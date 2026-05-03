"""
ChartNet — small CNN-LSTM model for Clone Hero chart generation.
~8M parameters. Predicts note placement + fret from beat-aligned mel spectrogram.

Architecture:
  Mel spectrogram (128 bins, beat-aligned 16th grid)
    → CNN feature extractor  (local spectral patterns)
    → BiLSTM                 (temporal context, phrase awareness)
    → Difficulty embedding   (conditions on Easy/Medium/Hard/Expert)
    → Output heads:
        note_head   → P(note at this grid position)  [sigmoid]
        fret_head   → P(fret=0|1|2 | note present)   [softmax]

Input shape:  (batch, seq_len, n_mels)   — seq_len = 16th-note grid positions
Output:       note_logits  (batch, seq_len, 1)
              fret_logits  (batch, seq_len, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_FRETS       = 3    # Green / Red / Yellow only
N_DIFFICULTIES = 4   # easy / medium / hard / expert
N_MELS        = 128
DIFF_EMB_DIM  = 16


class ConvBlock(nn.Module):
    """2-layer conv block with BatchNorm, ReLU, dropout."""
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ChartNet(nn.Module):
    def __init__(self, n_mels=N_MELS, diff_emb_dim=DIFF_EMB_DIM,
                 lstm_hidden=256, lstm_layers=2, dropout=0.2):
        super().__init__()

        # ── CNN feature extractor ─────────────────────────────────────────
        # Input: (batch, n_mels, seq_len)  — channels first for Conv1d
        self.cnn = nn.Sequential(
            ConvBlock(n_mels, 128, kernel=3, dropout=dropout),
            ConvBlock(128,    256, kernel=3, dropout=dropout),
            ConvBlock(256,    256, kernel=5, dropout=dropout),
        )
        cnn_out_dim = 256

        # ── Difficulty embedding ──────────────────────────────────────────
        self.diff_emb = nn.Embedding(N_DIFFICULTIES, diff_emb_dim)

        # ── Bidirectional LSTM ────────────────────────────────────────────
        lstm_in = cnn_out_dim + diff_emb_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2   # bidirectional

        # ── Output heads ──────────────────────────────────────────────────
        self.note_head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),    # logit → sigmoid = P(note)
        )
        self.fret_head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, N_FRETS),  # logits → softmax = P(fret)
        )

    def forward(self, mel, difficulty):
        """
        mel:        (B, seq_len, n_mels)
        difficulty: (B,)  int in [0,3]
        Returns:
            note_logits  (B, seq_len, 1)
            fret_logits  (B, seq_len, 3)
        """
        B, T, _ = mel.shape

        # CNN expects (B, channels, time)
        x = mel.permute(0, 2, 1)        # (B, n_mels, T)
        x = self.cnn(x)                  # (B, 256, T)
        x = x.permute(0, 2, 1)          # (B, T, 256)

        # Expand difficulty embedding across time
        diff_e = self.diff_emb(difficulty)           # (B, diff_emb_dim)
        diff_e = diff_e.unsqueeze(1).expand(-1, T, -1)  # (B, T, diff_emb_dim)

        x = torch.cat([x, diff_e], dim=-1)           # (B, T, 256+diff_emb_dim)

        # BiLSTM
        x, _ = self.lstm(x)                          # (B, T, lstm_hidden*2)

        note_logits = self.note_head(x)              # (B, T, 1)
        fret_logits = self.fret_head(x)              # (B, T, 3)

        return note_logits, fret_logits


# ── Loss ──────────────────────────────────────────────────────────────────────

class ChartLoss(nn.Module):
    """
    Combined loss:
      note_loss  — Binary cross-entropy with positive weight (notes are rare ~5-15%)
      fret_loss  — Cross-entropy on fret, but ONLY at positions where a note exists
    """
    def __init__(self, note_pos_weight=8.0, fret_weight=0.5):
        super().__init__()
        self.note_pos_weight = note_pos_weight
        self.fret_weight     = fret_weight

    def forward(self, note_logits, fret_logits, note_targets, fret_targets):
        """
        note_logits:  (B, T, 1)
        fret_logits:  (B, T, 3)
        note_targets: (B, T)     float 0/1
        fret_targets: (B, T)     int   0/1/2  (-1 = no note, ignored)
        """
        # Note loss
        pw  = torch.tensor([self.note_pos_weight], device=note_logits.device)
        note_loss = F.binary_cross_entropy_with_logits(
            note_logits.squeeze(-1), note_targets, pos_weight=pw)

        # Fret loss — only where note_targets == 1
        mask = (note_targets > 0.5)
        if mask.sum() > 0:
            fret_pred = fret_logits[mask]                     # (N, 3)
            fret_true = fret_targets[mask].long()             # (N,)
            fret_loss = F.cross_entropy(fret_pred, fret_true)
        else:
            fret_loss = torch.tensor(0.0, device=note_logits.device)

        total = note_loss + self.fret_weight * fret_loss
        return total, note_loss.detach(), fret_loss.detach()


# ── Inference helpers ─────────────────────────────────────────────────────────

DIFF_MAP = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}

@torch.no_grad()
def predict_chart(model, mel_tensor, difficulty_str,
                  note_threshold=0.35, device='cpu'):
    """
    Run model inference on a full-song mel tensor.
    mel_tensor: (seq_len, n_mels)  — one sequence per song
    Returns list of (grid_position, fret) tuples.
    """
    model.eval()
    mel = mel_tensor.unsqueeze(0).to(device)           # (1, T, n_mels)
    diff_idx = torch.tensor(
        [DIFF_MAP.get(difficulty_str, 3)], device=device)

    note_logits, fret_logits = model(mel, diff_idx)
    note_probs = torch.sigmoid(note_logits.squeeze(0).squeeze(-1))  # (T,)
    fret_preds = fret_logits.squeeze(0).argmax(dim=-1)               # (T,)

    notes = []
    for t in range(len(note_probs)):
        if float(note_probs[t]) >= note_threshold:
            notes.append((t, int(fret_preds[t])))

    return notes


def load_model(checkpoint_path, device='cpu'):
    """Load a trained ChartNet from a checkpoint file."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = ChartNet(**ckpt.get('model_kwargs', {}))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model.to(device)
