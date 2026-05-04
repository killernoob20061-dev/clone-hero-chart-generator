"""
ChartNet v2 — ~35M parameter CNN-LSTM model for Clone Hero chart generation.

Predicts 5 outputs per 16th-note grid position:
  note_head    → P(note here)               [sigmoid, binary]
  fret_head    → P(fret=0..4 | note)        [softmax, 5 classes]
  sustain_head → P(sustain bucket 0..3)     [softmax, 4 classes]
  chord_head   → P(chord fret 0..4 | none)  [softmax, 6 classes: 0-4 + no-chord]
  type_head    → P(note type 0..3)          [softmax, 4 classes]

Frets:         0=Green 1=Red 2=Yellow 3=Blue 4=Orange
Sustain buckets: 0=none 1=short(1/8-1/4) 2=medium(1/4-1beat) 3=long(>1beat)
Note types:    0=normal 1=HOPO 2=tap 3=open strum
Chord classes: 0-4=fret, 5=no chord

Architecture:
  Mel spectrogram (128 bins, beat-aligned 16th grid)
    → Deep CNN  (4 residual blocks, 512 channels)   ~local patterns
    → BiLSTM    (2 layers × 512 hidden)             ~phrase context
    → Difficulty embedding (16-dim)
    → 5 output heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────

N_FRETS         = 5    # 0-4
N_SUSTAIN       = 4    # buckets 0-3
N_CHORD         = 6    # frets 0-4  +  class 5 = no chord
N_TYPE          = 4    # normal / HOPO / tap / open
N_DIFFICULTIES  = 4
N_MELS          = 128
DIFF_EMB_DIM    = 16

DIFF_MAP = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}
CHORD_NO_CHORD  = 5   # special class index meaning "no chord"


# ── Building blocks ───────────────────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """Residual 1-D conv block with BN + GELU + dropout."""
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.15):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class OutputHead(nn.Module):
    """Small 2-layer MLP output head."""
    def __init__(self, in_dim, hidden, out_classes, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── ChartNet v2 ───────────────────────────────────────────────────────────────

class ChartNet(nn.Module):
    def __init__(self, n_mels=N_MELS, diff_emb_dim=DIFF_EMB_DIM,
                 lstm_hidden=512, lstm_layers=2, dropout=0.15):
        super().__init__()

        # ── Deep CNN — 4 residual blocks ──────────────────────────────────
        # 128 → 256 → 384 → 512 → 512 channels
        self.cnn = nn.Sequential(
            ResConvBlock(n_mels, 256, kernel=3,  dropout=dropout),
            ResConvBlock(256,    384, kernel=3,  dropout=dropout),
            ResConvBlock(384,    512, kernel=5,  dropout=dropout),
            ResConvBlock(512,    512, kernel=5,  dropout=dropout),
        )
        cnn_out = 512

        # ── Difficulty embedding ───────────────────────────────────────────
        self.diff_emb = nn.Embedding(N_DIFFICULTIES, diff_emb_dim)

        # ── Bidirectional LSTM ─────────────────────────────────────────────
        lstm_in = cnn_out + diff_emb_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2   # bidirectional

        # ── 5 output heads ─────────────────────────────────────────────────
        H = 128   # head hidden size
        self.note_head    = OutputHead(lstm_out, H, 1,          dropout)  # binary
        self.fret_head    = OutputHead(lstm_out, H, N_FRETS,    dropout)  # 5 classes
        self.sustain_head = OutputHead(lstm_out, H, N_SUSTAIN,  dropout)  # 4 classes
        self.chord_head   = OutputHead(lstm_out, H, N_CHORD,    dropout)  # 6 classes
        self.type_head    = OutputHead(lstm_out, H, N_TYPE,     dropout)  # 4 classes

    def forward(self, mel, difficulty):
        """
        mel:        (B, seq_len, n_mels)
        difficulty: (B,)  int in [0, 3]
        Returns dict of logits, all shape (B, seq_len, n_classes):
          note_logits    (B, T, 1)
          fret_logits    (B, T, 5)
          sustain_logits (B, T, 4)
          chord_logits   (B, T, 6)
          type_logits    (B, T, 4)
        """
        B, T, _ = mel.shape

        x = mel.permute(0, 2, 1)          # (B, n_mels, T)
        x = self.cnn(x)                    # (B, 512, T)
        x = x.permute(0, 2, 1)            # (B, T, 512)

        diff_e = self.diff_emb(difficulty)              # (B, diff_emb_dim)
        diff_e = diff_e.unsqueeze(1).expand(-1, T, -1)  # (B, T, diff_emb_dim)
        x = torch.cat([x, diff_e], dim=-1)              # (B, T, 512+16)

        x, _ = self.lstm(x)                             # (B, T, lstm_hidden*2)

        return {
            'note':    self.note_head(x),     # (B, T, 1)
            'fret':    self.fret_head(x),     # (B, T, 5)
            'sustain': self.sustain_head(x),  # (B, T, 4)
            'chord':   self.chord_head(x),    # (B, T, 6)
            'type':    self.type_head(x),     # (B, T, 4)
        }


# ── Multi-head loss ───────────────────────────────────────────────────────────

class ChartLoss(nn.Module):
    """
    Combined loss over all 5 heads.

    note_loss    — BCE with positive weight (notes are rare)
    fret_loss    — CE only at positions with a note
    sustain_loss — CE only at positions with a note
    chord_loss   — CE only at positions with a note
                   chord_target=-1 → class 5 (no chord)
    type_loss    — CE only at positions with a note

    Weights roughly calibrated so each head contributes meaningfully.
    """
    def __init__(self, note_pos_weight=8.0,
                 w_note=1.0, w_fret=0.6, w_sustain=0.4,
                 w_chord=0.4, w_type=0.3):
        super().__init__()
        self.note_pos_weight = note_pos_weight
        self.w = dict(note=w_note, fret=w_fret,
                      sustain=w_sustain, chord=w_chord, type=w_type)

    def forward(self, logits, note_targets, fret_targets,
                sustain_targets, chord_targets, type_targets):
        device = logits['note'].device

        # ── Note loss ──────────────────────────────────────────────────────
        pw = torch.tensor([self.note_pos_weight], device=device)
        note_loss = F.binary_cross_entropy_with_logits(
            logits['note'].squeeze(-1), note_targets, pos_weight=pw)

        # Mask: only compute other heads where a real note exists
        mask = (note_targets > 0.5)

        def _masked_ce(logit_key, targets, ignore=-1):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            pred = logits[logit_key][mask]     # (N, C)
            tgt  = targets[mask].long()
            # Remap -1 to ignore_index
            tgt  = tgt.clamp(min=0)            # avoid negative index crash
            return F.cross_entropy(pred, tgt)

        # Chord: remap -1 (no chord) → class 5
        chord_t_remapped = chord_targets.clone()
        chord_t_remapped[chord_targets < 0] = CHORD_NO_CHORD

        fret_loss    = _masked_ce('fret',    fret_targets)
        sustain_loss = _masked_ce('sustain', sustain_targets)
        chord_loss   = _masked_ce('chord',   chord_t_remapped)
        type_loss    = _masked_ce('type',    type_targets)

        total = (self.w['note']    * note_loss +
                 self.w['fret']    * fret_loss +
                 self.w['sustain'] * sustain_loss +
                 self.w['chord']   * chord_loss +
                 self.w['type']    * type_loss)

        return total, {
            'note':    note_loss.detach(),
            'fret':    fret_loss.detach(),
            'sustain': sustain_loss.detach(),
            'chord':   chord_loss.detach(),
            'type':    type_loss.detach(),
        }


# ── Inference ─────────────────────────────────────────────────────────────────

# Sustain bucket → approximate tick fraction of a beat (at res=192)
SUSTAIN_TICKS = {0: 0, 1: 64, 2: 144, 3: 288}

@torch.no_grad()
def predict_chart(model, mel_tensor, difficulty_str,
                  note_threshold=0.35, device='cpu'):
    """
    Run inference on a full-song mel tensor.
    mel_tensor: (seq_len, n_mels)

    Returns list of dicts:
      {
        'grid_pos': int,
        'fret':     int (0-4),
        'sustain':  int (ticks approx),
        'chord':    int (0-4) or -1,
        'type':     int (0-3),
      }
    """
    model.eval()
    mel      = mel_tensor.unsqueeze(0).to(device)
    diff_idx = torch.tensor([DIFF_MAP.get(difficulty_str, 3)], device=device)

    out = model(mel, diff_idx)

    note_probs    = torch.sigmoid(out['note'].squeeze(0).squeeze(-1))  # (T,)
    fret_preds    = out['fret'].squeeze(0).argmax(dim=-1)              # (T,)
    sustain_preds = out['sustain'].squeeze(0).argmax(dim=-1)           # (T,)
    chord_preds   = out['chord'].squeeze(0).argmax(dim=-1)             # (T,)
    type_preds    = out['type'].squeeze(0).argmax(dim=-1)              # (T,)

    notes = []
    for t in range(len(note_probs)):
        if float(note_probs[t]) < note_threshold:
            continue
        chord_idx = int(chord_preds[t])
        notes.append({
            'grid_pos': t,
            'fret':     int(fret_preds[t]),
            'sustain':  SUSTAIN_TICKS.get(int(sustain_preds[t]), 0),
            'chord':    chord_idx if chord_idx < CHORD_NO_CHORD else -1,
            'type':     int(type_preds[t]),
        })

    return notes


def load_model(checkpoint_path, device='cpu'):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ChartNet(**ckpt.get('model_kwargs', {}))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model.to(device)
