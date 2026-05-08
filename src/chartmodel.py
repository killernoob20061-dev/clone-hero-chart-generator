"""
ChartNet v3 — ~50M parameter CNN-Transformer model for Clone Hero chart generation.

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

Architecture (v3 upgrades over v2):
  Mel spectrogram (128 bins, beat-aligned 16th grid)
    → Deep CNN  (5 residual blocks, 512→768 channels)   ~local patterns
    → Positional encoding
    → Transformer encoder (8 heads, 6 layers, 768 dim)  ~long-range context
    → Difficulty embedding (32-dim)
    → 5 output heads (wider hidden 256)
    → ~50M parameters
"""

import math
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
DIFF_EMB_DIM    = 32

DIFF_MAP = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}
CHORD_NO_CHORD  = 5   # special class index meaning "no chord"


# ── Building blocks ───────────────────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """Residual 1-D conv block with BN + GELU + dropout."""
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.1):
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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OutputHead(nn.Module):
    """3-layer MLP output head — wider than v2."""
    def __init__(self, in_dim, hidden, out_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── ChartNet v3 ───────────────────────────────────────────────────────────────

class ChartNet(nn.Module):
    def __init__(self, n_mels=N_MELS, diff_emb_dim=DIFF_EMB_DIM,
                 transformer_dim=768, transformer_heads=8,
                 transformer_layers=6, dropout=0.1):
        super().__init__()

        # ── Deep CNN — 5 residual blocks ──────────────────────────────────
        # 128 → 256 → 384 → 512 → 640 → 768 channels
        self.cnn = nn.Sequential(
            ResConvBlock(n_mels, 256, kernel=3, dropout=dropout),
            ResConvBlock(256,    384, kernel=3, dropout=dropout),
            ResConvBlock(384,    512, kernel=5, dropout=dropout),
            ResConvBlock(512,    640, kernel=5, dropout=dropout),
            ResConvBlock(640,    transformer_dim, kernel=7, dropout=dropout),
        )

        # ── Difficulty embedding ───────────────────────────────────────────
        self.diff_emb = nn.Embedding(N_DIFFICULTIES, diff_emb_dim)

        # ── Project CNN output + diff embedding → transformer dim ──────────
        self.input_proj = nn.Linear(transformer_dim + diff_emb_dim, transformer_dim)

        # ── Positional encoding ────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(transformer_dim, dropout=dropout)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers,
            norm=nn.LayerNorm(transformer_dim),
        )

        # ── 5 output heads (wider hidden 256) ─────────────────────────────
        H = 256
        self.note_head    = OutputHead(transformer_dim, H, 1,         dropout)
        self.fret_head    = OutputHead(transformer_dim, H, N_FRETS,   dropout)
        self.sustain_head = OutputHead(transformer_dim, H, N_SUSTAIN, dropout)
        self.chord_head   = OutputHead(transformer_dim, H, N_CHORD,   dropout)
        self.type_head    = OutputHead(transformer_dim, H, N_TYPE,    dropout)

    def forward(self, mel, difficulty):
        """
        mel:        (B, seq_len, n_mels)
        difficulty: (B,)  int in [0, 3]
        Returns dict of logits, all shape (B, seq_len, n_classes)
        """
        B, T, _ = mel.shape

        x = mel.permute(0, 2, 1)           # (B, n_mels, T)
        x = self.cnn(x)                     # (B, transformer_dim, T)
        x = x.permute(0, 2, 1)             # (B, T, transformer_dim)

        diff_e = self.diff_emb(difficulty)               # (B, diff_emb_dim)
        diff_e = diff_e.unsqueeze(1).expand(-1, T, -1)   # (B, T, diff_emb_dim)
        x = torch.cat([x, diff_e], dim=-1)               # (B, T, transformer_dim+diff)
        x = self.input_proj(x)                           # (B, T, transformer_dim)

        x = self.pos_enc(x)                              # (B, T, transformer_dim)
        x = self.transformer(x)                          # (B, T, transformer_dim)

        return {
            'note':    self.note_head(x),
            'fret':    self.fret_head(x),
            'sustain': self.sustain_head(x),
            'chord':   self.chord_head(x),
            'type':    self.type_head(x),
        }


# ── Multi-head loss ───────────────────────────────────────────────────────────

class ChartLoss(nn.Module):
    """
    Combined loss over all 5 heads.

    note_loss    — BCE with positive weight (notes are rare)
    fret_loss    — CE only at positions with a note
    sustain_loss — CE only at positions with a note
    chord_loss   — CE with class weights (chord frets upweighted 2.7x vs no-chord)
    type_loss    — CE only at positions with a note
    """
    def __init__(self, note_pos_weight=8.0,
                 w_note=1.0, w_fret=0.6, w_sustain=0.4,
                 w_chord=0.8, w_type=0.3):
        super().__init__()
        self.note_pos_weight = note_pos_weight
        self.w = dict(note=w_note, fret=w_fret,
                      sustain=w_sustain, chord=w_chord, type=w_type)
        # Chord class weights: upweight fret classes (0-4) vs no-chord (5)
        # ~73% no-chord, ~27% chord → weight chord frets 2.7x
        self.chord_class_weights = [2.7, 2.7, 2.7, 2.7, 2.7, 1.0]

    def forward(self, logits, note_targets, fret_targets,
                sustain_targets, chord_targets, type_targets):
        device = logits['note'].device

        # ── Note loss ──────────────────────────────────────────────────────
        pw = torch.tensor([self.note_pos_weight], device=device)
        note_loss = F.binary_cross_entropy_with_logits(
            logits['note'].squeeze(-1), note_targets, pos_weight=pw)

        # Mask: only compute other heads where a real note exists
        mask = (note_targets > 0.5)

        def _masked_ce(logit_key, targets, class_weights=None):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            pred = logits[logit_key][mask]
            tgt  = targets[mask].long().clamp(min=0)
            w = torch.tensor(class_weights, device=device, dtype=torch.float32) if class_weights else None
            return F.cross_entropy(pred, tgt, weight=w)

        # Chord: remap -1 (no chord) → class 5
        chord_t_remapped = chord_targets.clone()
        chord_t_remapped[chord_targets < 0] = CHORD_NO_CHORD

        fret_loss    = _masked_ce('fret',    fret_targets)
        sustain_loss = _masked_ce('sustain', sustain_targets)
        chord_loss   = _masked_ce('chord',   chord_t_remapped, self.chord_class_weights)
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

SUSTAIN_TICKS = {0: 0, 1: 64, 2: 144, 3: 288}

@torch.no_grad()
def predict_chart(model, mel_tensor, difficulty_str,
                  note_threshold=0.35, device='cpu'):
    """
    Run inference on a full-song mel tensor.
    mel_tensor: (seq_len, n_mels)

    Handles songs longer than 2048 frames by running overlapping chunks
    of 2048 and stitching predictions from the non-boundary region of each.
    """
    MAX_LEN    = 2048
    CHUNK      = MAX_LEN          # chunk size = max positional encoding
    STRIDE     = 1536             # advance 1536 frames per chunk (512 overlap)
    GUARD      = 256              # ignore predictions within GUARD frames of each edge

    model.eval()
    diff_idx = torch.tensor([DIFF_MAP.get(difficulty_str, 3)], device=device)
    seq_len  = mel_tensor.shape[0]

    def _decode_chunk(chunk_mel, offset):
        """Run model on a (<=2048, n_mels) slice, return notes with global grid_pos."""
        mel = chunk_mel.unsqueeze(0).to(device)
        out = model(mel, diff_idx)
        note_probs    = torch.sigmoid(out['note'].squeeze(0).squeeze(-1))
        fret_preds    = out['fret'].squeeze(0).argmax(dim=-1)
        sustain_preds = out['sustain'].squeeze(0).argmax(dim=-1)
        chord_preds   = out['chord'].squeeze(0).argmax(dim=-1)
        type_preds    = out['type'].squeeze(0).argmax(dim=-1)
        clen = chunk_mel.shape[0]
        results = []
        for t in range(clen):
            if float(note_probs[t]) < note_threshold:
                continue
            chord_idx = int(chord_preds[t])
            results.append({
                'grid_pos': t + offset,
                'fret':     int(fret_preds[t]),
                'sustain':  SUSTAIN_TICKS.get(int(sustain_preds[t]), 0),
                'chord':    chord_idx if chord_idx < CHORD_NO_CHORD else -1,
                'type':     int(type_preds[t]),
            })
        return results

    if seq_len <= CHUNK:
        # Short song — single pass, no chunking needed
        return _decode_chunk(mel_tensor, offset=0)

    # Long song — chunked inference with overlap
    notes = []
    seen_positions = set()
    start = 0
    while start < seq_len:
        end   = min(start + CHUNK, seq_len)
        chunk = mel_tensor[start:end]

        # Determine which positions in this chunk are "trusted" (not near edges)
        local_start = GUARD if start > 0 else 0
        local_end   = (end - start) - GUARD if end < seq_len else (end - start)
        local_end   = max(local_end, local_start + 1)  # always keep at least 1

        chunk_notes = _decode_chunk(chunk, offset=start)
        for n in chunk_notes:
            local_pos = n['grid_pos'] - start
            if local_start <= local_pos < local_end:
                if n['grid_pos'] not in seen_positions:
                    seen_positions.add(n['grid_pos'])
                    notes.append(n)

        if end >= seq_len:
            break
        start += STRIDE

    return sorted(notes, key=lambda x: x['grid_pos'])


def load_model(checkpoint_path, device='cpu'):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ChartNet(**ckpt.get('model_kwargs', {}))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model.to(device)
