"""
Train ChartNet v2 on preprocessed Clone Hero chart pairs.
Usage: python train_chartnet.py --data ./processed --out ./checkpoints

Trains for ~40 epochs, saves best checkpoint by validation note F1.
Mixed precision (AMP) is enabled automatically on CUDA.
Estimated time: 4-7h on RTX 3090, 12-18h on RTX 3060 (with AMP).

After training:
    chartgen.py --model ./checkpoints/best.pt <song.mp3>
"""

import os, sys, json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_mel(mel):
    """SpecAugment-style augmentation: freq mask, time mask, pitch shift, noise."""
    mel = mel.clone()
    T, F = mel.shape
    # frequency masking
    f_w = torch.randint(1, 11, (1,)).item()
    f_s = torch.randint(0, max(1, F - f_w), (1,)).item()
    mel[:, f_s:f_s + f_w] = 0.0
    # time masking
    t_w = torch.randint(1, 9, (1,)).item()
    t_s = torch.randint(0, max(1, T - t_w), (1,)).item()
    mel[t_s:t_s + t_w, :] = 0.0
    # pitch shift simulation (roll mel bins)
    shift = torch.randint(-2, 3, (1,)).item()
    if shift != 0:
        mel = torch.roll(mel, shift, dims=1)
    # Gaussian noise
    mel = mel + torch.randn_like(mel) * 0.05
    return mel

from chartmodel  import ChartNet, ChartLoss, DIFF_MAP, N_FRETS, N_SUSTAIN, N_CHORD, N_TYPE
from preprocess_dataset import ChartDataset

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(logits, note_targets, fret_targets,
                    sustain_targets, chord_targets, type_targets,
                    threshold=0.35):
    note_probs = torch.sigmoid(logits['note'].squeeze(-1)).cpu().numpy().flatten()
    note_preds = (note_probs >= threshold).astype(int)
    note_true  = note_targets.cpu().numpy().flatten().astype(int)

    note_f1   = f1_score(note_true, note_preds, zero_division=0)
    note_prec = precision_score(note_true, note_preds, zero_division=0)
    note_rec  = recall_score(note_true, note_preds, zero_division=0)

    mask = (note_targets > 0.5).cpu()

    def _acc(logit_key, targets, offset=0):
        if mask.sum() == 0: return 0.0
        pred = logits[logit_key].argmax(-1)[mask].cpu().numpy()
        tgt  = targets[mask].cpu().numpy()
        tgt  = np.clip(tgt + offset, 0, None)
        return float((pred == tgt).mean())

    fret_acc    = _acc('fret',    fret_targets)
    sustain_acc = _acc('sustain', sustain_targets)
    chord_acc   = _acc('chord',   chord_targets.clamp(min=0))
    type_acc    = _acc('type',    type_targets)

    return {
        'note_f1':    note_f1,
        'note_prec':  note_prec,
        'note_rec':   note_rec,
        'fret_acc':   fret_acc,
        'sustain_acc': sustain_acc,
        'chord_acc':  chord_acc,
        'type_acc':   type_acc,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0,
                scaler=None, augment=True):
    model.train()
    use_amp = scaler is not None
    totals = {'loss': 0.0, 'note': 0.0, 'fret': 0.0,
              'sustain': 0.0, 'chord': 0.0, 'type': 0.0}

    for mel, note_t, fret_t, sustain_t, chord_t, type_t, diff_t in loader:
        mel, note_t, fret_t, sustain_t, chord_t, type_t, diff_t = (
            mel.to(device), note_t.to(device), fret_t.to(device),
            sustain_t.to(device), chord_t.to(device),
            type_t.to(device), diff_t.to(device))

        if augment and torch.rand(1).item() < 0.5:
            mel = augment_mel(mel)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mel, diff_t)
            loss, sub = criterion(logits, note_t, fret_t, sustain_t, chord_t, type_t)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        totals['loss']    += loss.item()
        totals['note']    += sub['note'].item()
        totals['fret']    += sub['fret'].item()
        totals['sustain'] += sub['sustain'].item()
        totals['chord']   += sub['chord'].item()
        totals['type']    += sub['type'].item()

    n = max(len(loader), 1)
    return {k: v/n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.35, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_metrics = []

    for mel, note_t, fret_t, sustain_t, chord_t, type_t, diff_t in loader:
        mel, note_t, fret_t, sustain_t, chord_t, type_t, diff_t = (
            mel.to(device), note_t.to(device), fret_t.to(device),
            sustain_t.to(device), chord_t.to(device),
            type_t.to(device), diff_t.to(device))

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mel, diff_t)
            loss, _ = criterion(logits, note_t, fret_t, sustain_t, chord_t, type_t)
        total_loss += loss.item()

        m = compute_metrics(logits, note_t, fret_t, sustain_t,
                            chord_t, type_t, threshold)
        all_metrics.append(m)

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg['loss'] = total_loss / max(len(loader), 1)
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      default='./processed',   help='Preprocessed .pt dir')
    ap.add_argument('--out',       default='./checkpoints', help='Checkpoint output dir')
    ap.add_argument('--epochs',    type=int,   default=40)
    ap.add_argument('--batch',     type=int,   default=48,
                    help='Batch size (reduce to 24 if OOM; AMP halves activation memory)')
    ap.add_argument('--lr',        type=float, default=2e-4)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--workers',   type=int,   default=4)
    ap.add_argument('--device',    default='auto')
    ap.add_argument('--resume',    default=None, help='Resume from checkpoint')
    ap.add_argument('--note_pos_weight', type=float, default=8.0)
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    print('Loading dataset...')
    full_ds = ChartDataset(args.data, window_beats=16, stride_beats=4)
    print(f'  Total windows: {len(full_ds)}')
    if len(full_ds) == 0:
        print('ERROR: No training samples. Run preprocess_dataset.py first.')
        sys.exit(1)

    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f'  Train={n_train}  Val={n_val}')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    # ── Model ─────────────────────────────────────────────────────────────
    model_kwargs = dict(n_mels=128, diff_emb_dim=16,
                        lstm_hidden=512, lstm_layers=2, dropout=0.15)
    model     = ChartNet(**model_kwargs).to(device)
    criterion = ChartLoss(note_pos_weight=args.note_pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_amp  = device.type == 'cuda'
    scaler   = GradScaler() if use_amp else None
    print(f'  Model: {n_params/1e6:.1f}M parameters')
    print(f'  Heads: note | fret(5) | sustain(4) | chord(6) | type(4)')
    print(f'  AMP: {"enabled (fp16)" if use_amp else "disabled (cpu)"}')

    start_epoch = 0
    best_f1     = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        if use_amp and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1     = ckpt.get('best_f1', 0.0)
        print(f'  Resumed from epoch {start_epoch}  best_f1={best_f1:.3f}')

    history = []

    print(f'\nTraining for {args.epochs} epochs...')
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device,
                         scaler=scaler)
        va = eval_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f'Epoch {epoch+1:3d}/{args.epochs}  '
            f'loss={tr["loss"]:.4f}  val_loss={va["loss"]:.4f}  '
            f'F1={va["note_f1"]:.3f}  P={va["note_prec"]:.3f}  R={va["note_rec"]:.3f}  '
            f'fret={va["fret_acc"]:.3f}  sus={va["sustain_acc"]:.3f}  '
            f'chord={va["chord_acc"]:.3f}  type={va["type_acc"]:.3f}  '
            f'lr={lr_now:.2e}  t={elapsed:.0f}s'
        )

        row = {'epoch': epoch+1, **tr,
               **{f'val_{k}': v for k, v in va.items()}}
        history.append(row)

        if va['note_f1'] > best_f1:
            best_f1 = va['note_f1']
            ckpt_best = {
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'best_f1':      best_f1,
                'val_metrics':  va,
            }
            if use_amp:
                ckpt_best['scaler_state'] = scaler.state_dict()
            torch.save(ckpt_best, out_dir / 'best.pt')
            print(f'  ✓ New best F1={best_f1:.3f} → saved best.pt')

        if (epoch+1) % 1 == 0:
            ckpt_periodic = {
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'best_f1':      best_f1,
            }
            if use_amp:
                ckpt_periodic['scaler_state'] = scaler.state_dict()
            torch.save(ckpt_periodic, out_dir / f'epoch_{epoch+1:03d}.pt')

        # Keep only the 3 most recent periodic checkpoints
        old_ckpts = sorted(out_dir.glob('epoch_*.pt'))
        for old in old_ckpts[:-3]:
            old.unlink(missing_ok=True)

    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'\nTraining complete. Best note F1: {best_f1:.3f}')
    print(f'Best model: {out_dir}/best.pt')
    print(f'\nTo use: python chartgen.py --model {out_dir}/best.pt song.mp3')


if __name__ == '__main__':
    main()
