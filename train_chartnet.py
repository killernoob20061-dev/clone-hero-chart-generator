"""
Train ChartNet on preprocessed Clone Hero chart pairs.
Usage: python train_chartnet.py --data ./processed --out ./checkpoints

Trains for ~50 epochs, saves best checkpoint by validation F1.
Estimated time: 4-8h on RTX 3090, 12-24h on RTX 3060.

After training, use:
    chartgen.py --model ./checkpoints/best.pt <song.mp3>
"""

import os, sys, json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score

from chartmodel  import ChartNet, ChartLoss, DIFF_MAP
from preprocess_dataset import ChartDataset

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(note_logits, note_targets, fret_logits, fret_targets,
                    threshold=0.35):
    """Compute note F1, fret accuracy."""
    probs = torch.sigmoid(note_logits.squeeze(-1)).cpu().numpy().flatten()
    preds = (probs >= threshold).astype(int)
    tgts  = note_targets.cpu().numpy().flatten().astype(int)

    note_f1   = f1_score(tgts, preds, zero_division=0)
    note_prec = precision_score(tgts, preds, zero_division=0)
    note_rec  = recall_score(tgts, preds, zero_division=0)

    # Fret accuracy — only on actual notes
    mask = (fret_targets >= 0).cpu()
    if mask.sum() > 0:
        fret_pred = fret_logits.argmax(-1)[mask].cpu().numpy()
        fret_true = fret_targets[mask].cpu().numpy()
        fret_acc  = (fret_pred == fret_true).mean()
    else:
        fret_acc = 0.0

    return {'note_f1': note_f1, 'note_prec': note_prec,
            'note_rec': note_rec, 'fret_acc': fret_acc}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = total_note = total_fret = 0.0
    for mel, note_t, fret_t, diff_t in loader:
        mel, note_t, fret_t, diff_t = (
            mel.to(device), note_t.to(device),
            fret_t.to(device), diff_t.to(device))

        optimizer.zero_grad()
        note_logits, fret_logits = model(mel, diff_t)
        loss, nl, fl = criterion(note_logits, fret_logits, note_t, fret_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_note += nl.item()
        total_fret += fl.item()

    n = len(loader)
    return {'loss': total_loss/n, 'note_loss': total_note/n, 'fret_loss': total_fret/n}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.35):
    model.eval()
    total_loss = 0.0
    all_metrics = []

    for mel, note_t, fret_t, diff_t in loader:
        mel, note_t, fret_t, diff_t = (
            mel.to(device), note_t.to(device),
            fret_t.to(device), diff_t.to(device))

        note_logits, fret_logits = model(mel, diff_t)
        loss, _, _ = criterion(note_logits, fret_logits, note_t, fret_t)
        total_loss += loss.item()

        m = compute_metrics(note_logits, note_t, fret_logits, fret_t, threshold)
        all_metrics.append(m)

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg['loss'] = total_loss / len(loader)
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      default='./processed',   help='Preprocessed .pt dir')
    ap.add_argument('--out',       default='./checkpoints', help='Checkpoint output dir')
    ap.add_argument('--epochs',    type=int,   default=60)
    ap.add_argument('--batch',     type=int,   default=32)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--val_split', type=float, default=0.1,  help='Validation fraction')
    ap.add_argument('--workers',   type=int,   default=4)
    ap.add_argument('--device',    default='auto')
    ap.add_argument('--resume',    default=None, help='Resume from checkpoint')
    ap.add_argument('--note_pos_weight', type=float, default=8.0)
    args = ap.parse_args()

    # Device
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
        print('ERROR: No training samples found. Run preprocess_dataset.py first.')
        sys.exit(1)

    n_val  = max(1, int(len(full_ds) * args.val_split))
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
                        lstm_hidden=256, lstm_layers=2, dropout=0.2)
    model     = ChartNet(**model_kwargs).to(device)
    criterion = ChartLoss(note_pos_weight=args.note_pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model: {n_params/1e6:.1f}M parameters')

    start_epoch = 0
    best_f1     = 0.0

    # Resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1     = ckpt.get('best_f1', 0.0)
        print(f'  Resumed from epoch {start_epoch}  best_f1={best_f1:.3f}')

    history = []

    # ── Train loop ─────────────────────────────────────────────────────────
    print(f'\nTraining for {args.epochs} epochs...')
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch+1:3d}/{args.epochs}  '
              f'loss={tr["loss"]:.4f}  '
              f'val_loss={va["loss"]:.4f}  '
              f'note_F1={va["note_f1"]:.3f}  '
              f'note_P={va["note_prec"]:.3f}  '
              f'note_R={va["note_rec"]:.3f}  '
              f'fret_acc={va["fret_acc"]:.3f}  '
              f'lr={lr_now:.2e}  '
              f't={elapsed:.0f}s')

        row = {'epoch': epoch+1, **tr,
               **{f'val_{k}': v for k, v in va.items()}}
        history.append(row)

        # Save best
        if va['note_f1'] > best_f1:
            best_f1 = va['note_f1']
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'best_f1':      best_f1,
                'val_metrics':  va,
            }, out_dir / 'best.pt')
            print(f'  ✓ New best F1={best_f1:.3f} → saved best.pt')

        # Save latest every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'best_f1':      best_f1,
            }, out_dir / f'epoch_{epoch+1:03d}.pt')

    # Save history
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'\nTraining complete. Best note F1: {best_f1:.3f}')
    print(f'Best model: {out_dir}/best.pt')
    print(f'\nTo use in chartgen: python chartgen.py --model {out_dir}/best.pt song.mp3')


if __name__ == '__main__':
    main()
