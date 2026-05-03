"""
Preprocess scraped Clone Hero chart pairs into training tensors.
Usage: python preprocess_dataset.py --data ./dataset --out ./processed

For each song folder (contains notes.chart + song.opus):
  1. Parse notes.chart → note grids per difficulty
  2. Load audio → beat-aligned log-mel spectrogram
  3. Align notes to 16th-note beat grid
  4. Save as .pt tensor pair (mel, labels)

Output: ./processed/<id>.pt   each containing:
  {
    'mel':           FloatTensor (seq_len, 128)   beat-aligned log-mel
    'note_targets':  dict[diff] → FloatTensor (seq_len,)   0/1
    'fret_targets':  dict[diff] → LongTensor  (seq_len,)   0/1/2 or -1
    'bpm':           float
    'title':         str
    'artist':        str
  }
"""

import os, sys, re, json, argparse
from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# ── Chart parser ──────────────────────────────────────────────────────────────

DIFF_SECTIONS = {
    'easy':   'EasySingle',
    'medium': 'MediumSingle',
    'hard':   'HardSingle',
    'expert': 'ExpertSingle',
}

def parse_chart(chart_path):
    """
    Parse a notes.chart file.
    Returns dict: {
        'resolution': int,
        'bpm': float,
        'notes': { 'expert': [(tick,fret,sustain),...], ... }
    }
    """
    text = Path(chart_path).read_text(encoding='utf-8', errors='replace')

    # Resolution
    res_m = re.search(r'Resolution\s*=\s*(\d+)', text)
    resolution = int(res_m.group(1)) if res_m else 192

    # BPM from SyncTrack
    bpm_events = []
    sync_m = re.search(r'\[SyncTrack\]\s*\{([^}]*)\}', text, re.S)
    if sync_m:
        for m in re.finditer(r'(\d+)\s*=\s*B\s+(\d+)', sync_m.group(1)):
            tick  = int(m.group(1))
            bpm   = int(m.group(2)) / 1000.0
            bpm_events.append((tick, bpm))
    global_bpm = bpm_events[0][1] if bpm_events else 120.0

    # Notes per difficulty
    notes = {}
    for diff, section in DIFF_SECTIONS.items():
        sec_m = re.search(rf'\[{re.escape(section)}\]\s*\{{([^}}]*)\}}', text, re.S)
        if not sec_m:
            continue
        diff_notes = []
        for m in re.finditer(r'(\d+)\s*=\s*N\s+(\d+)\s+(\d+)', sec_m.group(1)):
            tick    = int(m.group(1))
            fret    = int(m.group(2))
            sustain = int(m.group(3))
            if fret <= 2:          # only Green/Red/Yellow
                diff_notes.append((tick, fret, sustain))
        if diff_notes:
            notes[diff] = sorted(diff_notes, key=lambda x: x[0])

    return {'resolution': resolution, 'bpm': global_bpm,
            'bpm_events': bpm_events, 'notes': notes}


# ── Audio → beat-aligned mel ──────────────────────────────────────────────────

def audio_to_beat_mel(audio_path, bpm, bpm_events=None,
                       n_mels=128, hop_length=512, sr=22050):
    """
    Load audio and compute a log-mel spectrogram.
    Resample time axis so each frame = one 16th note grid position.
    Returns: mel FloatTensor (seq_len, n_mels), beat_times array, tps float
    """
    y, _ = librosa.load(str(audio_path), sr=sr)

    # Beat track on the audio itself for accurate grid
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames',
                                                  bpm=bpm, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    actual_bpm = float(np.atleast_1d(tempo)[0])
    tps        = (actual_bpm / 60.0) * 192   # ticks per second at res=192

    # Full log-mel spectrogram
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           hop_length=hop_length,
                                           n_fft=2048, fmin=20, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)   # (n_mels, frames)

    # Build 16th-note grid times
    grid_times = []
    for i in range(len(beat_times)-1):
        gap = beat_times[i+1] - beat_times[i]
        for k in range(4):
            grid_times.append(beat_times[i] + gap * k / 4)
    if len(beat_times) > 0:
        grid_times.append(beat_times[-1])
    grid_times = np.array(sorted(set(grid_times)))

    # For each grid time, extract the nearest mel frame
    mel_frames = []
    for t in grid_times:
        frame = int(librosa.time_to_frames(t, sr=sr, hop_length=hop_length))
        frame = max(0, min(frame, S_db.shape[1]-1))
        mel_frames.append(S_db[:, frame])

    mel = np.stack(mel_frames, axis=0)               # (seq_len, n_mels)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)   # normalise
    mel_tensor = torch.tensor(mel, dtype=torch.float32)

    return mel_tensor, grid_times, actual_bpm, tps


def ticks_to_grid_positions(notes, grid_times, tps, tolerance_sec=0.04):
    """
    Map chart notes (tick positions) → grid indices.
    tolerance_sec: notes within this distance snap to the nearest grid.
    Returns: note_targets (seq_len,) float,  fret_targets (seq_len,) long
    """
    T = len(grid_times)
    note_t = np.zeros(T, dtype=np.float32)
    fret_t = np.full(T, -1, dtype=np.int64)   # -1 = no note

    for tick, fret, _ in notes:
        t_sec = tick / tps
        # Find nearest grid position
        diffs = np.abs(grid_times - t_sec)
        idx   = int(diffs.argmin())
        if diffs[idx] <= tolerance_sec:
            note_t[idx] = 1.0
            fret_t[idx] = max(fret_t[idx], fret)   # take highest fret if chord

    return torch.tensor(note_t), torch.tensor(fret_t)


# ── Per-song processing ───────────────────────────────────────────────────────

def process_one(song_dir, n_mels=128, hop_length=512):
    song_dir   = Path(song_dir)
    chart_path = song_dir / 'notes.chart'
    audio_path = song_dir / 'song.opus'

    if not chart_path.exists() or not audio_path.exists():
        return None

    try:
        chart = parse_chart(chart_path)
        if not chart['notes']:
            return None

        mel, grid_times, bpm, tps = audio_to_beat_mel(
            audio_path, chart['bpm'],
            bpm_events=chart['bpm_events'],
            n_mels=n_mels, hop_length=hop_length)

        note_targets = {}
        fret_targets = {}
        for diff, diff_notes in chart['notes'].items():
            nt, ft = ticks_to_grid_positions(diff_notes, grid_times, tps)
            # Trim/pad to mel sequence length
            T = len(mel)
            nt = nt[:T] if len(nt) >= T else torch.cat([nt, torch.zeros(T-len(nt))])
            ft = ft[:T] if len(ft) >= T else torch.cat([ft, torch.full((T-len(ft),),-1)])
            note_targets[diff] = nt
            fret_targets[diff] = ft

        meta = {}
        if (song_dir / 'meta.json').exists():
            meta = json.loads((song_dir / 'meta.json').read_text())

        return {
            'mel':          mel,
            'note_targets': note_targets,
            'fret_targets': fret_targets,
            'bpm':          bpm,
            'title':        meta.get('title', song_dir.name),
            'artist':       meta.get('artist', ''),
        }
    except Exception as e:
        print(f'  [warn] {song_dir.name}: {e}')
        return None


# ── Dataset class ─────────────────────────────────────────────────────────────

class ChartDataset(Dataset):
    """
    Loads preprocessed .pt files from a directory.
    Returns windowed (mel_window, note_window, fret_window, diff_idx).
    window_beats: number of beats per training window (default 16 = 4 bars)
    """
    DIFFS = ['easy', 'medium', 'hard', 'expert']

    def __init__(self, processed_dir, window_beats=16, stride_beats=4,
                 sixteenths_per_beat=4):
        self.processed_dir     = Path(processed_dir)
        self.window_size       = window_beats * sixteenths_per_beat   # grid steps
        self.stride            = stride_beats * sixteenths_per_beat
        self.files             = sorted(self.processed_dir.glob('*.pt'))
        self.samples           = []   # (file_idx, start_pos, diff_idx)
        self._build_index()

    def _build_index(self):
        for fi, fp in enumerate(self.files):
            try:
                data = torch.load(str(fp), map_location='cpu')
            except Exception:
                continue
            T = data['mel'].shape[0]
            for diff_i, diff in enumerate(self.DIFFS):
                if diff not in data['note_targets']:
                    continue
                for start in range(0, T - self.window_size, self.stride):
                    self.samples.append((fi, start, diff_i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, start, diff_i = self.samples[idx]
        data  = torch.load(str(self.files[fi]), map_location='cpu')
        end   = start + self.window_size
        diff  = self.DIFFS[diff_i]
        mel   = data['mel'][start:end]                         # (W, n_mels)
        notes = data['note_targets'][diff][start:end]          # (W,)
        frets = data['fret_targets'][diff][start:end]          # (W,)
        diff_t = torch.tensor(diff_i, dtype=torch.long)
        return mel, notes, frets, diff_t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',  default='./dataset',   help='Scraped dataset dir')
    ap.add_argument('--out',   default='./processed', help='Output .pt dir')
    ap.add_argument('--mels',  type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()
                 and (d/'notes.chart').exists() and (d/'song.opus').exists()]
    print(f'Processing {len(song_dirs)} songs → {out_dir}')

    from concurrent.futures import ProcessPoolExecutor, as_completed

    ok = 0; fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, d, args.mels): d for d in song_dirs}
        for future in as_completed(futures):
            song_dir = futures[future]
            result   = future.result()
            if result:
                out_path = out_dir / f'{song_dir.name}.pt'
                torch.save(result, str(out_path))
                ok += 1
                if ok % 50 == 0:
                    print(f'  {ok}/{len(song_dirs)} processed')
            else:
                fail += 1

    print(f'\nDone: {ok} saved, {fail} failed → {out_dir}')


if __name__ == '__main__':
    main()
