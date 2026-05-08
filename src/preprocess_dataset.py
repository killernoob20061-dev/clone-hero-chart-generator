"""
Preprocess scraped Clone Hero chart pairs into training tensors.
Usage: python preprocess_dataset.py --data ./dataset --out ./processed

For each song folder (contains notes.chart + song.opus):
  1. Parse notes.chart → rich note grids per difficulty
  2. Load audio → beat-aligned log-mel spectrogram
  3. Align notes to 16th-note beat grid
  4. Save as .pt tensor

Output: ./processed/<id>.pt   each containing:
  {
    'mel':             FloatTensor (seq_len, 128)   beat-aligned log-mel
    'note_targets':    dict[diff] → FloatTensor (seq_len,)   0/1
    'fret_targets':    dict[diff] → LongTensor  (seq_len,)   0-4 or -1
    'sustain_targets': dict[diff] → LongTensor  (seq_len,)   sustain bucket 0-3
    'chord_targets':   dict[diff] → LongTensor  (seq_len,)   chord fret 0-4, or -1=none
    'type_targets':    dict[diff] → LongTensor  (seq_len,)   note type 0-3
    'bpm':             float
    'title':           str
    'artist':          str
  }

Sustain buckets:
  0 = none (< 1/8 note)
  1 = short (1/8 – 1/4 note)
  2 = medium (1/4 – 1 beat)
  3 = long (> 1 beat)

Note types:
  0 = normal strum
  1 = HOPO (forced)
  2 = tap  (fret 6 in chart)
  3 = open strum (fret 7 in chart)
"""

import os, sys, re, json, argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Chart parser ──────────────────────────────────────────────────────────────

DIFF_SECTIONS = {
    'easy':   'EasySingle',
    'medium': 'MediumSingle',
    'hard':   'HardSingle',
    'expert': 'ExpertSingle',
}

# HOPO window in ticks (at resolution 192): within 1/12 beat = ~forced HOPO threshold
HOPO_TICKS = 64   # roughly 1/3 beat at res=192

def parse_chart(chart_path):
    """
    Parse a notes.chart file into rich note events.
    Returns dict: {
        'resolution': int,
        'bpm': float,
        'bpm_events': [(tick, bpm), ...],
        'notes': {
            'expert': [
                {
                  'tick':    int,
                  'fret':    int (0-4),
                  'sustain': int (ticks),
                  'chord':   int or -1   (second fret if chord, else -1),
                  'type':    int (0=normal, 1=HOPO, 2=tap, 3=open)
                }, ...
            ], ...
        }
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
            tick = int(m.group(1))
            bpm  = int(m.group(2)) / 1000.0
            bpm_events.append((tick, bpm))
    global_bpm = bpm_events[0][1] if bpm_events else 120.0

    # HOPO flags — chart may explicitly mark HOPOs with fret 5 modifier
    # Note type 6 = tap, 7 = open in .chart format
    notes = {}
    for diff, section in DIFF_SECTIONS.items():
        sec_m = re.search(rf'\[{re.escape(section)}\]\s*\{{([^}}]*)\}}', text, re.S)
        if not sec_m:
            continue
        body = sec_m.group(1)

        # Collect all raw events at each tick: N lines and S lines
        raw = {}   # tick → {frets: set, sustains: dict, modifiers: set}
        for m in re.finditer(r'(\d+)\s*=\s*N\s+(\d+)\s+(\d+)', body):
            tick    = int(m.group(1))
            fret    = int(m.group(2))
            sustain = int(m.group(3))
            if tick not in raw:
                raw[tick] = {'frets': {}, 'modifiers': set()}
            if fret <= 4:
                raw[tick]['frets'][fret] = sustain
            elif fret == 5:
                raw[tick]['modifiers'].add('hopo')
            elif fret == 6:
                raw[tick]['modifiers'].add('tap')
            elif fret == 7:
                raw[tick]['modifiers'].add('open')

        if not raw:
            continue

        # Build note list sorted by tick
        ticks_sorted = sorted(raw.keys())
        diff_notes   = []

        for i, tick in enumerate(ticks_sorted):
            ev       = raw[tick]
            frets_d  = ev['frets']
            mods     = ev['modifiers']

            if 'open' in mods:
                # Open strum — no fret held
                diff_notes.append({
                    'tick': tick, 'fret': -1, 'sustain': 0,
                    'chord': -1, 'type': 3
                })
                continue

            if not frets_d:
                continue

            fret_list = sorted(frets_d.keys())
            main_fret = fret_list[0]
            sustain   = frets_d[main_fret]
            chord_fret = fret_list[1] if len(fret_list) > 1 else -1

            # Determine note type
            if 'tap' in mods:
                ntype = 2
            elif 'hopo' in mods:
                ntype = 1
            elif i > 0:
                # Auto-HOPO: within HOPO_TICKS of previous note, different fret
                prev_tick  = ticks_sorted[i-1]
                prev_frets = sorted(raw[prev_tick]['frets'].keys())
                prev_main  = prev_frets[0] if prev_frets else -1
                if (tick - prev_tick) <= HOPO_TICKS and main_fret != prev_main:
                    ntype = 1
                else:
                    ntype = 0
            else:
                ntype = 0

            diff_notes.append({
                'tick': tick, 'fret': main_fret, 'sustain': sustain,
                'chord': chord_fret, 'type': ntype
            })

        if diff_notes:
            notes[diff] = diff_notes

    return {
        'resolution': resolution,
        'bpm': global_bpm,
        'bpm_events': bpm_events,
        'notes': notes,
    }


# ── Sustain bucketing ─────────────────────────────────────────────────────────

def sustain_bucket(sustain_ticks, resolution):
    """Map raw sustain ticks → bucket 0-3."""
    eigth  = resolution // 2       # 1/8 note
    qtr    = resolution            # 1/4 note
    beat   = resolution            # 1 beat = 1/4 note at res=192
    if   sustain_ticks < eigth:    return 0   # none / very short
    elif sustain_ticks < qtr:      return 1   # short
    elif sustain_ticks < beat * 2: return 2   # medium
    else:                          return 3   # long


# ── Audio → beat-aligned mel ──────────────────────────────────────────────────

def audio_to_beat_mel(audio_path, bpm, bpm_events=None,
                       n_mels=128, hop_length=512, sr=22050):
    y, _ = librosa.load(str(audio_path), sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames',
                                                  bpm=bpm, trim=False)
    beat_times  = librosa.frames_to_time(beat_frames, sr=sr)
    actual_bpm  = float(np.atleast_1d(tempo)[0])
    tps         = (actual_bpm / 60.0) * 192

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           hop_length=hop_length,
                                           n_fft=2048, fmin=20, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Build 16th-note grid
    grid_times = []
    for i in range(len(beat_times)-1):
        gap = beat_times[i+1] - beat_times[i]
        for k in range(4):
            grid_times.append(beat_times[i] + gap * k / 4)
    if len(beat_times) > 0:
        grid_times.append(beat_times[-1])
    grid_times = np.array(sorted(set(grid_times)))

    mel_frames = []
    for t in grid_times:
        frame = int(librosa.time_to_frames(t, sr=sr, hop_length=hop_length))
        frame = max(0, min(frame, S_db.shape[1]-1))
        mel_frames.append(S_db[:, frame])

    mel   = np.stack(mel_frames, axis=0)
    mel   = (mel - mel.mean()) / (mel.std() + 1e-8)
    return torch.tensor(mel, dtype=torch.float32), grid_times, actual_bpm, tps


# ── Note → grid alignment ─────────────────────────────────────────────────────

def ticks_to_grid(notes, grid_times, tps, resolution, tolerance_sec=0.04):
    """
    Map rich note events → per-grid-position targets.
    Returns 5 tensors, each shape (seq_len,):
      note_t    float  0/1
      fret_t    long   0-4 or -1
      sustain_t long   bucket 0-3
      chord_t   long   0-4 or -1
      type_t    long   0-3
    """
    T = len(grid_times)
    note_t    = np.zeros(T, dtype=np.float32)
    fret_t    = np.full(T, -1, dtype=np.int64)
    sustain_t = np.zeros(T, dtype=np.int64)
    chord_t   = np.full(T, -1, dtype=np.int64)
    type_t    = np.zeros(T, dtype=np.int64)

    for ev in notes:
        t_sec = ev['tick'] / tps
        diffs = np.abs(grid_times - t_sec)
        idx   = int(diffs.argmin())
        if diffs[idx] > tolerance_sec:
            continue

        note_t[idx]    = 1.0
        fret           = ev['fret'] if ev['fret'] >= 0 else 0
        fret_t[idx]    = min(fret, 4)
        sustain_t[idx] = sustain_bucket(ev['sustain'], resolution)
        chord_t[idx]   = min(ev['chord'], 4) if ev['chord'] >= 0 else -1
        type_t[idx]    = ev['type']

    return (torch.tensor(note_t),
            torch.tensor(fret_t),
            torch.tensor(sustain_t),
            torch.tensor(chord_t),
            torch.tensor(type_t))


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

        T = len(mel)
        note_targets    = {}
        fret_targets    = {}
        sustain_targets = {}
        chord_targets   = {}
        type_targets    = {}

        def _trim_pad(t, fill):
            if len(t) >= T: return t[:T]
            pad = torch.full((T - len(t),), fill,
                             dtype=t.dtype)
            return torch.cat([t, pad])

        for diff, diff_notes in chart['notes'].items():
            nt, ft, st, ct, tt = ticks_to_grid(
                diff_notes, grid_times, tps, chart['resolution'])
            note_targets[diff]    = _trim_pad(nt, 0.0)
            fret_targets[diff]    = _trim_pad(ft, -1)
            sustain_targets[diff] = _trim_pad(st, 0)
            chord_targets[diff]   = _trim_pad(ct, -1)
            type_targets[diff]    = _trim_pad(tt, 0)

        meta = {}
        if (song_dir / 'meta.json').exists():
            meta = json.loads((song_dir / 'meta.json').read_text())

        return {
            'mel':             mel,
            'note_targets':    note_targets,
            'fret_targets':    fret_targets,
            'sustain_targets': sustain_targets,
            'chord_targets':   chord_targets,
            'type_targets':    type_targets,
            'bpm':             bpm,
            'title':           meta.get('title', song_dir.name),
            'artist':          meta.get('artist', ''),
        }
    except Exception as e:
        print(f'  [warn] {song_dir.name}: {e}')
        return None


# ── Dataset class ─────────────────────────────────────────────────────────────

class ChartDataset(Dataset):
    """
    Loads preprocessed .pt files.
    Returns windowed tensors: mel, note, fret, sustain, chord, type, diff_idx

    cache_size: number of song files to keep in memory (LRU).
    Set to len(dataset.files) or higher to cache everything and eliminate
    all disk I/O after the first epoch.
    """
    DIFFS = ['easy', 'medium', 'hard', 'expert']

    def __init__(self, processed_dir, window_beats=16, stride_beats=4,
                 sixteenths_per_beat=4, cache_size=256):
        self.processed_dir = Path(processed_dir)
        self.window_size   = window_beats * sixteenths_per_beat
        self.stride        = stride_beats * sixteenths_per_beat
        self.files         = sorted(self.processed_dir.glob('*.pt'))
        self.cache_size    = cache_size
        self._cache        = OrderedDict()
        self.samples       = []
        self._build_index()

    def _load(self, fi):
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        data = torch.load(str(self.files[fi]), map_location='cpu')
        self._cache[fi] = data
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def _build_index(self):
        pbar = tqdm(enumerate(self.files), total=len(self.files),
                    desc='Indexing', unit='file', leave=False, dynamic_ncols=True)
        for fi, _ in pbar:
            try:
                data = self._load(fi)
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
        data  = self._load(fi)
        end   = start + self.window_size
        diff  = self.DIFFS[diff_i]
        mel     = data['mel'][start:end]
        notes   = data['note_targets'][diff][start:end]
        frets   = data['fret_targets'][diff][start:end]
        sustain = data['sustain_targets'][diff][start:end]
        chord   = data['chord_targets'][diff][start:end]
        ntype   = data['type_targets'][diff][start:end]
        diff_t  = torch.tensor(diff_i, dtype=torch.long)
        return mel, notes, frets, sustain, chord, ntype, diff_t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',    default='./dataset',   help='Scraped dataset dir')
    ap.add_argument('--out',     default='./processed', help='Output .pt dir')
    ap.add_argument('--mels',    type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()
                and (d/'notes.chart').exists() and (d/'song.opus').exists()]
    song_dirs = [d for d in all_dirs if not (out_dir / f'{d.name}.pt').exists()]
    skip_count = len(all_dirs) - len(song_dirs)
    if skip_count:
        print(f'Skipping {skip_count} already-processed songs')
    print(f'Processing {len(song_dirs)} songs → {out_dir}')
    print('Targets: note, fret (0-4), sustain (4 buckets), chord fret, note type (4 types)')

    from concurrent.futures import ProcessPoolExecutor, as_completed

    ok = fail = 0
    pbar = tqdm(total=len(song_dirs), desc='Preprocess', unit='song',
                colour='cyan', dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, d, args.mels): d for d in song_dirs}
        for future in as_completed(futures):
            song_dir = futures[future]
            result   = future.result()
            if result:
                out_path = out_dir / f'{song_dir.name}.pt'
                torch.save(result, str(out_path))
                ok += 1
            else:
                fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok, fail=fail)

    pbar.close()
    print(f'\nDone: {ok} saved, {fail} failed → {out_dir}')


if __name__ == '__main__':
    main()
