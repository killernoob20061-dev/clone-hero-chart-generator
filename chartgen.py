"""
Clone Hero chart generator v6 — human-quality, solo-aware charts.
Usage: python chartgen.py <path_to_mp3> [<path_to_mp3> ...]
       python chartgen.py --frets 0,1,2,3,4 <song.mp3>   # all 5 frets
       python chartgen.py --frets 0,1,2     <song.mp3>   # GRY only (default)

v6 highlights:
- Note importance ranking: downbeats, chord roots, melody peaks scored first
- Phrase boundary detection: 2/4-bar motifs repeat identically across the song
- Guitar solo detection (pyin voiced prob + centroid + onset density)
- Note-for-note solo charting with high-resolution pitch tracking
- Genre-specific pattern generators (metal/pop/electronic/rock)
- Chromagram chord detection — chords only where music has actual chords
- Sustain fill on solo notes — every note held to the next
- Anti-HOPO violations, max 2-fret ergonomic jumps
- Configurable fret palette via --frets (e.g. 0,1,2 or 0,1,2,3,4)
  Frets: 0=Green 1=Red 2=Yellow 3=Blue 4=Orange
"""

import sys, os, re, subprocess, io, argparse
import librosa
import numpy as np
import whisper
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
from mutagen.id3 import ID3
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d, median_filter as sp_medfilt
from sklearn.preprocessing import normalize as sk_normalize

try:
    import madmom
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.features.onsets    import CNNOnsetProcessor, OnsetPeakPickingProcessor
    HAS_MADMOM = True
except ImportError:
    HAS_MADMOM = False

# Optional trained ChartNet model (set via --model flag or CHARTNET_MODEL env var)
try:
    from chartmodel import ChartNet, predict_chart, load_model as _load_chartnet
    HAS_CHARTNET = True
except ImportError:
    HAS_CHARTNET = False

_chartnet_model = None
_chartnet_device = 'cpu'

def load_chartnet(checkpoint_path):
    global _chartnet_model, _chartnet_device
    if not HAS_CHARTNET:
        print('  [warn] chartmodel.py not found — running without neural model')
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Loading ChartNet from {checkpoint_path} on {device}...')
    _chartnet_model  = _load_chartnet(checkpoint_path, device=device)
    _chartnet_device = device
    print('  ChartNet loaded — neural note prediction active')

# ── Constants ────────────────────────────────────────────────────────────────
RES  = 192
BEAT = RES        # quarter note
HALF = RES // 2   # 8th
QTR  = RES // 4   # 16th
EGTH = RES // 8   # 32nd
HOPO = HALF       # auto-HOPO threshold in CH

OUT_DIR = r'C:\Users\MrSchneider\OneDrive\Tiedostot\Clone Hero\Songs'

# ── Tick helpers ─────────────────────────────────────────────────────────────

def t2tick(t, tps):  return int(round(t * tps))
def snap(t, g):       return int(round(t / g)) * g

def filter_gap(ticks, mg):
    out, prev = [], -999999
    for t in ticks:
        if t - prev >= mg:
            out.append(t); prev = t
    return out

# ── Demucs ───────────────────────────────────────────────────────────────────

_demucs_model = None

def get_demucs():
    global _demucs_model
    if _demucs_model is None:
        print('  Loading Demucs htdemucs...')
        _demucs_model = get_model('htdemucs')
        _demucs_model.eval()
    return _demucs_model

def load_and_separate(mp3_path):
    print('  Demucs stem separation...')
    y_mix, sr_mix = librosa.load(mp3_path, sr=44100, mono=False)
    if y_mix.ndim == 1:
        y_mix = np.stack([y_mix, y_mix])
    model = get_demucs()
    wav = torch.tensor(y_mix, dtype=torch.float32)
    wav = convert_audio(wav, sr_mix, model.samplerate, model.audio_channels).unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(model, wav, progress=True)[0]
    stems = {n: sources[i].mean(0).numpy() for i, n in enumerate(model.sources)}
    SR  = 22050
    rs  = lambda a: librosa.resample(a, orig_sr=model.samplerate, target_sr=SR) if a is not None else None
    y   = rs(y_mix.mean(0))
    yg  = rs(stems.get('other'))
    yp  = rs(stems.get('drums'))
    yb  = rs(stems.get('bass'))      # bass stem — used for chord root anchoring
    if yg is None or np.abs(yg).max() < 1e-4:
        yg, yph = librosa.effects.hpss(y, margin=3.0)
        if yp is None: yp = yph
    if yp is None:
        _, yp = librosa.effects.hpss(y, margin=3.0)
    if yb is None:
        yb = np.zeros_like(y)
    dur = librosa.get_duration(y=y, sr=SR)
    return y, yg, yp, yb, SR, dur

# ── Genre detection ──────────────────────────────────────────────────────────

def detect_genre(y, sr):
    centroid   = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    rms_mean   = float(librosa.feature.rms(y=y)[0].mean())
    dur        = librosa.get_duration(y=y, sr=sr)
    n_onsets   = len(librosa.onset.onset_detect(y=y, sr=sr))
    onset_rate = n_onsets / max(dur, 1)
    _, bf      = librosa.beat.beat_track(y=y, sr=sr)
    bt         = librosa.frames_to_time(bf, sr=sr)
    regularity = 1.0 / (np.std(np.diff(bt)) / (np.mean(np.diff(bt)) + 1e-6) + 0.1) if len(bt) > 4 else 1.0
    if   onset_rate > 7.0 and centroid < 2500 and rms_mean > 0.04: genre = 'metal'
    elif centroid > 3500   and regularity > 6.0:                    genre = 'electronic'
    elif centroid > 2800   and onset_rate < 6.0:                    genre = 'pop'
    else:                                                            genre = 'rock'
    print(f'  Genre: {genre}  centroid={centroid:.0f}  onset_rate={onset_rate:.1f}/s')
    return genre

# ── Beat tracking ─────────────────────────────────────────────────────────────

def analyze_tempo(y_perc, sr, mp3_path):
    if HAS_MADMOM and mp3_path:
        try:
            act = RNNDownBeatProcessor()(mp3_path)
            res = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3,4], min_bpm=55, max_bpm=215,
                fps=100, transition_lambda=100, threshold=0.05, correct=True)(act)
            bt  = res[:,0]
            db  = res[res[:,1]==1, 0]
            bpm = 60.0 / float(np.median(np.diff(bt))) if len(bt) > 1 else 120.0
            print(f'  madmom BPM={bpm:.1f}  downbeats={len(db)}')
            return bpm, bt, _local_bpm(bt), db
        except Exception as e:
            print(f'  madmom beat fail ({e})')
    tempo, bf = librosa.beat.beat_track(y=y_perc, sr=sr, units='frames', trim=False)
    bt = librosa.frames_to_time(bf, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, bt, _local_bpm(bt), bt[::4]

def _local_bpm(bt):
    evs, win = [], max(8, len(bt)//8)
    for i in range(0, len(bt)-win, win//2):
        seg = bt[i:i+win]
        if len(seg) < 2: continue
        evs.append((seg[0], 60.0/float(np.median(np.diff(seg)))))
    return evs

def build_sync_track(bpm_events, tps, global_bpm):
    entries = {0: int(round(global_bpm*1000))}
    prev = global_bpm
    for t, b in sorted(bpm_events, key=lambda x: x[0]):
        rb = round(b*2)/2
        if abs(rb-prev) > 1.5:
            entries[max(1, int(round(t*tps)))] = int(round(rb*1000))
            prev = rb
    lines = ['[SyncTrack]', '{', '  0 = TS 4']
    for tick in sorted(entries):
        lines.append(f'  {tick} = B {entries[tick]}')
    lines.append('}')
    return '\n'.join(lines)

# ── Beat grids ────────────────────────────────────────────────────────────────

def build_grids(beat_times, tps):
    beats, eighths, sixteenths = [], [], []
    for i in range(len(beat_times)-1):
        g = beat_times[i+1] - beat_times[i]
        beats.append(snap(t2tick(beat_times[i], tps), BEAT))
        for n in range(2):
            eighths.append(snap(t2tick(beat_times[i]+g*n/2, tps), HALF))
        for n in range(4):
            sixteenths.append(snap(t2tick(beat_times[i]+g*n/4, tps), QTR))
    return sorted(set(beats)), sorted(set(eighths)), sorted(set(sixteenths))

# ── Onset detection ───────────────────────────────────────────────────────────

def get_onsets(y, y_perc, sr, tps, mp3_path):
    if HAS_MADMOM and mp3_path:
        try:
            act = CNNOnsetProcessor()(mp3_path)
            raw = OnsetPeakPickingProcessor(
                fps=100, threshold=0.25, smooth=0.04,
                pre_avg=0.1, post_avg=0.05, pre_max=0.03, post_max=0.07,
                combine=0.03)(act)
            return sorted(set(snap(t2tick(t, tps), QTR) for t in raw))
        except Exception as e:
            print(f'  madmom onset fail ({e})')
    o1 = librosa.onset.onset_detect(y=y_perc, sr=sr, units='frames', delta=0.03, backtrack=True)
    o2 = librosa.onset.onset_detect(y=y,      sr=sr, units='frames', delta=0.025, backtrack=True)
    times = librosa.frames_to_time(np.union1d(o1, o2), sr=sr)
    return sorted(set(snap(t2tick(t, tps), QTR) for t in times))

# ── Intensity curve ───────────────────────────────────────────────────────────

def build_intensity_curve(y, sr, beat_times, tps):
    """
    Per-beat intensity score (0.0–1.0) based on RMS energy of the full mix.
    Used to scale note density, sustain length, and fret jump range dynamically.
    Returns dict: beat_tick → intensity (float 0..1)
    """
    hop  = 512
    rms  = librosa.feature.rms(y=y, hop_length=hop)[0]
    # Smooth with a 3-beat window to avoid single-frame spikes
    rms_s = sp_medfilt(rms.astype(float), size=max(3, int(sr * 0.1 / hop) | 1))
    rms_max = rms_s.max() + 1e-8

    intensity = {}
    for bt in beat_times:
        frame = min(int(librosa.time_to_frames(bt, sr=sr, hop_length=hop)), len(rms_s)-1)
        tick  = snap(t2tick(bt, tps), BEAT)
        intensity[tick] = float(rms_s[frame]) / rms_max
    return intensity

# ── Kick / snare separation ───────────────────────────────────────────────────

def build_drum_maps(y_perc, sr, tps):
    """
    Separate kick (low) and snare (mid) onsets from the drum stem.
    Returns:
      kick_ticks  — set of ticks where kick drum fires  (anchor note placement)
      snare_ticks — set of ticks where snare fires      (beat 2 & 4 weight)
    """
    hop = 512
    # Low-pass for kick (~20–180 Hz), band-pass for snare (~180–5000 Hz)
    S     = np.abs(librosa.stft(y_perc, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    kick_mask  = freqs <= 180
    snare_mask = (freqs > 180) & (freqs <= 5000)

    kick_signal  = S[kick_mask].sum(0)
    snare_signal = S[snare_mask].sum(0)

    def _onsets_from_signal(sig):
        # Normalise and run onset on the filtered energy envelope
        sig_n = sig / (sig.max() + 1e-8)
        frames = librosa.onset.onset_detect(
            onset_envelope=sig_n, sr=sr, hop_length=hop,
            delta=0.1, backtrack=True)
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        return set(snap(t2tick(t, tps), QTR) for t in times)

    kick_ticks  = _onsets_from_signal(kick_signal)
    snare_ticks = _onsets_from_signal(snare_signal)
    return kick_ticks, snare_ticks

# ── Bass pitch map for chord root anchoring ───────────────────────────────────

def build_bass_root_map(y_bass, sr, tps):
    """
    Track the bass pitch per tick to identify chord roots.
    Returns dict: tick → bass_midi (float, 0 if unvoiced).
    The fret assigner uses this to prefer the lowest pool fret at root changes.
    """
    hop = 512
    f0, voiced, vprob = librosa.pyin(
        y_bass, sr=sr,
        fmin=librosa.note_to_hz('B0'), fmax=librosa.note_to_hz('G3'),
        frame_length=2048, hop_length=hop, fill_na=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = np.where(voiced & (f0 > 0) & (vprob >= 0.3),
                        12*np.log2(np.maximum(f0, 1e-6)/440)+69, 0.0)
    midi_s = sp_medfilt(midi, size=5)

    def bass_at(tick):
        t_s   = tick / tps
        frame = max(0, min(int(round(t_s * sr / hop)), len(midi_s)-1))
        return float(midi_s[frame])

    return bass_at

# ── Pitch map (pyin) ──────────────────────────────────────────────────────────

def build_pitch_map(y_guit, sr, tps):
    print('  pyin pitch analysis...')
    hop = 256
    f0, voiced, vprob = librosa.pyin(
        y_guit, sr=sr,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        frame_length=2048, hop_length=hop, fill_na=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = np.where(voiced & (f0>0) & (vprob>=0.4),
                        12*np.log2(np.maximum(f0,1e-6)/440)+69, 0.0)
    midi_s = sp_medfilt(midi, size=7)
    midi_s = np.where(midi > 0, midi_s, 0.0)
    vm = midi_s[midi_s>0]
    p33, p66 = (float(np.percentile(vm,33)), float(np.percentile(vm,66))) if len(vm)>10 else (55.0,67.0)
    print(f'  Pitch p33={p33:.1f} p66={p66:.1f}  voiced={len(vm)} frames')
    S  = np.abs(librosa.stft(y_guit, hop_length=hop, n_fft=2048))
    ff = librosa.fft_frequencies(sr=sr)
    le = S[ff<=300].sum(0); me = S[(ff>300)&(ff<=1500)].sum(0); he = S[ff>1500].sum(0)
    def fret_at(tick):
        t_s   = tick/tps
        frame = max(0, min(int(round(t_s*sr/hop)), len(midi_s)-1))
        m     = midi_s[frame]
        if m > 0: return 0 if m<p33 else (1 if m<p66 else 2)
        sf = min(int(librosa.time_to_frames(t_s, sr=sr, hop_length=hop)), le.shape[0]-1)
        return int(np.argmax([le[sf], me[sf], he[sf]]))
    return fret_at

# ── Guitar solo detection ─────────────────────────────────────────────────────

def detect_solo_regions(y_guit, y_full, sr, beat_times, tps, bpm):
    """
    Detect guitar solo regions by scoring each frame on five features:
      1. pyin voiced probability     — high = clear single-note melody line
      2. Spectral centroid (guitar)  — high = upper-register lead playing
      3. Onset density               — high = fast note runs
      4. Guitar RMS                  — high = guitar is prominent/loud
      5. Chroma peakiness            — high = single pitch (not chord wash)

    Regions where the combined score exceeds the 72nd percentile for
    at least 4 beats are classified as solos.
    Returns list of (start_tick, end_tick).
    """
    print('  Detecting guitar solo regions...')
    hop = 512

    # 1. Voiced probability from pyin
    _, voiced, vprob = librosa.pyin(
        y_guit, sr=sr,
        fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'),
        frame_length=2048, hop_length=hop, fill_na=0.0)
    vprob_s = uniform_filter1d(vprob.astype(float), size=20)

    # 2. Spectral centroid of guitar stem
    cent = librosa.feature.spectral_centroid(y=y_guit, sr=sr, hop_length=hop)[0]

    # 3. Onset density (rolling envelope)
    oe = librosa.onset.onset_strength(y=y_guit, sr=sr, hop_length=hop)
    od = uniform_filter1d(oe, size=30)

    # 4. Guitar RMS
    rms_g = librosa.feature.rms(y=y_guit, hop_length=hop)[0]

    # 5. Chroma peakiness: max_bin / mean → peaky = single note
    chroma = librosa.feature.chroma_cqt(y=y_guit, sr=sr, hop_length=hop)
    cp     = chroma.max(axis=0) / (chroma.mean(axis=0) + 1e-8)

    # Normalise all to [0,1]
    def norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    min_len = min(len(vprob_s), len(cent), len(od), len(rms_g), len(cp))
    score   = (0.30 * norm(vprob_s[:min_len]) +
               0.20 * norm(cent[:min_len])    +
               0.20 * norm(od[:min_len])      +
               0.15 * norm(rms_g[:min_len])   +
               0.15 * norm(cp[:min_len]))
    score   = uniform_filter1d(score, size=40)   # ~1 s smoothing at hop=512/sr=22050

    threshold = np.percentile(score, 72)
    above     = score > threshold

    # Minimum solo length in frames: 4 beats
    beat_dur_sec = 60.0 / max(bpm, 40)
    min_frames   = int(4 * beat_dur_sec * sr / hop)

    solo_regions = []
    in_solo = False; solo_start = 0
    for i, a in enumerate(above):
        if a and not in_solo:
            in_solo = True; solo_start = i
        elif not a and in_solo:
            if i - solo_start >= min_frames:
                solo_regions.append((
                    librosa.frames_to_time(solo_start, sr=sr, hop_length=hop),
                    librosa.frames_to_time(i,          sr=sr, hop_length=hop)))
            in_solo = False
    if in_solo and min_len - solo_start >= min_frames:
        solo_regions.append((
            librosa.frames_to_time(solo_start, sr=sr, hop_length=hop),
            librosa.frames_to_time(min_len-1,  sr=sr, hop_length=hop)))

    # Merge regions within 2 beats of each other
    merged = []
    for s, e in sorted(solo_regions):
        if merged and s - merged[-1][1] < 2 * beat_dur_sec:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Convert to ticks and filter out very short stubs
    tick_regions = []
    for s, e in merged:
        ts = snap(t2tick(s, tps), BEAT)
        te = snap(t2tick(e, tps), BEAT)
        if te - ts >= BEAT * 4:      # at least 4 beats long
            tick_regions.append((ts, te))

    if tick_regions:
        print(f'  Solos detected: {len(tick_regions)} region(s)')
        for ts, te in tick_regions:
            print(f'    tick {ts}–{te}  (~{ts/tps:.1f}s – {te/tps:.1f}s)')
    else:
        print('  No clear guitar solo detected')
    return tick_regions

# ── Note-for-note solo charting ───────────────────────────────────────────────

def chart_solo_region(y_guit, sr, tps, start_tick, end_tick, diff, bpm, fret_pool=None):
    """
    Chart a solo region note-for-note using high-resolution pyin.
    Every voiced note in the solo becomes a chart note.
    Notes are held (sustain) right up to the next note.
    """
    hop    = 128   # ~5.8 ms resolution — fine enough for fast solos
    t_s    = start_tick / tps
    t_e    = end_tick   / tps
    s0     = int(t_s * sr)
    s1     = min(int(t_e * sr), len(y_guit))
    y_seg  = y_guit[s0:s1]

    if len(y_seg) < sr // 8:
        return []

    # High-res pyin on the isolated solo segment
    f0, voiced, vprob = librosa.pyin(
        y_seg, sr=sr,
        fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'),
        frame_length=1024, hop_length=hop, fill_na=0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        midi = np.where(voiced & (f0 > 0) & (vprob >= 0.35),
                        12*np.log2(np.maximum(f0, 1e-6)/440)+69, 0.0)

    midi_s = sp_medfilt(midi, size=5)
    midi_s = np.where(midi > 0, midi_s, 0.0)

    # Detect note boundaries:
    # - silence → voiced transition
    # - pitch change > 1.5 semitones while voiced
    note_frames = []
    prev_m = 0.0
    for i, m in enumerate(midi_s):
        if m > 0:
            if prev_m == 0:                     # note onset from silence
                note_frames.append(i)
            elif abs(m - prev_m) > 1.5:         # pitch jump → new note
                note_frames.append(i)
        prev_m = m if m > 0 else prev_m

    # Fallback if pyin found nothing
    if not note_frames:
        frames = librosa.onset.onset_detect(
            y=y_seg, sr=sr, units='frames', hop_length=hop, delta=0.02, backtrack=True)
        note_frames = list(frames)

    if not note_frames:
        return []

    if fret_pool is None:
        fret_pool = [0, 1, 2]
    # Easy uses only the lower half of the pool
    if diff == 'easy':
        fret_pool = fret_pool[:max(1, (len(fret_pool)+1)//2)]
    max_idx = len(fret_pool) - 1

    # Pitch range for this solo section
    vm = midi_s[midi_s > 0]
    # Build percentile thresholds for each fret index boundary
    pcts = [float(np.percentile(vm, int(100*k/len(fret_pool)))) for k in range(1, len(fret_pool))] if len(vm) > 5 else []

    min_gap  = {
        'expert': QTR,
        'hard':   HALF,
        'medium': HALF,
        'easy':   BEAT,
    }[diff]

    raw_notes = []
    prev_tick = -99999
    prev_idx  = -1

    for frame in note_frames:
        t_abs = t_s + frame * hop / sr
        tick  = snap(t2tick(t_abs, tps), QTR if diff in ('expert','hard') else HALF)

        if tick - prev_tick < min_gap:
            continue
        if tick < start_tick or tick >= end_tick:
            continue

        m = midi_s[min(frame, len(midi_s)-1)]
        if m > 0 and pcts:
            idx = sum(1 for p in pcts if m >= p)
        else:
            # Map by position in solo
            idx = int((frame / max(len(midi_s),1)) * (max_idx+1)) % (max_idx+1)
        idx = min(idx, max_idx)

        # Avoid same-fret HOPO repeats
        if idx == prev_idx and tick - prev_tick <= HOPO:
            idx = min(prev_idx+1, max_idx) if prev_idx < max_idx else max(prev_idx-1, 0)

        raw_notes.append((tick, fret_pool[idx], 0))
        prev_tick = tick
        prev_idx  = idx

    if not raw_notes:
        return []

    # Aggressive sustains: hold each solo note all the way to the next
    result = []
    for i, (t, f, _) in enumerate(raw_notes):
        if i < len(raw_notes)-1:
            gap = raw_notes[i+1][0] - t
            sus = max(0, gap - EGTH)   # leave tiny gap before next note
        else:
            sus = BEAT * 2             # last note holds for 2 beats
        result.append((t, f, sus))

    return result

# ── Section detection ─────────────────────────────────────────────────────────

def detect_sections(y, sr, beat_times, tps, solo_regions=None):
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    oe  = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    sm  = uniform_filter1d(rms*oe, size=int(sr/hop*4))
    sm /= sm.max()+1e-8
    n   = min(16, max(4, int(librosa.get_duration(y=y, sr=sr)/20)))
    sb  = np.linspace(0, len(sm)-1, n+1).astype(int)
    se  = [sm[sb[i]:sb[i+1]].mean() for i in range(n)]
    ea  = np.array(se)
    lo, hi = np.percentile(ea,30), np.percentile(ea,70)

    solo_tick_set = set()
    if solo_regions:
        for ts, te in solo_regions:
            for tk in range(ts, te, BEAT):
                solo_tick_set.add(tk)

    events, dbb, prev, cnt = [], {}, None, {'v':0,'c':0,'b':0,'s':0}
    for i, e in enumerate(se):
        st   = librosa.frames_to_time(sb[i], sr=sr, hop_length=hop)
        tick = snap(t2tick(st, tps), BEAT)
        # Check if this segment overlaps a solo region
        if tick in solo_tick_set:
            cnt['s'] += 1; lbl = f'Solo {cnt["s"]}'; dns = 1.5; stype = 'solo'
        elif i == 0:               lbl,dns,stype = 'Intro',0.6,'intro'
        elif i == n-1:             lbl,dns,stype = 'Outro',0.7,'outro'
        elif e >= hi:
            cnt['c']+=1;           lbl,dns,stype = f'Chorus {cnt["c"]}',1.3,'chorus'
        elif e <= lo:
            cnt['v']+=1;           lbl,dns,stype = f'Verse {cnt["v"]}',0.8,'verse'
        else:
            cnt['b']+=1;           lbl,dns,stype = f'Bridge {cnt["b"]}',1.0,'bridge'
        if stype != prev:
            events.append((tick, f'section {lbl}')); prev = stype
        et = librosa.frames_to_time(sb[i+1], sr=sr, hop_length=hop)
        for bt in beat_times:
            if st <= bt < et:
                dbb[snap(t2tick(bt,tps),BEAT)] = dns
    return events, dbb

# ── Star power ────────────────────────────────────────────────────────────────

def make_star_power(beat_ticks, tps, duration, solo_regions=None):
    total  = int(duration*tps); margin = total//10
    # Prefer to place SP phrases right before solo regions
    sp_events = []
    solo_preamble = set()
    if solo_regions:
        for ts, _ in solo_regions:
            # SP phrase 4 beats before the solo = natural build
            pre = ts - BEAT*4
            if pre > margin:
                sp_events.append((pre, f'S 2 {BEAT*4}'))
                for tk in range(pre, ts):
                    solo_preamble.add(tk)

    usable = [t for t in beat_ticks if margin<t<total-margin and t not in solo_preamble]
    if not usable:
        return sp_events
    n_more = max(0, 7 - len(sp_events))
    step   = max(1, len(usable)//(n_more+1))
    for i in range(1, n_more+1):
        idx = i*step
        if idx < len(usable):
            sp_events.append((usable[idx], f'S 2 {BEAT*4}'))
    return sp_events

# ── Chromagram chord detection ────────────────────────────────────────────────

def build_chord_guide(y, y_guit, sr, tps, beat_ticks):
    """
    Use chromagram to detect when the music has multiple simultaneous pitches.
    Returns set of ticks where a chord is musically appropriate.
    """
    hop    = 512
    chroma = librosa.feature.chroma_cqt(y=y_guit, sr=sr, hop_length=hop)
    # "Chord frame": two or more chroma bins above 0.5 * peak
    chord_frames = set()
    for t in range(chroma.shape[1]):
        col  = chroma[:, t]
        peak = col.max()
        if peak < 0.3: continue       # too quiet
        strong = (col > 0.5*peak).sum()
        if strong >= 2:
            chord_frames.add(t)

    # Map beat ticks to chord or not
    chord_ticks = set()
    for tick in beat_ticks:
        t_s   = tick / tps
        frame = min(int(librosa.time_to_frames(t_s, sr=sr, hop_length=hop)),
                    chroma.shape[1]-1)
        if frame in chord_frames:
            chord_ticks.add(tick)
    return chord_ticks

# ── Fret assignment helpers ───────────────────────────────────────────────────
# All internal logic works with indices 0..N-1 into fret_pool.
# The final remap translates index → actual Clone Hero fret number.
# Default pool [0,1,2] = Green/Red/Yellow.  [0,1,2,3,4] = all five frets.

FRET_NAMES = {0:'Green', 1:'Red', 2:'Yellow', 3:'Blue', 4:'Orange'}

def _pool_contour(contour, n_frets):
    """Rescale a 0-based contour (originally 0..2) to 0..n_frets-1."""
    return np.clip(contour * (n_frets - 1) / 2.0, 0, n_frets - 1)

def _ergonomic(frets, max_idx):
    """Post-pass: cap index jumps > 2, respect pool bounds."""
    r = list(frets)
    for i in range(1, len(r)):
        if abs(r[i]-r[i-1]) > 2:
            r[i] = r[i-1] + (2 if r[i]>r[i-1] else -2)
            r[i] = max(0, min(r[i], max_idx))
    return r

def _assign_metal(ticks, contour, max_idx, dbb, tps):
    frets, base = [], contour[0] if len(contour) else 0
    for i, t in enumerate(ticks):
        if t % (BEAT*4) < QTR:
            base = min(int(contour[i]), max_idx)
        if t % BEAT < QTR and i > 0:
            d = int(contour[i]) - base
            if d != 0 and abs(d) <= 1:
                base = max(0, min(max_idx, base+int(np.sign(d))))
        frets.append(base)
    return frets

def _assign_pop(ticks, contour, max_idx):
    raw = [int(max(0, min(max_idx, v))) for v in contour]
    sm  = list(sp_medfilt(np.array(raw, dtype=float), size=5))
    r   = [int(max(0, min(max_idx, round(v)))) for v in sm]
    out = [r[0]]
    for i in range(1, len(r)):
        d = r[i]-out[-1]
        out.append(out[-1]+int(np.sign(d)) if abs(d)>1 else r[i])
    return out

def _assign_electronic(ticks, contour, max_idx, bpm):
    # Scale built-in patterns to available index range
    base_patterns = [[0,1,2,1],[2,1,0,1],[0,2,1,2],[1,0,1,2],[0,1,0,2],[2,0,1,0]]
    dom = int(np.median(contour)) if len(contour) else 1
    pat = [min(int(round(f * max_idx / 2)), max_idx) for f in base_patterns[dom % len(base_patterns)]]
    return [pat[((t//BEAT)%4*2+i) % len(pat)] for i, t in enumerate(ticks)]

def _assign_rock(ticks, contour, max_idx, dbb, tps):
    frets, prev, prev_dir = [], -1, 0
    for i, t in enumerate(ticks):
        dns = dbb.get(snap(t,BEAT), 1.0)
        pf  = min(int(contour[i]), max_idx)
        gap = (ticks[i+1]-t) if i<len(ticks)-1 else BEAT
        if   gap <= EGTH and prev >= 0: f = prev
        elif dns >= 1.3:                f = pf
        elif dns <= 0.75:               f = 0 if prev!=0 else min(1,max_idx)
        else:
            au = min(prev+1,max_idx) if prev>=0 else pf
            ad = max(prev-1,0)       if prev>=0 else pf
            f  = (au if pf>prev else ad) if pf!=prev else (au if prev_dir<=0 else ad)
        if prev>=0 and abs(f-prev)>2: f=prev+(2 if f>prev else -2); f=max(0,min(f,max_idx))
        if f==prev and gap<=HOPO and prev>=0:
            f = min(prev+1,max_idx) if prev<max_idx else max(prev-1,0)
        prev_dir=f-prev if prev>=0 else 0; prev=f; frets.append(f)
    return frets

def assign_frets_genre(ticks, fret_at, genre, dbb, tps, bpm=120, fret_pool=None,
                       intensity=None, bass_at=None, kick_ticks=None, snare_ticks=None):
    """
    Assign frets using genre patterns, then remap through fret_pool.
    intensity  — dict tick→float(0..1): loud sections allow wider fret jumps
    bass_at    — callable tick→midi: bass root anchors lowest fret at chord changes
    kick_ticks — set: kick hits snap to current base fret (steady, grounded feel)
    snare_ticks— set: snare hits nudge up one index (accentuate the backbeat)
    """
    if fret_pool is None:
        fret_pool = [0, 1, 2]
    if intensity is None:
        intensity = {}
    if kick_ticks is None:
        kick_ticks = set()
    if snare_ticks is None:
        snare_ticks = set()
    if not ticks:
        return []

    max_idx = len(fret_pool) - 1
    raw_contour = np.array([fret_at(t) for t in ticks], dtype=float)
    contour = _pool_contour(raw_contour, len(fret_pool))

    if   genre == 'metal':      indices = _assign_metal(ticks, contour, max_idx, dbb, tps)
    elif genre == 'electronic': indices = _assign_electronic(ticks, contour, max_idx, bpm)
    elif genre == 'pop':        indices = _assign_pop(ticks, contour, max_idx)
    else:                       indices = _assign_rock(ticks, contour, max_idx, dbb, tps)

    # ── Post-pass: intensity-driven jump expansion ─────────────────────────
    # In loud sections (intensity > 0.7) allow jumps up to 3; quiet < 0.3 → max 1
    for i in range(1, len(indices)):
        t    = ticks[i]
        lvl  = intensity.get(snap(t, BEAT), 0.5)
        max_jump = 1 if lvl < 0.3 else (3 if lvl > 0.7 else 2)
        if abs(indices[i] - indices[i-1]) > max_jump:
            indices[i] = indices[i-1] + (max_jump if indices[i] > indices[i-1] else -max_jump)
            indices[i] = max(0, min(max_idx, indices[i]))

    # ── Bass root anchoring ────────────────────────────────────────────────
    # When the bass note changes significantly, snap to index 0 (lowest fret)
    if bass_at is not None:
        prev_bass = None
        for i, t in enumerate(ticks):
            bm = bass_at(t)
            if bm > 0:
                if prev_bass is not None and abs(bm - prev_bass) >= 2.0:
                    # Root note change → anchor to lowest pool fret
                    indices[i] = 0
                prev_bass = bm

    # ── Kick / snare adjustments ───────────────────────────────────────────
    for i, t in enumerate(ticks):
        if t in kick_ticks:
            # Kick: hold current index (grounding effect — no jump on kick)
            if i > 0:
                indices[i] = indices[i-1]
        elif t in snare_ticks and max_idx > 0:
            # Snare: nudge one step up for a backbeat accent
            indices[i] = min(indices[i] + 1, max_idx)

    indices = _ergonomic(indices, max_idx)
    # Remap index → actual fret number
    return [(ticks[i], fret_pool[indices[i]], 0) for i in range(len(ticks))]

# ── SSM phrase replication ────────────────────────────────────────────────────

def detect_repeated_phrases(y, sr, tps):
    print('  SSM phrase analysis...')
    hop = 512
    try:
        chroma = sk_normalize(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop), axis=0)
        n  = min(12, max(4, int(librosa.get_duration(y=y, sr=sr)/15)))
        bf = librosa.segment.agglomerative(chroma, k=n)
        bt = np.concatenate([[0], librosa.frames_to_time(bf, sr=sr, hop_length=hop),
                              [librosa.get_duration(y=y, sr=sr)]])
        ns = len(bt)-1
        sc = []
        for i in range(ns):
            f0_=int(librosa.time_to_frames(bt[i],   sr=sr, hop_length=hop))
            f1_=min(int(librosa.time_to_frames(bt[i+1],sr=sr,hop_length=hop)),chroma.shape[1])
            sc.append(chroma[:,f0_:f1_].mean(1) if f1_>f0_ else np.zeros(12))
        tmap = {}
        for i in range(ns):
            for j in range(i+2, ns):
                sim = float(np.dot(sc[i],sc[j])/(np.linalg.norm(sc[i])*np.linalg.norm(sc[j])+1e-8))
                if sim > 0.82 and j not in tmap:
                    tmap[j] = i
        sticks = [(snap(t2tick(bt[i],tps),BEAT), snap(t2tick(bt[i+1],tps),BEAT)) for i in range(ns)]
        print(f'  SSM: {ns} sections, {len(tmap)} reuse pairs')
        return tmap, sticks
    except Exception as e:
        print(f'  SSM skipped ({e})')
        return {}, []

def _replicate_phrases(by_sec, tmap, sticks):
    for sec, tmpl in tmap.items():
        if tmpl not in by_sec or sec not in by_sec: continue
        tn=by_sec[tmpl]; sn=by_sec[sec]
        if not tn or not sn: continue
        tf=[f for _,f,_ in tn]; st_=[t for t,_,_ in sn]; ss_=[s for _,_,s in sn]
        nf=[tf[0]]
        for i in range(1,len(st_)):
            d=(tf[i]-tf[i-1]) if i<len(tf) else 0
            nf.append(max(0,min(2,nf[-1]+d)))
        by_sec[sec]=[(st_[i],nf[i],ss_[i]) for i in range(len(st_))]
    return by_sec

# ── Sustains ──────────────────────────────────────────────────────────────────

def add_sustains(notes, target_pct, intensity=None):
    """
    Add sustains to notes. Intense sections get shorter sustains (staccato attack),
    quiet sections get longer holds (smooth, melodic feel).
    """
    if not notes: return notes
    if intensity is None: intensity = {}
    notes = sorted(notes, key=lambda x: x[0])
    tl    = [t for t,_,_ in notes]
    res   = []
    for i,(t,f,_) in enumerate(notes):
        gap = (tl[i+1]-t) if i<len(notes)-1 else BEAT*4
        sus = (gap-QTR  if gap>=BEAT*2 else
               gap-QTR  if gap>=BEAT   else
               gap-EGTH if gap>=HALF   else 0)
        # Intensity modulation: loud → shorten sustain (punchy), quiet → extend
        if sus > 0 and intensity:
            lvl = intensity.get(snap(t, BEAT), 0.5)
            # loud (>0.7): cut sustain by up to 40%; quiet (<0.3): extend by 20%
            if   lvl > 0.7: sus = int(sus * (1.0 - 0.4 * (lvl - 0.7) / 0.3))
            elif lvl < 0.3: sus = int(sus * (1.0 + 0.2 * (0.3 - lvl) / 0.3))
            sus = max(0, sus)
        res.append((t,f,sus))
    si  = [i for i,(_,_,s) in enumerate(res) if s>0]
    act = len(si)/max(len(res),1)
    if act > target_pct:
        si.sort(key=lambda i: res[i][2])
        drop = set(si[:int((act-target_pct)*len(res))])
        res  = [(t,f,0 if i in drop else s) for i,(t,f,s) in enumerate(res)]
    return res

# ── Chords ────────────────────────────────────────────────────────────────────

def add_chords(notes, beat_ticks, dbb, diff, chord_guide=None, fret_pool=None):
    if fret_pool is None:
        fret_pool = [0, 1, 2]
    targets = {'expert':0.38,'hard':0.22,'medium':0.12,'easy':0.05}
    target  = targets.get(diff, 0.15)
    bset    = set(beat_ticks)
    tset    = {(t,f,s) for t,f,s in notes}
    extra   = []

    n = len(fret_pool)
    for t,f,s in notes:
        if t not in bset: continue
        if dbb.get(t,1.0) < 0.9: continue
        if s == 0: continue
        # Prefer chord guide: only add chord if chromagram supports it
        if chord_guide is not None and t not in chord_guide: continue
        # Pick a chord fret 2 steps away in the pool (wraps within pool)
        idx = fret_pool.index(f) if f in fret_pool else 0
        cf = fret_pool[(idx + 2) % n]
        if (t,cf,s) not in tset:
            extra.append((t,cf,s))

    total_t = len(set(t for t,_,_ in notes))
    if total_t > 0:
        chord_t = len(set(t for t,_,_ in extra))
        if chord_t/total_t > target:
            extra = extra[:int(target*total_t)]

    return sorted(list(notes)+extra, key=lambda x:(x[0],x[1]))

# ── Note importance ranking ──────────────────────────────────────────────────

def rank_onset_importance(onset_ticks, beat_times, tps, y, y_guit, sr,
                           chord_guide, fret_at,
                           kick_ticks=None, snare_ticks=None):
    """
    Score every detected onset 0–1 by musical importance.
    Human charters keep the most important notes; this replaces blind subsampling.

    Scoring factors:
      beat_strength   (0.25) — downbeat > beat > offbeat
      drum_anchor     (0.20) — kick/snare hit coincides with this onset
      onset_strength  (0.20) — how strong the attack is
      energy          (0.15) — RMS at that moment
      harmonic_weight (0.12) — is this onset a chord root / strong chroma peak?
      melodic_peak    (0.08) — local maximum in the pitch contour?
    """
    if not onset_ticks:
        return {}

    kick_ticks  = kick_ticks  or set()
    snare_ticks = snare_ticks or set()

    hop = 512
    # Pre-compute per-frame features
    oe      = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms     = librosa.feature.rms(y=y_guit, hop_length=hop)[0]
    chroma  = librosa.feature.chroma_cqt(y=y_guit, sr=sr, hop_length=hop)

    # Pitch contour for melodic peak detection
    f0, voiced, _ = librosa.pyin(y_guit, sr=sr,
                                  fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C7'),
                                  frame_length=2048, hop_length=hop, fill_na=0.0)
    midi = np.where(voiced & (f0 > 0),
                    12*np.log2(np.maximum(f0, 1e-6)/440)+69, 0.0)

    # Build beat strength map: tick → strength value
    beat_tick_set = set(snap(t2tick(bt, tps), BEAT) for bt in beat_times)
    half_tick_set = set()
    for i in range(len(beat_times)-1):
        g = beat_times[i+1] - beat_times[i]
        half_tick_set.add(snap(t2tick(beat_times[i] + g*0.5, tps), HALF))

    scores = {}
    for tick in onset_ticks:
        t_s   = tick / tps
        frame = min(int(librosa.time_to_frames(t_s, sr=sr, hop_length=hop)),
                    len(oe)-1)

        # 1. Beat strength
        if   tick in beat_tick_set:  bs = 1.0
        elif tick in half_tick_set:  bs = 0.6
        else:                        bs = 0.2

        # 2. Drum anchor — kick scores highest (strong downbeat hit), snare good too
        if   tick in kick_ticks:   da = 1.0
        elif tick in snare_ticks:  da = 0.7
        else:                      da = 0.0

        # 3. Onset strength (normalised)
        os_val = float(oe[frame]) / (float(oe.max()) + 1e-8)

        # 4. Energy
        en_val = float(rms[min(frame, len(rms)-1)]) / (float(rms.max()) + 1e-8)

        # 5. Harmonic weight — how "peaked" is the chroma at this frame?
        col    = chroma[:, min(frame, chroma.shape[1]-1)]
        peak   = float(col.max())
        mean_c = float(col.mean())
        hw     = (peak - mean_c) / (peak + 1e-8)

        # 6. Melodic peak — local max in pitch contour?
        lo = max(0, frame-3); hi = min(len(midi), frame+4)
        window = midi[lo:hi]
        mp = 1.0 if (len(window) > 0 and midi[frame] == window.max()
                     and midi[frame] > 0) else 0.0

        score = (0.25*bs + 0.20*da + 0.20*os_val + 0.15*en_val + 0.12*hw + 0.08*mp)
        scores[tick] = float(score)

    return scores


def select_by_importance(pool, scores, target, min_gap):
    """
    Pick `target` notes from pool, sorted by importance score,
    then re-sort chronologically. Enforces min_gap after selection.
    """
    if not pool:
        return []
    # Sort by score descending, then enforce min_gap
    ranked = sorted(pool, key=lambda t: scores.get(t, 0.0), reverse=True)
    chosen, prev = [], -999999
    for t in ranked:
        if t - prev >= min_gap or prev == -999999:
            chosen.append(t); prev = t   # note: prev not enforced in sorted order
        if len(chosen) >= target:
            break
    # Re-sort chronologically and re-enforce gap
    return filter_gap(sorted(chosen), min_gap)


# ── Phrase boundary & motif system ───────────────────────────────────────────

def build_phrase_map(y, sr, beat_times, tps, bars_per_phrase=4):
    """
    Split the song into musical phrases (default 4 bars each).
    Build a chroma fingerprint for every phrase.
    Return:
      phrases       — list of (start_tick, end_tick, fingerprint)
      phrase_groups — dict: phrase_idx → template_phrase_idx (for identical phrases)

    Two phrases are "identical" if chroma cosine similarity > 0.88.
    When charting, identical phrases get the *exact same* fret pattern.
    """
    if len(beat_times) < 2:
        return [], {}

    hop         = 512
    beats_per_phrase = bars_per_phrase * 4   # assuming 4/4
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma      = sk_normalize(chroma, axis=0)

    phrases = []
    i = 0
    while i + beats_per_phrase <= len(beat_times):
        t_start = beat_times[i]
        t_end   = beat_times[min(i + beats_per_phrase, len(beat_times)-1)]
        f_start = int(librosa.time_to_frames(t_start, sr=sr, hop_length=hop))
        f_end   = min(int(librosa.time_to_frames(t_end, sr=sr, hop_length=hop)),
                      chroma.shape[1])
        fp = chroma[:, f_start:f_end].mean(axis=1) if f_end > f_start else np.zeros(12)
        phrases.append((
            snap(t2tick(t_start, tps), BEAT),
            snap(t2tick(t_end,   tps), BEAT),
            fp
        ))
        i += beats_per_phrase

    # Find identical phrase pairs (similarity > 0.88)
    phrase_groups = {}   # phrase_j → phrase_i  (j copies pattern of i)
    for j in range(1, len(phrases)):
        for i in range(j):
            if i in phrase_groups.values():
                continue   # don't chain — always copy from original
            sim = float(np.dot(phrases[i][2], phrases[j][2]) /
                        (np.linalg.norm(phrases[i][2]) *
                         np.linalg.norm(phrases[j][2]) + 1e-8))
            if sim > 0.88 and j not in phrase_groups:
                phrase_groups[j] = i

    n_matches = len(phrase_groups)
    print(f'  Phrase map: {len(phrases)} phrases, {n_matches} repeating')
    return phrases, phrase_groups


def apply_phrase_motifs(notes, phrases, phrase_groups):
    """
    For every phrase marked as a repeat in phrase_groups,
    replace its notes with the exact fret pattern from the template phrase.
    Tick positions are preserved; only fret assignments are copied.
    """
    if not phrases or not phrase_groups:
        return notes

    # Index notes by phrase
    by_phrase = {i: [] for i in range(len(phrases))}
    for t, f, s in notes:
        for i, (ps, pe, _) in enumerate(phrases):
            if ps <= t < pe:
                by_phrase[i].append((t, f, s))
                break

    # Copy fret pattern from template → repeat phrase
    for j, i in phrase_groups.items():
        tmpl = by_phrase[i]
        copy = by_phrase[j]
        if not tmpl or not copy:
            continue
        t_frets = [f for _, f, _ in tmpl]
        new_notes = []
        for k, (t, _, s) in enumerate(copy):
            # Wrap around template if copy is longer
            f = t_frets[k % len(t_frets)]
            new_notes.append((t, f, s))
        by_phrase[j] = new_notes

    # Reassemble chronologically
    result = []
    for i in range(len(phrases)):
        result.extend(by_phrase[i])
    # Any notes outside phrase boundaries
    phrase_start = phrases[0][0]  if phrases else 0
    phrase_end   = phrases[-1][1] if phrases else 999999999
    for t, f, s in notes:
        if t < phrase_start or t >= phrase_end:
            result.append((t, f, s))
    return sorted(result, key=lambda x: x[0])


# ── Build one difficulty ──────────────────────────────────────────────────────

def build_diff(diff, beats, eighths, sixteenths, onset_ticks,
               fret_at, duration, dbb, genre, bpm, tps,
               solo_regions, y_guit, y, sr,
               tmap=None, sticks=None, chord_guide=None,
               importance_scores=None, phrases=None, phrase_groups=None,
               fret_pool=None, intensity=None, kick_ticks=None,
               snare_ticks=None, bass_at=None):

    if fret_pool is None:
        fret_pool = [0, 1, 2]
    if intensity is None:
        intensity = {}
    if kick_ticks is None:
        kick_ticks = set()
    if snare_ticks is None:
        snare_ticks = set()

    # Easy difficulty restricts to lower half of the fret pool
    easy_pool = fret_pool[:max(1, (len(fret_pool)+1)//2)]
    active_pool = easy_pool if diff == 'easy' else fret_pool

    scale    = duration / 180.0
    note_tgt = {'expert':int(1200*scale),'hard':int(850*scale),
                'medium':int(650*scale), 'easy':int(380*scale)}[diff]
    sus_tgt  = {'expert':0.10,'hard':0.15,'medium':0.22,'easy':0.35}[diff]
    min_gaps = {'expert':QTR, 'hard':HALF, 'medium':BEAT, 'easy':BEAT*2}
    min_gap  = min_gaps[diff]

    # Tick ranges occupied by solos
    solo_tick_set = set()
    for ts, te in (solo_regions or []):
        for tk in range(ts, te, QTR):
            solo_tick_set.add(tk)

    oset  = set(onset_ticks)
    onear = set()
    for o in onset_ticks:
        for d in range(-HALF, HALF+1, QTR): onear.add(o+d)

    # ── Candidate pool per difficulty ────────────────────────────────────
    if diff == 'expert':
        pool = [t for t in eighths if t in onear and t not in solo_tick_set]
        bursts = [o for o in onset_ticks
                  if ((o-QTR*2) in oset or (o+QTR*2) in oset)
                  and o not in solo_tick_set]
        pool = filter_gap(sorted(set(pool) | set(bursts)), QTR)
    elif diff == 'hard':
        pool = filter_gap(sorted(
            t for t in eighths if t in onear and t not in solo_tick_set), HALF)
    elif diff == 'medium':
        pool = filter_gap(sorted(
            t for t in set(eighths)|set(beats) if t not in solo_tick_set), BEAT)
    else:
        pool = filter_gap(sorted(
            t for t in beats if t not in solo_tick_set), BEAT*2)

    # Section density gate (Expert/Hard only)
    if diff in ('expert','hard') and dbb:
        pool = [t for t in pool
                if not (dbb.get(snap(t,BEAT),1.0) < 0.75
                        and np.random.random() > dbb.get(snap(t,BEAT),1.0))]

    # ── Intensity-aware note target: dense sections get more notes ────────
    # Boost/trim note_tgt per section based on local intensity
    if intensity:
        avg_intensity = float(np.mean(list(intensity.values()))) if intensity else 0.5
        # Scale pool size: quiet sections → 70% of target, loud → 130%
        intensity_scale = 0.70 + 0.60 * avg_intensity
        note_tgt = int(note_tgt * intensity_scale)

    # ── Kick-anchored boosting: kick hits must stay in pool ───────────────
    # Even if importance ranking would cut them, keep kick-coincident ticks
    kick_in_pool = set(t for t in pool if t in kick_ticks)

    # ── Importance-ranked note selection (replaces blind subsampling) ─────
    if importance_scores and len(pool) > note_tgt:
        pool = select_by_importance(pool, importance_scores, note_tgt, min_gap)
        # Re-insert any lost kick hits (up to 5% of target)
        for kt in sorted(kick_in_pool):
            if kt not in pool and len(pool) < int(note_tgt * 1.05):
                pool = sorted(pool + [kt])
    elif len(pool) > note_tgt:
        step = len(pool) / note_tgt
        pool = [pool[int(i*step)] for i in range(note_tgt)]

    # ── Genre fret assignment ─────────────────────────────────────────────
    notes = assign_frets_genre(pool, fret_at, genre, dbb, tps, bpm, active_pool,
                               intensity=intensity, bass_at=bass_at,
                               kick_ticks=kick_ticks, snare_ticks=snare_ticks)

    # ── Phrase motif repetition (identical phrases → identical patterns) ──
    if phrases and phrase_groups:
        notes = apply_phrase_motifs(notes, phrases, phrase_groups)

    # ── SSM macro-structure replication (fallback / complementary) ────────
    if tmap and sticks:
        by_sec = {si:[n for n in notes if s0<=n[0]<s1]
                  for si,(s0,s1) in enumerate(sticks)}
        by_sec = _replicate_phrases(by_sec, tmap, sticks)
        notes  = sorted([n for s in by_sec.values() for n in s], key=lambda x:x[0])

    notes = add_sustains(notes, sus_tgt, intensity=intensity)

    # ── Solo regions: note-for-note chart, merged in ───────────────────────
    solo_notes = []
    for ts, te in (solo_regions or []):
        solo_notes.extend(chart_solo_region(y_guit, sr, tps, ts, te, diff, bpm, active_pool))

    if solo_notes:
        notes = [(t,f,s) for t,f,s in notes if t not in solo_tick_set]
        notes = sorted(notes + solo_notes, key=lambda x:x[0])

    return notes

# ── Chart file output ─────────────────────────────────────────────────────────

def section_block(name, notes, sp_events=None):
    lines = [f'[{name}]', '{']
    all_  = [(t, f'N {f} {s}') for t,f,s in notes]
    if sp_events: all_ += [(t,e) for t,e in sp_events]
    for t,e in sorted(all_, key=lambda x:x[0]):
        lines.append(f'  {t} = {e}')
    lines.append('}')
    return '\n'.join(lines)

# ── Lyrics ────────────────────────────────────────────────────────────────────

def get_lyrics(mp3, tps, section_events):
    print('  Whisper large-v3 transcription...')
    model  = whisper.load_model('large-v3')
    result = model.transcribe(mp3, word_timestamps=True, verbose=False)
    evs    = list(section_events)
    for seg in result['segments']:
        txt = seg['text'].strip()
        if not txt: continue
        words=txt.split(); n=len(words)
        st,en=seg['start'],seg['end']
        evs.append((t2tick(st,tps),'phrase_start'))
        for i,w in enumerate(words):
            evs.append((t2tick(st+(en-st)*i/n,tps),
                        f'lyric {w.replace(chr(34),"").replace("=","")}'))
        evs.append((t2tick(en,tps),'phrase_end'))
    evs.sort(key=lambda x:x[0])
    return '[Events]\n{\n'+''.join(f'  {t} = E "{e}"\n' for t,e in evs)+'}'

# ── Main ──────────────────────────────────────────────────────────────────────

def generate(mp3_path, fret_pool=None):
    print(f'\n{"="*60}\n  {os.path.basename(mp3_path)}\n{"="*60}')
    mp3_path = os.path.abspath(mp3_path)

    try:
        tags   = ID3(mp3_path)
        title  = str(tags.get('TIT2', os.path.splitext(os.path.basename(mp3_path))[0]))
        artist = str(tags.get('TPE1', 'Unknown'))
        album  = str(tags.get('TALB', ''))
    except Exception:
        title  = os.path.splitext(os.path.basename(mp3_path))[0]
        artist = 'Unknown'; album = ''

    safe   = re.sub(r'[<>:"/\\|?*]', '', f'{artist} - {title}')
    folder = os.path.join(OUT_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    print(f'  {artist} - {title}')

    try:
        for k in ID3(mp3_path):
            if k.startswith('APIC'):
                img = Image.open(io.BytesIO(ID3(mp3_path)[k].data)).convert('RGB')
                img.save(os.path.join(folder,'album.jpg'),'JPEG')
                print(f'  Art: {img.size}'); break
    except Exception: pass

    # ── Analysis pipeline ─────────────────────────────────────────────────
    y, y_guit, y_perc, y_bass, sr, duration = load_and_separate(mp3_path)
    genre                            = detect_genre(y, sr)
    bpm, beat_times, bpm_events, _  = analyze_tempo(y_perc, sr, mp3_path)
    tps                              = (bpm/60.0)*RES
    if fret_pool is None:
        fret_pool = [0, 1, 2]
    fret_names = '/'.join(FRET_NAMES.get(f, str(f)) for f in fret_pool)
    print(f'  BPM={bpm:.1f}  Duration={duration:.1f}s  Genre={genre}  Frets={fret_names}')

    beats, eighths, sixteenths = build_grids(beat_times, tps)
    onset_ticks                = get_onsets(y, y_perc, sr, tps, mp3_path)
    fret_at                    = build_pitch_map(y_guit, sr, tps)
    solo_regions               = detect_solo_regions(y_guit, y, sr, beat_times, tps, bpm)
    sec_events, dbb            = detect_sections(y, sr, beat_times, tps, solo_regions)
    sp_events                  = make_star_power(beats, tps, duration, solo_regions)
    tmap, sticks               = detect_repeated_phrases(y, sr, tps)
    chord_guide                = build_chord_guide(y, y_guit, sr, tps, beats)

    # v7 additions — intensity, drums, bass
    print('  Building intensity curve...')
    intensity                  = build_intensity_curve(y, sr, beat_times, tps)
    print('  Separating kick / snare...')
    kick_ticks, snare_ticks    = build_drum_maps(y_perc, sr, tps)
    print('  Bass pitch tracking...')
    bass_at                    = build_bass_root_map(y_bass, sr, tps)

    print('  Ranking note importance...')
    importance_scores          = rank_onset_importance(
                                     onset_ticks, beat_times, tps,
                                     y, y_guit, sr, chord_guide, fret_at,
                                     kick_ticks=kick_ticks, snare_ticks=snare_ticks)
    print('  Building phrase map...')
    phrases, phrase_groups     = build_phrase_map(y, sr, beat_times, tps)

    avg_int = float(np.mean(list(intensity.values()))) if intensity else 0.0
    print(f'  Sections={len(sec_events)}  SP={len(sp_events)}  '
          f'Onsets={len(onset_ticks)}  Solos={len(solo_regions)}  '
          f'Phrases={len(phrases)} ({len(phrase_groups)} repeating)  '
          f'Kicks={len(kick_ticks)}  Snares={len(snare_ticks)}  '
          f'AvgIntensity={avg_int:.2f}')

    # ── Chart all difficulties ─────────────────────────────────────────────
    np.random.seed(42)
    diffs = {}
    for diff in ('expert','hard','medium','easy'):
        notes = build_diff(
            diff, beats, eighths, sixteenths, onset_ticks,
            fret_at, duration, dbb, genre, bpm, tps,
            solo_regions, y_guit, y, sr,
            tmap=tmap, sticks=sticks, chord_guide=chord_guide,
            importance_scores=importance_scores,
            phrases=phrases, phrase_groups=phrase_groups,
            fret_pool=fret_pool, intensity=intensity,
            kick_ticks=kick_ticks, snare_ticks=snare_ticks,
            bass_at=bass_at)
        notes = add_chords(notes, beats, dbb, diff, chord_guide, fret_pool=fret_pool)
        diffs[diff] = notes
        ps = round(100*sum(1 for _,_,s in notes if s>0)/max(len(notes),1))
        solo_n = sum(1 for t,_,_ in notes
                     if any(ts<=t<te for ts,te in solo_regions))
        print(f'  {diff:7s}: {len(notes):4d} notes  {ps}% sustain  '
              f'solo_notes={solo_n}  frets={sorted(set(f for _,f,_ in notes))}')

    sync         = build_sync_track(bpm_events, tps, bpm)
    events_block = get_lyrics(mp3_path, tps, sec_events)

    chart = '\n'.join([
        f'[Song]\n{{\n  Name = "{title}"\n  Artist = "{artist}"\n'
        f'  Charter = "Auto-Generated"\n  Offset = 0\n  Resolution = {RES}\n'
        f'  Player2 = bass\n  Difficulty = 0\n  PreviewStart = 0\n'
        f'  PreviewEnd = 0\n  Genre = "rock"\n  MediaType = "cd"\n'
        f'  MusicStream = "song.opus"\n}}',
        sync, events_block,
        section_block('ExpertSingle', diffs['expert'], sp_events),
        section_block('HardSingle',   diffs['hard'],   sp_events),
        section_block('MediumSingle', diffs['medium'], sp_events),
        section_block('EasySingle',   diffs['easy'],   sp_events),
    ])

    open(os.path.join(folder,'notes.chart'),'w',encoding='utf-8').write(chart)

    subprocess.run(['ffmpeg','-y','-i',mp3_path,'-c:a','libopus','-b:a','192k',
                    os.path.join(folder,'song.opus')], capture_output=True)

    bg = Image.new('RGB',(1920,1080),(10,10,10))
    d  = ImageDraw.Draw(bg)
    d.text((960,480), title,  fill=(255,255,255), anchor='mm')
    d.text((960,560), artist, fill=(180,180,180), anchor='mm')
    d.text((960,620), f'{genre.upper()}  •  {bpm:.0f} BPM  •  {len(phrases)} phrases', fill=(80,120,200), anchor='mm')
    if solo_regions:
        d.text((960,670), f'{len(solo_regions)} guitar solo(s) detected  •  {len(phrase_groups)} motif repeats',
               fill=(255,180,0), anchor='mm')
    bg.save(os.path.join(folder,'background.jpg'),'JPEG')

    open(os.path.join(folder,'song.ini'),'w',encoding='utf-8').write(
        f'[song]\nname = {title}\nartist = {artist}\nalbum = {album}\n'
        f'charter = Auto-Generated\nyear =\ngenre = {genre}\n'
        f'song_length = {int(duration*1000)}\ndiff_guitar = -1\n'
        f'preview_start_time = 0\nicon =\nloading_phrase =\n')

    print(f'  Done -> {folder}')
    return folder


def _mel_for_chartnet(y, sr, beat_times, n_mels=128, hop_length=512):
    """Build beat-aligned mel tensor for ChartNet inference (same as preprocess)."""
    import librosa as _lib
    S    = _lib.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                        hop_length=hop_length, n_fft=2048,
                                        fmin=20, fmax=8000)
    S_db = _lib.power_to_db(S, ref=np.max)
    grid_times = []
    for i in range(len(beat_times)-1):
        gap = beat_times[i+1] - beat_times[i]
        for k in range(4):
            grid_times.append(beat_times[i] + gap*k/4)
    if len(beat_times):
        grid_times.append(beat_times[-1])
    grid_times = np.array(sorted(set(grid_times)))
    frames = []
    for t in grid_times:
        f = min(int(_lib.time_to_frames(t, sr=sr, hop_length=hop_length)),
                S_db.shape[1]-1)
        frames.append(S_db[:, max(0, f)])
    mel = np.stack(frames, axis=0)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return torch.tensor(mel, dtype=torch.float32), grid_times


def chartnet_notes(y, sr, beat_times, tps, diff, solo_tick_set, fret_pool=None):
    """
    Use trained ChartNet v2 to predict full note events.
    Returns list of (tick, fret, sustain) with chord notes interleaved,
    or None if model not loaded / inference fails.

    ChartNet v2 predicts per grid position:
      fret     (0-4)  → remapped through fret_pool
      sustain  bucket (0-3) → converted to ticks
      chord    (0-4 or -1=none) → adds a second simultaneous note
      type     (0=normal, 1=HOPO, 2=tap, 3=open) → written as fret 5/6/7
    """
    if fret_pool is None:
        fret_pool = [0, 1, 2]
    if _chartnet_model is None:
        return None

    # Sustain bucket → approximate tick length
    SUSTAIN_TICKS = {0: 0, 1: QTR//2, 2: QTR, 3: BEAT*2}

    try:
        mel, grid_times = _mel_for_chartnet(y, sr, beat_times)
        raw = predict_chart(_chartnet_model, mel, diff, device=_chartnet_device)
        notes = []
        max_idx = len(fret_pool) - 1

        for ev in raw:
            gp = ev['grid_pos']
            if gp >= len(grid_times):
                continue
            tick = snap(t2tick(grid_times[gp], tps), QTR)
            if tick in solo_tick_set:
                continue

            # Remap fret through active pool
            mapped_fret = fret_pool[min(ev['fret'], max_idx)]
            sustain     = SUSTAIN_TICKS.get(ev['sustain'], 0)
            ntype       = ev['type']

            # Note type → Clone Hero special fret numbers
            if ntype == 2:     # tap
                notes.append((tick, 6, 0))
            elif ntype == 3:   # open strum
                notes.append((tick, 7, sustain))
            else:
                if ntype == 1:     # HOPO flag
                    notes.append((tick, 5, 0))
                notes.append((tick, mapped_fret, sustain))

                # Chord: add second note if model predicted one
                chord_idx = ev.get('chord', -1)
                if chord_idx >= 0:
                    chord_fret = fret_pool[min(chord_idx, max_idx)]
                    if chord_fret != mapped_fret:
                        notes.append((tick, chord_fret, sustain))

        return sorted(notes, key=lambda x: (x[0], x[1])) if notes else None
    except Exception as e:
        print(f'  [warn] ChartNet inference failed ({e}), using algorithmic fallback')
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Clone Hero chart generator v6',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fret numbers:  0=Green  1=Red  2=Yellow  3=Blue  4=Orange

Examples:
  python chartgen.py song.mp3
  python chartgen.py --frets 0,1,2,3,4 song.mp3        # all 5 frets
  python chartgen.py --frets 0,1,2     song.mp3        # GRY only (default)
  python chartgen.py --frets 2,3,4     song.mp3        # YBO only
  python chartgen.py --model best.pt --frets 0,1,2,3,4 song.mp3
""")
    ap.add_argument('--model', default=None,
                    help='Path to trained ChartNet checkpoint (optional)')
    ap.add_argument('--frets', default='0,1,2',
                    help='Comma-separated fret numbers to use, e.g. 0,1,2 or 0,1,2,3,4 '
                         '(0=Green 1=Red 2=Yellow 3=Blue 4=Orange, default: 0,1,2)')
    ap.add_argument('songs', nargs='*')
    known, _ = ap.parse_known_args()

    # Parse fret pool
    try:
        fret_pool = [int(f.strip()) for f in known.frets.split(',')]
        fret_pool = sorted(set(max(0, min(4, f)) for f in fret_pool))
        if not fret_pool:
            raise ValueError
    except Exception:
        print('ERROR: --frets must be comma-separated numbers 0-4, e.g. --frets 0,1,2')
        sys.exit(1)
    print(f'Fret palette: {", ".join(FRET_NAMES[f] for f in fret_pool)} '
          f'({",".join(str(f) for f in fret_pool)})')

    # Also check environment variable
    model_path = known.model or os.environ.get('CHARTNET_MODEL')
    if model_path and os.path.exists(model_path):
        load_chartnet(model_path)

    songs = known.songs or [a for a in sys.argv[1:] if not a.startswith('--')]
    if not songs:
        print('Usage: python chartgen.py [--model checkpoint.pt] [--frets 0,1,2] <song.mp3> ...')
        sys.exit(1)

    for mp3 in songs:
        generate(mp3, fret_pool=fret_pool)
