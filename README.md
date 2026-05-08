# Clone Hero Chart Generator

Automatically generates Clone Hero `.chart` files from MP3s — with guitar solo detection, genre-aware patterns, intensity-driven dynamics, instrument-aware note placement, and an optional trainable neural model.

## Features

- 🎸 **Guitar solo detection** — 5-signal detector (pyin voiced probability, spectral centroid, onset density, RMS, chroma peakiness). Solo regions are charted note-for-note at high resolution.
- 🎵 **Genre detection** — metal / pop / electronic / rock each get a different pattern generation strategy
- 🧠 **Phrase motif system** — repeated musical sections (chorus 1 ≈ chorus 2) get the **exact same** fret pattern
- 📊 **Note importance ranking** — downbeats, chord roots, melody peaks, kick/snare hits all scored; filler notes dropped
- 🔁 **SSM phrase replication** — Self-Similarity Matrix finds structurally similar sections
- 🎤 **Lyrics** — Whisper large-v3 transcription with word-level timestamps
- 🎯 **4 difficulties** — Easy / Medium / Hard / Expert, all calibrated from real chart analysis
- 🎛️ **Configurable fret palette** — choose any subset of the 5 frets via `--frets`
- 🥁 **Kick & snare awareness** — kick hits anchor fret position (grounded feel), snare hits accent upward
- 🎚️ **Intensity curve** — loud sections get denser notes and wider fret jumps; quiet sections get sparse, sustained notes
- 🎼 **Bass root anchoring** — bass pitch changes trigger a return to the lowest fret (chord root feel)
- 🤖 **Neural model (ChartNet v3, ~60M params)** — CNN + Transformer architecture predicts note placement, fret, sustain length, chords, and note type (HOPO/tap/open strum) all in one shot

## Requirements

```bash
pip install -r requirements.txt
pip install madmom   # optional but recommended for beat tracking
```

`ffmpeg` must be in PATH (for opus conversion). The generator checks at startup and prints install instructions if it is missing.

## Usage

### Generate a chart
```bash
python chartgen.py "MySong.mp3"
python chartgen.py --out "C:/Clone Hero/Songs" "MySong.mp3"
python chartgen.py --no-lyrics "MySong.mp3"   # skip Whisper, much faster
python chartgen.py "MySong.mp3" "Song2.mp3" "Song3.mp3"   # batch
```

Output goes to `<out>/Artist - Title/` (default: current directory):
```
Artist - Title/
  song.opus
  notes.chart
  song.ini
  album.jpg
  background.jpg
```

When processing multiple songs, a progress bar shows `1/N … 2/N …`. If one song fails (corrupt file, missing tags, etc.) the rest continue and a failure summary is printed at the end.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--out` | `.` | Output directory for chart folders |
| `--frets` | `0,1,2` | Fret palette — comma-separated numbers 0–4 |
| `--no-lyrics` | off | Skip Whisper transcription (much faster) |
| `--model` | none | Path to trained ChartNet checkpoint |

### Choosing which frets to use

Use `--frets` to pick any combination of the 5 Clone Hero buttons:

| Number | Color  |
|--------|--------|
| 0      | Green  |
| 1      | Red    |
| 2      | Yellow |
| 3      | Blue   |
| 4      | Orange |

```bash
# Default — Green / Red / Yellow
python chartgen.py "MySong.mp3"

# All 5 frets — Green / Red / Yellow / Blue / Orange
python chartgen.py --frets 0,1,2,3,4 "MySong.mp3"

# Top 3 only — Yellow / Blue / Orange
python chartgen.py --frets 2,3,4 "MySong.mp3"

# Just two frets — Green / Orange
python chartgen.py --frets 0,4 "MySong.mp3"
```

The pitch contour is rescaled to fill whatever range you pick. Easy difficulty automatically restricts to the lower half of your chosen palette. Works with the neural model too.

### With trained neural model
When `--model` is provided, ChartNet is used as the primary note predictor for all difficulties. The algorithmic pipeline is the fallback if the model isn't loaded or inference fails.

```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
python chartgen.py --model ./checkpoints/best.pt --frets 0,1,2,3,4 "MySong.mp3"
python chartgen.py --model ./checkpoints/best.pt --out "C:/Clone Hero/Songs" "MySong.mp3"
```

## Training your own model

### Step 1 — Scrape charts (2–5 hours for 5k+ songs)
```bash
python scraper.py --out ./dataset --limit 2000
python scraper.py --out ./dataset --limit 2000 --query "metal"   # genre-filtered
```
Downloads chart+audio pairs from [Enchor](https://enchor.us) (Chorus Encore). Each song is validated — must have an ExpertSingle track with 50+ notes and a valid audio file.

The song list is cached to `dataset/songs_list.json` after the first run. Re-running skips already-downloaded songs and resumes from where it left off. Use `--refresh` to re-fetch the song list from the API.

### Step 2 — Preprocess (~1–2 hours for 5k songs)
```bash
python preprocess_dataset.py --data ./dataset --out ./processed
```

Re-running skips songs whose `.pt` file already exists, so you can safely add new songs and re-run without reprocessing everything.

### Step 3 — Train
```bash
python train_chartnet.py --data ./processed --out ./checkpoints
```

Defaults: **160 epochs**, batch 48, mixed precision (AMP fp16) enabled automatically on CUDA. 5 minute GPU cooldown between epochs — safe to leave running overnight or for multiple days. Early stopping (patience=8) will stop training automatically if F1 stops improving, typically around epoch 30–50.

| GPU | ~5k songs / epoch | Full 160 epochs (max) |
|---|---|---|
| RTX 4070 Super | ~60 min | ~7 days (early stop ~2–3 days) |
| RTX 3090 | ~40 min | ~5 days (early stop ~1.5–2 days) |
| RTX 3060 | ~90 min | ~10 days (early stop ~3–4 days) |

> The GPU will not be damaged by sustained load — modern GPUs have automatic thermal throttling. The cooldown is just a precaution.

**Checkpoints:** `best.pt` is saved whenever validation note F1 improves. `epoch_NNN.pt` is saved after every epoch for crash recovery (only the 3 most recent are kept to save disk space). Resume with `--resume ./checkpoints/epoch_020.pt`.

**Early stopping:** Training stops automatically if val F1 does not improve for 8 consecutive epochs. Override with `--patience N` (set to `0` to disable).

**Training flags:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `160` | Maximum training epochs |
| `--batch` | `48` | Batch size (reduce to `24` if out of VRAM) |
| `--patience` | `8` | Early stopping patience (0 = disabled) |
| `--resume` | none | Resume from a periodic checkpoint |
| `--lr` | `2e-4` | Learning rate |
| `--note_pos_weight` | `8.0` | BCE positive weight for rare note class |

### Step 4 — Use it
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
```

## Files

| File | Purpose |
|---|---|
| `chartgen.py` | Main chart generator (v6) |
| `chartmodel.py` | ChartNet v3 model definition (~60M params, CNN+Transformer) |
| `scraper.py` | Downloads chart+audio pairs from Enchor API |
| `preprocess_dataset.py` | Converts charts to beat-aligned mel tensors |
| `train_chartnet.py` | Training loop with F1 tracking and early stopping |

## Notes

- Default fret palette is **Green (0), Red (1), Yellow (2)** — use `--frets` to change
- Audio output is always `song.opus` (192k)
- Album art saved as `album.jpg`
- `ffmpeg` must be in PATH — checked at startup with clear install instructions if missing
