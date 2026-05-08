# Clone Hero Chart Generator

Automatically generates Clone Hero `.chart` files from MP3s — with guitar solo detection, genre-aware patterns, intensity-driven dynamics, and a trainable neural model (ChartNet v3).

## GUI App

The easiest way to use this project is the **ChartGen.exe** desktop app — no command line needed.

**Download:** `release/ChartGen/ChartGen.exe`

**Or build it yourself:**
```bash
# Just double-click build.bat — installs everything and builds automatically
build.bat
```

> Built with **Nuitka** (Python → native machine code) — 0 antivirus detections on VirusTotal.

### App Features

| Tab | What it does |
|---|---|
| 🎸 **Generate** | Drag & drop MP3s, pick neural model, generate charts |
| 📥 **Scrape** | Download chart+audio pairs from Enchor by genre |
| ⚙️ **Preprocess** | Convert scraped songs to training tensors |
| 🧠 **Train** | Train ChartNet with all hyperparameters, resume support, stop button |

- **6 color themes** — Green, Blue Steel, Purple Haze, Fire Red, Gold Rush, Ice White
- **Custom background image** — any JPG/PNG, auto-blurred and darkened
- **Settings remembered** — output folder, model path, frets, theme all saved between sessions

---

## Features

- 🎸 **Guitar solo detection** — 5-signal detector (pyin voiced probability, spectral centroid, onset density, RMS, chroma peakiness). Solo regions are charted note-for-note at high resolution.
- 🎵 **Genre detection** — metal / pop / electronic / rock each get a different pattern generation strategy
- 🧠 **Phrase motif system** — repeated musical sections (chorus 1 ≈ chorus 2) get the **exact same** fret pattern
- 📊 **Note importance ranking** — downbeats, chord roots, melody peaks, kick/snare hits all scored; filler notes dropped
- 🔁 **SSM phrase replication** — Self-Similarity Matrix finds structurally similar sections
- 🎤 **Lyrics** — Whisper large-v3 transcription with word-level timestamps
- 🎯 **4 difficulties** — Easy / Medium / Hard / Expert, all calibrated from real chart analysis
- 🎛️ **Configurable fret palette** — choose any subset of the 5 frets via `--frets`
- 🥁 **Kick & snare awareness** — kick hits anchor fret position, snare hits accent upward
- 🎚️ **Intensity curve** — loud sections get denser notes; quiet sections get sparse, sustained notes
- 🎼 **Bass root anchoring** — bass pitch changes trigger a return to the lowest fret
- 🤖 **Neural model (ChartNet v3, ~50M params)** — CNN + Transformer predicts note placement, fret, sustain, chords, and note type (HOPO/tap/open strum) all in one shot

---

## Requirements

```bash
pip install -r requirements.txt
pip install madmom   # optional but recommended for beat tracking
```

`ffmpeg` must be in PATH (for opus conversion). The generator checks at startup and prints install instructions if missing.

---

## Command Line Usage

### Generate a chart
```bash
cd src/
python chartgen.py "MySong.mp3"
python chartgen.py --out "C:/Clone Hero/Songs" "MySong.mp3"
python chartgen.py --no-lyrics "MySong.mp3"        # skip Whisper, much faster
python chartgen.py "Song1.mp3" "Song2.mp3"         # batch
python chartgen.py --model ../checkpoints/best.pt "MySong.mp3"  # with neural model
```

Output goes to `<out>/Artist - Title/`:
```
Artist - Title/
  song.opus
  notes.chart
  song.ini
  album.jpg
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--out` | `.` | Output directory |
| `--frets` | `0,1,2` | Fret palette — comma-separated 0–4 |
| `--no-lyrics` | off | Skip Whisper transcription |
| `--model` | none | Path to ChartNet checkpoint |

### Fret palette

| Number | Color  |
|--------|--------|
| 0      | Green  |
| 1      | Red    |
| 2      | Yellow |
| 3      | Blue   |
| 4      | Orange |

```bash
python chartgen.py --frets 0,1,2,3,4 "MySong.mp3"   # all 5 frets
python chartgen.py --frets 2,3,4 "MySong.mp3"         # Yellow / Blue / Orange only
```

---

## Training Your Own Model

### Step 1 — Scrape charts
```bash
python scraper.py --out ./dataset --limit 2000 --query "metal" --workers 16
python scraper.py --out ./dataset --limit 2000 --query "rock"  --workers 16
```
Downloads chart+audio pairs from [Enchor](https://enchor.us). Each song is validated — must have ExpertSingle with 50+ notes. Skips already-downloaded songs on re-run.

### Step 2 — Preprocess
```bash
python preprocess_dataset.py --data ./dataset --out ./processed
```
Skips already-processed songs. ~1–2 hours for 5k songs.

### Step 3 — Train
```bash
python train_chartnet.py --data ./processed --out ./checkpoints --batch 600 --workers 16
```

AMP fp16 enabled automatically on CUDA. Early stopping (patience=8) stops training when F1 peaks.

| GPU | ~5k songs / epoch | Early stop (~40–60 epochs) |
|---|---|---|
| A100 SXM4 (cloud) | ~5–8 min | ~4–8 hours |
| RTX 4070 Super | ~60 min | ~2–3 days |
| RTX 3090 | ~40 min | ~1.5–2 days |
| RTX 3060 | ~90 min | ~3–4 days |

> **Cloud GPU recommended** — rent an A100 or RTX 3090 on [Vast.ai](https://vast.ai) for ~$0.20–0.50/hr. Training that takes 3 days locally finishes in ~6 hours.

**Checkpoints:** `best.pt` saved on every F1 improvement. `epoch_NNN.pt` saved each epoch (3 most recent kept). Resume with `--resume ./checkpoints/epoch_020.pt`.

**Training flags:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `160` | Max epochs |
| `--batch` | `48` | Batch size (increase on cloud GPU) |
| `--workers` | `4` | DataLoader workers (increase on cloud) |
| `--patience` | `8` | Early stopping patience (0 = disabled) |
| `--resume` | none | Resume from checkpoint |
| `--lr` | `2e-4` | Learning rate |
| `--reset-best` | off | Reset F1 baseline (use when switching datasets) |

### Step 4 — Use it
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
```

---

## Project Structure

```
clone-hero-chart-generator/
├── src/
│   ├── app.py                  GUI app (Generate / Scrape / Preprocess / Train)
│   ├── chartgen.py             Chart generator v6
│   ├── chartmodel.py           ChartNet v3 model (~50M params, CNN+Transformer)
│   ├── train_chartnet.py       Training loop with F1 tracking
│   ├── preprocess_dataset.py   Mel spectrogram preprocessing
│   ├── scraper.py              Enchor dataset downloader
│   ├── pipeline.sh             Full auto-pipeline script
│   └── chartgen.ico            App icon
├── release/
│   └── ChartGen.exe            Standalone Windows app
├── checkpoints/                Saved model checkpoints (gitignored)
├── build.bat                   One-click exe builder
└── requirements.txt
```

## Notes

- Default fret palette is **Green (0), Red (1), Yellow (2)** — use `--frets` to change
- Audio output is always `song.opus` (192k)
- `ffmpeg` must be in PATH
- ChartNet v3 best F1 achieved: **0.667** (trained on ~5.8k songs, A100 SXM4)
- Higher F1 = better charts. Target: 0.75+ for good quality, 0.80+ for great quality
