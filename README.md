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
- 🤖 **Neural model (ChartNet v2, ~35M params)** — predicts note placement, fret, sustain length, chords, and note type (HOPO/tap/open strum) all in one shot

## Requirements

```
pip install librosa numpy torch demucs openai-whisper mutagen Pillow scipy scikit-learn
pip install madmom          # optional but recommended for beat tracking
pip install requests        # for scraper
```

ffmpeg must be in PATH (for opus conversion).

## Usage

### Generate a chart
```bash
python chartgen.py "MySong.mp3"
python chartgen.py "MySong.mp3" "AnotherSong.mp3"
```

Output goes to:
```
C:\Users\<you>\OneDrive\Tiedostot\Clone Hero\Songs\Artist - Title\
  song.opus
  notes.chart
  song.ini
  album.jpg
  background.jpg
```

Edit `OUT_DIR` in `chartgen.py` to change the output path.

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
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
python chartgen.py --model ./checkpoints/best.pt --frets 0,1,2,3,4 "MySong.mp3"
```

## Training your own model

### Step 1 — Scrape charts (1–3 hours)
```bash
python scraper.py --out ./dataset --limit 2000
```
Downloads chart+audio pairs from [Enchor](https://enchor.us) (Chorus Encore). Each song is validated — must have an ExpertSingle track with 50+ notes and a valid audio file.

### Step 2 — Preprocess (~30 min)
```bash
python preprocess_dataset.py --data ./dataset --out ./processed
```

### Step 3 — Train (6–12h on RTX 3090)
```bash
python train_chartnet.py --data ./processed --out ./checkpoints --epochs 60
```

Saves `checkpoints/best.pt` (best validation note F1). If you run out of VRAM, reduce batch size: `--batch 16`.

### Step 4 — Use it
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
```

## Files

| File | Purpose |
|---|---|
| `chartgen.py` | Main chart generator (v6) |
| `chartmodel.py` | ChartNet model definition (~8M params) |
| `scraper.py` | Downloads chart+audio pairs from Enchor API |
| `preprocess_dataset.py` | Converts charts to beat-aligned mel tensors |
| `train_chartnet.py` | Training loop with F1 tracking |

## Notes

- Default fret palette is **Green (0), Red (1), Yellow (2)** — use `--frets` to change
- Audio output is always `song.opus` (192k)
- Album art saved as `album.jpg`
- Requires `ffmpeg` in PATH
