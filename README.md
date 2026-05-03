# Clone Hero Chart Generator

Automatically generates Clone Hero `.chart` files from MP3s — with guitar solo detection, genre-aware patterns, and an optional trainable neural model.

## Features

- 🎸 **Guitar solo detection** — 5-signal detector (pyin voiced probability, spectral centroid, onset density, RMS, chroma peakiness). Solo regions are charted note-for-note at high resolution.
- 🎵 **Genre detection** — metal / pop / electronic / rock each get a different pattern generation strategy
- 🧠 **Phrase motif system** — repeated musical sections (chorus 1 ≈ chorus 2) get the **exact same** fret pattern
- 📊 **Note importance ranking** — downbeats, chord roots, and melody peaks are kept; filler notes dropped
- 🔁 **SSM phrase replication** — Self-Similarity Matrix finds structurally similar sections
- 🎤 **Lyrics** — Whisper large-v3 transcription with word-level timestamps
- 🎯 **4 difficulties** — Easy / Medium / Hard / Expert, all calibrated from real chart analysis
- 🤖 **Optional neural model** — train ChartNet on real charts for human-level note placement

## Requirements

```
pip install librosa numpy torch demucs openai-whisper mutagen Pillow scipy scikit-learn
pip install madmom          # optional but recommended for beat tracking
pip install rarfile         # optional, for .rar chart archives in scraper
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

### With trained neural model
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
```

## Training your own model

### Step 1 — Scrape charts (1–3 days)
```bash
python scraper.py --out ./dataset --limit 2000
```
Downloads chart+audio pairs from [Chorus](https://chorus.fightthe.pw).

### Step 2 — Preprocess (~30 min)
```bash
python preprocess_dataset.py --data ./dataset --out ./processed
```

### Step 3 — Train (4–8h on RTX 3090)
```bash
python train_chartnet.py --data ./processed --out ./checkpoints --epochs 60
```

Saves `checkpoints/best.pt` (best validation note F1).

### Step 4 — Use it
```bash
python chartgen.py --model ./checkpoints/best.pt "MySong.mp3"
```

## Files

| File | Purpose |
|---|---|
| `chartgen.py` | Main chart generator (v6) |
| `chartmodel.py` | ChartNet model definition (~8M params) |
| `scraper.py` | Downloads chart+audio pairs from Chorus API |
| `preprocess_dataset.py` | Converts charts to beat-aligned mel tensors |
| `train_chartnet.py` | Training loop with F1 tracking |

## Notes

- Only uses frets **Green (0), Red (1), Yellow (2)** — never Blue or Orange
- Audio output is always `song.opus` (192k)
- Album art saved as `album.jpg`
- Requires `ffmpeg` in PATH
