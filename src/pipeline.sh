#!/bin/bash
# Auto pipeline: wait for training → scrape 10k (genre-diverse) → preprocess → retrain
cd "C:/Users/MrSchneider/Downloads/gg/spotifit"

echo "=== PIPELINE START ==="

# Step 1: Wait for current training to finish
echo "[pipeline] Waiting for current training to finish..."
while pgrep -f "train_chartnet.py" > /dev/null; do
    sleep 60
done
echo "[pipeline] Training finished!"

# Step 2: Genre-diverse scrape to 10k songs
# Scrape multiple genres so the model learns from varied music styles
echo "[pipeline] Scraping genre-diverse dataset to 10k songs..."
python scraper.py --out ./dataset --limit 2500 --query "metal"
python scraper.py --out ./dataset --limit 2500 --query "rock"
python scraper.py --out ./dataset --limit 2000 --query "punk"
python scraper.py --out ./dataset --limit 1500 --query "pop"
python scraper.py --out ./dataset --limit 1500 --query "alternative"
echo "[pipeline] Genre-diverse scrape done!"

# Step 3: Preprocess new songs
echo "[pipeline] Preprocessing..."
python preprocess_dataset.py --data ./dataset --out ./processed
echo "[pipeline] Preprocessing done!"

# Step 4: Resume training from best checkpoint with LR + best reset
# --reset-best: new val set is harder, reset best_f1 so early stopping is fair
# --reset-lr:   fresh LR schedule for the larger dataset
echo "[pipeline] Starting next training run (ChartNet v3, epochs=160)..."
python train_chartnet.py \
    --data ./processed \
    --out ./checkpoints \
    --epochs 160 \
    --reset-best \
    --reset-lr
echo "[pipeline] Training round 2 done!"

echo "=== PIPELINE COMPLETE ==="
