#!/bin/bash
set -e

source venv/bin/activate

echo "Training all models..."

# Train baseline
echo "Training baseline model..."
python python/scripts/train_baseline.py

# Train deep learning models
echo "Training CNN model..."
python python/scripts/train_cnn.py

echo "✓ All models trained!"
