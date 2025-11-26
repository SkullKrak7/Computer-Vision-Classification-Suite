#!/bin/bash
set -e

source venv/bin/activate

MODEL_PATH=${1:-"models/pytorch/best_model.pth"}
OUTPUT_DIR=${2:-"models/onnx"}

echo "Exporting model to ONNX..."
python -c "
import sys
sys.path.insert(0, 'python/src')
from export import export_model
export_model('$MODEL_PATH', '$OUTPUT_DIR')
print('✓ Export complete!')
"
