#!/bin/bash
set -e

echo "Setting up Computer Vision Classification Suite..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r python/requirements.txt

# Create model directories
mkdir -p models/{pytorch,tensorflow,onnx,baseline}

# Create dataset directories
mkdir -p datasets/{intel_images,car_damage,custom}

echo "✓ Setup complete!"
echo "Activate environment: source venv/bin/activate"
