#!/usr/bin/env python3
"""Capture images from webcam for custom dataset"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.capture import WebcamCapture

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'datasets/custom'
    capture = WebcamCapture(output_dir)
    capture.run()
