"""Numpy 2.x compatibility patch for tf2onnx"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np

if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'str'):
    np.str = str

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.protobuf')
