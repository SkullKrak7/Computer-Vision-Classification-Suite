#!/usr/bin/env python3
"""Test script for deep learning models"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.deep_learning import TFMobileNetClassifier, PyTorchCNNClassifier

def test_pytorch():
    print("\n=== Testing PyTorch CNN ===")
    # Create dummy data
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, 10)
    X_test = np.random.rand(2, 224, 224, 3).astype(np.float32)
    label_map = {0: 'class_0', 1: 'class_1', 2: 'class_2'}
    
    # Initialize and train
    model = PyTorchCNNClassifier(num_classes=3, input_shape=(224, 224, 3))
    print("Training PyTorch model...")
    model.train(X_train, y_train, label_map, epochs=2, batch_size=4)
    
    # Predict
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"Predicted classes: {[label_map[p] for p in predictions]}")
    
    # Save and load
    save_path = Path('models/pytorch/test_model.pth')
    model.save(str(save_path))
    loaded_model = PyTorchCNNClassifier.load(str(save_path))
    predictions2 = loaded_model.predict(X_test)
    print(f"Loaded model predictions: {predictions2}")
    
    assert np.array_equal(predictions, predictions2), "Predictions don't match!"
    print(" PyTorch CNN test passed!")
    return True

def test_tensorflow():
    print("\n=== Testing TensorFlow MobileNet ===")
    # Create dummy data
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, 10)
    X_test = np.random.rand(2, 224, 224, 3).astype(np.float32)
    label_map = {0: 'class_0', 1: 'class_1', 2: 'class_2'}
    
    # Initialize and train
    model = TFMobileNetClassifier(num_classes=3, input_shape=(224, 224, 3))
    print("Training TensorFlow model...")
    model.train(X_train, y_train, label_map, epochs=2, batch_size=4)
    
    # Predict
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"Predicted classes: {[label_map[p] for p in predictions]}")
    
    # Save and load
    save_path = Path('models/tensorflow/test_model')
    model.save(str(save_path))
    loaded_model = TFMobileNetClassifier.load(str(save_path))
    predictions2 = loaded_model.predict(X_test)
    print(f"Loaded model predictions: {predictions2}")
    
    assert np.array_equal(predictions, predictions2), "Predictions don't match!"
    print(" TensorFlow MobileNet test passed!")
    return True

if __name__ == '__main__':
    try:
        pytorch_ok = test_pytorch()
        tf_ok = test_tensorflow()
        
        if pytorch_ok and tf_ok:
            print("\n All tests passed!")
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
