#!/usr/bin/env python3
"""GPU Benchmark for RTX 3060"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def benchmark_pytorch():
    import torch
    from models.deep_learning import PyTorchCNNClassifier
    
    print("\n=== PyTorch GPU Benchmark ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create larger dataset for benchmarking
    X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, 100)
    X_test = np.random.rand(50, 224, 224, 3).astype(np.float32)
    label_map = {i: f'class_{i}' for i in range(10)}
    
    # Benchmark with AMP
    model = PyTorchCNNClassifier(num_classes=10, use_amp=True)
    
    start = time.time()
    model.train(X_train, y_train, label_map, epochs=5, batch_size=16)
    train_time = time.time() - start
    
    start = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - start
    
    print(f"✓ Training time: {train_time:.2f}s")
    print(f"✓ Inference time: {pred_time:.3f}s ({len(X_test)/pred_time:.1f} imgs/sec)")
    
    if torch.cuda.is_available():
        print(f"✓ Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    return train_time, pred_time

def benchmark_tensorflow():
    import tensorflow as tf
    from models.deep_learning import TFMobileNetClassifier
    
    print("\n=== TensorFlow GPU Benchmark ===")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs Available: {len(gpus)}")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    
    # Create dataset
    X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, 100)
    X_test = np.random.rand(50, 224, 224, 3).astype(np.float32)
    label_map = {i: f'class_{i}' for i in range(10)}
    
    # Benchmark with mixed precision
    model = TFMobileNetClassifier(num_classes=10, use_mixed_precision=True)
    
    start = time.time()
    model.train(X_train, y_train, label_map, epochs=5, batch_size=16)
    train_time = time.time() - start
    
    start = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - start
    
    print(f"✓ Training time: {train_time:.2f}s")
    print(f"✓ Inference time: {pred_time:.3f}s ({len(X_test)/pred_time:.1f} imgs/sec)")
    
    return train_time, pred_time

if __name__ == '__main__':
    print("=" * 60)
    print("GPU Acceleration Benchmark - RTX 3060")
    print("=" * 60)
    
    try:
        pt_train, pt_pred = benchmark_pytorch()
        tf_train, tf_pred = benchmark_tensorflow()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"PyTorch  - Train: {pt_train:.2f}s, Inference: {pt_pred:.3f}s")
        print(f"TensorFlow - Train: {tf_train:.2f}s, Inference: {tf_pred:.3f}s")
        print("\n✓ GPU acceleration configured successfully!")
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
