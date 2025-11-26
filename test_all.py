#!/usr/bin/env python3
"""Comprehensive test suite for entire project"""

import sys
import subprocess
from pathlib import Path

def run_test(name, command):
    """Run a test and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {name} PASSED")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} FAILED")
        print(e.stderr)
        return False

def main():
    """Run all tests"""
    results = {}
    
    # Python module imports
    results['Python Imports'] = run_test(
        'Python Imports',
        'cd python && python -c "from src.models.deep_learning import PyTorchCNNClassifier, TFMobileNetClassifier; from src.models.baseline import KNNClassifier, SVMClassifier; print(\'All imports successful\')"'
    )
    
    # GPU detection
    results['GPU Detection'] = run_test(
        'GPU Detection',
        'python -c "import torch; import tensorflow as tf; print(f\'PyTorch CUDA: {torch.cuda.is_available()}\'); print(f\'TensorFlow GPU: {len(tf.config.list_physical_devices(\\\"GPU\\\"))>0}\')"'
    )
    
    # Python tests
    results['Dataset Tests'] = run_test(
        'Dataset Tests',
        'python python/tests/test_dataset.py'
    )
    
    results['Training Tests'] = run_test(
        'Training Tests',
        'python python/tests/test_training.py'
    )
    
    # Backend API tests
    results['Backend API'] = run_test(
        'Backend API',
        'python backend/tests/test_api.py'
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
