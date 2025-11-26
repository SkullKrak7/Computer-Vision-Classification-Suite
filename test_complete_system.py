#!/usr/bin/env python3
"""Comprehensive system test for all models and components"""

import json
import sys
from pathlib import Path

def test_metrics_files():
    """Test that all metrics files exist and are valid"""
    print("\n" + "="*60)
    print("Testing Metrics Files")
    print("="*60)
    
    metrics_files = {
        'KNN': 'models/baseline/knn_metrics.json',
        'SVM': 'models/baseline/svm_metrics.json',
        'PyTorch CNN': 'models/pytorch_cnn_tuned_metadata.json',
        'TensorFlow MobileNet': 'models/tensorflow_mobilenet_tuned_metadata.json'
    }
    
    all_passed = True
    for model_name, filepath in metrics_files.items():
        path = Path(filepath)
        if not path.exists():
            print(f"❌ {model_name}: File not found - {filepath}")
            all_passed = False
            continue
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            required_keys = ['accuracy', 'precision', 'recall', 'f1_score']
            missing_keys = [k for k in required_keys if k not in data]
            
            if missing_keys:
                print(f"❌ {model_name}: Missing keys - {missing_keys}")
                all_passed = False
            else:
                print(f"✅ {model_name}: Accuracy={data['accuracy']:.4f}, F1={data['f1_score']:.4f}")
        except Exception as e:
            print(f"❌ {model_name}: Error reading file - {e}")
            all_passed = False
    
    return all_passed

def test_model_files():
    """Test that all model files exist"""
    print("\n" + "="*60)
    print("Testing Model Files")
    print("="*60)
    
    model_files = {
        'KNN': 'models/baseline/knn_model.pkl',
        'SVM': 'models/baseline/svm_model.pkl',
        'PyTorch CNN': 'models/pytorch_cnn_tuned.pth',
        'TensorFlow MobileNet': 'models/tensorflow_mobilenet_tuned.keras'
    }
    
    all_passed = True
    for model_name, filepath in model_files.items():
        path = Path(filepath)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name}: {size_mb:.2f} MB")
        else:
            print(f"❌ {model_name}: File not found - {filepath}")
            all_passed = False
    
    return all_passed

def test_backend_structure():
    """Test backend structure"""
    print("\n" + "="*60)
    print("Testing Backend Structure")
    print("="*60)
    
    required_files = [
        'backend/app/main.py',
        'backend/app/routes/metrics.py',
        'backend/app/routes/inference.py',
        'backend/app/models.py'
    ]
    
    all_passed = True
    for filepath in required_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            all_passed = False
    
    return all_passed

def test_frontend_structure():
    """Test frontend structure"""
    print("\n" + "="*60)
    print("Testing Frontend Structure")
    print("="*60)
    
    required_files = [
        'frontend/src/App.jsx',
        'frontend/src/components/MetricsChart.jsx',
        'frontend/src/components/ModelComparison.jsx',
        'frontend/src/services/api.js',
        'frontend/dist/index.html'
    ]
    
    all_passed = True
    for filepath in required_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            all_passed = False
    
    return all_passed

def print_summary():
    """Print model performance summary"""
    print("\n" + "="*60)
    print("Model Performance Summary")
    print("="*60)
    
    metrics_files = {
        'KNN': 'models/baseline/knn_metrics.json',
        'SVM': 'models/baseline/svm_metrics.json',
        'PyTorch CNN': 'models/pytorch_cnn_tuned_metadata.json',
        'TensorFlow MobileNet': 'models/tensorflow_mobilenet_tuned_metadata.json'
    }
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 73)
    
    for model_name, filepath in metrics_files.items():
        try:
            with open(filepath) as f:
                data = json.load(f)
            print(f"{model_name:<25} {data['accuracy']:<12.4f} {data['precision']:<12.4f} {data['recall']:<12.4f} {data['f1_score']:<12.4f}")
        except:
            print(f"{model_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

def main():
    print("\n" + "="*60)
    print("Computer Vision Classification Suite - System Test")
    print("="*60)
    
    tests = [
        ("Metrics Files", test_metrics_files),
        ("Model Files", test_model_files),
        ("Backend Structure", test_backend_structure),
        ("Frontend Structure", test_frontend_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print_summary()
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - System is ready!")
    else:
        print("⚠️  SOME TESTS FAILED - Please review errors above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
