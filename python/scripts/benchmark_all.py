"""
Benchmark all models and compare results
Usage: python scripts/benchmark_all.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import load_dataset
from src.models.baseline.knn import KNNClassifier
from src.models.baseline.svm import SVMClassifier
from src.models.deep_learning.pytorch_cnn import PyTorchCNNClassifier
from src.models.deep_learning.tf_mobilenet import TFMobileNetClassifier
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_model(model_name, model, X_test, y_test):
    """Benchmark a single model"""
    logger.info(f"Benchmarking {model_name}...")
    
    start_time = time.time()
    results = model.evaluate(X_test, y_test)
    inference_time = time.time() - start_time
    
    avg_time_per_sample = inference_time / len(X_test) * 1000
    
    return {
        'model': model_name,
        'accuracy': results['accuracy'],
        'inference_time_total': inference_time,
        'inference_time_per_sample_ms': avg_time_per_sample
    }


def main():
    print("\n" + "="*60)
    print("Model Benchmark Comparison")
    print("="*60 + "\n")
    
    (X_train, X_test, y_train, y_test), labels = load_dataset('datasets/custom', img_size=(224, 224))
    
    results = []
    
    try:
        knn = KNNClassifier.load('models/baseline/knn_model.pkl')
        X_test_64 = load_dataset('datasets/custom', img_size=(64, 64))[0][1]
        results.append(benchmark_model('KNN', knn, X_test_64, y_test))
    except FileNotFoundError:
        logger.warning("KNN model not found, skipping")
    
    try:
        svm = SVMClassifier.load('models/baseline/svm_model.pkl')
        X_test_64 = load_dataset('datasets/custom', img_size=(64, 64))[0][1]
        results.append(benchmark_model('SVM', svm, X_test_64, y_test))
    except FileNotFoundError:
        logger.warning("SVM model not found, skipping")
    
    try:
        pytorch_cnn = PyTorchCNNClassifier.load('models/pytorch/cnn_model.pth')
        results.append(benchmark_model('PyTorch CNN', pytorch_cnn, X_test, y_test))
    except FileNotFoundError:
        logger.warning("PyTorch CNN model not found, skipping")
    
    try:
        tf_mobilenet = TFMobileNetClassifier.load('models/tensorflow/mobilenet_model.h5')
        results.append(benchmark_model('TensorFlow MobileNetV2', tf_mobilenet, X_test, y_test))
    except FileNotFoundError:
        logger.warning("TensorFlow MobileNet model not found, skipping")
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "="*60)
    print("Benchmark Results (sorted by accuracy)")
    print("="*60)
    print(f"{'Model':<30} {'Accuracy':<12} {'Inference (ms/sample)'}")
    print("-"*60)
    
    for result in results:
        print(f"{result['model']:<30} {result['accuracy']:<12.2%} {result['inference_time_per_sample_ms']:.2f}")
    
    print("="*60 + "\n")
    
    best_model = results[0]
    print(f"Best Model: {best_model['model']} ({best_model['accuracy']:.2%})")
    print()


if __name__ == "__main__":
    main()
