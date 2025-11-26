"""
Train baseline models (KNN and SVM)
Usage: python scripts/train_baseline.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import load_dataset
from src.models.baseline.knn import KNNClassifier
from src.models.baseline.svm import SVMClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_knn(data_dir='datasets/custom', img_size=(64, 64)):
    """Train KNN baseline"""
    logger.info("Training KNN baseline...")
    
    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size)
    
    model = KNNClassifier(n_neighbors=3)
    model.train(X_train, y_train, labels)
    
    results = model.evaluate(X_test, y_test)
    logger.info(f"KNN Accuracy: {results['accuracy']:.2%}")
    
    model.save('models/baseline/knn_model.pkl')
    
    return model, results


def train_svm(data_dir='datasets/custom', img_size=(64, 64)):
    """Train SVM baseline"""
    logger.info("Training SVM baseline...")
    
    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size)
    
    model = SVMClassifier(kernel='rbf', C=1.0)
    model.train(X_train, y_train, labels)
    
    results = model.evaluate(X_test, y_test)
    logger.info(f"SVM Accuracy: {results['accuracy']:.2%}")
    
    model.save('models/baseline/svm_model.pkl')
    
    return model, results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Training Baseline Models")
    print("="*60 + "\n")
    
    knn_model, knn_results = train_knn()
    print("\n" + "-"*60 + "\n")
    svm_model, svm_results = train_svm()
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"KNN Accuracy: {knn_results['accuracy']:.2%}")
    print(f"SVM Accuracy: {svm_results['accuracy']:.2%}")
    print("="*60 + "\n")
