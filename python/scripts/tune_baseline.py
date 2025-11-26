"""
Automated hyperparameter tuning for baseline models (KNN and SVM)
Usage: python scripts/tune_baseline.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import logging

from src.data.dataset import load_dataset
from src.models.baseline.knn import KNNClassifier
from src.models.baseline.svm import SVMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tune_knn(data_dir="datasets/intel_images/seg_train/seg_train", img_size=(64, 64)):
    """Tune KNN with different k values"""
    logger.info("STARTING AUTO-TUNING: KNN")

    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size)

    best_acc = 0
    best_k = 3
    best_results = None

    for k in [1, 3, 5, 7, 9]:
        logger.info(f"Testing k={k}")
        model = KNNClassifier(n_neighbors=k)
        model.train(X_train, y_train, labels)
        results = model.evaluate(X_test, y_test)

        logger.info(f"k={k}, Accuracy: {results['accuracy']:.4f}")

        if results["accuracy"] > best_acc:
            best_acc = results["accuracy"]
            best_k = k
            best_results = results

    logger.info(f"Best k={best_k} with accuracy {best_acc:.4f}")

    model = KNNClassifier(n_neighbors=best_k)
    model.train(X_train, y_train, labels)
    model.save("models/baseline/knn_model.pkl")

    with open("models/baseline/knn_metrics.json", "w") as f:
        json.dump(best_results, f, indent=2)

    return model, best_results


def tune_svm(data_dir="datasets/intel_images/seg_train/seg_train", img_size=(64, 64)):
    """Tune SVM with different kernels and C values"""
    logger.info("STARTING AUTO-TUNING: SVM")

    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size)

    best_acc = 0
    best_params = {"kernel": "rbf", "C": 1.0}
    best_results = None

    for kernel in ["linear", "rbf"]:
        for C in [0.1, 1.0, 10.0]:
            logger.info(f"Testing kernel={kernel}, C={C}")
            model = SVMClassifier(kernel=kernel, C=C)
            model.train(X_train, y_train, labels)
            results = model.evaluate(X_test, y_test)

            logger.info(f"kernel={kernel}, C={C}, Accuracy: {results['accuracy']:.4f}")

            if results["accuracy"] > best_acc:
                best_acc = results["accuracy"]
                best_params = {"kernel": kernel, "C": C}
                best_results = results

    logger.info(f"Best params: {best_params} with accuracy {best_acc:.4f}")

    model = SVMClassifier(**best_params)
    model.train(X_train, y_train, labels)
    model.save("models/baseline/svm_model.pkl")

    with open("models/baseline/svm_metrics.json", "w") as f:
        json.dump(best_results, f, indent=2)

    return model, best_results


if __name__ == "__main__":
    knn_model, knn_results = tune_knn()
    svm_model, svm_results = tune_svm()

    logger.info("\nFINAL RESULTS:")
    logger.info(f"KNN Accuracy: {knn_results['accuracy']:.4f}")
    logger.info(f"SVM Accuracy: {svm_results['accuracy']:.4f}")
