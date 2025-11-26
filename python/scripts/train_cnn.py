"""
Train deep learning models (PyTorch CNN and TensorFlow MobileNet)
Usage: python scripts/train_cnn.py --model pytorch
       python scripts/train_cnn.py --model tensorflow
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging

from src.data.dataset import load_dataset
from src.models.deep_learning.pytorch_cnn import PyTorchCNNClassifier
from src.models.deep_learning.tf_mobilenet import TFMobileNetClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pytorch(data_dir="datasets/custom", epochs=20):
    """Train PyTorch CNN"""
    logger.info("Training PyTorch CNN...")

    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size=(224, 224))

    model = PyTorchCNNClassifier(num_classes=len(labels), learning_rate=0.001)
    model.train(X_train, y_train, labels, epochs=epochs, batch_size=32)

    results = model.evaluate(X_test, y_test)
    logger.info(f"PyTorch CNN Accuracy: {results['accuracy']:.2%}")

    model.save("models/pytorch/cnn_model.pth")

    return model, results


def train_tensorflow(data_dir="datasets/custom", epochs=20, fine_tune=False):
    """Train TensorFlow MobileNetV2"""
    logger.info("Training TensorFlow MobileNetV2...")

    (X_train, X_test, y_train, y_test), labels = load_dataset(data_dir, img_size=(224, 224))

    model = TFMobileNetClassifier(num_classes=len(labels), learning_rate=0.001, fine_tune=fine_tune)

    model.train(X_train, y_train, labels, X_val=X_test, y_val=y_test, epochs=epochs, batch_size=32)

    results = model.evaluate(X_test, y_test)
    logger.info(f"TensorFlow MobileNetV2 Accuracy: {results['accuracy']:.2%}")

    model.save("models/tensorflow/mobilenet_model.h5")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pytorch", "tensorflow", "both"],
        default="pytorch",
        help="Model to train",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/custom", help="Path to dataset directory"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument(
        "--fine-tune", action="store_true", help="Fine-tune MobileNet layers (TensorFlow only)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Training Deep Learning Models")
    print("=" * 60 + "\n")

    if args.model in ["pytorch", "both"]:
        pytorch_model, pytorch_results = train_pytorch(args.data, args.epochs)
        print(f"\nPyTorch CNN Accuracy: {pytorch_results['accuracy']:.2%}")

    if args.model in ["tensorflow", "both"]:
        print("\n" + "-" * 60 + "\n")
        tf_model, tf_results = train_tensorflow(args.data, args.epochs, args.fine_tune)
        print(f"\nTensorFlow MobileNetV2 Accuracy: {tf_results['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60 + "\n")
