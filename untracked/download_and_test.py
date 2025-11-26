#!/usr/bin/env python3
"""Download Kaggle datasets and test all models with proper tuning"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import kaggle
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset options
DATASETS = {
    "1": {
        "name": "Intel Image Classification",
        "kaggle_id": "puneet6060/intel-image-classification",
        "path": "datasets/intel_images",
        "classes": 6,
    },
    "2": {
        "name": "Cats vs Dogs",
        "kaggle_id": "tongpython/cat-and-dog",
        "path": "datasets/cats_dogs",
        "classes": 2,
    },
    "3": {
        "name": "Flowers Recognition",
        "kaggle_id": "alxmamaev/flowers-recognition",
        "path": "datasets/flowers",
        "classes": 5,
    },
}


def download_dataset(dataset_id):
    """Download dataset from Kaggle"""
    dataset = DATASETS[dataset_id]
    print(f"\nDownloading {dataset['name']}...")
    print(f"Kaggle ID: {dataset['kaggle_id']}")

    try:
        os.makedirs(dataset["path"], exist_ok=True)
        kaggle.api.dataset_download_files(dataset["kaggle_id"], path=dataset["path"], unzip=True)
        print(f"Downloaded to {dataset['path']}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Kaggle API credentials at ~/.kaggle/kaggle.json")
        print("2. Ensure you've accepted the dataset's terms on Kaggle website")
        print("3. Verify your internet connection")
        return False


def test_pytorch_model(X_train, X_val, y_train, y_val, label_map, num_classes):
    """Test PyTorch model with tuning"""
    from models.deep_learning import PyTorchCNNClassifier

    print("\n" + "=" * 70)
    print("TESTING PYTORCH CNN")
    print("=" * 70)

    # Hyperparameters tuned for bias-variance tradeoff
    config = {"num_classes": num_classes, "learning_rate": 0.001, "use_amp": True}

    print(f"Configuration: {config}")

    model = PyTorchCNNClassifier(**config)

    # Training with validation
    print("\nTraining...")
    model.train(X_train, y_train, label_map, X_val=X_val, y_val=y_val, epochs=10, batch_size=32)

    # Evaluate
    print("\nEvaluating...")
    train_preds = model.predict(X_train[:100])
    val_preds = model.predict(X_val[:100])

    from evaluation import compute_metrics

    train_metrics = compute_metrics(y_train[:100], train_preds)
    val_metrics = compute_metrics(y_val[:100], val_preds)

    print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Overfitting Check: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f}")

    if train_metrics["accuracy"] - val_metrics["accuracy"] > 0.15:
        print("WARNING: Model may be overfitting")
        print("Recommendation: Increase dropout, add regularization, or reduce model complexity")
    elif val_metrics["accuracy"] < 0.6:
        print("WARNING: Model may be underfitting")
        print("Recommendation: Increase model capacity, train longer, or reduce regularization")
    else:
        print("Model shows good bias-variance tradeoff")

    # Save model
    save_path = "models/pytorch/tested_model.pth"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return train_metrics, val_metrics


def test_tensorflow_model(X_train, X_val, y_train, y_val, label_map, num_classes):
    """Test TensorFlow model with tuning"""
    from models.deep_learning import TFMobileNetClassifier

    print("\n" + "=" * 70)
    print("TESTING TENSORFLOW MOBILENET")
    print("=" * 70)

    # Hyperparameters tuned for bias-variance tradeoff
    config = {
        "num_classes": num_classes,
        "learning_rate": 0.0001,  # Lower LR for transfer learning
        "fine_tune": False,  # Start without fine-tuning
        "use_mixed_precision": True,
    }

    print(f"Configuration: {config}")

    model = TFMobileNetClassifier(**config)

    # Training with validation
    print("\nTraining...")
    model.train(X_train, y_train, label_map, X_val=X_val, y_val=y_val, epochs=10, batch_size=16)

    # Evaluate
    print("\nEvaluating...")
    train_preds = model.predict(X_train[:100], batch_size=16)
    val_preds = model.predict(X_val[:100], batch_size=16)

    from evaluation import compute_metrics

    train_metrics = compute_metrics(y_train[:100], train_preds)
    val_metrics = compute_metrics(y_val[:100], val_preds)

    print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Overfitting Check: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f}")

    if train_metrics["accuracy"] - val_metrics["accuracy"] > 0.15:
        print("WARNING: Model may be overfitting")
        print("Recommendation: Add dropout, reduce learning rate, or use data augmentation")
    elif val_metrics["accuracy"] < 0.6:
        print("WARNING: Model may be underfitting")
        print("Recommendation: Enable fine-tuning or train longer")
    else:
        print("Model shows good bias-variance tradeoff")

    # Save model
    save_path = "models/tensorflow/tested_model"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return train_metrics, val_metrics


def load_and_prepare_data(dataset_path, num_classes, sample_size=1000):
    """Load and prepare dataset"""
    from data import DataConfig, DatasetLoader

    print(f"\nLoading dataset from {dataset_path}...")

    config = DataConfig(img_size=(224, 224), test_size=0.2, normalize=True)

    try:
        loader = DatasetLoader(dataset_path, config)
        (X_train, X_test, y_train, y_test), label_map = loader.load()

        # Sample for faster testing
        if len(X_train) > sample_size:
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        if len(X_test) > sample_size // 4:
            indices = np.random.choice(len(X_test), sample_size // 4, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"Classes: {label_map}")

        return X_train, X_test, y_train, y_test, label_map

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None


def main():
    print("=" * 70)
    print("KAGGLE DATASET TESTING WITH MODEL TUNING")
    print("=" * 70)

    # Show available datasets
    print("\nAvailable datasets:")
    for key, dataset in DATASETS.items():
        print(f"{key}. {dataset['name']} ({dataset['classes']} classes)")

    # Get user choice
    choice = input("\nSelect dataset (1-3) or 'q' to quit: ").strip()

    if choice.lower() == "q":
        print("Exiting...")
        return

    if choice not in DATASETS:
        print("Invalid choice")
        return

    dataset = DATASETS[choice]

    # Check if dataset exists
    if not os.path.exists(dataset["path"]) or len(os.listdir(dataset["path"])) == 0:
        print("\nDataset not found locally. Download it? (y/n)")
        download = input().strip().lower()

        if download == "y":
            if not download_dataset(choice):
                print("\nDownload failed. Please:")
                print("1. Visit https://www.kaggle.com/account")
                print("2. Create API token and download kaggle.json")
                print("3. Place it at ~/.kaggle/kaggle.json")
                print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
                return
        else:
            print("Cannot proceed without dataset")
            return

    # Load data
    X_train, X_test, y_train, y_test, label_map = load_and_prepare_data(
        dataset["path"], dataset["classes"]
    )

    if X_train is None:
        print("Failed to load dataset")
        return

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("\nFinal split:")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Test models
    print("\nWhich model to test?")
    print("1. PyTorch CNN")
    print("2. TensorFlow MobileNet")
    print("3. Both")

    model_choice = input("Choice (1-3): ").strip()

    results = {}

    if model_choice in ["1", "3"]:
        try:
            train_m, val_m = test_pytorch_model(
                X_train, X_val, y_train, y_val, label_map, dataset["classes"]
            )
            results["pytorch"] = {"train": train_m, "val": val_m}
        except Exception as e:
            print(f"\nPyTorch test failed: {e}")
            import traceback

            traceback.print_exc()

    if model_choice in ["2", "3"]:
        try:
            train_m, val_m = test_tensorflow_model(
                X_train, X_val, y_train, y_val, label_map, dataset["classes"]
            )
            results["tensorflow"] = {"train": train_m, "val": val_m}
        except Exception as e:
            print(f"\nTensorFlow test failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TESTING COMPLETE - SUMMARY")
    print("=" * 70)

    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Train Accuracy: {metrics['train']['accuracy']:.4f}")
        print(f"  Val Accuracy: {metrics['val']['accuracy']:.4f}")
        print(
            f"  Generalization Gap: {metrics['train']['accuracy'] - metrics['val']['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
