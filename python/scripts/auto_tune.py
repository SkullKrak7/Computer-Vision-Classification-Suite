#!/usr/bin/env python3
"""Automated hyperparameter tuning with iterative optimization"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import numpy as np
from data import DataConfig, DatasetLoader, augment_image
from evaluation import compute_metrics
from models.deep_learning import PyTorchCNNClassifier, TFMobileNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTuner:
    """Automated model tuning with bias-variance optimization"""

    def __init__(self, model_class, model_name, dataset_path, num_classes):
        self.model_class = model_class
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.best_model = None
        self.best_score = 0
        self.best_config = None
        self.history = []

    def load_data(self, sample_size=500):
        """Load and prepare data with class balancing"""
        config = DataConfig(img_size=(224, 224), test_size=0.2, normalize=True)
        loader = DatasetLoader(self.dataset_path, config)
        (X_train, X_test, y_train, y_test), label_map = loader.load()

        # Sample
        if len(X_train) > sample_size:
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]

        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")

        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 2.0:
            logger.warning(f"Class imbalance detected: {imbalance_ratio:.2f}x")
            # Compute class weights
            class_weights = compute_class_weight("balanced", classes=unique, y=y_train)
            class_weights = dict(zip(unique, class_weights))
        else:
            class_weights = None

        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        return (X_train, X_val, X_test, y_train, y_val, y_test, label_map, class_weights)

    def apply_augmentation(self, X, y, factor=2):
        """Apply data augmentation to reduce overfitting"""
        X_aug, y_aug = [], []
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
            for _ in range(factor - 1):
                aug_img = augment_image(X[i], flip=True, rotate=True, brightness=True)
                X_aug.append(aug_img)
                y_aug.append(y[i])
        return np.array(X_aug), np.array(y_aug)

    def evaluate_model(self, model, X_train, y_train, X_val, y_val):
        """Evaluate model and detect overfitting/underfitting"""
        train_preds = model.predict(X_train[:50])
        val_preds = model.predict(X_val[:50])

        train_metrics = compute_metrics(y_train[:50], train_preds)
        val_metrics = compute_metrics(y_val[:50], val_preds)

        gap = train_metrics["accuracy"] - val_metrics["accuracy"]

        return {
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "gap": gap,
            "train_f1": train_metrics["f1_score"],
            "val_f1": val_metrics["f1_score"],
        }

    def tune_iteration(self, config, X_train, y_train, X_val, y_val, label_map, iteration):
        """Single tuning iteration"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Iteration {iteration}: {config}")
        logger.info(f"{'='*70}")

        # Apply augmentation if needed
        if config.get("augment", False):
            X_train_aug, y_train_aug = self.apply_augmentation(X_train, y_train, factor=2)
        else:
            X_train_aug, y_train_aug = X_train, y_train

        # Train model
        model = self.model_class(num_classes=self.num_classes, **config["model_params"])
        model.train(
            X_train_aug,
            y_train_aug,
            label_map,
            X_val=X_val,
            y_val=y_val,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
        )

        # Evaluate
        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val)

        logger.info(
            f"Train Acc: {metrics['train_acc']:.4f}, Val Acc: {metrics['val_acc']:.4f}, Gap: {metrics['gap']:.4f}"
        )

        # Determine if optimal
        is_optimal = (
            metrics["gap"] < 0.15  # Not overfitting
            and metrics["val_acc"] > 0.65  # Good performance
            and metrics["val_f1"] > 0.60  # Good F1 score
        )

        # Track best
        score = metrics["val_acc"] - abs(metrics["gap"]) * 0.5
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
            self.best_config = config

        self.history.append(
            {"iteration": iteration, "config": config, "metrics": metrics, "is_optimal": is_optimal}
        )

        return metrics, is_optimal

    def get_next_config(self, current_config, metrics):
        """Determine next configuration based on results"""
        new_config = current_config.copy()

        if metrics["gap"] > 0.15:  # Overfitting
            logger.info("Overfitting detected - adjusting...")
            if "dropout" in new_config["model_params"]:
                new_config["model_params"]["dropout"] = min(
                    0.7, new_config["model_params"].get("dropout", 0.3) + 0.1
                )
            new_config["augment"] = True
            new_config["model_params"]["learning_rate"] *= 0.8

        elif metrics["val_acc"] < 0.65:  # Underfitting
            logger.info("Underfitting detected - adjusting...")
            new_config["epochs"] = min(20, new_config["epochs"] + 3)
            new_config["model_params"]["learning_rate"] *= 1.2
            if "dropout" in new_config["model_params"]:
                new_config["model_params"]["dropout"] = max(
                    0.1, new_config["model_params"].get("dropout", 0.3) - 0.1
                )

        return new_config

    def tune(self, max_iterations=5):
        """Run iterative tuning"""
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING AUTO-TUNING: {self.model_name}")
        logger.info(f"{'='*70}")

        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test, label_map, class_weights = self.load_data()

        # Initial config
        if "PyTorch" in self.model_name:
            config = {
                "model_params": {"learning_rate": 0.001, "use_amp": True},
                "epochs": 5,
                "batch_size": 32,
                "augment": False,
            }
        else:  # TensorFlow
            config = {
                "model_params": {
                    "learning_rate": 0.0001,
                    "use_mixed_precision": True,
                    "fine_tune": False,
                },
                "epochs": 5,
                "batch_size": 16,
                "augment": False,
            }

        # Iterative tuning
        for i in range(1, max_iterations + 1):
            metrics, is_optimal = self.tune_iteration(
                config, X_train, y_train, X_val, y_val, label_map, i
            )

            if is_optimal:
                logger.info(f"\nOptimal configuration found at iteration {i}!")
                break

            if i < max_iterations:
                config = self.get_next_config(config, metrics)

        # Save best model
        if "PyTorch" in self.model_name:
            save_path = f'models/{self.model_name.lower().replace(" ", "_")}_tuned.pth'
        else:
            save_path = f'models/{self.model_name.lower().replace(" ", "_")}_tuned.keras'

        if self.best_model:
            self.best_model.save(save_path)
            logger.info(f"\nBest model saved: {save_path}")

        return self.best_model, self.best_config, self.history


def main():
    """Run auto-tuning for all models"""
    dataset_path = "datasets/intel_images/seg_train/seg_train"
    num_classes = 6

    results = {}

    # Tune PyTorch
    logger.info("\n" + "=" * 70)
    logger.info("TUNING PYTORCH CNN")
    logger.info("=" * 70)

    pytorch_tuner = ModelTuner(PyTorchCNNClassifier, "PyTorch CNN", dataset_path, num_classes)
    model, config, history = pytorch_tuner.tune(max_iterations=5)
    results["pytorch"] = {"model": model, "config": config, "history": history}

    # Tune TensorFlow
    logger.info("\n" + "=" * 70)
    logger.info("TUNING TENSORFLOW MOBILENET")
    logger.info("=" * 70)

    tf_tuner = ModelTuner(TFMobileNetClassifier, "TensorFlow MobileNet", dataset_path, num_classes)
    model, config, history = tf_tuner.tune(max_iterations=5)
    results["tensorflow"] = {"model": model, "config": config, "history": history}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TUNING COMPLETE - SUMMARY")
    logger.info("=" * 70)

    for name, result in results.items():
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Best Config: {result['config']}")
        logger.info(f"  Iterations: {len(result['history'])}")
        final_metrics = result["history"][-1]["metrics"]
        logger.info(f"  Final Train Acc: {final_metrics['train_acc']:.4f}")
        logger.info(f"  Final Val Acc: {final_metrics['val_acc']:.4f}")
        logger.info(f"  Final Gap: {final_metrics['gap']:.4f}")


if __name__ == "__main__":
    main()
