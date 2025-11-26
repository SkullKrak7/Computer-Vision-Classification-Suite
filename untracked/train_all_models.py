#!/usr/bin/env python3
"""Train PyTorch and TensorFlow models with proper techniques"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from src.data.dataset import load_dataset
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications import EfficientNetB0
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PyTorch CNN with proper architecture
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_pytorch():
    """Train PyTorch CNN with proper techniques"""
    logger.info("=" * 60)
    logger.info("Training PyTorch CNN")
    logger.info("=" * 60)

    # Load data
    (X_train, X_test, y_train, y_test), labels = load_dataset(
        "datasets/intel_images/seg_train/seg_train", img_size=(64, 64)
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_test_t = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)

    # Data augmentation
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(len(labels)).to(device)

    # Class weights for imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training with early stopping
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_t.to(device)
            y_test_dev = y_test_t.to(device)
            val_outputs = model(X_test_dev)
            val_loss = criterion(val_outputs, y_test_dev).item()
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_test_dev).float().mean().item()

        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/pytorch_cnn_tuned.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load("models/pytorch_cnn_tuned.pth"))
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "label_map": {str(i): label for i, label in enumerate(labels)},
        "num_classes": len(labels),
    }

    with open("models/pytorch_cnn_tuned_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"PyTorch CNN - Acc: {acc:.4f}, F1: {f1:.4f}")
    return metrics


def train_tensorflow():
    """Train TensorFlow EfficientNetB0 (best for embedded)"""
    logger.info("=" * 60)
    logger.info("Training TensorFlow EfficientNetB0")
    logger.info("=" * 60)

    # Load data
    (X_train, X_test, y_train, y_test), labels = load_dataset(
        "datasets/intel_images/seg_train/seg_train",
        img_size=(224, 224),  # EfficientNet requires 224x224
    )

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Model - EfficientNetB0 is best for embedded (small, fast, accurate)
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze initially

    model = keras.Sequential(
        [
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(labels), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(patience=3, factor=0.5)

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Fine-tune
    logger.info("Fine-tuning last layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "label_map": {str(i): label for i, label in enumerate(labels)},
        "num_classes": len(labels),
    }

    model.save("models/tensorflow_efficientnet_tuned.keras")
    with open("models/tensorflow_efficientnet_tuned_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"TensorFlow EfficientNet - Acc: {acc:.4f}, F1: {f1:.4f}")
    return metrics


if __name__ == "__main__":
    logger.info("Starting model training with proper techniques...")

    pytorch_metrics = train_pytorch()
    tf_metrics = train_tensorflow()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"PyTorch CNN: {pytorch_metrics['accuracy']:.2%}")
    logger.info(f"TensorFlow EfficientNet: {tf_metrics['accuracy']:.2%}")
