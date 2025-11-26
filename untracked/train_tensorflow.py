#!/usr/bin/env python3
"""Train TensorFlow MobileNetV2 - best for embedded devices"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import logging

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from src.data.dataset import load_dataset
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.applications import MobileNetV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear GPU memory
tf.keras.backend.clear_session()

logger.info("Training TensorFlow MobileNetV2 (best for embedded)")

# Load data with smaller size
(X_train, X_test, y_train, y_test), labels = load_dataset(
    "datasets/intel_images/seg_train/seg_train", img_size=(96, 96)  # Smaller to fit in memory
)

# Class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# MobileNetV2 - optimized for mobile/embedded
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
base_model.trainable = False

model = keras.Sequential(
    [
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
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

# Train with smaller batch size
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,  # Smaller batch
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# Evaluate
y_pred = np.argmax(model.predict(X_test, batch_size=16), axis=1)
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

model.save("models/tensorflow_mobilenet_tuned.keras")
with open("models/tensorflow_mobilenet_tuned_metadata.json", "w") as f:
    json.dump(metrics, f, indent=2)

logger.info(f"TensorFlow MobileNetV2 - Acc: {acc:.4f}, F1: {f1:.4f}")
logger.info("Model saved successfully")
