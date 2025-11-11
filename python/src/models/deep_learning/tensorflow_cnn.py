"""
TensorFlow MobileNetV2 transfer learning classifier
Uses pre-trained MobileNetV2 for efficient image classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TFMobileNetClassifier:
    """
    Transfer learning classifier using MobileNetV2 backbone
    """
    
    def __init__(self, num_classes: int, input_shape: tuple = (224, 224, 3),
                 learning_rate: float = 0.001, fine_tune: bool = False):
        """
        Initialize MobileNetV2 classifier
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (H, W, C)
            learning_rate: Learning rate for optimizer
            fine_tune: Whether to fine-tune base model layers
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.fine_tune = fine_tune
        
        self.model = self._build_model()
        self.label_map = None
        self.is_trained = False
        
        logger.info(f"Initialized MobileNetV2 (fine_tune={fine_tune})")
    
    def _build_model(self):
        """Build transfer learning model"""
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = self.fine_tune
        
        if self.fine_tune:
            for layer in base_model.layers[:-30]:
                layer.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, label_map: dict,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 20, batch_size: int = 32):
        """
        Train MobileNetV2 model
        
        Args:
            X_train: Training images, shape (N, H, W, C)
            y_train: Training labels
            label_map: Dictionary mapping class indices to names
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info(f"Training MobileNetV2 for {epochs} epochs")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.label_map = label_map
        self.is_trained = True
        
        logger.info("MobileNetV2 training complete")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.argmax(axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model"""
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        target_names = [self.label_map[i] for i in sorted(self.label_map.keys())]
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def save(self, filepath: str):
        """Save model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'label_map': self.label_map,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'learning_rate': self.learning_rate,
                'fine_tune': self.fine_tune
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model"""
        import json
        
        filepath = Path(filepath)
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        classifier = cls(
            num_classes=metadata['num_classes'],
            input_shape=tuple(metadata['input_shape']),
            learning_rate=metadata['learning_rate'],
            fine_tune=metadata['fine_tune']
        )
        
        classifier.model = keras.models.load_model(filepath)
        classifier.label_map = metadata['label_map']
        classifier.label_map = {int(k): v for k, v in classifier.label_map.items()}
        classifier.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return classifier
