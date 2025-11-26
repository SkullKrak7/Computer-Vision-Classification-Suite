"""Model training utilities"""

import numpy as np
from sklearn.model_selection import train_test_split

from .config import TrainingConfig


class Trainer:
    """Generic model trainer"""

    def __init__(self, model, config: TrainingConfig = None):
        self.model = model
        self.config = config or TrainingConfig()

    def train(self, X: np.ndarray, y: np.ndarray, label_map: dict):
        """Train model with validation split"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, random_state=42
        )

        self.model.train(
            X_train,
            y_train,
            label_map,
            X_val=X_val,
            y_val=y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
        )

        return self.model
