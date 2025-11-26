"""Base model class with OOP design"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, num_classes: int, **kwargs):
        self.num_classes = num_classes
        self.label_map: Dict[int, str] = {}
        self.is_trained = False
        self._validate_params()
    
    def _validate_params(self):
        """Validate model parameters"""
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, label_map: Dict[int, str], **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        from ..evaluation.metrics import compute_metrics
        predictions = self.predict(X)
        return compute_metrics(y, predictions)
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'num_classes': self.num_classes,
            'is_trained': self.is_trained,
            'label_map': self.label_map
        }
