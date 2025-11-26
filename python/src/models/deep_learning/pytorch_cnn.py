"""PyTorch CNN with OOP design"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import logging
from ..base import BaseModel

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """CNN architecture"""
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PyTorchCNNClassifier(BaseModel):
    """PyTorch CNN classifier with GPU optimization"""
    
    def __init__(self, num_classes: int, input_shape: tuple = (224, 224, 3),
                 learning_rate: float = 0.001, device: str = None, use_amp: bool = True):
        super().__init__(num_classes)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device == 'cuda'
        
        self.model = SimpleCNN(num_classes, input_shape[2]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"Initialized on {self.device}, AMP: {self.use_amp}")
    
    def train(self, X: np.ndarray, y: np.ndarray, label_map: dict,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 20, batch_size: int = 32):
        """Train model"""
        self.label_map = label_map
        
        X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2)
        y_tensor = torch.LongTensor(y)
        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device=='cuda',
            num_workers=2
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(X_tensor)
            else:
                outputs = self.model(X_tensor)
            return outputs.argmax(dim=1).cpu().numpy()
    
    def save(self, filepath: str):
        """Save model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'label_map': self.label_map,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }, filepath)
        logger.info(f"Saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['num_classes'], tuple(checkpoint['input_shape']))
        model.model.load_state_dict(checkpoint['model_state'])
        model.label_map = {int(k): v for k, v in checkpoint['label_map'].items()}
        model.is_trained = True
        return model
