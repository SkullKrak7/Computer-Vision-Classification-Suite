"""
PyTorch CNN classifier for image classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class PyTorchCNNClassifier:
    def __init__(self, num_classes: int, input_shape: tuple = (224, 224, 3),
                 learning_rate: float = 0.001, device: str = None, use_amp: bool = True):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device == 'cuda'
        
        self.model = SimpleCNN(num_classes, input_shape[2]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.label_map = None
        self.is_trained = False
        
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}, AMP: {self.use_amp}")
        logger.info(f"Initialized PyTorch CNN on {self.device}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, label_map: dict,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 20, batch_size: int = 32):
        X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        y_train = torch.LongTensor(y_train)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, 
                                 shuffle=True, pin_memory=self.device=='cuda', num_workers=2)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
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
        
        self.label_map = label_map
        self.is_trained = True
        logger.info("PyTorch CNN training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        X = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(X)
            else:
                outputs = self.model(X)
            return outputs.argmax(dim=1).cpu().numpy()
    
    def save(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'label_map': self.label_map,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        checkpoint = torch.load(filepath)
        classifier = cls(checkpoint['num_classes'], tuple(checkpoint['input_shape']))
        classifier.model.load_state_dict(checkpoint['model_state'])
        classifier.label_map = {int(k): v for k, v in checkpoint['label_map'].items()}
        classifier.is_trained = True
        logger.info(f"Model loaded from {filepath}")
        return classifier
