"""SVM Classifier"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import numpy as np


class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, gamma='scale')
        self.label_map = None
    
    def train(self, X_train, y_train, label_map):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flat, y_train)
        self.label_map = label_map
    
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'label_map': self.label_map}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_map = data['label_map']
