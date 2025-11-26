"""Comprehensive model tests with OOP"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import tempfile

import numpy as np


class ModelTester:
    """Test harness for models"""

    def __init__(self, model_class, model_name):
        self.model_class = model_class
        self.model_name = model_name

    def generate_data(self, n_samples=10, n_classes=3):
        """Generate test data"""
        X = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        label_map = {i: f"class_{i}" for i in range(n_classes)}
        return X, y, label_map

    def test_all(self, **kwargs):
        """Run all tests"""
        print(f"\nTesting {self.model_name}:")

        X, y, label_map = self.generate_data()
        model = self.model_class(num_classes=3, **kwargs)

        # Train
        model.train(X, y, label_map, epochs=1, batch_size=4)
        print("  [PASS] Training")

        # Predict
        X_test = np.random.rand(2, 224, 224, 3).astype(np.float32)
        preds = model.predict(X_test)
        assert len(preds) == 2
        print("  [PASS] Prediction")

        # Save/Load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)
            loaded = self.model_class.load(save_path)
            preds2 = loaded.predict(X_test)
            assert np.array_equal(preds, preds2)
        print("  [PASS] Save/Load")

        return True


if __name__ == "__main__":
    print("=" * 70)
    print("MODEL TESTING SUITE")
    print("=" * 70)

    from models.deep_learning import PyTorchCNNClassifier, TFMobileNetClassifier

    try:
        ModelTester(PyTorchCNNClassifier, "PyTorch CNN").test_all(use_amp=True)
        ModelTester(TFMobileNetClassifier, "TensorFlow MobileNet").test_all(
            use_mixed_precision=True
        )
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
