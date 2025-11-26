"""Test dataset loading"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data import DatasetLoader, DataConfig, augment_image

def test_augmentation():
    """Test image augmentation"""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    augmented = augment_image(img)
    assert augmented.shape == img.shape
    print("✓ Augmentation test passed")

def test_config():
    """Test data config"""
    config = DataConfig(img_size=(128, 128), test_size=0.3)
    assert config.img_size == (128, 128)
    assert config.test_size == 0.3
    print("✓ Config test passed")

if __name__ == '__main__':
    test_augmentation()
    test_config()
    print("\n✓ All dataset tests passed!")
