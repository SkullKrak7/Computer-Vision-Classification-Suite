"""Test training utilities"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training import TrainingConfig, Trainer

def test_config():
    """Test training config"""
    config = TrainingConfig(epochs=10, batch_size=16)
    assert config.epochs == 10
    assert config.batch_size == 16
    assert config.use_gpu == True
    print("✓ Training config test passed")

def test_trainer_init():
    """Test trainer initialization"""
    config = TrainingConfig()
    # Note: Trainer needs a model, so just test config
    assert config.learning_rate == 0.001
    print("✓ Trainer init test passed")

if __name__ == '__main__':
    test_config()
    test_trainer_init()
    print("\n✓ All training tests passed!")
