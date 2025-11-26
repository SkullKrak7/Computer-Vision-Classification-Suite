"""Training configuration"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    use_gpu: bool = True
    use_mixed_precision: bool = True
