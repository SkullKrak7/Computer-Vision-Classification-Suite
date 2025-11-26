# Kaggle Dataset Testing Guide

## Overview

This guide explains how to test the Computer Vision Classification Suite with real Kaggle datasets, including automatic bias-variance analysis and model tuning.

## Prerequisites

### Kaggle API Setup

Your Kaggle API is already configured. If you need to reconfigure:

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

### Verify Authentication

```bash
python -c "import kaggle; kaggle.api.authenticate(); print('Authenticated')"
```

## Available Datasets

### 1. Intel Image Classification (ALREADY DOWNLOADED)
- **Classes**: 6 (Buildings, Forest, Glacier, Mountain, Sea, Street)
- **Images**: ~14,000
- **Location**: `datasets/intel_images/`
- **Kaggle ID**: `puneet6060/intel-image-classification`

### 2. Cats vs Dogs
- **Classes**: 2 (Cat, Dog)
- **Images**: ~25,000
- **Kaggle ID**: `tongpython/cat-and-dog`
- **Auto-download**: Yes

### 3. Flowers Recognition
- **Classes**: 5 (Daisy, Dandelion, Rose, Sunflower, Tulip)
- **Images**: ~4,000
- **Kaggle ID**: `alxmamaev/flowers-recognition`
- **Auto-download**: Yes

## Running Tests

### Interactive Mode

```bash
python python/scripts/download_and_test.py
```

The script will:
1. Show available datasets
2. Download if needed (with auth error handling)
3. Let you choose which models to test
4. Provide real-time analysis

### What Gets Tested

#### PyTorch CNN
- **Architecture**: 3 conv layers + 2 FC layers
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 10
- **Optimization**: AMP (Automatic Mixed Precision)
- **Regularization**: Dropout 0.3

#### TensorFlow MobileNetV2
- **Architecture**: Transfer learning from ImageNet
- **Learning Rate**: 0.0001 (lower for fine-tuning)
- **Batch Size**: 16
- **Epochs**: 10
- **Optimization**: Mixed precision (FP16)
- **Fine-tuning**: Optional

## Bias-Variance Analysis

The script automatically analyzes model performance:

### Overfitting Detection
**Condition**: Train accuracy - Val accuracy > 0.15

**Symptoms**:
- High training accuracy
- Low validation accuracy
- Large generalization gap

**Recommendations**:
- Increase dropout rate (0.3 -> 0.5)
- Add data augmentation
- Reduce model complexity
- Add L2 regularization
- Use early stopping

### Underfitting Detection
**Condition**: Validation accuracy < 0.6

**Symptoms**:
- Low training accuracy
- Low validation accuracy
- Model not learning patterns

**Recommendations**:
- Train for more epochs
- Increase model capacity
- Reduce regularization
- Lower learning rate
- Check data quality

### Good Tradeoff
**Condition**: Gap < 0.15 and Val accuracy > 0.6

**Indicators**:
- Similar train/val accuracy
- Reasonable performance
- Good generalization

## Model Tuning Parameters

### If Overfitting

**PyTorch**:
```python
model = PyTorchCNNClassifier(
    num_classes=num_classes,
    learning_rate=0.0005,  # Reduce LR
    use_amp=True
)
# Increase dropout in model architecture
# Add data augmentation
```

**TensorFlow**:
```python
model = TFMobileNetClassifier(
    num_classes=num_classes,
    learning_rate=0.00005,  # Reduce LR
    fine_tune=False,  # Don't fine-tune base
    use_mixed_precision=True
)
```

### If Underfitting

**PyTorch**:
```python
model = PyTorchCNNClassifier(
    num_classes=num_classes,
    learning_rate=0.001,
    use_amp=True
)
# Train for more epochs (20-30)
# Reduce dropout (0.3 -> 0.1)
```

**TensorFlow**:
```python
model = TFMobileNetClassifier(
    num_classes=num_classes,
    learning_rate=0.0001,
    fine_tune=True,  # Enable fine-tuning
    use_mixed_precision=True
)
# Train for more epochs
```

## Output

### Saved Models

**PyTorch**: `models/pytorch/tested_model.pth`
**TensorFlow**: `models/tensorflow/tested_model.keras`

### Metrics Reported

- Train Accuracy
- Validation Accuracy
- Generalization Gap
- Precision, Recall, F1-score
- Confusion Matrix

### Example Output

```
======================================================================
TESTING PYTORCH CNN
======================================================================
Configuration: {'num_classes': 6, 'learning_rate': 0.001, 'use_amp': True}

Training...
Epoch 1/10, Loss: 1.7234
Epoch 2/10, Loss: 1.2156
...

Evaluating...

RESULTS:
Train Accuracy: 0.8750
Val Accuracy: 0.8200
Generalization Gap: 0.0550

ANALYSIS: Good bias-variance tradeoff

Model saved to models/pytorch/tested_model.pth
```

## Troubleshooting

### Kaggle Authentication Failed

```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# If not, download from Kaggle and place it
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Dataset Download Failed

1. Visit the dataset page on Kaggle
2. Accept the dataset's terms and conditions
3. Try downloading again

### Out of Memory

Reduce sample size in the script:
```python
sample_size = 500  # Reduce to 200 or 100
```

Or reduce batch size:
```python
batch_size = 16  # Reduce to 8 or 4
```

### Model Not Learning

- Check data normalization
- Verify label encoding
- Ensure sufficient training data
- Try different learning rates

## Advanced Usage

### Custom Dataset

To test with your own dataset:

1. Organize as: `dataset_name/class1/`, `dataset_name/class2/`, etc.
2. Modify script to add your dataset
3. Run testing

### Hyperparameter Search

For systematic tuning, use the script as a template and implement grid search:

```python
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        # Train and evaluate
        # Track best configuration
```

## Best Practices

1. **Always use validation set** for hyperparameter tuning
2. **Test set only once** for final evaluation
3. **Monitor both metrics** (train and val) during training
4. **Save best model** based on validation performance
5. **Document experiments** with different configurations
6. **Use early stopping** to prevent overfitting
7. **Apply data augmentation** for better generalization

## Next Steps

After testing:

1. Export best model to ONNX
2. Test C++ inference
3. Deploy via FastAPI backend
4. Monitor production performance
5. Iterate based on real-world results

## Support

If you encounter issues:
1. Check this guide
2. Review error messages
3. Verify Kaggle authentication
4. Check dataset availability
5. Ensure sufficient system resources (RAM, GPU memory)
