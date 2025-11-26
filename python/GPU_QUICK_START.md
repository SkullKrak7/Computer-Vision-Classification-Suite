# GPU Quick Start - RTX 3060

## Verify GPU is Working

```bash
python -c "import torch; print(f'PyTorch GPU: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

## PyTorch - GPU Accelerated

```python
from models.deep_learning import PyTorchCNNClassifier

# GPU + Mixed Precision (Recommended)
model = PyTorchCNNClassifier(num_classes=10, use_amp=True)
model.train(X_train, y_train, label_map, epochs=20, batch_size=32)
predictions = model.predict(X_test)
```

## TensorFlow - GPU Accelerated

```python
from models.deep_learning import TFMobileNetClassifier

# GPU + Mixed Precision (Recommended)
model = TFMobileNetClassifier(num_classes=10, use_mixed_precision=True)
model.train(X_train, y_train, label_map, epochs=20, batch_size=16)
predictions = model.predict(X_test, batch_size=32)
```

## Monitor GPU

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or in Python
import torch
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

## Benchmark Performance

```bash
python benchmark_gpu.py
```

## Optimal Settings for RTX 3060

| Setting          | PyTorch | TensorFlow |
|------------------|---------|------------|
| Batch Size       | 32-64   | 16-32      |
| Mixed Precision  |        |           |
| Workers          | 2-4     | N/A        |

## Troubleshooting

**Out of Memory?** → Reduce batch size
**Slow inference?** → Increase batch size or warmup model
**GPU not used?** → Check `torch.cuda.is_available()` or `tf.config.list_physical_devices('GPU')`

See `GPU_OPTIMIZATION.md` for detailed guide.
