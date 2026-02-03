# Troubleshooting

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
training:
  batch_size: 16  # Instead of 32

# Disable mixed precision
gpu:
  mixed_precision: false

# Check GPU memory
nvidia-smi
```

### Docker Build Slow
First build: ~18 minutes (installs torch/tensorflow)
Subsequent builds: <2 minutes (cached)

```bash
# Build once
docker-compose build

# Fast restarts
docker-compose up
```

### Tests Failing
```bash
# Clean environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run with PYTHONPATH
PYTHONPATH=. pytest backend/tests/
```

### Import Errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use make
make test
```

### Port Already in Use
```bash
# Check port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn backend.app.main:app --port 8001
```

### Model Not Found
```bash
# Train models first
python python/scripts/auto_tune.py

# Check models exist
ls -la models/
```

## Performance Issues

### Slow Training
- Enable GPU: `nvidia-smi` should show GPU
- Enable mixed precision: `use_amp=True`
- Increase batch size if memory allows
- Reduce data augmentation

### Slow Inference
- Use C++ ONNX engine
- Batch predictions
- Enable GPU
- Use smaller models (MobileNet)

### High Memory Usage
- Reduce batch size
- Clear cache: `torch.cuda.empty_cache()`
- Use gradient checkpointing
- Monitor: `nvidia-smi`

## Debugging

### Enable Debug Logging
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env
LOG_LEVEL=DEBUG
```

### Check Logs
```bash
# Docker logs
docker-compose logs -f backend

# Filter errors
docker-compose logs backend | grep ERROR

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Health Check
```bash
# Check service health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```

### Database Issues
```bash
# Check migrations
alembic current

# Run migrations
alembic upgrade head

# Reset database
rm cv_classification.db
alembic upgrade head
```

## Getting Help

1. Check logs first
2. Search GitHub Issues
3. Enable debug logging
4. Run health checks
5. [Open an issue](https://github.com/SkullKrak7/Computer-Vision-Classification-Suite/issues)

## Quick Fixes

```bash
# Reset everything
docker-compose down -v
docker-compose up --build

# Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```
