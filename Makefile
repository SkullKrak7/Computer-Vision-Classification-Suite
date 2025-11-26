.PHONY: setup train test benchmark clean

setup:
	@bash scripts/setup.sh

train:
	@bash scripts/train_all.sh

benchmark:
	@bash scripts/benchmark.sh

export:
	@bash scripts/export_onnx.sh

test:
	@source venv/bin/activate && python python/test_models.py

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"
