.PHONY: setup train test benchmark clean format lint lint-fix ci install-hooks

VENV = venv/bin/activate

setup:
	@bash scripts/setup.sh

train:
	@bash scripts/train_all.sh

benchmark:
	@bash scripts/benchmark.sh

export:
	@bash scripts/export_onnx.sh

test:
	@. $(VENV) && pytest python/tests/ backend/tests/ -v

format:
	@. $(VENV) && black python/ backend/
	@echo "✓ Formatted with Black"

lint:
	@. $(VENV) && ruff check python/ backend/
	@echo "✓ Linted with Ruff"

lint-fix:
	@. $(VENV) && ruff check --fix python/ backend/
	@echo "✓ Fixed with Ruff"

ci: format lint test
	@echo "✓ CI checks passed"

install-hooks:
	@. $(VENV) && pre-commit install
	@echo "✓ Pre-commit hooks installed"

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache .coverage coverage.xml .ruff_cache
	@echo "✓ Cleaned"
