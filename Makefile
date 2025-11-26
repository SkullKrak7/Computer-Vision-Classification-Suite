.PHONY: setup train test benchmark clean format lint lint-fix ci install-hooks

setup:
	@bash scripts/setup.sh

train:
	@bash scripts/train_all.sh

benchmark:
	@bash scripts/benchmark.sh

export:
	@bash scripts/export_onnx.sh

test:
	@pytest python/tests/ backend/tests/ -v

format:
	@black python/ backend/
	@echo "✓ Formatted with Black"

lint:
	@ruff check python/ backend/
	@echo "✓ Linted with Ruff"

lint-fix:
	@ruff check --fix python/ backend/
	@echo "✓ Fixed with Ruff"

ci: format lint test
	@echo "✓ CI checks passed"

install-hooks:
	@pre-commit install
	@echo "✓ Pre-commit hooks installed"

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache .coverage coverage.xml .ruff_cache
	@echo "✓ Cleaned"
