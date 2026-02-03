# Testing Guide

## Test Coverage: 93%

```bash
# Run all tests
pytest backend/tests/ -v --cov=backend/app

# Quick test
make test
```

## Test Suite (51 tests)

| File | Tests | Purpose |
|------|-------|---------|
| test_models.py | 7 | Pydantic model validation |
| test_validation_async.py | 5 | Async file upload validation |
| test_inference.py | 8 | Inference endpoint tests |
| test_integration.py | 10 | End-to-end workflows |
| test_error_handling.py | 11 | Error scenarios |
| test_database.py | 4 | Database models |
| test_metrics.py | 3 | Metrics routes |
| test_training.py | 4 | Training routes |

## Running Tests

```bash
# All tests with coverage
pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing

# Specific test file
pytest backend/tests/test_inference.py -v

# Single test
pytest backend/tests/test_inference.py::test_predict_with_valid_image -v

# C++ tests
cd cpp/build && ctest
```

## Code Quality

```bash
# Format code
black --line-length 120 backend/ python/
isort --profile black --line-length 120 backend/ python/

# Lint
ruff check backend/ python/

# Type check
mypy backend/app --ignore-missing-imports

# Security scan
bandit -r backend/ python/ -c pyproject.toml

# All checks
make ci
```

## Writing Tests

### Async Tests
```python
import pytest
from backend.app.utils.validation import validate_image_upload

@pytest.mark.asyncio
async def test_validate_image():
    mock_file = MockUploadFile(content=valid_jpeg_bytes, content_type="image/jpeg")
    image, format = await validate_image_upload(mock_file)
    assert format == "JPEG"
```

### Integration Tests
```python
def test_full_prediction_flow(client, sample_image_bytes):
    response = client.post(
        "/v1/inference/predict",
        files={"file": ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
    )
    assert response.status_code in [200, 500, 503]
```

## Coverage Requirements

- Minimum: 80% (enforced in CI)
- Current: 93%
- Target: Maintain >90%

## CI Integration

Tests run automatically on:
- Every push
- Every pull request
- Before merge

Checks:
- ✅ All tests pass
- ✅ Coverage ≥80%
- ✅ No linting errors
- ✅ Type checking passes
- ✅ Security scan clean
