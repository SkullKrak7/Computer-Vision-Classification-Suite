#!/bin/bash
# Post-deployment verification script

set -e

echo "🔍 Gold Standard Deployment Verification"
echo "========================================"

# Check Python version
echo "✓ Python version:"
python --version

# Check dependencies
echo ""
echo "✓ Checking critical dependencies..."
python -c "import fastapi, pydantic, prometheus_client, alembic; print('  - FastAPI, Pydantic, Prometheus, Alembic: OK')"

# Check app loads
echo ""
echo "✓ Checking app loads..."
PYTHONPATH=. python -c "from backend.app.main import app; from backend.app.config import settings; print(f'  - {settings.api_title} v{settings.api_version}: OK')"

# Check middleware
echo ""
echo "✓ Checking middleware..."
PYTHONPATH=. python -c "from backend.app.middleware import RateLimitMiddleware, SecurityHeadersMiddleware; print('  - Security middleware: OK')"

# Check validation
echo ""
echo "✓ Checking validation..."
PYTHONPATH=. python -c "from backend.app.utils.validation import validate_image_upload, sanitize_filename; print('  - Image validation: OK')"

# Check metrics
echo ""
echo "✓ Checking metrics..."
PYTHONPATH=. python -c "from backend.app.utils.metrics import metrics_endpoint; print('  - Prometheus metrics: OK')"

# Check config
echo ""
echo "✓ Checking configuration..."
PYTHONPATH=. python -c "from backend.app.config import settings; print(f'  - Rate limit: {settings.rate_limit_per_minute} req/min')"
PYTHONPATH=. python -c "from backend.app.config import settings; print(f'  - Max file size: {settings.max_file_size_mb}MB')"

# Run quick tests
echo ""
echo "✓ Running quick tests..."
PYTHONPATH=. pytest backend/tests/test_models.py -q --tb=no

echo ""
echo "========================================"
echo "✅ Verification Complete!"
echo ""
echo "Next steps:"
echo "  1. Start server: uvicorn backend.app.main:app --reload"
echo "  2. Check health: curl http://localhost:8000/health"
echo "  3. Check metrics: curl http://localhost:8000/metrics"
echo "  4. View docs: http://localhost:8000/docs"
