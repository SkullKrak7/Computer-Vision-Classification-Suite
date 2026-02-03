#!/bin/bash
# Production deployment script

set -e

APP_NAME="cv-classification"
VERSION=${1:-latest}
PORT=${2:-8000}

echo "Deploying $APP_NAME version $VERSION..."

# 1. Pull latest code
echo "1. Pulling latest code..."
git pull origin main

# 2. Install dependencies
echo "2. Installing dependencies..."
pip install -r requirements.txt

# 3. Run migrations
echo "3. Running database migrations..."
PYTHONPATH=. alembic upgrade head

# 4. Run tests
echo "4. Running tests..."
PYTHONPATH=. pytest backend/tests/ -q --cov=backend/app --cov-fail-under=50

# 5. Stop existing service
echo "5. Stopping existing service..."
pkill -f "uvicorn backend.app.main:app" || true

# 6. Start service
echo "6. Starting service..."
nohup uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT > app.log 2>&1 &

# 7. Wait for health check
echo "7. Waiting for health check..."
sleep 5
curl -f http://localhost:$PORT/health || (echo "Health check failed!" && exit 1)

echo ""
echo "Deployment complete!"
echo "Service running on port $PORT"
echo "Logs: tail -f app.log"
