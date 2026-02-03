#!/bin/bash
# Rollback to previous version

set -e

REVISION=${1:-"-1"}

echo "Rolling back to revision $REVISION..."

# 1. Stop service
echo "1. Stopping service..."
pkill -f "uvicorn backend.app.main:app" || true

# 2. Rollback database
echo "2. Rolling back database..."
PYTHONPATH=. alembic downgrade $REVISION

# 3. Rollback code
if [ "$REVISION" != "-1" ]; then
    echo "3. Rolling back code to commit..."
    git reset --hard HEAD~1
fi

# 4. Reinstall dependencies
echo "4. Reinstalling dependencies..."
pip install -r requirements.txt

# 5. Restart service
echo "5. Restarting service..."
nohup uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &

# 6. Health check
echo "6. Checking health..."
sleep 5
curl -f http://localhost:8000/health || (echo "Rollback failed!" && exit 1)

echo ""
echo "Rollback complete!"
