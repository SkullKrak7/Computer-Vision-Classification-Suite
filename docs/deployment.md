# Deployment Guide

## Local Development

### Python Backend
```bash
source venv/bin/activate
uvicorn backend.app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Docker Deployment

### Build and Run All Services
```bash
docker-compose up --build
```

Services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Individual Services

**Backend Only:**
```bash
docker-compose up backend
```

**Frontend Only:**
```bash
docker-compose up frontend
```

**Training:**
```bash
docker-compose run python python python/scripts/train_cnn.py
```

## Production Deployment

### Environment Variables
```bash
export PYTHONUNBUFFERED=1
export NODE_ENV=production
```

### Build Production Frontend
```bash
cd frontend
npm run build
```

### Serve with Nginx
```nginx
server {
    listen 80;
    
    location / {
        root /app/frontend/dist;
        try_files $uri /index.html;
    }
    
    location /api {
        proxy_pass http://backend:8000;
    }
}
```

## GPU Support in Docker

Add to docker-compose.yml:
```yaml
services:
  python:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Health Checks

```bash
# Backend
curl http://localhost:8000/

# Frontend
curl http://localhost:3000/
```

## Monitoring

Use docker logs:
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Scaling

```bash
docker-compose up --scale backend=3
```
