"""Security middleware for FastAPI"""

import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware - 100 requests per minute per IP"""

    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[datetime]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request"""
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests (older than 1 minute)
        now = datetime.now()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] if now - req_time < timedelta(minutes=1)
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                headers={"Retry-After": "60"},
            )

        # Add current request
        self.requests[client_ip].append(now)

        response = await call_next(request)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID header"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers"""
        response = await call_next(request)

        # Security headers per OWASP recommendations
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing header"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Measure and add request processing time"""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
