"""Custom timeout middleware for FastAPI"""
from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio


class CustomTimeoutMiddleware(BaseHTTPMiddleware):
    """Custom timeout middleware for requests"""

    def __init__(self, app, timeout_seconds: int = 60):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
            return response
        except asyncio.TimeoutError:
            return Response(
                content='{"error": "RequestTimeout", "message": "Request processing exceeded timeout limit"}',
                status_code=504,
                media_type="application/json",
            )
