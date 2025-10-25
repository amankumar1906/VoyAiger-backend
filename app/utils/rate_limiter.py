"""Rate limiter for API endpoints - in-memory implementation"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List


class InMemoryRateLimiter:
    """
    In-memory rate limiter using IP address as key.
    Stores request timestamps and checks if limit exceeded.
    """

    def __init__(self, requests_per_hour: int = 2):
        """
        Initialize rate limiter.

        Args:
            requests_per_hour: Max requests allowed per IP per hour (default: 2)
        """
        self.requests_per_hour = requests_per_hour
        self.requests: Dict[str, List[datetime]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """
        Check if request from client IP is allowed.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Remove requests older than 1 hour
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] if req_time > cutoff
        ]

        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.requests_per_hour:
            return False

        # Record this request
        self.requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """
        Get remaining requests for client IP in current hour.

        Args:
            client_ip: Client IP address

        Returns:
            Number of remaining requests
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Count recent requests
        recent = [
            req_time for req_time in self.requests[client_ip] if req_time > cutoff
        ]
        return max(0, self.requests_per_hour - len(recent))
