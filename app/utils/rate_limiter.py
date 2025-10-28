"""Rate limiter for API endpoints - in-memory implementation"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List


class InMemoryRateLimiter:
    """
    In-memory rate limiter with per-IP and global limits.
    Prevents DDoS attacks using different IPs by enforcing a global request cap.
    """

    def __init__(self, requests_per_hour: int = 2, global_requests_per_minute: int = 10):
        """
        Initialize rate limiter.

        Args:
            requests_per_hour: Max requests allowed per IP per hour (default: 2)
            global_requests_per_minute: Max total requests globally per minute (default: 10)
        """
        self.requests_per_hour = requests_per_hour
        self.global_requests_per_minute = global_requests_per_minute
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.global_requests: List[datetime] = []

    def is_allowed(self, client_ip: str) -> bool:
        """
        Check if request from client IP is allowed.
        Enforces both per-IP limit and global limit.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        now = datetime.now()
        cutoff_hour = now - timedelta(hours=1)
        cutoff_minute = now - timedelta(minutes=1)

        # Clean up old per-IP requests (older than 1 hour)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] if req_time > cutoff_hour
        ]

        # Check per-IP limit
        if len(self.requests[client_ip]) >= self.requests_per_hour:
            return False

        # Clean up old global requests (older than 1 minute)
        self.global_requests = [
            req_time for req_time in self.global_requests if req_time > cutoff_minute
        ]

        # Check global limit (prevent DDoS with different IPs)
        if len(self.global_requests) >= self.global_requests_per_minute:
            return False

        # Record this request on both per-IP and global trackers
        self.requests[client_ip].append(now)
        self.global_requests.append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """
        Get remaining requests for client IP in current hour.
        Returns the minimum of per-IP and global remaining requests.

        Args:
            client_ip: Client IP address

        Returns:
            Number of remaining requests (limited by both per-IP and global caps)
        """
        now = datetime.now()
        cutoff_hour = now - timedelta(hours=1)
        cutoff_minute = now - timedelta(minutes=1)

        # Count recent per-IP requests
        recent_ip = [
            req_time for req_time in self.requests[client_ip] if req_time > cutoff_hour
        ]
        per_ip_remaining = max(0, self.requests_per_hour - len(recent_ip))

        # Count recent global requests
        recent_global = [
            req_time for req_time in self.global_requests if req_time > cutoff_minute
        ]
        global_remaining = max(0, self.global_requests_per_minute - len(recent_global))

        # Return the limiting factor
        return min(per_ip_remaining, global_remaining)
