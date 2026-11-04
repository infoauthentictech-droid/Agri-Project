"""API Rate Limiting and Access Control

Implements rate limiting, access control, and abuse protection for the FastAPI backend.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class RateLimiter:
    """In-memory rate limiter using token bucket algorithm."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self.tokens: Dict[str, float] = defaultdict(lambda: self.config.burst_size)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> tuple[bool, Optional[str]]:
        """Check if request is allowed.

        Args:
            identifier: Client identifier (IP, API key, etc.)

        Returns:
            Tuple of (allowed, error_message)
        """
        with self.lock:
            now = time.time()
            last_time = self.last_update[identifier]

            # Replenish tokens based on time elapsed
            elapsed = now - last_time
            replenished = elapsed * (self.config.requests_per_minute / 60.0)
            self.tokens[identifier] = min(
                self.config.burst_size,
                self.tokens[identifier] + replenished
            )
            self.last_update[identifier] = now

            # Check if token available
            if self.tokens[identifier] >= 1:
                self.tokens[identifier] -= 1
                return True, None
            else:
                return False, f"Rate limit exceeded for {identifier}"

    def reset(self, identifier: str) -> None:
        """Reset rate limit for a client.

        Args:
            identifier: Client identifier
        """
        with self.lock:
            self.tokens[identifier] = self.config.burst_size
            self.last_update[identifier] = time.time()


class APIKeyAuth(APIKeyHeader):
    """API key authentication with validation."""

    def __init__(self, name: str = "X-API-Key", auto_error: bool = True):
        """Initialize API key auth.

        Args:
            name: Header name
            auto_error: Whether to auto-raise errors
        """
        super().__init__(name=name, auto_error=auto_error)
        self.valid_keys = self._load_valid_keys()

    def _load_valid_keys(self) -> set:
        """Load valid API keys from environment or config.

        Returns:
            Set of valid API keys
        """
        import os

        keys = os.environ.get("GROUNDNUT_API_KEYS", "").split(",")
        return {k.strip() for k in keys if k.strip()}

    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key.

        Args:
            request: FastAPI request

        Returns:
            API key if valid, None otherwise

        Raises:
            HTTPException if key is invalid
        """
        api_key = await super().__call__(request)

        if api_key not in self.valid_keys:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return api_key


class AccessControl:
    """Access control and rate limiting middleware."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        require_api_key: bool = False,
    ):
        """Initialize access control.

        Args:
            rate_limiter: Rate limiter instance
            require_api_key: Whether to require API key authentication
        """
        self.rate_limiter = rate_limiter or RateLimiter()
        self.require_api_key = require_api_key
        self.api_key_auth = APIKeyAuth() if require_api_key else None

    def check_rate_limit(self, request: Request) -> tuple[bool, Optional[str]]:
        """Check rate limit for request.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (allowed, error_message)
        """
        # Key on the client IP only: the X-API-Key header is client-chosen and
        # unvalidated here, so keying on it would let a caller mint a fresh
        # bucket per request.
        identifier = request.client.host if request.client else "unknown"
        return self.rate_limiter.is_allowed(identifier)

    def log_request(self, request: Request, response_status: int) -> None:
        """Log request for security monitoring.

        Args:
            request: FastAPI request
            response_status: HTTP status code
        """
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host} "
            f"status={response_status}"
        )


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_access_control: Optional[AccessControl] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter.

    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_access_control() -> AccessControl:
    """Get the global access control instance.

    Returns:
        AccessControl instance
    """
    global _access_control
    if _access_control is None:
        import os

        require_api_key = os.environ.get("GROUNDNUT_REQUIRE_API_KEY", "false").lower() == "true"
        _access_control = AccessControl(require_api_key=require_api_key)
    return _access_control
