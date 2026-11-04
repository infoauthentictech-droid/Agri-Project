"""Lightweight production-appropriate security helpers.

- API-key authentication (constant-time compare) via ``X-API-Key`` header.
- In-memory fixed-window rate limiting per client IP (bounded, worker-local).
- The API key is read from the ``GROUNDNUT_API_KEY`` environment variable;
  when unset a random ephemeral dev key is generated and logged once (the
  service is never deployed with a blank key by accident).

Rate limiting notes:
- ``X-Forwarded-For`` is only trusted when ``GROUNDNUT_TRUSTED_PROXY=1`` is
  set. Do NOT run this service directly behind a reverse proxy without
  configuring that flag, otherwise any client could spoof its IP.
- The store is in-memory and therefore per-worker/per-process. For a
  multi-worker or distributed deployment use a shared limiter (e.g. Redis);
  this module documents that limitation rather than pretending to be
  globally consistent.

No external dependencies are required beyond the FastAPI stack.
"""
from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time
from collections import deque
from typing import Optional

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

#: Ephemeral dev key when env var is missing (never blank).
API_KEY = os.environ.get("GROUNDNUT_API_KEY") or f"dev-{secrets.token_hex(16)}"
if not os.environ.get("GROUNDNUT_API_KEY"):
    logger.warning("GROUNDNUT_API_KEY unset; using ephemeral dev key %s...",
                   API_KEY[:12])

RATE_LIMIT_PER_MINUTE = int(os.environ.get("GROUNDNUT_RATE_LIMIT", "60"))

#: Only trust X-Forwarded-For when the deployment is behind a trusted proxy
#: that overwrites/validates it (e.g. nginx/traefik/AWS ALB). Direct clients
#: must never be able to choose their own rate-limit identity.
TRUST_PROXY = os.environ.get("GROUNDNUT_TRUSTED_PROXY", "0") == "1"

#: Upper bound on tracked client keys so a hostile flood of spoofed IPs can
#: not grow the in-memory store without limit. Oldest entries are evicted.
MAX_RATE_LIMIT_KEYS = int(os.environ.get("GROUNDNUT_RATE_LIMIT_KEYS", "10000"))

_guard = threading.Lock()
_store: dict = {}
_last_insert_ts: dict = {}


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            # RFC 7239: the right-most entry is the client as appended by the
            # trusted proxies; when a single proxy is used it is the second
            # entry. We take the first entry only when explicitly trusted.
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Reject requests without a valid API key (constant-time compare)."""
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")


def rate_limit(request: Request) -> None:
    """Reject requests exceeding the per-minute budget for a client identity."""
    global _store, _last_insert_ts  # noqa: PLW0603
    now = time.time()
    key = _client_ip(request)
    with _guard:
        window = _store.setdefault(key, deque())
        _last_insert_ts[key] = now
        while window and window[0] < now - 60.0:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded")
        window.append(now)
        # Bound the store: evict the least-recently-inserted keys when the
        # tracked-client count exceeds the configured cap.
        if len(_store) > MAX_RATE_LIMIT_KEYS:
            oldest = sorted(_last_insert_ts, key=_last_insert_ts.get)[: len(_store) - MAX_RATE_LIMIT_KEYS]
            for k in oldest:
                _store.pop(k, None)
                _last_insert_ts.pop(k, None)


def api_key() -> str:
    """Return the configured key (test helper)."""
    return API_KEY