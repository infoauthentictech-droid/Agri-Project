"""
FastAPI main application for Groundnut AI.
"""

from contextlib import asynccontextmanager
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.utils import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

#: Hard cap on the request body (bytes). Images are base64 and validated up to
#: 10 MiB *decoded*; base64 inflates binary by ~4/3, so a 10 MiB image encodes
#: to ~13.4 MiB of ASCII. A 16 MiB cap accepts every valid image the validator
#: permits while still rejecting degenerate payloads before JSON parsing.
MAX_BODY_BYTES = 16 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize services once on startup.

    The four services are created here a single time and handed to the
    endpoints through ``app.state.services`` (see ``app.api.endpoints``), so
    no endpoint can accidentally create a second, duplicated set. Heavy models
    are loaded lazily by the services; only the default detection model is
    pre-warmed below (and only when preloading is enabled).
    """
    from app.api.endpoints import build_services, warm_up_services

    logger.info("Starting Groundnut AI API...")

    # Build the shared service singletons (cheap: constructors no longer load
    # the heavy YOLO weights eagerly).
    app.state.services = build_services()

    # Explicitly pre-warm the default detection model + XAI engine so the first
    # user request does not pay the one-time YOLO load cost. Skipped when
    # GROUNDNUT_PRELOAD_MODELS=0 (e.g. tests or memory-constrained hosts).
    if os.environ.get("GROUNDNUT_PRELOAD_MODELS", "1") != "0":
        warm_up_services(app.state.services)

    model_service = app.state.services["model"]
    logger.info(f"Model loaded: {model_service.is_loaded()}")
    logger.info(f"Classifiers available: {app.state.services['classifier'].available_models}")
    logger.info(f"Recommendation engine loaded: {app.state.services['recommendation'].is_loaded()}")
    logger.info(f"XAI engine loaded: {app.state.services['xai'].is_loaded()}")

    yield

    logger.info("Shutting down Groundnut AI API...")


# Create FastAPI app. Docs are env-gated (they expose the API surface; disable
# in production with GROUNDNUT_DISABLE_DOCS=1).
import os

docs_enabled = os.environ.get("GROUNDNUT_DISABLE_DOCS", "0") != "1"

app = FastAPI(
    title="Groundnut AI API",
    description="API for groundnut disease detection, weed management, and herbicide recommendations",
    version="1.1.0",
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    lifespan=lifespan,
)

# CORS: mobile clients do not rely on CORS, but the Flutter *web* build calls
# the API from another origin (e.g. http://localhost:PORT), which browsers
# enforce. No credentials/cookies are used (only the X-API-Key header).
#
# The default is deliberately NOT "*": local web development on any
# localhost/127.0.0.1 port is allowed via a regex, but every other origin must
# be explicitly listed with GROUNDNUT_CORS_ORIGINS (comma-separated exact
# strings) or GROUNDNUT_CORS_ORIGIN_REGEX. This prevents a permissive default
# combined with the static mobile API key from being abused by arbitrary web
# origins.
cors_origins = [
    o.strip()
    for o in os.environ.get("GROUNDNUT_CORS_ORIGINS", "").split(",")
    if o.strip()
]
cors_origin_regex = os.environ.get(
    "GROUNDNUT_CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress JSON responses (images are small base64 blobs; gzip helps on slow
# mobile links without hurting latency meaningfully).
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def security_and_tracing(request: Request, call_next):
    """Per-request security headers, body-size guard, and request-ID tracing.

    Rate limiting is NOT done here: the /api/v1 router applies
    ``app.utils.security.rate_limit`` per client IP, so a second limiter here
    would only double-count requests (and skip CORS headers on its 429s).
    """
    # Reject oversized bodies early (before FastAPI parses JSON).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"success": False, "error": "Request body too large"},
                )
        except ValueError:
            pass  # malformed header -> let FastAPI reject it instead

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    # Security hardening headers (OWASP recommended).
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    return response


# Include API router (all /api/v1 routes require the X-API-Key header).
app.include_router(router, prefix="/api/v1")


@app.get("/healthz")
async def healthz():
    """Liveness probe for container platforms (Cloud Run / HF / Load balancers).

    Deliberately OUTSIDE the /api/v1 router so it needs no API key - these
    platforms probe the container before routing user traffic and cannot send
    the app's X-API-Key header.
    """
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler — no internal details leak to the client."""
    logger.error("Unhandled exception on %s %s",
                 request.method, request.url.path,
                 exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
