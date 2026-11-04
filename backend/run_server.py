"""Start the Groundnut AI backend.

Usage (from the backend/ directory):
    python run_server.py [api_key] [port]

If arguments are omitted, values are read from environment variables:
    GROUNDNUT_API_KEYS   (comma-separated valid keys; first one is used)
    PORT                 (default 8000)

Render sets PORT automatically; set GROUNDNUT_API_KEYS in the Render dashboard.
"""
import os
import sys


def main() -> int:
    # API key: prefer explicit arg, else env var, else a dev default
    if len(sys.argv) > 1:
        key = sys.argv[1]
    else:
        keys = os.environ.get("GROUNDNUT_API_KEYS", "")
        key = keys.split(",")[0].strip() if keys.strip() else "dev-local-key"

    os.environ["GROUNDNUT_API_KEY"] = key
    # Also expose the plural form the auth layer reads
    if keys.strip():
        os.environ["GROUNDNUT_API_KEYS"] = keys

    # Port: explicit arg, else $PORT (Render), else 8000
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = int(os.environ.get("PORT", "8000"))

    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())