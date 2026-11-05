"""Start the Groundnut AI backend with an explicit API key.

Usage (from the backend/ directory):
    python run_server.py dev-my-secret-key [port]

Sets the GROUNDNUT_API_KEY inside this process (before importing the app) so the
key can never fail to propagate via shell/environment inheritance.
"""
import os
import sys


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "dev-local-key"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    os.environ["GROUNDNUT_API_KEY"] = key

    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())