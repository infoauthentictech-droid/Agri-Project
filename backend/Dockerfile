# =============================================================================
# Groundnut AI Backend — Dockerfile for Render
# =============================================================================
# Build context is the REPOSITORY ROOT (the backend imports the top-level `ml`
# and `herbicide` packages and reads model weights from `experiments/`).
#
#   docker build -f backend/Dockerfile -t groundnut-ai-backend .
#   docker run -p 8000:8000 -e GROUNDNUT_API_KEYS=your-key groundnut-ai-backend
#
# The container mirrors the repo layout (/app/backend, /app/ml, /app/experiments,
# ...) because the services resolve the project root as `parents[3]` of their
# own file — flattening backend/ into /app would break every model path.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries needed by opencv (cv2) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# CPU-only torch/torchvision FIRST (Render has no GPU). If ultralytics were
# installed first it would pull the multi-GB CUDA build of torch from PyPI, and
# the later CPU install would be skipped as "already satisfied".
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Source packages
COPY backend/app ./backend/app
COPY backend/run_server.py ./backend/
COPY ml ./ml
COPY herbicide ./herbicide
COPY configs ./configs

# Inference weights (only the files whitelisted in .gitignore are in the repo)
COPY experiments ./experiments
COPY yolov8n.pt yolov8n-seg.pt ./

WORKDIR /app/backend

# Render injects $PORT; run_server.py reads PORT and GROUNDNUT_API_KEYS from env.
ENV PORT=8000 \
    GROUNDNUT_PRELOAD_MODELS=1 \
    GROUNDNUT_DISABLE_DOCS=0

EXPOSE 8000

CMD ["python", "run_server.py"]
