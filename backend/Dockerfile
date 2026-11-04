# =============================================================================
# Groundnut AI Backend — Dockerfile for Render
# =============================================================================
# Build: docker build -t groundnut-ai-backend ./backend
# Run : docker run -p 8000:8000 -e GROUNDNUT_API_KEYS=your-key groundnut-ai-backend
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies (needed by opencv-python-headless, pillow, numpy)
# libgl1 + libxcb1 needed by cv2 (opencv) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add CPU-only torch/torchvision (Render has no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy backend source
COPY backend/app ./app
COPY backend/run_server.py ./

# Copy ml package (needed by classifier_service.py)
COPY ml ./ml

# Copy model weights (kept small; tracked in git via !*.pt un-ignore)
COPY *.pt ./

# Port Render injects via $PORT
ENV PORT=8000
ENV GROUNDNUT_PRELOAD_MODELS=1
ENV GROUNDNUT_DISABLE_DOCS=0

EXPOSE $PORT

# Production entrypoint: read API key + port from environment
CMD sh -c "python run_server.py \"${GROUNDNUT_API_KEY:-dev-local-key}\" \"${PORT:-8000}\""