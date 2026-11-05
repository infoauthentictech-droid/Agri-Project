# =============================================================================
# Groundnut AI Backend — Dockerfile for Hugging Face Spaces
# =============================================================================
# HF Spaces exposes port 7860. Build with:
#   docker build -t groundnut-ai-backend ./deploy
# Run locally:
#   docker run -p 7860:7860 -e GROUNDNUT_API_KEYS=your-key groundnut-ai-backend
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by opencv-python-headless, torch, numpy
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

# CPU-only torch/torchvision (HF Spaces has no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy backend source
COPY backend/app ./app
COPY backend/run_server.py ./

# Copy ml package (needed by classifier_service.py)
COPY ml ./ml

# Copy herbicide package (needed by recommendation_service.py)
COPY herbicide ./herbicide

# Copy model weights (tracked in git via .gitignore un-ignore)
COPY *.pt ./

# HF Spaces uses port 7860
ENV PORT=7860
ENV GROUNDNUT_PRELOAD_MODELS=0
ENV GROUNDNUT_DISABLE_DOCS=1

EXPOSE 7860

# Production entrypoint
CMD sh -c "python run_server.py \"${GROUNDNUT_API_KEY:-dev-local-key}\" \"${PORT:-7860}\""