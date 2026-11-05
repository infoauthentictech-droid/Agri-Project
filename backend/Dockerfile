# =============================================================================
# Groundnut AI Backend — Dockerfile for Google Cloud Run
# =============================================================================
# Build locally:
#   docker build -t gcr.io/YOUR_PROJECT_ID/groundnut-ai-backend ./backend
#
# Deploy to Cloud Run:
#   gcloud run deploy groundnut-ai-backend \
#     --image gcr.io/YOUR_PROJECT_ID/groundnut-ai-backend \
#     --platform managed \
#     --region us-central1 \
#     --allow-unauthenticated \
#     --set-env-vars GROUNDNUT_API_KEYS=your-secret-key
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by opencv-python, torch, numpy
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

# CPU-only torch/torchvision (Cloud Run has no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy backend source
COPY backend/app ./app
COPY backend/run_server.py ./

# Copy ml package (needed by classifier_service.py)
COPY ml ./ml

# Copy herbicide package (needed by recommendation_service.py)
COPY herbicide ./herbicide

# Copy configs (needed by model loading)
COPY configs ./configs

# Copy model weights
COPY yolov8n.pt ./
COPY yolov8n-seg.pt ./

# Copy experiment weights if they exist (for production inference)
COPY experiments/yolo/agri_yolov8n/weights/best.pt ./experiments/yolo/agri_yolov8n/weights/best.pt
COPY experiments/yoloseg/agri_yolov8n_seg/weights/best.pt ./experiments/yoloseg/agri_yolov8n_seg/weights/best.pt
COPY experiments/by_dataset/hybrid/plant_gate/plant_gate_model.joblib ./experiments/by_dataset/hybrid/plant_gate/plant_gate_model.joblib
COPY experiments/by_dataset/hybrid/plant_gate/plant_gate_model.json ./experiments/by_dataset/hybrid/plant_gate/plant_gate_model.json

# Cloud Run sets PORT automatically
ENV PORT=8080
ENV GROUNDNUT_PRELOAD_MODELS=0
ENV GROUNDNUT_DISABLE_DOCS=1

EXPOSE $PORT

# Production entrypoint: read API key + port from environment
CMD sh -c "python run_server.py \"\${GROUNDNUT_API_KEY:-dev-local-key}\" \"\${PORT:-8080}\""