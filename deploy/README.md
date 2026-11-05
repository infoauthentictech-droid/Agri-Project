---
title: Groundnut AI Backend
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Groundnut AI Backend

FastAPI backend for groundnut weed detection and herbicide recommendations.

## API Endpoints

| Endpoint | Method | Auth |
|----------|--------|------|
| `/healthz` | GET | No |
| `/api/v1/health` | GET | Yes (`X-API-Key`) |
| `/api/v1/predict` | POST | Yes |
| `/api/v1/detect` | POST | Yes |
| `/api/v1/analyze` | POST | Yes |
| `/api/v1/recommend` | POST | Yes |
| `/api/v1/explain` | POST | Yes |
| `/api/v1/segment` | POST | Yes |

## Environment Variables

Set these in your Hugging Face Space settings:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROUNDNUT_API_KEYS` | Yes | Comma-separated valid API keys (e.g. `dev-my-secret-key`) |
| `GROUNDNUT_PRELOAD_MODELS` | No | Set to `0` to save memory (default: `0`) |
| `GROUNDNUT_DISABLE_DOCS` | No | Set to `1` to disable Swagger UI (default: `1`) |

## Usage

Send requests with the `X-API-Key` header:

```bash
curl https://your-username.space-name.hf.space/api/v1/health \
  -H "X-API-Key: dev-my-secret-key"
```

## Notes

- Free HF Spaces have CPU-only inference (no GPU).
- First request after idle time may take 30–60 seconds to wake the container.
- ML model weights are bundled in the image (~13 MB total).