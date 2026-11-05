# Google Cloud Run Deployment — Groundnut AI Backend

This guide deploys the FastAPI backend to [Google Cloud Run](https://cloud.google.com/run),
which supports Docker containers and has a generous free tier (much better memory than Render's free tier).

## Prerequisites

1. A Google Cloud account: https://console.cloud.google.com
2. [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install) installed on your PC
3. Docker Desktop installed and running
4. A credit card on file with Google Cloud (you won't be charged within free limits)

## Step 1: Set up Google Cloud

```powershell
# Log in to Google Cloud
gcloud auth login

# Set your project (create one first in the console if needed)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## Step 2: Create Artifact Registry repository

```powershell
gcloud artifacts repositories create groundnut-ai-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Groundnut AI backend images"
```

## Step 3: Build and push the Docker image

```powershell
# Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build the image
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/groundnut-ai-repo/groundnut-ai-backend:latest ./backend

# Push the image
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/groundnut-ai-repo/groundnut-ai-backend:latest
```

## Step 4: Deploy to Cloud Run

```powershell
gcloud run deploy groundnut-ai-backend \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/groundnut-ai-repo/groundnut-ai-backend:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --set-env-vars "GROUNDNUT_API_KEYS=your-secret-key-here" \
    --set-env-vars "GROUNDNUT_PRELOAD_MODELS=0" \
    --set-env-vars "GROUNDNUT_DISABLE_DOCS=1"
```

Replace `your-secret-key-here` with your chosen API key.

## Step 5: Get your service URL

```powershell
gcloud run services describe groundnut-ai-backend --region us-central1 --format="value(status.url)"
```

This returns something like:
```
https://groundnut-ai-backend-xxxxxx-uc.a.run.app
```

## Step 6: Test the backend

```powershell
# Health check (no auth needed)
curl.exe https://YOUR_SERVICE_URL/healthz

# API endpoint (requires X-API-Key header)
curl.exe https://YOUR_SERVICE_URL/api/v1/health -H "X-API-Key: your-secret-key-here"
```

## Step 7: Connect your Flutter app

Update your Flutter run/build command:

```powershell
flutter run --dart-define=GROUNDNUT_API_KEY=your-secret-key-here --dart-define=API_BASE_URL=https://YOUR_SERVICE_URL/api/v1
```

Or rebuild the release APK:

```powershell
flutter build apk --release --dart-define=GROUNDNUT_API_KEY=your-secret-key-here --dart-define=API_BASE_URL=https://YOUR_SERVICE_URL/api/v1
```

## Free Tier Limits

Cloud Run free tier includes:
- 2 million requests/month
- 360,000 GB-seconds of memory
- 200,000 GHz-seconds of CPU
- 1 GiB memory per instance

This is significantly more generous than Render's free tier (512 MB RAM), so PyTorch + YOLO should work fine.

## One-Liner Deploy Script

Use `deploy\deploy_cloud_run.ps1` to automate steps 2–4:

```powershell
.\deploy\deploy_cloud_run.ps1 -ProjectId YOUR_PROJECT_ID -ApiKey your-secret-key-here -Region us-central1
```