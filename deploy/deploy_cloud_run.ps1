# =============================================================================
# Deploy Groundnut AI Backend to Google Cloud Run
# =============================================================================
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. Docker Desktop running
#   3. Project ID set (gcloud config set project YOUR_PROJECT_ID)
#
# Usage:
#   .\deploy\deploy_cloud_run.ps1 -ProjectId YOUR_PROJECT_ID -ApiKey your-secret-key -Region us-central1
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,

    [Parameter(Mandatory=$true)]
    [string]$ApiKey,

    [string]$Region = "us-central1",

    [string]$ServiceName = "groundnut-ai-backend",

    [string]$RepoName = "groundnut-ai-repo"
)

$ImageName = "$Region-docker.pkg.dev/$ProjectId/$RepoName/$ServiceName:latest"

Write-Host "🔐 Step 1: Authenticating Docker with Artifact Registry..." -ForegroundColor Cyan
gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet

Write-Host "📦 Step 2: Creating Artifact Registry repository (if not exists)..." -ForegroundColor Cyan
gcloud artifacts repositories create $RepoName `
    --repository-format=docker `
    --location=$Region `
    --description="Groundnut AI backend images" `
    --quiet 2>$null

Write-Host "🔧 Step 3: Enabling required APIs..." -ForegroundColor Cyan
gcloud services enable run.googleapis.com --quiet
gcloud services enable artifactregistry.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet

Write-Host "🏗️  Step 4: Building Docker image..." -ForegroundColor Cyan
docker build -t $ImageName ./backend

Write-Host "📤 Step 5: Pushing image to Artifact Registry..." -ForegroundColor Cyan
docker push $ImageName

Write-Host "🚀 Step 6: Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy $ServiceName `
    --image $ImageName `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --set-env-vars "GROUNDNUT_API_KEYS=$ApiKey" `
    --set-env-vars "GROUNDNUT_PRELOAD_MODELS=0" `
    --set-env-vars "GROUNDNUT_DISABLE_DOCS=1"

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "🔗 Get your service URL:" -ForegroundColor Cyan
Write-Host "   gcloud run services describe $ServiceName --region $Region --format='value(status.url)'" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Green
Write-Host "🧪 Test endpoints:" -ForegroundColor Cyan
Write-Host "   curl.exe https://YOUR_SERVICE_URL/healthz" -ForegroundColor Cyan
Write-Host "   curl.exe https://YOUR_SERVICE_URL/api/v1/health -H `"X-API-Key: $ApiKey`"" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Green
Write-Host "📱 Connect Flutter:" -ForegroundColor Cyan
Write-Host "   flutter run --dart-define=GROUNDNUT_API_KEY=$ApiKey --dart-define=API_BASE_URL=https://YOUR_SERVICE_URL/api/v1" -ForegroundColor Cyan