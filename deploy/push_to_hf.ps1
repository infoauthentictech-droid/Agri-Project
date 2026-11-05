# =============================================================================
# Push Groundnut AI Backend to Hugging Face Spaces
# =============================================================================
# Prerequisites:
#   1. Create a Hugging Face account at https://huggingface.co
#   2. Create a new Space (SDK: Docker) at https://huggingface.co/spaces
#   3. Get your HF token from https://huggingface.co/settings/tokens
#
# Usage:
#   .\deploy\push_to_hf.ps1 -HfUser your-username -Token hf_xxx
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$HfUser,

    [Parameter(Mandatory=$true)]
    [string]$Token
)

$SpaceName = "groundnut-ai-backend"
$RemoteUrl = "https://$HfUser:$Token@huggingface.co/spaces/$HfUser/$SpaceName"

Write-Host "📦 Creating temporary clone of HF Space..." -ForegroundColor Cyan

# Create a temp directory with just the deploy files
$TempDir = Join-Path $env:TEMP "hf-space-$SpaceName"
if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Copy deploy contents to temp dir
Get-ChildItem "deploy\*" -File | ForEach-Object { Copy-Item $_.FullName -Destination $TempDir }

# Move up one level so Dockerfile is at root of the space repo
# (HF expects Dockerfile at the root)
Move-Item -Path "$TempDir\*" -Destination $TempDir -Force

Write-Host "🔧 Initializing git in temp directory..." -ForegroundColor Cyan
Push-Location $TempDir
git init
git add -A
git commit -m "Initial deploy: Groundnut AI backend" | Out-Null

Write-Host "🔗 Adding HF remote..." -ForegroundColor Cyan
git remote add origin $RemoteUrl

Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Cyan
git push -u origin master --force

Pop-Location

Write-Host "✅ Done! Your Space is building at:" -ForegroundColor Green
Write-Host "   https://huggingface.co/spaces/$HfUser/$SpaceName" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Go to https://huggingface.co/spaces/$HfUser/$SpaceName/settings" -ForegroundColor Cyan
Write-Host "   2. Add Environment Variable: GROUNDNUT_API_KEYS = dev-my-secret-key" -ForegroundColor Cyan
Write-Host "   3. Wait for the build to complete (check the 'Builds' tab)" -ForegroundColor Cyan
Write-Host "   4. Test: https://$HfUser-$SpaceName.hf.space/healthz" -ForegroundColor Cyan