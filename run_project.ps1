<#
.SYNOPSIS
    One-command setup + run for the Groundnut AI project (USB connection).

    Does everything in one go:
      1. Installs Flutter dependencies        (flutter pub get)
      2. Starts the FastAPI backend on 0.0.0.0:8000  (background window)
      3. Builds the debug APK configured to talk to localhost via USB tunnel
      4. Installs it on your connected Android phone
      5. Launches the app on the phone

.PARAMETER DeviceId
    Android device serial (e.g. UCS8PJVSM7RWDYSW). Auto-detected when empty.

.PARAMETER ApiKey
    Shared secret used by both the backend (env var GROUNDNUT_API_KEY) and the
    app (--dart-define). A random one is generated per run when not given.

.PARAMETER InstallBackendDeps
    If set, installs backend/requirements.txt into the detected Python before
    starting the backend.

.PARAMETER SkipBuild
    If set, skips building the APK and uses the existing one (faster for development).

.PARAMETER SkipInstall
    If set, skips phone detection and APK installation. The APK is built but not installed.

.EXAMPLE
    .\run_project.ps1
.EXAMPLE
    .\run_project.ps1 -ApiKey my-secret-key
.EXAMPLE
    .\run_project.ps1 -InstallBackendDeps
.EXAMPLE
    .\run_project.ps1 -SkipBuild
.EXAMPLE
    .\run_project.ps1 -SkipInstall
#>
param(
    [string]$DeviceId = "",
    [string]$ApiKey = "dev-groundnut-$([guid]::NewGuid().ToString('N').Substring(0,8))",
    [switch]$InstallBackendDeps = $false,
    [switch]$SkipBuild = $false,
    [switch]$SkipInstall = $false
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "================ Groundnut AI: one-click run (USB) ================" -ForegroundColor Cyan

# USB tunnel mode: phone reaches backend via localhost
$BackendUrl = "http://127.0.0.1:8000/api/v1"
Write-Host "[0] Backend URL for the app = $BackendUrl (USB tunnel mode)" -ForegroundColor Cyan

# --- 0.5  Stop any leftover backend from a previous run (idempotent) ---------
$old = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($old) {
    foreach ($conn in $old) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -like '*python*') {
            Write-Host "        stopping previous backend PID $($conn.OwningProcess) ..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
}
# --- 1. Python discovery + optional backend deps -------------------------
# Prefer an explicit venv, else the `py` launcher, else whatever `python` is
# on PATH â€” never a hard-coded absolute path.
$PythonExe = ""
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyOut = (py -3 -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
    if ($pyOut) { $PythonExe = $pyOut }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    $candidate = Get-Command python -ErrorAction SilentlyContinue
    if ($candidate) { $PythonExe = $candidate.Source }
}
if (-not $PythonExe -or -not (Test-Path $PythonExe)) { throw "Python not found. Install Python 3.10+ and put `py`/`python` on PATH." }
Write-Host "[1/6] Python: $PythonExe" -ForegroundColor Cyan

if ($InstallBackendDeps) {
    Write-Host "    Installing backend dependencies (backend/requirements.txt) ..." -ForegroundColor Yellow
    Push-Location (Join-Path $PSScriptRoot "backend")
    try {
        & $PythonExe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install of backend requirements failed." }
    } finally { Pop-Location }
}

# --- 2. Flutter dependencies ----------------------------------------------
Write-Host "[2/6] flutter pub get ..." -ForegroundColor Cyan
flutter pub get
if ($LASTEXITCODE -ne 0) { throw "flutter pub get failed." }

# --- 3. Start the FastAPI backend ------------------------------------------
Write-Host "[3/6] Starting backend at http://0.0.0.0:8000 (log: backend_run.log) ..." -ForegroundColor Cyan
$env:GROUNDNUT_API_KEY = $ApiKey
$BackendLog = Join-Path $PSScriptRoot "backend_run.log"
$BackendErr = Join-Path $PSScriptRoot "backend_run.err.log"
# run_server.py passes the key INSIDE the python process (more reliable than
# inheriting it through the shell).
$backend = Start-Process -FilePath $PythonExe `
    -ArgumentList @("run_server.py", $ApiKey) `
    -WorkingDirectory (Join-Path $PSScriptRoot "backend") `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $BackendLog `
    -RedirectStandardError $BackendErr
Write-Host "        backend PID = $($backend.Id)"

$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" `
            -Headers @{ "X-API-Key" = $ApiKey } -TimeoutSec 3
        $healthy = $true
        Write-Host "        OK  model=$($r.model_loaded) recs=$($r.recommendation_engine_loaded) xai=$($r.xai_loaded)" -ForegroundColor Green
        break
    } catch { }
}
if (-not $healthy) {
    Write-Host "        WARNING: backend did not respond yet. See backend_run.err.log" -ForegroundColor Red
}

# --- 3. Build debug APK (points at the backend + API key) -------------
$apk = Join-Path $PSScriptRoot "build\app\outputs\flutter-apk\app-debug.apk"
if ($SkipBuild -and (Test-Path $apk)) {
    Write-Host "[4/6] Skipping build (using existing APK) ..." -ForegroundColor Cyan
    Write-Host "        APK: $apk" -ForegroundColor Green
} else {
    Write-Host "[4/6] Building debug APK (backend: $BackendUrl) ..." -ForegroundColor Cyan
    flutter build apk --debug `
        --dart-define="API_BASE_URL=$BackendUrl" `
        --dart-define="GROUNDNUT_API_KEY=$ApiKey"
    if ($LASTEXITCODE -ne 0) { throw "flutter build apk --debug failed." }
    Write-Host "        APK ready: $apk" -ForegroundColor Green
}

# --- 4. Detect the Android phone -------------------------------------------
if (-not $SkipInstall) {
    if (-not $DeviceId) {
        # Try flutter devices first
        $devices = (flutter devices | Out-String)
        $m = $devices -split "`r?`n" | Where-Object { $_ -match "\(mobile\)\s+[^\s\r\n]+\s+(\S+)" } | Select-Object -First 1
        if ($m) { $DeviceId = ([regex]::Match($m, "\(mobile\)\s+[^\s\r\n]+\s+(\S+)")).Groups[1].Value }

        # Fallback to adb devices if flutter didn't detect
        if (-not $DeviceId) {
            $adb = "adb"
            if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
                $adb = "$env:LOCALAPPDATA\Android\sdk\platform-tools\adb.exe"
            }
            $adbDevices = (& $adb devices 2>&1 | Out-String)
            $m = $adbDevices -split "`r?`n" | Where-Object { $_ -match "^\s*(\S+)\s+device$" } | Select-Object -First 1
            if ($m) { $DeviceId = ([regex]::Match($m, "^\s*(\S+)\s+device$")).Groups[1].Value }
        }
    }
    if (-not $DeviceId) {
        Write-Host "[4/6] No Android phone detected. Skipping install." -ForegroundColor Yellow
        Write-Host "    APK is ready at: $apk" -ForegroundColor Cyan
        Write-Host "    Install manually: adb install -r $apk" -ForegroundColor Cyan
        Write-Host "    Or connect your phone and re-run this script." -ForegroundColor Cyan
        $SkipInstall = $true
    }
}

# --- 5. Install + launch on the phone ---------------------------------------
if (-not $SkipInstall) {
    Write-Host "[5/6] Installing on phone ($DeviceId) ..." -ForegroundColor Cyan
    $adb = "adb"
    if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
        $adb = "$env:LOCALAPPDATA\Android\sdk\platform-tools\adb.exe"
    }
    & $adb install -r $apk
    if ($LASTEXITCODE -ne 0) { throw "adb install failed. Unlock the phone and accept the install dialog." }
    Write-Host "[6/6] Launching the app ..." -ForegroundColor Cyan
    # USB tunnel so the phone's localhost:8000 reaches this PC's backend.
    & $adb reverse tcp:8000 tcp:8000 2>&1 | Out-Null
    & $adb -s $DeviceId shell am start -n com.groundnutai.groundnut_ai/.MainActivity
} else {
    Write-Host "[5/6] Skipping install (phone not connected or -SkipInstall used)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== Done! ==" -ForegroundColor Green
if (-not $SkipInstall) {
    Write-Host "  Backend : localhost:8000 (phone reaches it via USB tunnel)  docs: http://localhost:8000/docs"
} else {
    Write-Host "  Backend : localhost:8000  docs: http://localhost:8000/docs"
}
Write-Host "  API key : $ApiKey"
Write-Host "  Backend PID: $($backend.Id)   (to stop later:  Stop-Process -Id $($backend.Id))"
Write-Host "  Backend logs: backend_run.log / backend_run.err.log"
if ($SkipInstall) {
    Write-Host ""
    Write-Host "  APK location: $apk" -ForegroundColor Cyan
    Write-Host "  To install manually: adb install -r $apk" -ForegroundColor Cyan
}