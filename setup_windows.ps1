# PBG Grading — one-time station setup (Windows).
# Run via setup_windows.bat (double-click), or directly:
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1
# It:
#   1. downloads micromamba (a single small .exe; no admin rights, does not
#      touch any system Python),
#   2. builds the pinned conda environment from environment.yml into
#      .\envs\grading inside this folder,
#   3. puts a "PBG Grading" shortcut on the Desktop.
# Re-running is safe: it rebuilds the environment from scratch (package
# downloads are cached in .\.micromamba, so a rebuild is fast).
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# TLS 1.2 for Windows PowerShell 5.x
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$mmDir = Join-Path $PSScriptRoot ".micromamba"
$mm    = Join-Path $mmDir "micromamba.exe"
if (!(Test-Path -LiteralPath $mm)) {
    Write-Host "==> Downloading micromamba (win-64)..."
    New-Item -ItemType Directory -Force -Path $mmDir | Out-Null
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64" `
        -OutFile $mm
}
Write-Host "==> micromamba $(& $mm --version)"

$env:MAMBA_ROOT_PREFIX = $mmDir
$envPrefix = Join-Path $PSScriptRoot "envs\grading"

if (Test-Path -LiteralPath $envPrefix) {
    Write-Host "==> Existing environment found - rebuilding it."
    Remove-Item -Recurse -Force -LiteralPath $envPrefix
}

Write-Host "==> Building environment from environment.yml (several minutes on first run)..."
& $mm create -y -p $envPrefix -f (Join-Path $PSScriptRoot "environment.yml")
if ($LASTEXITCODE -ne 0) { throw "Environment build failed (micromamba exit code $LASTEXITCODE)." }

Write-Host "==> Verifying imports..."
& $mm run -p $envPrefix python -c "import dash, plotly, numpy, pandas, scipy, cv2, yaml, dotenv, requests, dropbox, depthai, PyQt6.QtCore; import sys; print('  python', sys.version.split()[0]); print('  depthai', depthai.__version__); print('  opencv', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Import verification failed." }

Write-Host "==> Creating Desktop shortcut..."
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "PBG Grading.lnk"))
$lnk.TargetPath       = Join-Path $PSScriptRoot "Start PBG Grading.bat"
$lnk.WorkingDirectory = $PSScriptRoot
$lnk.Description      = "PBG Grading station"
$lnk.Save()

Write-Host ""
Write-Host "Setup complete. Launch with the 'PBG Grading' Desktop shortcut"
Write-Host "(or 'Start PBG Grading.bat' in this folder)."
