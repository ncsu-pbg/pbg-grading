#!/bin/bash
# PBG Grading — one-time station setup (macOS).
# Double-click this file (or run it in Terminal). It:
#   1. downloads micromamba (a single small binary; no admin rights, does
#      not touch the system/Anaconda Python),
#   2. builds the pinned conda environment from environment.yml into
#      ./envs/grading inside this folder.
# Re-running is safe: it rebuilds the environment from scratch (package
# downloads are cached in ./.micromamba, so a rebuild is fast).
set -euo pipefail
cd "$(dirname "$0")"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64)  PLAT="osx-arm64" ;;
  x86_64) PLAT="osx-64" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

MM=".micromamba/micromamba"
if [ ! -x "$MM" ]; then
  echo "==> Downloading micromamba ($PLAT)…"
  mkdir -p .micromamba
  curl -fL --retry 3 \
    "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$PLAT" \
    -o "$MM"
  chmod +x "$MM"
fi
echo "==> micromamba $("$MM" --version)"

export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
ENV_PREFIX="$PWD/envs/grading"

# If this folder lives inside Dropbox, keep Dropbox away from the env:
# syncing offloads/deletes binaries out of it and breaks the app.
mkdir -p envs
xattr -w com.dropbox.ignored 1 envs 2>/dev/null || true
xattr -w com.dropbox.ignored 1 .micromamba 2>/dev/null || true

if [ -d "$ENV_PREFIX" ]; then
  echo "==> Existing environment found — rebuilding it."
  rm -rf "$ENV_PREFIX"
  if [ -d "$ENV_PREFIX" ]; then
    echo "ERROR: could not fully remove the old environment at:" >&2
    echo "  $ENV_PREFIX" >&2
    echo "Close the PBG Grading app (and any terminal running it), pause" >&2
    echo "Dropbox syncing, delete the envs folder by hand, then re-run." >&2
    exit 1
  fi
fi

echo "==> Building environment from environment.yml (several minutes on first run)…"
"$MM" create -y -p "$ENV_PREFIX" -f environment.yml

echo "==> Verifying imports…"
"$MM" run -p "$ENV_PREFIX" python -c "import dash, plotly, numpy, pandas, scipy, cv2, yaml, dotenv, requests, dropbox, depthai, PyQt6.QtCore; print('  python', __import__('sys').version.split()[0]); print('  depthai', depthai.__version__); print('  opencv', cv2.__version__)"

echo ""
echo "Setup complete. Double-click 'Start PBG Grading.command' to launch the app."
