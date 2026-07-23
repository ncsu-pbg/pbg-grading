#!/bin/bash
# PBG Grading — station launcher (macOS). Double-click to start.
# Starts the app in the local environment; the browser opens by itself.
#
# Per-station overrides (uncomment to change):
#   export PBG_HOST=0.0.0.0   # also serve the app to other machines on the LAN
#   export PBG_PORT=8050      # change the port
#   export PBG_NO_BROWSER=1   # don't auto-open the browser
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".micromamba/micromamba" ] || [ ! -d "envs/grading" ]; then
  echo "Environment not found — run setup_macos.command first." >&2
  exit 1
fi

# Activate the env and run python directly.
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
eval "$(./.micromamba/micromamba shell hook --shell bash)"
micromamba activate "$PWD/envs/grading"
exec python PBG_grading.py
