#!/usr/bin/env bash
# Download yisol/IDM-VTON into models/IDM-VTON.
# If hf_transfer is installed, enables fast parallel downloads (see HF docs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source ".venv/bin/activate"
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true
  echo "Note: hf_transfer not installed — using standard downloader. Run: pip install hf_transfer"
fi
exec huggingface-cli download yisol/IDM-VTON \
  --local-dir models/IDM-VTON \
  --resume-download
