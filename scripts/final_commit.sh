#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🔒 SILICONVTON FINAL COMMIT"
echo "==========================="
echo ""

echo "Running pre-commit checks..."

if compgen -G "$HOME/.cache/huggingface/hub/models--yisol--IDM-VTON/blobs/"*.incomplete >/dev/null 2>&1; then
  echo "❌ Incomplete HF blobs still present. Finish download first."
  exit 1
fi

if [ ! -f "assets/outputs/final_inference_test.png" ]; then
  echo "❌ assets/outputs/final_inference_test.png missing. Run:"
  echo "   SILICONVTON_FULL_INFERENCE=1 PYTHONPATH=. python scripts/validate_siliconvton.py"
  exit 1
fi

if [ ! -f "assets/demo_backup.mp4" ]; then
  echo "⚠️  assets/demo_backup.mp4 not found."
  if [ -t 0 ]; then
    read -r -p "Continue without demo video? (y/n): " confirm
    if [ "${confirm:-n}" != "y" ]; then
      exit 1
    fi
  else
    echo "(non-interactive: set ALLOW_NO_DEMO=1 to skip)"
    if [ "${ALLOW_NO_DEMO:-0}" != "1" ]; then
      exit 1
    fi
  fi
fi

echo "Running test suite..."
PYTHONPATH=. pytest tests/ -q --tb=short

echo "Checking resume alignment..."
PYTHONPATH=. python scripts/verify_resume_alignment.py

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  git init
  git branch -M main 2>/dev/null || true
fi

echo ""
git status

if [ -n "${COMMIT_MSG:-}" ]; then
  message="$COMMIT_MSG"
elif [ -t 0 ]; then
  echo ""
  echo "Enter commit message (or set COMMIT_MSG):"
  read -r message
else
  message="SiliconVTON: validation, docs, demo workflow"
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$message"

if git remote | grep -q .; then
  git push || echo "⚠️  git push failed (check remote / auth)"
else
  echo "ℹ️  No git remote configured; skipping push."
fi

echo ""
echo "✅ Final commit recorded."
echo "   Add remote: git remote add origin <url> && git push -u origin main"
