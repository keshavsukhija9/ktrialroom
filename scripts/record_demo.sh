#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🎬 SILICONVTON DEMO RECORDING SETUP"
echo "===================================="
echo ""
echo "MANUAL STEPS (follow carefully):"
echo ""
echo "1. Open QuickTime Player"
echo "   → File → New Screen Recording"
echo "   → Click record, select full screen"
echo ""
echo "2. In terminal, Gradio app will launch"
echo "   → Wait for 'Running on local URL' message"
echo ""
echo "3. In browser (http://127.0.0.1:7860):"
echo "   → Upload: assets/sample_inputs/person_1.jpg"
echo "   → Upload: assets/sample_inputs/garment_1.jpg"
echo "   → Run try-on / generate"
echo "   → Wait for result (minutes on first cold start)"
echo ""
echo "4. When result appears:"
echo "   → Show before/after"
echo "   → Show metrics if displayed (SSIM, LPIPS, time, memory)"
echo "   → Stop QuickTime recording"
echo "   → Save as: assets/demo_backup.mp4"
echo ""
echo "5. Verify recording:"
echo "   → ls -lh assets/demo_backup.mp4"
echo "   → Should be >5MB for a full run"
echo ""
if [ -t 0 ]; then
  read -r -p "Press Enter to launch Gradio app..."
else
  echo "(non-interactive: launch Gradio yourself: PYTHONPATH=. python ui/gradio_app.py)"
  exit 0
fi

source .venv/bin/activate
export PYTHONPATH=.
exec python ui/gradio_app.py
