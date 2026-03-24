#!/usr/bin/env python3
"""
Quick diagnostic: where do imports hang? Run before minimal_inference.

  cd /path/to/resume_tryon && source .venv/bin/activate
  PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/test_critical_imports.py

For the numbered 5-test flow (torch → pipeline init), use:

  PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py

import_tryon_modules() takes no arguments (vendor path is third_party/idm-vton).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step(name: str):
    t = time.perf_counter()
    print(f"→ {name} …", flush=True)
    return t


def done(t0: float, name: str):
    print(f"  OK {name} ({time.perf_counter() - t0:.2f}s)", flush=True)


def main() -> int:
    print("=== SiliconVTON critical import test ===\n", flush=True)

    t = step("import torch")
    import torch

    done(t, f"torch {torch.__version__}")

    t = step("import mediapipe")
    import mediapipe as mp

    done(t, f"mediapipe {getattr(mp, '__version__', '?')}")

    t = step("import diffusers")
    import diffusers

    done(t, f"diffusers {diffusers.__version__}")

    t = step("import_tryon_modules() [needs third_party/idm-vton]")
    from siliconvton.models.model_loader import import_tryon_modules

    import_tryon_modules()
    done(t, "TryonPipeline / hacked UNets importable")

    t = step("resolve_model_id + local weights folder")
    from siliconvton.models.model_loader import resolve_model_id
    from siliconvton.utils.project_config import load_merged_config, repo_root

    cfg = load_merged_config(repo_root())
    mid = resolve_model_id(cfg["model"]["name"])
    print(f"  model path/id: {mid}", flush=True)
    done(t, "config OK")

    print("\n✅ All critical imports passed. If minimal_inference still hangs, watch its [n/9] lines.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n❌ Failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
