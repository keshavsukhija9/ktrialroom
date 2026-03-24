#!/usr/bin/env python3
"""
Five-step import debug (run before minimal_inference).

On a healthy Mac venv, ``import torch`` and this whole script usually finish in
well under a minute (often ~10–30s for torch alone on first load). If step 1
hangs for minutes, reinstall PyTorch or recreate ``.venv`` — see
``docs/COMPLETION_CHECKLIST.md`` §2.

  cd /path/to/resume_tryon && source .venv/bin/activate
  PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py

Test 4 does NOT load multi-GB weights — import_tryon_modules() only imports
Python classes from third_party/idm-vton. Weights load later in
load_tryon_pipeline() / first VTONPipeline.generate().

Wrong (will TypeError): import_tryon_modules('models/IDM-VTON')
Correct: import_tryon_modules()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(name: str, fn) -> bool:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
        dt = time.perf_counter() - t0
        print(f"✅ OK ({dt:.2f}s)", flush=True)
        return True
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"❌ FAIL after {dt:.2f}s: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    t_all = time.perf_counter()
    print("SiliconVTON — 5 import tests (torch → pipeline init, no weight load in step 4)", flush=True)

    ok = True

    def t1():
        import torch

        print(f"  torch {torch.__version__}", flush=True)
        print(f"  MPS available: {torch.backends.mps.is_available()}", flush=True)

    ok = run("Test 1: torch + MPS", t1) and ok

    def t2():
        import mediapipe as mp

        print(f"  mediapipe {getattr(mp, '__version__', '?')}", flush=True)

    ok = run("Test 2: mediapipe", t2) and ok

    def t3():
        import diffusers

        print(f"  diffusers {diffusers.__version__}", flush=True)

    ok = run("Test 3: diffusers", t3) and ok

    def t4():
        from siliconvton.models.model_loader import import_tryon_modules

        print("  import_tryon_modules() — vendor TryonPipeline + UNets (no disk weights yet)", flush=True)
        TryonPipeline, UNetTryon, UNetEncoder = import_tryon_modules()
        assert TryonPipeline and UNetTryon and UNetEncoder

    ok = run("Test 4: model loader", t4) and ok

    def t5():
        from siliconvton.core.vton_pipeline import VTONPipeline
        from siliconvton.utils.project_config import load_merged_config, repo_root

        cfg = load_merged_config(repo_root())
        cfg["inference"]["num_inference_steps"] = 1
        VTONPipeline(cfg)
        print("  VTONPipeline initialized (lazy_load: weights on first generate)", flush=True)

    ok = run("Test 5: VTONPipeline init", t5) and ok

    elapsed = time.perf_counter() - t_all
    if ok:
        print(
            f"\n========================================\n"
            f"✅ ALL 5 TESTS PASSED (total {elapsed:.1f}s)\n"
            f"========================================\n"
            f"Next: PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py",
            flush=True,
        )
        return 0
    print(
        f"\n❌ One or more tests failed (ran {elapsed:.1f}s) — fix the first failure before minimal_inference.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
