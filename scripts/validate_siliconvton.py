#!/usr/bin/env python3
"""
SiliconVTON validation runner (QA).

Phases:
  1) Import all modules
  2) Instantiate preprocess + memory on sample images
  3) Import upstream IDM-VTON Tryon modules (requires third_party + einops)
  4) Optional: load HF weights + one inference (SILICONVTON_FULL_INFERENCE=1, long + large download)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "siliconvton.core.vton_pipeline",
    "siliconvton.core.diffusion_engine",
    "siliconvton.core.quality_metrics",
    "siliconvton.preprocessing.pose_estimator",
    "siliconvton.preprocessing.segmenter",
    "siliconvton.preprocessing.garment_warper",
    "siliconvton.preprocessing.image_validator",
    "siliconvton.optimization.memory_manager",
    "siliconvton.optimization.precision_handler",
    "siliconvton.models.model_loader",
    "siliconvton.utils.device_utils",
    "ui.gradio_app",
    "benchmarks.fp32_vs_fp16",
]


def phase1_imports() -> int:
    print("=== PHASE 1: import validation ===")
    failed = []
    for mod in MODULES:
        try:
            __import__(mod)
            print(f"  OK {mod}")
        except Exception as e:
            print(f"  FAIL {mod}: {e}")
            failed.append(mod)
    if failed:
        print(f"\nFAILED: {len(failed)} modules")
        return 1
    print(f"\nAll {len(MODULES)} modules imported.\n")
    return 0


def phase2_preprocess() -> int:
    print("=== PHASE 2: preprocessing on sample images ===")
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    p = ROOT / "assets/sample_inputs/person_1.jpg"
    g = ROOT / "assets/sample_inputs/garment_1.jpg"
    if not p.is_file() or not g.is_file():
        print("MISSING sample images under assets/sample_inputs/")
        return 1

    person = Image.open(p).convert("RGB")
    garment = Image.open(g).convert("RGB")

    from siliconvton.preprocessing.image_validator import ImageValidator
    from siliconvton.preprocessing.pose_estimator import PoseEstimator
    from siliconvton.preprocessing.segmenter import HumanSegmenter
    from siliconvton.preprocessing.pose_canvas import keypoints_to_pose_image
    from siliconvton.preprocessing.mask_builder import torso_inpaint_region, inpaint_mask_to_pil
    from siliconvton.preprocessing.garment_warper import GarmentWarper
    from siliconvton.optimization.memory_manager import MemoryManager

    v = ImageValidator(768, 1024)
    if not v.validate(person)[0]:
        return 1
    pv = v.letterbox(person)
    gv = v.letterbox(garment)
    pe = PoseEstimator()
    bgr = cv2.cvtColor(np.asarray(pv), cv2.COLOR_RGB2BGR)
    kps = pe.extract_keypoints(bgr)
    assert len(kps) == 33
    seg = HumanSegmenter(device=torch.device("cpu"))
    mask = seg.get_segmentation_mask(pv)
    keypoints_to_pose_image(kps, 768, 1024)
    inpaint_mask_to_pil(torso_inpaint_region(mask))
    GarmentWarper(768, 1024).prepare(cv2.cvtColor(np.asarray(gv), cv2.COLOR_RGB2BGR))
    mm = MemoryManager()
    mm.get_usage()
    mm.clear_mps()
    print("  Preprocessing + memory: OK\n")
    return 0


def phase3_vendor() -> int:
    print("=== PHASE 3: upstream IDM-VTON (TryonPipeline) imports ===")
    try:
        from siliconvton.models.model_loader import ensure_idm_on_path, import_tryon_modules

        ensure_idm_on_path()
        import_tryon_modules()
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    print("  Vendor modules: OK\n")
    return 0


def phase4_inference_optional() -> int:
    if os.environ.get("SILICONVTON_FULL_INFERENCE") != "1":
        print("=== PHASE 4: full inference SKIPPED (set SILICONVTON_FULL_INFERENCE=1) ===\n")
        return 0
    print("=== PHASE 4: full inference (downloads weights + runs MPS/CPU) ===")
    from PIL import Image

    from siliconvton.core.vton_pipeline import VTONPipeline
    from siliconvton.utils.project_config import load_merged_config

    cfg = load_merged_config(ROOT)
    cfg["inference"]["num_inference_steps"] = 2
    cfg["optimization"]["precision"] = "fp16"
    # Prefer sequential offload for full validation on Apple Silicon (lower peak RAM than model offload alone).
    cfg["optimization"]["enable_sequential_cpu_offload"] = True
    cfg["optimization"]["enable_model_cpu_offload"] = True

    pipe = VTONPipeline(cfg)
    person = Image.open(ROOT / "assets/sample_inputs/person_1.jpg").convert("RGB")
    garment = Image.open(ROOT / "assets/sample_inputs/garment_1.jpg").convert("RGB")
    out = pipe(person, garment, "Short Sleeve T-shirt", num_inference_steps=2, seed=0)
    assert out["result_image"].size == (768, 1024)

    out_dir = ROOT / "assets" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_inference_test.png"
    out["result_image"].save(out_path)
    print(f"  Saved: {out_path}")
    print("  Inference: OK\n")
    return 0


def phase5_ui_smoke() -> int:
    print("=== PHASE 5: Gradio app smoke (import only, no server) ===")
    import ui.gradio_app as g

    assert callable(g.run_tryon)
    assert callable(g.main)
    print("  Gradio: OK\n")
    return 0


def main() -> int:
    code = phase1_imports()
    if code:
        return code
    code = phase2_preprocess()
    if code:
        return code
    code = phase3_vendor()
    if code:
        return code
    code = phase4_inference_optional()
    if code:
        return code
    code = phase5_ui_smoke()
    if code:
        return code
    print("=== VALIDATION COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
