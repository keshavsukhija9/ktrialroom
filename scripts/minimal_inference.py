#!/usr/bin/env python3
"""
Minimal inference smoke test: small resolution + 1 step + aggressive offload.

Run in Terminal.app with unbuffered I/O so you see progress before weights load::

    PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py 2>&1 | tee inference_log.txt

If RSS stays ~tens of MB for minutes, you are stuck *before* weights — check tail of the log for the last [n/N] line.
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    _log("[1/9] minimal_inference: entering main()")

    _log("[2/9] importing PIL…")
    from PIL import Image

    _log("[3/9] importing torch…")
    import torch

    _log(f"     torch {torch.__version__} | mps_built={torch.backends.mps.is_built()}")

    _log("[4/9] importing cv2, numpy…")
    import cv2
    import numpy as np

    _log("[5/9] loading project config…")
    from siliconvton.utils.project_config import load_merged_config, repo_root

    _log("[6/9] device + preprocessing (MediaPipe / torchvision DeepLab load here)…")
    from siliconvton.preprocessing.pose_estimator import PoseEstimator
    from siliconvton.preprocessing.segmenter import HumanSegmenter
    from siliconvton.utils.device_utils import get_device

    person_p = ROOT / "assets" / "sample_inputs" / "person_1.jpg"
    garment_p = ROOT / "assets" / "sample_inputs" / "garment_1.jpg"
    if not person_p.is_file() or not garment_p.is_file():
        _log("❌ Need assets/sample_inputs/person_1.jpg and garment_1.jpg")
        return 1

    dev = get_device("auto")
    _log(f"✅ Device: {dev.type}")

    person = Image.open(person_p).convert("RGB")
    garment = Image.open(garment_p).convert("RGB")

    bgr = cv2.cvtColor(np.asarray(person), cv2.COLOR_RGB2BGR)
    kps = PoseEstimator().extract_keypoints(bgr)
    _log(f"✅ Pose extracted: {len(kps)} keypoints")

    mask = HumanSegmenter(device=torch.device("cpu")).get_segmentation_mask(person)
    _log(f"✅ Segmentation mask generated ({mask.shape[0]}×{mask.shape[1]})")

    _log("[7/9] importing VTONPipeline (pulls diffusion_engine; still lazy until generate)…")
    from siliconvton.core.vton_pipeline import VTONPipeline

    cfg = load_merged_config(repo_root())
    w = int(os.environ.get("SILICONVTON_MIN_WIDTH", "256"))
    h = int(os.environ.get("SILICONVTON_MIN_HEIGHT", "256"))
    cfg["inference"]["width"] = w
    cfg["inference"]["height"] = h
    cfg["inference"]["num_inference_steps"] = 1
    cfg["optimization"]["precision"] = "fp16"
    cfg["optimization"]["enable_sequential_cpu_offload"] = True
    cfg["optimization"]["enable_model_cpu_offload"] = True

    _log("[8/9] constructing VTONPipeline…")
    t0 = time.perf_counter()
    pipe = VTONPipeline(cfg)
    _log(f"✅ Pipeline object ready ({time.perf_counter() - t0:.1f}s) — weights load on first generate")

    _log("[9/9] running diffusion (expect RSS to jump when UNet loads)…")
    t_inf = time.perf_counter()
    out = pipe(
        person,
        garment,
        "Short Sleeve T-shirt",
        num_inference_steps=1,
        seed=0,
    )
    infer_s = time.perf_counter() - t_inf
    total_s = time.perf_counter() - t0

    out_dir = ROOT / "assets" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "minimal_test.png"
    out["result_image"].save(path)
    kb = path.stat().st_size / 1024

    m = out.get("metrics", {})
    rss = out.get("performance", {}).get("memory_usage", {})
    rss_mb = rss.get("rss_mb", 0) if isinstance(rss, dict) else 0

    _log(f"✅ Inference complete in {infer_s:.1f}s (total since pipeline construct {total_s:.1f}s)")
    _log(f"✅ Output saved: {path} ({kb:.1f} KB)")
    if m:
        _log(f"✅ SSIM: {m.get('ssim', 'n/a')}, LPIPS: {m.get('lpips', 'n/a')}")
    if rss_mb:
        _log(f"✅ Memory RSS (sample): {rss_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    _log("minimal_inference: starting (use PYTHONUNBUFFERED=1 for line-by-line in pipes)")
    try:
        sys.exit(main())
    except Exception as e:
        _log(f"❌ Minimal inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
