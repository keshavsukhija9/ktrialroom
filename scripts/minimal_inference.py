#!/usr/bin/env python3
"""
Minimal inference smoke test: small resolution + 1 step + aggressive offload.

Run in Terminal.app on your Mac (not Cursor’s agent) — needs real unified memory for ~12GB weights.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from siliconvton.core.vton_pipeline import VTONPipeline
from siliconvton.preprocessing.pose_estimator import PoseEstimator
from siliconvton.preprocessing.segmenter import HumanSegmenter
from siliconvton.utils.device_utils import get_device
from siliconvton.utils.project_config import load_merged_config, repo_root


def main() -> int:
    person_p = ROOT / "assets" / "sample_inputs" / "person_1.jpg"
    garment_p = ROOT / "assets" / "sample_inputs" / "garment_1.jpg"
    if not person_p.is_file() or not garment_p.is_file():
        print("❌ Need assets/sample_inputs/person_1.jpg and garment_1.jpg", flush=True)
        return 1

    dev = get_device("auto")
    print(f"✅ Device: {dev.type}", flush=True)

    person = Image.open(person_p).convert("RGB")
    garment = Image.open(garment_p).convert("RGB")

    bgr = cv2.cvtColor(np.asarray(person), cv2.COLOR_RGB2BGR)
    kps = PoseEstimator().extract_keypoints(bgr)
    print(f"✅ Pose extracted: {len(kps)} keypoints", flush=True)

    mask = HumanSegmenter(device=torch.device("cpu")).get_segmentation_mask(person)
    print(f"✅ Segmentation mask generated ({mask.shape[0]}×{mask.shape[1]})", flush=True)

    cfg = load_merged_config(repo_root())
    cfg["inference"]["width"] = 256
    cfg["inference"]["height"] = 256
    cfg["inference"]["num_inference_steps"] = 1
    cfg["optimization"]["precision"] = "fp16"
    cfg["optimization"]["enable_sequential_cpu_offload"] = True
    cfg["optimization"]["enable_model_cpu_offload"] = True

    print("⏳ Building pipeline (Tryon UNet loads on first generate — may take minutes first time)…", flush=True)
    t0 = time.perf_counter()
    pipe = VTONPipeline(cfg)
    print("✅ Pipeline ready", flush=True)

    print("🚀 Running minimal inference (1 step, 256×256, sequential offload)…", flush=True)
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

    print(f"✅ Inference complete in {infer_s:.1f}s (total setup+run {total_s:.1f}s)", flush=True)
    print(f"✅ Output saved: {path} ({kb:.1f} KB)", flush=True)
    if m:
        print(f"✅ SSIM: {m.get('ssim', 'n/a')}, LPIPS: {m.get('lpips', 'n/a')}", flush=True)
    if rss_mb:
        print(f"✅ Memory RSS (sample): {rss_mb:.1f} MB", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ Minimal inference failed: {e}", flush=True)
        sys.exit(1)
