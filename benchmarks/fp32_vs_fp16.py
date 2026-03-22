"""
Compare FP32 vs FP16 inference (run on device with weights cached).

Example:
  PYTHONPATH=. python benchmarks/fp32_vs_fp16.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siliconvton.core.vton_pipeline import VTONPipeline
from siliconvton.utils.project_config import load_merged_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    person = Image.open(ROOT / "assets/sample_inputs/person_1.jpg").convert("RGB")
    garment = Image.open(ROOT / "assets/sample_inputs/garment_1.jpg").convert("RGB")

    for label, prec in [("fp32", "fp32"), ("fp16", "fp16")]:
        cfg = load_merged_config(ROOT)
        cfg["optimization"]["precision"] = prec
        cfg["optimization"]["enable_model_cpu_offload"] = True
        cfg["inference"]["num_inference_steps"] = 20
        pipe = VTONPipeline(cfg)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            pipe(person, garment, "Short Sleeve T-shirt", enable_benchmark=False)
            times.append(time.perf_counter() - t0)
        print(f"{label}: avg {sum(times)/len(times):.2f}s over {args.runs} runs")


if __name__ == "__main__":
    main()
