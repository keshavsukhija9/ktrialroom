#!/usr/bin/env python3
"""Verify inference output: full-res or minimal PoC image under assets/outputs/."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

OUT_DIR = Path("assets/outputs")
# (filename, min_size_kb) — minimal run uses smaller PNGs
CANDIDATES: list[tuple[str, float]] = [
    ("final_inference_test.png", 100.0),
    ("minimal_test.png", 50.0),
]


def verify_output() -> bool:
    chosen: Path | None = None
    min_kb = 0.0
    for name, min_k in CANDIDATES:
        p = OUT_DIR / name
        if p.is_file() and p.stat().st_size / 1024 >= min_k:
            chosen = p
            min_kb = min_k
            break

    if chosen is None:
        print("❌ No valid output found. Expected one of:")
        for name, min_k in CANDIDATES:
            print(f"   - {OUT_DIR / name} (≥ {min_k:.0f} KB)")
        return False

    size_kb = chosen.stat().st_size / 1024
    img = Image.open(chosen)
    w, h = img.size
    if w < 128 or h < 128:
        print(f"❌ Image too small: {w}x{h}")
        return False

    print(f"✅ Output verified: {w}x{h}, {size_kb:.1f}KB ({chosen.name})")
    return True


if __name__ == "__main__":
    sys.exit(0 if verify_output() else 1)
