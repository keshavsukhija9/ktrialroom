#!/usr/bin/env python3
"""Verify inference output meets quality standards."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

OUTPUT = Path("assets/outputs/final_inference_test.png")


def verify_output() -> bool:
    if not OUTPUT.exists():
        print("❌ Output file not found")
        return False

    size_kb = OUTPUT.stat().st_size / 1024
    if size_kb < 100:
        print(f"❌ Output file too small: {size_kb:.1f}KB")
        return False

    img = Image.open(OUTPUT)
    w, h = img.size
    if w < 256 or h < 256:
        print(f"❌ Image too small: {w}x{h}")
        return False

    print(f"✅ Output verified: {w}x{h}, {size_kb:.1f}KB")
    return True


if __name__ == "__main__":
    success = verify_output()
    sys.exit(0 if success else 1)
