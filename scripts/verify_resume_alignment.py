#!/usr/bin/env python3
"""Verify shipped docs/code do not claim third-party pose stacks we do not use."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
FORBIDDEN = ("DWPose", "SCHP", "DensePose")
SUFFIXES = {".md", ".py", ".txt"}
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    "third_party",
    "models",
    ".mypy_cache",
    ".ruff_cache",
    ".cursor",
}


def iter_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in SUFFIXES:
            continue
        if p.resolve() == SELF:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        out.append(p)
    return out


def check_references() -> bool:
    hits: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for bad in FORBIDDEN:
            if bad in text:
                hits.append(f"{path.relative_to(ROOT)}: contains {bad!r}")

    if hits:
        print("❌ Found inaccurate product references:")
        print("\n".join(hits))
        print("\nReplace with MediaPipe pose + DeepLabV3 where describing this repo.")
        return False

    print("✅ No DWPose / SCHP / DensePose strings in tracked *.md, *.py, *.txt (excl. this script)")
    print("✅ Documentation aligns with shipped stack")
    return True


if __name__ == "__main__":
    sys.exit(0 if check_references() else 1)
