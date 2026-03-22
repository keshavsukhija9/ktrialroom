"""Lightweight RSS sampling (best-effort; not a GPU profiler)."""

from __future__ import annotations

import os

import psutil


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    print(f"RSS MB: {rss_mb():.1f}")
