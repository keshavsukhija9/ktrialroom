"""Lightweight timing helpers (used by benchmarks/)."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
