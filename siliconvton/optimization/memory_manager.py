from __future__ import annotations

import os
from typing import Any, Dict

import psutil
import torch


class MemoryManager:
    """Process RSS / unified-memory friendly tracking (best-effort on macOS)."""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())

    def get_usage(self) -> Dict[str, Any]:
        mi = self._proc.memory_info()
        return {
            "rss_mb": float(mi.rss) / 1024 / 1024,
            "vms_mb": float(mi.vms) / 1024 / 1024,
            "percent": float(self._proc.memory_percent()),
        }

    def clear_mps(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
