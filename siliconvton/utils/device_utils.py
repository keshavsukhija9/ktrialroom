from __future__ import annotations

import platform
from typing import Literal

import torch


def get_device(prefer: Literal["mps", "cpu", "auto"] = "auto") -> torch.device:
    """Return MPS on Apple Silicon when available; otherwise CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_mps(device: torch.device) -> bool:
    return device.type == "mps"
