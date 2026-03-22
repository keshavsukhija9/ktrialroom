from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def inference_autocast(device: torch.device, use_fp16: bool) -> Iterator[None]:
    """Autocast only on backends where it is typically stable (MPS/CUDA)."""
    if use_fp16 and device.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.float16):
            yield
    elif use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield
