from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


@contextmanager
def inference_autocast(device: torch.device, use_fp16: bool) -> Iterator[None]:
    """Autocast on CUDA; MPS skips autocast (torch <2.3 has no MPS autocast support)."""
    if not use_fp16 or device.type == "mps":
        with nullcontext():
            yield
    else:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            yield
