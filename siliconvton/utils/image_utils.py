from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
