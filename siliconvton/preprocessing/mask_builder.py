from __future__ import annotations

import numpy as np
from PIL import Image


def torso_inpaint_region(person_mask: np.ndarray) -> np.ndarray:
    """
    Build bool mask of inpaint region (upper torso) from a person segmentation mask.
    """
    if person_mask.dtype != bool:
        person_mask = person_mask.astype(bool)
    ys, xs = np.where(person_mask)
    if len(ys) == 0:
        return np.zeros_like(person_mask, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = max(1, y1 - y0 + 1)
    y_end = y0 + int(0.72 * h)
    out = np.zeros_like(person_mask, dtype=bool)
    out[y0:y_end, :] = person_mask[y0:y_end, :]
    return out & person_mask


def inpaint_mask_to_pil(mask_inpaint: np.ndarray) -> Image.Image:
    """White (255) = repaint; black (0) = preserve (per diffusers inpaint convention)."""
    if mask_inpaint.dtype != bool:
        mask_inpaint = mask_inpaint.astype(bool)
    u8 = (mask_inpaint.astype(np.uint8)) * 255
    return Image.fromarray(u8, mode="L")
