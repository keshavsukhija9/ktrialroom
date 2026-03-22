from __future__ import annotations

import cv2
import numpy as np


class GarmentWarper:
    """
    Lightweight alignment: letterbox garment to target size.
    Full TPS is optional; IDM-VTON uses garment via IP-Adapter style conditioning.
    """

    def __init__(self, width: int = 768, height: int = 1024) -> None:
        self.width = width
        self.height = height

    def prepare(self, garment_bgr: np.ndarray) -> np.ndarray:
        """Resize garment to HxW with letterboxing (white background)."""
        if garment_bgr.ndim != 3 or garment_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR image")
        h, w = garment_bgr.shape[:2]
        tw, th = self.width, self.height
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(garment_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized
        return canvas
