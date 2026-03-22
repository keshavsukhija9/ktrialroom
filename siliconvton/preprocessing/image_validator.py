from __future__ import annotations

from typing import Tuple

from PIL import Image


class ImageValidator:
    """Validate and letterbox/pad inputs to the pipeline resolution."""

    def __init__(self, target_width: int = 768, target_height: int = 1024, min_size: int = 256):
        self.target_width = target_width
        self.target_height = target_height
        self.min_size = min_size

    def validate(self, image: Image.Image) -> Tuple[bool, str]:
        img = image.convert("RGB")
        w, h = img.size
        if w < self.min_size or h < self.min_size:
            return False, f"Image too small: {w}x{h} (min {self.min_size})"
        return True, "ok"

    def letterbox(self, image: Image.Image) -> Image.Image:
        """Resize with aspect ratio preserved, pad to (target_width, target_height)."""
        image = image.convert("RGB")
        tw, th = self.target_width, self.target_height
        w, h = image.size
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (255, 255, 255))
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas.paste(resized, (ox, oy))
        return canvas
