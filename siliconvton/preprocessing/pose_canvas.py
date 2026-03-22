from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# MediaPipe Pose connections (pairs of landmark indices) — upper-body focused subset
_POSE_LINES: List[Tuple[int, int]] = [
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
]


def keypoints_to_pose_image(
    keypoints: Dict[int, Dict[str, float]],
    width: int,
    height: int,
    line_thickness: int = 4,
    point_radius: int = 6,
) -> Image.Image:
    """
    Draw a simple RGB skeleton map (proxy for dense pose) at (width, height).
    Used as conditioning for IDM-VTON when full dense body maps are not available.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pts: Dict[int, Tuple[int, int]] = {}
    for idx, kp in keypoints.items():
        x = int(np.clip(kp["x"], 0, width - 1))
        y = int(np.clip(kp["y"], 0, height - 1))
        pts[idx] = (x, y)

    for a, b in _POSE_LINES:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 200), line_thickness, cv2.LINE_AA)

    for p in pts.values():
        cv2.circle(canvas, p, point_radius, (255, 120, 0), -1, cv2.LINE_AA)

    return Image.fromarray(canvas, mode="RGB")
