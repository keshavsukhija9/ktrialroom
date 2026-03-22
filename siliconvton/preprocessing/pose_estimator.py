from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore


class PoseEstimator:
    """MediaPipe pose landmarks (33 keypoints)."""

    def __init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for PoseEstimator")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def extract_keypoints(self, image_bgr: np.ndarray) -> Dict[int, Dict[str, float]]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR image")
        res = self._pose.process(image_bgr)
        if res.pose_landmarks is None:
            raise ValueError("No pose detected in image")
        h, w = image_bgr.shape[:2]
        kps: Dict[int, Dict[str, float]] = {}
        for idx, lm in enumerate(res.pose_landmarks.landmark):
            kps[idx] = {
                "x": float(lm.x * w),
                "y": float(lm.y * h),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
        return kps

    def close(self) -> None:
        self._pose.close()
