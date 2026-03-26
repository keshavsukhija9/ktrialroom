from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from siliconvton.core.diffusion_engine import DiffusionEngine
from siliconvton.core.quality_metrics import QualityMetrics
from siliconvton.optimization.memory_manager import MemoryManager
from siliconvton.preprocessing.garment_warper import GarmentWarper
from siliconvton.preprocessing.image_validator import ImageValidator
from siliconvton.preprocessing.mask_builder import inpaint_mask_to_pil, torso_inpaint_region
from siliconvton.preprocessing.pose_canvas import keypoints_to_pose_image
from siliconvton.preprocessing.pose_estimator import PoseEstimator
from siliconvton.preprocessing.segmenter import HumanSegmenter


class VTONPipeline:
    """Orchestrates preprocessing + IDM-VTON inference + evaluation metrics."""

    def _pose_fallback_keypoints(self) -> Dict[int, Dict[str, float]]:
        """Deterministic dummy keypoints (for smoke tests only)."""
        cx = float(self._w) / 2.0
        cy = float(self._h) / 2.0
        return {i: {"x": cx, "y": cy, "z": 0.0, "visibility": 0.0} for i in range(33)}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        inf = config.get("inference", {})
        w, h = int(inf.get("width", 768)), int(inf.get("height", 1024))
        self._w, self._h = w, h

        self.validator = ImageValidator(target_width=w, target_height=h)
        self.pose_estimator = PoseEstimator()
        self.segmenter = HumanSegmenter(device=torch.device("cpu"))
        self.garment_warper = GarmentWarper(width=w, height=h)
        self.diffusion_engine = DiffusionEngine(config, lazy_load=True)
        self.quality_metrics = QualityMetrics(device=torch.device("cpu"))
        self.memory_manager = MemoryManager()

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str,
        *,
        enable_benchmark: bool = True,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        ok, msg = self.validator.validate(person_image)
        if not ok:
            raise ValueError(msg)
        ok2, msg2 = self.validator.validate(garment_image)
        if not ok2:
            raise ValueError(msg2)

        t0 = time.perf_counter()
        person_v = self.validator.letterbox(person_image)
        garment_v = self.validator.letterbox(garment_image)

        bgr = cv2.cvtColor(np.asarray(person_v), cv2.COLOR_RGB2BGR)
        try:
            keypoints = self.pose_estimator.extract_keypoints(bgr)
        except ValueError as e:
            # MediaPipe can fail at tiny resolutions or edge-case inputs.
            # Allow a fallback skeleton for smoke tests when explicitly enabled.
            if os.environ.get("SILICONVTON_POSE_FALLBACK", "0") == "1" and "No pose" in str(e):
                keypoints = self._pose_fallback_keypoints()
            else:
                raise

        mask_bool = self.segmenter.get_segmentation_mask(person_v)
        torso = torso_inpaint_region(mask_bool)
        mask_pil = inpaint_mask_to_pil(torso)

        pose_pil = keypoints_to_pose_image(keypoints, self._w, self._h)

        g_bgr = cv2.cvtColor(np.asarray(garment_v), cv2.COLOR_RGB2BGR)
        g_prep = self.garment_warper.prepare(g_bgr)
        garment_rgb = Image.fromarray(cv2.cvtColor(g_prep, cv2.COLOR_BGR2RGB))

        t_inf0 = time.perf_counter()
        result_image = self.diffusion_engine.generate(
            person_image=person_v,
            garment_image=garment_rgb,
            pose_image=pose_pil,
            mask_image=mask_pil,
            garment_description=garment_description,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        inference_time = time.perf_counter() - t_inf0

        metrics = self.quality_metrics.calculate(person_v, result_image)
        total_time = time.perf_counter() - t0

        out: Dict[str, Any] = {
            "result_image": result_image,
            "metrics": metrics,
            "performance": {
                "total_time": total_time,
                "inference_time": inference_time,
                "memory_usage": self.memory_manager.get_usage(),
            },
            "aux": {
                "pose_image": pose_pil,
                "mask_image": mask_pil,
            },
        }
        if enable_benchmark:
            out["benchmark"] = out["performance"]
        return out
