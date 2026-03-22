"""
Component instantiation checks (device, preprocess, metrics, vendor imports, pipeline, UI).

Aligned with actual APIs: there is no VTONModelLoader or VTONInterface — use
``import_tryon_modules`` / ``load_merged_config`` and ``ui.gradio_app.run_tryon``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def test_device_detection():
    from siliconvton.utils.device_utils import get_device

    device = get_device("auto")
    assert device.type in ("mps", "cpu"), f"Unexpected device: {device}"


def test_memory_manager():
    from siliconvton.optimization.memory_manager import MemoryManager

    mm = MemoryManager()
    usage = mm.get_usage()
    assert "rss_mb" in usage
    assert usage["rss_mb"] > 0


def test_pose_estimator_on_sample():
    import cv2
    import numpy as np

    from siliconvton.preprocessing.pose_estimator import PoseEstimator

    sample = REPO / "assets" / "sample_inputs" / "person_1.jpg"
    if not sample.is_file():
        pytest.skip("assets/sample_inputs/person_1.jpg not found")

    bgr = cv2.imread(str(sample))
    assert bgr is not None
    pe = PoseEstimator()
    keypoints = pe.extract_keypoints(bgr)
    assert len(keypoints) == 33


def test_pose_estimator_random_image_may_fail():
    """Random noise rarely contains a detectable pose — expect ValueError."""
    import numpy as np

    from siliconvton.preprocessing.pose_estimator import PoseEstimator

    noise = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    pe = PoseEstimator()
    with pytest.raises(ValueError, match="No pose"):
        pe.extract_keypoints(noise)


def test_segmenter():
    from siliconvton.preprocessing.segmenter import HumanSegmenter

    test_img = Image.new("RGB", (512, 512), color="white")
    seg = HumanSegmenter(device=torch.device("cpu"))
    mask = seg.get_segmentation_mask(test_img)
    assert mask.shape == (512, 512)
    assert mask.dtype == bool


def test_quality_metrics():
    from siliconvton.core.quality_metrics import QualityMetrics

    img1 = Image.new("RGB", (256, 256), color="red")
    img2 = Image.new("RGB", (256, 256), color="blue")
    qm = QualityMetrics(device=torch.device("cpu"))
    metrics = qm.calculate(img1, img2)
    assert "ssim" in metrics and "lpips" in metrics


def test_vendor_tryon_modules_import():
    """Critical: upstream IDM-VTON TryonPipeline + hacked UNets must import."""
    from siliconvton.models.model_loader import import_tryon_modules

    TryonPipeline, UNetTryon, UNetEncoder = import_tryon_modules()
    assert TryonPipeline is not None
    assert UNetTryon is not None
    assert UNetEncoder is not None


def test_pipeline_initialization():
    from siliconvton.core.vton_pipeline import VTONPipeline
    from siliconvton.utils.project_config import load_merged_config, repo_root

    cfg = load_merged_config(repo_root())
    cfg["inference"]["num_inference_steps"] = 1
    pipeline = VTONPipeline(cfg)
    assert pipeline.diffusion_engine is not None
    assert pipeline.pose_estimator is not None


def test_ui_imports():
    import gradio as gr

    import ui.gradio_app as app

    assert hasattr(app, "run_tryon")
    assert hasattr(app, "main")
    assert gr is not None


if __name__ == "__main__":
    print("=" * 60)
    print("COMPONENT INSTANTIATION TESTS (run via pytest)")
    print("=" * 60)
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
