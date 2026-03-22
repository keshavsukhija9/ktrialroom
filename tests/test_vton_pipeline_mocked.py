import numpy as np
import pytest
import torch
from PIL import Image

from siliconvton.core.vton_pipeline import VTONPipeline


@pytest.fixture
def minimal_config():
    return {
        "model": {"name": "yisol/IDM-VTON"},
        "inference": {"width": 768, "height": 1024, "num_inference_steps": 1, "guidance_scale": 2.0, "seed": 0},
        "optimization": {"precision": "fp32", "enable_model_cpu_offload": False, "enable_sequential_cpu_offload": False},
    }


class _FakePose:
    def extract_keypoints(self, bgr):
        return {i: {"x": 384.0, "y": 400.0, "z": 0.0, "visibility": 1.0} for i in range(33)}


class _FakeSeg:
    def __init__(self, device=None):
        pass

    def get_segmentation_mask(self, image):
        return np.ones((1024, 768), dtype=bool)


class _FakeEngine:
    def __init__(self, *a, **k):
        pass

    def generate(self, **kwargs):
        return Image.new("RGB", (768, 1024), color=(10, 120, 200))


def test_pipeline_with_mocks(monkeypatch, minimal_config, rgb_image):
    garment = Image.fromarray(np.full((400, 300, 3), 128, dtype=np.uint8), mode="RGB")

    monkeypatch.setattr("siliconvton.core.vton_pipeline.PoseEstimator", _FakePose)
    monkeypatch.setattr("siliconvton.core.vton_pipeline.HumanSegmenter", _FakeSeg)
    monkeypatch.setattr("siliconvton.core.vton_pipeline.DiffusionEngine", _FakeEngine)

    pipe = VTONPipeline(minimal_config)
    out = pipe(rgb_image, garment, "test garment", enable_benchmark=False)
    assert out["result_image"].size == (768, 1024)
    assert "ssim" in out["metrics"]
