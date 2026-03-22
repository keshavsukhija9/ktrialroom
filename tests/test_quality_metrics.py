import numpy as np
import pytest
import torch
from PIL import Image

from siliconvton.core.quality_metrics import QualityMetrics


@pytest.fixture
def identical_pair():
    arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    return im, im.copy()


def test_ssim_high_for_identical(identical_pair):
    a, b = identical_pair
    m = QualityMetrics(device=torch.device("cpu"))
    out = m.calculate(a, b)
    assert out["ssim"] > 0.99
    assert out["lpips"] < 0.05
