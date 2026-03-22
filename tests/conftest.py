import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    arr = np.zeros((512, 384, 3), dtype=np.uint8)
    arr[:, :] = (200, 210, 220)
    return Image.fromarray(arr, mode="RGB")
