from PIL import Image

from siliconvton.preprocessing.image_validator import ImageValidator


def test_letterbox_output_size(rgb_image):
    v = ImageValidator(768, 1024)
    out = v.letterbox(rgb_image)
    assert out.size == (768, 1024)


def test_validate_tiny_fails():
    v = ImageValidator(768, 1024, min_size=600)
    tiny = Image.new("RGB", (32, 32), color="white")
    ok, _ = v.validate(tiny)
    assert ok is False
