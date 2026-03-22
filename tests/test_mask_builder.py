import numpy as np

from siliconvton.preprocessing.mask_builder import inpaint_mask_to_pil, torso_inpaint_region


def test_torso_inpaint_smaller_than_full():
    m = np.zeros((100, 80), dtype=bool)
    m[40:90, 20:60] = True
    t = torso_inpaint_region(m)
    assert t.sum() <= m.sum()
