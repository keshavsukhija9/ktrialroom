import torch

from siliconvton.utils.device_utils import get_device, is_mps


def test_get_device_cpu_explicit():
    d = get_device("cpu")
    assert d.type == "cpu"


def test_is_mps_false_for_cpu():
    assert is_mps(torch.device("cpu")) is False
