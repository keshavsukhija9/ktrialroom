from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import lpips as _lpips
except ImportError:  # pragma: no cover
    _lpips = None


class QualityMetrics:
    """SSIM + LPIPS for evaluation (not training)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self._lpips_model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def _get_lpips(self):
        if _lpips is None:
            raise ImportError("lpips package required for LPIPS")
        if self._lpips_model is None:
            self._lpips_model = _lpips.LPIPS(net="vgg").to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    @staticmethod
    def _ssim_tensor(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Simple SSIM on NCHW tensors in [0,1]."""
        c1, c2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, 11, 1, padding=5)
        mu2 = F.avg_pool2d(img2, 11, 1, padding=5)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu12 = mu1 * mu2
        sig1 = F.avg_pool2d(img1 * img1, 11, 1, padding=5) - mu1_sq
        sig2 = F.avg_pool2d(img2 * img2, 11, 1, padding=5) - mu2_sq
        sig12 = F.avg_pool2d(img1 * img2, 11, 1, padding=5) - mu12
        ssim = ((2 * mu12 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
        return float(ssim.mean().item())

    @torch.inference_mode()
    def calculate(self, original: Image.Image, generated: Image.Image) -> Dict[str, float]:
        o = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        g = self.transform(generated.convert("RGB")).unsqueeze(0).to(self.device)
        ssim = self._ssim_tensor(o, g)
        lp = self._get_lpips()(o * 2 - 1, g * 2 - 1).item()
        return {"ssim": ssim, "lpips": float(lp)}
