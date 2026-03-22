from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class HumanSegmenter:
    """DeepLabV3 (torchvision) — binary mask for the 'person' VOC class."""

    PERSON_CLASS = 15  # Pascal VOC semantic labels used by torchvision COCO-with-VOC weights

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        self._weights_meta = weights.meta
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def get_segmentation_mask(self, image: Image.Image) -> np.ndarray:
        """Return HxW bool mask (True = person)."""
        img = image.convert("RGB")
        inp = self.transform(img).unsqueeze(0).to(self.device)
        out = self.model(inp)["out"][0]
        pred = out.argmax(0)
        mask = pred == self.PERSON_CLASS
        return mask.cpu().numpy().astype(bool)
