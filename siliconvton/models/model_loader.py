"""
Load IDM-VTON components from Hugging Face.

Uses the official hacked UNet classes from `third_party/idm-vton` (not stock diffusers UNet2DConditionModel).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

def idm_vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "idm-vton"


def ensure_idm_on_path() -> Path:
    root = idm_vendor_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing IDM-VTON vendor at {root}. Clone with: "
            "git clone https://github.com/yisol/IDM-VTON.git third_party/idm-vton"
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


_DEFAULT_LOCAL_WEIGHTS = "models/IDM-VTON"
_HUB_ID = "yisol/IDM-VTON"


def resolve_model_id(model_id: str) -> str:
    """Prefer a local folder (repo-relative or absolute); fall back to Hub if default path missing."""
    from siliconvton.utils.project_config import repo_root

    p = Path(model_id).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    candidate = (repo_root() / model_id).resolve()
    if candidate.is_dir():
        return str(candidate)
    if model_id.replace("\\", "/") == _DEFAULT_LOCAL_WEIGHTS:
        return _HUB_ID
    return model_id


def import_tryon_modules():
    ensure_idm_on_path()
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetEncoder
    from src.unet_hacked_tryon import UNet2DConditionModel as UNetTryon
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

    return TryonPipeline, UNetTryon, UNetEncoder


def load_tryon_pipeline(
    model_id: str,
    *,
    torch_dtype: torch.dtype,
) -> Tuple[Any, dict]:
    """Build TryonPipeline + return component handles for optional offload hooks."""
    TryonPipeline, UNetTryon, UNetEncoder = import_tryon_modules()
    model_id = resolve_model_id(model_id)

    unet = UNetTryon.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)
    unet.requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch_dtype
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)
    unet_encoder = UNetEncoder.from_pretrained(model_id, subfolder="unet_encoder", torch_dtype=torch_dtype)

    for m in (unet_encoder, vae, text_encoder_one, text_encoder_two, image_encoder, unet):
        m.requires_grad_(False)

    pipe = TryonPipeline.from_pretrained(
        model_id,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.unet_encoder = unet_encoder

    meta = {"dtype": torch_dtype}
    return pipe, meta
