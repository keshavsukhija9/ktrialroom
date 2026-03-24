from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

from siliconvton.models.model_loader import load_tryon_pipeline
from siliconvton.optimization.precision_handler import inference_autocast
from siliconvton.utils.device_utils import get_device


class DiffusionEngine:
    """Wraps upstream IDM-VTON TryonPipeline with FP16 + optional CPU offload."""

    def __init__(self, config: Dict[str, Any], *, lazy_load: bool = True) -> None:
        self.config = config
        backend = str(config.get("device", {}).get("backend", "auto")).lower().strip()
        if backend == "cpu":
            self.device = get_device("cpu")
        elif backend == "mps":
            self.device = get_device("mps")
        else:
            self.device = get_device("auto")
        opt = config.get("optimization", {})
        self.use_fp16 = str(opt.get("precision", "fp16")).lower() == "fp16"
        self.enable_model_cpu_offload = bool(opt.get("enable_model_cpu_offload", True))
        self.enable_sequential_cpu_offload = bool(opt.get("enable_sequential_cpu_offload", False))

        self._pipe = None
        self._lazy = lazy_load

        self.tensor_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if not lazy_load:
            self._ensure_pipe()

    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return
        dtype = torch.float16 if self.use_fp16 else torch.float32
        model_id = self.config["model"]["name"]
        pipe, _ = load_tryon_pipeline(model_id, torch_dtype=dtype)

        if self.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        elif self.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)

        self._pipe = pipe

    def _pipe_device(self) -> torch.device:
        self._ensure_pipe()
        p = self._pipe
        assert p is not None
        d = getattr(p, "device", None)
        if isinstance(d, torch.device):
            return d
        try:
            return next(p.unet.parameters()).device
        except Exception:
            return self.device

    @property
    def pipe(self):
        self._ensure_pipe()
        assert self._pipe is not None
        return self._pipe

    @torch.inference_mode()
    def generate(
        self,
        *,
        person_image: Image.Image,
        garment_image: Image.Image,
        pose_image: Image.Image,
        mask_image: Image.Image,
        garment_description: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        inf = self.config.get("inference", {})
        steps = int(num_inference_steps or inf.get("num_inference_steps", 30))
        g = float(guidance_scale or inf.get("guidance_scale", 2.0))
        seed_val = int(seed if seed is not None else inf.get("seed", 42))

        w, h = int(inf.get("width", 768)), int(inf.get("height", 1024))
        person_image = person_image.convert("RGB").resize((w, h))
        garment_image = garment_image.convert("RGB").resize((w, h))
        pose_image = pose_image.convert("RGB").resize((w, h))
        mask_image = mask_image.convert("L").resize((w, h))

        pipe = self.pipe
        device = self._pipe_device()
        dtype = torch.float16 if self.use_fp16 else torch.float32

        prompt = "model is wearing " + garment_description.strip()
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        prompt_c = "a photo of " + garment_description.strip()
        (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
            prompt_c,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
        )

        pose_t = self.tensor_rgb(pose_image).unsqueeze(0).to(device, dtype=dtype)
        garm_t = self.tensor_rgb(garment_image).unsqueeze(0).to(device, dtype=dtype)

        gen_dev = device
        if device.type == "mps":
            gen_dev = torch.device("mps")
        elif device.type == "cuda":
            gen_dev = torch.device("cuda")
        else:
            gen_dev = torch.device("cpu")
        generator = torch.Generator(device=gen_dev).manual_seed(seed_val)

        def _run():
            return pipe(
                prompt_embeds=prompt_embeds.to(device, dtype=dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(device, dtype=dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype=dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype=dtype),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_t,
                text_embeds_cloth=prompt_embeds_c.to(device, dtype=dtype),
                cloth=garm_t,
                mask_image=mask_image,
                image=person_image,
                height=h,
                width=w,
                ip_adapter_image=garment_image,
                guidance_scale=g,
            )[0]

        with inference_autocast(device, self.use_fp16 and device.type in ("mps", "cuda")):
            images = _run()

        return images[0]
