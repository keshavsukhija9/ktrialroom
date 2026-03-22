from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siliconvton.core.vton_pipeline import VTONPipeline
from siliconvton.utils.project_config import load_merged_config


def _build_pipeline(precision: str, steps: int, seed: int) -> VTONPipeline:
    cfg = load_merged_config(ROOT)
    cfg["optimization"]["precision"] = "fp16" if precision.startswith("FP16") else "fp32"
    cfg["inference"]["num_inference_steps"] = int(steps)
    cfg["inference"]["seed"] = int(seed)
    return VTONPipeline(cfg)


def run_tryon(
    person,
    garment,
    garment_desc: str,
    precision: str,
    steps: float,
    seed: float,
):
    if person is None or garment is None:
        raise gr.Error("Upload both person and garment images.")
    desc = (garment_desc or "").strip()
    if not desc:
        desc = str(load_merged_config(ROOT)["extras"].get("garment_description_default", "shirt"))

    pipe = _build_pipeline(precision, steps, seed)
    out = pipe(
        person.convert("RGB"),
        garment.convert("RGB"),
        desc,
        num_inference_steps=int(steps),
        seed=int(seed),
    )
    perf = out["performance"]
    m = out["metrics"]
    return (
        out["result_image"],
        out["aux"]["pose_image"],
        m["ssim"],
        m["lpips"],
        perf["inference_time"],
        perf["memory_usage"]["rss_mb"],
    )


def main():
    cfg0 = load_merged_config(ROOT)
    default_desc = cfg0["extras"].get("garment_description_default", "Short Sleeve Round Neck T-shirt")
    dev_label = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU"

    with gr.Blocks(title="SiliconVTON", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SiliconVTON — Virtual Try-On (IDM-VTON inference)\n"
            "**Inference-only** pipeline with FP16 + CPU offload on Apple Silicon. "
            "Not real-time; quality metrics are for evaluation, not training."
        )
        gr.Markdown(f"**Compute:** `{dev_label}`")

        with gr.Row():
            person_in = gr.Image(label="Person", type="pil", height=360)
            garment_in = gr.Image(label="Garment", type="pil", height=360)
        desc = gr.Textbox(label="Garment description", value=default_desc)

        with gr.Row():
            prec = gr.Radio(["FP16 (recommended)", "FP32 (baseline)"], value="FP16 (recommended)", label="Precision")
            steps = gr.Slider(15, 40, value=30, step=1, label="Denoising steps")
            seed = gr.Number(value=42, label="Seed", precision=0)

        go = gr.Button("Generate try-on", variant="primary")

        with gr.Row():
            out_img = gr.Image(label="Output", type="pil")
            pose_vis = gr.Image(label="Pose map (MediaPipe proxy)", type="pil")

        with gr.Row():
            ssim_v = gr.Number(label="SSIM (eval)")
            lpips_v = gr.Number(label="LPIPS (eval)")
            time_v = gr.Number(label="Inference time (s)")
            mem_v = gr.Number(label="RSS (MB)")

        go.click(
            fn=run_tryon,
            inputs=[person_in, garment_in, desc, prec, steps, seed],
            outputs=[out_img, pose_vis, ssim_v, lpips_v, time_v, mem_v],
        )

    demo.launch()


if __name__ == "__main__":
    main()
