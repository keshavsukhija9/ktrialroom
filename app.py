from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Gradio 4.4x + gradio-client: JSON Schema allows `additionalProperties: true` (bool); get_type() assumes dict.
import gradio_client.utils as _gcu

_orig_jst = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _get_type_safe(schema: object):
    if not isinstance(schema, dict):
        return {}
    return _orig_get_type(schema)


def _json_schema_to_python_type_safe(schema: object, defs) -> str:
    if not isinstance(schema, dict):
        return "Any"
    return _orig_jst(schema, defs)


_gcu.get_type = _get_type_safe
_gcu._json_schema_to_python_type = _json_schema_to_python_type_safe

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siliconvton.core.vton_pipeline import VTONPipeline
from siliconvton.utils.project_config import load_merged_config, repo_root

OUTPUT_DIR = ROOT / "assets" / "outputs"

_pipeline: VTONPipeline | None = None


def _get_pipeline() -> VTONPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VTONPipeline(load_merged_config(repo_root()))
    return _pipeline


def run_tryon(
    person: Image.Image | None,
    garment: Image.Image | None,
    garment_description: str,
) -> tuple[Image.Image | None, str]:
    if person is None or garment is None:
        raise gr.Error("Please upload both a person photo and a garment photo.")

    desc = (garment_description or "").strip()
    if not desc:
        desc = str(
            load_merged_config(repo_root())["extras"].get(
                "garment_description_default", "Short Sleeve Round Neck T-shirt"
            )
        )

    person_rgb = person.convert("RGB")
    garment_rgb = garment.convert("RGB")

    out = _get_pipeline()(person_rgb, garment_rgb, desc)
    result: Image.Image = out["result_image"]
    inference_time = float(out["performance"]["inference_time"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUTPUT_DIR / f"tryon_{stamp}.png"
    result.save(path)

    timing = (
        f"Inference time: {inference_time:.3f} s · Saved to assets/outputs/{path.name}"
    )
    return result, timing


def main() -> None:
    defaults = load_merged_config(repo_root())
    default_desc = defaults["extras"].get(
        "garment_description_default", "Short Sleeve Round Neck T-shirt"
    )

    theme = gr.themes.Soft(
        primary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="SiliconVTON", theme=theme) as demo:
        gr.Markdown("# SiliconVTON")
        gr.Markdown("Upload images and run virtual try-on. First run loads model weights.")

        with gr.Row():
            person_in = gr.Image(label="Person Photo", type="pil", height=380)
            garment_in = gr.Image(label="Garment Photo", type="pil", height=380)

        desc_in = gr.Textbox(
            label="Garment Description",
            value=str(default_desc),
            lines=2,
        )

        go = gr.Button("Try On", variant="primary")

        out_img = gr.Image(label="Result", type="pil", height=420)
        timing_out = gr.Textbox(
            label="Timing",
            value="Inference time will appear here after a run.",
            interactive=False,
            lines=2,
        )

        go.click(
            fn=run_tryon,
            inputs=[person_in, garment_in, desc_in],
            outputs=[out_img, timing_out],
        )

    # show_api=False avoids a Gradio 4.44 / gradio-client schema bug with some outputs.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
    )


if __name__ == "__main__":
    main()
