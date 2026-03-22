# SiliconVTON

Local-first **virtual try-on** inference using the pre-trained **IDM-VTON** weights (`yisol/IDM-VTON`) on **Apple Silicon (PyTorch MPS)** with **FP16** and **CPU offloading**. This repository is **inference-only** (no model training).

## Highlights

- **1024×768** try-on pipeline (configurable in `configs/model_config.yaml`)
- **PyTorch MPS** backend with **CPU** fallback
- **FP16** + `enable_model_cpu_offload()` for memory pressure on consumer Macs
- **SSIM / LPIPS** for **evaluation** (not training loss)
- **MediaPipe** pose → **RGB pose map** (lightweight alternative to dense body maps used in some upstream demos)
- **DeepLabV3** (torchvision) → torso **inpaint mask**
- **Gradio** UI: `ui/gradio_app.py`

## Honest scope

- **Not** “real-time” (sub-100 ms) — diffusion is **seconds** on laptop hardware.
- **Not** ONNX-first — the path is **PyTorch + diffusers + upstream IDM** code.
- **Not** training / fine-tuning IDM-VTON — weights are **pre-trained**; this repo **engineers inference** and **benchmarks**.
- **Pose:** Some upstream demos use heavier pose/parsing stacks; this project uses a **lighter** MediaPipe skeleton + pose image for portability on M4.

## Project layout

```
configs/          YAML for model, optimization, UI defaults
siliconvton/      Python package (named to avoid clashing with upstream `src/`)
third_party/idm-vton/   Vendored upstream IDM-VTON (TryonPipeline + hacked UNets)
ui/               Gradio demo
assets/sample_inputs/   Example person + garment
tests/            pytest
benchmarks/     FP32 vs FP16 script (run locally; fill README table with real numbers)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Dependency notes (important):**

- **`diffusers==0.25.1`** — Vendored IDM-VTON UNet imports symbols removed in newer diffusers (e.g. `PositionNet`). Do not upgrade diffusers casually.
- **`huggingface_hub`** — Kept in the `0.19.x–0.20.x` range compatible with that diffusers line and **Gradio 4.x** (`gradio>=4.40,<5`).
- **`mediapipe==0.10.14`** — Some newer macOS wheels omit `mediapipe.solutions` (pose). This pin keeps the classic pose API.
- **`einops`** — Required by upstream `third_party/idm-vton` UNets.

## QA validation (before demos / interviews)

```bash
PYTHONPATH=. python scripts/validate_siliconvton.py
PYTHONPATH=. pytest tests/ -q
```

**Weights on disk** (fast; checks `configs/model_config.yaml` → `models/IDM-VTON`):

```bash
python tests/test_weights_loaded.py
```

Optional end-to-end inference (downloads **multi-GB** HF weights; slow):

```bash
SILICONVTON_FULL_INFERENCE=1 PYTHONPATH=. python scripts/validate_siliconvton.py
```

Run this in **Terminal.app** on your Mac (not Cursor’s agent): full diffusion needs real unified memory; **exit code 138** in constrained environments usually means the process was stopped under memory pressure—not necessarily a code bug.

**Tighter memory (proof-of-concept):** 256×256, 1 step, sequential offload — still loads full weights but smaller activations:

```bash
PYTHONPATH=. python scripts/minimal_inference.py
```

Saves `assets/outputs/minimal_test.png`. If full validation fails, lower resolution in `configs/inference_config.yaml` or set `enable_sequential_cpu_offload: true` in `configs/optimization_config.yaml`.

Ensure `third_party/idm-vton` exists (clone if missing):

```bash
git clone --depth 1 https://github.com/yisol/IDM-VTON.git third_party/idm-vton
```

Weights download from Hugging Face on first run (~several GB).

**Resumable download (recommended entry point):** `scripts/download_idm_weights.sh` only sets `HF_HUB_ENABLE_HF_TRANSFER=1` if `hf_transfer` is importable, so you never hit the “package not available” error from an orphan env var.

```bash
bash scripts/download_idm_weights.sh
```

**Optional: faster downloads with `hf_transfer`** (parallel Rust-based transfers; see [HF docs](https://huggingface.co/docs/huggingface_hub/hf_transfer)):

```bash
source .venv/bin/activate
pip install -U "huggingface_hub>=0.19,<0.21" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download yisol/IDM-VTON \
  --local-dir models/IDM-VTON \
  --resume-download
```

Do **not** export `HF_HUB_ENABLE_HF_TRANSFER=1` unless `hf_transfer` is installed — the CLI raises `ValueError` and does not fall back. **Quick check** (run in a second terminal; leave the download tab alone):

```bash
source .venv/bin/activate
python -c "import hf_transfer; print('hf_transfer OK')" 2>/dev/null || echo "hf_transfer not installed — use script above or omit HF_HUB_ENABLE_HF_TRANSFER"
```

**If you already see the error:** stop the process (Ctrl+C), `unset HF_HUB_ENABLE_HF_TRANSFER`, then resume with the default downloader (slower but reliable; still resumable):

```bash
cd /path/to/resume_tryon
source .venv/bin/activate
huggingface-cli download yisol/IDM-VTON \
  --local-dir models/IDM-VTON \
  --resume-download
```

`configs/model_config.yaml` points `model.name` at `models/IDM-VTON` (resolved from the repo root). To use the Hub directly again, set `model.name` to `yisol/IDM-VTON`.

## Run Gradio

```bash
PYTHONPATH=. python ui/gradio_app.py
```

Open the printed local URL (default `http://127.0.0.1:7860`).

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

After full inference, verify the output image:

```bash
PYTHONPATH=. python scripts/verify_inference_output.py
```

## Benchmark results (M4 — fill in from your runs)

| Metric | Value | Notes |
|--------|-------|--------|
| Minimal inference (256², 1 step) | *TBD s* | `python scripts/minimal_inference.py` (proof-of-concept) |
| Inference time (2 steps validation) | *TBD s* | FP16 + offload (`validate_siliconvton.py`) |
| Peak memory | *TBD GB* | Unified memory; Activity Monitor / `ps` |
| SSIM | *TBD* | From pipeline output dict when enabled |
| LPIPS | *TBD* | From pipeline output dict when enabled |
| Model weights (disk) | ~12 GB | HF cache + `models/IDM-VTON` |
| Preprocessing | *TBD s* | MediaPipe + DeepLabV3 |

## Verification status

| Check | Status | Date |
|-------|--------|------|
| Unit tests | run locally | *YYYY-MM-DD* |
| Weights on disk | `python tests/test_weights_loaded.py` | *YYYY-MM-DD* |
| Full or minimal inference | `validate_siliconvton.py` and/or `minimal_inference.py` | *YYYY-MM-DD* |
| MPS | when available on Apple Silicon | *YYYY-MM-DD* |
| Demo video | `assets/demo_backup.mp4` (manual) | *YYYY-MM-DD* |

## Demo

Record with QuickTime/OBS while running `python ui/gradio_app.py`; save as `assets/demo_backup.mp4`. For GitHub, link or attach the file from your release/assets (large binaries are often gitignored).

## Known limitations

- Uses **MediaPipe** and **DeepLabV3** for lightweight preprocessing—not the heaviest upstream pose/segmentation stacks.
- First run downloads large weights (~12 GB class); cached afterward.
- Optimized for **Apple Silicon** (MPS); other devices need validation.
- Inference time varies with resolution, step count, and thermal state.

## Performance table (fill in on your M4)

| Mode | Avg inference (s) | Peak RSS (MB) | Notes |
|------|-------------------|---------------|--------|
| FP32 | *measure* | *measure* | Baseline |
| FP16 + offload | *measure* | *measure* | Optimized |

Use `benchmarks/fp32_vs_fp16.py` after caching weights.

## Docs

- `SYSTEM_ARCHITECTURE.md` — architecture
- `PRD.md` — product scope
- `docs/INTERVIEW_PREP.md` — talking points / guardrails
- `docs/COMPLETION_CHECKLIST.md` — **finish inference + demo on your Mac (Terminal)**

To declare the project “done” for interviews, complete at least **minimal inference** and fill benchmark rows in this README using numbers from your machine.

## Licenses

- This scaffolding: **MIT** (`LICENSE`)
- **IDM-VTON** weights & upstream code: **CC BY-NC-SA 4.0** (see `third_party/idm-vton` and Hugging Face model card)
