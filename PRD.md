# SiliconVTON — Product Requirements Document

**Version:** 2.0 (Mac Optimized)  
**Target hardware:** MacBook Air M4 (Unified Memory)  
**Core focus:** Inference optimization, memory management, pipeline integration  
**Development tool:** Cursor IDE

---

## 1. Problem Statement

High-quality Virtual Try-On models are typically cloud-dependent due to heavy VRAM requirements. This project engineers a **local-first VTON pipeline** optimized for Apple Silicon (M4), demonstrating how to run SOTA diffusion models (IDM-VTON) on consumer hardware using memory offloading and precision quantization.

---

## 2. Key Changes from Original Plan

| Original claim (risky on Mac) | Modified claim (feasible & strong) | Why |
|-------------------------------|-------------------------------------|-----|
| ONNX optimizations | Memory offloading & FP16 quantization | ONNX support for diffusion on Mac is buggy. Memory offloading (`cpu_offload`) is native to PyTorch MPS and works well. |
| Custom perceptual loss (training) | Perceptual quality evaluation (LPIPS/SSIM) | No training (requires A100-class resources). Evaluating quality using loss-style metrics is honest and sound. |
| Real-time rendering | Low-latency inference for consumer hardware | “Real-time” implies under 100 ms; diffusion takes seconds. “Low-latency” is accurate for FP16 vs FP32. |
| Heavy third-party pose/parsing stacks | MediaPipe + DeepLabV3 pipeline | Default stack uses **MediaPipe** pose and **DeepLabV3** torso masks; lighter RAM footprint on M-series Macs. |

---

## 3. Resume Bullets (copy-paste)

- Engineered a 1024×768 garment transfer pipeline using **IDM-VTON** with **pose and garment conditioning** (not generic ControlNet), optimized for Apple Silicon (MPS) for local deployment.
- Engineered lightweight preprocessing (**MediaPipe** pose + **DeepLabV3** segmentation), optimizing for consumer hardware; implemented **FP16** and **sequential CPU offloading**, reducing memory footprint by **~50%** on Apple Silicon M4 (measure and quote your own numbers).
- Integrated **LPIPS/SSIM** evaluation (not training loss) for structural and perceptual checks without retraining the base model.

*Note: Do **not** claim **ControlNet** or **TPS warping** for this repo. Do not name third-party pose/segmentation stacks that this codebase does not integrate — describe **MediaPipe**, **DeepLabV3**, and **letterboxed garment prep** only.*

---

## 4. Technical Architecture (M4-friendly)

### 4.1 Tech stack

| Layer | Choice |
|-------|--------|
| Framework | PyTorch 2.4+ (MPS) |
| Libraries | Hugging Face `diffusers`, `accelerate` (offloading) |
| Model | IDM-VTON (pre-trained weights) |
| Preprocessing | MediaPipe Pose + DeepLabV3 (segmentation) |
| UI | Gradio (local) |
| Evaluation | `torchmetrics` (SSIM, LPIPS) |

### 4.2 Optimization strategy (“40%” claim)

Compare:

- **Baseline:** PyTorch MPS, FP32, full model in memory.
- **Optimized:** PyTorch MPS, FP16, `enable_model_cpu_offload()`.

This comparison should show genuine speed/memory gains on Mac without relying on ONNX for diffusion.

---

## 5. Implementation Plan (Cursor)

### Phase 1: Setup & baseline (Days 1–3)

**Goal:** Run IDM-VTON on M4 without OOM.

**Cursor prompt (example):**  
On MacBook M4, set up PyTorch with `diffusers` to load `yisol/IDM-VTON`, move tensors to `mps`, implement basic inference (person + garment images), and handle OOM with batch size / dtype / `enable_model_cpu_offload()` as needed.

### Phase 2: Optimization (Days 4–7)

**Goal:** FP16 + offloading + benchmarks.

**Deliverable:** Script reporting e.g. FP32 time, FP16 time, memory saved (FP32 vs FP16 vs offloaded).

### Phase 3: Preprocessing & quality (Days 8–10)

**Goal:** MediaPipe pose + DeepLabV3 parsing; LPIPS/SSIM between input person and try-on output.

### Phase 4: UI & demo (Days 11–14)

**Goal:** Gradio: person + garment uploads, Generate, show result + inference time + SSIM/LPIPS.

---

## 6. Risk Management (M4)

| Risk | Mitigation |
|------|------------|
| Model too large (OOM) | `enable_sequential_cpu_offload()` or `enable_model_cpu_offload()`; unified memory helps. |
| MPS bugs | Fall back specific ops/layers to CPU; iterate with targeted fixes. |
| Slow inference | Accept absolute latency; emphasize FP16 vs FP32 relative gains. |

---

## 7. Interview Defense (concise)

- **Why not ONNX?** Diffusion UNets on Apple Silicon with ONNX can be unstable (dynamic shapes). PyTorch MPS + FP16 + offloading was chosen for stability and measurable gains.
- **Did you train perceptual loss?** No multi-GPU training. LPIPS/VGG-style metrics were used for **evaluation**, not fine-tuning.
- **Business value:** Shows path to edge/local deployment and lower inference cost vs cloud-only GPU reliance.

---

## 8. Success Checklist

- [ ] GitHub README with **Performance benchmarks** table (FP32 vs FP16).
- [ ] Short demo video (e.g. Gradio on Mac) for portfolio/LinkedIn.
- [ ] Type hints and docstrings on public APIs.
- [ ] Wording: **inference pipeline** for IDM-VTON — not “trained IDM-VTON.”

---

*This PRD keeps the project credible on M4 hardware and defensible in technical interviews.*
