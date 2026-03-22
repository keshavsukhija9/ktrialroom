# SiliconVTON Interview Preparation

## 30-second elevator pitch

"I built a production-oriented virtual try-on setup optimized for Apple Silicon. It runs IDM-VTON with FP16 and CPU offloading so diffusion fits consumer unified memory. I wired preprocessing with MediaPipe and DeepLabV3, added evaluation metrics and tests, and shipped a Gradio UI—focused on inference and deployment, not training."

## Technical deep-dive points

### 1. Why MediaPipe instead of heavier pose stacks?

Trade-off for RAM and dependencies. MediaPipe gives a usable pose image for conditioning with lower overhead on a laptop. For this project the goal was **reproducible local inference**, not matching every upstream demo pixel-for-pixel.

### 2. How did you optimize memory?

- FP16 for diffusion where supported  
- `enable_model_cpu_offload()` (and optional sequential offload) so not all weights live on MPS at once  

Measure your own peaks; quote numbers from **your** runs only.

### 3. Why pinned dependency versions?

Vendored IDM-VTON UNet code targets an older diffusers API (e.g. symbols that moved in newer releases). Pinning keeps the stack **reproducible** and avoids silent breakage.

### 4. What was the hardest part?

Integrating upstream code that was not a clean pip package: vendoring, path setup, matching `diffusers`/`transformers`/`accelerate`/`huggingface_hub` versions, and adding a validation script so changes stay testable.

### 5. Did you train the model?

No. Inference-only; weights are pre-trained. Contribution is **pipeline, optimization, testing, and UI** around those weights.

## What not to say

| Avoid | Prefer |
|-------|--------|
| "I trained IDM-VTON" | "I engineered inference and deployment" |
| "Real-time" (for full diffusion) | "Optimized inference; seconds on laptop" |
| Heavy third-party pose/seg products we did not ship | "MediaPipe + DeepLabV3 in this repo" |
| "ONNX is the main path" | "PyTorch MPS + diffusers; FP16 + offload" |
| "Works everywhere out of the box" | "Tuned for Apple Silicon; CUDA needs different tuning" |

## Code walkthrough order

1. `siliconvton/core/vton_pipeline.py` — orchestration  
2. `siliconvton/core/diffusion_engine.py` — Tryon pipeline + offload  
3. `siliconvton/preprocessing/pose_estimator.py` — MediaPipe  
4. `siliconvton/preprocessing/segmenter.py` — DeepLabV3  
5. `tests/test_instantiation.py` — smoke tests  
6. `ui/gradio_app.py` — UI entrypoints  

## Demo strategy

1. Prefer a **recorded** demo if live network or load is risky.  
2. Show **tests** and **validation script** to show rigor.  
3. Be honest about **latency** and **first-download** weight size.  

## Guardrails

- Do not claim model training.  
- Do not claim sub-100 ms end-to-end diffusion.  
- Align resume bullets with **MediaPipe + DeepLabV3** and this repo’s actual code.  
- Avoid running multiple huge GPU/MPS workloads simultaneously on unified memory.  
