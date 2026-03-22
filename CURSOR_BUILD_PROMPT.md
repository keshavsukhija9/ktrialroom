# SiliconVTON — Master Build Prompt for Cursor (Vibe Coder)

**How to use:** Copy everything inside the `---COPY BELOW---` … `---END COPY---` block into **Cursor Chat (Cmd+L)** or **Composer (Cmd+I)**. Re-run or append sections by phase if context limits require chunking.

---

## ---COPY BELOW---

```markdown
# Role
You are a senior ML engineer and release engineer. Build **SiliconVTON**: a **local-first**, **inference-only** Virtual Try-On system for a **B.Tech IT** internship portfolio (e.g. VIT Vellore). Code must be **GitHub-ready**, **honest**, and **test-backed**.

# Project root
Use the repository root as the project root (this workspace). If the folder is named `resume_tryon` or `silicon-vton`, keep paths relative to that root—**do not** hardcode absolute paths.

---

## NON-NEGOTIABLE: VERIFY AFTER EVERY COMPONENT (TESTS FIRST MENTALITY)

**Critical rule:** After **every** module, submodule, or feature you add, you MUST:

1. **Add or extend automated tests** (prefer `pytest`) that cover:
   - **Happy path** (valid inputs)
   - **At least one failure path** (invalid input, missing pose, wrong shape) where applicable
   - **Device-agnostic logic** on CPU where possible; use `@pytest.mark.skipif` for MPS-only smoke tests if CI has no MPS
2. **Run tests** (`pytest tests/ -q` or targeted file) and **fix failures** before moving on.
3. **Add a short “Verification” note** in the PR/commit description or module docstring: what was tested, what command was run.
4. **Smoke script** optional but encouraged: `python -m scripts.smoke_<module>` for manual one-shot checks on M4.

**Definition of done for any file in `src/`:** corresponding tests exist under `tests/` OR are explicitly justified (e.g. thin Gradio wrapper tested via integration test).

Do **not** accumulate untested code across phases.

---

## CRITICAL CONSTRAINTS (INTERVIEW-SAFE)

### Hardware
- Target: **MacBook Air M4**, **unified memory (~16 GB typical)**.
- **PyTorch `mps`** — **not** CUDA. All device logic must **fallback** to CPU.
- Expect **OOM**; design for **batch size 1**, **resolution caps**, **`enable_model_cpu_offload()`** / sequential offload.

### Technical honesty
- **DO:** Inference pipeline for **pre-trained IDM-VTON** (Hugging Face).
- **DO:** FP16 + **accelerate** CPU offloading; benchmark **FP32 vs FP16** (relative gains).
- **DO:** **SSIM** & **LPIPS** as **evaluation metrics** (not training losses).
- **DO NOT:** Claim **training** or **fine-tuning** IDM-VTON.
- **DO NOT:** Claim **“real-time”** or **sub-100 ms** end-to-end. Use **“low-latency vs baseline”** or **“optimized inference.”**
- **DO NOT:** Make **ONNX** the primary path. If mentioned: **experimental / optional** only.

### Resume alignment (implementation must support these claims honestly)
1. Garment transfer pipeline at **1024×768** (or configurable; document default) using **IDM-VTON**-class pipeline.
2. **Relative** memory reduction via **FP16** + **offloading** (measure; do not fabricate numbers—**README table from real runs**).
3. **LPIPS/SSIM** integrated; **pose/parsing** via **MediaPipe** + **DeepLabV3** (or documented fallback). Do **not** claim third-party pose/segmentation products in README or resume unless they are actually shipped—use “MediaPipe pose” and “DeepLabV3 segmentation.”

---

## AUTHORITATIVE REFERENCES IN REPO
- Read and follow **`PRD.md`** and **`SYSTEM_ARCHITECTURE.md`** in this repo. If code and docs diverge, **update docs** in the same PR.

---

## TECHNOLOGY STACK (PIN IN `requirements.txt`)

Use **compatible** versions; if resolution conflicts on M4, document the exact working set in README.

**Core**
- Python **3.10+**
- PyTorch **2.4+** with MPS
- `diffusers`, `transformers`, `accelerate`, `safetensors`

**Preprocessing**
- `opencv-python`, `mediapipe`, `pillow`, `numpy`
- Human parsing: **DeepLabV3** via `torchvision` hub **or** `segmentation_models_pytorch`—pick one and test it.

**Metrics**
- `torchmetrics` (SSIM), `lpips` (or equivalent for LPIPS)

**UI**
- `gradio` (primary). **Next.js** is **out of scope** unless explicitly requested later—do **not** scaffold Next.js by default (reduces scope creep).

**Dev / test**
- `pytest`, `pytest-cov` (optional), `ruff` or `black` (optional)

**Add:** `scipy` only if TPS/RBF warping is kept; prefer **documented, tested** warping. If warping is stubbed, **tests must assert stub behavior** and README must say “simplified alignment.”

---

## DIRECTORY STRUCTURE (CREATE AND MAINTAIN)

```
.
├── README.md
├── SYSTEM_ARCHITECTURE.md
├── PRD.md
├── requirements.txt
├── .gitignore
├── LICENSE                    # MIT if you add one
├── configs/
│   ├── model_config.yaml
│   ├── optimization_config.yaml
│   └── inference_config.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── vton_pipeline.py
│   │   ├── diffusion_engine.py    # Thin wrapper around official IDM-VTON pipeline
│   │   └── quality_metrics.py
│   ├── preprocessing/
│   │   ├── pose_estimator.py
│   │   ├── segmenter.py
│   │   ├── garment_warper.py
│   │   └── image_validator.py
│   ├── optimization/
│   │   ├── memory_manager.py      # Peak memory / timing helpers
│   │   ├── precision_handler.py
│   │   └── benchmark.py
│   ├── models/
│   │   └── model_loader.py        # HF loading + offload hooks
│   └── utils/
│       ├── logger.py
│       ├── image_utils.py
│       └── device_utils.py
├── ui/
│   └── gradio_app.py
├── benchmarks/
│   ├── fp32_vs_fp16.py
│   └── memory_profiler.py         # Best-effort on macOS
├── tests/
│   ├── conftest.py                # Fixtures: tiny RGB images, skip markers
│   ├── test_device_utils.py
│   ├── test_image_validator.py
│   ├── test_pose_estimator.py
│   ├── test_segmenter.py
│   ├── test_garment_warper.py
│   ├── test_quality_metrics.py
│   ├── test_diffusion_engine.py   # May skip if no weights in CI
│   └── test_vton_pipeline.py      # Integration; optional heavy skip
├── scripts/
│   └── smoke_inference.py
├── assets/
│   ├── sample_inputs/
│   └── outputs/
└── notebooks/
    └── exploration.ipynb          # Optional
```

**Omit** `api/FastAPI` unless you need it—Gradio is enough for the demo.

---

## CONFIG FILES (INITIAL CONTENT)

### `configs/model_config.yaml`
- `model_id`: Hugging Face repo for **IDM-VTON** (e.g. community ID—**verify** exact ID from current `diffusers` / model card).
- `revision`, `use_safetensors`
- `resolution`: height × width consistent with model (note: **IDM-VTON** may expect fixed sizes—**read model card** and encode in config).

### `configs/optimization_config.yaml`
- `precision`: `fp16` | `fp32`
- `enable_model_cpu_offload`, `enable_sequential_cpu_offload` (booleans)
- `benchmarking`: warmup runs, timed runs, `measure_memory: true`

### `configs/inference_config.yaml`
- `num_inference_steps`, `guidance` / model-specific knobs **as required by the real pipeline API**
- `seed`, `device`: `mps` with `cpu` fallback

---

## IMPLEMENTATION PHASES (WITH TEST GATES)

### PHASE 0 — Repo hygiene
**Deliverables:** `.gitignore` (Python, `__pycache__`, `.venv`, `*.pt`, `hf_cache/`, `outputs/`), `requirements.txt`, package layout, `pytest` runs empty suite.

**Verification:** `pytest` exits 0; `python -c "import src"` works.

---

### PHASE 1 — Device utilities + config loading
**Implement:** `src/utils/device_utils.py` — `get_device()` → MPS > CPU; no CUDA assumptions.

**Tests:** `tests/test_device_utils.py` — mock or skip MPS; assert CPU fallback on non-Darwin.

**Verification:** `pytest tests/test_device_utils.py -v`

---

### PHASE 2 — Image validator + preprocessing contracts
**Implement:** `image_validator.py` — RGB, min size, resize/pad to **target resolution** from config.

**Tests:** valid image; too small; non-RGB converted; output shape matches config.

**Verification:** `pytest tests/test_image_validator.py -v`

---

### PHASE 3 — Pose (MediaPipe)
**Implement:** `pose_estimator.py` — keypoints dict, clear error if no pose.

**Tests:** synthetic or tiny real image **in repo** under `assets/sample_inputs/` (add one **permissively licensed** or generated image); test raises on blank image.

**Verification:** `pytest tests/test_pose_estimator.py -v`

---

### PHASE 4 — Segmentation (DeepLabV3)
**Implement:** `segmenter.py` — binary or label mask; run on CPU in tests if faster.

**Tests:** output shape matches input; values in expected range; model `eval()` and `no_grad`.

**Verification:** `pytest tests/test_segmenter.py -v`

---

### PHASE 5 — Garment warping
**Implement:** `garment_warper.py` — if full TPS is too brittle, implement **minimal** alignment (e.g. resize/crop) and **document**; tests must match documented behavior.

**Tests:** known control points → expected array shape; no NaNs in map.

**Verification:** `pytest tests/test_garment_warper.py -v`

---

### PHASE 6 — Quality metrics
**Implement:** `quality_metrics.py` — SSIM + LPIPS **evaluation**; same spatial size handling (resize in metric fn if needed).

**Tests:** identical images → SSIM high, LPIPS low; random noise → worse scores.

**Verification:** `pytest tests/test_quality_metrics.py -v`

---

### PHASE 7 — Model loader + diffusion engine (CORE)

**Do not invent a minimal DDPM loop** unless you verify it matches **IDM-VTON**. Preferred approach:

1. Find the **official** or **community-documented** way to run **IDM-VTON** in `diffusers` (custom pipeline class or `DiffusionPipeline.from_pretrained` with correct custom modules).
2. Wrap in `DiffusionEngine` with:
   - `torch_dtype` fp16/fp32 from config
   - `pipe.enable_model_cpu_offload()` when requested
   - **MPS** device placement; document any op that must fall back to CPU

**Tests:**
- **Mock** HF download in CI if needed (`pytest` marker `heavy` for real download).
- **Smoke test** on developer machine: one inference with tiny resolution **if** model allows—else mark `skip` with reason.

**Verification:** `pytest tests/test_diffusion_engine.py -v` (may skip heavy); `python scripts/smoke_inference.py` works locally with weights cached.

---

### PHASE 8 — `VTONPipeline` orchestration
**Implement:** `vton_pipeline.py` — wires preprocessing → conditioning inputs required by **real** IDM-VTON call → decode → metrics.

**Tests:** integration test with **mocked** `DiffusionEngine.generate` returning a PIL image; assert metrics keys exist.

**Verification:** `pytest tests/test_vton_pipeline.py -v`

---

### PHASE 9 — Benchmarks
**Implement:** `benchmarks/fp32_vs_fp16.py` — prints **relative** speedup and **peak memory** (use `resource`, `psutil`, or Apple-specific notes; document limitations).

**Tests:** benchmark functions runnable with **mocked** short run.

**Verification:** script runs; README table filled with **real** numbers from your M4.

---

### PHASE 10 — Gradio UI
**Implement:** `ui/gradio_app.py` — person + garment uploads, run pipeline, show output, **inference time**, **SSIM/LPIPS**.

**Tests:** lightweight test that imports app and mocks pipeline callback if possible.

**Verification:** manual: `python -m ui.gradio_app` (or documented entry point); add **README** section **Performance benchmarks**.

---

## CODE QUALITY
- Type hints on public functions; concise docstrings.
- **No** misleading comments (“trained model”).
- Centralize **random seeds** for reproducibility in benchmarks.

---

## README SECTIONS (REQUIRED)
1. Honest **scope** (inference only).
2. **Install** + **Apple Silicon** notes.
3. **Performance benchmarks** table (FP32 vs FP16 + offload)—**measured**.
4. **Limitations** (not real-time, MPS ops, memory).

---

## GUARDRAILS CHECKLIST (BEFORE YOU FINISH)
- [ ] No claim of training IDM-VTON
- [ ] No “real-time” / sub-100 ms claims
- [ ] ONNX not primary
- [ ] Tests exist for each `src/` module
- [ ] At least one end-to-end mocked integration test
- [ ] Benchmarks reproducible from CLI

---

## IF BLOCKED
- **OOM:** reduce resolution, enable sequential offload, fp16.
- **MPS unsupported op:** isolate and move **that** op or tensor to CPU (document).
- **IDM-VTON API unclear:** read Hugging Face model card + `diffusers` issues; prefer **one** working minimal example over speculative UNet code.

---

# Your task
Implement the project **phase by phase** in order. After **each** phase, **run tests**, fix failures, then proceed. Keep **`SYSTEM_ARCHITECTURE.md`** and **`PRD.md`** aligned with behavior. Start with **Phase 0** and report test commands and results as you go.

```

## ---END COPY---

---

## Notes for you (not part of the Cursor prompt)

- **IDM-VTON** integration must follow the **actual** Hugging Face / `diffusers` API for that model. The draft snippets in some blog prompts (generic `UNet2DConditionModel` + `DDPMScheduler`) are **often wrong** for IDM-VTON; the master prompt above tells Cursor to **verify** against the model card.
- **TPS snippet bugs:** e.g. `tps_y` must map to `target_points[:, 1]`; RBF APIs differ—**tests** catch this.
- **Pose/segmentation:** This build targets **MediaPipe + DeepLabV3**; keep README and resume aligned (not extra heavy stacks unless added).

This file is the **single long structured prompt** to paste into Cursor; adjust phase chunking if the chat hits token limits (paste Phases 0–5, then 6–10).
