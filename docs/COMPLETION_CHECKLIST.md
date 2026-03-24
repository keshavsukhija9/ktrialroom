# SiliconVTON — completion checklist (run on your Mac)

Inference **must** run in **Terminal.app** on your machine (not Cursor’s agent). Unified memory and ~12 GB weights are too heavy for constrained sandboxes.

## 1. Dependencies & vendor

```bash
cd /path/to/resume_tryon
source .venv/bin/activate
pip install -r requirements.txt
test -d third_party/idm-vton || git clone --depth 1 https://github.com/yisol/IDM-VTON.git third_party/idm-vton
python tests/test_weights_loaded.py
```

## 2. PyTorch: `import torch` should finish in ~10–30s (not minutes)

If **`import torch` hangs for minutes** or errors with **missing / mmap failures on `libtorch_python.dylib`**, treat the install as **broken** (partial wheel, bad pip cache, or security software blocking `.so` mmap)—not “slow Mac.”

**Step A — force reinstall (same major line as `requirements.txt`):**

```bash
cd /path/to/resume_tryon
source .venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir "torch==2.4.0" "torchvision==0.19.0"
# torchaudio optional unless you use audio elsewhere:
# pip install --no-cache-dir torchaudio==2.4.0
```

**Step B — verify (should return in seconds):**

```bash
python -c "import torch; print('✅ torch:', torch.__version__)"
python -c "import torch; print('✅ MPS:', torch.backends.mps.is_available())"
```

**Step C — re-run the five-step import script (total usually well under a minute, no weights loaded):**

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py
```

**Step D — still broken, or `pip` itself is stuck?** If **`pip uninstall` or `pip install` runs many minutes** (normal is seconds to a few minutes for the full tree), assume **corrupted venv / pip metadata** — do not try to “repair” in place. Use **§2.1** (delete `.venv` and recreate).

**Step E — `pip` says torch is installed but `import torch` fails, or `which python` is not `.venv/bin/python`?** Use **§2.1b** (explicit interpreter paths, clear user pip cache, install **torch first**, then `requirements.txt`).

**If `import torch` is fast but inference fails on MPS:** use CPU for the diffusion device — set `device.backend` to `"cpu"` in `configs/model_config.yaml`, or run with `export PYTORCH_ENABLE_MPS_FALLBACK=1` (see README). `PYTORCH_MPS_HIGH_WATERMARK_RATIO` does **not** fix a broken `import torch`; it only tweaks MPS memory behavior during **runtime**.

### 2.1 Nuclear option: delete and recreate `.venv` (pip or venv corrupted)

**Signals:** `pip` hangs on trivial commands, `pip uninstall` takes **4+ minutes**, repeated `errno`/mmap/dlopen errors, or §2 Step A–C did not fix `import torch`. **Do not try to repair the old venv** — delete it and start clean.

#### Copy-paste one-liner (Mac Terminal)

```bash
cd /Users/keshavsukhija/Desktop/resume_tryon && deactivate 2>/dev/null || true && rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
```

*(If your repo lives elsewhere, change the `cd` path.)*

| Step | ~Time |
|------|--------|
| `deactivate` | 1 s |
| `rm -rf .venv` | ~5 s |
| `python3 -m venv .venv` | ~10 s |
| `pip install --upgrade pip` | ~30 s |
| `pip install -r requirements.txt` | **10–20 min** |
| `pip install -e .` | ~30 s |
| **Total** | **~15–25 min** |

**While `pip install` runs (~15–20 min), don’t stare at the terminal:**

- [ ] Update resume VTON bullets to match **`PRD.md` §3**
- [ ] Skim **`docs/INTERVIEW_PREP.md`**
- [ ] Confirm GitHub repo is visible (e.g. [github.com/keshavsukhija9/ktrialroom](https://github.com/keshavsukhija9/ktrialroom))

#### After installation — verify

Run the three checks in **§2.2** (torch, MPS, `debug_five_imports.py`).

#### If `pip install -r requirements.txt` hangs (>5 min)

Ctrl+C, then install in batches (still in activated `.venv`):

```bash
source .venv/bin/activate

pip install --no-cache-dir "torch>=2.4.0" "torchvision>=0.19.0"
pip install --no-cache-dir diffusers==0.25.1 transformers==4.36.2 accelerate==0.25.0
pip install --no-cache-dir mediapipe==0.10.14 opencv-python Pillow
pip install --no-cache-dir "gradio>=4.40.0,<5.0.0" pytest torchmetrics lpips
# Remaining pins from requirements.txt (Hub, utils, tests)
pip install --no-cache-dir "huggingface_hub>=0.19.0,<0.21" hf_transfer safetensors einops \
  "numpy>=1.26.0" "scipy>=1.11.0" "PyYAML>=6.0.1" "tqdm>=4.66.0" "psutil>=5.9.0"

pip install -e .
```

If anything is still missing, run `pip install -r requirements.txt` again once the network is stable.

### 2.1b Complete rebuild: torch missing after “successful” pip (explicit paths + torch-first)

**Signals:** `pip` / `pip list` claims **torch** is installed, but **`ModuleNotFoundError: No module named 'torch'`** when you run `python`; **`which python`** is **not** `…/resume_tryon/.venv/bin/python` (wrong interpreter / broken venv metadata). Also use this if **§2.1** did not fix imports.

**Python version:** This project needs **Python ≥ 3.10** (`pyproject.toml`). Check before creating the venv:

```bash
/usr/bin/python3 --version
```

If that is **&lt; 3.10**, do **not** use it — pick a newer interpreter (e.g. Homebrew `$(brew --prefix)/bin/python3`, or `python3.11` from pyenv) and substitute it everywhere below instead of `/usr/bin/python3`.

**Step 1 — remove venv and (optional) user pip cache**

```bash
cd /path/to/resume_tryon
deactivate 2>/dev/null || true
rm -rf .venv
ls -la .venv 2>&1 || echo "✅ .venv deleted"
rm -rf ~/Library/Caches/pip    # user-wide pip cache; clears corrupted wheels
```

**Step 2 — new venv with an explicit `python3`**

```bash
/usr/bin/python3 -m venv .venv    # or your ≥3.10 python path
ls -la .venv/bin/python
```

**Step 3 — activate and confirm the interpreter**

```bash
source /path/to/resume_tryon/.venv/bin/activate
which python
# MUST print: /path/to/resume_tryon/.venv/bin/python
```

If it prints anything else, **stop** — fix `PATH` / shell config so the venv’s `python` wins, or always use **`.venv/bin/python`** (below).

**Step 4 — upgrade `pip` using the venv binary**

```bash
.venv/bin/pip install --upgrade pip
```

**Step 5 — install only Torch + Torchvision, then test immediately**

```bash
cd /path/to/resume_tryon
.venv/bin/pip install --no-cache-dir torch torchvision
.venv/bin/python -c "import torch; print('✅ torch:', torch.__version__)"
```

If this **fails**, stop and capture the full error. If it **works**, continue.

**Step 6 — rest of dependencies + editable package**

```bash
cd /path/to/resume_tryon
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

**Step 7 — final checks (always use `.venv/bin/python` if unsure)**

```bash
cd /path/to/resume_tryon
.venv/bin/python -c "import torch; print('✅ torch:', torch.__version__)"
.venv/bin/python -c "import mediapipe; print('✅ mediapipe:', mediapipe.__version__)"
.venv/bin/python -c "import diffusers; print('✅ diffusers:', diffusers.__version__)"
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/debug_five_imports.py
```

**While installs run:** confirm the venv is growing (torch alone is large):

```bash
du -sh /path/to/resume_tryon/.venv
```

Repeat every minute or use `watch` if you have it (`brew install watch`).

**Copy-paste — torch-first through Step 5** (adjust `cd` if needed; **~10–15 min** for torch on first download):

```bash
cd /Users/keshavsukhija/Desktop/resume_tryon && deactivate 2>/dev/null || true && rm -rf .venv && rm -rf ~/Library/Caches/pip && /usr/bin/python3 -m venv .venv && source .venv/bin/activate && which python && .venv/bin/pip install --upgrade pip && .venv/bin/pip install --no-cache-dir torch torchvision && .venv/bin/python -c "import torch; print('✅ torch:', torch.__version__)"
```

When that prints a torch version, run **Step 6** and **Step 7** above.

**If torch install fails**

| Error | What to try |
|--------|-------------|
| No matching distribution | `.venv/bin/pip install --upgrade pip` then retry |
| `SSL: CERTIFICATE_VERIFY_FAILED` | `.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch torchvision` |
| Permission denied on `.venv` | `sudo chown -R "$(whoami)" .venv` (then avoid `sudo pip`) |
| No space left on device | `df -h` — need on the order of **~15 GB** free for torch + deps |

**Report back after Step 5**

1. Output of `which python`
2. Output of `.venv/bin/python -c "import torch; print('✅ torch:', torch.__version__)"`
3. Output of `du -sh .venv`

### 2.2 Environment ready — verify, then minimal inference

Typical successful install includes (versions may vary; `requirements.txt` uses **`torch>=2.4.0`** so pip may pick e.g. **2.11.x** on Apple Silicon):

- `torch` + `torchvision` (ARM64 macOS wheels)
- `diffusers==0.25.1`, `transformers==4.36.2`, `accelerate==0.25.0`
- `mediapipe==0.10.14`, `gradio` 4.x
- `siliconvton` editable (`pip install -e .`)

**Torch note:** Newer torch than 2.4.x is **fine** if **`import torch`** is fast and **`MPS: True`**. To force older pins: `pip install torch==2.4.0 torchvision==0.19.0 --force-reinstall`.

#### Three verification commands (in order)

```bash
cd /Users/keshavsukhija/Desktop/resume_tryon
source .venv/bin/activate

python -c "import torch; print('✅ torch:', torch.__version__)"
python -c "import torch; print('✅ MPS:', torch.backends.mps.is_available())"
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py
```

**One-liner:**

```bash
python -c "import torch; print('✅ torch:', torch.__version__)" && python -c "import torch; print('✅ MPS:', torch.backends.mps.is_available())" && PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py
```

#### Example success output (illustrative; seconds vary)

```text
✅ torch: 2.11.0
✅ MPS: True
SiliconVTON — 5 import tests (torch → pipeline init, no weight load in step 4)

=== Test 1: torch + MPS ===
  torch 2.11.0
  MPS available: True
✅ OK (2.1s)

=== Test 2: mediapipe ===
  mediapipe 0.10.14
✅ OK (0.6s)

=== Test 3: diffusers ===
  diffusers 0.25.1
✅ OK (0.4s)

=== Test 4: model loader ===
  import_tryon_modules() — vendor TryonPipeline + UNets (no disk weights yet)
✅ OK (1.1s)

=== Test 5: VTONPipeline init ===
  VTONPipeline constructed (lazy_load: weights on first generate)
✅ OK (3.2s)

========================================
✅ ALL 5 TESTS PASSED (total 7.4s)
========================================
Next: PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py
```

#### Report back (after the 3 verification commands)

1. Torch version: **\_\_\_**
2. MPS available: **YES / NO**
3. `debug_five_imports.py`: all 5 passed? **YES / NO**
4. Total time for all 5 tests: **\_\_ s**
5. *(If run)* `minimal_inference.py`: completed? **YES / NO**; output size: **\_\_ KB**

#### If all 5 pass — minimal inference (~3–5 min; loads weights on first run)

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py 2>&1 | tee inference_log.txt
# optional second tab: tail -f inference_log.txt
```

Expect `assets/outputs/minimal_test.png` (256×256, often ~50–100 KB). Then:

```bash
ls -lh assets/outputs/minimal_test.png
PYTHONPATH=. python scripts/verify_inference_output.py
```

#### What happens next (optional)

| Step | Command | ~Time |
|------|---------|--------|
| Verify imports | `debug_five_imports.py` | often &lt;1 min |
| Minimal inference | `minimal_inference.py` | ~3–5 min (first weight load) |
| Check output | `ls -lh assets/outputs/minimal_test.png` | seconds |
| Demo | `./scripts/record_demo.sh` | ~5 min |
| Final commit | `COMMIT_MSG="..." ./scripts/final_commit.sh` | ~1 min |

#### If a debug test fails

| Test | Likely fix |
|------|------------|
| 1 (torch) | `pip install --force-reinstall "torch>=2.4.0" "torchvision>=0.19.0"` or §2.1 fresh venv |
| 2 (mediapipe) | `pip install mediapipe==0.10.14` |
| 3 (diffusers) | `pip install diffusers==0.25.1` |
| 4 (model loader) | Clone vendor: `git clone --depth 1 https://github.com/yisol/IDM-VTON.git third_party/idm-vton` |
| 5 (pipeline init) | Fix **first** error traceback (configs, `opencv-python`, imports). **HF weights are not required** for init — they load on first `generate()`. |

#### Report back (fresh venv / pip timing)

1. `pip install -r requirements.txt` duration: **\_\_ min**
2. Torch version: **\_\_\_**
3. MPS: **YES / NO**
4. All 5 debug tests: **YES / NO**
5. Total debug time: **\_\_ s**

## 3. If the process shows ~75 MB RSS for minutes

It is probably stuck **before** weights (import / preprocessing), not in diffusion.

```bash
pkill -f minimal_inference || true
# Numbered tests 1–5 (correct API: import_tryon_modules with NO args — it does not load GB weights)
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/debug_five_imports.py
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/test_critical_imports.py   # optional: + resolve_model_id
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py 2>&1 | tee inference_log.txt
# other tab: tail -f inference_log.txt
```

Watch **`[1/9]` … `[9/9]`** lines in `minimal_inference.py` — the **last line printed** tells you the bottleneck.

## 4. Proof-of-concept inference (fastest)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # optional, if MPS glitches
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/minimal_inference.py
ls -lh assets/outputs/minimal_test.png
PYTHONPATH=. python scripts/verify_inference_output.py
```

## 5. Full validation (optional)

```bash
SILICONVTON_FULL_INFERENCE=1 PYTHONPATH=. python scripts/validate_siliconvton.py
PYTHONPATH=. python scripts/verify_inference_output.py
```

## 6. Demo & commit

```bash
./scripts/record_demo.sh
COMMIT_MSG="SiliconVTON: minimal inference verified" ALLOW_NO_DEMO=1 ./scripts/final_commit.sh
# With demo: omit ALLOW_NO_DEMO or answer y when prompted
```

**Docs-only / CI sync** (no PNG yet — not “interview complete”):

```bash
ALLOW_NO_INFERENCE=1 ALLOW_NO_DEMO=1 SKIP_PYTEST=1 COMMIT_MSG="docs: sync" ./scripts/final_commit.sh
```

## 7. Interview

Read `docs/INTERVIEW_PREP.md`.
