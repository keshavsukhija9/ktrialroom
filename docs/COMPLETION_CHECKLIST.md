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

## 2. Proof-of-concept inference (fastest)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # optional, if MPS glitches
PYTHONPATH=. python scripts/minimal_inference.py
ls -lh assets/outputs/minimal_test.png
PYTHONPATH=. python scripts/verify_inference_output.py
```

## 3. Full validation (optional)

```bash
SILICONVTON_FULL_INFERENCE=1 PYTHONPATH=. python scripts/validate_siliconvton.py
PYTHONPATH=. python scripts/verify_inference_output.py
```

## 4. Demo & commit

```bash
./scripts/record_demo.sh
COMMIT_MSG="SiliconVTON: minimal inference verified" ALLOW_NO_DEMO=1 ./scripts/final_commit.sh
# With demo: omit ALLOW_NO_DEMO or answer y when prompted
```

**Docs-only / CI sync** (no PNG yet — not “interview complete”):

```bash
ALLOW_NO_INFERENCE=1 ALLOW_NO_DEMO=1 SKIP_PYTEST=1 COMMIT_MSG="docs: sync" ./scripts/final_commit.sh
```

## 5. Interview

Read `docs/INTERVIEW_PREP.md`.
