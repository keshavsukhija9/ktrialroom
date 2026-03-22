"""Verify IDM-VTON weight files exist on disk (no torch; pytest optional)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Import pytest only when collected by pytest (avoids pygments/import issues in some environments)
try:
    import pytest
except ImportError:
    pytest = None  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _min_bytes(path: Path, minimum: int) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size >= minimum, f"too small ({path.stat().st_size} B): {path}"


def _model_dir() -> Path:
    cfg = yaml.safe_load((_repo_root() / "configs" / "model_config.yaml").read_text())
    name = cfg["model"]["name"]
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidate = (_repo_root() / name).resolve()
    if candidate.is_dir():
        return candidate
    if pytest:
        pytest.skip(f"Local weights not at {candidate}; config has {name!r}")
    print(f"SKIP: Local weights not at {candidate}; config has {name!r}", file=sys.stderr)
    sys.exit(0)


def test_idm_vton_weight_files_present():
    root = _model_dir()
    assert root.is_dir(), f"model folder missing: {root}"

    unet = root / "unet"
    w = list(unet.glob("diffusion_pytorch_model.*"))
    assert w, f"no diffusion weights under {unet}"
    _min_bytes(w[0].resolve(), 1_000_000_000)

    uenc = root / "unet_encoder"
    w2 = list(uenc.glob("diffusion_pytorch_model.*"))
    assert w2, f"no unet_encoder weights under {uenc}"
    _min_bytes(w2[0].resolve(), 1_000_000_000)

    vae = root / "vae"
    wv = list(vae.glob("diffusion_pytorch_model.*"))
    assert wv, f"no vae weights under {vae}"
    _min_bytes(wv[0].resolve(), 100_000_000)

    for sub, name in (
        ("text_encoder", "model.safetensors"),
        ("text_encoder_2", "model.safetensors"),
        ("image_encoder", "model.safetensors"),
    ):
        p = (root / sub / name).resolve()
        _min_bytes(p, 100_000_000)

    assert (root / "scheduler" / "scheduler_config.json").is_file()
    assert (root / "tokenizer" / "tokenizer_config.json").is_file()
    assert (root / "tokenizer_2" / "tokenizer_config.json").is_file()


if __name__ == "__main__":
    test_idm_vton_weight_files_present()
    print("test_weights_loaded: PASSED", file=sys.stderr)
