from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from siliconvton.utils.config_loader import load_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_merged_config(root: Path | None = None) -> Dict[str, Any]:
    root = root or repo_root()
    model_cfg = load_yaml(root / "configs/model_config.yaml")
    opt_cfg = load_yaml(root / "configs/optimization_config.yaml")
    inf_cfg = load_yaml(root / "configs/inference_config.yaml")
    return {
        "model": model_cfg["model"],
        "device": model_cfg.get("device", {}),
        "inference": model_cfg.get("inference", {}),
        "optimization": opt_cfg.get("optimization", {}),
        "benchmarking": opt_cfg.get("benchmarking", {}),
        "extras": inf_cfg,
    }
