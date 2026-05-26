"""Config dataclasses + YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimConfig:
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500


@dataclass
class TrainConfig:
    backbone: str = "siglip2_base_256"
    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 4
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = True
    log_interval: int = 50
    eval_interval: int = 500
    ckpt_dir: str = "checkpoints"
    run_name: str = "default"
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, float] = field(default_factory=lambda: {"ce_weight": 1.0, "mse_weight": 1.0})
    wandb_project: str | None = None


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    optim_raw = raw.pop("optim", {}) or {}
    return TrainConfig(optim=OptimConfig(**optim_raw), **raw)
