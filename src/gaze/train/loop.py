"""Training loop. Supports SigLIP 2 / MobileCLIP2 / ResNet-50 backbones uniformly."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze.data.dataset import build_dataset
from gaze.models.gaze_model import GazeModel
from gaze.models.head import angular_error_deg, l2cs_loss
from gaze.train.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed) if hasattr(torch, "mps") else None


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, yaws, pitches = zip(*batch)
    return (
        torch.stack(list(imgs), dim=0),
        torch.tensor(list(yaws), dtype=torch.float32),
        torch.tensor(list(pitches), dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(model: GazeModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_err = 0.0
    total_yaw_l1 = 0.0
    total_pitch_l1 = 0.0
    n = 0
    for imgs, yaws, pitches in loader:
        imgs = imgs.to(device)
        yaws = yaws.to(device)
        pitches = pitches.to(device)
        out = model(imgs)
        err = angular_error_deg(out.yaw_deg, out.pitch_deg, yaws, pitches)
        total_err += err.sum().item()
        total_yaw_l1 += (out.yaw_deg - yaws).abs().sum().item()
        total_pitch_l1 += (out.pitch_deg - pitches).abs().sum().item()
        n += imgs.size(0)
    return {
        "angular_err_deg": total_err / max(n, 1),
        "yaw_l1_deg": total_yaw_l1 / max(n, 1),
        "pitch_l1_deg": total_pitch_l1 / max(n, 1),
        "n_samples": n,
    }


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"[train] device={device} backbone={cfg.backbone}")

    # ── Data
    train_ds = build_dataset(cfg.data, train=True)
    val_ds = build_dataset(cfg.data, train=False)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    # ── Model
    model = GazeModel(backbone_name=cfg.backbone).to(device)
    print(f"[train] params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Optim
    optim = AdamW(
        model.trainable_param_groups(backbone_lr=cfg.optim.backbone_lr, head_lr=cfg.optim.head_lr),
        weight_decay=cfg.optim.weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.epochs * steps_per_epoch
    sched = CosineAnnealingLR(optim, T_max=max(total_steps - cfg.optim.warmup_steps, 1))

    # ── Optional wandb
    use_wandb = bool(cfg.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.__dict__)

    # ── AMP (bfloat16 on M-series MPS, fp16 on CUDA)
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    use_amp = cfg.amp and device.type in {"cuda", "mps"}

    ckpt_dir = Path(cfg.ckpt_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_err = math.inf
    best_path = ckpt_dir / "best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for imgs, yaws, pitches in pbar:
            imgs = imgs.to(device, non_blocking=True)
            yaws = yaws.to(device, non_blocking=True)
            pitches = pitches.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(imgs)
                    loss, metrics = l2cs_loss(
                        out.l2cs,
                        yaws,
                        pitches,
                        ce_weight=cfg.loss.get("ce_weight", 1.0),
                        mse_weight=cfg.loss.get("mse_weight", 1.0),
                    )
            else:
                out = model(imgs)
                loss, metrics = l2cs_loss(
                    out.l2cs,
                    yaws,
                    pitches,
                    ce_weight=cfg.loss.get("ce_weight", 1.0),
                    mse_weight=cfg.loss.get("mse_weight", 1.0),
                )

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            if global_step >= cfg.optim.warmup_steps:
                sched.step()

            global_step += 1
            if global_step % cfg.log_interval == 0:
                pbar.set_postfix(loss=f"{metrics['total']:.3f}", ce=f"{metrics['ce']:.3f}", mse=f"{metrics['mse']:.3f}")
                if use_wandb:
                    wandb.log({**metrics, "step": global_step})

            if global_step % cfg.eval_interval == 0:
                eval_metrics = evaluate(model, val_loader, device)
                model.train()
                print(f"[eval @ step {global_step}] {eval_metrics}")
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in eval_metrics.items()} | {"step": global_step})
                if eval_metrics["angular_err_deg"] < best_err:
                    best_err = eval_metrics["angular_err_deg"]
                    torch.save(
                        {"model": model.state_dict(), "cfg": cfg.__dict__, "best_err": best_err},
                        best_path,
                    )
                    print(f"[ckpt] best {best_err:.2f}° → {best_path}")

        # Epoch-end full eval
        eval_metrics = evaluate(model, val_loader, device)
        print(f"[eval @ epoch {epoch + 1}] {eval_metrics}")
        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_metrics.items()} | {"epoch": epoch + 1})
        if eval_metrics["angular_err_deg"] < best_err:
            best_err = eval_metrics["angular_err_deg"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "best_err": best_err},
                best_path,
            )
            print(f"[ckpt] best {best_err:.2f}° → {best_path}")

    print(f"[done] best val angular error = {best_err:.2f}° at {best_path}")
    return best_path
