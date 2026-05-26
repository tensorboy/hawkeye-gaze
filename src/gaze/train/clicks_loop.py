"""Training loop for click-as-label end-to-end screen-coord regression.

Uses ScreenGazeModel (backbone + ScreenCoordHead) on ClickLabelDataset. Loss is
Huber on normalized (x, y); we additionally monitor mean pixel error so you can
gauge real-world accuracy on the user's display.
"""

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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from tqdm import tqdm

from gaze.data.dataset import build_dataset
from gaze.models.gaze_model import ScreenGazeModel
from gaze.models.head import screen_coord_loss, screen_pixel_error
from gaze.train.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, xs, ys, sws, shs = zip(*batch)
    return (
        torch.stack(list(imgs), dim=0),
        torch.tensor(list(xs), dtype=torch.float32),
        torch.tensor(list(ys), dtype=torch.float32),
        torch.tensor([(sw, sh) for sw, sh in zip(sws, shs)], dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(model: ScreenGazeModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_pixel_err = 0.0
    total_l2_norm = 0.0
    n = 0
    for imgs, xs, ys, screen_wh in loader:
        imgs = imgs.to(device)
        xs = xs.to(device)
        ys = ys.to(device)
        target = torch.stack([xs, ys], dim=-1)
        out = model(imgs)
        delta = (out.screen.xy_norm - target) * screen_wh.to(device)
        pix = delta.pow(2).sum(dim=-1).sqrt()
        total_pixel_err += pix.sum().item()
        total_l2_norm += (out.screen.xy_norm - target).pow(2).sum(dim=-1).sqrt().sum().item()
        n += imgs.size(0)
    return {
        "pixel_err_mean": total_pixel_err / max(n, 1),
        "l2_norm_mean": total_l2_norm / max(n, 1),
        "n_samples": n,
    }


def split_dataset(ds: torch.utils.data.Dataset, val_fraction: float = 0.15, seed: int = 42):
    n = len(ds)
    val_n = max(1, int(n * val_fraction))
    train_n = n - val_n
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=gen)
    return train_ds, val_ds


def compute_stratified_weights(
    train_subset: torch.utils.data.Subset,
    grid_size: int = 10,
) -> torch.Tensor:
    """Inverse-density weights for WeightedRandomSampler.

    Walks the training subset's underlying ClickRecord list to find each
    sample's (x_norm, y_norm) → grid zone, counts zone populations, and
    assigns weight = 1 / count[zone] to each sample. Samples in rare zones
    get sampled more often per epoch, balancing the batch composition
    without throwing data away.
    """
    base = train_subset.dataset  # ClickLabelDataset
    indices = list(train_subset.indices)
    zones: list[tuple[int, int]] = []
    for idx in indices:
        rec = base.records[idx]
        gx = min(grid_size - 1, max(0, int(rec.click_x / rec.screen_w * grid_size)))
        gy = min(grid_size - 1, max(0, int(rec.click_y / rec.screen_h * grid_size)))
        zones.append((gx, gy))
    counts: dict[tuple[int, int], int] = {}
    for z in zones:
        counts[z] = counts.get(z, 0) + 1
    weights = torch.tensor([1.0 / counts[z] for z in zones], dtype=torch.double)
    # Normalize so the mean weight is 1 (cosmetic; sampler only cares about ratios)
    weights = weights * (len(weights) / weights.sum())
    n_unique = len(counts)
    print(f"[stratified] {len(weights)} train samples across {n_unique} occupied zones (of {grid_size**2})")
    print(f"[stratified] weight range [{weights.min():.3f}, {weights.max():.3f}], ratio {weights.max() / weights.min():.1f}×")
    return weights


def train_clicks(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"[train-clicks] device={device} backbone={cfg.backbone}")

    # ── Data
    full_ds = build_dataset(cfg.data | {"type": "clicks"}, train=True)
    print(f"[train-clicks] total click samples: {len(full_ds)}")
    train_ds, val_ds = split_dataset(full_ds, val_fraction=cfg.data.get("val_fraction", 0.15), seed=cfg.seed)
    print(f"[train-clicks] train={len(train_ds)} val={len(val_ds)}")

    # Stratified sampler if enabled (default): each batch becomes spatially balanced
    use_strat = cfg.data.get("stratified_sampling", True)
    sampler = None
    shuffle = True
    if use_strat:
        weights = compute_stratified_weights(train_ds, grid_size=cfg.data.get("grid_size", 10))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # mutually exclusive with sampler in DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        drop_last=False,
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
    model = ScreenGazeModel(backbone_name=cfg.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train-clicks] params={n_params:.1f}M")

    # ── Optim
    optim = AdamW(
        model.trainable_param_groups(backbone_lr=cfg.optim.backbone_lr, head_lr=cfg.optim.head_lr),
        weight_decay=cfg.optim.weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.epochs * steps_per_epoch
    sched = CosineAnnealingLR(optim, T_max=max(total_steps - cfg.optim.warmup_steps, 1))

    use_wandb = bool(cfg.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.__dict__)

    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    use_amp = cfg.amp and device.type in {"cuda", "mps"}

    ckpt_dir = Path(cfg.ckpt_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_pixel_err = math.inf
    best_path = ckpt_dir / "best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for imgs, xs, ys, screen_wh in pbar:
            imgs = imgs.to(device, non_blocking=True)
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            target = torch.stack([xs, ys], dim=-1)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(imgs)
                    loss, metrics = screen_coord_loss(out.screen, target)
            else:
                out = model(imgs)
                loss, metrics = screen_coord_loss(out.screen, target)

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            if global_step >= cfg.optim.warmup_steps:
                sched.step()

            global_step += 1
            if global_step % cfg.log_interval == 0:
                pbar.set_postfix(
                    huber=f"{metrics['huber']:.4f}",
                    l2_p50=f"{metrics['l2_p50']:.3f}",
                )
                if use_wandb:
                    wandb.log({**metrics, "step": global_step})

            if global_step % cfg.eval_interval == 0:
                eval_metrics = evaluate(model, val_loader, device)
                model.train()
                print(f"[eval @ step {global_step}] {eval_metrics}")
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in eval_metrics.items()} | {"step": global_step})
                if eval_metrics["pixel_err_mean"] < best_pixel_err:
                    best_pixel_err = eval_metrics["pixel_err_mean"]
                    torch.save(
                        {"model": model.state_dict(), "cfg": cfg.__dict__, "best_pixel_err": best_pixel_err, "model_type": "screen_gaze"},
                        best_path,
                    )
                    print(f"[ckpt] best {best_pixel_err:.1f}px → {best_path}")

        # epoch eval
        eval_metrics = evaluate(model, val_loader, device)
        print(f"[eval @ epoch {epoch + 1}] {eval_metrics}")
        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in eval_metrics.items()} | {"epoch": epoch + 1})
        if eval_metrics["pixel_err_mean"] < best_pixel_err:
            best_pixel_err = eval_metrics["pixel_err_mean"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "best_pixel_err": best_pixel_err, "model_type": "screen_gaze"},
                best_path,
            )
            print(f"[ckpt] best {best_pixel_err:.1f}px → {best_path}")

    print(f"[done] best val pixel error = {best_pixel_err:.1f}px at {best_path}")
    return best_path
