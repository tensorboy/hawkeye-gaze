"""Overfit test: can SigLIP 2 + L2CS head memorize a tiny fixed batch?

This is a diagnostic, not an accuracy benchmark. We generate N fixed (image, gaze)
pairs and train for K steps. The loss should drop from ~12 (random) toward 0.
If it doesn't, something is broken in:
  - gradient flow through the backbone
  - layer-wise LR setup
  - L2CS loss formulation
  - autocast / amp config

Run:
    uv run python scripts/test_overfit.py --backbone siglip2_base_256 --steps 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.optim import AdamW

from gaze.models.gaze_model import GazeModel
from gaze.models.head import angular_error_deg, l2cs_loss


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    args = p.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(42)

    print(f"[overfit] device={device} backbone={args.backbone}")
    t0 = time.perf_counter()
    model = GazeModel(backbone_name=args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[overfit] {n_params:.1f}M params loaded in {time.perf_counter() - t0:.1f}s")

    # Fixed synthetic batch — N images, N gaze labels
    if args.backbone.startswith("resnet"):
        size = 224
    else:
        size = args.input_size

    imgs = torch.randn(args.n_samples, 3, size, size, device=device)
    yaw_targets = torch.linspace(-30, 30, args.n_samples, device=device)
    pitch_targets = torch.linspace(-15, 15, args.n_samples, device=device)

    optim = AdamW(
        model.trainable_param_groups(backbone_lr=args.backbone_lr, head_lr=args.head_lr),
        weight_decay=1e-4,
    )

    # Initial loss + angular error
    model.eval()
    with torch.inference_mode():
        out = model(imgs)
        init_loss, init_metrics = l2cs_loss(out.l2cs, yaw_targets, pitch_targets)
        init_err = angular_error_deg(out.yaw_deg, out.pitch_deg, yaw_targets, pitch_targets).mean()
    print(f"[overfit] step   0: loss={init_metrics['total']:.3f}  ang_err={init_err.item():.2f}°")

    # Train
    model.train()
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        optim.zero_grad(set_to_none=True)
        out = model(imgs)
        loss, metrics = l2cs_loss(out.l2cs, yaw_targets, pitch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(metrics["total"])
        if step % max(args.steps // 10, 1) == 0:
            with torch.inference_mode():
                err = angular_error_deg(out.yaw_deg, out.pitch_deg, yaw_targets, pitch_targets).mean()
            print(
                f"[overfit] step {step:3d}: loss={metrics['total']:.3f}  "
                f"ce={metrics['ce']:.3f}  mse={metrics['mse']:.3f}  ang_err={err.item():.2f}°"
            )

    elapsed = time.perf_counter() - t0
    final_loss = losses[-1]
    drop_ratio = (init_metrics["total"] - final_loss) / max(init_metrics["total"], 1e-6)

    # Final eval
    model.eval()
    with torch.inference_mode():
        out = model(imgs)
        final_err = angular_error_deg(out.yaw_deg, out.pitch_deg, yaw_targets, pitch_targets).mean()

    print()
    print(f"[overfit] {args.steps} steps in {elapsed:.1f}s ({elapsed / args.steps * 1000:.0f}ms/step)")
    print(f"[overfit] loss     {init_metrics['total']:.2f} → {final_loss:.2f}  (-{drop_ratio * 100:.0f}%)")
    print(f"[overfit] ang_err  {init_err.item():.2f}° → {final_err.item():.2f}°")
    print()

    if drop_ratio < 0.5:
        print("[overfit] ❌ FAIL: loss did not drop ≥ 50%. Check optimizer / loss / gradients.")
        return 1
    if final_err > 5.0:
        print("[overfit] ⚠ WARN: final angular error > 5° even on memorized batch — increase steps or LR.")
        return 0
    print("[overfit] ✅ PASS: model learns the fixed batch end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
