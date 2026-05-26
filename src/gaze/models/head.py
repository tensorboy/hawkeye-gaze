"""L2CS gaze head — yaw/pitch bin classification + regression refinement.

Paper: Abdelrahman et al., "L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained
Environments", arXiv:2203.03339.

The trick: regress angle as (a) classify into N bins of width B degrees, (b) compute
expected angle from softmax over bins. Joint loss = cross-entropy on bin + MSE on
the expected angle. More stable than naive regression alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


# Standard L2CS setup: 90 bins of 4°, covering [-180°, +180°]
N_BINS = 90
BIN_WIDTH_DEG = 4.0


@dataclass
class L2CSOutput:
    yaw_logits: Tensor  # (B, N_BINS)
    pitch_logits: Tensor  # (B, N_BINS)
    yaw_deg: Tensor  # (B,) expected yaw in degrees
    pitch_deg: Tensor  # (B,) expected pitch in degrees


class L2CSHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int = N_BINS, bin_width_deg: float = BIN_WIDTH_DEG) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.bin_width_deg = bin_width_deg
        # bin centers: -180 + 2°, -176 + 2°, ... +180 - 2°
        half = bin_width_deg / 2.0
        centers = torch.linspace(-180.0 + half, 180.0 - half, steps=n_bins)
        self.register_buffer("bin_centers_deg", centers, persistent=False)

        # Shared trunk then two heads (one per axis)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.yaw_head = nn.Linear(512, n_bins)
        self.pitch_head = nn.Linear(512, n_bins)

    def forward(self, features: Tensor) -> L2CSOutput:
        h = self.trunk(features)
        yaw_logits = self.yaw_head(h)
        pitch_logits = self.pitch_head(h)

        yaw_probs = torch.softmax(yaw_logits, dim=-1)
        pitch_probs = torch.softmax(pitch_logits, dim=-1)

        yaw_deg = (yaw_probs * self.bin_centers_deg).sum(dim=-1)
        pitch_deg = (pitch_probs * self.bin_centers_deg).sum(dim=-1)

        return L2CSOutput(
            yaw_logits=yaw_logits,
            pitch_logits=pitch_logits,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
        )


def gaze_to_bin_target(angle_deg: Tensor, n_bins: int = N_BINS, bin_width_deg: float = BIN_WIDTH_DEG) -> Tensor:
    """Convert continuous angle in degrees to bin index. Clamps to [0, n_bins-1]."""
    idx = ((angle_deg + 180.0) / bin_width_deg).long()
    return idx.clamp(0, n_bins - 1)


def l2cs_loss(
    out: L2CSOutput,
    yaw_target_deg: Tensor,
    pitch_target_deg: Tensor,
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Joint classification + regression loss from the L2CS paper."""
    yaw_tgt_bin = gaze_to_bin_target(yaw_target_deg, out.yaw_logits.size(-1))
    pitch_tgt_bin = gaze_to_bin_target(pitch_target_deg, out.pitch_logits.size(-1))

    ce = nn.functional.cross_entropy(out.yaw_logits, yaw_tgt_bin) + nn.functional.cross_entropy(
        out.pitch_logits, pitch_tgt_bin
    )
    mse = nn.functional.mse_loss(out.yaw_deg, yaw_target_deg) + nn.functional.mse_loss(
        out.pitch_deg, pitch_target_deg
    )

    total = ce_weight * ce + mse_weight * mse
    metrics = {"ce": ce.item(), "mse": mse.item(), "total": total.item()}
    return total, metrics


def angular_error_deg(pred_yaw: Tensor, pred_pitch: Tensor, tgt_yaw: Tensor, tgt_pitch: Tensor) -> Tensor:
    """3D angular error between predicted and target gaze vectors (degrees)."""

    def to_vec(yaw_deg: Tensor, pitch_deg: Tensor) -> Tensor:
        yaw = torch.deg2rad(yaw_deg)
        pitch = torch.deg2rad(pitch_deg)
        x = -torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)
        return torch.stack([x, y, z], dim=-1)

    v_pred = to_vec(pred_yaw, pred_pitch)
    v_tgt = to_vec(tgt_yaw, tgt_pitch)
    cos = (v_pred * v_tgt).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos))
