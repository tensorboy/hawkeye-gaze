"""Full GazeModel = Backbone + L2CSHead. Optional ScreenMappingHead for on-device use."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gaze.models.backbone import BackboneName, build_backbone
from gaze.models.head import L2CSHead, L2CSOutput


@dataclass
class GazeOutput:
    l2cs: L2CSOutput
    yaw_deg: Tensor  # convenience alias
    pitch_deg: Tensor


class GazeModel(nn.Module):
    """Backbone → L2CS head. All parameters trainable by default."""

    def __init__(self, backbone_name: BackboneName = "siglip2_base_256") -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        self.head = L2CSHead(in_dim=self.backbone.feature_dim)
        self.backbone_name = backbone_name

    def forward(self, pixel_values: Tensor) -> GazeOutput:
        feats = self.backbone(pixel_values).features
        l2cs = self.head(feats)
        return GazeOutput(l2cs=l2cs, yaw_deg=l2cs.yaw_deg, pitch_deg=l2cs.pitch_deg)

    def trainable_param_groups(
        self, backbone_lr: float = 1e-5, head_lr: float = 1e-3
    ) -> list[dict]:
        """Layer-wise learning rates: low lr on pretrained backbone, high lr on fresh head."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": self.head.parameters(), "lr": head_lr, "name": "head"},
        ]


class ScreenMappingHead(nn.Module):
    """Per-user, on-device fine-tuned. (yaw, pitch, head_pose_6dof) → (screen_x, screen_y)."""

    def __init__(self, in_dim: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),  # output in [-1, 1], scaled to screen size downstream
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
