"""Full GazeModel = Backbone + L2CSHead. Optional ScreenMappingHead for on-device use.

Two model flavors:
  * `GazeModel`     — backbone + L2CS head, output (yaw_deg, pitch_deg).
  * `ScreenGazeModel` — backbone + ScreenCoordHead, output (x_norm, y_norm)
    in [0, 1]². Trained end-to-end from (face_image, click_xy) click pairs.
Both keep all parameters trainable by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gaze.models.backbone import BackboneName, build_backbone
from gaze.models.head import L2CSHead, L2CSOutput, ScreenCoordHead, ScreenXYOutput


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


@dataclass
class ScreenGazeOutput:
    screen: ScreenXYOutput
    x_norm: Tensor  # convenience alias
    y_norm: Tensor


class ScreenGazeModel(nn.Module):
    """Backbone → ScreenCoordHead. End-to-end face-image → normalized (x, y).

    Trained on click-as-label data: input = face crop at click time, target =
    normalized click position on screen. Skips the (yaw, pitch) intermediate
    representation entirely.
    """

    def __init__(
        self,
        backbone_name: BackboneName = "siglip2_base_256",
        head_hidden: int = 512,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        self.head = ScreenCoordHead(
            in_dim=self.backbone.feature_dim, hidden=head_hidden, dropout=head_dropout
        )
        self.backbone_name = backbone_name

    def forward(self, pixel_values: Tensor) -> ScreenGazeOutput:
        feats = self.backbone(pixel_values).features
        screen = self.head(feats)
        return ScreenGazeOutput(screen=screen, x_norm=screen.xy_norm[:, 0], y_norm=screen.xy_norm[:, 1])

    def trainable_param_groups(
        self, backbone_lr: float = 1e-5, head_lr: float = 1e-3
    ) -> list[dict]:
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "name": "backbone"},
            {"params": self.head.parameters(), "lr": head_lr, "name": "head"},
        ]
