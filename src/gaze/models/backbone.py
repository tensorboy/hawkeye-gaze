"""Backbone wrappers — SigLIP 2 / MobileCLIP2 / MobileOne / custom.

Selectable via config. All backbones output a per-image feature tensor of shape (B, D)
where D depends on the backbone. Downstream heads adapt to D automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


BackboneName = Literal[
    "siglip2_base_256",
    "siglip2_large_256",
    "mobileclip2_s0",
    "mobileclip2_s2",
    "resnet50",
]


@dataclass
class BackboneOutput:
    features: Tensor  # (B, D) pooled, or (B, N, D) tokens if return_tokens
    feature_dim: int


class SigLIP2Backbone(nn.Module):
    """Wraps HF transformers Siglip2VisionModel."""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-256") -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).vision_model
        self.model_id = model_id
        # SigLIP 2 Base hidden_size = 768, Large = 1024
        self.feature_dim: int = self.model.config.hidden_size

    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        out = self.model(pixel_values=pixel_values, output_hidden_states=False)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        return BackboneOutput(features=pooled, feature_dim=self.feature_dim)


class MobileCLIP2Backbone(nn.Module):
    """Wraps timm mobileclip2 image encoder. Fallback when SigLIP 2 too slow."""

    def __init__(self, variant: str = "mobileclip2_s0") -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(variant, pretrained=True, num_classes=0)
        self.feature_dim: int = self.model.num_features

    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        features = self.model(pixel_values)
        return BackboneOutput(features=features, feature_dim=self.feature_dim)


class ResNet50Backbone(nn.Module):
    """ImageNet-pretrained ResNet-50 baseline. BSD-style commercial-safe."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.model = backbone
        self.feature_dim: int = 2048

    def forward(self, pixel_values: Tensor) -> BackboneOutput:
        features = self.model(pixel_values)
        return BackboneOutput(features=features, feature_dim=self.feature_dim)


def build_backbone(name: BackboneName) -> nn.Module:
    if name == "siglip2_base_256":
        return SigLIP2Backbone("google/siglip2-base-patch16-256")
    if name == "siglip2_large_256":
        return SigLIP2Backbone("google/siglip2-large-patch16-256")
    if name in ("mobileclip2_s0", "mobileclip2_s2"):
        return MobileCLIP2Backbone(name)
    if name == "resnet50":
        return ResNet50Backbone()
    raise ValueError(f"unknown backbone: {name}")
