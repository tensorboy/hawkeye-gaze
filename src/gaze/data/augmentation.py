"""Gaze-aware data augmentation.

Critical detail: horizontal flip *must* negate yaw (left/right swaps). Most other
augmentations (color jitter, blur, slight crops) are gaze-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# SigLIP normalization (matches HF preprocessor)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# ImageNet normalization for ResNet baselines
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugmentedSample:
    image: torch.Tensor  # (3, H, W), normalized
    yaw_deg: float
    pitch_deg: float


class GazeAwareAugment:
    """Train-time augmentation that respects gaze geometry."""

    def __init__(
        self,
        input_size: int = 256,
        mean: tuple[float, float, float] = SIGLIP_MEAN,
        std: tuple[float, float, float] = SIGLIP_STD,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
    ) -> None:
        self.input_size = input_size
        self.hflip_prob = hflip_prob
        # Color aug is applied to PIL before tensor conversion
        self.color = T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
        )
        self.to_tensor = T.Compose(
            [
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image, yaw_deg: float, pitch_deg: float) -> AugmentedSample:
        if np.random.rand() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            yaw_deg = -yaw_deg  # critical: hflip negates yaw, pitch unchanged
        image = self.color(image)
        tensor = self.to_tensor(image)
        return AugmentedSample(image=tensor, yaw_deg=yaw_deg, pitch_deg=pitch_deg)


class GazeEvalTransform:
    """Eval-time transform: just resize + normalize, no augmentation."""

    def __init__(
        self,
        input_size: int = 256,
        mean: tuple[float, float, float] = SIGLIP_MEAN,
        std: tuple[float, float, float] = SIGLIP_STD,
    ) -> None:
        self.transform = T.Compose(
            [
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image, yaw_deg: float, pitch_deg: float) -> AugmentedSample:
        return AugmentedSample(image=self.transform(image), yaw_deg=yaw_deg, pitch_deg=pitch_deg)
