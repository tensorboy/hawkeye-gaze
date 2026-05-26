"""Smoke test: model loads, forward pass runs, output shapes are sane.

Skips SigLIP 2 download by default (use --siglip to force). Always runs ResNet-50
which is bundled in torchvision.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from gaze.models.gaze_model import GazeModel
from gaze.models.head import angular_error_deg, l2cs_loss


def test_resnet_forward() -> None:
    """Fastest backbone — sanity check the pipeline."""
    model = GazeModel(backbone_name="resnet50").eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        out = model(x)
    assert out.yaw_deg.shape == (2,)
    assert out.pitch_deg.shape == (2,)
    assert out.l2cs.yaw_logits.shape == (2, 90)
    assert out.l2cs.pitch_logits.shape == (2, 90)
    # Expected angle from random softmax should be roughly centered (near 0)
    assert out.yaw_deg.abs().max() < 180.0


def test_l2cs_loss_runs() -> None:
    model = GazeModel(backbone_name="resnet50")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    yaw_tgt = torch.tensor([10.0, -20.0])
    pitch_tgt = torch.tensor([5.0, -3.0])
    total, metrics = l2cs_loss(out.l2cs, yaw_tgt, pitch_tgt)
    assert torch.isfinite(total)
    assert "ce" in metrics and "mse" in metrics
    total.backward()  # gradients flow end-to-end
    n_with_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert n_with_grad > 0


def test_angular_error_zero_for_identity() -> None:
    yaw = torch.tensor([0.0, 30.0])
    pitch = torch.tensor([0.0, -10.0])
    err = angular_error_deg(yaw, pitch, yaw, pitch)
    # float32 acos near 1.0 has limited precision; 0.05° is well below gaze noise floor
    assert err.abs().max() < 0.05


@pytest.mark.skip(reason="downloads ~340MB SigLIP 2 weights; opt-in only")
def test_siglip2_forward() -> None:
    model = GazeModel(backbone_name="siglip2_base_256").eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        out = model(x)
    assert out.yaw_deg.shape == (1,)
