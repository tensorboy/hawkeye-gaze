"""Export a trained GazeModel to a CoreML mlpackage.

Known gotcha (coremltools issue #2311): PyTorch's `_native_multi_head_attention`
fast path emits ops that current coremltools cannot trace for SigLIP / ViT
backbones. We must force the math attention path before tracing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from gaze.models.gaze_model import GazeModel


def disable_attention_fast_paths() -> None:
    """Force the math/eager attention path. Required for ViT-based backbones."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except (AttributeError, RuntimeError):
        # Older PyTorch or non-CUDA platforms: safe to skip
        pass


class GazeExportWrapper(torch.nn.Module):
    """Wrap GazeModel so it returns only the two scalars CoreML needs.

    GazeModel forward returns a dataclass; we collapse it to a plain (yaw, pitch) tuple
    so coremltools' tracer can map outputs cleanly.
    """

    def __init__(self, model: GazeModel) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values)
        return out.yaw_deg, out.pitch_deg


def export(
    checkpoint: str | Path,
    out_path: str | Path,
    backbone: str = "siglip2_base_256",
    input_size: int = 256,
    quantize: str = "fp16",
) -> Path:
    import coremltools as ct

    disable_attention_fast_paths()

    model = GazeModel(backbone_name=backbone)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()

    wrapper = GazeExportWrapper(model)
    example = torch.randn(1, 3, input_size, input_size)

    print(f"[export] tracing wrapper (input {tuple(example.shape)})")
    traced = torch.jit.trace(wrapper, example, strict=False)

    print("[export] converting to CoreML")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape)],
        outputs=[ct.TensorType(name="yaw_deg"), ct.TensorType(name="pitch_deg")],
        compute_precision=ct.precision.FLOAT16 if quantize == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,  # CPU / GPU / ANE — let CoreML pick best per op
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    if quantize == "int8":
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)
        print("[export] applied int8 weight quantization")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"[export] saved → {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="exports/hawkeye-gaze.mlpackage")
    p.add_argument("--backbone", default="siglip2_base_256")
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--quantize", choices=["fp32", "fp16", "int8"], default="fp16")
    args = p.parse_args()
    export(
        checkpoint=args.checkpoint,
        out_path=args.out,
        backbone=args.backbone,
        input_size=args.input_size,
        quantize=args.quantize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
