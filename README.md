# hawkeye-gaze

End-to-end trainable, on-device personalizable gaze estimation for macOS.

Independent R&D project. Output: a single `.mlpackage` consumed by the main Hawkeye app.

## Architecture

```
camera face crop (256×256)
   ↓
[trainable] SigLIP 2 ViT-B/16-256       (86M, Apache 2.0)
   ↓
[trainable] L2CS head                   (yaw bins + pitch bins + regression)
   ↓
3D gaze (yaw, pitch)
   ↓
[trainable] Screen Mapping Head         (per-user, on-device fine-tune)
   ↓
screen (x, y)
```

All ~86M params trainable. Backbone is selectable (SigLIP 2 / MobileCLIP2 / MobileOne / custom)
via YAML config — see `configs/`.

## Why fully trainable

WebGazer-style ridge regression on hand-crafted features cannot absorb continuous user data
(hardcoded 50-sample ring buffer). Foundation-model-frozen approaches (Gaze-LLE family) target
third-person gaze-target estimation, not first-person face → screen mapping. This project
follows the L2CS-Net / MPIIFaceGaze first-person paradigm with a modern backbone, everything
end-to-end trainable.

## Quick start

```bash
uv sync                                      # install deps
uv run scripts/demo_webcam.py                # live demo on your face
uv run scripts/train.py --config configs/siglip2_base.yaml
uv run scripts/export.py --checkpoint best.pt --out exports/hawkeye-gaze.mlpackage
```

## Layout

```
src/gaze/
├── models/         backbone wrappers, L2CS head, full GazeModel
├── data/           dataset loaders, augmentation
├── train/          training loop, eval, config
├── deploy/         CoreML / ONNX export
└── utils/          face detection, screen mapping geometry

scripts/            CLI entry points (uv run scripts/X.py)
configs/            YAML configs (one per backbone)
checkpoints/        training outputs (gitignored)
exports/            production .mlpackage files (gitignored)
tests/              pytest
```

## Datasets (commercial use)

All real gaze datasets are research-only (Gaze360, MPIIFaceGaze, ETH-XGaze, GazeCapture).
For commercial deployment we will either:
1. Self-collect (100-200 people, ~$5-10k)
2. Use GazeGene synthetic + domain adaptation
3. Negotiate commercial license

Public datasets are usable for development/research only.
