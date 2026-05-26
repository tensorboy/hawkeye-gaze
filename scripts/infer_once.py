"""One-shot inference: capture a single webcam frame (or take --image),
run the model, print yaw/pitch and timing, save annotated PNG.

Designed to be runnable headless — no cv2.imshow loop.

Usage:
    uv run scripts/infer_once.py                              # webcam, ResNet-50
    uv run scripts/infer_once.py --backbone siglip2_base_256  # webcam, SigLIP 2
    uv run scripts/infer_once.py --image path.jpg             # use file instead
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import torch
from PIL import Image

from gaze.models.gaze_model import GazeModel


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def preprocess(frame_bgr: np.ndarray, input_size: int, mean: tuple, std: tuple) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((input_size, input_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def draw(frame: np.ndarray, yaw: float, pitch: float, infer_ms: float, dev: str) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    L = min(w, h) // 3
    dx = -L * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
    dy = -L * math.sin(math.radians(pitch))
    cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(frame, f"yaw={yaw:+.1f}  pitch={pitch:+.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"infer {infer_ms:.0f} ms on {dev}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image", default=None, help="if set, use this image instead of webcam")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default="exports/infer_once.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = select_device(args.device)

    # Pick normalization matching backbone
    if args.backbone.startswith("siglip2"):
        input_size, mean, std = 256, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        input_size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    print(f"[infer] device={device} backbone={args.backbone} input={input_size}")

    t0 = time.perf_counter()
    model = GazeModel(backbone_name=args.backbone).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[infer] loaded {n_params:.1f}M params in {time.perf_counter() - t0:.1f}s")

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        print(f"[infer] checkpoint {args.checkpoint}")
    else:
        print("[infer] no checkpoint — head is random, output is meaningless gaze direction")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[infer] could not read {args.image}", file=sys.stderr)
            return 1
        print(f"[infer] reading {args.image} ({frame.shape})")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[infer] could not open camera {args.camera}", file=sys.stderr)
            return 1
        # warmup grab a few frames so exposure stabilizes
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()
        cap.release()
        if not ok:
            print("[infer] failed to grab webcam frame", file=sys.stderr)
            return 1
        print(f"[infer] grabbed webcam frame {frame.shape}")

    # Warmup forward pass (first run on MPS is slow)
    with torch.inference_mode():
        warm = preprocess(frame, input_size, mean, std).to(device)
        _ = model(warm)
        if device.type == "mps":
            torch.mps.synchronize()

    # Timed forward pass
    inp = preprocess(frame, input_size, mean, std).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(inp)
        if device.type == "mps":
            torch.mps.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    yaw = float(out.yaw_deg.item())
    pitch = float(out.pitch_deg.item())

    print(f"[infer] yaw={yaw:+.2f}°  pitch={pitch:+.2f}°  in {infer_ms:.1f} ms")

    annotated = draw(frame.copy(), yaw, pitch, infer_ms, str(device))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"[infer] saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
