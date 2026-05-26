"""Live webcam demo: load GazeModel, run on webcam frames, draw gaze arrow on face.

Usage:
    uv run scripts/demo_webcam.py
    uv run scripts/demo_webcam.py --backbone siglip2_base_256
    uv run scripts/demo_webcam.py --checkpoint checkpoints/best.pt

Press 'q' to quit. With no checkpoint, runs the backbone with a randomly-initialized
L2CS head — output will be garbage, but proves the pipeline is wired up. Real gaze
predictions require training first.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Allow `from gaze.X` imports when invoked via `uv run scripts/demo_webcam.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import torch
from PIL import Image

from gaze.models.gaze_model import GazeModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="siglip2_base_256")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def preprocess(frame_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    """BGR np.ndarray -> normalized (1, 3, H, W) tensor matching SigLIP 2 preprocessing."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((input_size, input_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # SigLIP normalization (mean=0.5, std=0.5)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def draw_gaze(frame: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Draw a green arrow from the frame center indicating gaze direction."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    length = min(w, h) // 3

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    dx = -length * math.cos(pitch) * math.sin(yaw)
    dy = -length * math.sin(pitch)
    end = (int(cx + dx), int(cy + dy))

    cv2.arrowedLine(frame, (cx, cy), end, (0, 255, 0), 3, tipLength=0.2)
    text = f"yaw={yaw_deg:+.1f} pitch={pitch_deg:+.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    print(f"[demo] device={device}")
    print(f"[demo] loading backbone={args.backbone} (first run downloads weights)")

    model = GazeModel(backbone_name=args.backbone).to(device).eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        print(f"[demo] loaded checkpoint {args.checkpoint}")
    else:
        print("[demo] WARNING: no checkpoint, L2CS head is random — output is meaningless")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[demo] could not open camera {args.camera}", file=sys.stderr)
        return 1

    print("[demo] press 'q' to quit")
    fps_smoothed = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            inp = preprocess(frame, args.input_size).to(device)
            with torch.inference_mode():
                t0 = time.perf_counter()
                out = model(inp)
                latency_ms = (time.perf_counter() - t0) * 1000.0

            yaw = float(out.yaw_deg.item())
            pitch = float(out.pitch_deg.item())
            frame = draw_gaze(frame, yaw, pitch)

            t_now = time.perf_counter()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps
            cv2.putText(
                frame,
                f"infer {latency_ms:.0f}ms  display {fps_smoothed:.1f}fps  {device}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("hawkeye-gaze demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
