"""Live webcam demo + click-as-label data collector + screen-overlay preview.

What this does, running together in one process:
1. Opens a webcam capture loop in the main thread, runs the gaze model on each
   frame, draws the predicted gaze arrow on the frame.
2. Periodically grabs a full-screen screenshot, draws the *predicted* gaze
   point on it (mapping yaw/pitch → screen xy via a simple linear heuristic
   until the model is trained), then stacks webcam-on-top + screenshot-on-
   bottom into a single cv2 window so you can see both views at once.
3. Starts a pynput global mouse listener in a background thread. On every left
   click anywhere on the screen, snapshots the most-recent webcam frame and
   appends a row to data/clicks/clicks.csv:
       image_path, click_x, click_y, screen_w, screen_h, timestamp_ms
   The frame is saved as a JPEG under data/clicks/frames/.

The model is OPTIONAL — pass --no-model to skip inference and just collect data
faster (no MPS warmup, no per-frame compute). Default uses ResNet-50 (no
download required) so you see something on screen immediately.

macOS permissions: global mouse capture requires Accessibility permission for
your terminal app (System Settings → Privacy & Security → Accessibility).
First run will prompt; subsequent runs are silent.

Usage:
    uv run scripts/live_collect.py                                 # demo + collect, ResNet
    uv run scripts/live_collect.py --backbone siglip2_base_256     # use SigLIP 2
    uv run scripts/live_collect.py --no-model                      # data collection only
    uv run scripts/live_collect.py --collect-only --headless       # no window at all

Press 'q' in the window to quit (or Ctrl-C in terminal for headless).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import torch
from PIL import Image
from pynput import mouse


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


def get_primary_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary display in points."""
    try:
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return int(frame.size.width), int(frame.size.height)
    except Exception:
        return 1920, 1080  # fallback


def grab_screen_bgr() -> np.ndarray | None:
    """Full primary-screen screenshot as a BGR numpy array. Returns None on failure."""
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab(all_screens=False)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[screen] capture failed: {e}", file=sys.stderr)
        return None


def yaw_pitch_to_screen_xy(
    yaw_deg: float, pitch_deg: float, screen_w: int, screen_h: int
) -> tuple[int, int]:
    """Linear heuristic for visualization before the model is trained.

    Maps yaw ∈ [-30°, +30°] linearly to x ∈ [0, screen_w].
    Maps pitch ∈ [+15°, -15°] linearly to y ∈ [0, screen_h] (positive pitch = up).
    Clamps to the screen bounds. Replace with the trained ScreenMappingHead once
    we have a calibrated model.
    """
    yaw_norm = max(-30.0, min(30.0, yaw_deg)) / 30.0  # [-1, +1]
    pitch_norm = -max(-15.0, min(15.0, pitch_deg)) / 15.0  # [-1, +1] flipped so up = top
    x = int((yaw_norm * 0.5 + 0.5) * screen_w)
    y = int((pitch_norm * 0.5 + 0.5) * screen_h)
    return x, y


def compose_split_view(
    webcam_bgr: np.ndarray,
    screen_bgr: np.ndarray | None,
    yaw: float | None,
    pitch: float | None,
    screen_w: int,
    screen_h: int,
    infer_ms: float,
    count: int,
    dev: str,
    target_w: int = 1280,
) -> np.ndarray:
    """Top = webcam with overlay. Bottom = screen capture with predicted gaze dot.
    Both resized to target_w and stacked vertically."""

    # Top: webcam
    top = webcam_bgr.copy()
    if yaw is not None and pitch is not None:
        h, w = top.shape[:2]
        cx, cy = w // 2, h // 2
        L = min(w, h) // 3
        dx = -L * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
        dy = -L * math.sin(math.radians(pitch))
        cv2.arrowedLine(top, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(top, f"yaw={yaw:+.1f}  pitch={pitch:+.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(top, f"WEBCAM   infer {infer_ms:.0f}ms on {dev}   samples {count}", (10, top.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    top_resized = cv2.resize(top, (target_w, int(target_w * top.shape[0] / top.shape[1])))

    # Bottom: screen capture with predicted gaze dot
    if screen_bgr is not None:
        bot = screen_bgr.copy()
        if yaw is not None and pitch is not None:
            px, py = yaw_pitch_to_screen_xy(yaw, pitch, screen_w, screen_h)
            cv2.circle(bot, (px, py), 30, (0, 0, 255), 4)  # outer red ring
            cv2.circle(bot, (px, py), 8, (0, 255, 255), -1)  # inner yellow dot
            cv2.putText(bot, f"predicted: ({px}, {py})", (px + 40, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(bot, f"SCREEN  {screen_bgr.shape[1]}x{screen_bgr.shape[0]}  (heuristic mapping until trained)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        bot_resized = cv2.resize(bot, (target_w, int(target_w * bot.shape[0] / bot.shape[1])))
    else:
        bot_resized = np.zeros((target_w * 9 // 16, target_w, 3), dtype=np.uint8)
        cv2.putText(bot_resized, "screen capture unavailable", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.vstack([top_resized, bot_resized])


# ────────────────────────────────────────────────────────────────────────────
# Shared state between main webcam thread and pynput mouse-listener thread.
# Wrapped in a tiny lock-free pattern: the listener pulls the *latest* frame
# pointer atomically — even a stale-by-30ms frame is fine for our purposes.
# ────────────────────────────────────────────────────────────────────────────


class FrameBuffer:
    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.time()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), self._timestamp)


class ClickCollector:
    def __init__(self, out_dir: Path, buffer: FrameBuffer) -> None:
        self.out_dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "clicks.csv"
        self.buffer = buffer
        self.count = 0
        self.screen_w, self.screen_h = get_primary_screen_size()

        # Write CSV header once
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "click_x", "click_y", "screen_w", "screen_h", "timestamp_iso"])
        else:
            # Count existing rows for resume
            with self.csv_path.open() as f:
                self.count = sum(1 for _ in f) - 1
        print(f"[collect] output dir: {self.out_dir} (existing: {self.count} samples)")
        print(f"[collect] primary screen: {self.screen_w}×{self.screen_h}")

    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        # Only on press (not release) and only left button
        if not pressed or button != mouse.Button.left:
            return
        frame, ts = self.buffer.latest()
        if frame is None:
            return

        now = datetime.fromtimestamp(ts, tz=timezone.utc)
        stem = now.strftime("%Y%m%dT%H%M%S_") + f"{int((ts % 1) * 1000):03d}"
        img_path = self.frames_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([str(img_path.relative_to(self.out_dir)), x, y, self.screen_w, self.screen_h, now.isoformat()])

        self.count += 1
        print(f"[collect] {self.count:5d} ← click ({x:>5},{y:>5}) → {img_path.name}")


def draw_overlay(frame: np.ndarray, yaw: float | None, pitch: float | None, infer_ms: float, count: int, dev: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if yaw is not None and pitch is not None:
        cx, cy = w // 2, h // 2
        L = min(w, h) // 3
        dx = -L * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
        dy = -L * math.sin(math.radians(pitch))
        cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, f"yaw={yaw:+.1f}  pitch={pitch:+.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"samples collected: {count}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if yaw is not None:
        cv2.putText(frame, f"infer {infer_ms:.0f}ms on {dev}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "click anywhere on screen → labeled sample", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="data/clicks")
    p.add_argument("--no-model", action="store_true", help="skip inference, collect-only mode")
    p.add_argument("--headless", action="store_true", help="no cv2 window")
    p.add_argument("--collect-only", action="store_true", help="alias for --no-model --headless")
    args = p.parse_args()
    if args.collect_only:
        args.no_model = True
        args.headless = True

    # ── Model (optional)
    model = None
    device = select_device(args.device)
    input_size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if not args.no_model:
        from gaze.models.gaze_model import GazeModel

        if args.backbone.startswith("siglip2"):
            input_size = 256
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        print(f"[live] loading model={args.backbone} on {device}…")
        t0 = time.perf_counter()
        model = GazeModel(backbone_name=args.backbone).to(device).eval()
        print(f"[live] loaded {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params in {time.perf_counter() - t0:.1f}s")
    else:
        print("[live] running in collect-only mode (no model)")

    # ── Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[live] could not open camera {args.camera}", file=sys.stderr)
        return 1

    # ── Shared frame buffer + click listener
    frame_buffer = FrameBuffer()
    collector = ClickCollector(Path(args.out_dir), frame_buffer)
    listener = mouse.Listener(on_click=collector.on_click)
    listener.start()
    print("[live] global mouse listener started — click anywhere to log a sample")
    print("[live] (macOS: grant Accessibility permission to your terminal app if prompted)")

    last_infer_ms = 0.0
    yaw = pitch = None
    screen_bgr: np.ndarray | None = None
    last_screen_grab_ts = 0.0
    SCREEN_GRAB_INTERVAL = 0.25  # 4 fps screen refresh — keeps CPU sane

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_buffer.push(frame)

            # Inference (optional)
            if model is not None:
                inp = preprocess(frame, input_size, mean, std).to(device)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(inp)
                    if device.type == "mps":
                        torch.mps.synchronize()
                last_infer_ms = (time.perf_counter() - t0) * 1000.0
                yaw = float(out.yaw_deg.item())
                pitch = float(out.pitch_deg.item())

            if not args.headless:
                # Refresh screen capture at SCREEN_GRAB_INTERVAL
                now = time.time()
                if now - last_screen_grab_ts > SCREEN_GRAB_INTERVAL:
                    screen_bgr = grab_screen_bgr()
                    last_screen_grab_ts = now

                composite = compose_split_view(
                    webcam_bgr=frame,
                    screen_bgr=screen_bgr,
                    yaw=yaw,
                    pitch=pitch,
                    screen_w=collector.screen_w,
                    screen_h=collector.screen_h,
                    infer_ms=last_infer_ms,
                    count=collector.count,
                    dev=str(device),
                )
                cv2.imshow("hawkeye-gaze live + collect", composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless: minimal CPU when no display
                time.sleep(0.03)
    except KeyboardInterrupt:
        print("[live] interrupted")
    finally:
        listener.stop()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[live] final sample count: {collector.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
