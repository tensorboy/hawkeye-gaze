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


def _find_cv2_window_id(title: str) -> int | None:
    """Look up our cv2 window's CGWindowID by its visible title."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionOnScreenOnly,
        )

        infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for w in infos:
            name = w.get("kCGWindowName", "") or ""
            if title in name:
                return int(w["kCGWindowNumber"])
    except Exception as e:
        print(f"[screen] window lookup failed: {e}", file=sys.stderr)
    return None


def grab_screen_below_window_bgr(window_id: int | None) -> np.ndarray | None:
    """Capture the primary screen *excluding* the given window (and the windows
    above it). Uses Quartz's native `kCGWindowListOptionOnScreenBelowWindow`,
    so the demo window can be moved/resized freely without breaking the view.
    Falls back to a full PIL.ImageGrab when the window id is unknown.
    """
    if window_id is None:
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab(all_screens=False)
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[screen] PIL fallback failed: {e}", file=sys.stderr)
            return None

    try:
        from Quartz import (
            CGDataProviderCopyData,
            CGImageGetBytesPerRow,
            CGImageGetDataProvider,
            CGImageGetHeight,
            CGImageGetWidth,
            CGRectInfinite,
            CGWindowListCreateImage,
            kCGWindowImageDefault,
            kCGWindowListOptionOnScreenBelowWindow,
        )

        img_ref = CGWindowListCreateImage(
            CGRectInfinite,
            kCGWindowListOptionOnScreenBelowWindow,
            window_id,
            kCGWindowImageDefault,
        )
        if img_ref is None:
            return None
        w = CGImageGetWidth(img_ref)
        h = CGImageGetHeight(img_ref)
        bpr = CGImageGetBytesPerRow(img_ref)
        data = CGDataProviderCopyData(CGImageGetDataProvider(img_ref))
        buf = bytes(data)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpr // 4, 4))[:, :w, :]
        # macOS captures as BGRA → drop alpha
        return arr[:, :, :3].copy()
    except Exception as e:
        print(f"[screen] CGWindowListCreateImage failed: {e}", file=sys.stderr)
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab(all_screens=False)
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        except Exception:
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
    mouse_xy: tuple[int, int] | None = None,
    target_w: int = 1280,
) -> np.ndarray:
    """Top = webcam with overlay. Bottom = screen capture with predicted gaze
    dot AND current mouse cursor position. Both resized to target_w and
    stacked vertically. Mouse position is drawn in cyan so it's visually
    distinct from the red/yellow gaze prediction."""

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

    # Bottom: screen capture with predicted gaze dot + mouse cursor marker
    if screen_bgr is not None:
        bot = screen_bgr.copy()
        bot_h, bot_w = bot.shape[:2]
        # Captured image may be retina-scaled (e.g. 3200×1800 for a 1600×900 logical screen).
        # Convert logical (screen_w, screen_h) coords to capture-pixel coords.
        scale_x = bot_w / max(screen_w, 1)
        scale_y = bot_h / max(screen_h, 1)

        # ── Predicted gaze: bright LIME + black outline (visible on any background)
        # BGR: (0, 255, 0) is pure green; outline first, then colored line over it.
        if yaw is not None and pitch is not None:
            px_log, py_log = yaw_pitch_to_screen_xy(yaw, pitch, screen_w, screen_h)
            px = int(px_log * scale_x)
            py = int(py_log * scale_y)
            # outline ring (black, thicker)
            cv2.circle(bot, (px, py), 50, (0, 0, 0), 12)
            cv2.circle(bot, (px, py), 50, (0, 255, 0), 6)  # bright lime ring
            cv2.circle(bot, (px, py), 14, (0, 0, 0), -1)
            cv2.circle(bot, (px, py), 12, (0, 255, 0), -1)  # center dot
            # text with black shadow then green fg
            text_g = f"GAZE pred ({px_log}, {py_log})"
            tpos = (px + 60, py + 10)
            cv2.putText(bot, text_g, tpos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(bot, text_g, tpos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # ── Mouse cursor: bright MAGENTA crosshair + ring + outlines
        if mouse_xy is not None:
            mx = int(mouse_xy[0] * scale_x)
            my = int(mouse_xy[1] * scale_y)
            magenta = (255, 0, 255)
            # outline (black) then magenta on top, for both ring and crosshair
            cv2.circle(bot, (mx, my), 32, (0, 0, 0), 10)
            cv2.circle(bot, (mx, my), 32, magenta, 5)
            cv2.line(bot, (mx - 48, my), (mx + 48, my), (0, 0, 0), 8)
            cv2.line(bot, (mx - 48, my), (mx + 48, my), magenta, 4)
            cv2.line(bot, (mx, my - 48), (mx, my + 48), (0, 0, 0), 8)
            cv2.line(bot, (mx, my - 48), (mx, my + 48), magenta, 4)
            text_m = f"MOUSE ({mouse_xy[0]}, {mouse_xy[1]})"
            tpos_m = (mx + 60, my + 60)
            cv2.putText(bot, text_m, tpos_m, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(bot, text_m, tpos_m, cv2.FONT_HERSHEY_SIMPLEX, 1.0, magenta, 2)

        cv2.putText(bot, f"SCREEN  {bot.shape[1]}x{bot.shape[0]}  (heuristic mapping until trained)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
    """FIFO-per-zone collector.

    Each grid zone is a fixed-capacity deque. When a new click arrives in a
    full zone, the oldest sample in that zone is evicted (jpeg deleted,
    CSV row removed) before the new sample is appended. This keeps the
    training set bounded above (grid_size² × max_per_zone) while letting
    appearance drift (new glasses, new lighting, new haircut) replace stale
    samples organically.
    """

    CSV_FIELDS = ["image_path", "click_x", "click_y", "screen_w", "screen_h", "timestamp_iso"]

    def __init__(
        self,
        out_dir: Path,
        buffer: FrameBuffer,
        grid_size: int = 10,
        max_per_zone: int = 50,
    ) -> None:
        from collections import deque

        self.out_dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "clicks.csv"
        self.buffer = buffer
        self.screen_w, self.screen_h = get_primary_screen_size()
        self.grid_size = grid_size
        self.max_per_zone = max_per_zone
        self.evicted = 0
        # zone → deque[row_dict]; FIFO order. Restored from CSV on startup.
        self.zone_records: dict[tuple[int, int], deque] = {}

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                w.writeheader()
        else:
            with self.csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        z = self._zone_for(
                            float(row["click_x"]), float(row["click_y"]),
                            int(row["screen_w"]), int(row["screen_h"]),
                        )
                    except (KeyError, ValueError):
                        continue
                    self.zone_records.setdefault(z, deque()).append(row)

        print(f"[collect] output dir: {self.out_dir} (existing: {self.count} samples)")
        print(f"[collect] primary screen: {self.screen_w}×{self.screen_h}")
        print(f"[collect] FIFO: {grid_size}×{grid_size} grid, max {max_per_zone} samples/zone")
        full_zones = sum(1 for d in self.zone_records.values() if len(d) >= max_per_zone)
        if full_zones:
            print(f"[collect] {full_zones} / {grid_size * grid_size} zones at cap — new clicks evict oldest in those zones")

    @property
    def count(self) -> int:
        return sum(len(d) for d in self.zone_records.values())

    def _zone_for(self, x: float, y: float, sw: int, sh: int) -> tuple[int, int]:
        gx = min(self.grid_size - 1, max(0, int(x / sw * self.grid_size)))
        gy = min(self.grid_size - 1, max(0, int(y / sh * self.grid_size)))
        return gx, gy

    def _rewrite_csv(self) -> None:
        # Order rows by timestamp so the on-disk CSV stays chronologically sorted
        all_rows = [r for d in self.zone_records.values() for r in d]
        all_rows.sort(key=lambda r: r.get("timestamp_iso", ""))
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)

    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        from collections import deque

        # Only on press (not release) and only left button
        if not pressed or button != mouse.Button.left:
            return
        frame, ts = self.buffer.latest()
        if frame is None:
            return

        zone = self._zone_for(x, y, self.screen_w, self.screen_h)
        zone_deque = self.zone_records.setdefault(zone, deque())

        action = "add"
        if len(zone_deque) >= self.max_per_zone:
            oldest = zone_deque.popleft()
            oldest_path = self.out_dir / oldest["image_path"]
            if oldest_path.exists():
                try:
                    oldest_path.unlink()
                except OSError as e:
                    print(f"[collect]   warn: failed to delete {oldest_path}: {e}", file=sys.stderr)
            self.evicted += 1
            action = f"FIFO evict {Path(oldest['image_path']).name}"

        now = datetime.fromtimestamp(ts, tz=timezone.utc)
        stem = now.strftime("%Y%m%dT%H%M%S_") + f"{int((ts % 1) * 1000):03d}"
        img_path = self.frames_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        row = {
            "image_path": str(img_path.relative_to(self.out_dir)),
            "click_x": str(x),
            "click_y": str(y),
            "screen_w": str(self.screen_w),
            "screen_h": str(self.screen_h),
            "timestamp_iso": now.isoformat(),
        }
        zone_deque.append(row)
        self._rewrite_csv()

        n_total = self.count
        n_zone = len(zone_deque)
        print(f"[collect] {n_total:5d} ← click ({x:>5},{y:>5}) zone {zone} ({n_zone}/{self.max_per_zone}) {action}")


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
    p.add_argument("--grid-size", type=int, default=10, help="grid size for spatial balance")
    p.add_argument("--max-per-zone", type=int, default=50, help="max samples per grid zone")
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
    collector = ClickCollector(
        Path(args.out_dir),
        frame_buffer,
        grid_size=args.grid_size,
        max_per_zone=args.max_per_zone,
    )
    listener = mouse.Listener(on_click=collector.on_click)
    listener.start()
    mouse_ctrl = mouse.Controller()  # for per-frame cursor position query
    print("[live] global mouse listener started — click anywhere to log a sample")
    print("[live] (macOS: grant Accessibility permission to your terminal app if prompted)")

    last_infer_ms = 0.0
    yaw = pitch = None
    screen_bgr: np.ndarray | None = None
    last_screen_grab_ts = 0.0
    SCREEN_GRAB_INTERVAL = 0.25  # 4 fps screen refresh — keeps CPU sane

    WINDOW_NAME = "hawkeye-gaze live + collect"
    cv2_window_id: int | None = None  # discovered after first imshow

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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
                # Refresh screen capture at SCREEN_GRAB_INTERVAL.
                # The Quartz native API gets only the area BELOW our window,
                # so wherever the user drags the window, recursion is impossible.
                now = time.time()
                if now - last_screen_grab_ts > SCREEN_GRAB_INTERVAL:
                    if cv2_window_id is None:
                        cv2_window_id = _find_cv2_window_id(WINDOW_NAME)
                    screen_bgr = grab_screen_below_window_bgr(cv2_window_id)
                    last_screen_grab_ts = now

                try:
                    mx, my = mouse_ctrl.position
                    mouse_xy = (int(mx), int(my))
                except Exception:
                    mouse_xy = None
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
                    mouse_xy=mouse_xy,
                )
                cv2.imshow(WINDOW_NAME, composite)
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
