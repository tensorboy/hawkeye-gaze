"""Train ScreenGazeModel on click-as-label data.

Usage:
    uv run scripts/train_clicks.py --config configs/siglip2_clicks.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaze.train.clicks_loop import train_clicks
from gaze.train.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_clicks(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
