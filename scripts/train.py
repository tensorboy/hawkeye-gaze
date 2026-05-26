"""Training entry point.

Usage:
    uv run scripts/train.py --config configs/siglip2_base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaze.train.config import load_config
from gaze.train.loop import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML training config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
