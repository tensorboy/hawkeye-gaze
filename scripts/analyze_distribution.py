"""Inspect the spatial distribution of collected click samples.

Prints a 5×5 zone histogram of normalized click positions and writes a
density heatmap PNG. Use this to see whether the training set is dominated
by a few hot zones (menu bar, dock, frequently-used buttons) — which would
bias the model toward those locations regardless of true gaze.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/clicks/clicks.csv")
    p.add_argument("--grid", type=int, default=5, help="grid size for zone count, default 5×5")
    p.add_argument("--out", default="exports/click_density.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = Path(args.csv)
    if not p.exists():
        print(f"[analyze] no CSV at {p}", file=sys.stderr)
        return 1

    xs, ys = [], []
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sw, sh = int(row["screen_w"]), int(row["screen_h"])
            xs.append(float(row["click_x"]) / sw)
            ys.append(float(row["click_y"]) / sh)

    if not xs:
        print("[analyze] no samples")
        return 1

    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)
    print(f"[analyze] n = {n}")
    print(f"[analyze] x_norm  mean {xs.mean():.3f}  std {xs.std():.3f}  min {xs.min():.3f}  max {xs.max():.3f}")
    print(f"[analyze] y_norm  mean {ys.mean():.3f}  std {ys.std():.3f}  min {ys.min():.3f}  max {ys.max():.3f}")

    # Zone counts on a grid×grid grid
    g = args.grid
    bins = np.linspace(0, 1, g + 1)
    counts, _, _ = np.histogram2d(xs, ys, bins=[bins, bins])
    counts = counts.astype(int)
    if_uniform = n / (g * g)
    imbalance_ratio = counts.max() / max(1.0, if_uniform)

    print(f"\n[analyze] {g}×{g} zone counts (rows=x left→right, cols=y top→bottom):")
    for row in counts:
        print("  " + "  ".join(f"{c:4d}" for c in row))
    print(f"\n[analyze] if uniform: {if_uniform:.1f} samples / zone")
    print(f"[analyze] hottest zone has {counts.max()} samples → {imbalance_ratio:.1f}× more than uniform")
    print(f"[analyze] empty zones: {(counts == 0).sum()} / {g * g}")

    # Heatmap PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6 * counts.shape[1] / counts.shape[0]))
        # Transpose so visual axes match screen: x→right, y→down
        im = ax.imshow(counts.T, origin="upper", cmap="hot", aspect="auto", extent=[0, 1, 1, 0])
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                v = counts[i, j]
                if v > 0:
                    color = "black" if v > counts.max() * 0.5 else "white"
                    ax.text((i + 0.5) / g, (j + 0.5) / g, str(v), ha="center", va="center", color=color, fontsize=9)
        ax.set_xlabel("x_norm")
        ax.set_ylabel("y_norm (top → bottom)")
        ax.set_title(f"click density (n={n}, {g}×{g} grid)")
        plt.colorbar(im, ax=ax, label="samples")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\n[analyze] heatmap → {out}")
    except ImportError:
        print("[analyze] matplotlib not installed, skipped PNG output")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
