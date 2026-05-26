"""Dataset loaders.

Three loader flavors:

1. `LabelFileDataset` — generic: CSV/TSV with columns (image_path, yaw_deg, pitch_deg)
   Use for self-collected data or any preprocessed dataset.

2. `Gaze360Dataset` — official Gaze360 split files (train.txt / val.txt / test.txt)
   Format per line: <image_path> <gaze_x> <gaze_y> <gaze_z>

3. `MPIIFaceGazeDataset` — official MPIIFaceGaze label format
   Per-subject .label files with normalized face images + 2D gaze in head coords.

All three return (image_tensor, yaw_deg, pitch_deg) so the training loop is uniform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gaze.data.augmentation import AugmentedSample, GazeAwareAugment, GazeEvalTransform


TransformFn = Callable[[Image.Image, float, float], AugmentedSample]


@dataclass
class GazeRecord:
    image_path: Path
    yaw_deg: float
    pitch_deg: float


def gaze_vector_to_yaw_pitch_deg(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert a 3D gaze direction vector to (yaw, pitch) in degrees.

    Convention matches Gaze360 / MPIIFaceGaze:
      yaw   = atan2(-x, -z)   (positive = looking right)
      pitch = asin(-y)        (positive = looking up)
    """
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / norm, y / norm, z / norm
    yaw = math.degrees(math.atan2(-x, -z))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -y))))
    return yaw, pitch


class LabelFileDataset(Dataset):
    """Generic CSV/TSV reader.

    File format (1 header line, then rows):
        image_path,yaw_deg,pitch_deg
        /abs/or/relative/path.jpg,12.3,-4.5
        ...
    """

    def __init__(
        self,
        label_file: str | Path,
        image_root: str | Path | None = None,
        transform: TransformFn | None = None,
        delimiter: str = ",",
    ) -> None:
        self.label_file = Path(label_file)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform or GazeEvalTransform()
        self.records: list[GazeRecord] = []
        self._load(delimiter)

    def _load(self, delimiter: str) -> None:
        with self.label_file.open() as f:
            header = f.readline().strip().split(delimiter)
            try:
                ipath_idx = header.index("image_path")
                yaw_idx = header.index("yaw_deg")
                pitch_idx = header.index("pitch_deg")
            except ValueError as e:
                raise ValueError(f"label file {self.label_file} missing required columns: {e}") from e
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(delimiter)
                p = Path(cols[ipath_idx])
                if self.image_root and not p.is_absolute():
                    p = self.image_root / p
                self.records.append(
                    GazeRecord(image_path=p, yaw_deg=float(cols[yaw_idx]), pitch_deg=float(cols[pitch_idx]))
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        rec = self.records[idx]
        img = Image.open(rec.image_path).convert("RGB")
        sample = self.transform(img, rec.yaw_deg, rec.pitch_deg)
        return sample.image, sample.yaw_deg, sample.pitch_deg


class Gaze360Dataset(Dataset):
    """Gaze360 official split format: `<image_relative_path> <gx> <gy> <gz>` per line."""

    def __init__(
        self,
        split_file: str | Path,
        image_root: str | Path,
        transform: TransformFn | None = None,
    ) -> None:
        self.split_file = Path(split_file)
        self.image_root = Path(image_root)
        self.transform = transform or GazeEvalTransform()
        self.records: list[GazeRecord] = []
        self._load()

    def _load(self) -> None:
        with self.split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                rel, gx, gy, gz = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                yaw, pitch = gaze_vector_to_yaw_pitch_deg(gx, gy, gz)
                self.records.append(
                    GazeRecord(image_path=self.image_root / rel, yaw_deg=yaw, pitch_deg=pitch)
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        rec = self.records[idx]
        img = Image.open(rec.image_path).convert("RGB")
        sample = self.transform(img, rec.yaw_deg, rec.pitch_deg)
        return sample.image, sample.yaw_deg, sample.pitch_deg


class MPIIFaceGazeDataset(Dataset):
    """MPIIFaceGaze normalized format.

    Each subject has a .label file. Each row:
        <img_path> <gaze_pitch_rad> <gaze_yaw_rad> <head_pitch_rad> <head_yaw_rad> ...

    We pull gaze_yaw / gaze_pitch and convert to degrees.
    """

    def __init__(
        self,
        label_files: list[str | Path],
        image_root: str | Path,
        transform: TransformFn | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform or GazeEvalTransform()
        self.records: list[GazeRecord] = []
        for lf in label_files:
            self._load(Path(lf))

    def _load(self, label_file: Path) -> None:
        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rel = parts[0]
                pitch_rad = float(parts[1])
                yaw_rad = float(parts[2])
                self.records.append(
                    GazeRecord(
                        image_path=self.image_root / rel,
                        yaw_deg=math.degrees(yaw_rad),
                        pitch_deg=math.degrees(pitch_rad),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        rec = self.records[idx]
        img = Image.open(rec.image_path).convert("RGB")
        sample = self.transform(img, rec.yaw_deg, rec.pitch_deg)
        return sample.image, sample.yaw_deg, sample.pitch_deg


@dataclass
class ClickRecord:
    image_path: Path
    click_x: float
    click_y: float
    screen_w: int
    screen_h: int


ClickTransformFn = Callable[[Image.Image, float, float, int, int], "ClickSample"]


@dataclass
class ClickSample:
    image: "torch.Tensor"
    x_norm: float  # click_x / screen_w
    y_norm: float  # click_y / screen_h
    screen_w: int
    screen_h: int


class ClickLabelDataset(Dataset):
    """Click-as-label dataset for end-to-end screen-coord training.

    Reads the CSV produced by `scripts/live_collect.py`:
        image_path, click_x, click_y, screen_w, screen_h, timestamp_iso

    Returns (face_image_tensor, x_norm, y_norm, screen_w, screen_h).
    The label is the *normalized* click position, so the model output space
    is device-independent. Screen dimensions are returned alongside so the
    training loop can compute pixel-space errors for monitoring.
    """

    def __init__(
        self,
        label_file: str | Path,
        image_root: str | Path | None = None,
        transform: TransformFn | None = None,
        hflip_prob: float = 0.5,  # gaze-aware flip — also negates x_norm
        input_size: int = 256,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        train: bool = True,
    ) -> None:
        import csv

        self.label_file = Path(label_file)
        self.image_root = Path(image_root) if image_root else self.label_file.parent
        self.records: list[ClickRecord] = []
        with self.label_file.open() as f:
            reader = csv.DictReader(f)
            required = {"image_path", "click_x", "click_y", "screen_w", "screen_h"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"click CSV {self.label_file} missing columns: {missing}")
            for row in reader:
                p = Path(row["image_path"])
                if not p.is_absolute():
                    p = self.image_root / p
                self.records.append(
                    ClickRecord(
                        image_path=p,
                        click_x=float(row["click_x"]),
                        click_y=float(row["click_y"]),
                        screen_w=int(row["screen_w"]),
                        screen_h=int(row["screen_h"]),
                    )
                )

        from torchvision import transforms as T

        ops = [
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transform = T.Compose(ops)
        self.hflip_prob = hflip_prob if train else 0.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        rec = self.records[idx]
        img = Image.open(rec.image_path).convert("RGB")
        x_norm = rec.click_x / rec.screen_w
        y_norm = rec.click_y / rec.screen_h

        # Horizontal flip is the only safe gaze-preserving spatial aug:
        # flip the face image AND flip x_norm around 0.5.
        if np.random.rand() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_norm = 1.0 - x_norm

        tensor = self.transform(img)
        return tensor, float(x_norm), float(y_norm), rec.screen_w, rec.screen_h


def build_dataset(cfg: dict, train: bool) -> Dataset:
    """Factory dispatched on cfg['type']."""
    kind = cfg["type"]
    transform: TransformFn
    if train:
        transform = GazeAwareAugment(
            input_size=cfg.get("input_size", 256),
            mean=tuple(cfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(cfg.get("std", (0.5, 0.5, 0.5))),
            hflip_prob=cfg.get("hflip_prob", 0.5),
            color_jitter=cfg.get("color_jitter", 0.2),
        )
    else:
        transform = GazeEvalTransform(
            input_size=cfg.get("input_size", 256),
            mean=tuple(cfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(cfg.get("std", (0.5, 0.5, 0.5))),
        )

    if kind == "label_file":
        return LabelFileDataset(
            label_file=cfg["label_file_train" if train else "label_file_val"],
            image_root=cfg.get("image_root"),
            transform=transform,
            delimiter=cfg.get("delimiter", ","),
        )
    if kind == "gaze360":
        return Gaze360Dataset(
            split_file=cfg["split_train" if train else "split_val"],
            image_root=cfg["image_root"],
            transform=transform,
        )
    if kind == "mpiifacegaze":
        return MPIIFaceGazeDataset(
            label_files=cfg["label_files_train" if train else "label_files_val"],
            image_root=cfg["image_root"],
            transform=transform,
        )
    if kind == "clicks":
        return ClickLabelDataset(
            label_file=cfg["label_file_train" if train else "label_file_val"],
            image_root=cfg.get("image_root"),
            hflip_prob=cfg.get("hflip_prob", 0.5),
            input_size=cfg.get("input_size", 256),
            mean=tuple(cfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(cfg.get("std", (0.5, 0.5, 0.5))),
            train=train,
        )
    raise ValueError(f"unknown dataset type: {kind}")
