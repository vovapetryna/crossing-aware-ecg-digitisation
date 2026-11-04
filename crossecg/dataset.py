"""Cached series crops and random training windows."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def coordinate_channel(height: int, width: int) -> np.ndarray:
    return np.broadcast_to(np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None], (height, width))


def to_input(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 crop -> [4, H, W] RGB/255 plus a vertical-coordinate channel."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    stacked = np.concatenate((rgb.transpose(2, 0, 1), coordinate_channel(height, width)[None]), axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def two_row_target(y: np.ndarray, mask: np.ndarray, height: int) -> np.ndarray:
    """Distribute unit mass between floor(y) and floor(y)+1 by the fractional part."""
    target = np.zeros((height, len(y)), dtype=np.float32)
    columns = np.flatnonzero(mask)
    lower = np.clip(np.floor(y[columns]).astype(np.int64), 0, height - 1)
    upper = np.clip(lower + 1, 0, height - 1)
    fraction = np.clip(y[columns] - lower, 0.0, 1.0).astype(np.float32)
    target[lower, columns] += 1.0 - fraction
    target[upper, columns] += fraction
    return target


def crossing_targets(y: np.ndarray, mask: np.ndarray, height: int, min_step: float = 2.0, band: int = 6):
    """Row-reading targets for steep strokes.

    Between columns c and c+1 whose trace rows differ by >= ``min_step``, the polyline crosses
    every integer row level in between at a fractional column. Pixel (row, c) gets crossing=1 and
    offset = fractional position in [0, 1). ``supervised`` marks the band around the stroke where
    the crossing BCE is applied (neighbouring traces elsewhere stay unsupervised).
    """
    width = len(y)
    crossing = np.zeros((height, width), np.float32)
    offset = np.zeros((height, width), np.float32)
    supervised = np.zeros((height, width), bool)
    steep = np.flatnonzero(mask[:-1] & mask[1:] & (np.abs(np.diff(y)) >= min_step))
    for column in steep:
        y0, y1 = y[column], y[column + 1]
        low, high = min(y0, y1), max(y0, y1)
        supervised[max(int(low) - band, 0):min(int(high) + band + 1, height), column] = True
        rows = np.arange(int(np.ceil(low)), int(np.floor(high)) + 1)
        rows = rows[(rows > low) & (rows < high) & (rows >= 0) & (rows < height)]
        if len(rows) == 0:
            continue
        crossing[rows, column] = 1.0
        offset[rows, column] = (rows - y0) / (y1 - y0)
    return crossing, offset, supervised


class CacheIndex:
    """manifest.csv rows: one image with 4 series crops and their labels."""

    def __init__(self, root: str | Path, split: str, kinds: tuple[str, ...] | None = None, min_contrast: float = 0.0,
                 label_stride: int = 1):
        self.root = Path(root)
        frame = pd.concat([pd.read_csv(path, dtype={"record_id": str, "kind": str})
                           for path in sorted(self.root.glob("manifest*.csv")) if "before" not in path.name])
        frame = frame[(frame.split == split) & (frame.status == "ok")]
        if kinds:
            frame = frame[frame.kind.isin(kinds)]
        if min_contrast:
            frame = frame[frame.contrast >= min_contrast]
        self.frame = frame.reset_index(drop=True)
        self.geometry = json.loads((self.root / "geometry.json").read_text())
        cache_info = self.root / "cache.json"
        self.upsample = json.loads(cache_info.read_text())["upsample"] if cache_info.exists() else 1.0
        # label_stride > 1 reads the cache at a coarser horizontal scale: fewer label columns, less upsampling.
        self.label_stride = label_stride
        if label_stride > 1:
            self.geometry = {**self.geometry, "horizontal_scale": self.geometry["horizontal_scale"] / label_stride}
            self.upsample = self.upsample / label_stride

    def __len__(self) -> int:
        return len(self.frame)

    def image_dir(self, index: int) -> Path:
        row = self.frame.iloc[index]
        return self.root / "images" / f"{row.record_id}-{row.kind}"

    def crop(self, index: int, series: int) -> np.ndarray:
        image = cv2.imread(str(self.image_dir(index) / f"s{series}.jpg"), cv2.IMREAD_COLOR)
        if self.upsample != 1.0:
            width = int(round((self.geometry["x_end"] - self.geometry["x_start"]) * self.geometry["horizontal_scale"]))
            image = cv2.resize(image, (width, image.shape[0]), interpolation=cv2.INTER_CUBIC)
        return image

    def labels(self, index: int) -> dict[str, np.ndarray]:
        with np.load(self.image_dir(index) / "labels.npz") as data:
            labels = {key: data[key] for key in data.files}
        if self.label_stride > 1:
            labels = {key: value[:, :: self.label_stride] for key, value in labels.items()}
        return labels


class WindowDataset(Dataset):
    """Random fixed-width windows of series crops with photometric augmentation."""

    def __init__(self, index: CacheIndex, window: int, samples: int, seed: int = 0, augment: bool = True,
                 mask_key: str = "ordinate_mask", crossing: bool = False):
        self.index = index
        self.mask_key = mask_key
        self.crossing = crossing
        self.window = window
        self.samples = samples
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, item, torch.initial_seed() % (2**31)))
        image_index = int(rng.integers(len(self.index)))
        series = int(rng.integers(4))
        crop = self.index.crop(image_index, series)
        labels = self.index.labels(image_index)
        height, width = crop.shape[:2]
        y, mask, exists = labels["y"][series].copy(), labels[self.mask_key][series], labels["exists"][series]
        start = int(rng.integers(0, max(width - self.window, 0) + 1))
        stop = start + self.window
        crop, y, mask, exists = crop[:, start:stop], y[start:stop], mask[start:stop], exists[start:stop]
        if self.augment:
            crop, y = augment(crop, y, rng)
        mask = mask & (y >= 0) & (y <= height - 1)
        extra = {}
        if self.crossing:
            crossing, offset, supervised = crossing_targets(y, mask, height)
            extra = {"crossing": torch.from_numpy(crossing), "crossing_offset": torch.from_numpy(offset),
                     "crossing_mask": torch.from_numpy(supervised)}
        return {
            **extra,
            "image": to_input(crop),
            "ordinate_target": torch.from_numpy(two_row_target(y, mask, height)),
            "ordinate_y": torch.from_numpy(y.astype(np.float32)),
            "ordinate_mask": torch.from_numpy(mask),
            "path_exists": torch.from_numpy(exists.astype(np.float32)),
        }


def augment(crop: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shift = float(rng.uniform(-12, 12))
    height, width = crop.shape[:2]
    crop = cv2.warpAffine(
        crop, np.float32([[1, 0, 0], [0, 1, shift]]), (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    y = y + shift
    image = crop.astype(np.float32)
    if rng.random() < 0.8:
        image = image * rng.uniform(0.75, 1.25) + rng.uniform(-30, 30)
        image = image * rng.uniform(0.9, 1.1, size=(1, 1, 3))
    if rng.random() < 0.2:
        image = image.mean(axis=2, keepdims=True).repeat(3, axis=2)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        image = image + rng.normal(0, rng.uniform(2, 10), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), y
