"""Stream records from the competition zip without extracting it."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


@dataclass(frozen=True)
class RecordInfo:
    record_id: str
    fs: float
    sig_len: int


class CompetitionSource:
    """Reads either the original zip or an extracted folder with the same layout."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.archive = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None
        self.names = set(self.archive.namelist()) if self.archive else None

    def read_bytes(self, name: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(name)
        return (self.path / name).read_bytes()

    def exists(self, name: str) -> bool:
        return name in self.names if self.names is not None else (self.path / name).exists()

    def records(self) -> list[RecordInfo]:
        frame = pd.read_csv(io.BytesIO(self.read_bytes("train.csv")))
        return [RecordInfo(str(row.id), float(row.fs), int(row.sig_len)) for row in frame.itertuples()]

    def image_types(self, record_id: str) -> list[str]:
        return [kind for kind in IMAGE_TYPES if self.exists(f"train/{record_id}/{record_id}-{kind}.png")]

    def image(self, record_id: str, kind: str) -> np.ndarray:
        data = np.frombuffer(self.read_bytes(f"train/{record_id}/{record_id}-{kind}.png"), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {record_id}-{kind}")
        return image

    def signals(self, record_id: str) -> dict[str, np.ndarray]:
        frame = pd.read_csv(io.BytesIO(self.read_bytes(f"train/{record_id}/{record_id}.csv")))
        return {column: frame[column].to_numpy(dtype=np.float64) for column in frame.columns}
