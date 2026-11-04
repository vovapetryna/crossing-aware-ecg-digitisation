"""Original 500 Hz PTB-XL signals for competition records (matched by lead-II correlation)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metric import LEADS


class PtbxlLabels:
    def __init__(self, zip_path: str | Path, matches_csv: str | Path, min_corr: float = 0.99):
        self.archive = zipfile.ZipFile(zip_path)
        self.headers = {Path(n).stem: n for n in self.archive.namelist() if re.search(r"records500/.*_hr\.hea$", n)}
        matches = pd.read_csv(matches_csv, dtype={"record_id": str})
        self.matches = dict(zip(matches.record_id[matches["corr"] > min_corr], matches.ptb[matches["corr"] > min_corr]))

    def record(self, name: str) -> np.ndarray:
        header = self.headers[name]
        lines = self.archive.read(header).decode().splitlines()
        gains = [float(line.split()[2].split("/")[0].split("(")[0]) for line in lines[1:13]]
        raw = np.frombuffer(self.archive.read(header[:-4] + ".dat"), dtype="<i2").reshape(-1, 12)
        return raw / np.array(gains)

    def signals(self, record_id: str, csv: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        """500 Hz signals laid out with the CSV's per-lead time windows, or None if unmatched."""
        name = self.matches.get(record_id)
        if name is None:
            return None
        original = self.record(name)
        output = {}
        for index, lead in enumerate(LEADS):
            values = original[:, index].astype(np.float64)
            if lead == "II":
                output[lead] = values
                continue
            finite = np.isfinite(csv[lead])
            first = np.flatnonzero(finite)[0] / len(finite) * len(values)
            last = (np.flatnonzero(finite)[-1] + 1) / len(finite) * len(values)
            mask = np.zeros(len(values), bool)
            mask[int(round(first)):int(round(last))] = True
            output[lead] = np.where(mask, values, np.nan)
        return output
