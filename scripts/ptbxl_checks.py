"""Match competition records to PTB-XL 1.0.3 and check patient overlap between train and development records.

The match table is also the input of `crossecg.prepare --ptbxl-matches` (500 Hz labels for the ablation cache).
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

from crossecg.prepare import split_records
from crossecg.zip_io import CompetitionSource


def load_ptbxl(path: Path) -> tuple[list[str], np.ndarray]:
    """All 500 Hz PTB-XL records as [N, 5000, 12] mV, read from the zip."""
    archive = zipfile.ZipFile(path)
    headers = sorted(n for n in archive.namelist() if re.search(r"records500/.*_hr\.hea$", n))
    names, data = [], []
    for header in headers:
        lines = archive.read(header).decode().splitlines()
        gains = [float(line.split()[2].split("/")[0].split("(")[0]) for line in lines[1:13]]
        raw = np.frombuffer(archive.read(header[:-4] + ".dat"), dtype="<i2").reshape(-1, 12)
        names.append(Path(header).stem)
        data.append((raw / np.array(gains)).astype(np.float32))
    return names, np.stack(data)


def match_records(source: CompetitionSource, names: list[str], ptb: np.ndarray) -> pd.DataFrame:
    """Best PTB-XL record for each competition record by lead II correlation at 100 Hz."""
    bank = ptb[:, ::5, 1]
    bank = (bank - bank.mean(1, keepdims=True)) / (bank.std(1, keepdims=True) + 1e-6)
    rows = []
    for record in source.records():
        query = resample(np.nan_to_num(source.signals(record.record_id)["II"]), 1000)
        query = (query - query.mean()) / (query.std() + 1e-6)
        corr = bank @ query / 1000
        best = int(np.argmax(corr))
        rows.append({"record_id": record.record_id, "fs": record.fs, "ptb": names[best], "ptb_index": best, "corr": float(corr[best])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--ptbxl", required=True, help="PTB-XL 1.0.3 zip from PhysioNet")
    parser.add_argument("--output", default="outputs/ptbxl_matches.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    arguments = parser.parse_args()
    source = CompetitionSource(arguments.source)
    names, ptb = load_ptbxl(Path(arguments.ptbxl))
    matches = match_records(source, names, ptb)
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(arguments.output, index=False)

    database = pd.read_csv(zipfile.ZipFile(arguments.ptbxl).open(
        next(n for n in zipfile.ZipFile(arguments.ptbxl).namelist() if n.endswith("ptbxl_database.csv"))))
    database["ptb"] = database.filename_hr.str.split("/").str[-1]
    matches = matches.merge(database[["ptb", "patient_id"]], on="ptb", how="left")
    split = split_records(matches.record_id.tolist(), arguments.validation_fraction, arguments.seed)
    matches["development"] = matches.record_id.map(split) == "validation"
    shared = set(matches.patient_id[matches.development]) & set(matches.patient_id[~matches.development])
    print("matched (corr > 0.99):", int((matches["corr"] > 0.99).sum()), "of", len(matches))
    print("development records sharing a patient with training:",
          matches.record_id[matches.development & matches.patient_id.isin(shared)].tolist())


if __name__ == "__main__":
    main()
