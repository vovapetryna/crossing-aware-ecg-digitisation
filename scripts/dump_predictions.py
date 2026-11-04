"""Save a checkpoint's per-page lead predictions and reference signals for scoring and figures."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from crossecg.dataset import CacheIndex
from crossecg.layout import KIND_GROUP, CropGeometry, time_offsets_for, truth_segments
from crossecg.predict import assemble, predict_rows
from crossecg.train import build_model
from crossecg.zip_io import CompetitionSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--source", required=True, help="competition zip or extracted folder")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"], device, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    index = CacheIndex(arguments.cache_dir, arguments.split, label_stride=int(checkpoint["config"].get("label_stride", 1)))
    geometry = CropGeometry(**index.geometry)
    source = CompetitionSource(arguments.source)
    pages = []
    for item in range(len(index)):
        row = index.frame.iloc[item]
        fs, sig_len = float(row.fs), int(row.sig_len)
        rows, valid = predict_rows(model, [index.crop(item, s) for s in range(4)], device)
        offsets = time_offsets_for(KIND_GROUP[row.kind])
        pages.append({
            "record_id": row.record_id, "kind": row.kind, "fs": fs, "sig_len": sig_len,
            "rows": rows.astype(np.float32), "valid": valid, "offsets": offsets,
            "post": assemble(rows, valid, fs, sig_len, geometry, True, offsets),
            "truth": truth_segments(source.signals(row.record_id)),
        })
        if item % 100 == 0:
            print(item, len(index), flush=True)
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.output, "wb") as handle:
        pickle.dump(pages, handle)
    print("saved", len(pages))


if __name__ == "__main__":
    main()
