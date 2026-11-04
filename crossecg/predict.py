"""Full-width strip inference, 12-lead assembly, and official-metric evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional

from .dataset import CacheIndex, to_input
from .decoding import decode_ordinate_logits
from .layout import KIND_GROUP, SERIES_TIME_OFFSET_X, CropGeometry, time_offsets_for, truth_segments
from .metric import image_snr, score_db
from .signals import complete_leads, einthoven_project, fuse_lead_ii, series_to_leads


@torch.no_grad()
def predict_rows(model, crops: list[np.ndarray], device: torch.device, top_k: int = 5):
    """Decoded row per column and accepted mask for the four strips of a page; crossing outputs are ignored."""
    rows, accepted = [], []
    for crop in crops:
        image = to_input(crop)[None].to(device)
        width = image.shape[-1]
        padded = functional.pad(image, (0, (-width) % 32, 0, 0), mode="replicate")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = {key: value.float() for key, value in model(padded).items()}
        decoded = decode_ordinate_logits(outputs["ordinate_logits"][..., :width], outputs["existence_logits"][..., :width],
                                         top_k=top_k)
        rows.append(decoded.y[0].cpu().numpy())
        accepted.append(decoded.accepted[0].cpu().numpy())
    return np.stack(rows), np.stack(accepted)


def assemble(rows: np.ndarray, valid: np.ndarray, fs: float, sig_len: int, geometry: CropGeometry, post: bool = True,
             time_offsets: tuple[float, ...] = SERIES_TIME_OFFSET_X) -> dict[str, np.ndarray]:
    leads, short_ii = series_to_leads(rows, valid, fs, sig_len, geometry, time_offsets)
    if post:
        fuse_lead_ii(leads, short_ii)
        einthoven_project(leads)
    return complete_leads(leads)


def evaluate_cache(model, index: CacheIndex, signals_loader, device: torch.device, limit: int | None = None) -> dict[str, object]:
    """Official per-image SNR over cached development crops, overall and per image type."""
    geometry = CropGeometry(**index.geometry)
    model.eval()
    per_kind: dict[str, list[float]] = defaultdict(list)
    count = len(index) if limit is None else min(limit, len(index))
    for item in range(count):
        row = index.frame.iloc[item]
        rows, valid = predict_rows(model, [index.crop(item, series) for series in range(4)], device)
        prediction = assemble(rows, valid, float(row.fs), int(row.sig_len), geometry, True,
                              time_offsets_for(KIND_GROUP.get(row.kind)))
        truth = truth_segments(signals_loader(row.record_id))
        per_kind[row.kind].append(image_snr(truth, prediction, float(row.fs)))
    scores = [value for values in per_kind.values() for value in values]
    return {
        "snr_db": score_db(scores),
        "images": len(scores),
        "per_kind_db": {kind: round(score_db(values), 3) for kind, values in sorted(per_kind.items())},
    }
