"""Convert decoded series-crop rows into the 12 scored lead signals."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded

from .layout import EDGE_MARGIN_SECONDS, PIXELS_PER_MV, PIXELS_PER_SECOND, SERIES_TIME_OFFSET_X, CropGeometry, lead_windows
from .metric import LEADS


def fill_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not valid.any():
        return np.zeros_like(values)
    index = np.arange(len(values))
    return np.interp(index, index[valid], values[valid])


def segment_bounds(sig_len: int, index: int, parts: int = 4) -> tuple[int, int]:
    return (index * sig_len) // parts, ((index + 1) * sig_len) // parts


def samples_from_columns(column_times: np.ndarray, values: np.ndarray, sample_times: np.ndarray, smoothness: float = 1e-3,
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Invert the polyline rendering: columns are linear interpolations of the samples.

    Solves min ||A s - values||^2 + smoothness * ||D2 s||^2, which recovers sharp
    peaks that interpolating columns back to sample times would cut off.
    """
    count = len(sample_times)
    if count < 3:
        return np.interp(sample_times, column_times, values)
    step = sample_times[1] - sample_times[0]
    position = (column_times - sample_times[0]) / step
    keep = (position >= 0) & (position <= count - 1)
    weights = np.ones(len(position)) if weights is None else np.asarray(weights, dtype=np.float64)
    position, values, weights = position[keep], values[keep], weights[keep]
    left = np.minimum(np.floor(position).astype(np.int64), count - 2)
    fraction = position - left
    # Normal equations: A^T A is tridiagonal, D2^T D2 pentadiagonal -> banded upper form, 3 superdiagonals.
    bands = np.zeros((3, count))
    np.add.at(bands[2], left, weights * (1 - fraction) ** 2)
    np.add.at(bands[2], left + 1, weights * fraction**2)
    np.add.at(bands[1], left + 1, weights * (1 - fraction) * fraction)
    rhs = np.zeros(count)
    np.add.at(rhs, left, weights * (1 - fraction) * values)
    np.add.at(rhs, left + 1, weights * fraction * values)
    second = np.zeros((3, count))
    second[2] = 6.0
    second[2][[0, -1]] = 1.0
    second[2][[1, -2]] = 5.0
    second[1, 1:] = -4.0
    second[1, [1, -1]] = -2.0
    second[0, 2:] = 1.0
    scale = max(float(bands[2].mean()), 1e-6)
    return solveh_banded(bands + smoothness * scale * second, rhs)


def series_to_leads(
    rows: np.ndarray,
    valid: np.ndarray,
    fs: float,
    sig_len: int,
    geometry: CropGeometry,
    time_offsets: tuple[float, ...] = SERIES_TIME_OFFSET_X,
    edge_margin: float = EDGE_MARGIN_SECONDS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return per-lead mV signals at the scored sample times and the full-length short II.

    ``rows``/``valid`` are [4, W] crop rows and accepted columns; ``time_offsets`` are
    per-series page-pixel shifts of the drawn ink relative to the nominal time axis.
    """
    leads: dict[str, np.ndarray] = {}
    short_ii = None
    for series in range(rows.shape[0]):
        times = geometry.column_times() - time_offsets[series] / PIXELS_PER_SECOND
        millivolts = (0.5 * geometry.height - fill_gaps(rows[series], valid[series])) / (
            geometry.vertical_scale * PIXELS_PER_MV
        )
        windows = lead_windows(series)
        for index, (lead, start, end) in enumerate(windows):
            first, last = (0, sig_len) if len(windows) == 1 else segment_bounds(sig_len, index)
            sample_times = np.arange(first, last) / fs
            # Separator bars and the calibration pulse sit at segment edges; extend the interior instead.
            interior = (times >= start + edge_margin) & (times <= end - edge_margin)
            held = np.interp(times, times[interior], millivolts[interior]) if edge_margin > 0 else millivolts
            values = samples_from_columns(times, held, sample_times)
            if series == 1 and lead == "II":
                short_ii = values
            else:
                leads[lead] = values
    return leads, short_ii


def fuse_lead_ii(leads: dict[str, np.ndarray], short_ii: np.ndarray | None, weight: float = 0.5) -> None:
    """Average the first 2.5 s of the rhythm strip with the short lead II panel."""
    if short_ii is None:
        return
    count = len(short_ii)
    rhythm = leads["II"]
    offset = np.median(rhythm[:count] - short_ii[:count])
    rhythm[:count] = (1.0 - weight) * rhythm[:count] + weight * (short_ii[:count] + offset)


def einthoven_project(leads: dict[str, np.ndarray]) -> None:
    """Least-squares projection of the six limb leads onto the 2-D Einthoven subspace per segment.

    Limb leads are drawn in different 2.5 s windows except II, so the constraint only
    links leads sharing a window: (I, II, III) in 0-2.5 s and (aVR, aVL, aVF) in 2.5-5 s.
    Only the rank-deficient relations within a window are used.
    """
    first = slice(0, len(leads["I"]))
    i, ii, iii = leads["I"], leads["II"][first], leads["III"]
    if len(i) == len(iii) == len(ii):
        residual = (i - i.mean()) + (iii - iii.mean()) - (ii - ii.mean())
        leads["I"] = i - residual / 3.0
        leads["III"] = iii - residual / 3.0
        leads["II"][first] = leads["II"][first] + residual / 3.0
    avr, avl, avf = leads["aVR"], leads["aVL"], leads["aVF"]
    if len(avr) == len(avl) == len(avf):
        residual = (avr - avr.mean()) + (avl - avl.mean()) + (avf - avf.mean())
        leads["aVR"], leads["aVL"], leads["aVF"] = (avr - residual / 3.0, avl - residual / 3.0, avf - residual / 3.0)


def complete_leads(leads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {lead: np.nan_to_num(np.asarray(leads[lead], dtype=np.float64)) for lead in LEADS}
