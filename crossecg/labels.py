"""Series-crop ordinate targets from ground-truth CSV signals, plus ink registration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .layout import PIXELS_PER_MV, PIXELS_PER_SECOND, CropGeometry, lead_windows

BOUNDARY_SECONDS = 0.02


@dataclass
class SeriesTarget:
    y: np.ndarray  # [W] float crop row of the latent trace
    ordinate_mask: np.ndarray  # [W] bool, supervise the ordinate here
    exists: np.ndarray  # [W] bool, a trace should be emitted here


def lead_samples(signals: dict[str, np.ndarray], lead: str, fs: float, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signals[lead], dtype=np.float64)
    index = np.arange(len(values))
    times = index / fs
    keep = np.isfinite(values) & (times >= start - 0.5 / fs) & (times < end)
    return times[keep], values[keep]


def series_target(
    signals: dict[str, np.ndarray],
    fs: float,
    series: int,
    geometry: CropGeometry,
    dx: float = 0.0,
    dy: float = 0.0,
) -> SeriesTarget:
    """``dx``/``dy`` are page-pixel offsets of the drawn ink from the nominal layout."""
    width = geometry.width
    times = geometry.column_times() - dx / PIXELS_PER_SECOND
    y = np.full(width, np.nan)
    exists = np.zeros(width, dtype=bool)
    ordinate_mask = np.zeros(width, dtype=bool)
    for lead, start, end in lead_windows(series):
        sample_times, values = lead_samples(signals, lead, fs, start, end)
        if len(values) < 2:
            continue
        inside = (times >= sample_times[0]) & (times <= sample_times[-1])
        y[inside] = (
            0.5 * geometry.height
            + (dy - PIXELS_PER_MV * np.interp(times[inside], sample_times, values)) * geometry.vertical_scale
        )
        exists |= inside
        ordinate_mask |= inside & (times >= start + BOUNDARY_SECONDS) & (times <= end - BOUNDARY_SECONDS)
    ordinate_mask &= np.isfinite(y) & (y >= 0) & (y <= geometry.height - 1)
    return SeriesTarget(np.nan_to_num(y, nan=0.0).astype(np.float32), ordinate_mask, exists)


def ink_map(crop: np.ndarray) -> np.ndarray:
    """Soft darkness relative to the local paper background, in [0, 1]."""
    darkest = crop.min(axis=2).astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    background = cv2.dilate(darkest, np.ones((1, 15), np.uint8))
    background = cv2.medianBlur(background.astype(np.uint8), 31).astype(np.float32)
    contrast = np.clip((background - darkest) / np.maximum(background, 32.0), 0.0, 1.0)
    return cv2.GaussianBlur(contrast, (0, 0), 1.0)


def path_score(ink: np.ndarray, target: SeriesTarget) -> float:
    columns = np.flatnonzero(target.ordinate_mask)
    if len(columns) == 0:
        return 0.0
    return float(map_coordinates(ink, [target.y[columns], columns], order=1).mean())


@dataclass
class Registration:
    dx: tuple[float, ...]
    dy: tuple[float, ...]
    score: float
    contrast: float


def _refine(score, center: float, half_width: float, step: float) -> float:
    """Grid search plus parabolic interpolation of the peak."""
    candidates = np.arange(center - half_width, center + half_width + step / 2, step)
    values = np.array([score(value) for value in candidates])
    best = int(np.argmax(values))
    if 0 < best < len(values) - 1:
        left, middle, right = values[best - 1:best + 2]
        curvature = left - 2 * middle + right
        if curvature < 0:
            return float(candidates[best] + 0.5 * step * (left - right) / curvature)
    return float(candidates[best])


def register_from_prior(
    crops: list[np.ndarray],
    signals: dict[str, np.ndarray],
    fs: float,
    geometry: CropGeometry,
    prior_dx: tuple[float, ...],
    prior_dy: tuple[float, ...],
    half_dx: float = 1.0,
    half_dy: float = 3.0,
) -> Registration:
    """Per-series refinement around known offsets; ~4x cheaper than the global search in ``register``."""
    inks = [ink_map(crop) for crop in crops]
    dxs, dys, contrasts = list(prior_dx), list(prior_dy), []
    for series, ink in enumerate(inks):
        def score_at(dx: float, dy: float) -> float:
            return path_score(ink, series_target(signals, fs, series, geometry, dx, dy))

        dys[series] = _refine(lambda value: score_at(dxs[series], value), dys[series], half_dy, 0.5)
        grid = np.arange(dxs[series] - half_dx, dxs[series] + half_dx + 0.01, 0.25)
        grid_scores = [score_at(value, dys[series]) for value in grid]
        dxs[series] = float(grid[int(np.argmax(grid_scores))])
        for _ in range(2):
            dxs[series] = _refine(lambda value: score_at(value, dys[series]), dxs[series], 0.3, 0.05)
            dys[series] = _refine(lambda value: score_at(dxs[series], value), dys[series], 0.6, 0.1)
        contrasts.append(score_at(dxs[series], dys[series]) / max(float(np.median(grid_scores)), 1e-3))
    score = float(np.mean([
        path_score(ink, series_target(signals, fs, series, geometry, dxs[series], dys[series]))
        for series, ink in enumerate(inks)
    ]))
    return Registration(tuple(dxs), tuple(dys), score, float(np.mean(contrasts)))


def register(
    crops: list[np.ndarray],
    signals: dict[str, np.ndarray],
    fs: float,
    geometry: CropGeometry,
    max_dx: float = 16.0,
    max_dy: float = 16.0,
) -> Registration:
    """Find the shared time offset and per-series vertical offsets of the drawn trace."""
    inks = [ink_map(crop) for crop in crops]

    def total(dx: float, dys: list[float]) -> float:
        return float(np.mean([
            path_score(ink, series_target(signals, fs, series, geometry, dx, dys[series]))
            for series, ink in enumerate(inks)
        ]))

    def best_dy(series: int, dx: float, candidates: np.ndarray) -> float:
        scores = [path_score(inks[series], series_target(signals, fs, series, geometry, dx, dy)) for dy in candidates]
        return float(candidates[int(np.argmax(scores))])

    dys = [0.0] * len(crops)
    coarse_dy = np.arange(-max_dy, max_dy + 0.01, 2.0)
    rhythm = len(crops) - 1
    dx_candidates = np.arange(-max_dx, max_dx + 0.01, 2.0)
    dx_scores = []
    for dx in dx_candidates:
        dy = best_dy(rhythm, dx, coarse_dy)
        dx_scores.append(path_score(inks[rhythm], series_target(signals, fs, rhythm, geometry, dx, dy)))
    dx = float(dx_candidates[int(np.argmax(dx_scores))])
    for _ in range(2):
        dys = [best_dy(series, dx, np.arange(-max_dy, max_dy + 0.01, 0.5)) for series in range(len(crops))]
        fine = np.arange(dx - 2.0, dx + 2.01, 0.25)
        dx = float(fine[int(np.argmax([total(value, dys) for value in fine]))])
    # Rectification leaves sub-pixel, row-dependent time offsets; refine each series separately.
    dxs = [dx] * len(crops)
    for series, ink in enumerate(inks):
        for _ in range(2):
            dxs[series] = _refine(
                lambda value: path_score(ink, series_target(signals, fs, series, geometry, value, dys[series])),
                dxs[series], 0.6, 0.05)
            dys[series] = _refine(
                lambda value: path_score(ink, series_target(signals, fs, series, geometry, dxs[series], value)),
                dys[series], 0.6, 0.1)
    score = float(np.mean([
        path_score(ink, series_target(signals, fs, series, geometry, dxs[series], dys[series]))
        for series, ink in enumerate(inks)
    ]))
    contrast = score / max(float(np.median(dx_scores)), 1e-3)
    return Registration(tuple(dxs), tuple(dys), score, contrast)
