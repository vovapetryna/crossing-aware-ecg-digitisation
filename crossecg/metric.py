"""Exact port of the Kaggle PhysioNet ECG signal-extraction metric on arrays.

Reference: kaggle.com/code/metric/physionet-ecg-signal-extraction-metric.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.signal

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 1e6


def align_signals(label: np.ndarray, prediction: np.ndarray, max_shift: float = float("inf")) -> np.ndarray:
    label = np.asarray(label, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.all(np.isfinite(label)):
        raise ValueError("label must be finite")
    correlation = scipy.signal.correlate(label - label.mean(), prediction - prediction.mean(), mode="full")
    lags = scipy.signal.correlation_lags(label.size, prediction.size, mode="full")
    valid = (lags >= -max_shift) & (lags <= max_shift)
    best_value = np.nanmax(correlation[valid])
    candidates = np.flatnonzero(correlation == best_value)
    shift = int(lags[min(candidates, key=lambda index: abs(lags[index]))])
    aligned = np.concatenate((
        np.full(max(shift, 0), np.nan),
        prediction[max(-shift, 0):min(label.size - shift, prediction.size)],
        np.full(max(label.size - prediction.size - shift, 0), np.nan),
    ))
    if np.any(np.isfinite(label) & np.isfinite(aligned)):
        offset = scipy.optimize.minimize_scalar(
            lambda value: np.nansum((label - (aligned - value)) ** 2), method="Brent"
        ).x
        aligned = aligned - offset
    return aligned


def lead_powers(label: np.ndarray, prediction: np.ndarray, fs: float) -> tuple[float, float]:
    aligned = align_signals(label, prediction, int(fs * MAX_TIME_SHIFT))
    aligned[~np.isfinite(aligned)] = 0.0
    return float(np.sum(label**2)), float(np.sum((label - aligned) ** 2))


def image_snr(truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray], fs: float) -> float:
    """Linear SNR of one image, with powers summed over the 12 leads."""
    signal = noise = 0.0
    for lead in LEADS:
        predicted = np.asarray(prediction[lead], dtype=np.float64)
        if predicted.shape != truth[lead].shape or not np.all(np.isfinite(predicted)):
            raise ValueError(f"lead {lead}: prediction must be finite with shape {truth[lead].shape}")
        lead_signal, lead_noise = lead_powers(truth[lead], predicted.copy(), fs)
        signal += lead_signal
        noise += lead_noise
    if noise == 0:
        return PERFECT_SCORE
    if signal == 0:
        return 0.0
    return min(signal / noise, PERFECT_SCORE)


def score_db(image_snrs: list[float] | np.ndarray) -> float:
    """Competition score: dB of the mean linear per-image SNR."""
    mean = float(np.mean(image_snrs))
    return max(10.0 * np.log10(mean), -PERFECT_SCORE) if mean > 0 else -PERFECT_SCORE
