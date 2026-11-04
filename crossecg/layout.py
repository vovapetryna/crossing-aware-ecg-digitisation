"""Page geometry of the 3x4 + rhythm layout and series-crop coordinates.

Page coordinates are those of the rectifier's 2200x1700 (200 DPI) output page.
The time axis was fitted on rectified rhythm strips of 45 validation pages:
10 s span x=117.76..2080.06 (std 0.3 px). Raw -0001 images use a slightly
wider axis (196.85 px/s), so always rectify first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAGE_WIDTH, PAGE_HEIGHT = 2200, 1700
TIME_ORIGIN_X = 117.76
PIXELS_PER_SECOND = 196.23
PIXELS_PER_MV = 78.74
ZERO_MV_Y = (707.5, 991.25, 1274.5, 1534.25)
SERIES_LEADS = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
    ("II",),
)
# Mean registered ink offset per series row over 3519 cached pages (rectifier residual skew).
SERIES_TIME_OFFSET_X = (-0.239, -0.175, -0.129, -0.050)
# Per acquisition group (train split means): print/scan/photo shifts the trace relative to the grid.
GROUP_TIME_OFFSET_X = {
    "clean": (0.115, 0.124, 0.126, 0.098),
    "scan": (-0.177, -0.081, -0.042, 0.056),
    "photo": (-0.357, -0.288, -0.212, -0.132),
    "screen": (-0.487, -0.518, -0.501, -0.405),
}
KIND_GROUP = {"0001": "clean", "0003": "scan", "0004": "scan", "0011": "scan", "0012": "scan",
              "0005": "photo", "0009": "photo", "0010": "photo", "0006": "screen"}
# Mean registered vertical ink offset per series (page px), over 3519 cached pages.
SERIES_ZERO_OFFSET_Y = (-4.446, -3.855, -3.232, -2.658)
SEGMENT_SECONDS = 2.5
EDGE_MARGIN_SECONDS = 0.0
TOTAL_SECONDS = 10.0


def image_group(width: int, height: int) -> str | None:
    """Acquisition group from the original image size; sizes separate the groups exactly in train."""
    long_side, short_side = max(width, height), min(width, height)
    if (long_side, short_side) == (2200, 1700):
        return "clean"
    if short_side == 1652:
        return "scan"
    if (long_side, short_side) == (4000, 3000):
        return "screen"
    if (long_side, short_side) == (4032, 3024):
        return "photo"
    return None


def time_offsets_for(group: str | None) -> tuple[float, ...]:
    return GROUP_TIME_OFFSET_X.get(group, SERIES_TIME_OFFSET_X)


@dataclass(frozen=True)
class CropGeometry:
    """Each series crop resamples a page band at fixed horizontal/vertical scales."""

    x_start: float = 70.0
    x_end: float = 2120.0
    horizontal_scale: float = 2.0
    vertical_scale: float = 0.75
    height: int = 352

    @property
    def width(self) -> int:
        return int(round((self.x_end - self.x_start) * self.horizontal_scale))

    def page_x(self, column: np.ndarray) -> np.ndarray:
        return self.x_start + np.asarray(column, dtype=np.float64) / self.horizontal_scale

    def column(self, page_x: np.ndarray) -> np.ndarray:
        return (np.asarray(page_x, dtype=np.float64) - self.x_start) * self.horizontal_scale

    def page_y(self, series: int, row: np.ndarray) -> np.ndarray:
        top = ZERO_MV_Y[series] - 0.5 * self.height / self.vertical_scale
        return top + np.asarray(row, dtype=np.float64) / self.vertical_scale

    def column_times(self) -> np.ndarray:
        return (self.page_x(np.arange(self.width)) - TIME_ORIGIN_X) / PIXELS_PER_SECOND

    def maps(self, series: int, page_to_source: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """cv2.remap maps from crop pixels to source-image pixels.

        ``page_to_source`` optionally gives dense (map_x, map_y) arrays over the
        page that locate each page pixel in the original high-resolution image.
        """
        xs = self.page_x(np.arange(self.width)).astype(np.float32)
        ys = self.page_y(series, np.arange(self.height)).astype(np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        if page_to_source is None:
            return grid_x, grid_y
        import cv2

        source_x, source_y = page_to_source
        return (
            cv2.remap(source_x, grid_x, grid_y, cv2.INTER_LINEAR),
            cv2.remap(source_y, grid_x, grid_y, cv2.INTER_LINEAR),
        )


def lead_windows(series: int) -> list[tuple[str, float, float]]:
    """(lead, start second, end second) for each lead drawn on a series row."""
    leads = SERIES_LEADS[series]
    if len(leads) == 1:
        return [(leads[0], 0.0, TOTAL_SECONDS)]
    return [(lead, index * SEGMENT_SECONDS, (index + 1) * SEGMENT_SECONDS) for index, lead in enumerate(leads)]


def truth_segments(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scored ground truth per lead: lead II is 10 s, other leads their 2.5 s window."""
    return {
        lead: np.nan_to_num(np.asarray(column, dtype=np.float64)) if lead == "II"
        else np.asarray(column, dtype=np.float64)[np.isfinite(column)]
        for lead, column in signals.items()
    }
