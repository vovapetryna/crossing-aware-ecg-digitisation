"""Page rectification with hengck23's public stage 0 (keypoints) and stage 1 (grid) models.

The result is a dense page->source map: for every pixel of the 2200x1700 page
it gives the location in the original full-resolution image, so crops are
resampled once from the original pixels.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional

from .layout import PAGE_HEIGHT, PAGE_WIDTH

HENGCK_DIR = Path(__file__).resolve().parents[1] / "third_party" / "hengck23"


@dataclass
class Rectification:
    rotation: int
    page_to_source: tuple[np.ndarray, np.ndarray]
    grid_found: float


def _import_hengck():
    if str(HENGCK_DIR) not in sys.path:
        sys.path.insert(0, str(HENGCK_DIR))
    with contextlib.redirect_stdout(io.StringIO()):
        import stage0_common
        import stage0_model
        import stage1_common
        import stage1_model
    return stage0_common, stage0_model, stage1_common, stage1_model


class Rectifier:
    def __init__(self, device: torch.device, stage0_tta: int = 1):
        self.s0c, s0m, self.s1c, s1m = _import_hengck()
        self.device = device
        self.stage0_tta = stage0_tta
        with contextlib.redirect_stdout(io.StringIO()):
            self.stage0 = self.s0c.load_net(s0m.Net(pretrained=False), HENGCK_DIR / "weight/stage0-last.checkpoint.pth").to(device)
            self.stage1 = self.s1c.load_net(s1m.Net(pretrained=False), HENGCK_DIR / "weight/stage1-last.checkpoint.pth").to(device)

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> Rectification:
        return self.finish(self.network(image_bgr))

    @torch.no_grad()
    def network(self, image_bgr: np.ndarray) -> dict:
        """GPU part: keypoints, homography and the stage 1 grid outputs."""
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch = self.s0c.image_to_batch(image)
        marker, orientation = 0, 0
        flips = batch["tta"][: self.stage0_tta]
        # One flip at a time: the 4-image batch peaks above 11 GB on MPS.
        for index, tta in enumerate(flips):
            output = self.stage0({"image": batch["image"][[index]]})
            mk, on = output["marker"].float().cpu(), output["orientation"][0].float().cpu()
            if tta == 1:
                mk, on = torch.flip(mk, [2]), on[[4, 5, 6, 7, 0, 1, 2, 3]]
            elif tta == 2:
                mk, on = torch.flip(mk, [3]), on[[6, 7, 4, 5, 2, 3, 0, 1]]
            elif tta == 3:
                mk, on = torch.flip(mk, [2, 3]), on[[2, 3, 0, 1, 6, 7, 4, 5]]
            orientation = orientation + on
            marker = marker + mk[..., : batch["sH"], : batch["sW"]]
        keypoint, rotation = self.s0c.marker_to_keypoint(image, orientation, marker / len(flips), batch["scale"])
        rotation = int(rotation)
        rotated = np.ascontiguousarray(np.rot90(image, rotation, axes=(0, 1)))
        normalised, homography, _ = self.s0c.normalise_image(rotated, [[k[0], k[1]] for k in keypoint])
        tensor = torch.from_numpy(np.ascontiguousarray(normalised.transpose(2, 0, 1)))[None]
        output = {key: value.float().cpu() for key, value in self.stage1({"image": tensor}).items()}
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return {"output": output, "normalised": normalised, "homography": homography,
                "rotation": rotation, "shape": image.shape[:2], "tensor": tensor}

    def finish(self, state: dict) -> Rectification:
        """CPU part: grid post-processing and the dense page -> source map."""
        normalised = state["normalised"]
        with contextlib.redirect_stdout(io.StringIO()):
            gridpoint_xy, _ = self.s1c.output_to_predict(normalised, {"image": state["tensor"]}, state["output"])
        grid_found = float(np.mean(np.any(gridpoint_xy != 0, axis=-1)))

        # Dense page -> normalised coordinates, as in hengck23's rectify_image.
        sparse = torch.from_numpy(np.ascontiguousarray(gridpoint_xy.transpose(2, 0, 1)))[None].float()
        dense = functional.interpolate(sparse, size=(PAGE_HEIGHT, PAGE_WIDTH), mode="bilinear", align_corners=True)[0].numpy()
        height, width = normalised.shape[:2]
        norm_x = dense[0] * width / (width - 1) - 0.5
        norm_y = dense[1] * height / (height - 1) - 0.5

        # Normalised -> rotated full-resolution image through the inverse homography.
        inverse = np.linalg.inv(state["homography"])
        points = np.stack([norm_x, norm_y, np.ones_like(norm_x)], axis=0).reshape(3, -1)
        mapped = inverse @ points
        rot_x = (mapped[0] / mapped[2]).reshape(PAGE_HEIGHT, PAGE_WIDTH)
        rot_y = (mapped[1] / mapped[2]).reshape(PAGE_HEIGHT, PAGE_WIDTH)
        source_x, source_y = unrotate(rot_x, rot_y, state["rotation"], state["shape"])
        return Rectification(state["rotation"], (source_x.astype(np.float32), source_y.astype(np.float32)), grid_found)


def unrotate(x: np.ndarray, y: np.ndarray, k: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates in np.rot90(image, k) back to the original image."""
    height, width = shape
    k %= 4
    if k == 0:
        return x, y
    if k == 1:  # rotated[r, c] = image[c, W-1-r]
        return width - 1 - y, x
    if k == 2:
        return width - 1 - x, height - 1 - y
    return y, height - 1 - x  # rotated[r, c] = image[H-1-c, r]
