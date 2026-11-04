"""Rectify competition images and cache 4 series crops plus registered labels per image."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .labels import register_from_prior, series_target
from .layout import KIND_GROUP, SERIES_ZERO_OFFSET_Y, CropGeometry, time_offsets_for
from .zip_io import IMAGE_TYPES, CompetitionSource

FIELDS = ("record_id", "kind", "split", "fs", "sig_len", "status", "rotation", "grid_found",
          "dx0", "dx1", "dx2", "dx3", "dy0", "dy1", "dy2", "dy3", "score", "contrast", "seconds", "error", "ptbxl_labels")

_local = threading.local()


def split_records(record_ids: list[str], validation_fraction: float, seed: int) -> dict[str, str]:
    order = np.random.default_rng(seed).permutation(sorted(record_ids))
    count = int(round(len(order) * validation_fraction))
    return {record_id: ("validation" if position < count else "train") for position, record_id in enumerate(order)}


def _load(source_path: str, job: tuple) -> tuple:
    if not hasattr(_local, "source"):
        _local.source = CompetitionSource(source_path)
    record_id, kind = job[0], job[1]
    try:
        return job, _local.source.image(record_id, kind), _local.source.signals(record_id), None
    except Exception as error:  # noqa: BLE001
        return job, None, None, error


def _finish(rectifier, state: dict, image: np.ndarray, signals: dict, job: tuple, output: Path, quality: int, started: float,
            geometry: CropGeometry, label_geometry: CropGeometry, labels=None) -> dict:
    record_id, kind, split, fs, sig_len = job
    row = {"record_id": record_id, "kind": kind, "split": split, "fs": fs, "sig_len": sig_len}
    try:
        rectification = rectifier.finish(state)
        crops = [
            cv2.remap(image, *geometry.maps(series, rectification.page_to_source), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            for series in range(4)
        ]
        original = labels.signals(record_id, signals) if labels is not None else None
        label_signals, label_fs = (original, 500.0) if original is not None else (signals, fs)
        registration = register_from_prior(
            crops, label_signals, label_fs, geometry,
            time_offsets_for(KIND_GROUP.get(kind)), SERIES_ZERO_OFFSET_Y,
        )
        targets = [series_target(label_signals, label_fs, series, label_geometry, registration.dx[series], registration.dy[series])
                   for series in range(4)]
        row["ptbxl_labels"] = original is not None
        target_dir = output / "images" / f"{record_id}-{kind}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for series, crop in enumerate(crops):
            cv2.imwrite(str(target_dir / f"s{series}.jpg"), crop, [
                cv2.IMWRITE_JPEG_QUALITY, quality,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ])
        np.savez_compressed(
            target_dir / "labels.npz",
            y=np.stack([t.y for t in targets]),
            ordinate_mask=np.stack([t.ordinate_mask for t in targets]),
            exists=np.stack([t.exists for t in targets]),
        )
        row.update(
            status="ok", rotation=rectification.rotation, grid_found=round(rectification.grid_found, 4),
            score=round(registration.score, 4), contrast=round(registration.contrast, 4),
            **{f"dy{series}": value for series, value in enumerate(registration.dy)},
            **{f"dx{series}": value for series, value in enumerate(registration.dx)},
        )
    except Exception as error:  # noqa: BLE001 - one bad page must not stop the cache build
        row.update(status="error", error=repr(error)[:200])
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/physionet-ecg-image-digitization.zip")
    parser.add_argument("--output-dir", default="outputs/physionet_cache")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-kinds-per-record", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--splits", default="validation,train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage0-tta", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--vertical-scale", type=float, default=0.75)
    parser.add_argument("--image-scale", type=float, default=2.0, help="stored image columns per page px")
    parser.add_argument("--label-scale", type=float, default=2.0, help="label columns per page px (image is upsampled on load)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ptbxl", default=None, help="PTB-XL zip; with --ptbxl-matches, label matched records at 500 Hz")
    parser.add_argument("--ptbxl-matches", default=None)
    arguments = parser.parse_args()

    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = CropGeometry(horizontal_scale=arguments.image_scale, vertical_scale=arguments.vertical_scale, height=arguments.height)
    label_geometry = CropGeometry(horizontal_scale=arguments.label_scale, vertical_scale=arguments.vertical_scale, height=arguments.height)
    (output / "geometry.json").write_text(json.dumps(asdict(label_geometry)))
    (output / "cache.json").write_text(json.dumps({"upsample": arguments.label_scale / arguments.image_scale}))
    labels = None
    if arguments.ptbxl and arguments.ptbxl_matches:
        from .ptbxl import PtbxlLabels

        labels = PtbxlLabels(arguments.ptbxl, arguments.ptbxl_matches)
    source = CompetitionSource(arguments.source)
    records = source.records()
    if arguments.max_records:
        records = records[: arguments.max_records]
    splits = split_records([r.record_id for r in records], arguments.validation_fraction, arguments.seed)
    manifest = output / ("manifest.csv" if arguments.num_shards == 1 else f"manifest_shard{arguments.shard}.csv")
    done = set()
    for existing in output.glob("manifest*.csv"):
        with existing.open() as handle:
            done |= {(row["record_id"], row["kind"]) for row in csv.DictReader(handle)}
    rng = np.random.default_rng(arguments.seed)
    jobs = []
    wanted_splits = arguments.splits.split(",")
    for wanted in wanted_splits:
        for record in records:
            if splits[record.record_id] != wanted:
                continue
            kinds = [kind for kind in IMAGE_TYPES if source.exists(f"train/{record.record_id}/{record.record_id}-{kind}.png")]
            if wanted == "train" and arguments.train_kinds_per_record < len(kinds):
                kinds = sorted(rng.choice(kinds, arguments.train_kinds_per_record, replace=False).tolist())
            jobs += [(record.record_id, kind, wanted, record.fs, record.sig_len) for kind in kinds if (record.record_id, kind) not in done]
    jobs = jobs[arguments.shard::arguments.num_shards]
    print(f"{len(jobs)} images to process ({len(done)} already cached)", flush=True)
    new_file = not manifest.exists()
    import torch

    from .rectify import Rectifier

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    rectifier = Rectifier(torch.device(arguments.device), stage0_tta=arguments.stage0_tta)
    started = time.perf_counter()
    with manifest.open("a", newline="") as handle, \
            ThreadPoolExecutor(2) as loaders, ThreadPoolExecutor(arguments.workers) as finishers:
        writer = csv.DictWriter(handle, FIELDS)
        if new_file:
            writer.writeheader()
        loads = (loaders.submit(_load, str(source.path), job) for job in jobs)
        prefetched = [next(loads, None) for _ in range(4)]
        pending, count = set(), 0

        def drain(block: bool) -> None:
            nonlocal pending, count
            if block:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            else:
                finished = {future for future in pending if future.done()}
                pending -= finished
            for future in finished:
                row = future.result()
                writer.writerow(row)
                handle.flush()
                count += 1
                if count % 20 == 0 or row["status"] != "ok":
                    rate = count / (time.perf_counter() - started)
                    print(f"{count}/{len(jobs)} {rate:.2f} img/s eta {(len(jobs) - count) / rate / 60:.0f} min "
                          f"last={row['record_id']}-{row['kind']} {row['status']} {row.get('error') or ''}", flush=True)

        while prefetched and prefetched[0] is not None:
            job, image, signals, error = prefetched.pop(0).result()
            prefetched.append(next(loads, None))
            job_started = time.perf_counter()
            if error is None:
                try:
                    state = rectifier.network(image)
                except Exception as network_error:  # noqa: BLE001
                    error = network_error
            if error is not None:
                writer.writerow({"record_id": job[0], "kind": job[1], "split": job[2], "status": "error", "error": repr(error)[:200]})
                continue
            pending.add(finishers.submit(_finish, rectifier, state, image, signals, job, output, arguments.quality, job_started,
                                         geometry, label_geometry, labels))
            while len(pending) >= 2 * arguments.workers:
                drain(True)
            drain(False)
        while pending:
            drain(True)


if __name__ == "__main__":
    main()
