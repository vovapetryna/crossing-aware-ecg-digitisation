"""Paired comparison of ablation A (no crossing head) and C (crossing head) with a record-clustered bootstrap."""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from crossecg.metric import image_snr, score_db


def page_snrs(path) -> dict[str, tuple[str, float]]:
    return {p["record_id"] + "-" + p["kind"]: (p["kind"], image_snr(p["truth"], p["post"], p["fs"]))
            for p in pickle.load(open(path, "rb"))}


def record_bootstrap(records: np.ndarray, sa: np.ndarray, sc: np.ndarray, draws: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Paired C-A score differences (dB) over draws that resample whole records, same draws for A and C."""
    ids, index = np.unique(records, return_inverse=True)
    pages = np.bincount(index).astype(float)
    sum_a, sum_c = np.bincount(index, sa), np.bincount(index, sc)
    counts = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(draws)])
    total = counts @ pages
    return 10 * np.log10(counts @ sum_c / total) - 10 * np.log10(counts @ sum_a / total)


def records_improved(records: np.ndarray, sa: np.ndarray, sc: np.ndarray) -> tuple[int, int]:
    _, index = np.unique(records, return_inverse=True)
    return int(np.sum(np.bincount(index, sc) > np.bincount(index, sa))), int(index.max() + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", required=True, help="A.pkl:C.pkl per seed")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--exclude", nargs="*", default=[], help="record ids to leave out")
    arguments = parser.parse_args()
    rng = np.random.default_rng(0)
    for pair in arguments.pairs:
        a, c = (page_snrs(path) for path in pair.split(":"))
        keys = [k for k in sorted(set(a) & set(c)) if k.rsplit("-", 1)[0] not in arguments.exclude]
        sa, sc = np.array([a[k][1] for k in keys]), np.array([c[k][1] for k in keys])
        records = np.array([k.rsplit("-", 1)[0] for k in keys])
        boot = record_bootstrap(records, sa, sc, arguments.bootstrap, rng)
        better, total = records_improved(records, sa, sc)
        print(f"{pair}: pages {len(keys)}  A {score_db(sa):.2f}  C {score_db(sc):.2f}  diff {score_db(sc) - score_db(sa):+.3f}  "
              f"95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  records improved {better}/{total}")


if __name__ == "__main__":
    main()
