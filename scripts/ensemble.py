"""Score the signal average of two dumped models (Table 1)."""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from crossecg.metric import LEADS, image_snr, score_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    arguments = parser.parse_args()
    a, b = ({p["record_id"] + "-" + p["kind"]: p for p in pickle.load(open(path, "rb"))}
            for path in (arguments.first, arguments.second))
    keys = sorted(set(a) & set(b))
    for name, pages in (("first", a), ("second", b)):
        print(name, round(score_db([image_snr(pages[k]["truth"], pages[k]["post"], pages[k]["fs"]) for k in keys]), 3))
    average = [image_snr(a[k]["truth"], {lead: (np.asarray(a[k]["post"][lead]) + np.asarray(b[k]["post"][lead])) / 2
                                         for lead in LEADS}, a[k]["fs"]) for k in keys]
    print("average", round(score_db(average), 3))


if __name__ == "__main__":
    main()
