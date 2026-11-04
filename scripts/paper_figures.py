"""Paper figures from saved development-set predictions (no GPU). Numbers are cached in figures/data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib
import matplotlib.font_manager
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from crossecg.layout import SERIES_LEADS  # noqa: E402
from crossecg.metric import LEADS, MAX_TIME_SHIFT, align_signals, image_snr, score_db  # noqa: E402

RUNS = Path("outputs/predictions")
OUT = Path("outputs/figures")
KINDS = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")
KIND_NAMES = {"0001": "clean", "0003": "scan", "0004": "scan bw", "0005": "photo", "0006": "screen",
              "0009": "photo", "0010": "photo", "0011": "scan", "0012": "scan bw"}
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"

# Libertinus Sans matches the CEURART template; set LIBERTINUS_DIR to its .otf folder, else matplotlib's default font is used.
for _font in Path(os.environ.get("LIBERTINUS_DIR", ".")).expanduser().glob("LibertinusSans-*.otf"):
    matplotlib.font_manager.fontManager.addfont(str(_font))
plt.rcParams["font.family"] = ["Libertinus Sans", "DejaVu Sans"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "lines.linewidth": 2, "pdf.fonttype": 42,
})


def cached(name: str, compute):
    path = OUT / "data" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    path.write_text(json.dumps(value, indent=1))
    return value


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def lead_errors(page):
    """Aligned per-lead error (as the metric aligns it) and QRS centres from lead II truth."""
    fs, n = page["fs"], len(page["truth"]["II"])
    gradient = np.abs(np.gradient(np.asarray(page["truth"]["II"], float)))
    centres, _ = find_peaks(gradient, distance=int(0.3 * fs), height=np.percentile(gradient, 99) * 0.3)
    where = {lead: (column, len(leads)) for leads in SERIES_LEADS for column, lead in enumerate(leads)}
    out, signal = {}, 0.0
    for lead in LEADS:
        truth = np.asarray(page["truth"][lead], float)
        aligned = align_signals(truth, np.asarray(page["post"][lead], float), int(fs * MAX_TIME_SHIFT))
        aligned[~np.isfinite(aligned)] = 0
        column, count = where[lead]
        offset = 0 if count == 1 else int(np.floor(column * n / 4))
        out[lead] = (truth - aligned, centres - offset)
        signal += float(np.sum(truth**2))
    return out, signal


def qrs_mask(length, centres, half):
    mask = np.zeros(length, bool)
    for k in centres:
        lo, hi = k - half, k + half + 1
        if hi > 0 and lo < length:
            mask[max(lo, 0):min(hi, length)] = True
    return mask


def error_oracle_data():
    pages = pickle.load(open(RUNS / "base_long_val_pred.pkl", "rb"))
    widths, cuts = (0.04, 0.06, 0.1), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    snr = {f"{w}:{c}": [] for w in widths for c in cuts}
    share = {str(w): [] for w in widths}
    for page in pages:
        errors, signal = lead_errors(page)
        for w in widths:
            half = int(w * page["fs"])
            inside = outside = 0.0
            for e, centres in errors.values():
                m = qrs_mask(len(e), centres, half)
                inside += float(np.sum(e[m] ** 2))
                outside += float(np.sum(e[~m] ** 2))
            share[str(w)].append(inside / max(inside + outside, 1e-12))
            for c in cuts:
                snr[f"{w}:{c}"].append(min(signal / max(outside + (1 - c) ** 2 * inside, 1e-12), 1e6))
    return {"share": {w: float(np.mean(v)) for w, v in share.items()},
            "oracle": {k: score_db(v) for k, v in snr.items()}, "cuts": list(cuts), "widths": list(widths)}


def fig_error_oracle():
    d = cached("error_oracle", error_oracle_data)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for w, color in zip(d["widths"], (BLUE, ORANGE, AQUA)):
        y = [d["oracle"][f"{w}:{c}"] for c in d["cuts"]]
        ax.plot([100 * c for c in d["cuts"]], y, color=color, marker="o", markersize=3.5)
        ax.annotate(f"±{int(w * 1000)} ms ({100 * d['share'][str(w)]:.0f}% of error)", (100, y[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_xlabel("QRS-window error amplitude removed (%)")
    ax.set_ylabel("Development-set SNR (dB)")
    ax.set_xlim(0, 100)
    save(fig, "fig_error_oracle")


def value_data():
    pages = pickle.load(open(RUNS / "base_long_val_pred.pkl", "rb"))
    windows, signals, noises = [], [], []
    for index, page in enumerate(pages):
        errors, signal = lead_errors(page)
        half = int(0.06 * page["fs"])
        noise = sum(float(np.sum(e**2)) for e, _ in errors.values())
        signals.append(signal)
        noises.append(noise)
        for lead, (e, centres) in errors.items():
            for k in centres:
                lo, hi = max(k - half, 0), min(k + half + 1, len(e))
                if hi - lo > 5:
                    windows.append((index, lead, page["kind"], float(np.sum(e[lo:hi] ** 2))))
    signals, noises = np.array(signals), np.array(noises)
    snr = np.minimum(signals / noises, 1e6)
    value = np.array([snr[i] * n / noises[i] for i, _, _, n in windows])
    order = np.argsort(-value)
    base = score_db(snr)

    def gain(count):
        removed = np.zeros(len(pages))
        for i in order[:count]:
            removed[windows[i][0]] += windows[i][3]
        return score_db(np.minimum(signals / np.maximum(noises - removed, 1e-12), 1e6)) - base

    fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    top = order[: len(windows) // 20]
    by_lead = {lead: float(sum(value[i] for i in top if windows[i][1] == lead)) for lead in LEADS}
    by_kind = {kind: float(sum(value[i] for i in top if windows[i][2] == kind)) for kind in KINDS}
    total = sum(by_lead.values())
    page_db = 10 * np.log10(snr)
    low = score_db(np.where(page_db < 17, 10 ** 2.26, snr)) - base
    return {"fractions": fractions, "gain": [gain(int(len(windows) * f)) for f in fractions],
            "by_lead": {k: v / total for k, v in by_lead.items()}, "by_kind": {k: v / total for k, v in by_kind.items()},
            "fix_low_pages": low, "windows": len(windows)}


def fig_value_concentration():
    d = cached("value_concentration", value_data)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.35})
    picks = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    gains = [d["gain"][d["fractions"].index(f)] for f in picks]
    labels = [f"{int(f * 100)}%" for f in picks]
    left.bar(range(len(picks)), gains, color=BLUE, width=0.7)
    left.bar(len(picks) + 0.4, d["fix_low_pages"], color=GRAY, width=0.7)
    for x, g in zip(list(range(len(picks))) + [len(picks) + 0.4], gains + [d["fix_low_pages"]]):
        left.annotate(f"+{g:.1f}" if g >= 0.95 else f"+{g:.2f}", (x, g), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
    left.set_xticks(list(range(len(picks))) + [len(picks) + 0.4], labels + ["pages\n<17 dB"], fontsize=7)
    left.set_ylabel("SNR gain if made perfect (dB)")
    left.set_xlabel("Top-ranked QRS windows")
    left.grid(axis="x", visible=False)
    leads = sorted(d["by_lead"], key=d["by_lead"].get, reverse=True)
    right.bar(range(len(leads)), [100 * d["by_lead"][k] for k in leads], color=BLUE, width=0.7)
    right.set_xticks(range(len(leads)), leads, rotation=45, fontsize=7)
    right.set_ylabel("Share of top-5% value (%)")
    right.set_xlabel("Lead")
    right.grid(axis="x", visible=False)
    save(fig, "fig_value_concentration")


def crossing_data():
    from score_ablation import page_snrs, record_bootstrap, records_improved

    out = {}
    for seed, (a, c) in {"seed 1": ("ablA_seed1", "ablC_seed1"), "seed 2": ("ablA_seed2", "ablC_seed2")}.items():
        sa, sc = page_snrs(RUNS / f"{a}_val_pred.pkl"), page_snrs(RUNS / f"{c}_val_pred.pkl")
        keys = sorted(set(sa) & set(sc))
        kinds = np.array([sa[k][0] for k in keys])
        records = np.array([k.rsplit("-", 1)[0] for k in keys])
        va, vc = np.array([sa[k][1] for k in keys]), np.array([sc[k][1] for k in keys])
        rng = np.random.default_rng(0)
        boot = record_bootstrap(records, va, vc, 5000, rng)
        per_kind = {}
        for kind in KINDS:
            m = kinds == kind
            kind_boot = record_bootstrap(records[m], va[m], vc[m], 5000, rng)
            per_kind[kind] = [score_db(vc[m]) - score_db(va[m]), *np.percentile(kind_boot, [2.5, 97.5]).tolist()]
        better, total = records_improved(records, va, vc)
        out[seed] = {"A": score_db(va), "C": score_db(vc), "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                     "records_improved": better, "records": total, "pages": len(keys), "per_kind": per_kind}
    return out


def fig_crossing_per_type():
    d = cached("crossing", crossing_data)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(KINDS))
    for offset, (seed, color) in zip((-0.19, 0.19), (("seed 1", BLUE), ("seed 2", ORANGE))):
        values = np.array([d[seed]["per_kind"][k] for k in KINDS])
        ax.bar(x + offset, values[:, 0], width=0.36, color=color, label=f"{seed} ({d[seed]['C'] - d[seed]['A']:+.2f} dB overall)")
        ax.errorbar(x + offset, values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]],
                    fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, [f"{k}\n{KIND_NAMES[k]}" for k in KINDS], fontsize=8)
    ax.set_ylabel("SNR gain, C − A (dB)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, "fig_crossing_per_type")


def fig_crossing_curves():
    # seed-2 checkpoints, 180-page subset, crossing fusion off (scored during the run)
    minutes = [15, 31, 47, 63]
    a, c = [19.88, 22.40, 23.26, 23.68], [20.64, 22.70, 23.56, 23.96]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(minutes, a, color=BLUE, marker="o", markersize=3.5)
    ax.plot(minutes, c, color=ORANGE, marker="o", markersize=3.5)
    ax.annotate("A: no head", (minutes[-1], a[-1]), xytext=(5, -4), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.annotate("C: crossing head", (minutes[-1], c[-1]), xytext=(5, 4), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlim(10, 82)
    ax.set_xticks([15, 30, 45, 60])
    ax.set_xlabel("Training time (min)")
    ax.set_ylabel("Subset SNR (dB)")
    save(fig, "fig_crossing_curves")


def example_data():
    a = {p["record_id"] + "-" + p["kind"]: p for p in pickle.load(open(RUNS / "ablA_seed1_val_pred.pkl", "rb"))}
    c = {p["record_id"] + "-" + p["kind"]: p for p in pickle.load(open(RUNS / "ablC_seed1_val_pred.pkl", "rb"))}
    candidates = []
    for key, pa in a.items():
        pc = c[key]
        leads_c = pc["post"]
        for lead in ("V2", "V3", "V4", "V5"):
            truth = np.asarray(pa["truth"][lead], float)
            fs = pa["fs"]
            al_a = align_signals(truth, np.asarray(pa["post"][lead], float), int(fs * MAX_TIME_SHIFT))
            al_c = align_signals(truth, np.asarray(leads_c[lead], float), int(fs * MAX_TIME_SHIFT))
            if not (np.all(np.isfinite(al_a)) and np.all(np.isfinite(al_c))):
                continue
            peak = int(np.argmax(np.abs(np.gradient(truth))))
            lo, hi = max(peak - int(0.12 * fs), 0), min(peak + int(0.12 * fs), len(truth))
            ea, ec = np.sum((truth - al_a)[lo:hi] ** 2), np.sum((truth - al_c)[lo:hi] ** 2)
            if pa["kind"] in ("0001", "0004", "0006") and np.ptp(truth[lo:hi]) > 1.0:
                candidates.append((ea - ec, ea / max(ec, 1e-9), key, lead, lo, hi, truth[lo:hi].tolist(),
                                   al_a[lo:hi].tolist(), al_c[lo:hi].tolist(), fs, pa["kind"]))
    picks, seen = [], set()
    for cand in sorted(candidates, key=lambda t: -t[0]):
        if cand[10] not in seen and cand[1] > 1.5:
            picks.append(cand)
            seen.add(cand[10])
        if len(picks) == 3:
            break
    return [{"key": p[2], "lead": p[3], "kind": p[10], "fs": p[9], "truth": p[6], "a": p[7], "c": p[8]} for p in picks]


def fig_examples():
    d = cached("examples", example_data)
    fig, axes = plt.subplots(1, len(d), figsize=(7.2, 2.4))
    for ax, ex in zip(np.atleast_1d(axes), d):
        t = np.arange(len(ex["truth"])) / ex["fs"] * 1000
        ax.plot(t, ex["truth"], color=INK, linewidth=2.5, label="truth")
        ax.plot(t, ex["a"], color=BLUE, linewidth=1.3, label="A: no head")
        ax.plot(t, ex["c"], color=ORANGE, linewidth=1.3, label="C: crossing head")
        ax.set_title(f"{KIND_NAMES[ex['kind']]} ({ex['kind']}), lead {ex['lead']}", fontsize=8, color=INK)
        ax.set_xlabel("Time (ms)")
    np.atleast_1d(axes)[0].set_ylabel("mV")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    save(fig, "fig_examples")


def fig_pipeline():
    steps = ["Page image:\nscan, photo\nor render", "Rectification\n(keypoints,\ngrid map)", "4 row\nstrips\n(±3 mV)",
             "U-Net reader:\nheight per\ncolumn;\ncrossing head\nin training", "Least-squares\ncolumns to\nsamples",
             "Lead fusion,\nEinthoven\nprojection", "12 lead\nsignals"]
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_axis_off()
    width, gap = 0.128, 0.016
    for i, text in enumerate(steps):
        x = i * (width + gap)
        highlight = "crossing" in text
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.1), width, 0.8, boxstyle="round,pad=0.005,rounding_size=0.02",
                                                      facecolor="#fdeee7" if highlight else "#f3f2ef",
                                                      edgecolor=ORANGE if highlight else GRAY, linewidth=1))
        ax.text(x + width / 2, 0.5, text, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + width + gap, 0.5), xytext=(x + width, 0.5),
                        arrowprops={"arrowstyle": "-|>", "color": INK2, "linewidth": 0.8})
    ax.set_xlim(-0.01, len(steps) * (width + gap))
    ax.set_ylim(0, 1)
    save(fig, "fig_pipeline")


FIGURES = {f.__name__: f for f in (fig_error_oracle, fig_value_concentration, fig_crossing_per_type,
                                   fig_crossing_curves, fig_examples, fig_pipeline)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(FIGURES))
    arguments = parser.parse_args()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for name in arguments.names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
