"""A second composite exhibit with CRPS: python make_evidence_crps_figure.py [--combine mean|mixture] [--tag name]

Writes figures/fig_learning_and_evidence_crps[_<tag>]{,_dark}.{pdf,png} and the tables behind them,
figures/evidence_crps.csv (the ratio rows) and figures/evidence_crps_test.csv (the CRPS block). The architecture panel
and the palette are imported from make_submission_figure.py, which is left as it is (its figure remains the current
exhibit; this one is an alternative).

Bottom-left: one probability-ratio row per evaluated season, the full model against the strict control on the same
players: exp(mean paired log-score difference) with the paired 95% interval. Every season is out of sample: the models
behind a row are trained on earlier seasons only (rolling held-out seasons 2021-22 and 2022-23; the registered one-shot
test 2025-26). Bottom-right: CRPS skill by statistic on the test season, skill = (CRPS without - CRPS with) /
CRPS without, with the paired 95% interval of the numerator over the same denominator.

Seasons with two registered fits are shown as one row. --combine mean (default) averages the two fits' paired
differences player by player (the expected single-fit comparison; the registered per-fit numbers in the text are its
parts); --combine mixture scores the equal mixture of the two fits against the mixture of the two controls (one
ensemble forecast per player, from the pooled per-unit files). A season with one fit uses that fit. Single-fit rows
keep the registered log-score values of figures/evidence_source.csv (the scored values from the per-unit files are
printed beside them as a check). The per-unit files are written by scripts/crps_lane.sh and scripts/crps_lane2.sh
(docs/CRPS_*.per_unit.csv). Nothing is retrained here.
"""
from pathlib import Path
import argparse
import csv
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox

from make_submission_figure import THEMES, OUT, draw_idea

ROOT = Path(__file__).resolve().parent
LEFT, RIGHT = 0.04, 0.975        # the figure's text margins (figure fractions)
LABELS = {"games": "games played", "minutes": "season minutes", "points": "season points", "rebounds": "season rebounds",
          "assists": "season assists", "mpg": "minutes per game", "ppg": "points per game", "per": "PER"}
# (group, season, per-fit per-unit file stems, pooled-mixture per-unit file stem). Groups are drawn in this order.
SEASONS = [("Held-Out Seasons", "2021–22", ["CRPS_2022_s1_history"], None),
           ("Held-Out Seasons", "2022–23", ["CRPS_2023_s1_history", "CRPS_2023_s2_history"], "CRPS_2023_pooled_history"),
           ("Out-of-Time Test Season", "2025–26", ["CRPS_TEST_2026_s1_primary", "CRPS_TEST_2026_s2_primary"], "CRPS_TEST_2026_pooled_primary")]
TEST = SEASONS[-1]


def paired(d):
    """mean of the paired differences with its 95% interval."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    se = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se, len(d)


def load_units(docs: Path, stems):
    """Per-unit tables indexed by unit_id, or None when a file is missing."""
    out = []
    for stem in stems:
        f = docs / f"{stem}.per_unit.csv"
        out.append(pd.read_csv(f).set_index("unit_id") if f.exists() else None)
    return out


def combined(tables, combine):
    """One per-unit table of paired differences: columns d_nll and d_<stat>, den_<stat> (the control's CRPS)."""
    tables = [t for t in tables if t is not None]
    if not tables:
        return None
    idx = tables[0].index
    for t in tables[1:]:
        idx = idx.intersection(t.index)
    out = pd.DataFrame(index=idx)
    k = len(tables) if combine == "mean" else 1
    out["d_nll"] = sum(t.loc[idx, "nll_without"] - t.loc[idx, "nll_with"] for t in tables) / k
    for s in LABELS:
        out[f"d_{s}"] = sum(t.loc[idx, f"crps_{s}_without"] - t.loc[idx, f"crps_{s}_with"] for t in tables) / k
        out[f"den_{s}"] = sum(t.loc[idx, f"crps_{s}_without"] for t in tables) / k
    return out


def season_table(docs: Path, season, combine):
    """The per-unit differences behind one season's row, and the name of what was combined."""
    group, label, fits, pooled = season
    if len(fits) == 1:
        return combined(load_units(docs, fits), "mean"), "fit 1"
    if combine == "mixture":
        return combined(load_units(docs, [pooled]), "mixture"), "mixture of the fits"
    return combined(load_units(docs, fits), "mean"), "mean over the fits"


def build_rows(docs: Path, combine: str):
    registered = {}
    with (OUT / "evidence_source.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            registered[(r["season"], r["fit"])] = r
    rows = []
    for season in SEASONS:
        group, label, fits, pooled = season
        tab, how = season_table(docs, season, combine)
        row = dict(evaluation=group, season=label, combined=how, n="", source="")
        reg = registered.get((label, "1")) if len(fits) == 1 else None
        if reg is not None:                                   # single fit: the registered log-score values
            row.update(mean_log_score_gain=float(reg["mean_log_score_gain"]), ci95_low_log=float(reg["ci95_low_log"]),
                       ci95_high_log=float(reg["ci95_high_log"]), source=f"registered: {reg['source']}")
        if tab is not None:
            g, lo, hi, n = paired(tab["d_nll"])
            row["n"] = str(n)
            check = f"  scored {g:.3f} [{lo:.3f}, {hi:.3f}]" if reg is not None else ""
            if reg is None:
                row.update(mean_log_score_gain=g, ci95_low_log=lo, ci95_high_log=hi,
                           source=f"{how} from the per-unit files {', '.join(fits if combine == 'mean' or len(fits) == 1 else [pooled])}")
        else:
            check = "  (no per-unit file)"
        if "mean_log_score_gain" not in row:
            print(f"skip {group} {label}: no registered value and no per-unit file")
            continue
        if not row["ci95_low_log"] <= row["mean_log_score_gain"] <= row["ci95_high_log"]:
            raise ValueError(f"Interval does not contain estimate: {row}")
        row["geometric_mean_probability_ratio"] = math.exp(row["mean_log_score_gain"])
        row["ci95_low_ratio"] = math.exp(row["ci95_low_log"])
        row["ci95_high_ratio"] = math.exp(row["ci95_high_log"])
        if row["ci95_low_ratio"] < .5 or row["ci95_high_ratio"] > 16:
            raise ValueError("Result exceeds axis [0.5,16]; revise limits explicitly.")
        rows.append(row)
        print(f"{group} {label} ({how}): gain {row['mean_log_score_gain']:.3f} [{row['ci95_low_log']:.3f}, {row['ci95_high_log']:.3f}] "
              f"= {row['geometric_mean_probability_ratio']:.2f}x [{row['ci95_low_ratio']:.2f}, {row['ci95_high_ratio']:.2f}] n {row['n']}{check}")
    if len(rows) < 2:
        raise ValueError("Fewer than two rows available.")
    fields = ["evaluation", "season", "combined", "n", "mean_log_score_gain", "ci95_low_log", "ci95_high_log",
              "geometric_mean_probability_ratio", "ci95_low_ratio", "ci95_high_ratio", "source"]
    with (OUT / "evidence_crps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def crps_by_statistic(docs: Path, stats, combine):
    """CRPS skill per statistic on the test season."""
    tab, how = season_table(docs, TEST, combine)
    if tab is None:
        print("no per-unit file for the test season: the CRPS block is skipped")
        return []
    out = []
    for k in stats:
        m, lo, hi, n = paired(tab[f"d_{k}"])
        den = tab[f"den_{k}"].mean()
        out.append(dict(stat=k, label=LABELS[k].replace("season ", ""), skill=100 * m / den, lo=100 * lo / den, hi=100 * hi / den, n=n, combined=how))
        print(f"CRPS {LABELS[k]} ({how}): {out[-1]['skill']:+.1f}% [{out[-1]['lo']:+.1f}, {out[-1]['hi']:+.1f}] (n {n})")
    with (OUT / "evidence_crps_test.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stat", "label", "skill", "lo", "hi", "n", "combined"])
        w.writeheader()
        w.writerows(out)
    return out


def _pct(v):
    """Signed whole percent; a value that rounds to zero prints as 0, never -0."""
    r = int(round(v))
    return f"{r:+d}" if r else "0"


def draw_evidence(fig, c, rows, crps_rows, texts, crps_label, show_training=False):
    """Left: one probability-ratio row per season. Right: CRPS skill by statistic on the test season.
    Returns the figure fraction below which the canvas is empty (the saved figure is cropped there)."""
    ys, headers, gaps = [], [], []
    y, previous = 0.0, None
    for row in rows:
        if row["evaluation"] != previous:
            if previous is not None:
                gaps.append(y + 0.15)
                y -= 0.75
            headers.append((y + (0.70 if show_training else 0.62), row["evaluation"]))
            previous = row["evaluation"]
        ys.append(y)
        y -= 1
    ys2 = [-i for i in range(len(crps_rows))]
    ylim = (min(ys[-1], ys2[-1] if ys2 else 0) - 0.6, 1.0)
    top = 0.372
    h = min(0.30, 0.037 * (ylim[1] - ylim[0]))
    box, box2 = [0.235, top - h, 0.17, h], [0.715, top - h, 0.125, h]
    groups = list(dict.fromkeys(r["evaluation"] for r in rows))
    colors = {g: (c["purple"] if i == 0 else c["green"]) for i, g in enumerate(groups)}
    val_fs = 9.3

    # Titles
    texts.append(fig.text(LEFT, 0.445, "Model Evaluation", fontsize=15, weight="bold", color=c["ink"], va="bottom"))
    texts.append(fig.text(LEFT, 0.421, "comparison of the model with and without post-NCAA data", fontsize=10.5, color=c["muted"], va="bottom"))
    texts.append(fig.text(LEFT, 0.400, "every row is out of sample: its models are trained only on seasons before the one scored",
                          fontsize=9.6, color=c["muted"], va="bottom"))

    # Left block: probability ratios
    ax = fig.add_axes(box, facecolor=c["bg"])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 16)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator([.5, 1, 2, 4, 8, 16]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
    ax.minorticks_off()
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=c["muted"], length=0, pad=7, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvspan(.5, 1, color=c["shade"], zorder=0)
    ax.grid(axis="x", color=c["grid"], lw=.7, zorder=1)
    ax.axvline(1, color=c["muted"], ls=(0, (3, 3)), lw=1.1, zorder=2)
    for row, yy in zip(rows, ys):
        col = colors[row["evaluation"]]
        ratio, lo, hi = row["geometric_mean_probability_ratio"], row["ci95_low_ratio"], row["ci95_high_ratio"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=3, alpha=0.85)
        ax.plot(ratio, yy, "o", ms=11, mfc=col, mec=c["bg"], mew=1.5, zorder=4)
    left_lab = (LEFT - box[0]) / box[2]
    right_edge = (0.60 - box[0]) / box[2]
    for row, yy in zip(rows, ys):
        n = row["n"].strip()
        a = ax.text(left_lab, yy + (0.17 if show_training else 0), row["season"] + (f" · n = {n}" if n else ""), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=10.5, color=c["ink"])
        texts.append(a)
        if show_training:                                      # fold v trains through season v-2 (fpp/experiments/train.py)
            end = int(row["season"][:4]) + 1
            through = f"{end - 3}–{(end - 2) % 100:02d}"
            a = ax.text(left_lab, yy - 0.27, f"trained through {through}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.4, color=c["muted"])
            texts.append(a)
        a = ax.text(1.04, yy, f"{row['geometric_mean_probability_ratio']:.2f}× [{row['ci95_low_ratio']:.2f}, {row['ci95_high_ratio']:.2f}]",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=val_fs, color=c["ink"])
        texts.append(a)
    for yy, label in headers:                                   # a long group name may run over the axis's shaded edge; the bbox keeps it clean
        a = ax.text(left_lab, yy, label, transform=ax.get_yaxis_transform(), fontsize=10.0, weight="bold", color=colors[label], ha="left", va="center",
                    zorder=10, bbox=dict(facecolor=c["bg"], edgecolor="none", pad=1.5))
        texts.append(a)
    for yy in gaps:
        ax.plot([left_lab, right_edge], [yy, yy], transform=ax.get_yaxis_transform(), color=c["border"], lw=.7, clip_on=False)
    a = ax.text(1.04, headers[0][0], "ratio [95% CI]", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.8, color=c["muted"])
    texts.append(a)
    x_one = box[0] + box[2] * math.log(2) / math.log(32)
    texts.append(fig.text(x_one, top + 0.004, "no change", ha="center", va="bottom", fontsize=9.8, color=c["muted"]))
    cap = box[1] - 0.055
    x_col = (LEFT + 0.60) / 2 + 0.03
    texts.append(fig.text(x_col, cap, "times more probable the actual seasons were with the post-NCAA data", ha="center", fontsize=10.0, color=c["ink"]))
    texts.append(fig.text(x_col, cap - 0.025, "geometric mean over players, log scale; 1× = no change", ha="center", fontsize=9.3, color=c["muted"]))

    # Right block: CRPS by statistic on the test season
    if crps_rows:
        ax2 = fig.add_axes(box2, facecolor=c["bg"])
        lo_all, hi_all = min(r["lo"] for r in crps_rows), max(r["hi"] for r in crps_rows)
        pad = 0.10 * (hi_all - lo_all + 1e-9)
        ax2.set_xlim(min(lo_all - pad, -1.0), hi_all + pad)
        ax2.set_ylim(*ylim)
        ax2.set_yticks([])
        x_lo, x_hi = ax2.get_xlim()
        step = 5 if x_hi - x_lo <= 24 else (10 if x_hi - x_lo <= 60 else 20)
        ax2.xaxis.set_major_locator(FixedLocator([v for v in range(int(math.floor(x_lo / step)) * step, int(math.ceil(x_hi / step)) * step + 1, step) if x_lo <= v <= x_hi]))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _pct(v)))
        ax2.tick_params(axis="x", colors=c["muted"], length=0, pad=7, labelsize=10)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.axvspan(x_lo, 0, color=c["shade"], zorder=0)
        ax2.grid(axis="x", color=c["grid"], lw=.7, zorder=1)
        ax2.axvline(0, color=c["muted"], ls=(0, (3, 3)), lw=1.1, zorder=2)
        col = c["green"]
        lab_x = (0.625 - box2[0]) / box2[2]
        for r, yy in zip(crps_rows, ys2):
            ax2.plot([r["lo"], r["hi"]], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=3, alpha=0.85)
            ax2.plot(r["skill"], yy, "o", ms=11, mfc=col, mec=c["bg"], mew=1.5, zorder=4)
            a = ax2.text(lab_x, yy, r["label"], transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=10.0, color=c["ink"])
            texts.append(a)
            a = ax2.text(1.05, yy, f"{_pct(r['skill'])}% [{_pct(r['lo'])}, {_pct(r['hi'])}]", transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=val_fs, color=c["ink"])
            texts.append(a)
        a = ax2.text(1.05, 0.62, "skill [95% CI]", transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=9.8, color=c["muted"])
        texts.append(a)
        if crps_label:
            a = ax2.text(lab_x, 0.62, crps_label, transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=10.0, weight="bold", color=col)
            texts.append(a)
        texts.append(fig.text(0.625, top + 0.004, "CRPS by statistic, 2025–26 test season", ha="left", va="bottom", fontsize=9.8, color=c["muted"]))
        x2_col = (0.625 + RIGHT) / 2
        texts.append(fig.text(x2_col, cap, "CRPS improvement over the control, %", ha="center", fontsize=10.0, color=c["ink"]))
        texts.append(fig.text(x2_col, cap - 0.025, "share of the control's error removed", ha="center", fontsize=9.3, color=c["muted"]))
    return max(0.0, cap - 0.05)


def make_figure(theme_name, rows, crps_rows, crps_label, tag="", show_training=False):
    c = THEMES[theme_name]
    fig = plt.figure(figsize=(9.0, 8.6), facecolor=c["bg"])
    texts = []
    draw_idea(fig, c, texts)
    crop = draw_evidence(fig, c, rows, crps_rows, texts, crps_label, show_training)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, hgt = fig.get_size_inches()
    keep = Bbox.from_extents(0, crop * hgt, w, hgt)             # the empty strip under the captions is cut off
    floor = fig.bbox.y0 + crop * fig.bbox.height
    for a in texts:                                             # every label on the canvas
        b = a.get_window_extent(renderer)
        assert b.x0 >= fig.bbox.x0 - 1 and b.y0 >= floor - 1 and b.x1 <= fig.bbox.x1 + 1 and b.y1 <= fig.bbox.y1 + 1, (a.get_text(), b)
    suffix = "" if theme_name == "light" else "_dark"
    base = OUT / ("fig_learning_and_evidence_crps" + (f"_{tag}" if tag else "") + suffix)
    fig.savefig(base.with_suffix(".pdf"), facecolor=c["bg"], bbox_inches=keep, pad_inches=0, metadata={
        "Title": "Learn backward. Predict forward.", "Author": "Future Proves Past",
        "Subject": "Exponentiated paired log-score differences and CRPS skill; provenance in evidence_crps.csv and evidence_crps_test.csv"})
    fig.savefig(base.with_suffix(".png"), facecolor=c["bg"], dpi=220, bbox_inches=keep, pad_inches=0)
    strings = [a.get_text() for a in texts] + ["0.5×", "1×", "2×", "4×", "8×", "16×"]
    print(f"{base.name}: {sum(len(s.split()) for s in strings)} words including ticks")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combine", default="mean", choices=["mean", "mixture"], help="how a season's two fits become one row (see the module docstring)")
    parser.add_argument("--docs", type=Path, default=ROOT.parents[1] / "docs")
    parser.add_argument("--tag", default="", help="suffix for a variant file name, e.g. --tag mixture")
    parser.add_argument("--crps-stats", default="minutes,points,rebounds,assists", help="statistics in the CRPS block, in order")
    parser.add_argument("--crps-label", default="", help="optional header of the CRPS block (empty: none)")
    parser.add_argument("--show-training", action="store_true", help="a muted sublabel per row with the last season its models were trained on")
    args = parser.parse_args()
    rows = build_rows(args.docs, args.combine)
    crps_rows = crps_by_statistic(args.docs, args.crps_stats.split(","), args.combine)
    for theme in THEMES:
        make_figure(theme, rows, crps_rows, args.crps_label, args.tag, args.show_training)
