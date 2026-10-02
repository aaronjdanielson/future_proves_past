"""Rebuild the composite exhibit: python make_submission_figure.py [--results results.csv].

Top half: the idea as a player timeline. Pre-college games (left) feed the forward task, which forecasts the NCAA
season (the target block); the later career abroad (right) feeds the backward task, which reconstructs that same,
already-known season. Both pass through one shared tower (game encoder -> history pooling -> translator -> season
distribution), so the ~10,000 former players teach the network that evaluates recruits.

Bottom half: the paired evidence. Each row is one fit of the full model against the strict control on the same players,
as the geometric-mean ratio of the probability assigned to the seasons that happened (exp of the mean paired log-score
difference) with the transformed 95% interval. Source values: figures/evidence_source.csv (provenance column kept).
Nothing is recomputed here. Requires Python 3 and matplotlib.
"""
from pathlib import Path
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True})

# Palette shared with scripts/make_figures.py (dark: navy ground, purple = with the post-NCAA data, red = control,
# gold = actual, green = accent); the light theme is the paper's print palette.
THEMES = {
    "light": dict(bg="#FFFFFF", ink="#111827", muted="#5B6473", border="#D1D5DB", box="#F3F4F6", grid="#E5E7EB",
                  purple="#6D4FC2", red="#C0504D", gold="#D97706", green="#0F8A6B", shade="#F3F4F6", tint="#EDE9FB"),
    "dark": dict(bg="#0F172A", ink="#F1F5F9", muted="#94A3B8", border="#334155", box="#1E293B", grid="#1E293B",
                 purple="#A78BFA", red="#F87171", gold="#FBBF24", green="#34D399", shade="#162032", tint="#35345E"),
}

# Illustrative game dates (ages) for the timeline: pre-college games and the later career abroad. Positions only.
PRE_GAMES = [16.4, 16.7, 16.9, 17.2, 17.4, 17.5, 17.9, 18.1, 18.3, 18.6, 18.8, 19.0, 19.2, 19.35]
POST_GAMES = [23.3, 23.5, 23.8, 24.0, 24.2, 24.6, 24.9, 25.1, 25.4, 25.7, 26.0, 26.3, 26.6, 26.9, 27.2]
AGE_MIN, AGE_MAX = 15.8, 27.8
NCAA = (19.6, 20.4)              # the target season (age 19-20); later college seasons follow it


def load_results(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not 2 <= len(rows) <= 8:
        raise ValueError("Layout supports 2-8 comparison rows.")
    for row in rows:
        for key in ["mean_log_score_gain", "ci95_low_log", "ci95_high_log"]:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Nonfinite result: {row}")
        if not row["ci95_low_log"] <= row["mean_log_score_gain"] <= row["ci95_high_log"]:
            raise ValueError(f"Interval does not contain estimate: {row}")
        if not row["source"].strip():
            raise ValueError("Every result needs provenance in source.")
        row["geometric_mean_probability_ratio"] = math.exp(row["mean_log_score_gain"])
        row["ci95_low_ratio"] = math.exp(row["ci95_low_log"])
        row["ci95_high_ratio"] = math.exp(row["ci95_high_log"])
        if row["ci95_low_ratio"] < .5 or row["ci95_high_ratio"] > 16:
            raise ValueError("Result exceeds axis [0.5,16]; revise limits explicitly.")
    with (OUT / "learning_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _age_x(age):
    return (age - AGE_MIN) / (AGE_MAX - AGE_MIN)


def draw_idea(fig, c, texts):
    """The timeline and the shared tower, in a figure-fraction axes."""
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.44])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def t(x, y, s, **kw):
        kw.setdefault("color", c["ink"])
        a = ax.text(x, y, s, **kw)
        texts.append(a)
        return a

    # Title
    t(0.0, 0.99, "Learn backward. Predict forward.", fontsize=19, weight="bold", va="top")
    t(0.0, 0.905, "one shared network, trained by both tasks", fontsize=10.5, va="top", color=c["muted"])

    # The shared tower: four stages in one rounded band
    band_y, band_h = 0.56, 0.15
    ax.add_patch(FancyBboxPatch((0.17, band_y), 0.66, band_h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                lw=1.4, edgecolor=c["purple"], facecolor=c["tint"], zorder=2))
    stages = ["game\nencoder", "history\npooling", "translator", "season\ndistribution"]
    xs = [0.245, 0.405, 0.565, 0.735]
    for i, (x, s) in enumerate(zip(xs, stages)):
        ax.add_patch(FancyBboxPatch((x - 0.062, band_y + 0.025), 0.124, band_h - 0.05, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    lw=1.0, edgecolor=c["purple"], facecolor=c["box"], zorder=3))
        t(x, band_y + band_h / 2, s, fontsize=10.2, ha="center", va="center", weight="bold", zorder=4)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 0.064, band_y + band_h / 2), (xs[i + 1] - 0.064, band_y + band_h / 2),
                                         arrowstyle="-|>", mutation_scale=11, lw=1.2, color=c["purple"], zorder=4))
    t(0.17, band_y + band_h + 0.03, "Forward (→):\nforecast the season\nfrom what came before", fontsize=9.6, ha="left", va="bottom", color=c["green"], weight="bold")
    t(0.83, band_y + band_h + 0.03, "Backward (←):\nreconstruct the known season\nfrom what came after", fontsize=9.6, ha="right", va="bottom", color=c["purple"], weight="bold")

    # Timeline
    ty = 0.22
    ax.plot([0.0, 1.0], [ty, ty], color=c["border"], lw=1.6, zorder=1)
    for age in range(16, 28, 2):
        x = _age_x(age)
        ax.plot([x, x], [ty - 0.018, ty + 0.018], color=c["border"], lw=1.2)
        if not NCAA[0] <= age <= NCAA[1]:              # the tick under the target block stays unlabelled
            t(x, ty - 0.045, f"{age}", fontsize=9.5, ha="center", va="top", color=c["muted"])
    t(1.0, ty - 0.045, "age", fontsize=9.5, ha="right", va="top", color=c["muted"])
    # the target season
    x0, x1 = _age_x(NCAA[0]), _age_x(NCAA[1])
    ax.add_patch(Rectangle((x0, ty - 0.075), x1 - x0, 0.15, facecolor=c["gold"], edgecolor="none", alpha=0.95, zorder=3))
    t((x0 + x1) / 2, ty + 0.095, "NCAA season\n(the target)", fontsize=10, ha="center", va="bottom", weight="bold", color=c["gold"])
    # later college seasons, faint
    x2 = _age_x(23.0)
    ax.add_patch(Rectangle((x1 + 0.004, ty - 0.05), x2 - x1 - 0.004, 0.10, facecolor=c["box"], edgecolor=c["border"], lw=0.8, zorder=2))
    t((x1 + x2) / 2, ty - 0.075, "later college seasons", fontsize=8.8, ha="center", va="top", color=c["muted"])
    # games
    for age in PRE_GAMES:
        ax.plot(_age_x(age), ty, "o", ms=6.5, color=c["green"], mec=c["bg"], mew=0.8, zorder=4)
    for age in POST_GAMES:
        ax.plot(_age_x(age), ty, "o", ms=6.5, color=c["purple"], mec=c["bg"], mew=0.8, zorder=4)
    t(_age_x(17.9), ty - 0.075, "pre-college games\nclubs, youth and national teams", fontsize=9.5, ha="center", va="top", color=c["green"])
    t(_age_x(25.3), ty - 0.075, "the career abroad after college\nleagues and national teams", fontsize=9.5, ha="center", va="top", color=c["purple"])

    # The two tasks: curved arrows from the games into the tower, and the tower's output onto the target season
    kw = dict(arrowstyle="-|>", mutation_scale=15, lw=2.0, zorder=5)
    ax.add_patch(FancyArrowPatch((_age_x(18.6), ty + 0.035), (0.215, band_y + 0.005), connectionstyle="arc3,rad=0.2", color=c["green"], **kw))
    ax.add_patch(FancyArrowPatch((_age_x(24.6), ty + 0.035), (0.785, band_y + 0.005), connectionstyle="arc3,rad=-0.2", color=c["purple"], **kw))
    ax.add_patch(FancyArrowPatch((0.735, band_y - 0.005), ((x0 + x1) / 2 + 0.01, ty + 0.17), connectionstyle="arc3,rad=0.15", color=c["gold"], **kw))
    t(0.47, 0.335, "same season,\nsame weights", fontsize=9.5, ha="left", va="center", color=c["gold"])
    t(0.0, 0.0, "one recruit", fontsize=9.5, ha="left", va="bottom", color=c["green"])
    t(1.0, 0.0, "≈10,000 former NCAA players", fontsize=9.5, ha="right", va="bottom", color=c["purple"])
    return ax


def _has_crps(rows):
    return all(str(r.get("crps_skill_pct", "")).strip() for r in rows)


def draw_evidence(fig, c, rows, texts):
    with_crps = _has_crps(rows)
    ax = fig.add_axes([0.26, 0.095, 0.30, 0.29] if with_crps else [0.30, 0.095, 0.44, 0.29], facecolor=c["bg"])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 16)
    n = len(rows)
    ys, headers, gaps = [], [], []
    y, previous = 0.0, None
    for row in rows:
        if row["evaluation"] != previous:
            if previous is not None:
                gaps.append(y + 0.15)
                y -= 0.75
            headers.append((y + 0.62, row["evaluation"] + (f"  ·  n = {row['n'].strip()}" if row["n"].strip() else ""), row["evaluation"]))
            previous = row["evaluation"]
        ys.append(y)
        y -= 1
    ax.set_ylim(ys[-1] - 0.6, 1.0)
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
    groups = list(dict.fromkeys(r["evaluation"] for r in rows))
    colors = {g: (c["purple"] if i == 0 else c["green"]) for i, g in enumerate(groups)}
    seen = {}
    for row, yy in zip(rows, ys):
        col = colors[row["evaluation"]]
        ratio, lo, hi = row["geometric_mean_probability_ratio"], row["ci95_low_ratio"], row["ci95_high_ratio"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=3, alpha=0.85)
        ax.plot(ratio, yy, "o", ms=11, mfc=col, mec=c["bg"], mew=1.5, zorder=4)
        key = (row["evaluation"], row["season"])
        seen[key] = seen.get(key, 0) + 1
    repeats = {k: v for k, v in seen.items()}
    left_lab = -0.78 if with_crps else -0.52
    val_x = 1.05 if with_crps else 1.06
    for row, yy in zip(rows, ys):
        label = row["season"] + (f" · fit {row['fit']}" if repeats[(row["evaluation"], row["season"])] > 1 else "")
        a = ax.text(left_lab, yy, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10.5, color=c["ink"])
        texts.append(a)
        a = ax.text(val_x, yy, f"{row['geometric_mean_probability_ratio']:.2f}×  [{row['ci95_low_ratio']:.2f}, {row['ci95_high_ratio']:.2f}]",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.6 if with_crps else 10.2, color=c["ink"])
        texts.append(a)
    for yy, label, group in headers:
        a = ax.text(left_lab, yy, label, transform=ax.get_yaxis_transform(), fontsize=10.2, weight="bold", color=colors[group], ha="left", va="center")
        texts.append(a)
    right_edge = 2.45 if with_crps else 1.62
    for yy in gaps:
        ax.plot([left_lab, right_edge], [yy, yy], transform=ax.get_yaxis_transform(), color=c["border"], lw=.7, clip_on=False)
    a = ax.text(val_x, headers[0][0], "ratio [95% CI]", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10, color=c["muted"])
    texts.append(a)
    x_axis = (0.26 + 0.30 / 5) if with_crps else (0.30 + 0.44 / 5)
    texts.append(fig.text(x_axis, 0.385, "no change", ha="center", va="bottom", fontsize=9.8, color=c["muted"]))
    texts.append(fig.text(0.04, 0.44, "What the later careers add", fontsize=15, weight="bold", color=c["ink"], va="bottom"))
    texts.append(fig.text(0.04, 0.415, "the full model against the strict control, paired on the same players", fontsize=10.5, color=c["muted"], va="bottom"))
    if with_crps:
        # Second column: CRPS skill for one featured statistic, the same rows. skill = (CRPS without - CRPS with) / CRPS without.
        ax2 = fig.add_axes([0.72, 0.095, 0.19, 0.29], facecolor=c["bg"])
        vals = [(float(r["crps_skill_pct"]), float(r["crps_lo_pct"]), float(r["crps_hi_pct"])) for r in rows]
        lo_all = min(v[1] for v in vals); hi_all = max(v[2] for v in vals)
        pad = 0.12 * (hi_all - lo_all + 1e-9)
        ax2.set_xlim(min(lo_all - pad, -1.0), hi_all + pad)
        ax2.set_ylim(ys[-1] - 0.6, 1.0)
        ax2.set_yticks([])
        ax2.tick_params(axis="x", colors=c["muted"], length=0, pad=7, labelsize=10)
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:+.0f}%" if v else "0"))
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.grid(axis="x", color=c["grid"], lw=.7, zorder=1)
        ax2.axvline(0, color=c["muted"], ls=(0, (3, 3)), lw=1.1, zorder=2)
        for (s, lo, hi), row, yy in zip(vals, rows, ys):
            col = colors[row["evaluation"]]
            ax2.plot([lo, hi], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=3, alpha=0.85)
            ax2.plot(s, yy, "o", ms=11, mfc=col, mec=c["bg"], mew=1.5, zorder=4)
            a = ax2.text(1.06, yy, f"{s:+.0f}%  [{lo:+.0f}, {hi:+.0f}]", transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=9.6, color=c["ink"])
            texts.append(a)
        stat = rows[0].get("crps_stat", "").strip() or "season production"
        a = ax2.text(1.06, headers[0][0], "CRPS skill [95% CI]", transform=ax2.get_yaxis_transform(), ha="left", va="center", fontsize=10, color=c["muted"])
        texts.append(a)
        texts.append(fig.text(0.72 + 0.19 / 2, 0.385, f"CRPS, {stat}", ha="center", va="bottom", fontsize=9.8, color=c["muted"]))
        texts.append(fig.text(0.41, 0.04, "probability assigned to the seasons that happened, as a multiple of the control's", ha="center", fontsize=10.2, color=c["ink"]))
        texts.append(fig.text(0.41, 0.015, "geometric mean across players; log scale", ha="center", fontsize=9.4, color=c["muted"]))
        texts.append(fig.text(0.815, 0.04, "CRPS improvement over the control", ha="center", fontsize=10.2, color=c["ink"]))
        texts.append(fig.text(0.815, 0.015, "share of the control's CRPS removed", ha="center", fontsize=9.4, color=c["muted"]))
    else:
        texts.append(fig.text(0.52, 0.04, "probability assigned to the seasons that happened, as a multiple of the control's", ha="center", fontsize=10.5, color=c["ink"]))
        texts.append(fig.text(0.52, 0.015, "geometric mean across players; log scale", ha="center", fontsize=9.6, color=c["muted"]))
    return ax


def make_figure(theme_name, rows):
    c = THEMES[theme_name]
    fig = plt.figure(figsize=(9.0, 8.6), facecolor=c["bg"])
    texts = []
    draw_idea(fig, c, texts)
    draw_evidence(fig, c, rows, texts)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in texts:                                             # every label on the canvas
        b = a.get_window_extent(renderer)
        assert b.x0 >= fig.bbox.x0 - 1 and b.y0 >= fig.bbox.y0 - 1 and b.x1 <= fig.bbox.x1 + 1 and b.y1 <= fig.bbox.y1 + 1, (a.get_text(), b)
    suffix = "" if theme_name == "light" else "_dark"
    base = OUT / ("fig_learning_and_evidence" + suffix)
    fig.savefig(base.with_suffix(".pdf"), facecolor=c["bg"], metadata={
        "Title": "Learn backward. Predict forward.", "Author": "Future Proves Past",
        "Subject": "Exponentiated paired log-score differences; provenance in evidence_source.csv"})
    fig.savefig(base.with_suffix(".png"), facecolor=c["bg"], dpi=220)
    strings = [a.get_text() for a in texts] + ["0.5×", "1×", "2×", "4×", "8×", "16×"]
    print(f"{base.name}: {sum(len(s.split()) for s in strings)} words including ticks")
    print("Text: " + " | ".join(s.replace(chr(10), " ") for s in strings))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT / "evidence_source.csv")
    rows = load_results(parser.parse_args().results)
    for theme in THEMES:
        make_figure(theme, rows)
