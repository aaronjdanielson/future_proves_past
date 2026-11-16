"""The README's headline scoreboard and its evidence figure.

    python3 replication/readme/make_evidence_figures.py [--csv replication/readme/evidence.csv] [--out assets]

Reads only replication/readme/evidence.csv (built by build_evidence_csv.py; every plotted value is a row of it) and writes
assets/fig_scoreboard{,_dark}.png and assets/fig_evidence{,_dark}.png. Needs matplotlib.

Both show one quantity for freshmen with tracked pre-college games: how many times more probable the season that happened
was under the model trained with the former players' later careers than under its strict control (e^gain, geometric mean
over players), with 95% intervals over players. Colour identifies the kind of evidence, never a value: purple for the
development seasons, green for the registered one-shot test.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "axes.unicode_minus": True})
THEMES = {
    "light": dict(bg="#FFFFFF", ink="#111827", muted="#5B6473", border="#D1D5DB", grid="#E5E7EB", band="#F6F7F9", purple="#6D4FC2", green="#0F8A6B"),
    "dark": dict(bg="#0F172A", ink="#F1F5F9", muted="#94A3B8", border="#334155", grid="#1E293B", band="#141E33", purple="#A78BFA", green="#34D399"),
}


def rows_with_history(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    return d[d["cohort"] == "with history"].reset_index(drop=True)


def pick(d: pd.DataFrame, season: str, ahead: bool = False):
    m = d[(d["season"] == season) & (d["design"].str.contains("further ahead") == ahead)]
    return None if m.empty else m.iloc[0]


def scoreboard(d: pd.DataFrame, theme: str, out: Path) -> Path:
    c = THEMES[theme]
    W, H = 12.0, 3.25
    fig = plt.figure(figsize=(W, H), facecolor=c["bg"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    tiles = [("2021–22", False, "held-out season"), ("2022–23", False, "held-out season"), ("2023–24", True, "held-out season"), ("2025–26", False, "registered one-shot test")]
    gap, left = 0.22, 0.40
    width = (W - 2 * left - gap * (len(tiles) - 1)) / len(tiles)
    ax.text(left, H - 0.40, "How many times more probable the seasons that happened were, with later-career training", color=c["muted"], fontsize=11.5, va="center")
    for i, (season, ahead, kind) in enumerate(tiles):
        r = pick(d, season, ahead); test = season == "2025–26"
        x0 = left + i * (width + gap); cx = x0 + width / 2
        ax.add_patch(FancyBboxPatch((x0, 0.26), width, 2.28, boxstyle="round,pad=0,rounding_size=0.10", facecolor=c["band"], edgecolor=c["border"], linewidth=0.8))
        ax.add_patch(FancyBboxPatch((x0 + 0.24, 2.27), 0.34, 0.055, boxstyle="round,pad=0,rounding_size=0.027", facecolor=c["green"] if test else c["purple"], edgecolor="none"))
        ax.text(x0 + 0.24, 2.05, season, color=c["ink"], fontsize=13, fontweight="bold", va="center")
        ax.text(x0 + 0.24, 1.78, kind, color=c["muted"], fontsize=9.6, va="center")
        ax.text(cx, 1.13, f"{r['ratio']:.1f}×", color=c["ink"], fontsize=37, fontweight="bold", ha="center", va="center")
        note = "scored September 2026" if test else "a season further ahead" if ahead else f"{int(r['fits'])} fits"
        ax.text(cx, 0.52, f"n {int(r['n'])} · {note}", color=c["muted"], fontsize=9.6, ha="center", va="center")
    path = out / f"fig_scoreboard{'_dark' if theme == 'dark' else ''}.png"
    fig.savefig(path, dpi=200, facecolor=c["bg"]); plt.close(fig)
    return path


def evidence(d: pd.DataFrame, theme: str, out: Path) -> Path:
    c = THEMES[theme]
    plan = [("2021–22", [(False, "four fits")]),
            ("2022–23", [(False, "four fits"), (True, "a season further ahead · two fits mixed")]),
            ("2023–24", [(True, "a season further ahead · two fits mixed")]),
            ("2025–26", [(False, "two fits · scored September 2026")])]
    y, ys, heads = 0.0, [], []
    for season, items in plan:
        heads.append((y, season)); y -= 0.92
        for item in items:
            ys.append((y, season, item)); y -= 0.84
        y -= 0.40
    y_lo, y_hi = y + 0.40 - 0.30, 0.6
    W, H = 12.0, 1.70 + 0.37 * (y_hi - y_lo)
    fig = plt.figure(figsize=(W, H), facecolor=c["bg"])
    top, bottom = 1 - 1.16 / H, 0.62 / H
    ax = fig.add_axes([0.345, bottom, 0.455, top - bottom], facecolor=c["bg"])
    ax.set_xscale("log"); ax.set_xlim(0.72, 9.5); ax.set_ylim(y_lo, y_hi)
    ticks = [1, 2, 4, 8]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}×" for t in ticks], color=c["muted"], fontsize=10.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    for t in ticks:
        ax.axvline(t, color=c["border"] if t == 1 else c["grid"], linewidth=1.2 if t == 1 else 0.8, zorder=0)
    ax.text(1, 1.012, "no change", color=c["muted"], fontsize=9.5, ha="center", va="bottom", transform=ax.get_xaxis_transform())
    fy = lambda yy: bottom + (yy - y_lo) / (y_hi - y_lo) * (top - bottom)        # noqa: E731
    for yy, season in heads:
        n = int(pick(d, season, season == "2023–24")["n"])
        fig.text(0.04, fy(yy), season, color=c["ink"], fontsize=12.5, fontweight="bold", va="center")
        fig.text(0.112, fy(yy), f"n {n}" + (" · registered one-shot test" if season == "2025–26" else ""), color=c["muted"], fontsize=10.5, va="center")
    for yy, season, (ahead, label) in ys:
        r = pick(d, season, ahead); colour = c["green"] if season == "2025–26" else c["purple"]
        fig.text(0.055, fy(yy), label, color=c["ink"], fontsize=10.6, va="center")
        ax.plot([r["ci95_low_ratio"], r["ci95_high_ratio"]], [yy, yy], color=colour, linewidth=2.4, solid_capstyle="round", zorder=2)
        if isinstance(r["per_fit_gains"], str) and r["per_fit_gains"].strip():
            fits = [math.exp(float(v)) for v in r["per_fit_gains"].split()]
            ax.plot(fits, [yy - 0.27] * len(fits), linestyle="none", marker="o", markersize=4.2, color=colour, alpha=0.8, markeredgecolor=c["bg"], markeredgewidth=0.8, zorder=3)
        ax.plot([r["ratio"]], [yy], marker="o", markersize=11.5, color=colour, markeredgecolor=c["bg"], markeredgewidth=2.0, zorder=4)
        fig.text(0.825, fy(yy), f"{r['ratio']:.1f}×", color=c["ink"], fontsize=12, fontweight="bold", va="center")
        fig.text(0.876, fy(yy), f"[{r['ci95_low_ratio']:.1f}, {r['ci95_high_ratio']:.1f}]", color=c["muted"], fontsize=10.5, va="center")
    fig.text(0.04, 1 - 0.40 / H, "Later-career training, season by season", color=c["ink"], fontsize=15.5, fontweight="bold", va="center")
    fig.text(0.04, 1 - 0.74 / H, "how many times more probable the seasons that happened were · freshmen with tracked pre-college games · 95% intervals",
             color=c["muted"], fontsize=10.2, va="center")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=9.5, color=c["purple"], markeredgecolor=c["bg"], markeredgewidth=1.5, label="development seasons"),
               Line2D([], [], linestyle="none", marker="o", markersize=9.5, color=c["green"], markeredgecolor=c["bg"], markeredgewidth=1.5, label="registered test"),
               Line2D([], [], linestyle="none", marker="o", markersize=4.2, color=c["muted"], alpha=0.85, label="single fits")]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.036, 0.02 / H), ncol=3, frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(c["muted"])
    path = out / f"fig_evidence{'_dark' if theme == 'dark' else ''}.png"
    fig.savefig(path, dpi=200, facecolor=c["bg"]); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "replication/readme/evidence.csv")); ap.add_argument("--out", default=str(ROOT / "assets"))
    args = ap.parse_args()
    d, out = rows_with_history(Path(args.csv)), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        print("wrote", scoreboard(d, theme, out)); print("wrote", evidence(d, theme, out))


if __name__ == "__main__":
    main()
