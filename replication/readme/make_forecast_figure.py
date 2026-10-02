"""The README's 2026-27 forecast figure: where the former players' careers change the forecast most.

    python3 replication/readme/make_forecast_figure.py [--docs docs] [--out assets] [--up 7] [--down 5]

Reads only files in the release (`results_docs.zip`, unpacked into docs/) and the public cohort list:
    docs/FORECASTS_2026_27_deploy_v11_pooled.json          the model with the post-NCAA data (frozen 2026-10-01)
    docs/FORECASTS_2026_27_deploy_control_v11_pooled.json  the strict control
    docs/FORECAST_RATES_2026_27_deploy_v11_pooled.json     rate forecasts (PER with and without)
    data/forecast_2027_cohort.csv                          names, where the frozen files left one blank
Players: the --up largest and --down largest changes in P(400+ minutes) between the two models, the same rule as the
README's tables. Panels: P(400+ minutes); season minutes, mean and 90% interval (5th to 95th percentile of 2,000
simulated seasons per model); PER (rate forecast). Writes assets/fig_forecast_2027{,_dark}.png. Needs matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "axes.unicode_minus": True})
THEMES = {
    "light": dict(bg="#FFFFFF", ink="#111827", muted="#5B6473", border="#D1D5DB", grid="#E5E7EB", band="#F6F7F9",
                  purple="#6D4FC2", red="#C0504D"),
    "dark": dict(bg="#0F172A", ink="#F1F5F9", muted="#94A3B8", border="#334155", grid="#1E293B", band="#141E33",
                 purple="#A78BFA", red="#F87171"),
}


def load(docs: Path, select: str, up: int, down: int) -> pd.DataFrame:
    w = pd.DataFrame(json.loads((docs / "FORECASTS_2026_27_deploy_v11_pooled.json").read_text()))
    c = pd.DataFrame(json.loads((docs / "FORECASTS_2026_27_deploy_control_v11_pooled.json").read_text()))
    r = pd.DataFrame(json.loads((docs / "FORECAST_RATES_2026_27_deploy_v11_pooled.json").read_text()))
    cohort = pd.read_csv(ROOT / "data" / "forecast_2027_cohort.csv")[["player_id", "name", "nationality", "of_interest"]]
    cohort = cohort.rename(columns={"name": "cohort_name"})
    m = w.merge(c[["player_id", "p_rotation_400min", "minutes"]], on="player_id", suffixes=("", "_c"))
    m = m.merge(r[["player_id", "per_with", "per_without"]], on="player_id").merge(cohort, on="player_id", how="left")
    m["name"] = m["name"].where(m["name"].notna() & (m["name"].astype(str) != "nan"), m["cohort_name"])
    m["delta"] = m["p_rotation_400min"] - m["p_rotation_400min_c"]
    if select == "international":
        # The players of interest whose nationality is not the United States alone, by the full model's P(400+ min).
        intl = (m["of_interest"] == 1) & m["nationality"].notna() & (m["nationality"] != "United States")
        return m[intl].sort_values("p_rotation_400min", ascending=False).assign(group="international").reset_index(drop=True)
    m = m.sort_values("delta", ascending=False)
    raised, lowered = m.head(up).assign(group="raised"), m.tail(down).iloc[::-1].assign(group="lowered")
    return pd.concat([raised, lowered], ignore_index=True)


def draw(rows: pd.DataFrame, theme: str, out: Path, title: str, stem: str) -> Path:
    c = THEMES[theme]
    grouped = rows["group"].iloc[0] == "raised"
    n_up = int((rows["group"] == "raised").sum())
    ys, y = [], 0.0
    for i in range(len(rows)):
        if grouped and i == n_up:
            y -= 1.0                                           # gap and header for the second group
        ys.append(y)
        y -= 1.0
    ylim = (ys[-1] - 0.7, 1.15 if grouped else 0.6)
    height = 1.55 + 0.40 * (ylim[1] - ylim[0])
    fig = plt.figure(figsize=(10.5, height), facecolor=c["bg"])
    top, bottom = 1 - 1.05 / height, 0.55 / height
    panels = [("chance of a 400-minute season", 0.30, 0.205, (0, 1)),
              ("season minutes, 90% range", 0.545, 0.215, None),
              ("PER", 0.80, 0.165, None)]
    span_min = (min(rows["minutes"].map(lambda v: v[1]).min(), rows["minutes_c"].map(lambda v: v[1]).min()),
                max(rows["minutes"].map(lambda v: v[2]).max(), rows["minutes_c"].map(lambda v: v[2]).max()))
    span_per = (min(rows["per_with"].min(), rows["per_without"].min()), max(rows["per_with"].max(), rows["per_without"].max()))
    axes = []
    for k, (label, x0, wdt, xlim) in enumerate(panels):
        ax = fig.add_axes([x0, bottom, wdt, top - bottom], facecolor="none")   # transparent: the row bands show through
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(axis="x", color=c["grid"], lw=0.8, zorder=0)
        ax.tick_params(axis="x", colors=c["muted"], length=0, labelsize=9.5, pad=4)
        if k == 0:
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        elif k == 1:
            pad = 0.05 * (span_min[1] - span_min[0])
            ax.set_xlim(span_min[0] - pad, span_min[1] + pad)          # room for the rounded caps at zero minutes
        else:
            pad = 0.12 * (span_per[1] - span_per[0])
            ax.set_xlim(span_per[0] - pad, span_per[1] + pad)
        ax.text(0.0, 1.0, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=c["muted"])
        axes.append(ax)
    a0, a1, a2 = axes

    def dumbbell(ax, yy, without, with_):
        # A plain link between the two forecasts; the colours carry the direction (purple = with the careers).
        ax.plot([without, with_], [yy, yy], color=c["purple"], lw=2.2, alpha=0.6, solid_capstyle="butt", zorder=2)
        ax.plot(without, yy, "o", ms=8, color=c["red"], mec=c["bg"], mew=1.2, zorder=3)
        ax.plot(with_, yy, "o", ms=9.5, color=c["purple"], mec=c["bg"], mew=1.2, zorder=4)

    def fig_y(yy):
        return a0.transData.transform((0, yy))[1] / fig.bbox.height

    for i, ((_, r), yy) in enumerate(zip(rows.iterrows(), ys)):
        if i % 2 == 0:                                         # alternating row bands across the full width
            y0, y1 = fig_y(yy - 0.5), fig_y(yy + 0.5)
            fig.add_artist(Rectangle((0.02, y0), 0.96, y1 - y0, transform=fig.transFigure, facecolor=c["band"],
                                     edgecolor="none", zorder=-1))
        dumbbell(a0, yy, r["p_rotation_400min_c"], r["p_rotation_400min"])
        for vals, col, off in ((r["minutes_c"], c["red"], -0.17), (r["minutes"], c["purple"], 0.17)):
            a1.plot([vals[1], vals[2]], [yy + off, yy + off], color=col, lw=3.0, solid_capstyle="round", alpha=0.85, zorder=3)
            a1.plot(vals[0], yy + off, "o", ms=6.5, color=col, mec=c["bg"], mew=1.0, zorder=4)
        dumbbell(a2, yy, r["per_without"], r["per_with"])
        fig.text(0.03, a0.transData.transform((0, yy))[1] / fig.bbox.height + 0.006, r["name"], ha="left", va="bottom",
                 fontsize=10.5, color=c["ink"], weight="bold")
        where = r["team"] if grouped else f"{r['team']} · {str(r['nationality']).replace('/', ' / ')}"
        fig.text(0.03, a0.transData.transform((0, yy))[1] / fig.bbox.height - 0.004, f"{where} · {int(r['input_games'])} tracked games",
                 ha="left", va="top", fontsize=8.8, color=c["muted"])
    if grouped:
        for i, label in ((0, "Raised most by the careers"), (n_up, "Lowered most by the careers")):
            yy = ys[i] + 0.78
            fig.text(0.03, a0.transData.transform((0, yy))[1] / fig.bbox.height, label, ha="left", va="center", fontsize=10.5,
                     color=c["muted"], style="italic")

    fig.text(0.03, 1 - 0.30 / height, title, ha="left", va="center", fontsize=16, weight="bold", color=c["ink"])
    lx, ly = 0.03, 1 - 0.66 / height
    fig.text(lx, ly, "●", color=c["purple"], fontsize=13, va="center")
    fig.text(lx + 0.018, ly, "with post-NCAA data", color=c["ink"], fontsize=10, va="center")
    fig.text(lx + 0.175, ly, "●", color=c["red"], fontsize=13, va="center")
    fig.text(lx + 0.193, ly, "without", color=c["ink"], fontsize=10, va="center")
    fig.text(lx + 0.27, ly, "frozen 1 October 2026; outcomes arrive in March 2027", color=c["muted"], fontsize=9.5, va="center")

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}{'' if theme == 'light' else '_dark'}.png"
    fig.savefig(path, facecolor=c["bg"], dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=ROOT / "docs")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    ap.add_argument("--select", default="international", choices=["international", "change"],
                    help="international: the high-profile international players of interest (default, the README figure); "
                         "change: the --up/--down largest changes in P(400+ minutes)")
    ap.add_argument("--up", type=int, default=7)
    ap.add_argument("--down", type=int, default=5)
    args = ap.parse_args()
    rows = load(args.docs, args.select, args.up, args.down)
    for _, r in rows.iterrows():
        print(f"{r['group']:13s} {r['name']:26s} {r['team']:18s} P400 {r['p_rotation_400min_c']:.2f} -> {r['p_rotation_400min']:.2f}  "
              f"PER {r['per_without']:.1f} -> {r['per_with']:.1f}")
    title, stem = (("2026–27: High-Profile International Freshmen", "fig_forecast_2027") if args.select == "international"
                   else ("2026–27 Forecasts: What the Former Players' Careers Change", "fig_forecast_2027_changes"))
    for theme in THEMES:
        print("wrote", draw(rows, theme, args.out, title, stem))


if __name__ == "__main__":
    main()
