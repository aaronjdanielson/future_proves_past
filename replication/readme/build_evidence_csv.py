"""Collect the numbers behind the README's scoreboard and evidence figure into one small file.

    python3 replication/readme/build_evidence_csv.py [--docs docs] [--out replication/readme/evidence.csv]

Every row is a paired comparison on one cohort of freshmen: the log-score gain of the model trained with the former
players' later careers over its strict control, nats per player, with a 95% interval over players (conditional on the
fitted models), the probability ratio e^gain and, where per-player gains are at hand, the share of players improved.

  development seasons    2021–22 and 2022–23: four paired fits per season, gains averaged player by player, from
                         docs/FIRST_RESULT_V12v9statP20_vs_V12AexclStatP20_<fold>{,_s2,_s3,_s4}.per_unit.csv;
                         2022–23 and 2023–24 a season further ahead: two fits mixed, from docs/LATENT_CAREER_COMPARISON.json
                         (its "A: strict → AR" tables)
  registered test        2025–26, scored once in September 2026 with the inputs the model had then, copied from
                         replication/abstract/figures/evidence_crps.csv; a one-shot test is not rescored
The development rows' docs/ files join a release at the end.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COHORTS = (("first_year_with_history", "with history"), ("intl_entrant", "international entrants"),
           ("first_year_without_history", "no history"), ("first_year", "all freshmen"))
SUFFIXES = ("", "_s2", "_s3", "_s4")


def interval(x: np.ndarray) -> tuple[float, float, float]:
    mean = float(x.mean()); se = float(x.std(ddof=1)) / math.sqrt(len(x))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "docs")); ap.add_argument("--out", default=str(ROOT / "replication/readme/evidence.csv"))
    args = ap.parse_args()
    docs, rows = Path(args.docs), []

    def add(evidence, season, design, cohort, fits, n, gain, lo, hi, share, per_fit, source):
        rows.append({"evidence": evidence, "season": season, "design": design, "cohort": cohort, "fits": fits, "n": n,
                     "gain_nats": gain, "ci95_low_nats": lo, "ci95_high_nats": hi, "ratio": math.exp(gain), "ci95_low_ratio": math.exp(lo),
                     "ci95_high_ratio": math.exp(hi), "share_improved": share, "per_fit_gains": per_fit, "source": source})

    for fold, season in ((2022, "2021–22"), (2023, "2022–23")):
        frames = [pd.read_csv(docs / f"FIRST_RESULT_V12v9statP20_vs_V12AexclStatP20_{fold}{s}.per_unit.csv").set_index("unit_id") for s in SUFFIXES]
        if any(not f.index.equals(frames[0].index) for f in frames[1:]):
            raise SystemExit(f"fold {fold}: the four fits score different units")
        gains = np.stack([(f["without_joint"] - f["with_joint"]).to_numpy() for f in frames])
        if not np.isfinite(gains).all():
            raise SystemExit(f"fold {fold}: non-finite gain")
        for key, name in COHORTS:
            m = frames[0][key].to_numpy(bool) & frames[0]["first_year"].to_numpy(bool)
            if not frames[0].loc[m, "player_id"].is_unique:
                raise SystemExit(f"fold {fold}, {name}: a player appears twice")
            avg = gains[:, m].mean(0)
            mean, lo, hi = interval(avg)
            add("development season", season, "four paired fits, gains averaged player by player", name, 4, int(m.sum()), mean, lo, hi,
                float((avg > 0).mean()), " ".join(f"{g:+.3f}" for g in gains[:, m].mean(1)),
                f"docs/FIRST_RESULT_V12v9statP20_vs_V12AexclStatP20_{fold}{{,_s2,_s3,_s4}}.per_unit.csv")

    tables = json.loads((docs / "LATENT_CAREER_COMPARISON.json").read_text())["tables"]
    for key, name in COHORTS:
        for season, s in zip(("2022–23", "2023–24"), tables[f"A: strict → AR|mixture|{key}"]["splits"]):
            add("development season", season, "a season further ahead, two fits mixed", name, 2, int(s["n"]), float(s["mean"]), float(s["lo"]), float(s["hi"]),
                float("nan"), "", "docs/LATENT_CAREER_COMPARISON.json, table 'A: strict → AR', mixture")

    record = pd.read_csv(ROOT / "replication/abstract/figures/evidence_crps.csv")
    test = record[record["season"] == "2025–26"].iloc[0]
    add("registered test", "2025–26", "registered one-shot test, two fits averaged; scored September 2026", "with history", 2, int(test.n),
        float(test.mean_log_score_gain), float(test.ci95_low_log), float(test.ci95_high_log), 0.50, "+0.26 +0.52",
        "replication/abstract/figures/evidence_crps.csv; share improved and per-fit gains from TEST_RESULTS_2026.md")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.6g")
    print(out[out.cohort == "with history"][["evidence", "season", "design", "fits", "n", "gain_nats", "ratio", "ci95_low_ratio", "ci95_high_ratio", "share_improved"]].to_string(index=False))
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
