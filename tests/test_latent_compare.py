"""The D-115 comparison report (scripts/latent_compare.py): finite-input checks, the draw ladder and the verdict gating, on
synthetic score files (no tables, no fitted model)."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latent_compare", ROOT / "scripts" / "latent_compare.py")
LCMP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LCMP)

FOLDS, SEEDS, N, ENTRANTS = [2022, 2023], [20260901, 20260902], 400, 60


def _fixture(tmp: Path, *, entrant_shift=0.0, draw_bias=0.0, penalty=0.1, hetero=False, checks=None, edit=None) -> list[str]:
    """Score files of the registered design. C's entrant loss = A's − entrant_shift + draw_bias·(64/K) + noise, so a positive
    draw_bias reproduces the upward finite-draw bias that shrinks with the draw count (``hetero``: that bias on every other
    unit only, doubled); ``penalty`` is C's loss on the other units. ``checks``: a function fit name → reload result (True,
    False or None for no check file), written as scripts/latent_usage.py writes usage.json."""
    rng = np.random.default_rng(7)
    runs, docs = tmp / "runs", tmp / "docs"
    runs.mkdir(); docs.mkdir(); (tmp / "checks").mkdir()
    bias = draw_bias * (2.0 * (np.arange(N + 20) % 2) if hetero else 1.0)
    for fold in FOLDS:
        ids = np.array([f"u{fold}_{i}" for i in range(N + 20)])
        fy = np.arange(N + 20) < N
        ent = np.arange(N + 20) < ENTRANTS
        hist = np.arange(N + 20) < ENTRANTS + 20
        base = 45 + rng.normal(0, 3, N + 20)
        for j, seed in enumerate(SEEDS):
            a = base + rng.normal(0, 0.3, N + 20)
            ref = pd.DataFrame({"unit_id": ids, "player_id": np.arange(N + 20) + 10 * fold, "first_year": fy.astype(int), "intl_entrant": (ent & fy).astype(int),
                                "first_year_with_history": (hist & fy).astype(int), "first_year_without_history": (~hist & fy).astype(int),
                                "with_raw": a, "without_raw": a + 0.05, "with_point": a - 0.15, "without_point": a - 0.1})
            ref.to_csv(docs / f"B2_{fold}{'' if j == 0 else f'_s{j + 1}'}.per_unit.csv", index=False)
            for arm in ("later", "strict"):
                d = runs / f"lc_{arm}_{fold}_{seed}"
                d.mkdir()
                noise = lambda: rng.normal(0, 0.02, N + 20)                    # noqa: E731
                c = a + penalty - np.where(ent, entrant_shift + penalty, 0.0) + (0.05 if arm == "strict" else 0.0)
                pu = ref[["unit_id", "player_id", "first_year", "intl_entrant", "first_year_with_history", "first_year_without_history"]].copy()
                pu["team_id"], pu["season"] = np.arange(N + 20) % 37, fold + 1
                pu["nll"], pu["nll_replicate"] = c + bias + noise(), c + bias + noise()
                pu["nll_opportunity"], pu["nll_without_makes"] = 0.4 * pu["nll"], 0.9 * pu["nll"]
                pu["nll_draws_256"] = np.where(ent & fy, c + bias / 4 + noise() / 2, np.nan)
                pu = pu.sample(frac=1.0, random_state=3)                       # the join must not depend on row order
                if edit:
                    edit(pu, ref, d, arm, fold, j)
                pu.to_csv(d / f"per_unit_{fold + 1}.csv", index=False)
                curve = [{"epoch": e + 1, "stop_nll": 45.4 - 0.03 * e, "train_loss": 97 - 0.2 * e, "mean_kl_per_career": 2.0 + 0.05 * e} for e in range(20)]
                ev = {"units": N + 20, "nonfinite": 0, "nll_all": 45.0, "nll_first_year": 45.0, "nll_intl_entrants": 45.0, "draws": 64}
                run = {"arm": arm, "fold": fold, "seed": seed, "code": {"x": "abc"}, "config": {**LCMP.REGISTERED_CONFIG, "tables_manifest": {"payload_sha256": "t"}},
                       "evaluation": {str(fold): ev, str(fold + 1): ev},
                       "training": {"curve": curve, "best_stop_nll": curve[-1]["stop_nll"], "epochs_run": 20, "steps": 6700, "wall_seconds": 17000.0, "budget_exhausted": None},
                       "latent": {"sigma": [0.3] * 8, "state_map_column_norms": [0.5] * 8}}
                (d / "run.json").write_text(json.dumps(run))
                ok = checks(d.name) if checks else None
                if ok is not None:
                    (tmp / "checks" / d.name).mkdir()
                    gap = 2e-7 if ok else 0.3
                    (tmp / "checks" / d.name / "usage.json").write_text(json.dumps({
                        "run": d.name, "tolerance": 1e-5, "reproduction_gaps": {"joint": gap, "opportunity": gap, "makes_removed": gap},
                        "reload": {"artifact_hash_matches_run_json": True, "static_vocabularies_match": True, "saved_scores_reproduced": bool(ok)}}))
                if edit:
                    edit(None, None, d, arm, fold, j)
    return ["--runs", str(runs), "--folds", *map(str, FOLDS), "--seeds", *map(str, SEEDS), "--b2", str(docs / "B2_2022"), str(docs / "B2_2023"),
            "--exports", "none", "none", "--split-labels", "2022–23", "2023–24", "--skip-calibration", "--checks", str(tmp / "checks"),
            "--out", str(tmp / "report")]


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["latent_compare.py", *argv])
    LCMP.main()
    out = Path(argv[argv.index("--out") + 1])
    return json.loads(out.with_suffix(".json").read_text()), out.with_suffix(".md").read_text()


def test_registered_design_gives_a_verdict_on_matching_readings(tmp_path, monkeypatch):
    rec, md = _run(_fixture(tmp_path, entrant_shift=0.3), monkeypatch)
    rule = rec["rule"]
    assert rec["complete"] and rec["registered_design"] and rule["stable_across_draws"] and rule["score_conditions_met"] is False
    assert rule["conditions"]["registered"]["entrant_positive_both"] and rule["conditions"]["registered"]["entrant_aggregate_above_zero"]
    assert not rule["conditions"]["registered"]["no_history_noninferior"]            # C is 0.1 worse without history: the guard fails
    assert "Score conditions: not met" in md and "384 (all saved draws pooled)" in md
    # no reload check has been written: reproduction is pending and nothing is eligible
    assert rec["checkpoint_reproduction"]["status"] == "pending" and "Checkpoint reproduction: pending" in md
    assert rec["candidate_requirements"] == {"score_conditions_met": False, "checkpoint_reproduction": "pending", "eligible_on_scores_and_reproduction": False}


def test_a_loss_on_entrants_is_reported_as_not_met(tmp_path, monkeypatch):
    rec, _ = _run(_fixture(tmp_path, entrant_shift=-0.3), monkeypatch)
    assert rec["rule"]["stable_across_draws"] and rec["rule"]["score_conditions_met"] is False
    assert not rec["rule"]["conditions"]["registered"]["entrant_positive_both"]


def test_draw_dependent_answer_withholds_the_verdict(tmp_path, monkeypatch):
    # entrants: true gain +0.03; the 64-draw scores carry +0.04, the 256-draw scores +0.01, so the sign changes up the ladder
    rec, md = _run(_fixture(tmp_path, entrant_shift=0.03, draw_bias=0.04), monkeypatch)
    rule = rec["rule"]
    assert not rule["conditions"]["registered"]["entrant_positive_both"] and rule["conditions"]["highest"]["entrant_positive_both"]
    assert rule["stable_across_draws"] is False and rule["score_conditions_met"] is None and "Numerically unsettled" in md
    ladder = rec["draw_ladder"]["intl_entrant"]
    means = [lv["aggregate"]["mean"] for lv in ladder["levels"]]
    assert [lv["draws"] for lv in ladder["levels"]] == [64, 64, 128, 256, 384] and means[3] > means[4] > means[2] and means[2] < 0 < means[4]
    # the extrapolation is the stated formula applied to the reported means
    want = means[4] + (means[4] - 0.5 * (means[0] + means[1])) / (384 / 64 - 1)
    assert abs(ladder["extrapolated"]["aggregate"]["mean"] - want) < 1e-12
    other = rec["draw_ladder"]["first_year"]
    assert [lv["draws"] for lv in other["levels"]] == [64, 64, 128]


def test_first_order_extrapolation_recovers_the_mixture_score_on_real_draws():
    # log densities of independent draws: the K-draw estimate of −log E[p] is biased upward by about c/K
    rng = np.random.default_rng(11)
    sd, units = 1.0, 20000
    lp = rng.normal(-45.0 - sd ** 2 / 2, sd, size=(384, units))             # E[exp(lp)] = exp(−45)
    est = lambda x: -(np.logaddexp.reduce(x, axis=0) - math.log(len(x)))      # noqa: E731
    a, b, c = est(lp[:64]), est(lp[64:128]), est(lp[128:])
    hi = LCMP.pooled([a, b, c], [64, 64, 256])
    bias64, bias_hi = 0.5 * (a.mean() + b.mean()) - 45.0, hi.mean() - 45.0
    assert bias64 > 4 * bias_hi > 0                                           # the bias falls roughly as 1/K
    corrected = hi.mean() - (0.5 * (a.mean() + b.mean()) - hi.mean()) / (384 / 64 - 1)
    assert abs(corrected - 45.0) < 0.2 * bias_hi + 1e-3


@pytest.mark.parametrize("target", ["baseline", "replicate", "entrant_256", "component", "run_json"])
def test_non_finite_inputs_stop_the_report(tmp_path, monkeypatch, target):
    def edit(pu, ref, d, arm, fold, j):
        if (arm, fold, j) != ("later", 2023, 1):
            return
        if pu is not None:
            if target == "replicate":
                pu.iloc[5, pu.columns.get_loc("nll_replicate")] = np.nan
            if target == "component":
                pu.iloc[9, pu.columns.get_loc("nll_without_makes")] = np.inf
            if target == "entrant_256":
                row = pu.index[pu["intl_entrant"] == 1][3]
                pu.loc[row, "nll_draws_256"] = np.nan
        elif target == "run_json":
            run = json.loads((d / "run.json").read_text())
            run["evaluation"]["2024"]["nonfinite"] = 2
            (d / "run.json").write_text(json.dumps(run))
    argv = _fixture(tmp_path, entrant_shift=0.3, edit=edit)
    if target == "baseline":
        f = tmp_path / "docs" / "B2_2022_s2.per_unit.csv"
        t = pd.read_csv(f)
        t.loc[17, "with_point"] = np.nan
        t.to_csv(f, index=False)
    with pytest.raises(SystemExit) as err:
        _run(argv, monkeypatch)
    assert "non-finite" in str(err.value)
    assert not (tmp_path / "report.md").exists()


def test_a_nan_gap_fails_the_tolerance_check_and_cohort_or_config_drift_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        LCMP.within(float("nan"), 1e-3, "rebuild")
    with pytest.raises(SystemExit):
        LCMP.within(0.5, 1e-3, "rebuild")
    assert LCMP.within(1e-6, 1e-3, "rebuild") == 1e-6

    def edit(pu, ref, d, arm, fold, j):
        if pu is None and (arm, fold, j) == ("strict", 2022, 0):
            run = json.loads((d / "run.json").read_text())
            run["config"]["eval_draws"] = 32
            (d / "run.json").write_text(json.dumps(run))
    with pytest.raises(SystemExit) as err:
        _run(_fixture(tmp_path, edit=edit), monkeypatch)
    assert "registered" in str(err.value)


def test_pooled_draws_and_subset_preview(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    lp = rng.normal(-45, 2, size=(384, 5))                                   # log densities of 384 draws for 5 units
    est = lambda x: -(np.logaddexp.reduce(x, axis=0) - math.log(len(x)))      # noqa: E731
    got = LCMP.pooled([est(lp[:64]), est(lp[64:128]), est(lp[128:])], [64, 64, 256])
    assert np.allclose(got, est(lp))
    argv = _fixture(tmp_path, entrant_shift=0.3)
    argv[argv.index("--seeds") + 1:argv.index("--seeds") + 3] = [str(SEEDS[0])]
    rec, md = _run(argv, monkeypatch)
    assert rec["complete"] and not rec["registered_design"] and "rule" not in rec and "Subset preview" in md


def _entrant_parts(tmp: Path, fold: int):
    """The entrants' per-player mixture gains of one fold at the registered draws, the replicate and all draws pooled."""
    refs = [pd.read_csv(tmp / "docs" / f"B2_{fold}{'' if j == 0 else f'_s{j + 1}'}.per_unit.csv") for j in range(2)]
    ids = refs[0].loc[(refs[0]["first_year"] == 1) & (refs[0]["intl_entrant"] == 1), "unit_id"]
    a = LCMP.ens(np.stack([r.set_index("unit_id").loc[ids, "with_raw"].to_numpy() for r in refs]))
    pus = [pd.read_csv(tmp / "runs" / f"lc_later_{fold}_{seed}" / f"per_unit_{fold + 1}.csv").set_index("unit_id").loc[ids] for seed in SEEDS]
    col = lambda name: np.stack([p[name].to_numpy() for p in pus])           # noqa: E731
    hi = np.stack([LCMP.pooled([p["nll"].to_numpy(), p["nll_replicate"].to_numpy(), p["nll_draws_256"].to_numpy()], [64, 64, 256]) for p in pus])
    return a - LCMP.ens(col("nll")), a - LCMP.ens(col("nll_replicate")), a - LCMP.ens(hi), pus[0]["player_id"].to_numpy()


def test_extrapolated_interval_is_recomputed_from_per_player_gains(tmp_path, monkeypatch):
    # the finite-draw bias sits on every other entrant: the correction varies across players, so shifting the 384-draw interval
    # by the mean correction is not the interval of the corrected gains
    rec, md = _run(_fixture(tmp_path, entrant_shift=0.5, draw_bias=0.3, hetero=True), monkeypatch)
    ladder = rec["draw_ladder"]["intl_entrant"]
    hi, ex = ladder["levels"][-1], ladder["extrapolated"]
    parts = []
    for i, fold in enumerate(FOLDS):
        g0, g1, gh, players = _entrant_parts(tmp_path, fold)
        want = LCMP.clustered(gh + (gh - 0.5 * (g0 + g1)) / (384 / 64 - 1), players)
        got = ex["splits"][i]
        assert all(abs(got[k] - want[k]) < 1e-9 for k in ("mean", "se", "lo", "hi"))
        shifted_lo = hi["splits"][i]["lo"] + (got["mean"] - hi["splits"][i]["mean"])
        assert abs(got["lo"] - shifted_lo) > 1e-3 and abs(got["se"] - hi["splits"][i]["se"]) > 1e-3
        parts.append(want)
    agg = LCMP.equal_weight(parts)
    assert all(abs(ex["aggregate"][k] - agg[k]) < 1e-9 for k in ("mean", "lo", "hi"))
    assert "(per player)" in md
    # Monte Carlo standard errors: first-order variance factors of the independent blocks
    assert abs(LCMP.mc_factor([1 / 6, 1 / 6, 2 / 3], [64, 64, 256]) - 1 / 6) < 1e-12 and abs(LCMP.mc_factor([0.1, 0.1, 0.8], [64, 64, 256]) - 0.18) < 1e-12
    base = ladder["levels"][0]["mc_se"]
    assert base > 0 and abs(ladder["levels"][2]["mc_se"] / base - math.sqrt(0.5)) < 1e-9
    assert abs(hi["mc_se"] / base - math.sqrt(1 / 6)) < 1e-9 and abs(ex["mc_se"] / base - math.sqrt(0.18)) < 1e-9
    other = rec["draw_ladder"]["first_year"]
    assert abs(other["extrapolated"]["mc_se"] / other["levels"][0]["mc_se"] - math.sqrt(0.5)) < 1e-9


def test_checkpoint_reproduction_is_an_adoption_requirement(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir(); (tmp_path / "c").mkdir()
    kw = dict(entrant_shift=0.3, penalty=0.0)
    rec, md = _run(_fixture(tmp_path / "a", checks=lambda name: True, **kw), monkeypatch)
    assert rec["rule"]["score_conditions_met"] is True and rec["checkpoint_reproduction"]["status"] == "met"
    assert rec["candidate_requirements"]["eligible_on_scores_and_reproduction"] is True and "Checkpoint reproduction: met" in md
    # one checkpoint does not reproduce: the scores still say met, the candidate is not eligible
    rec, md = _run(_fixture(tmp_path / "b", checks=lambda name: name != "lc_later_2023_20260902", **kw), monkeypatch)
    assert rec["rule"]["score_conditions_met"] is True and rec["checkpoint_reproduction"]["status"] == "not met"
    assert rec["checkpoint_reproduction"]["fits"]["lc_later_2023_20260902"]["failed"] == ["saved_scores_reproduced"]
    assert rec["candidate_requirements"]["eligible_on_scores_and_reproduction"] is False and "cannot be adopted" in md
    # a check that is missing leaves the requirement pending, never met
    rec, md = _run(_fixture(tmp_path / "c", checks=lambda name: None if name == "lc_strict_2022_20260901" else True, **kw), monkeypatch)
    assert rec["checkpoint_reproduction"]["status"] == "pending" and rec["candidate_requirements"]["eligible_on_scores_and_reproduction"] is False


def test_mixture_conditional_reweights_the_components_by_the_conditioning_information():
    import torch
    from scipy.stats import betabinom, poisson
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_diagnostics as CD
    rng = np.random.default_rng(5)
    n = 40000
    lam, ab = (5.0, 30.0), ((2.0, 8.0), (8.0, 2.0))          # two components: few attempts and poor shooting, many and good
    r = rng.integers(0, 2, n)
    a = np.maximum(poisson.rvs(np.where(r == 0, lam[0], lam[1]), random_state=rng), 1)
    k = betabinom.rvs(a, np.where(r == 0, ab[0][0], ab[1][0]), np.where(r == 0, ab[0][1], ab[1][1]), random_state=rng)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)       # noqa: E731
    grids = [CD.bb_grid(t(a), t(np.full(n, al)), t(np.full(n, be)), 0) for al, be in ab]
    # attempts were floored at one: the weight of a component is its probability of the observed (floored) attempts
    pa = lambda l: np.where(a == 1, poisson.cdf(1, l), poisson.pmf(a, l))    # noqa: E731
    logw = t(np.log(np.stack([pa(lam[0]), pa(lam[1])])))
    weighted, equal = CD.mixture_logpmf(grids, logw), CD.mixture_logpmf(grids)
    # the weighted mixture is the analytic conditional; the equal-weight one is the plain average, as before
    w0 = pa(lam[0]) / (pa(lam[0]) + pa(lam[1]))
    direct = w0[:, None] * grids[0].exp().numpy() + (1 - w0)[:, None] * grids[1].exp().numpy()
    assert np.allclose(weighted.exp().numpy(), direct, atol=1e-12)
    assert torch.equal(equal, torch.log((torch.nan_to_num(grids[0].exp()) + torch.nan_to_num(grids[1].exp())) / 2))
    ones = np.ones(n, dtype=bool)
    good = CD.summarize(CD.pit_discrete(weighted, torch.as_tensor(k), 0, rng), ones)
    poor = CD.summarize(CD.pit_discrete(equal, torch.as_tensor(k), 0, rng), ones)
    assert good["ks"] < 0.012 and abs(good["cov90"] - 0.90) < 0.01          # calibrated under the mixture's own conditional
    assert poor["ks"] > 0.03 and poor["ks"] > 3 * good["ks"]                # the fixed-exposure average is not that conditional
    with pytest.raises(ValueError):
        CD.mixture_logpmf(grids, torch.full((2, n), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError):
        CD.mixture_logpmf(grids, logw[:1])


def test_the_two_later_career_gains_are_compared_on_the_same_players(tmp_path, monkeypatch):
    rec, md = _run(_fixture(tmp_path, entrant_shift=0.3), monkeypatch)
    assert "Difference between the architectures" in md
    for i, fold in enumerate(FOLDS):
        refs = [pd.read_csv(tmp_path / "docs" / f"B2_{fold}{'' if j == 0 else f'_s{j + 1}'}.per_unit.csv") for j in range(2)]
        ids = refs[0].loc[(refs[0]["first_year"] == 1) & (refs[0]["intl_entrant"] == 1), "unit_id"]
        a = np.mean([(r.set_index("unit_id").loc[ids, "without_raw"] - r.set_index("unit_id").loc[ids, "with_raw"]).to_numpy() for r in refs], axis=0)
        col = lambda arm, seed: pd.read_csv(tmp_path / "runs" / f"lc_{arm}_{fold}_{seed}" / f"per_unit_{fold + 1}.csv").set_index("unit_id").loc[ids, "nll"].to_numpy()   # noqa: E731
        c = np.mean([col("strict", seed) - col("later", seed) for seed in SEEDS], axis=0)
        got = rec["tables"]["later-career gain, A minus C|seed average|intl_entrant"]["splits"][i]
        want = LCMP.clustered(a - c, refs[0].set_index("unit_id").loc[ids, "player_id"].to_numpy())
        assert all(abs(got[k] - want[k]) < 1e-9 for k in ("mean", "lo", "hi"))
    agg = rec["tables"]["later-career gain, A minus C|mixture|first_year"]["aggregate"]
    assert math.isfinite(agg["mean"]) and agg["lo"] < agg["mean"] < agg["hi"]


def test_equal_season_aggregate_clusters_players_across_seasons():
    """scripts/latent_all_units.py: returning players appear in both evaluation seasons, so the equal-season aggregate
    clusters by player across seasons; with disjoint players it equals the independence formula."""
    sys.path.insert(0, str(ROOT / "scripts"))
    spec_all = importlib.util.spec_from_file_location("latent_all_units", ROOT / "scripts" / "latent_all_units.py")
    mod = importlib.util.module_from_spec(spec_all); spec_all.loader.exec_module(mod)
    from bayes_two_split import clustered, equal_weight
    rng = np.random.default_rng(3)
    g1, g2 = rng.normal(0.3, 1.0, 400), rng.normal(0.1, 1.0, 500)
    p1, p2 = np.arange(400), 1000 + np.arange(500)
    joint, indep = mod.joint_equal_season([(g1, p1), (g2, p2)]), equal_weight([clustered(g1, p1), clustered(g2, p2)])
    assert joint["players_in_every_season"] == 0 and joint["clusters"] == 900
    assert abs(joint["mean"] - indep["mean"]) < 1e-12 and abs(joint["se"] / indep["se"] - 1) < 5e-3
    # the same players with the same gains in both seasons: the two means move together, so the aggregate is as
    # uncertain as one season, where the independence formula would claim 1/sqrt(2) of that
    one = clustered(g1, p1)
    same = mod.joint_equal_season([(g1, p1), (g1, p1)])
    assert same["players_in_every_season"] == 400 and abs(same["se"] / one["se"] - 1) < 5e-3
    assert same["se"] > 1.3 * equal_weight([one, one])["se"]
    # opposite gains for the same players cancel exactly
    opposite = mod.joint_equal_season([(g1, p1), (-g1, p1)])
    assert abs(opposite["mean"]) < 1e-12 and opposite["se"] < 1e-12
    # several units of one player within a season stay one cluster
    multi = mod.joint_equal_season([(np.r_[g1, g1], np.r_[p1, p1]), (g2, p2)])
    assert multi["clusters"] == 900
