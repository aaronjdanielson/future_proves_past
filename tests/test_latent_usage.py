"""The frozen-checkpoint usage diagnostic (scripts/latent_usage.py) on a fabricated model and fabricated units: the
quantities are finite, respect their bounds, and report exactly zero when the decoder ignores the state."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fpp.experiments import latent as L
from fpp.model.latent_career import DT, LatentCareerModel
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latent_usage", ROOT / "scripts" / "latent_usage.py")
LU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LU)


class FakeSource:
    strength_mu, strength_sd = np.zeros(4), np.ones(4)
    features = L.SpellSource.features


def _setup(n=6, seed=4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = LatentCareerModel(n_features=L.N_FEATURES, n_competitions=5, n_static=4, static_vocab_sizes=[3, 2], n_covariates=L.N_COVARIATES)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape))
    model.eval()
    counts = np.full((n, 13), 12, dtype=np.int64)
    counts[:, L.LC.MAKE_INDEX] = 5
    frame = pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "player_id": np.arange(n), "season": 2024, "age_at_cutoff": 19.0 + 0.3 * np.arange(n)})
    ua = L.UnitArrays(frame, rng.normal(size=(n, 4)).astype(np.float32), np.ones((n, 4), dtype=np.float32), rng.integers(0, 3, size=(n, 2)).astype(np.int64) % 2,
                      rng.normal(size=(n, 4)).astype(np.float32),
                      {"games": np.full(n, 10), "starts": np.full(n, 4), "minutes": np.full(n, 250.0), "counts": counts, "overtime": np.zeros(n),
                       "overtime_known": np.ones(n, dtype=bool), "schedule": np.full(n, 30)}, np.ones(n))
    src, seqs = FakeSource(), []
    for i in range(n):
        k = 1 + i % 3
        c = rng.integers(3, 30, size=(k, 13)).astype(float)
        c[:, L.LC.MAKE_INDEX] = np.minimum(c[:, L.LC.MAKE_INDEX], c[:, L.LC.ATTEMPT_INDEX])
        sp = {"season": np.sort(rng.integers(2021, 2024, size=k)), "comp": rng.integers(0, 6, size=k), "type": rng.integers(1, 5, size=k),
              "minutes": rng.uniform(50, 600, size=k), "games": rng.integers(3, 30, size=k).astype(float), "starts": np.zeros(k), "starts_known": np.zeros(k), "counts": c}
        for name in ("team_adj_o", "team_adj_d", "opp_adj_o", "opp_adj_d"):
            sp[name], sp[name + "_n"] = np.zeros(k), np.zeros(k)
        seqs.append(L._sequence(src, sp, target_season=2024, age_at_target=float(frame["age_at_cutoff"][i]), hw=ua.hw[i], unit=i))
    return model, DifferentiableSeasonLikelihood(None, quadrature_nodes=16), src, ua, seqs


def test_usage_quantities_are_finite_bounded_and_reproducible():
    model, lik, src, ua, seqs = _setup()
    idx = np.arange(len(seqs))
    f1, parts = LU.usage_arrays(model, lik, src, ua, seqs, idx, draws=8, seed=3, chunk=4)
    f2, _ = LU.usage_arrays(model, lik, src, ua, seqs, idx, draws=8, seed=3, chunk=4)
    assert np.isfinite(f1.to_numpy()).all() and np.array_equal(f1.to_numpy(), f2.to_numpy())
    assert (f1["kl_prior"] > 0).all() and (f1["kl_no_history"] > 0).all() and (f1["variance_ratio"] < 1).all() and (f1["variance_ratio"] > 0).all()
    assert (f1["nll_with"] != f1["nll_without"]).all() and (f1["state_shift"] > 0).all()
    assert parts["context"].shape == (len(seqs), 32) and {"abs_log_change_rates", "abs_log_change_makes_alpha"} <= set(f1.columns)


def test_a_decoder_that_ignores_the_state_shows_no_sensitivity_to_history():
    model, lik, src, ua, seqs = _setup()
    with torch.no_grad():
        model.state_map.weight.zero_()
    f, _ = LU.usage_arrays(model, lik, src, ua, seqs, np.arange(len(seqs)), draws=8, seed=3)
    assert np.allclose(f["nll_with"], f["nll_without"], atol=1e-10) and float(f["state_shift"].abs().max()) == 0.0
    assert float(f["abs_log_change_rates"].abs().max()) == 0.0 and (f["kl_prior"] > 0).all()      # the state still moves; the forecast does not


def test_gaussian_kl_matches_the_closed_form_and_reload_restores_the_model():
    m1, m0 = torch.tensor([[0.5, -1.0]], dtype=DT), torch.zeros(1, 2, dtype=DT)
    P1, P0 = torch.diag(torch.tensor([0.5, 2.0], dtype=DT)).unsqueeze(0), torch.eye(2, dtype=DT).unsqueeze(0)
    want = 0.5 * (0.5 + 2.0 + 0.25 + 1.0 - 2 - np.log(0.5 * 2.0))
    assert abs(float(LU.gaussian_kl(m1, P1, m0, P0)) - want) < 1e-9 and abs(float(LU.gaussian_kl(m1, P1, m1, P1))) < 1e-8
    model, lik, src, ua, seqs = _setup()
    clone = LU.load_model({"architecture": model.architecture, "state_dict": model.state_dict()})
    a = L.score_units(model, lik, seqs, ua, draws=8, seed=1)
    assert np.array_equal(a, L.score_units(clone, lik, seqs, ua, draws=8, seed=1))


def test_plug_in_forecast_is_one_parameter_set_and_the_mixture_differs_from_it():
    model, lik, src, ua, seqs = _setup()
    idx = np.arange(len(seqs))
    plug = LU.plug_in_nll(model, lik, ua, seqs, idx, chunk=4)
    assert np.isfinite(plug).all() and np.allclose(plug, LU.plug_in_nll(model, lik, ua, seqs, idx), atol=1e-5)     # batching moves the float32 decoder by ~1e-7
    mix = L.score_units(model, lik, seqs, ua, draws=32, seed=2)
    assert not np.allclose(plug, mix)
    with torch.no_grad():                                           # a decoder that ignores the state: nothing to integrate over
        model.state_map.weight.zero_()
    assert np.allclose(LU.plug_in_nll(model, lik, ua, seqs, idx), L.score_units(model, lik, seqs, ua, draws=8, seed=2), atol=1e-9)
