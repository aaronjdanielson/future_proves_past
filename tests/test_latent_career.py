"""Latent-career model (D-115): the inference algebra and the information rules that need no data tables.

The checks on the real tables (post-cutoff rows, the strict exclusion, aggregation, reload) are run by
``scripts/latent_career.py validate`` and reported in ``docs/LATENT_VALIDATION.md``.
"""
import math

import numpy as np
import torch

from fpp.model.latent_career import DT, LatentCareerModel, age_basis, draw, kalman_filter, rts_smoother


def _chain(B=3, T=4, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=DT, generator=g)        # noqa: E731
    A0 = r(B, d, d)
    return {"m0": r(B, d), "P0": A0 @ A0.transpose(-1, -2) + torch.eye(d, dtype=DT), "F": torch.eye(d, dtype=DT) + 0.2 * r(B, T, d, d),
            "b": 0.3 * r(B, T, d), "Q": torch.diag_embed(0.2 + torch.rand(B, T, d, dtype=DT, generator=g)),
            "J": 2 * torch.rand(B, T, d, dtype=DT, generator=g), "h": r(B, T, d)}


def test_filter_smoother_and_normalizer_match_gaussian_conditioning():
    c = _chain()
    c["J"][:, 2], c["h"][:, 2] = 0.0, 0.0                       # a period without spells
    mp, Pp, mf, Pf, logz = kalman_filter(**c)
    ms, Ps = rts_smoother(mp, Pp, mf, Pf, c["F"])
    B, T, d = c["J"].shape
    for i in range(B):
        mean, cov = [c["m0"][i]], torch.zeros(T * d, T * d, dtype=DT)
        cov[:d, :d] = c["P0"][i]
        for t in range(1, T):
            mean.append(c["F"][i, t] @ mean[-1] + c["b"][i, t])
            prev, cur = slice((t - 1) * d, t * d), slice(t * d, (t + 1) * d)
            cov[cur, :t * d] = c["F"][i, t] @ cov[prev, :t * d]
            cov[:t * d, cur] = cov[cur, :t * d].T
            cov[cur, cur] = c["F"][i, t] @ cov[prev, prev] @ c["F"][i, t].T + c["Q"][i, t]
        mean = torch.cat(mean)
        Jf, hf = torch.diag(c["J"][i].reshape(-1)), c["h"][i].reshape(-1)
        prec = torch.linalg.inv(cov) + Jf
        post_cov = torch.linalg.inv(prec)
        post_mean = post_cov @ (torch.linalg.solve(cov, mean) + hf)
        assert torch.allclose(post_mean.view(T, d), ms[i], atol=1e-8)
        for t in range(T):
            assert torch.allclose(post_cov[t * d:(t + 1) * d, t * d:(t + 1) * d], Ps[i, t], atol=1e-8)
        ref = (-0.5 * torch.linalg.slogdet(torch.eye(T * d, dtype=DT) + cov @ Jf)[1] + 0.5 * post_mean @ prec @ post_mean
               - 0.5 * mean @ torch.linalg.solve(cov, mean))
        assert torch.allclose(ref, logz[i], atol=1e-8)


def test_filter_ignores_later_potentials_and_padding_carries_the_state():
    c = _chain()
    _, _, mf, Pf, _ = kalman_filter(**c)
    later = {**c, "J": c["J"].clone(), "h": c["h"].clone()}
    later["J"][:, 3] += 5.0
    later["h"][:, 3] -= 2.0
    _, _, mf2, Pf2, _ = kalman_filter(**later)
    assert torch.equal(mf[:, :3], mf2[:, :3]) and torch.equal(Pf[:, :3], Pf2[:, :3])
    # two padded periods (identity transition, no noise, no spells) leave the last state and the normalizer unchanged
    B, T, d = c["J"].shape
    pad = lambda x, fill: torch.cat([x, fill.expand(B, 2, *x.shape[2:])], dim=1)        # noqa: E731
    padded = {"m0": c["m0"], "P0": c["P0"], "F": pad(c["F"], torch.eye(d, dtype=DT)), "b": pad(c["b"], torch.zeros(d, dtype=DT)),
              "Q": pad(c["Q"], torch.zeros(d, d, dtype=DT)), "J": pad(c["J"], torch.zeros(d, dtype=DT)), "h": pad(c["h"], torch.zeros(d, dtype=DT))}
    mp3, Pp3, mf3, Pf3, logz3 = kalman_filter(**padded)
    _, _, _, _, logz = kalman_filter(**c)
    assert torch.allclose(mf3[:, -1], mf[:, -1]) and torch.allclose(Pf3[:, -1], Pf[:, -1]) and torch.allclose(logz3, logz)
    ms3, _ = rts_smoother(mp3, Pp3, mf3, Pf3, padded["F"])
    ms, _ = rts_smoother(*kalman_filter(**c)[:4], c["F"])
    assert torch.allclose(ms3[:, :T], ms, atol=1e-6)


def _model(seed=1):
    torch.manual_seed(seed)
    m = LatentCareerModel(n_features=38, n_competitions=5, n_static=4, static_vocab_sizes=[3, 2], n_covariates=8)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape))
    return m.eval()


def _seq(B=4, T=3, n=5, seed=2):
    g = torch.Generator().manual_seed(seed)
    return {"cov": torch.randn(B, 8, generator=g), "dt": torch.tensor([[0.0, 1.0, 2.0]] * B, dtype=DT), "age_prev": torch.full((B, T), 19.0, dtype=DT),
            "obs_seq": torch.tensor([0, 0, 1, 2, 2]), "obs_period": torch.tensor([0, 1, 0, 1, 1]),
            "obs_feat": torch.randn(n, 38, generator=g), "obs_comp": torch.tensor([1, 2, 0, 3, 5])}


def test_kl_is_nonnegative_and_zero_without_spells():
    m = _model()
    with torch.no_grad():
        _, _, kl = m.infer(_seq(), smooth=True)
        empty = {**_seq(), "obs_seq": torch.zeros(0, dtype=torch.long), "obs_period": torch.zeros(0, dtype=torch.long),
                 "obs_feat": torch.zeros(0, 38), "obs_comp": torch.zeros(0, dtype=torch.long)}
        ms, Ps, kl0 = m.infer(empty, smooth=True)
        m0, P0 = m.initial(empty["cov"])
    assert bool((kl > -1e-9).all()) and float(kl[3]) < 1e-9           # sequence 3 has no spells
    assert torch.allclose(kl0, torch.zeros_like(kl0), atol=1e-9)
    assert torch.allclose(ms[:, 0], m0) and torch.allclose(Ps[:, 0], P0)


def test_unseen_competition_has_zero_deviation_and_missing_age_is_flagged():
    m = LatentCareerModel(n_features=38, n_competitions=5, n_static=4, static_vocab_sizes=[3, 2], n_covariates=8)   # as constructed
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    z = torch.randn(4, m.latent, dtype=DT)
    counts = torch.ones(4, 13, dtype=torch.long)
    loss = -m.emission_log_prob(z, torch.tensor([0, 0, 1, 2]), torch.tensor([1, 1, 2, 3]), torch.ones(4, dtype=DT), counts, torch.ones(4, dtype=torch.bool)).sum()
    loss.backward()
    opt.step()                                                  # the unseen row receives no gradient, so it stays at zero
    with torch.no_grad():
        assert float(m.comp_effect.weight[0].abs().sum()) == 0.0 and float(m.comp_embed.weight[0].abs().sum()) == 0.0
        assert float(m.comp_effect.weight[1].abs().sum()) > 0.0
    basis = age_basis(torch.tensor([20.0, float("nan"), 40.0], dtype=DT))
    assert basis[0].tolist() == [1.0, 0.0, 0.0, 0.0] and basis[1].tolist() == [1.0, 0.0, 0.0, 1.0] and float(basis[2, 1]) == 2.5


def test_predictive_mixture_is_logsumexp_not_a_mean_of_logs():
    lp = torch.tensor([[-50.0, -40.0], [-45.0, -60.0], [-70.0, -41.0]], dtype=DT)
    mix = torch.logsumexp(lp, dim=0) - math.log(3)
    assert bool((mix > lp.mean(dim=0)).all())
    assert np.allclose(mix.numpy(), np.log(np.exp(lp.numpy()).mean(axis=0)))


def test_draws_reproduce_the_requested_moments():
    g = torch.Generator().manual_seed(5)
    A = torch.randn(3, 3, dtype=DT, generator=g)
    P, mean = A @ A.T + torch.eye(3, dtype=DT), torch.tensor([1.0, -2.0, 0.5], dtype=DT)
    z = draw(mean, P, torch.randn(200000, 3, dtype=DT, generator=g))
    assert torch.allclose(z.mean(0), mean, atol=0.02) and torch.allclose(torch.cov(z.T), P, atol=0.05)
