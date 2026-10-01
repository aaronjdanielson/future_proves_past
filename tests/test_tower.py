"""Reference tower: initialization at the reference, order and padding invariance, empty histories."""
import unittest

import numpy as np
import torch

from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
from fpp.model.tower import ReferenceTower

VOCAB = [5, 4, 6, 4, 3, 4]
STATIC_VOCAB = [4, 6, 4, 5]


def _channel(rng, T, present=True, seasons=(2022, 2023)):
    Fn = len(F.FIELD_NAMES)
    if not present or T == 0:
        return F.Channel.empty(Fn)
    x = rng.normal(size=(T, Fn)).astype(np.float32)
    mask = rng.random((T, Fn)) > 0.3
    x = np.where(mask, x, 0.0).astype(np.float32)
    cats = rng.integers(0, 3, size=(T, len(F.CATEGORICAL))).astype(np.int32)
    dates = np.sort(rng.uniform(-3.0, -0.1, size=T))
    season = np.where(dates < -1.0, seasons[0], seasons[1]).astype(np.int32)
    recency = (dates.max() - dates).astype(np.float32)
    clocks = np.stack([dates, np.sign(dates) * np.log1p(np.abs(dates)), np.log1p(recency),
                       np.log1p(np.r_[0, np.diff(dates)]), np.r_[1, season[1:] != season[:-1]], np.r_[1, np.zeros(T - 1)]], axis=1).astype(np.float32)
    weight = (rng.uniform(1, 30, size=T) + 1).astype(np.float32)
    ev = rng.normal(size=len(F.EVIDENCE_NAMES)).astype(np.float32)
    ch = F.Channel(x, mask, cats, clocks, weight, recency, season, np.arange(T, dtype=np.int32), ev, present)
    return ch if present else F.Channel.empty(Fn)


def _example(rng, T_intl, T_ncaa, uid="u"):
    static = rng.normal(size=len(F.STATIC_CONTINUOUS)).astype(np.float32)
    t_v, t_m = F.time_basis(-0.4, 19.5, 19.9)
    targets = {"games": 20, "starts": 5, "minutes": 400.0, "counts": np.array([80, 40, 30, 10, 40, 30, 20, 60, 30, 10, 5, 25, 40]),
               "overtime": 1.0, "overtime_known": True, "schedule": 31}
    return F.Example(uid, "-", (), _channel(rng, T_intl, T_intl > 0), _channel(rng, T_ncaa, T_ncaa > 0), static, static > -9,
                     rng.integers(0, 3, size=len(F.STATIC_CATEGORICAL)).astype(np.int32), t_v, t_m, targets)


def _permute_channel(ch: dict, perm: torch.Tensor, row: int) -> None:
    for key in ("x", "mask", "cats", "clocks", "weight", "recency", "valid", "season", "order"):
        ch[key][row] = ch[key][row][perm]


class TowerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.tower = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
        self.examples = [_example(self.rng, 12, 0, "a"), _example(self.rng, 25, 7, "b"), _example(self.rng, 0, 0, "c")]
        self.batch = F.collate(self.examples)

    def test_untrained_heads_reproduce_the_reference_up_to_small_perturbation(self):
        params = self.tower(self.batch)
        ref = InterceptOnlySeasonModel(SeasonParameters()).constrained()
        for name, value in ref.items():
            rel = ((params[name] - value.to(params[name].dtype)).abs() / value.abs().clamp(min=1e-6)).max()
            self.assertLess(float(rel), 0.25, name)   # small random weights around the reference bias

    def test_order_invariance(self):
        base = self.tower.represent(self.batch).detach()
        perm = torch.randperm(25)
        _permute_channel(self.batch["intl"], perm, 1)
        out = self.tower.represent(self.batch).detach()
        self.assertLess(float((out - base).abs().max()), 1e-4)

    def test_padding_invariance(self):
        base = self.tower.represent(self.batch).detach()
        longer = F.collate(self.examples + [_example(self.rng, 60, 30, "d")])   # pads every channel further
        out = self.tower.represent(longer).detach()[:3]
        self.assertLess(float((out - base).abs().max()), 1e-4)

    def test_empty_history_uses_learned_vector(self):
        e = self.tower.encoder(self.batch["intl"])
        h = self.tower.pool_intl(e, self.batch["intl"])
        self.assertTrue(torch.allclose(h[2], self.tower.pool_intl.empty))
        self.assertFalse(torch.allclose(h[0], self.tower.pool_intl.empty))

    def test_aggregate_baseline_runs_and_is_order_invariant(self):
        from fpp.model.baseline import AggregateBaseline
        model = AggregateBaseline(VOCAB, STATIC_VOCAB, SeasonParameters())
        base = model.represent(self.batch).detach()
        self.assertTrue(torch.isfinite(base).all())
        perm = torch.randperm(25)
        _permute_channel(self.batch["intl"], perm, 1)
        out = model.represent(self.batch).detach()
        self.assertLess(float((out - base).abs().max()), 1e-4)
        e = model.intl(self.batch["intl"])
        self.assertTrue(torch.allclose(e[2], model.intl.empty))

    def test_likelihood_from_heads_is_finite_and_differentiable(self):
        params = self.tower(self.batch)
        lik = DifferentiableSeasonLikelihood(None)
        b = SeasonBatch(self.batch["games"], self.batch["starts"], self.batch["minutes"], self.batch["counts"], self.batch["schedule"],
                        self.batch["overtime"].long(), self.batch["overtime_known"])
        lp, _ = lik.log_prob(b, params=params)
        self.assertTrue(torch.isfinite(lp).all())
        (-lp).sum().backward()
        grads = [p.grad for p in self.tower.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(bool((g != 0).any()) for g in grads))


if __name__ == "__main__":
    unittest.main()
