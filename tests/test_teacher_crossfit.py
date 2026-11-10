"""D-110 amendments: cross-fitting covers preprocessing; the future-view weight is explicit; the cache checks itself."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fpp.experiments.teacher import refit_preprocessing, rescale_to_mean_one, step_loss
from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.teacher import PARAM_KEYS, TeacherCache, TeacherTower, checkpoint_sha256, collate_pairs, pair_examples
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood
from fpp.model.tower import ReferenceTower
from tests.test_features import _games
from tests.test_teacher import _window
from tests.test_tower import STATIC_VOCAB, VOCAB, _example


def _two_players(scale_7: float = 1.0, scale_8: float = 1.0) -> pd.DataFrame:
    a = _games()
    b = _games()
    b["player_id"] = 8
    b["record_id"] = [f"intl:{i}:8" for i in range(len(b))]
    b["game_id"] = b["game_id"] + 100
    for frame, s in ((a, scale_7), (b, scale_8)):
        frame["minutes"] = frame["minutes"] * s
        frame["pts"] = (frame["pts"] * s).round()
    return pd.concat([a, b], ignore_index=True)


class CrossFitTests(unittest.TestCase):
    def test_refit_preprocessing_ignores_held_out_players(self):
        cutoff = F._days(pd.Series([pd.Timestamp("2024-09-30")]))[0]
        units = pd.DataFrame({"unit_id": ["7:2025", "8:2025"], "player_id": [7, 8], "season": [2025, 2025], "position": ["G", "F"]})
        dobs = pd.Series({7: pd.Timestamp("2004-03-01"), 8: pd.Timestamp("2004-06-01")})

        def fit(scale_7, scale_8):
            store = F.GameStore.from_frame(_two_players(scale_7, scale_8), dobs=dobs)
            return refit_preprocessing(store, units, cutoff_days=cutoff, allowed_players={7}, excluded_rows=None, seed=1)

        base, _ = fit(1.0, 1.0)
        held_changed, vocab = fit(1.0, 3.0)        # the held-out player's games change: nothing may move
        own_changed, _ = fit(3.0, 1.0)             # the teacher's own player's games change: the statistics move
        np.testing.assert_array_equal(base.mu, held_changed.mu)
        np.testing.assert_array_equal(base.sigma, held_changed.sigma)
        self.assertFalse(np.array_equal(base.mu, own_changed.mu))
        self.assertEqual(vocab["position"], {"G": 1})                       # the vocabulary is built from the teacher's units only

    def test_step_loss_normalizes_window_rows_separately(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        forward = [_example(rng, 12, 0, "a"), _example(rng, 25, 7, "b"), _example(rng, 0, 0, "c")]
        windows = [_window(rng, 30, "a", (1,)), _window(rng, 60, "a", (None,)), _window(rng, 20, "b", (1,))]
        pairs = pair_examples(forward, windows)
        fwd, win = [p for p in pairs if p.v is None], [p for p in pairs if p.v is not None]
        teacher = TeacherTower(ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters()), SeasonParameters(), use_future=True).eval()
        lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=8)
        with torch.no_grad():
            l_fwd = float(step_loss(teacher, lik, fwd, [], 1.0))
            l1 = float(step_loss(teacher, lik, fwd, win, 1.0))
            l_dup = float(step_loss(teacher, lik, fwd, win + win, 1.0))        # twice the windows: same self-normalized term
            l_half = float(step_loss(teacher, lik, fwd, win, 0.5))
            l_zero = float(step_loss(teacher, lik, fwd, win, 0.0))
        self.assertAlmostEqual(l1, l_dup, places=6)                       # equal up to float summation order
        self.assertAlmostEqual(l_half, l_fwd + 0.5 * (l1 - l_fwd), places=6)
        self.assertAlmostEqual(l_zero, l_fwd, places=12)
        self.assertGreater(abs(l1 - l_fwd), 1e-6)

    def test_minibatch_estimator_is_unbiased_for_the_weighted_loss(self):
        """With weights rescaled to mean one and the 'size' normalizer, the mean of singleton-batch losses equals the full
        weighted loss Σwℓ/Σw exactly (linearity); a per-batch weight-sum denominator does not (the reviewer's (1, 9) example)."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        forward = [_example(rng, 12, 0, "a"), _example(rng, 25, 7, "b"), _example(rng, 0, 0, "c"), _example(rng, 9, 3, "d")]
        pairs = pair_examples(forward, [])
        for p, w in zip(pairs, (1.0, 9.0, 4.0, 2.0)):
            p.weight = w
        rescale_to_mean_one(pairs)
        self.assertAlmostEqual(float(np.mean([p.weight for p in pairs])), 1.0, places=12)
        teacher = TeacherTower(ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters()), SeasonParameters(), use_future=False).eval()
        lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=8)
        with torch.no_grad():
            full = float(step_loss(teacher, lik, pairs, [], 1.0, normalizer="size"))
            full_weight = float(step_loss(teacher, lik, pairs, [], 1.0, normalizer="weight"))
            singles_size = [float(step_loss(teacher, lik, [p], [], 1.0, normalizer="size")) for p in pairs]
            singles_weight = [float(step_loss(teacher, lik, [p], [], 1.0, normalizer="weight")) for p in pairs]
        self.assertAlmostEqual(full, full_weight, places=6)                           # mean-one weights: the two agree on the full set
        self.assertAlmostEqual(float(np.mean(singles_size)), full, places=6)         # the estimator we use: unbiased
        self.assertGreater(abs(float(np.mean(singles_weight)) - full), 1e-3)         # self-normalized singletons: biased

    def test_cache_integrity_and_identity_checks(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        forward = [_example(rng, 12, 0, "a"), _example(rng, 25, 7, "b")]
        pairs = [p for p in pair_examples(forward, [_window(rng, 30, "a", (1,)), _window(rng, 20, "b", (1,))]) if p.v is not None]
        teacher = TeacherTower(ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters()), SeasonParameters(), use_future=True)
        bx, bv, hf, w, ids = collate_pairs(pairs)
        with torch.no_grad():
            out = teacher(bx, bv, hf)
        meta = {"fold": 2023, "split": 0, "n_splits": 2, "use_future": True, "training_cutoff": "2021-09-30",
                "checkpoint_sha256": checkpoint_sha256(teacher.state_dict()), "feature_schema": {"name": "v9", "blocks": ["prior_league"]}}
        with tempfile.TemporaryDirectory() as d:
            rows = TeacherCache.empty_rows()
            TeacherCache.append(rows, ids, [p.k for p in pairs], [0, 0], w.tolist(), out)
            cache = TeacherCache.read(TeacherCache.write(Path(d) / "ok", rows, meta))
            cache.validate({"fold": 2023, "feature_schema": {"name": "v9", "blocks": ("prior_league",)}})   # tuple vs list agree
            with pytest.raises(ValueError, match="fold"):
                cache.validate({"fold": 2024})
            with pytest.raises(ValueError, match="checkpoint_sha256"):
                cache.validate({"checkpoint_sha256": "0" * 64})
            # non-finite parameter → refused on read
            bad = TeacherCache.empty_rows()
            TeacherCache.append(bad, ids, [p.k for p in pairs], [0, 0], w.tolist(), out)
            bad["pi_g"][0][0] = np.nan
            path = TeacherCache.write(Path(d) / "nan", bad, meta)
            with pytest.raises(ValueError, match="non-finite"):
                TeacherCache.read(path)
            # duplicate (outcome, window) keys → refused on read
            dup = TeacherCache.empty_rows()
            TeacherCache.append(dup, ["a", "a"], [(1,), (1,)], [0, 0], [1.0, 1.0], {k: out[k][:2] for k in PARAM_KEYS + ("representation",)})
            path = TeacherCache.write(Path(d) / "dup", dup, meta)
            with pytest.raises(ValueError, match="duplicate"):
                TeacherCache.read(path)


if __name__ == "__main__":
    unittest.main()
