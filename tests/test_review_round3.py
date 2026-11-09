"""D-095: third external review. Score checks keep every row; pretrained heads keep the recorded architecture; one row
policy for unresolved competition periods; frozen prior-NCAA context in reconstruction; the registered test season;
table and rebuild gates for reports; the truncated foul mean."""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from fpp.data.folds import UNRESOLVED_GUARD, Fold
from fpp.experiments import train as T
from fpp.experiments.report import check_tables, row_guard, truncated_nb_mean
from fpp.model import features as F
from fpp.model.dataset import FoldDataset, future_targets
from fpp.model.distributions import SeasonParameters, truncated_negative_binomial_log_prob
from fpp.model.pool import PoolSampler
from fpp.model.tower import OutcomeHeads, ReferenceTower
from tests.test_capacity_options import _batch
from tests.test_pool_and_arms import _pool_games


def _checked(*arrays, max_nodes=256):
    """Run score_checked with the scorer replaced by successive arrays (lower resolution first)."""
    values = iter([np.asarray(a, dtype=float) for a in arrays])
    examples = [SimpleNamespace(unit_id=f"u{i}") for i in range(len(arrays[0]))]
    with mock.patch.object(T, "_score", lambda *a, **k: next(values)):
        return T.score_checked(examples, batch_size=2, nodes=8, tolerance=1e-6, max_nodes=max_nodes)


class ScoreCheckTests(unittest.TestCase):
    def test_nan_anywhere_raises(self):
        with self.assertRaisesRegex(RuntimeError, "invalid scores"):
            _checked([1.0, np.nan], [1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "invalid scores"):
            _checked([np.nan, np.nan], [np.nan, np.nan])          # all invalid: previously certified with gap 0

    def test_negative_infinity_raises(self):
        with self.assertRaisesRegex(RuntimeError, r"u1"):
            _checked([1.0, -np.inf], [1.0, -np.inf])

    def test_finiteness_differing_between_resolutions_is_an_unbounded_gap(self):
        with self.assertRaisesRegex(RuntimeError, "finiteness differs"):
            _checked([1.0, np.inf], [1.0, 2.0], [1.0, np.inf], [1.0, 2.0], max_nodes=32)

    def test_infinite_score_is_kept_and_recorded(self):
        scores, info = _checked([1.0, np.inf, 3.0], [1.0, np.inf, 3.0])
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.isposinf(scores[1]))
        self.assertEqual(info["infinite_scores"], 1)
        self.assertEqual(info["infinite_units"], ["u1"])
        self.assertEqual(info["rows_checked"], 3)

    def test_finite_rows_pass_and_gap_is_measured_on_all_rows(self):
        scores, info = _checked([1.0, 2.0], [1.0, 2.0 + 1e-7])
        self.assertTrue(info["within_tolerance"])
        self.assertAlmostEqual(info["max_abs_gap"], 1e-7, places=12)


class HeadResetTests(unittest.TestCase):
    def test_reset_keeps_width_and_hidden_head_and_round_trips(self):
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        for width, head_hidden in ((32, 0), (32, 16), (64, 0), (64, 16)):
            arch = dict(width=width, hidden=64, head_hidden=head_hidden)
            tower = ReferenceTower(vocab, [2] * 6, SeasonParameters(), n_context=4, **arch)
            tower.reset_heads(SeasonParameters())
            self.assertEqual(tower.heads.linear.in_features, head_hidden or width)
            self.assertEqual(tower.heads.pre is not None, head_hidden > 0)
            with torch.no_grad():
                out = tower(batch)
            self.assertTrue(torch.isfinite(out["rates"]).all())
            fresh = ReferenceTower(vocab, [2] * 6, SeasonParameters(), n_context=4, **arch)
            fresh.load_state_dict(tower.state_dict(), strict=True)       # the saved architecture loads exactly

    def test_default_reset_matches_the_pre_d095_heads(self):
        """Registered heads (width 32, linear): the reset draws the same initialization as the old call."""
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        tower = ReferenceTower(vocab, [2] * 6, SeasonParameters())
        torch.manual_seed(5)
        old = OutcomeHeads(SeasonParameters())
        torch.manual_seed(5)
        tower.reset_heads(SeasonParameters())
        for (k, a), (_, b) in zip(old.state_dict().items(), tower.heads.state_dict().items()):
            torch.testing.assert_close(a, b, msg=k)


def _guarded_games():
    games = _pool_games(n_players=2, n_games=8)
    games["season_guard"] = ""
    bad = (games["player_id"] == 1) & (games.index % 8 < 3)          # player 1's first three games: unresolved period
    games.loc[bad, "season_guard"] = UNRESOLVED_GUARD
    return games, set(games.loc[bad, "record_id"])


class RowPolicyTests(unittest.TestCase):
    def test_guarded_rows_leave_every_store_consumer(self):
        games, bad = _guarded_games()
        store = F.GameStore.from_frame(games, row_guard=True)
        legacy = F.GameStore.from_frame(games)
        self.assertEqual(store.vocab, legacy.vocab)                        # vocabularies (and checkpoints) unchanged
        self.assertTrue(store.row_guard and not legacy.row_guard)
        self.assertFalse(set(store.record_id[store.rows(1)]) & bad)        # forward and forecast channels
        self.assertEqual(set(legacy.record_id[legacy.rows(1)]) & bad, bad)  # legacy stores keep them (old runs rebuild exactly)
        self.assertFalse(set(store.record_id[store.eligible()]) & bad)     # normalization rows
        ds = FoldDataset(None, store, None, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), set())
        unit = SimpleNamespace(player_id=1, season=2023)
        cutoff = float(F._days(pd.Series([pd.Timestamp("2022-10-01")]))[0])
        intl, _ = ds._channels(unit, cutoff)
        self.assertFalse(set(store.record_id[intl]) & bad)
        norm = F.Normalizer.fit(store, np.flatnonzero(store.eligible()), cutoff, np.zeros((2, len(F.STATIC_CONTINUOUS))))
        pool = PoolSampler(store, norm, cutoff_days=cutoff, heldout_players=set(), seed=1)
        self.assertFalse(set(store.record_id[pool.admissible]) & bad)      # the pretraining pool

    def test_future_targets_follow_the_row_policy(self):
        rows = []
        for p in (1, 2):
            for i in range(12):
                d = pd.Timestamp("2021-06-01") + pd.Timedelta(days=7 * i)
                rows.append({"player_id": p, "source": "intl", "date": d, "release_date": d, "minutes": 30.0, "pts": 12,
                             "orb": 1, "drb": 3, "ast": 2 + p, "season_guard": UNRESOLVED_GUARD if p == 1 else ""})
        units = pd.DataFrame({"unit_id": ["a", "b"], "player_id": [1, 2], "period_end": [pd.Timestamp("2021-04-30")] * 2})
        tables = {"units": units, "games": pd.DataFrame(rows)}
        fold = Fold(2023, "development")
        self.assertEqual(len(future_targets(tables, fold, units, set())), 2)
        self.assertEqual(len(future_targets(tables, fold, units, set(), row_guard=True)), 0)   # one player left: no z-score

    def test_row_policy_recorded_per_run(self):
        self.assertFalse(row_guard({"config": {}}))                         # runs before D-095
        self.assertTrue(row_guard({"config": {"row_guard": True}}))


class FrozenPriorNcaaContextTests(unittest.TestCase):
    def test_reconstruction_prior_ncaa_context_matches_the_forward_example(self):
        games = _pool_games(n_players=1, n_games=6)
        prior = games.index < 3
        games.loc[prior, "source"] = "ncaa"
        games.loc[prior, "season"] = 2022
        games.loc[prior, "opp_adj_o"] = 130.0
        games.loc[prior, "opp_strength_available"] = pd.Timestamp("2023-03-01")   # after the target's forecast cutoff
        store = F.GameStore.from_frame(games)
        c0 = float(F._days(pd.Series([pd.Timestamp("2022-10-01")]))[0])          # target forecast cutoff
        c1 = float(F._days(pd.Series([pd.Timestamp("2024-09-30")]))[0])          # training cutoff
        norm = F.Normalizer(np.zeros(len(F.FIELD_NAMES)), np.ones(len(F.FIELD_NAMES)), np.zeros(1), np.ones(1))
        unit = SimpleNamespace(player_id=1, season=2023, period_start=pd.Timestamp("2022-11-01"), dob=pd.Timestamp("2003-01-01"),
                               age_at_cutoff=19.0, minutes=100.0, games_played=10, starts=0, overtime_periods=0,
                               overtime_known=False, scheduled_games=30, unit_id="u", **{k: 0 for k in F.TARGET_COUNTS})
        rows = store.rows(1)
        ncaa_rows, intl_rows = rows[store.is_ncaa[rows]], rows[~store.is_ncaa[rows]]
        kw = dict(store=store, normalizer=norm, static=np.zeros(1), static_mask=np.ones(1, bool), static_cats=np.zeros(1, int),
                  intl_rows=intl_rows, ncaa_rows=ncaa_rows)
        fwd = F.build_example(unit, cutoff_days=c0, direction="-", **kw)
        rec = F.build_example(unit, cutoff_days=c1, direction="+", ncaa_cutoff_days=c0, **kw)
        old = F.build_example(unit, cutoff_days=c1, direction="+", **kw)
        np.testing.assert_array_equal(fwd.ncaa.x, rec.ncaa.x)
        np.testing.assert_array_equal(fwd.ncaa.mask, rec.ncaa.mask)
        self.assertFalse(np.array_equal(fwd.ncaa.x, old.ncaa.x))            # the pre-D-095 call saw the later rating


class ProtocolAndReportGateTests(unittest.TestCase):
    def test_test_season_comes_from_the_protocol(self):
        registered = json.loads(T.PROTOCOL_PATH.read_text())["experiment"]["test_season"]
        self.assertEqual(T.protocol_test_season(), int(registered))

    def test_table_digest_gate(self):
        run = {"config": {"tables_manifest": {"payload_sha256": "aaa"}}}
        self.assertTrue(check_tables(run, {"manifest": {"payload_sha256": "aaa"}})["match"])
        with self.assertRaisesRegex(ValueError, "differs from the run's training tables"):
            check_tables(run, {"manifest": {"payload_sha256": "bbb"}})
        allowed = check_tables(run, {"manifest": {"payload_sha256": "bbb"}}, allow_mismatch=True)
        self.assertFalse(allowed["match"])
        self.assertIsNone(check_tables({"config": {}}, {"manifest": {"payload_sha256": "bbb"}})["match"])   # unverifiable


class MomentTailBoundTests(unittest.TestCase):
    def test_moment_sum_stops_within_its_own_bound(self):
        """The first-moment sum (latent minutes) stops under L g Pr(O > N) + t mu Pr(O >= N); with heavy overtime its
        adaptive value stays within the registered tolerance of a long fixed sum, and the reported bound covers the error."""
        from fpp.experiments.gradcheck import structural_cases
        from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch
        params = SeasonParameters(overtime_rate_per_game=0.6)
        lik = DifferentiableSeasonLikelihood(InterceptOnlySeasonModel(params), quadrature_nodes=32)
        _, draws = structural_cases(params, np.random.default_rng(3), 32)
        b = SeasonBatch.from_outcomes(draws, 32)
        with torch.no_grad():
            adaptive, diag = lik.log_prob(b, production=False, moment=True)
            exact, _ = lik.log_prob(b, production=False, moment=True, overtime_terms=120)
        ok = torch.isfinite(exact) & (b.games > 0)
        err = float((adaptive[ok] - exact[ok]).abs().max())
        self.assertLessEqual(err, lik.accuracy.overtime_log_tolerance)
        self.assertLessEqual(err, diag.max_overtime_log_error_bound + 1e-12)


def _fitted(tmp, **over):
    from fpp.model.artifact import FittedModel, save_fitted_model
    store, exs, batch = _batch()
    vocab_sizes = [len(store.vocab[c]) for c in F.CATEGORICAL]
    arch = {"width": 32, "hidden": 64, "direction_film": False, "future_head": False, "head_hidden": 16, "attention_pool": False}
    torch.manual_seed(3)
    tower = ReferenceTower(vocab_sizes, [2] * 6, SeasonParameters(), **arch)
    norm = F.Normalizer(np.arange(39, dtype=np.float32), np.ones(39, dtype=np.float32), np.zeros(12), np.ones(12))
    fitted = FittedModel(model_kind="tower", state_dict=tower.state_dict(),
                         constructor={"vocab_sizes": vocab_sizes, "static_sizes": [2] * 6, "n_context": 0, "n_fields": 39,
                                      "n_static": 12, "architecture": arch},
                         feature_schema={"name": "v7", "blocks": []}, vocab=store.vocab, static_vocab={"a": {"x": 1}},
                         normalizer=norm, row_guard=True, seasons={"fold": 2023}, tables_manifest={"payload_sha256": "aaa"},
                         data_policy={}, conventions={}, **over)
    path = tmp / "model_artifact.pt"
    return tower, batch, fitted, path, save_fitted_model(path, fitted)


class FittedModelArtifactTests(unittest.TestCase):
    def test_round_trip_restores_the_model_and_preprocessing_exactly(self):
        import tempfile
        from pathlib import Path
        from fpp.model.artifact import load_fitted_model
        with tempfile.TemporaryDirectory() as d:
            tower, batch, fitted, path, sha = _fitted(Path(d))
            back = load_fitted_model(path, expected_sha256=sha)
            model = back.build_model()
            tower.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(batch)["rates"], tower(batch)["rates"], rtol=0, atol=0)
            self.assertEqual(back.normalizer.mu.dtype, np.float32)               # dtypes kept: bit-identical inputs
            self.assertTrue(back.row_guard and back.check_tables({"manifest": {"payload_sha256": "aaa"}}))
            back.check_preprocessing(vocab=fitted.vocab, static_vocab=fitted.static_vocab, normalizer=fitted.normalizer)
            moved = F.Normalizer(fitted.normalizer.mu + 1e-3, fitted.normalizer.sigma, fitted.normalizer.static_mu, fitted.normalizer.static_sigma)
            with self.assertRaisesRegex(ValueError, "normalizer"):
                back.check_preprocessing(normalizer=moved)
            with self.assertRaisesRegex(ValueError, "game vocabulary"):
                back.check_preprocessing(vocab={**fitted.vocab, "competition": {"new": 1}})
            with self.assertRaisesRegex(ValueError, "does not match the digest"):
                load_fitted_model(path, expected_sha256="0" * 64)


class PoolContractTests(unittest.TestCase):
    def test_contract_digest_changes_with_anything_the_pool_depends_on(self):
        from fpp.model.artifact import canonical_sha256, pool_contract
        games, _ = _guarded_games()
        store, guarded = F.GameStore.from_frame(games), F.GameStore.from_frame(games, row_guard=True)
        norm = F.Normalizer(np.zeros(39, np.float32), np.ones(39, np.float32), np.zeros(12), np.ones(12))
        base = dict(tables_sha256="t", schema={"name": "v8"}, store=store, normalizer=norm, heldout={1, 2}, excluded_rows=None,
                    architecture={"width": 32}, vocab_sizes=[3], static_sizes=[2], n_context=4, settings={"epochs": 3}, code={"x": "1"})
        digest = canonical_sha256(pool_contract(**base))
        self.assertEqual(digest, canonical_sha256(pool_contract(**{**base, "heldout": {2, 1}})))      # order-free
        for change in ({"heldout": {1}}, {"store": guarded}, {"architecture": {"width": 64}}, {"settings": {"epochs": 1}},
                       {"normalizer": F.Normalizer(np.ones(39, np.float32), np.ones(39, np.float32), np.zeros(12), np.ones(12))},
                       {"excluded_rows": np.zeros(len(games), bool)}, {"tables_sha256": "u"}, {"code": {"x": "2"}}):
            self.assertNotEqual(digest, canonical_sha256(pool_contract(**{**base, **change})), msg=str(list(change)))


class TruncatedFoulMeanTests(unittest.TestCase):
    def test_matches_the_implemented_distribution(self):
        for mean, disp, cap in ((10.0, 2.0, 5), (45.0, 3.0, 150), (1.2, 0.8, 5), (60.0, 1.5, 60)):
            brute = sum(k * np.exp(truncated_negative_binomial_log_prob(k, mean, disp, cap)) for k in range(cap + 1))
            self.assertAlmostEqual(float(truncated_nb_mean(mean, disp, cap)), brute, places=8)

    def test_high_and_low_truncation_limits(self):
        self.assertAlmostEqual(float(truncated_nb_mean(10.0, 2.0, 5)), 2.9004933142931324, places=10)   # the review's case
        self.assertAlmostEqual(float(truncated_nb_mean(10.0, 2.0, 10 ** 6)), 10.0, places=8)          # untruncated limit
        np.testing.assert_allclose(truncated_nb_mean(np.array([0.0, 3.0]), 2.0, np.array([5, 0])), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
