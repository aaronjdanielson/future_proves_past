"""Regression tests for the review findings resolved in D-026."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.data.assemble import team_season_ledger
from fpp.experiments.gradcheck import structural_cases
from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.observation import OnceRoundedMinutesKernel, SummedRoundingSensitivityKernel
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood, InterceptOnlySeasonModel, SeasonBatch, value_agreement
from fpp.model.tower import PARAMETER_FLOORS, OutcomeHeads, ReferenceTower


class LedgerTests(unittest.TestCase):
    """Finding 1: a team game without player logs must reduce coverage, not the schedule."""

    def test_missing_logs_do_not_shorten_the_denominator(self):
        team_games = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 10], "season": [2024, 2024],
                                   "coverage_certified": [True, True]})
        logs = pd.DataFrame({"game_id": [1], "team_id": [10], "season": [2024], "coverage_certified": [True]})
        ncaa_games = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 20], "date": pd.to_datetime(["2023-11-10", "2023-11-14"])})
        out = team_season_ledger(team_games, logs, ncaa_games)
        row = out.iloc[0]
        self.assertEqual(int(row.scheduled_games), 2)
        self.assertEqual(int(row.certified_games), 1)
        self.assertFalse(bool(row.coverage_complete))
        self.assertEqual(row.period_end, pd.Timestamp("2023-11-14"))   # dated through the opponent's log

    def test_undated_game_is_kept_uncertified(self):
        team_games = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 10], "season": [2024, 2024], "coverage_certified": [True, True]})
        logs = pd.DataFrame({"game_id": [1], "team_id": [10], "season": [2024], "coverage_certified": [True]})
        ncaa_games = pd.DataFrame({"game_id": [1], "team_id": [10], "date": pd.to_datetime(["2023-11-10"])})
        out = team_season_ledger(team_games, logs, ncaa_games)
        self.assertEqual(int(out.iloc[0].scheduled_games), 2)
        self.assertEqual(int(out.iloc[0].undated_games), 1)
        self.assertFalse(bool(out.iloc[0].coverage_complete))


class TeamHistoryAvailabilityTests(unittest.TestCase):
    """Finding 2: the player's own team record obeys the same availability date as the opponent's."""

    def test_team_history_masked_before_availability(self):
        from tests.test_features import _games
        games = _games()
        games["team_hist_available"] = pd.Timestamp("2023-06-30")
        store = F.GameStore.from_frame(games)
        rows = store.rows(7)
        col = F.FIELD_NAMES.index("team_hist_winpct")
        early = store.continuous(rows[:3], F._days(pd.Series([pd.Timestamp("2023-02-01")]))[0])
        late = store.continuous(rows[:3], F._days(pd.Series([pd.Timestamp("2023-09-30")]))[0])
        self.assertTrue(np.isnan(early[:, col]).all())
        self.assertEqual(list(late[:, col]), [0.5] * 3)


class ContrastOrderTests(unittest.TestCase):
    """Finding 6: the within-season contrast ranks games by stored order, not tensor position."""

    def test_permuting_complete_records_leaves_the_contrast_unchanged(self):
        from tests.test_tower import STATIC_VOCAB, VOCAB, _channel, _permute_channel
        rng = np.random.default_rng(3)
        torch.manual_seed(3)
        tower = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
        ch = _channel(rng, 14, True, seasons=(2023, 2023))      # 14 games, all in the latest season → contrast active
        static = np.zeros(len(F.STATIC_CONTINUOUS), dtype=np.float32)
        t_v, t_m = F.time_basis(-0.4, 19.5, 19.9)
        targets = {"games": 10, "starts": 2, "minutes": 150.0, "counts": np.zeros(13, dtype=np.int64), "overtime": 0.0,
                   "overtime_known": True, "schedule": 30}
        ex = F.Example("u", "-", (), ch, F.Channel.empty(len(F.FIELD_NAMES)), static, static > -9,
                       np.zeros(len(F.STATIC_CATEGORICAL), dtype=np.int32), t_v, t_m, targets)
        batch = F.collate([ex])
        e = tower.encoder(batch["intl"])
        d0, flags0 = tower.pool_intl.contrasts(e, batch["intl"], batch["intl"]["weight"] * batch["intl"]["valid"])
        self.assertEqual(float(flags0[0, 0]), 1.0)                       # within-season contrast present
        perm = torch.flip(torch.arange(14), dims=[0])                     # reversed order is the worst case
        _permute_channel(batch["intl"], perm, 0)
        e1 = tower.encoder(batch["intl"])
        d1, _ = tower.pool_intl.contrasts(e1, batch["intl"], batch["intl"]["weight"] * batch["intl"]["valid"])
        self.assertLess(float((d1 - d0).abs().max()), 1e-5)


class HeadDomainTests(unittest.TestCase):
    """Finding 4: heads stay inside a registered domain and reproduce the reference at initialization."""

    def test_floors_and_initialization(self):
        heads = OutcomeHeads(SeasonParameters(), init_scale=0.0)
        u = torch.randn(5, 32)
        out = heads(u)
        ref = InterceptOnlySeasonModel(SeasonParameters()).constrained()
        for name, value in ref.items():
            self.assertLess(float((out[name] - value.to(out[name].dtype)).abs().max()), 1e-6, name)
        heads = OutcomeHeads(SeasonParameters(), init_scale=50.0)         # violent weights
        out = heads(torch.randn(64, 32) * 10)
        for name, floor in PARAMETER_FLOORS.items():
            self.assertGreaterEqual(float(out[name].min()), floor, name)
        self.assertGreater(float(out["pi_g"].min()), 0.0)
        self.assertLess(float(out["pi_m"].max()), 1.0)


class QuadratureDomainTests(unittest.TestCase):
    """Finding 4: at 32 nodes the likelihood meets its tolerance across the registered domain,
    and the reviewer's out-of-domain case is indeed outside it."""

    def _agreement(self, **overrides):
        params = SeasonParameters(**overrides)
        rng = np.random.default_rng(7)
        cases, draws = structural_cases(params, rng, 32)
        outcomes = list(cases.values()) + draws
        lik = DifferentiableSeasonLikelihood(InterceptOnlySeasonModel(params), quadrature_nodes=32)
        return value_agreement(lik, outcomes, 32)

    def test_shape_grid_at_the_floors(self):
        fl = PARAMETER_FLOORS
        grid = [dict(minutes_alpha=fl["alpha_m"], minutes_beta=fl["beta_m"]),
                dict(minutes_alpha=fl["alpha_m"], minutes_beta=8.0),
                dict(minutes_alpha=12.0, minutes_beta=fl["beta_m"]),
                dict(minutes_alpha=40.0, minutes_beta=40.0),
                dict(minutes_alpha=0.6, minutes_beta=0.6, minutes_max_probability=0.3)]
        for overrides in grid:
            result = self._agreement(**overrides)
            self.assertTrue(result["same_support"], overrides)
            self.assertLess(result["max_abs_diff_nats"], 5e-5, (overrides, result["max_abs_diff_nats"]))

    def test_reviewer_case_is_outside_the_domain(self):
        self.assertLess(0.001, PARAMETER_FLOORS["alpha_m"])


class KernelSensitivityTests(unittest.TestCase):
    """Finding 5: the summed-rounding probe widens the once-rounded cell by the Irwin–Hall width."""

    def test_probe_reduces_to_once_rounded_without_widening(self):
        once = OnceRoundedMinutesKernel()
        probe = SummedRoundingSensitivityKernel(sd_multiple=0.0)
        for recorded, maximum, games in ((100.0, 800.0, 20), (0.0, 40.0, 1), (800.0, 800.0, 20)):
            self.assertEqual(once.cell(recorded, maximum), probe.cell(recorded, maximum, games))

    def test_probe_widens_with_games_and_scores(self):
        probe = SummedRoundingSensitivityKernel()
        lo, hi = probe.cell(400.0, 1200.0, 30)
        self.assertAlmostEqual(hi - lo, 2 * np.sqrt(30 / 12), places=9)
        params = SeasonParameters()
        cases, draws = structural_cases(params, np.random.default_rng(1), 32)
        batch = SeasonBatch.from_outcomes(list(cases.values()) + draws, 32)
        model = InterceptOnlySeasonModel(params)
        once, _ = DifferentiableSeasonLikelihood(model).log_prob(batch)
        wide, _ = DifferentiableSeasonLikelihood(model, kernel=probe).log_prob(batch)
        finite = torch.isfinite(once)
        self.assertTrue(torch.equal(finite, torch.isfinite(wide)))
        self.assertTrue(bool((wide[finite] - once[finite]).abs().max() > 0))   # the probe is not a no-op
        # A wider cell holds more of the continuous mass, so the probe's score is never far below.
        self.assertGreater(float((wide[finite] - once[finite]).min()), -1e-9)


if __name__ == "__main__":
    unittest.main()
