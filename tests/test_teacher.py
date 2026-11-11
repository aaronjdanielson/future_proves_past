"""D-110 teacher: the no-future twin cannot respond to V; the missing view is the same pathway; pairing, splits and the
cache keep windows separate and reproduce the teacher on reload."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.teacher import PARAM_KEYS, TeacherCache, TeacherTower, checkpoint_sha256, collate_pairs, pair_examples, player_split
from fpp.model.tower import ReferenceTower
from tests.test_tower import STATIC_VOCAB, VOCAB, _example


def _window(rng, T, uid, k=(1,)):
    e = _example(rng, T, 0, uid)
    e.direction, e.k = "+", tuple(k)
    e.time, e.time_mask = F.time_basis(1.3, 21.5, 19.9)      # a later-career view: positive gap
    return e


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.forward = [_example(self.rng, 12, 0, "a"), _example(self.rng, 25, 7, "b"), _example(self.rng, 0, 0, "c")]
        self.windows = [_window(self.rng, 30, "a", (1,)), _window(self.rng, 60, "a", (None,)), _window(self.rng, 20, "b", (1,)),
                        _window(self.rng, 15, "zzz", (1,))]          # zzz has no forward example

    def _pairs(self):
        return pair_examples(self.forward, self.windows)

    def test_pairing_keeps_every_window_and_skips_orphans(self):
        pairs = self._pairs()
        self.assertEqual(len(pairs), 3 + 3)                                  # 3 missing-view entries + 3 windows (zzz skipped)
        self.assertEqual([p.unit_id for p in pairs if p.v is not None], ["a", "a", "b"])
        self.assertEqual([p.k for p in pairs if p.v is not None], [(1,), (None,), (1,)])
        self.assertTrue(all(p.v is None for p in pairs[:3]))
        bx, bv, has_future, w, ids = collate_pairs(pairs)
        self.assertEqual(bx["games"].shape[0], 6)
        self.assertEqual(has_future.tolist(), [False, False, False, True, True, True])
        self.assertEqual(ids, ["a", "b", "c", "a", "a", "b"])

    def test_no_future_teacher_is_invariant_to_the_window_batch(self):
        base = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
        d0 = TeacherTower(base, SeasonParameters(), use_future=False)
        bx, bv, has_future, _, _ = collate_pairs(self._pairs())
        other = collate_pairs(pair_examples(self.forward, [_window(np.random.default_rng(9), 40, "a"), _window(np.random.default_rng(9), 9, "b")]))[1]
        with torch.no_grad():
            o1, o2, o3 = d0(bx, bv, has_future), d0(bx, other, has_future), d0(bx)
        for key in PARAM_KEYS + ("representation",):
            self.assertTrue(torch.equal(o1[key], o2[key]), key)
            self.assertTrue(torch.equal(o1[key], o3[key]), key)

    def test_future_teacher_reads_the_window_only_where_it_exists(self):
        base = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
        teacher = TeacherTower(base, SeasonParameters(), use_future=True)
        pairs = self._pairs()
        bx, bv, has_future, _, _ = collate_pairs(pairs)
        rng9 = np.random.default_rng(9)                                      # the same pairing structure, different windows
        other = collate_pairs(pair_examples(self.forward, [_window(rng9, 40, "a", (1,)), _window(rng9, 50, "a", (None,)), _window(rng9, 9, "b", (1,))]))[1]
        with torch.no_grad():
            o1, o2 = teacher(bx, bv, has_future), teacher(bx, other, has_future)
            none = teacher(bx)
        rep1, rep2 = o1["representation"], o2["representation"]
        self.assertTrue(torch.equal(rep1[:3], rep2[:3]))                    # missing-view rows ignore the placeholder window
        self.assertTrue(torch.equal(rep1[:3], none["representation"][:3]))   # and equal the no-window pathway
        self.assertGreater(float((rep1[3:] - rep2[3:]).abs().max()), 1e-6)  # paired rows respond to V
        # The missing view is a learned parameter shared by both teachers; D⁰ and D⁺ differ only in whether V is read.
        self.assertEqual(tuple(teacher.missing.shape), (teacher.width,))
        self.assertEqual(teacher.architecture["teacher_use_future"], True)

    def test_player_split_is_deterministic_balanced_and_player_level(self):
        players = list(range(1000, 1101))
        s1, s2 = player_split(players, 2), player_split(players, 2)
        self.assertEqual(s1, s2)
        counts = np.bincount(list(s1.values()))
        self.assertTrue(abs(counts[0] - counts[1]) <= 1)
        self.assertEqual(player_split(players, 3)[1000], player_split(players[::-1], 3)[1000])
        with self.assertRaises(ValueError):
            player_split(players, 1)

    def test_cache_round_trip_reproduces_the_teacher(self):
        base = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
        teacher = TeacherTower(base, SeasonParameters(), use_future=True)
        pairs = [p for p in self._pairs() if p.v is not None]
        bx, bv, has_future, w, ids = collate_pairs(pairs)
        with torch.no_grad():
            out = teacher(bx, bv, has_future)
        rows = TeacherCache.empty_rows()
        TeacherCache.append(rows, ids, [p.k for p in pairs], range(len(pairs)), w.tolist(), out)
        meta = {"fold": 2023, "split": 0, "n_splits": 2, "use_future": True, "checkpoint_sha256": checkpoint_sha256(teacher.state_dict()), "cutoff": "2021-09-30"}
        with tempfile.TemporaryDirectory() as d:
            path = TeacherCache.write(Path(d) / "teacher_s0", rows, meta)
            cache = TeacherCache.read(path)
        self.assertEqual(cache.metadata["rows"], 3)
        self.assertEqual(cache.metadata["width"], teacher.width)
        self.assertEqual(list(cache.rows["k"]), ["1", "all", "1"])
        self.assertEqual(cache.by_unit(), {"a": [0, 1], "b": [2]})           # two windows of 'a' stay separate rows
        np.testing.assert_allclose(cache.rows["representation"], out["representation"].numpy(), rtol=0, atol=1e-12)
        for key in PARAM_KEYS:
            np.testing.assert_allclose(cache.rows[key], out[key].numpy(), rtol=0, atol=1e-12)
        self.assertEqual(len(cache.metadata["checkpoint_sha256"]), 64)
        with self.assertRaises(ValueError):
            cache.unit_scores()                                              # no held-out scores in this cache
        # With held-out log scores and cohort flags: per-unit scores and flags come back. The default statistic is the score
        # of the ω-mixture of a unit's window laws (what distillation supervises with), not the ω-mean of the window scores.
        rows2 = TeacherCache.empty_rows()
        TeacherCache.append(rows2, ids, [p.k for p in pairs], range(len(pairs)), [1.0, 3.0, 2.0], out,
                            logscore=[-40.0, -44.0, -30.0], logscore_makes=[-5.0, -6.0, -4.0],
                            logscore_makes_by_shot=[[-2.0, -2.0, -1.0], [-3.0, -1.0, -2.0], [-1.0, -2.0, -1.0]], player_ids=[11, 11, 22],
                            flags={"first_year": [1, 1, 0], "intl_entrant": [1, 1, 0], "has_history": [1, 1, 1]})
        with tempfile.TemporaryDirectory() as d:
            cache2 = TeacherCache.read(TeacherCache.write(Path(d) / "teacher_s0b", rows2, meta))
        s, s_mean = cache2.unit_scores(), cache2.unit_scores(statistic="mean")
        self.assertAlmostEqual(s["a"][0], math.log(0.25 * math.exp(-40.0) + 0.75 * math.exp(-44.0)))   # mixture over a's two windows
        self.assertAlmostEqual(s_mean["a"][0], (1.0 * -40.0 + 3.0 * -44.0) / 4.0)                      # ω-mean of the window scores
        self.assertGreater(s["a"][0], s_mean["a"][0])                                                  # Jensen: the mixture is never lower
        self.assertEqual(s["b"], (-30.0, False, False, True))                                          # one window: both agree
        self.assertEqual(s_mean["b"], s["b"])
        makes = cache2.unit_scores("makes_given_attempts")                      # the joint shooting law mixed over windows
        self.assertAlmostEqual(makes["a"][0], math.log(0.25 * math.exp(-5.0) + 0.75 * math.exp(-6.0)))
        self.assertEqual(makes["b"][0], -4.0)
        # Per shot type: each marginal mixed over windows, and their sum — the transfer score, what a factorized student can
        # learn — which differs from the joint shooting mixture when the windows disagree on more than one shot type.
        mix = lambda x, y: math.log(0.25 * math.exp(x) + 0.75 * math.exp(y))   # noqa: E731
        self.assertAlmostEqual(cache2.unit_scores("makes_K2")["a"][0], mix(-2.0, -3.0))
        self.assertAlmostEqual(cache2.unit_scores("makes_K3")["a"][0], mix(-2.0, -1.0))
        self.assertAlmostEqual(cache2.unit_scores("makes_Kf")["a"][0], mix(-1.0, -2.0))
        transfer = cache2.unit_scores("makes_transfer")
        self.assertAlmostEqual(transfer["a"][0], mix(-2.0, -3.0) + mix(-2.0, -1.0) + mix(-1.0, -2.0))
        self.assertNotAlmostEqual(transfer["a"][0], makes["a"][0])
        self.assertAlmostEqual(cache2.unit_scores("makes_transfer", "mean")["a"][0], 0.25 * -5.0 + 0.75 * -6.0)   # ω-means add up
        self.assertEqual(transfer["b"][0], -4.0)                                  # one window: every shooting score agrees
        self.assertEqual(cache2.unit_players(), {"a": 11, "b": 22})
        with self.assertRaises(ValueError):
            cache.unit_players()
        with self.assertRaises(ValueError):
            cache2.unit_scores("opportunity")
        with self.assertRaises(ValueError):
            cache2.unit_scores(statistic="median")


if __name__ == "__main__":
    unittest.main()
