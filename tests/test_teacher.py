"""D-110 teacher: the no-future twin cannot respond to V; the missing view is the same pathway; pairing, splits and the
cache keep windows separate and reproduce the teacher on reload."""
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


if __name__ == "__main__":
    unittest.main()
