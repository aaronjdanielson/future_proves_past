"""D-109/D-110 student side: the makes factor leaves the season likelihood exactly; the shooting cross-entropy is minimized
at the teacher's law; targets index and validate; alignment and representation loss behave."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

from fpp.experiments.distill import (Alignments, TeacherTargets, assignment_digest, distilled_rows, load_teacher_caches,
                                     makes_logprob_by_shot, representation_loss, shooting_cross_entropy, student_makes_logprob)
from fpp.experiments.train import season_batch
from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.teacher import PARAM_KEYS, TeacherCache, TeacherTower, checkpoint_sha256, collate_pairs, pair_examples
from fpp.model.torch_likelihood import DifferentiableSeasonLikelihood
from fpp.model.tower import ReferenceTower
from tests.test_teacher import _window
from tests.test_tower import STATIC_VOCAB, VOCAB, _example

D = torch.float64


def _student_and_batch(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    examples = [_example(rng, 12, 0, "a"), _example(rng, 25, 7, "b"), _example(rng, 0, 0, "c")]
    batch = F.collate(examples)
    tower = ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters())
    return tower, batch, examples


def _cache_dir(root: Path, split: int, n_splits: int, teacher: TeacherTower, pairs, players: dict, fold=2023, sha=None) -> Path:
    bx, bv, hf, w, ids = collate_pairs(pairs)
    with torch.no_grad():
        out = teacher(bx, bv, hf)
    rows = TeacherCache.empty_rows()
    position, seen = [], {}
    for p in pairs:
        position.append(seen.get(p.unit_id, 0))
        seen[p.unit_id] = position[-1] + 1
    TeacherCache.append(rows, ids, [p.k for p in pairs], position, w.tolist(), out)
    meta = {"fold": fold, "split": split, "n_splits": n_splits, "use_future": True, "training_cutoff": "2021-09-30",
            "tables_manifest": {"payload_sha256": "tabsha"}, "feature_schema": {"name": "v9", "blocks": ["prior_league"]},
            "split_assignment_sha256": sha or assignment_digest(players), "checkpoint_sha256": checkpoint_sha256(teacher.state_dict()), "teacher_lambda": 1.0}
    d = root / f"teacher_s{split}"
    d.mkdir()
    TeacherCache.write(d / "targets", rows, meta)
    (d / "split_assignment.json").write_text(json.dumps({"n_splits": n_splits, "split": split, "salt": 1, "players": players}))
    return d


class DistillTests(unittest.TestCase):
    def test_makes_factor_leaves_the_likelihood_exactly(self):
        tower, batch, _ = _student_and_batch()
        lik = DifferentiableSeasonLikelihood(None, quadrature_nodes=8)
        with torch.no_grad():
            params = tower(batch)
            sb = season_batch(batch)
            lp_full, _ = lik.log_prob(sb, params=params)
            lp_makes = student_makes_logprob(params, batch["counts"])
            other = {k: v.clone() for k, v in params.items()}
            other["makes_alpha"] = other["makes_alpha"] * 1.7 + 0.3          # change only the makes law
            other["makes_beta"] = other["makes_beta"] * 0.6 + 0.2
            lp_full2, _ = lik.log_prob(sb, params=other)
            lp_makes2 = student_makes_logprob(other, batch["counts"])
        self.assertTrue(torch.isfinite(lp_full).all())
        np.testing.assert_allclose((lp_full2 - lp_full).numpy(), (lp_makes2 - lp_makes).numpy(), rtol=0, atol=1e-9)
        # zero attempts carry no shooting information: a unit with all-zero counts has a zero makes factor
        zero = batch["counts"].clone()
        zero[:] = 0
        self.assertTrue(bool((student_makes_logprob(params, zero) == 0).all()))
        # per shot type (2P, 3P, FT): the three marginal terms whose sum is the makes factor
        by_shot = makes_logprob_by_shot(params, batch["counts"])
        self.assertEqual(tuple(by_shot.shape), (batch["counts"].shape[0], 3))
        np.testing.assert_allclose(by_shot.sum(1).numpy(), lp_makes.numpy(), rtol=0, atol=1e-12)

    def test_shooting_cross_entropy_is_minimized_at_the_teacher_law_and_blends_correctly(self):
        tower, batch, examples = _student_and_batch()
        with torch.no_grad():
            params = {k: v.clone() for k, v in tower(batch).items()}
        n = batch["counts"].shape[0]
        arrays = {k: params[k].numpy().copy() for k in PARAM_KEYS}
        arrays.update({"representation": np.zeros((n, 32)), "weight": np.ones(n), "split": np.zeros(n, dtype=np.int64)})
        targets = TeacherTargets(arrays, {"a": np.array([0]), "b": np.array([1])}, np.ones(n), 32, 1, True, [])
        rows, pos, omega, split = targets.rows_for([e.unit_id for e in examples])
        ce_match = shooting_cross_entropy(params, batch["counts"], targets, rows, pos, omega)
        # at q = p the cross-entropy is the entropy: non-negative, and below any perturbed student's cross-entropy
        self.assertTrue(bool((ce_match[:2] >= 0).all()) and float(ce_match[2]) == 0.0)
        worse = {k: v.clone() for k, v in params.items()}
        worse["makes_alpha"] = worse["makes_alpha"] * 2.5
        ce_worse = shooting_cross_entropy(worse, batch["counts"], targets, rows, pos, omega)
        self.assertTrue(bool((ce_worse[:2] > ce_match[:2]).all()))
        # the gradient of CE in the student's makes parameters vanishes at the teacher's law
        ma = params["makes_alpha"].clone().requires_grad_(True)
        p2 = {**params, "makes_alpha": ma}
        shooting_cross_entropy(p2, batch["counts"], targets, rows, pos, omega).sum().backward()
        self.assertLess(float(ma.grad[:2].abs().max()), 1e-6)
        # the blend: α = 0 is the plain NLL; α = 1 replaces the shooting NLL by the CE on rows with targets
        lp_full = torch.tensor([-50.0, -40.0, -10.0], dtype=D)
        lp_makes = torch.tensor([-5.0, -4.0, 0.0], dtype=D)
        has = torch.tensor([True, True, False])
        ce = torch.tensor([6.0, 3.0, 0.0], dtype=D)
        np.testing.assert_allclose(distilled_rows(lp_full, lp_makes, ce, has, 0.0).numpy(), [50.0, 40.0, 10.0])
        np.testing.assert_allclose(distilled_rows(lp_full, lp_makes, ce, has, 1.0).numpy(), [45.0 + 6.0, 36.0 + 3.0, 10.0])
        np.testing.assert_allclose(distilled_rows(lp_full, lp_makes, ce, has, 0.5).numpy(), [45.0 + 2.5 + 3.0, 36.0 + 2.0 + 1.5, 10.0])

    def test_alignment_and_representation_loss(self):
        u = torch.randn(4, 32, dtype=D)
        align = Alignments(2, 32, 32)
        targets = TeacherTargets({"representation": u.numpy().copy(), "split": np.array([0, 1, 0, 1])}, {}, np.ones(4), 32, 2, True, [])
        rows, pos, omega, split = np.arange(4), np.arange(4), np.ones(4), np.array([0, 1, 0, 1])
        zero = representation_loss(u, align, targets, rows, pos, omega, split, torch.ones(4))
        self.assertAlmostEqual(float(zero), 0.0, places=12)                  # identity alignment, targets equal u
        with torch.no_grad():
            align.maps["1"].bias.fill_(1.0)                                     # split 1's map moves; split 0's rows are unaffected
        loss = representation_loss(u, align, targets, rows, pos, omega, split, torch.ones(4))
        self.assertAlmostEqual(float(loss), (2 * 32.0) / 4, places=9)           # two rows of split 1, each ‖1‖² = 32, mean over 4 units

    def test_loader_validates_and_indexes_caches(self):
        rng = np.random.default_rng(1)
        torch.manual_seed(1)
        forward = [_example(rng, 10, 0, "7:2021"), _example(rng, 20, 5, "8:2021"), _example(rng, 15, 0, "9:2021")]
        windows = [_window(rng, 30, "7:2021", (1,)), _window(rng, 40, "7:2021", (None,)), _window(rng, 20, "8:2021", (1,)), _window(rng, 25, "9:2021", (1,))]
        pairs = [p for p in pair_examples(forward, windows) if p.v is not None]
        teacher = TeacherTower(ReferenceTower(VOCAB, STATIC_VOCAB, SeasonParameters()), SeasonParameters(), use_future=True)
        players = {"7": 0, "8": 1, "9": 0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d0 = _cache_dir(root, 0, 2, teacher, [p for p in pairs if p.unit_id != "8:2021"], players)
            d1 = _cache_dir(root, 1, 2, teacher, [p for p in pairs if p.unit_id == "8:2021"], players)
            t = load_teacher_caches([d1, d0], fold_season=2023, training_cutoff="2021-09-30", tables_sha256="tabsha",
                                    feature_schema={"name": "v9", "blocks": ("prior_league",)},
                                    unit_player={"7:2021": 7, "8:2021": 8, "9:2021": 9})
            self.assertEqual(t.n_rows, 4)
            self.assertEqual(sorted(t.index), ["7:2021", "8:2021", "9:2021"])
            rows, pos, omega, split = t.rows_for(["9:2021", "zzz", "7:2021"])
            self.assertEqual(pos.tolist(), [0, 2, 2])
            self.assertAlmostEqual(float(omega[1] + omega[2]), 1.0)              # 7:2021's two windows share unit mass
            self.assertEqual(sorted(set(split.tolist())), [0])
            self.assertEqual(t.describe()["splits"], [0, 1])
            good = {"7:2021": 7, "8:2021": 8, "9:2021": 9}
            fs = {"name": "v9", "blocks": ["prior_league"]}
            with pytest.raises(ValueError, match="fold"):
                load_teacher_caches([d0, d1], fold_season=2024, training_cutoff="2021-09-30", tables_sha256="tabsha", feature_schema=fs, unit_player=good)
            with pytest.raises(ValueError, match="tables"):
                load_teacher_caches([d0, d1], fold_season=2023, training_cutoff="2021-09-30", tables_sha256="other", feature_schema=fs, unit_player=good)
            with pytest.raises(ValueError, match="every split"):
                load_teacher_caches([d0], fold_season=2023, training_cutoff="2021-09-30", tables_sha256="tabsha", feature_schema=fs, unit_player=good)
            common = dict(fold_season=2023, training_cutoff="2021-09-30", tables_sha256="tabsha", feature_schema=fs)
            with pytest.raises(ValueError, match="not in split"):
                load_teacher_caches([d0, d1], unit_player={"7:2021": 8, "8:2021": 8, "9:2021": 9}, **common)
            with pytest.raises(ValueError, match="not among the student's training units"):          # an outcome the student cannot place
                load_teacher_caches([d0, d1], unit_player={"7:2021": 7, "8:2021": 8}, **common)
            with pytest.raises(ValueError, match="does not match the digest"):                        # tampered assignment file
                (d1 / "split_assignment.json").write_text(json.dumps({"n_splits": 2, "split": 1, "salt": 1, "players": {"7": 1, "8": 1, "9": 0}}))
                load_teacher_caches([d0, d1], unit_player=good, **common)
            (d1 / "split_assignment.json").unlink()
            with pytest.raises(ValueError, match="no split_assignment.json"):                         # missing evidence is not verification
                load_teacher_caches([d0, d1], unit_player=good, **common)
            # an outcome supplied by two teachers is refused
            d1b = _cache_dir(root / "again", 1, 2, teacher, [p for p in pairs if p.unit_id in ("8:2021", "7:2021")], players) if (root / "again").mkdir() is None else None
            with pytest.raises(ValueError, match="supplied by teachers of splits"):
                load_teacher_caches([d0, d1b], unit_player=good, **common)


if __name__ == "__main__":
    unittest.main()
