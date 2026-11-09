"""D-097: options for freshmen without tracked history — whole-history dropout and the forward-trained static branch."""
import dataclasses
import unittest

import numpy as np
import torch

from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.tower import ARCHITECTURE_DEFAULTS, ReferenceTower
from tests.test_capacity_options import _batch


class HistoryDropoutTests(unittest.TestCase):
    def test_forward_examples_lose_the_whole_history_and_its_clocks(self):
        store, exs, _ = _batch(k=3)
        fwd, rec = exs[0], exs[1]                      # directions "-" and "+"
        rng = np.random.default_rng(0)
        d = F.drop_history(fwd, 1.0, rng)
        self.assertFalse(d.intl.present)
        self.assertEqual(d.intl.x.shape[0], 0)
        self.assertTrue(d.time_mask[1] == fwd.time_mask[1])                 # target age kept
        self.assertFalse(d.time_mask[[0] + list(range(2, len(d.time_mask)))].any())   # source age and gap unobserved
        self.assertIs(F.drop_history(rec, 1.0, rng), rec)                  # reconstruction examples untouched
        self.assertIs(F.drop_history(fwd, 0.0, rng), fwd)
        self.assertIs(F.drop_history(d, 1.0, rng), d)                      # nothing left to drop

    def test_dropped_example_matches_a_built_no_history_example(self):
        """The augmented example is what build_example gives a player without history (same time encoding)."""
        t_v, t_m = F.time_basis(np.nan, np.nan, 19.0)
        store, exs, _ = _batch(k=1)
        e = exs[0]
        e = F.Example(e.unit_id, "-", (), e.intl, e.ncaa, e.static, e.static_mask, e.static_cats,
                      *F.time_basis(-1.0, 18.0, 19.0), e.targets)
        d = F.drop_history(e, 1.0, np.random.default_rng(1))
        np.testing.assert_array_equal(d.time_mask, t_m)
        np.testing.assert_array_equal(d.time, t_v)


class HistoryDropoutScopeTests(unittest.TestCase):
    def test_no_ncaa_scope_leaves_returners_alone(self):
        from fpp.experiments.train import _drop_histories
        store, exs, _ = _batch(k=3)
        freshman = exs[0]                                                    # forward, history, no NCAA season
        returner = dataclasses.replace(freshman, ncaa=freshman.intl)        # same, with an NCAA channel present
        self.assertTrue(freshman.intl.present and not freshman.ncaa.present and returner.ncaa.present)
        out = _drop_histories([returner, freshman], 1.0, "no-ncaa", np.random.default_rng(0))
        self.assertIs(out[0], returner)
        self.assertFalse(out[1].intl.present)
        out = _drop_histories([returner, freshman], 1.0, "any", np.random.default_rng(0))
        self.assertFalse(out[0].intl.present or out[1].intl.present)

    def test_any_scope_draws_the_same_stream_as_the_first_runs(self):
        from fpp.experiments.train import _drop_histories
        store, exs, _ = _batch(k=4)
        group = list(exs) * 8
        a = _drop_histories(group, 0.5, "any", np.random.default_rng(3))
        rng = np.random.default_rng(3)
        b = [F.drop_history(e, 0.5, rng) for e in group]
        self.assertEqual([e.intl.present for e in a], [e.intl.present for e in b])
        self.assertIn(True, [e.intl.present for e in a]); self.assertIn(False, [e.intl.present for e in a])


class StaticBranchTests(unittest.TestCase):
    def test_default_off_and_main_path_initialisation_unchanged(self):
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        torch.manual_seed(0)
        plain = ReferenceTower(vocab, [2] * 6, SeasonParameters())
        torch.manual_seed(0)
        branched = ReferenceTower(vocab, [2] * 6, SeasonParameters(), static_branch=True)
        self.assertEqual(plain.architecture, ARCHITECTURE_DEFAULTS)
        self.assertTrue(branched.architecture["static_branch"])
        for k, v in plain.state_dict().items():                          # the branch is built after every main module
            torch.testing.assert_close(branched.state_dict()[k], v, msg=k)
        with torch.no_grad():
            out = branched(batch)
        self.assertTrue(torch.isfinite(out["rates"]).all())

    def test_static_branch_learns_from_forward_examples_only(self):
        store, exs, batch = _batch(k=4)                                     # directions -, +, -, +
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        tower = ReferenceTower(vocab, [2] * 6, SeasonParameters(), static_branch=True)
        direction = batch["direction"]
        for keep in (direction > 0, direction < 0):                       # reconstruction rows, then forward rows
            tower.zero_grad()
            out = tower(batch)
            out["rates"][keep].sum().backward()
            g = sum(float(p.grad.abs().sum()) for p in tower.static_branch.parameters() if p.grad is not None)
            if bool((keep == (direction > 0)).all()):
                self.assertEqual(g, 0.0)                                   # reconstruction: no gradient reaches the branch
            else:
                self.assertGreater(g, 0.0)

    def test_gate_reads_evidence(self):
        store, exs, batch = _batch(k=2)
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        tower = ReferenceTower(vocab, [2] * 6, SeasonParameters(), static_branch=True)
        g = tower.gate(batch)
        torch.testing.assert_close(g, torch.full_like(g, 0.5))            # g = 1/2 at initialisation
        with torch.no_grad():
            tower.gate.linear.weight[0, 0] = 2.0                           # intl presence raises g
            self.assertTrue(bool((tower.gate(batch) > 0.5).all()))


if __name__ == "__main__":
    unittest.main()
