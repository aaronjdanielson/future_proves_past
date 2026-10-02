"""D-070: capacity and training options — width, direction FiLM, the future head, game dropout — leave the registered
architecture unchanged by default and work when switched on."""
import unittest

import numpy as np
import pandas as pd
import torch

from fpp.model import features as F
from fpp.model.distributions import SeasonParameters
from fpp.model.tower import ARCHITECTURE_DEFAULTS, ReferenceTower
from tests.test_pool_and_arms import _pool_games


def _batch(n_fields=39, n_static=12, k=3):
    games = _pool_games(n_players=2, n_games=6)
    store = F.GameStore.from_frame(games)
    cutoff = float(store.date.max()) + 1.0
    norm = F.Normalizer.fit(store, np.arange(len(games)), cutoff, np.zeros((2, n_static)))
    rows = store.rows(1)
    ch = F.build_channel(store, rows, cutoff_days=cutoff, target_days=cutoff, normalizer=norm)
    t_v, t_m = F.time_basis(-1.0, 18.0, 19.0)
    targets = {"games": 3, "starts": 1, "minutes": 60.0, "counts": np.zeros(13, dtype=np.int64), "overtime": 0.0, "overtime_known": True, "schedule": 5}
    exs = []
    for i in range(k):
        tg = dict(targets)
        if i == 0:
            tg["future"], tg["future_known"] = 0.7, True
        exs.append(F.Example(f"u{i}", "-" if i % 2 == 0 else "+", (), ch, F.Channel.empty(n_fields), np.zeros(n_static, np.float32),
                             np.zeros(n_static, bool), np.zeros(6, np.int32), t_v, t_m, tg))
    return store, exs, F.collate(exs)


class CapacityTests(unittest.TestCase):
    def test_defaults_reproduce_the_registered_parameter_count(self):
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        tower = ReferenceTower(vocab, [2] * 6, SeasonParameters())
        self.assertEqual(tower.architecture, ARCHITECTURE_DEFAULTS)
        n = sum(p.numel() for p in tower.parameters())
        wide = ReferenceTower(vocab, [2] * 6, SeasonParameters(), width=64, hidden=128)
        self.assertGreater(sum(p.numel() for p in wide.parameters()), 3 * n)
        with torch.no_grad():
            out = wide(batch)
        self.assertEqual(out["rates"].shape[0], 3)
        self.assertNotIn("future", out)

    def test_direction_film_is_identity_at_init_and_future_head_outputs(self):
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        torch.manual_seed(0)
        plain = ReferenceTower(vocab, [2] * 6, SeasonParameters())
        torch.manual_seed(0)
        film = ReferenceTower(vocab, [2] * 6, SeasonParameters(), direction_film=True, future_head=True)
        # Same random init for the shared modules; FiLM starts at zero, so the heads agree before training.
        plain_sd = plain.state_dict()
        film.load_state_dict({**film.state_dict(), **{k: v for k, v in plain_sd.items()}}, strict=False)
        with torch.no_grad():
            a, b = plain(batch), film(batch)
        torch.testing.assert_close(a["rates"], b["rates"])
        self.assertEqual(b["future"].shape, (3,))
        self.assertTrue(batch["future_known"][0].item() and not batch["future_known"][1].item())
        self.assertAlmostEqual(float(batch["future"][0]), 0.7, places=6)
        self.assertTrue(torch.isnan(batch["future"][1]))
        # After a FiLM perturbation the two directions differ while the forward example changes too (the shift is per direction).
        with torch.no_grad():
            film.translator.film.weight[1, :10].fill_(0.5)
            c = film(batch)
        self.assertFalse(torch.allclose(a["rates"][1], c["rates"][1]))      # reconstruction example (direction +1) moved
        torch.testing.assert_close(a["rates"][0], c["rates"][0])            # forward example unchanged

    def test_drop_games_keeps_at_least_one_row_and_recomputes_evidence(self):
        store, exs, _ = _batch()
        rng = np.random.default_rng(1)
        e = exs[0]
        d = F.drop_games(e, 0.5, rng)
        self.assertGreaterEqual(d.intl.x.shape[0], 1)
        self.assertLessEqual(d.intl.x.shape[0], e.intl.x.shape[0])
        self.assertAlmostEqual(float(d.intl.evidence[0]), np.log1p(d.intl.x.shape[0]), places=5)
        self.assertEqual(float(d.intl.clocks[0, 5]), 1.0)
        self.assertEqual(float(d.intl.clocks[1:, 5].sum()) if d.intl.x.shape[0] > 1 else 0.0, 0.0)
        self.assertIs(F.drop_games(e, 0.0, rng), e)
        heavy = F.drop_games(e, 0.999, rng)
        self.assertEqual(heavy.intl.x.shape[0], 1)
        self.assertEqual(heavy.ncaa.x.shape[0], 0)                          # an empty channel stays empty

    def test_attention_pool_and_two_layer_heads(self):
        """D-074: both options are recorded, add parameters, start harmlessly (uniform attention, heads near the
        reference) and run on a batch with an absent channel."""
        from fpp.model.production import RATE_NAMES
        store, exs, batch = _batch()
        vocab = [len(store.vocab[c]) for c in F.CATEGORICAL]
        plain = ReferenceTower(vocab, [2] * 6, SeasonParameters())
        tower = ReferenceTower(vocab, [2] * 6, SeasonParameters(), head_hidden=64, attention_pool=True)
        self.assertEqual(tower.architecture["head_hidden"], 64)
        self.assertTrue(tower.architecture["attention_pool"])
        self.assertEqual(plain.architecture, ARCHITECTURE_DEFAULTS)
        self.assertGreater(sum(p.numel() for p in tower.parameters()), sum(p.numel() for p in plain.parameters()))
        self.assertTrue((tower.pool_intl.score.weight == 0).all())            # uniform attention at init
        with torch.no_grad():
            out = tower(batch)
        self.assertEqual(out["rates"].shape, (3, len(RATE_NAMES)))
        for key in ("pi_g", "alpha_m", "rates", "dispersions", "makes_alpha"):
            self.assertTrue(torch.isfinite(out[key]).all(), key)
        ref = SeasonParameters()
        self.assertLess(float((out["pi_g"] - ref.participation_probability).abs().max()), 0.1)   # heads start near the reference
        # A non-uniform score vector changes the fused channel (a uniform one would not: each game encoding is
        # LayerNorm-ed, so its feature sum is zero for every game and the attention would stay uniform).
        with torch.no_grad():
            tower.pool_intl.score.weight.copy_(torch.linspace(-1.0, 1.0, tower.pool_intl.width).unsqueeze(0))
            moved = tower(batch)
        self.assertFalse(torch.allclose(out["rates"], moved["rates"]))

    def test_lr_decay_helper(self):
        from fpp.experiments.train import _decay_lr
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        self.assertAlmostEqual(_decay_lr(opt, 0.5), 5e-4)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 5e-4)
        self.assertAlmostEqual(_decay_lr(opt, 1.0), 5e-4)


if __name__ == "__main__":
    unittest.main()
