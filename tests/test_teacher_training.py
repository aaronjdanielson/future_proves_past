"""D-110 teacher training: split exclusion is structural (held-out players never reach the teacher's labels, the stopping
season reaches neither side, windows follow their units) and the test season is out of bounds."""
import unittest

import numpy as np
import pandas as pd
import pytest

from fpp.experiments.teacher import prepare_dataset, teacher_masks, train_teacher
from fpp.model.teacher import player_split


class TeacherMaskTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for player in range(1, 9):
            for season in (2019, 2020, 2021, 2022):          # 2022 is the stopping season of fold 2023
                rows.append({"unit_id": f"{player}:{season}", "player_id": player, "season": season})
        self.units = pd.DataFrame(rows)
        self.windows = pd.DataFrame({"unit_id": [f"{p}:{s}" for p in (1, 2, 5, 8) for s in (2019, 2021)] + ["3:2022"],
                                     "player_id": [p for p in (1, 2, 5, 8) for _ in (2019, 2021)] + [3], "k": [(1,)] * 9})
        self.split = player_split(self.units["player_id"], 2)

    def test_masks_partition_the_training_era_by_player_and_drop_the_stop_season(self):
        for k in (0, 1):
            train_mask, held_mask, train_win, held_win = teacher_masks(self.units, self.windows, fold_season=2023, split=self.split, k=k)
            self.assertFalse((train_mask & held_mask).any())
            self.assertFalse(self.units.loc[train_mask | held_mask, "season"].eq(2022).any())      # stopping season on neither side
            held_players = set(self.units.loc[held_mask, "player_id"])
            train_players = set(self.units.loc[train_mask, "player_id"])
            self.assertFalse(held_players & train_players)
            self.assertEqual(held_players, {p for p, s in self.split.items() if s == k})
            # windows follow their units, and the stop-season window of player 3 is on neither side
            self.assertEqual(set(self.windows.loc[train_win, "unit_id"]) | set(self.windows.loc[held_win, "unit_id"]),
                             set(self.windows["unit_id"]) - {"3:2022"})
            self.assertTrue(set(self.windows.loc[held_win, "player_id"]) <= held_players)
        # the two splits' held-out sets cover every training-era unit exactly once
        _, h0, _, _ = teacher_masks(self.units, self.windows, fold_season=2023, split=self.split, k=0)
        _, h1, _, _ = teacher_masks(self.units, self.windows, fold_season=2023, split=self.split, k=1)
        self.assertEqual(int(h0.sum() + h1.sum()), int((self.units["season"] <= 2021).sum()))

    def test_test_season_is_out_of_bounds_before_any_table_is_read(self):
        with pytest.raises(ValueError, match="test season"):
            prepare_dataset("no_such_tables", 2026, feature_schema="auto", feature_blocks=None, row_guard=True, recon_k=(1,), recon_max_years=None, seed=1)
        with pytest.raises(ValueError, match="split must lie"):
            train_teacher(tables="no_such_tables", fold_season=2023, n_splits=2, split=2, use_future=True, out="no_out", seed=1)


if __name__ == "__main__":
    unittest.main()
