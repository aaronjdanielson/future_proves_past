"""D-091: high school strength block — dated alumni scores, the unit join, and the feature schema."""
import numpy as np
import pandas as pd

from fpp.data.assemble import attach_hs_block
from fpp.data.hs_schools import school_scores, school_scores_decay
from fpp.model.features import BLOCKS, feature_schema


def _talent():
    # school 1 alumni: freshmen in 2010 (talent 0.9) and 2012 (0.1); school 2: one alumnus in 2011 (0.5)
    return pd.DataFrame({"player_id": [1, 2, 3], "season": [2010, 2012, 2011], "decile": [9, 0, 5],
                         "min_share": [0.6, 0.1, 0.3], "prod40": [20.0, 5.0, 10.0], "talent": [0.9, 0.1, 0.5]})


def _entries():
    return pd.DataFrame({"player_id": [1, 2, 3, 10, 11, 12], "entry_season": [2010, 2012, 2011, 2012, 2013, 2011],
                         "hs_id": [1, 1, 2, 1, 1, 3], "hs_region": ["Texas", "Texas", "Texas", "Texas", "Texas", "Ohio"]})


def test_alumni_only_from_earlier_seasons():
    t = school_scores(_talent(), _entries(), k=0.0).set_index("player_id")
    # entering 2012 from school 1: only the 2010 alumnus counts (the 2012 freshman is a contemporary, not an alumnus)
    assert t.loc[10, "hs_alumni_n"] == 1 and np.isclose(t.loc[10, "hs_talent"], 0.9)
    # entering 2013: both school-1 alumni count
    assert t.loc[11, "hs_alumni_n"] == 2 and np.isclose(t.loc[11, "hs_talent"], 0.5)
    assert np.isclose(t.loc[11, "hs_top2_share"], 0.5) and np.isclose(t.loc[11, "hs_alumni_min_share"], 0.35)
    # the first alumnus himself had no earlier alumni
    assert t.loc[1, "hs_alumni_n"] == 0 and np.isnan(t.loc[1, "hs_alumni_min_share"])


def test_shrinkage_toward_earlier_regional_mean():
    t = school_scores(_talent(), _entries(), k=2.0).set_index("player_id")
    # player 10 (2012, school 1): one alumnus at 0.9; Texas alumni before 2012 = {0.9 (2010), 0.5 (2011)} -> prior 0.7
    assert np.isclose(t.loc[10, "hs_talent"], (0.9 + 2 * 0.7) / 3)
    # player 12 (Ohio, no Ohio alumni yet): prior falls back to the global mean of alumni before 2011 (0.9)
    assert np.isclose(t.loc[12, "hs_talent"], 0.9)


def test_attach_marks_unknown_schools_and_keeps_later_units():
    hs = school_scores(_talent(), _entries(), k=0.0)
    units = pd.DataFrame({"player_id": [11, 11, 99], "season": [2013, 2014, 2013]})
    out = attach_hs_block(units, hs).set_index(["player_id", "season"])
    assert out.loc[(11, 2013), "hs_known"] == 1.0 and out.loc[(11, 2014), "hs_talent"] == out.loc[(11, 2013), "hs_talent"]
    assert out.loc[(99, 2013), "hs_known"] == 0.0 and np.isnan(out.loc[(99, 2013), "hs_talent"])
    assert "entry_season" not in out.columns


def test_schema_block():
    assert "hs" in BLOCKS
    s = feature_schema("v9", ["hs"])
    assert {"hs_known", "hs_talent", "hs_alumni_n_log", "hs_alumni_min_share", "hs_top2_share"} <= set(s.static_names)
    assert "hs_region" in s.static_categorical
    v8 = feature_schema("v8")
    assert not set(v8.static_names) & {"hs_talent"} and set(v8.static_names) <= set(s.static_names)


def test_alumni_only_block_has_no_school_indicators():
    s = feature_schema("v9", ["hs_alumni"])
    assert {"hs_talent", "hs_alumni_n_log", "hs_alumni_min_share", "hs_alumni_level"} <= set(s.static_names)
    assert "hs_known" not in s.static_names and "hs_region" not in s.static_categorical
    assert "hs_alumni_level" in s.static_names and "hs_top2_share" not in s.static_names


def test_decay_keeps_plain_count_and_level_uses_earlier_seasons():
    seasons = pd.DataFrame({"player_id": [1, 1, 2], "hs_id": [1, 1, 1], "season": [2010, 2011, 2012],
                            "min": [100.0, 300.0, 200.0], "team_em": [10.0, 20.0, -5.0]})
    t = school_scores_decay(_talent(), _entries(), k=0.0, half_life=None, alumni_seasons=seasons).set_index("player_id")
    # player 10 enters 2012: alumni seasons 2010-2011 only (minutes-weighted: (100*10 + 300*20) / 400 = 17.5)
    assert np.isclose(t.loc[10, "hs_alumni_level"], 17.5) and t.loc[10, "hs_alumni_n"] == 1
    # player 11 enters 2013: all three seasons ((1000 + 6000 - 1000) / 600 = 10)
    assert np.isclose(t.loc[11, "hs_alumni_level"], 10.0) and t.loc[11, "hs_alumni_n"] == 2
    d = school_scores_decay(_talent(), _entries(), k=0.0, half_life=1.0, alumni_seasons=seasons).set_index("player_id")
    assert d.loc[11, "hs_alumni_n"] == 2                                   # the count stays plain under decay
    assert d.loc[11, "hs_talent"] < 0.5 + 1e-9                              # recent alumnus (0.1) outweighs the older (0.9)


def test_first_year_only_masks_later_seasons():
    hs = school_scores(_talent(), _entries(), k=0.0)
    units = pd.DataFrame({"player_id": [11, 11], "season": [2013, 2014], "first_year": [True, False]})
    out = attach_hs_block(units, hs, first_year_only=True).set_index("season")
    assert out.loc[2013, "hs_known"] == 1.0 and out.loc[2014, "hs_known"] == 0.0 and np.isnan(out.loc[2014, "hs_talent"])
