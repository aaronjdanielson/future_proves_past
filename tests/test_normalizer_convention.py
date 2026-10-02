"""D-078: a rebuilt run must use the normalization convention it was trained with. Manifests after the D-072 fix
record it; earlier manifests are judged by the run's completion time, which is exact because no strict control was
training across the fix."""
import json
import tempfile
import unittest
from pathlib import Path

from fpp.experiments.report import NORMALIZER_FIX_TIME, normalizer_exclusion


class NormalizerConventionTests(unittest.TestCase):
    def test_recorded_flag_wins(self):
        self.assertTrue(normalizer_exclusion({"config": {"normalizer_exclusion": True}}, None))
        self.assertFalse(normalizer_exclusion({"config": {"normalizer_exclusion": False}}, None))

    def test_completion_time_decides_for_older_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "status.json").write_text(json.dumps({"stage": "done", "updated_at": "2026-09-29T08:33:35"}))
            self.assertFalse(normalizer_exclusion({"config": {}}, run))            # the registered 2025-26 control
            (run / "status.json").write_text(json.dumps({"stage": "done", "updated_at": "2026-10-01T10:20:45"}))
            self.assertTrue(normalizer_exclusion({"config": {}}, run))             # the v11 deployment control
            (run / "status.json").write_text(json.dumps({"stage": "done", "updated_at": NORMALIZER_FIX_TIME}))
            self.assertTrue(normalizer_exclusion({"config": {}}, run))

    def test_no_evidence_means_the_old_convention(self):
        self.assertFalse(normalizer_exclusion({"config": {}}, None))
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(normalizer_exclusion({"config": {}}, Path(d)))      # no status file


if __name__ == "__main__":
    unittest.main()
