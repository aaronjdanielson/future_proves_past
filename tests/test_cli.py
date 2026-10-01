from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from fpp.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_zero_schedule_end_to_end_and_write_once_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke.json"
            argv = ["--project", str(ROOT), "smoke", "--out", str(output),
                    "--schedule", "0", "--draws", "4", "--score-first", "1"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            data = json.loads(output.read_text())
            self.assertEqual(data["summary"]["mean_recorded_minutes"], 0)
            self.assertTrue(data["kernel_provenance"]["synthetic_only"])
            self.assertEqual(data["scored_draws"][0]["score"]["log_prob"], 0)
            original = output.read_bytes()
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 1)
            self.assertEqual(output.read_bytes(), original)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.db"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--project", str(ROOT), "audit-db", "--db", str(path)]), 1)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
