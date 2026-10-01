import json
from pathlib import Path
import shutil
import tempfile
import unittest

from fpp.specification import verify_specification


ROOT = Path(__file__).resolve().parents[1]


class SpecificationTests(unittest.TestCase):
    def test_actual_source_matches_frozen_decision(self):
        self.assertTrue(verify_specification(ROOT)["verified"])

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["config/protocol.json", "paper/international_ncaa_translation.tex",
                         "IMPLEMENTATION_PLAN.md", "DECISIONS.md"]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            source = root / "paper/international_ncaa_translation.tex"
            source.write_text(source.read_text() + "\n% unregistered change\n")
            with self.assertRaisesRegex(ValueError, "hash disagrees"):
                verify_specification(root)


if __name__ == "__main__":
    unittest.main()
