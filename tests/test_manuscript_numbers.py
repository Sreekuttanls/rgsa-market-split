"""The manuscript's quoted numbers match the data, and the checker notices a
changed number or leftover revision wording (so a pass means something)."""

import tempfile
import unittest
from pathlib import Path

from src import verify_manuscript_numbers as v


@unittest.skipUnless((v.RES / "statistics.json").exists() and v.TEX.exists(),
                     "results or manuscript not present")
class TestManuscriptNumbers(unittest.TestCase):

    def run_on(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.tex"
            path.write_text(text)
            old, v.TEX = v.TEX, path
            try:
                return v.main()
            finally:
                v.TEX = old

    def test_manuscript_passes(self):
        self.assertEqual(v.main(), 0)

    def test_changed_number_is_caught(self):
        text = v.TEX.read_text()
        self.assertIn("better on 105 instances", text)
        self.assertEqual(self.run_on(text.replace("better on 105 instances",
                                                  "better on 106 instances")), 1)

    def test_revision_wording_is_caught(self):
        text = v.TEX.read_text()
        self.assertEqual(self.run_on(text + "\n% as the editor asked\n"), 1)


if __name__ == "__main__":
    unittest.main()
