"""The QOBLIB checker used by src.check_exact_solutions accepts a known
solution and rejects a corrupted one (so a passing check means something)."""

import shutil
import subprocess
import unittest

from src.check_exact_solutions import CHECK_DIR, CHECKER, INST

SOL = CHECK_DIR.parent / "solutions" / "ms_03_050_002.opt.sol"


def read_bits(path):
    """The .opt.sol files hold the solution as space-separated 0/1 values."""
    return "".join(t for t in path.read_text().split() if t in ("0", "1"))


def available():
    if CHECKER.exists():
        return True
    if shutil.which("cargo") is None:
        return False
    return subprocess.run(["cargo", "build", "--release"], cwd=CHECK_DIR,
                          capture_output=True).returncode == 0


@unittest.skipUnless(available() and SOL.exists(), "QOBLIB checker or solution file unavailable")
class TestQoblibChecker(unittest.TestCase):

    def run_check(self, bits):
        return subprocess.run([str(CHECKER), str(INST / "ms_03_050_002.dat"), bits],
                              capture_output=True, text=True).returncode

    def test_known_solution_accepted(self):
        self.assertEqual(self.run_check(read_bits(SOL)), 0)

    def test_corrupted_solution_rejected(self):
        bits = read_bits(SOL)
        flipped = ("1" if bits[0] == "0" else "0") + bits[1:]
        self.assertNotEqual(self.run_check(flipped), 0)


if __name__ == "__main__":
    unittest.main()
