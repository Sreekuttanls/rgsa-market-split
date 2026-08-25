import json
import tempfile
import unittest
from pathlib import Path

from src.validate_publication import PublicationValidationError, validate_results


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_benchmarks" / "paper_benchmark_results.json"


class TestPublicationValidation(unittest.TestCase):
    def test_committed_results_have_expected_shape(self):
        summary = validate_results(RESULTS)
        self.assertEqual(summary["runs"], 160)
        self.assertEqual(summary["instances"], 8)
        self.assertEqual(summary["groups"], 32)
        self.assertEqual(summary["feasible_runs"], 0)

    def test_duplicate_run_is_rejected(self):
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        payload["runs"].append(dict(payload["runs"][0]))
        payload["num_runs"] += 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(PublicationValidationError):
                validate_results(handle.name)

    def test_feasibility_flag_must_match_residual(self):
        payload = json.loads(RESULTS.read_text(encoding="utf-8"))
        payload["runs"][0]["feasible"] = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaises(PublicationValidationError):
                validate_results(handle.name)


if __name__ == "__main__":
    unittest.main()
