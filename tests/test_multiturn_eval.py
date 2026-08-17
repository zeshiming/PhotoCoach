import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_multiturn_eval import evaluate_case, load_cases


class MultiTurnEvalTests(unittest.TestCase):
    def test_all_multiturn_cases_pass_without_network(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            results = [
                evaluate_case(case, Path(directory))
                for case in load_cases()
            ]

        self.assertEqual(len(results), 8)
        self.assertTrue(all(result["passed"] for result in results))


if __name__ == "__main__":
    unittest.main()
