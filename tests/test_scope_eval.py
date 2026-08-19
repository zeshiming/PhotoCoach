import asyncio
import unittest

from eval.run_scope_eval import CASES_PATH, load_cases, run


class ScopeEvalTests(unittest.TestCase):
    def test_dataset_has_all_categories(self):
        cases = load_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(
            {case["category"] for case in cases},
            {"in_scope", "out_of_scope", "sensitive", "clarify", "multiturn"},
        )

    def test_mock_evaluation_is_deterministic(self):
        report = asyncio.run(run("mock"))
        self.assertEqual(report["summary"]["total"], 30)
        self.assertEqual(report["summary"]["passed"], 30)
        self.assertEqual(report["summary"]["unsafe_allow_rate"], 0.0)

    def test_holdout_ids_are_disjoint_from_development_set(self):
        holdout_path = CASES_PATH.with_name("scope_eval_holdout.jsonl")
        development_ids = {case["id"] for case in load_cases(CASES_PATH)}
        holdout_cases = load_cases(holdout_path)
        self.assertEqual(len(holdout_cases), 26)
        self.assertTrue(
            development_ids.isdisjoint({case["id"] for case in holdout_cases})
        )


if __name__ == "__main__":
    unittest.main()
