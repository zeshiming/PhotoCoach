import asyncio
import unittest

from eval.run_capability_eval import load_cases, run


class CapabilityEvalTests(unittest.TestCase):
    def test_mock_capability_eval_passes(self):
        report = asyncio.run(run("mock"))
        self.assertEqual(len(load_cases()), 12)
        self.assertTrue(report["summary"]["all_checks_passed"])
        self.assertEqual(report["summary"]["capability_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
