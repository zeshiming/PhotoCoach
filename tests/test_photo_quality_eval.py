import unittest
from pathlib import Path

from eval.validate_photo_eval import load_cases, validate


class PhotoQualityEvalTests(unittest.TestCase):
    def test_dataset_has_40_cases_and_dev_holdout_split(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "photo_quality_eval.jsonl"
        report = validate(load_cases(path))
        self.assertEqual(report["total"], 40)
        self.assertEqual(report["dev"], 32)
        self.assertEqual(report["holdout"], 8)
        self.assertEqual(report["errors"], [])

    def test_local_image_references_exist(self):
        root = Path(__file__).resolve().parents[1] / "eval"
        path = Path(__file__).resolve().parents[1] / "eval" / "photo_quality_eval.jsonl"
        report = validate(load_cases(path), root)
        self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
