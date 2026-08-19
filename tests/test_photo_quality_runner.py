import unittest
from pathlib import Path

from eval.run_photo_quality_eval import load_cases, resolve_images


class PhotoQualityRunnerTests(unittest.TestCase):
    def test_loads_dev_and_holdout_cases(self):
        cases = load_cases()
        self.assertEqual(len(cases), 40)
        self.assertEqual(len(load_cases(split="dev")), 32)
        self.assertEqual(len(load_cases(split="holdout")), 8)

    def test_resolves_images_inside_eval_root(self):
        paths = resolve_images(["assets/outdoors-man-portrait.jpg"])
        self.assertTrue(paths[0].endswith("eval/assets/outdoors-man-portrait.jpg"))
        self.assertTrue(Path(paths[0]).is_file())


if __name__ == "__main__":
    unittest.main()
