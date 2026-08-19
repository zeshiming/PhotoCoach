import unittest

from photo_coach.context_compaction import CompactionConfig, ContextCompactor


class ContextCompactionTests(unittest.TestCase):
    def test_compaction_keeps_recent_items_and_summary(self):
        compactor = ContextCompactor(
            CompactionConfig(max_tokens=10, keep_recent_items=2, chars_per_token=1)
        )
        items = [
            {"role": "user", "content": "第一轮摄影问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "最后一轮问题"},
        ]
        result = compactor.compact(items, "之前讨论了曝光")
        self.assertTrue(result.compacted)
        self.assertEqual(result.removed_items, 1)
        self.assertIn("之前讨论了曝光", result.items[0]["content"])
        self.assertEqual(result.items[-1]["content"], "最后一轮问题")

    def test_small_context_is_not_changed(self):
        compactor = ContextCompactor(CompactionConfig(max_tokens=1000))
        items = [{"role": "user", "content": "构图"}]
        result = compactor.compact(items)
        self.assertFalse(result.compacted)
        self.assertEqual(result.items, items)


if __name__ == "__main__":
    unittest.main()
