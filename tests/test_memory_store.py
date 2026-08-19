import tempfile
import unittest
from pathlib import Path

from photo_coach.memory_store import LongTermMemoryStore, UserMemory


class LongTermMemoryStoreTests(unittest.TestCase):
    def test_explicit_memory_persists_and_is_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LongTermMemoryStore(Path(directory) / "agent_sessions.db")
            memory = store.remember_explicit("u1", "请记住我喜欢日系胶片风", "s1")
            self.assertIsNotNone(memory)
            results = store.search("u1", "我喜欢什么风格")
            self.assertEqual(results[0].memory_value, "我喜欢日系胶片风")

    def test_sensitive_memory_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LongTermMemoryStore(Path(directory) / "agent_sessions.db")
            self.assertIsNone(store.remember_explicit("u1", "请记住我的身份证号", "s1"))
            self.assertEqual(store.list("u1"), [])

    def test_session_summary_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "agent_sessions.db"
            LongTermMemoryStore(db_path).upsert_summary("s1", "讨论了人像曝光")
            reopened = LongTermMemoryStore(db_path)
            self.assertEqual(reopened.get_summary("s1").summary, "讨论了人像曝光")


if __name__ == "__main__":
    unittest.main()
