import tempfile
import unittest
from pathlib import Path

from photo_coach.scope_memory import ScopeMemoryStore
from photo_coach.scope_models import ScopeDecision, ScopeResult


class ScopeMemoryTests(unittest.TestCase):
    def _result(self, decision="allow", reason_code="photography_question"):
        return ScopeResult(
            decision=ScopeDecision(decision),
            scope="in_scope" if decision == "allow" else "restricted",
            risk_level="low" if decision == "allow" else "high",
            reason_code=reason_code,
            message="test",
        )

    def test_record_and_read_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScopeMemoryStore(Path(directory) / "agent_sessions.db")
            state = store.record(
                "s1",
                self._result(),
                goal="帮助用户学习摄影",
                last_image_id="image#1",
            )
            self.assertEqual(state.goal, "帮助用户学习摄影")
            self.assertEqual(store.get("s1").last_image_id, "image#1")
            self.assertIn("会话目标", store.summary("s1"))

    def test_denied_topic_is_retained_across_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScopeMemoryStore(Path(directory) / "agent_sessions.db")
            store.record("s1", self._result(), goal="摄影分析")
            store.record(
                "s1",
                self._result("refuse", "sensitive_inference"),
            )
            state = store.get("s1")
            self.assertEqual(state.goal, "摄影分析")
            self.assertEqual(state.denied_topics, ["sensitive_inference"])

    def test_state_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "agent_sessions.db"
            ScopeMemoryStore(db_path).record("s1", self._result(), goal="摄影")
            reopened = ScopeMemoryStore(db_path)
            self.assertEqual(reopened.get("s1").goal, "摄影")

    def test_clear_removes_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScopeMemoryStore(Path(directory) / "agent_sessions.db")
            store.record("s1", self._result(), goal="摄影")
            store.clear("s1")
            self.assertIsNone(store.get("s1"))


if __name__ == "__main__":
    unittest.main()
