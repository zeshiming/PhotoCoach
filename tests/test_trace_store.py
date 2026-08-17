import json
import tempfile
import unittest
from pathlib import Path

from photo_coach.trace_store import JsonlTraceStore, TraceRecord


class TraceStoreTests(unittest.TestCase):
    def test_appends_trace_without_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            store = JsonlTraceStore(path)
            store.append(
                TraceRecord(
                    trace_id="t1",
                    session_id="s1",
                    model="test-model",
                    status="success",
                    duration_ms=123,
                    usage={"total_tokens": 10},
                )
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["trace_id"], "t1")
            self.assertNotIn("api_key", record)


if __name__ == "__main__":
    unittest.main()
