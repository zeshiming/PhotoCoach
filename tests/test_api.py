import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from photo_coach.api import app


class ApiTests(unittest.TestCase):
    def test_health(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_frontend_is_served(self):
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("PhotoCoach", response.text)

    def test_chat_requires_text_or_image(self):
        response = TestClient(app).post("/api/v1/chat", data={"message": ""})
        self.assertEqual(response.status_code, 400)

    def test_chat_error_returns_id_and_logs_redacted_details(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "api_errors.jsonl"
            failing = AsyncMock(side_effect=RuntimeError("provider failed sk-secret-token"))
            with patch("photo_coach.api.run_photo_agent", failing), patch(
                "photo_coach.api.API_ERROR_LOG", log_path
            ):
                response = TestClient(app).post(
                    "/api/v1/chat",
                    data={"message": "怎么拍人像？", "session_id": "api-error-test"},
                )

            self.assertEqual(response.status_code, 502)
            self.assertIn("错误编号：", response.json()["detail"])
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("error_id", content)
            self.assertNotIn("sk-secret-token", content)

    def test_sessions_endpoint(self):
        response = TestClient(app).get("/api/v1/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_session_messages_endpoint(self):
        response = TestClient(app).get("/api/v1/sessions/new-session/messages")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "new-session")

    def test_delete_unknown_session_is_safe(self):
        response = TestClient(app).delete("/api/v1/sessions/not-found")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["deleted"])


if __name__ == "__main__":
    unittest.main()
