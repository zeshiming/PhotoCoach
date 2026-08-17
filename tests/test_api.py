import unittest

from fastapi.testclient import TestClient

from photo_coach.api import app


class ApiTests(unittest.TestCase):
    def test_health(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_chat_requires_text_or_image(self):
        response = TestClient(app).post("/api/v1/chat", data={"message": ""})
        self.assertEqual(response.status_code, 400)

    def test_sessions_endpoint(self):
        response = TestClient(app).get("/api/v1/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_session_messages_endpoint(self):
        response = TestClient(app).get("/api/v1/sessions/new-session/messages")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "new-session")


if __name__ == "__main__":
    unittest.main()
