import tempfile
import unittest
from pathlib import Path

from photo_coach.image_session_store import ImageSessionStore


class ImageSessionStoreTests(unittest.TestCase):
    def test_register_get_and_list(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ImageSessionStore(Path(directory) / "agent_sessions.db")
            image_id = store.register("s1", "/tmp/photo.jpg", 1)
            self.assertEqual(image_id, "image#1")
            self.assertEqual(store.get("s1", image_id).ref, "/tmp/photo.jpg")
            self.assertEqual(len(store.list("s1")), 1)

    def test_second_turn_without_new_image_resolves_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ImageSessionStore(Path(directory) / "agent_sessions.db")
            image_id = store.register("s1", "/tmp/photo.jpg", 1)
            self.assertEqual(store.resolve_reference("s1", "那光线怎么样"), image_id)

    def test_persists_after_reopening_store(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "agent_sessions.db"
            ImageSessionStore(db_path).register("s1", "/tmp/photo.jpg", 1)
            reopened = ImageSessionStore(db_path)
            self.assertEqual(reopened.latest("s1").image_id, "image#1")

    def test_invalid_local_path_is_detectable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ImageSessionStore(Path(directory) / "agent_sessions.db")
            record = store.get("s1", store.register("s1", "/tmp/missing.jpg", 1))
            self.assertIsNotNone(record)
            self.assertFalse(Path(record.ref).is_file())


if __name__ == "__main__":
    unittest.main()
