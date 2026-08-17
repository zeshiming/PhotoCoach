import unittest
from types import SimpleNamespace

from photo_coach.reliability import (
    RetryTracker,
    build_retry_policy,
    classify_retry_context,
)


class ReliabilityTests(unittest.TestCase):
    def test_network_error_is_retryable(self):
        tracker = RetryTracker()
        policy = build_retry_policy(tracker)
        context = SimpleNamespace(
            attempt=1,
            error=TimeoutError(),
            normalized=SimpleNamespace(
                is_network_error=True,
                is_timeout=False,
                status_code=None,
            ),
        )

        self.assertTrue(policy(context))
        self.assertEqual(tracker.retry_count, 1)
        self.assertEqual(tracker.failure_type, "network_error")

    def test_auth_error_is_not_retryable(self):
        tracker = RetryTracker()
        policy = build_retry_policy(tracker)
        context = SimpleNamespace(
            attempt=1,
            error=ValueError(),
            normalized=SimpleNamespace(
                is_network_error=False,
                is_timeout=False,
                status_code=401,
            ),
        )

        self.assertFalse(policy(context))
        self.assertEqual(classify_retry_context(context), "http_401")


if __name__ == "__main__":
    unittest.main()
