import unittest

from photo_coach.tool_permissions import PermissionDecision, ToolPermissionPolicy


class ToolPermissionTests(unittest.TestCase):
    def test_read_only_tools_are_allowed(self):
        policy = ToolPermissionPolicy()
        self.assertEqual(policy.decision("analyze_current_image"), PermissionDecision.ALLOW)
        self.assertEqual(policy.decision("read_image_metadata"), PermissionDecision.ALLOW)

    def test_side_effect_tools_require_approval(self):
        policy = ToolPermissionPolicy()
        self.assertEqual(policy.decision("edit_image"), PermissionDecision.ASK)
        self.assertEqual(policy.decision("upload_external"), PermissionDecision.ASK)

    def test_unknown_tools_fail_closed_to_approval(self):
        policy = ToolPermissionPolicy()
        self.assertEqual(policy.decision("future_tool"), PermissionDecision.ASK)

    def test_high_risk_tools_are_denied(self):
        policy = ToolPermissionPolicy()
        self.assertEqual(policy.decision("execute_shell"), PermissionDecision.DENY)


if __name__ == "__main__":
    unittest.main()
