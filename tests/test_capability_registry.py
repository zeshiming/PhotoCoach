import unittest

from photo_coach.capability_registry import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilitySpec,
)


class CapabilityRegistryTests(unittest.TestCase):
    def test_default_registry_maps_capabilities_to_tools(self):
        self.assertEqual(
            DEFAULT_CAPABILITY_REGISTRY.tool_names_for(["analyze_image"]),
            {"analyze_current_image"},
        )
        self.assertEqual(
            DEFAULT_CAPABILITY_REGISTRY.tool_names_for(["read_metadata"]),
            {"read_image_metadata"},
        )

    def test_unknown_capability_is_filtered(self):
        self.assertEqual(
            DEFAULT_CAPABILITY_REGISTRY.validate(["analyze_image", "unknown"]),
            ["analyze_image"],
        )

    def test_registry_can_register_direct_answer_capability(self):
        registry = type(DEFAULT_CAPABILITY_REGISTRY)()
        registry.register(
            CapabilitySpec(id="test_capability", description="test", tools=())
        )
        self.assertEqual(registry.tool_names_for(["test_capability"]), set())


if __name__ == "__main__":
    unittest.main()
