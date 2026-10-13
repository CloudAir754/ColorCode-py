import unittest

from FlaskServer import display_test_config


class DisplayTestConfigTest(unittest.TestCase):
    def test_disabled_mode_returns_none(self):
        original_mode = display_test_config.DISPLAY_TEST_MODE
        try:
            display_test_config.DISPLAY_TEST_MODE = None
            self.assertIsNone(display_test_config.get_display_test_result())
        finally:
            display_test_config.DISPLAY_TEST_MODE = original_mode

    def test_initialization_mode_returns_initialization_schema(self):
        result = display_test_config.get_display_test_result("initialization")

        self.assertEqual(set(result["info"]), {"stages", "time_differences"})
        self.assertEqual(set(result["info"]["stages"]), {"RGB", "RG", "G"})

    def test_preset_mode_returns_preset_schema(self):
        result = display_test_config.get_display_test_result("preset")

        self.assertEqual(set(result["info"]), {
            "name", "stretch", "preset_time_differences", "stretch_speed"
        })

    def test_result_is_a_copy_of_the_editable_configuration(self):
        result = display_test_config.get_display_test_result("preset")
        result["info"]["name"] = "changed"

        self.assertNotEqual(
            result["info"]["name"],
            display_test_config.PRESET_TEST_INFO["name"],
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            display_test_config.get_display_test_result("unknown")


if __name__ == "__main__":
    unittest.main()
