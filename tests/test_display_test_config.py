import unittest

from FlaskServer import display_test_config


EXPECTED_RECORD_PREVIEW = (
    "Image infomation:\n"
    "Details:\n"
    "T0: 0s RGB\n"
    "T1: 1.50s RG\n"
    "T2: 2.50s G\n"
    "ΔT1=1.50；ΔT2=2.50；"
)

EXPECTED_MEASURE_PREVIEW = (
    "Image infomation:\n"
    "Name: 5D code42\n"
    "Details:\n"
    "RGB α0：100.00%\n"
    "RG α1：127.00%\n"
    "G α2：145.00%\n"
    "ΔT1=1.50；ΔT2=2.50；（known）\n"
    "\n"
    "Speed:\n"
    "337.50 mm/min"
)


class DisplayTestConfigTest(unittest.TestCase):
    def test_disabled_mode_returns_none(self):
        original_mode = display_test_config.DISPLAY_TEST_MODE
        try:
            display_test_config.DISPLAY_TEST_MODE = None
            self.assertIsNone(display_test_config.get_display_test_result())
        finally:
            display_test_config.DISPLAY_TEST_MODE = original_mode

    def test_record_mode_returns_record_schema(self):
        result = display_test_config.get_display_test_result("record")

        self.assertEqual(set(result["info"]), {"mode", "stages", "time_differences"})
        self.assertEqual(result["info"]["mode"], "record")
        self.assertEqual(set(result["info"]["stages"]), {"T0", "T1", "T2"})
        self.assertEqual(result["info"]["time_differences"], {
            "delta_t1": 1.5,
            "delta_t2": 2.5,
            "unit": "s",
        })

    def test_measure_mode_returns_measure_schema(self):
        result = display_test_config.get_display_test_result("measure")

        self.assertEqual(set(result["info"]), {
            "mode", "name", "stretch_percent", "time_differences", "speed"
        })
        self.assertEqual(result["info"]["mode"], "measure")
        self.assertEqual(result["info"]["stretch_percent"], {
            "alpha0": 100.0,
            "alpha1": 127.0,
            "alpha2": 145.0,
        })
        self.assertEqual(result["info"]["speed"]["value"], 337.5)

    def test_legacy_mode_names_map_to_new_schemas(self):
        self.assertEqual(
            display_test_config.get_display_test_result("initialization")["info"]["mode"],
            "record",
        )
        self.assertEqual(
            display_test_config.get_display_test_result("preset")["info"]["mode"],
            "measure",
        )

    def test_record_preview_matches_android_template(self):
        self.assertEqual(
            display_test_config.render_preview("record"),
            EXPECTED_RECORD_PREVIEW,
        )

    def test_measure_preview_matches_android_template(self):
        self.assertEqual(
            display_test_config.render_preview("measure"),
            EXPECTED_MEASURE_PREVIEW,
        )

    def test_result_is_a_copy_of_the_editable_configuration(self):
        result = display_test_config.get_display_test_result("measure")
        result["info"]["name"] = "changed"

        self.assertNotEqual(
            result["info"]["name"],
            display_test_config.MEASURE_TEST_INFO["name"],
        )

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            display_test_config.get_display_test_result("unknown")


if __name__ == "__main__":
    unittest.main()
