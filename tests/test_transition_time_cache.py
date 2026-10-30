import io
import json
import unittest
from contextlib import redirect_stdout

from FlaskServer.video_processor import (
    PROGRAM_TRANSITION_TIME_PRESET,
    SAMPLE_TRANSITION_TIME_CACHE,
    TIME_MAPPING_SCOPE_IMAGE,
    TIME_MAPPING_SCOPE_PROGRAM,
    VideoProcessor,
)


MATRIX = [
    ["Green", "Red", "Blue"],
    ["Red", "Green", "Blue"],
    ["Blue", "Red", "Green"],
]
OTHER_MATRIX = [
    ["Green", "Blue", "Blue"],
    ["Red", "Green", "Blue"],
    ["Blue", "Red", "Green"],
]


def stage_data(timestamp, ratio, frame_number, matrix=MATRIX):
    return {
        "color_matrix": matrix,
        "stretch_ratio": ratio,
        "frame_info": {"frame_number": frame_number, "timestamp": timestamp},
    }


class TransitionTimeCacheTest(unittest.TestCase):
    def setUp(self):
        SAMPLE_TRANSITION_TIME_CACHE.clear()
        PROGRAM_TRANSITION_TIME_PRESET["times"] = None

    def make_processor(self, times, ratios, scope=TIME_MAPPING_SCOPE_IMAGE,
                       enabled=True, matrix=MATRIX, material_length_mm=None):
        processor = VideoProcessor(
            use_stored_transition_times=enabled,
            time_mapping_scope=scope,
            material_length_mm=material_length_mm,
        )
        processor.stage_transitions = {
            stage: stage_data(times[stage], ratios[stage], stage * 10, matrix)
            for stage in (1, 2, 3)
        }
        processor.stretch_samples = [
            {
                "timestamp": times[stage],
                "frame_number": stage * 10,
                "stretch_ratio": ratios[stage],
                "color_matrix": matrix,
            }
            for stage in (1, 2, 3)
        ]
        return processor

    def run_silently(self, processor):
        debug = io.StringIO()
        with redirect_stdout(debug):
            result = processor.get_transition_info()
        json.dumps(result, ensure_ascii=False)
        return result, debug.getvalue()

    def initialize_image_preset(self):
        first = self.make_processor(
            {1: 2.0, 2: 3.5, 3: 6.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
        )
        return self.run_silently(first)

    def test_record_output_contains_stage_times_and_differences(self):
        result, debug = self.initialize_image_preset()

        self.assertIn("Cache MISS: using ORIGINAL", debug)
        self.assertEqual(set(result), {"mode", "stages", "time_differences"})
        self.assertEqual(result["mode"], "record")
        self.assertNotIn("name", result)
        self.assertEqual(result["stages"], {
            "T0": {"time": 0.0, "color": "RGB"},
            "T1": {"time": 1.5, "color": "RG"},
            "T2": {"time": 2.5, "color": "G"},
        })
        self.assertEqual(result["time_differences"], {
            "delta_t1": 1.5,
            "delta_t2": 2.5,
            "unit": "s",
        })

    def test_measure_output_uses_stored_times_name_percent_and_speed(self):
        self.initialize_image_preset()
        second = self.make_processor(
            {1: 10.0, 2: 15.0, 3: 20.0},
            {1: 0.9, 2: 1.5, 3: 2.0},
        )
        second.stretch_samples.extend([
            {"timestamp": 11.5, "frame_number": 115, "stretch_ratio": 1.15, "color_matrix": MATRIX},
            {"timestamp": 14.0, "frame_number": 140, "stretch_ratio": 1.4, "color_matrix": MATRIX},
        ])

        result, debug = self.run_silently(second)

        self.assertIn("Cache HIT: using STORED", debug)
        self.assertEqual(set(result), {
            "mode", "name", "stretch_percent", "time_differences", "speed"
        })
        self.assertEqual(result["mode"], "measure")
        self.assertEqual(result["name"], "5D code42")
        self.assertEqual(result["stretch_percent"], {
            "alpha0": 90.0,
            "alpha1": 115.0,
            "alpha2": 140.0,
        })
        self.assertEqual(result["time_differences"], {
            "delta_t1": 1.5,
            "delta_t2": 2.5,
            "unit": "s",
            "known": True,
        })
        # v = (1.40 - 0.90) × 50 mm ÷ 4.0 s × 60 = 375.0 mm/min
        self.assertEqual(result["speed"], {
            "value": 375.0,
            "unit": "mm/min",
            "material_length_mm": 50.0,
        })

    def test_speed_uses_configured_material_length(self):
        self.initialize_image_preset()
        second = self.make_processor(
            {1: 10.0, 2: 11.0, 3: 12.0},
            {1: 1.0, 2: 1.2, 3: 1.4},
            material_length_mm=25.0,
        )

        result, _ = self.run_silently(second)

        # 落点：10.0、11.5、14.0 s，分别取到 10.0、11.0、12.0 s 的采样。
        # v = (1.40 - 1.00) × 25 mm ÷ 4.0 s × 60 = 150.0 mm/min
        self.assertEqual(result["speed"]["value"], 150.0)
        self.assertEqual(result["speed"]["material_length_mm"], 25.0)

    def test_material_length_must_be_positive(self):
        with self.assertRaises(ValueError):
            VideoProcessor(material_length_mm=0)
        with self.assertRaises(ValueError):
            VideoProcessor(material_length_mm="not-a-number")

    def test_image_binding_keeps_different_matrices_separate(self):
        self.initialize_image_preset()
        different = self.make_processor(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
            matrix=OTHER_MATRIX,
        )

        result, _ = self.run_silently(different)

        self.assertEqual(set(result), {"mode", "stages", "time_differences"})
        self.assertEqual(len(SAMPLE_TRANSITION_TIME_CACHE), 2)

    def test_program_binding_reuses_first_preset_for_a_different_matrix(self):
        first = self.make_processor(
            {1: 2.0, 2: 3.0, 3: 5.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
            scope=TIME_MAPPING_SCOPE_PROGRAM,
        )
        first_result, _ = self.run_silently(first)

        second = self.make_processor(
            {1: 10.0, 2: 20.0, 3: 30.0},
            {1: 1.0, 2: 1.5, 3: 2.0},
            scope=TIME_MAPPING_SCOPE_PROGRAM,
            matrix=OTHER_MATRIX,
        )
        second.stretch_samples.extend([
            {"timestamp": 11.0, "frame_number": 110, "stretch_ratio": 1.2, "color_matrix": OTHER_MATRIX},
            {"timestamp": 13.0, "frame_number": 130, "stretch_ratio": 1.6, "color_matrix": OTHER_MATRIX},
        ])
        second_result, debug = self.run_silently(second)

        self.assertEqual(set(first_result), {"mode", "stages", "time_differences"})
        self.assertIn("'program' binding", debug)
        self.assertEqual(second_result["mode"], "measure")
        self.assertIn("name", second_result)
        self.assertEqual(second_result["time_differences"]["delta_t1"], 1.0)
        self.assertEqual(second_result["time_differences"]["delta_t2"], 2.0)
        self.assertEqual(SAMPLE_TRANSITION_TIME_CACHE, {})

    def test_switch_off_uses_original_times_without_caching(self):
        processor = self.make_processor(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
            enabled=False,
        )

        result, debug = self.run_silently(processor)

        self.assertIn("Switch OFF: using ORIGINAL", debug)
        self.assertEqual(set(result), {"mode", "stages", "time_differences"})
        self.assertEqual(result["mode"], "record")
        self.assertEqual(SAMPLE_TRANSITION_TIME_CACHE, {})
        self.assertIsNone(PROGRAM_TRANSITION_TIME_PRESET["times"])


if __name__ == "__main__":
    unittest.main()
