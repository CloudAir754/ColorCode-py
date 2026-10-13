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
                       enabled=True, matrix=MATRIX):
        processor = VideoProcessor(
            use_stored_transition_times=enabled,
            time_mapping_scope=scope,
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

    def test_initialization_output_contains_only_stage_times_and_differences(self):
        result, debug = self.initialize_image_preset()

        self.assertIn("Cache MISS: using ORIGINAL", debug)
        self.assertEqual(set(result), {"stages", "time_differences"})
        self.assertNotIn("name", result)
        self.assertNotIn("stretch", result)
        self.assertEqual(result["stages"], {
            "RGB": {"time": 0.0, "description": "一阶段RGB"},
            "RG": {"time": 1.5, "description": "二阶段RG"},
            "G": {"time": 4.0, "description": "三阶段G"},
        })
        self.assertEqual(result["time_differences"], {"t1": 1.5, "t2": 2.5, "unit": "s"})

    def test_image_preset_output_uses_stored_times_and_numbered_name(self):
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
            "name", "stretch", "preset_time_differences", "stretch_speed"
        })
        self.assertEqual(result["name"], "5D code42")
        self.assertEqual(result["stretch"], {
            "RGB": "α：90.00%",
            "RG": "α：115.00%",
            "G": "α：140.00%",
        })
        self.assertEqual(result["preset_time_differences"], {"t1": 1.5, "t2": 2.5, "unit": "s"})
        self.assertEqual(result["stretch_speed"], {
            "value": 12.5,
            "unit": "%/s",
            "display": "12.50 %/s",
        })

    def test_image_binding_keeps_different_matrices_separate(self):
        self.initialize_image_preset()
        different = self.make_processor(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
            matrix=OTHER_MATRIX,
        )

        result, _ = self.run_silently(different)

        self.assertEqual(set(result), {"stages", "time_differences"})
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

        self.assertEqual(set(first_result), {"stages", "time_differences"})
        self.assertIn("'program' binding", debug)
        self.assertIn("name", second_result)
        self.assertEqual(second_result["preset_time_differences"]["t1"], 1.0)
        self.assertEqual(second_result["preset_time_differences"]["t2"], 2.0)
        self.assertEqual(SAMPLE_TRANSITION_TIME_CACHE, {})

    def test_switch_off_uses_original_times_without_caching(self):
        processor = self.make_processor(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 1.0, 2: 1.1, 3: 1.2},
            enabled=False,
        )

        result, debug = self.run_silently(processor)

        self.assertIn("Switch OFF: using ORIGINAL", debug)
        self.assertEqual(set(result), {"stages", "time_differences"})
        self.assertEqual(SAMPLE_TRANSITION_TIME_CACHE, {})
        self.assertIsNone(PROGRAM_TRANSITION_TIME_PRESET["times"])


if __name__ == "__main__":
    unittest.main()
