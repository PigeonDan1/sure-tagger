#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import cluster_scenes


MODEL_ID = cluster_scenes.DEFAULT_WER_INPUT_MODEL


def sample(error_rate=None, metric_name="wer"):
    valid = error_rate is not None
    return {
        "scores": {
            MODEL_ID: {
                "error_rate": error_rate,
                "error_valid": valid,
                "metric_name": metric_name if valid else None,
                "scoring_protocol": "test_protocol" if valid else None,
                "scoring_config_id": "test_config" if valid else None,
            }
        }
    }


class WerInputAxisTest(unittest.TestCase):
    def test_fit_uses_log1p_and_tracks_mixed_metrics(self):
        samples = [sample(0.0, "wer"), sample(0.5, "cer"), sample(1.0, "wer")]
        spec = cluster_scenes.fit_error_preprocess(samples, MODEL_ID)

        self.assertEqual(spec["n_valid"], 3)
        self.assertEqual(spec["metric_name_counts"], {"wer": 2, "cer": 1})
        self.assertEqual(spec["transform"], "log1p")
        self.assertAlmostEqual(spec["center"], math.log1p(0.5))

    def test_error_axis_uses_robust_z_distance_and_tau(self):
        spec = {"model_id": MODEL_ID, "center": 0.0, "spread": 0.5}
        actual = cluster_scenes.error_axis_distance(sample(0.0), sample(1.0), spec, 3.0)
        expected = min(math.log1p(1.0) / 0.5 / 3.0, 1.0)

        self.assertAlmostEqual(actual, expected)

    def test_error_axis_is_fifth_equal_weight_axis(self):
        spec = {"model_id": MODEL_ID, "center": 0.0, "spread": 1.0}
        axis_values = {
            "content": 0.1,
            "interaction": 0.2,
            "quality": 0.3,
            "spatial": 0.4,
        }
        with patch.object(
            cluster_scenes,
            "axis_distance",
            side_effect=lambda _a, _b, axis, _scales, _tau: axis_values[axis],
        ):
            actual = cluster_scenes.tag_distance(sample(0.0), sample(1.0), {}, 3.0, spec)

        error_distance = math.log1p(1.0) / 3.0
        self.assertAlmostEqual(actual, (0.1 + 0.2 + 0.3 + 0.4 + error_distance) / 5.0)

    def test_error_axis_missing_values_are_not_filled_with_zero(self):
        spec = {"model_id": MODEL_ID, "center": 0.0, "spread": 1.0}

        self.assertIsNone(cluster_scenes.error_axis_distance(sample(), sample(), spec, 3.0))
        self.assertEqual(
            cluster_scenes.error_axis_distance(sample(), sample(0.0), spec, 3.0),
            cluster_scenes.MISSING_WEIGHT_CAP,
        )


class AllSureTaggerLabelsTest(unittest.TestCase):
    def test_cli_flag_defaults_to_false_and_can_enable_all_labels(self):
        with patch.object(sys, "argv", ["cluster_scenes.py"]):
            args = cluster_scenes.parse_args()
        self.assertFalse(args.use_all_sure_tagger_labels)

        with patch.object(sys, "argv", ["cluster_scenes.py", "--use-all-tags"]):
            args = cluster_scenes.parse_args()
        self.assertTrue(args.use_all_sure_tagger_labels)

    def test_default_schema_remains_calibrated_subset(self):
        specs = cluster_scenes.default_feature_specs()

        self.assertFalse(specs["all_labels"])
        self.assertEqual(specs["axes"], cluster_scenes.AXES)
        self.assertNotIn("language_content.language", [path for path, _axis in specs["categorical"]])

    def test_all_labels_discovers_nested_and_list_features(self):
        rows = [
            {
                "basic_acoustic": {"duration_sec": 1.0, "channels": 1},
                "language_content": {"language": "en"},
                "speaker": {
                    "profiles": [
                        {"gender": "female", "speech_rate": {"value": 120.0}},
                        {"gender": "male", "speech_rate": {"value": 80.0}},
                    ]
                },
            },
            {
                "basic_acoustic": {"duration_sec": 2.0, "channels": 2},
                "language_content": {"language": "zh"},
                "speaker": {"profiles": None},
            },
        ]
        specs = cluster_scenes.discover_all_feature_specs(rows)
        feature = cluster_scenes.extract_tag_features(rows[0], specs)

        self.assertTrue(specs["all_labels"])
        self.assertIn("language", specs["axes"])
        self.assertIn(("basic_acoustic.channels", "spatial"), specs["numeric"])
        self.assertEqual(feature["categorical"]["language_content.language"], "en")
        self.assertEqual(
            feature["sets"]["speaker.profiles[].gender"], frozenset(("female", "male"))
        )
        self.assertEqual(feature["numeric"]["speaker.profiles.__count"], 2.0)

    def test_all_labels_change_distance_when_only_extra_label_differs(self):
        rows = [
            {"basic_acoustic": {"duration_sec": 1.0}, "language_content": {"language": "en"}},
            {"basic_acoustic": {"duration_sec": 1.0}, "language_content": {"language": "zh"}},
        ]
        specs = cluster_scenes.discover_all_feature_specs(rows)
        samples = [
            {"features": cluster_scenes.extract_tag_features(row, specs)} for row in rows
        ]
        scales = cluster_scenes.fit_numeric_preprocess(samples, specs)

        distance = cluster_scenes.tag_distance(
            samples[0], samples[1], scales, 3.0, feature_specs=specs
        )
        self.assertGreater(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
