import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import score_benchmark_v1 as scorer


class SilenceRatioTest(unittest.TestCase):
    def test_ami_turn_complement_is_the_official_ratio(self):
        values = scorer.ami_speaker_values(
            [
                {"speaker": "A", "start": 0.0, "end": 2.0},
                {"speaker": "B", "start": 3.0, "end": 5.0},
            ],
            10.0,
        )
        self.assertEqual(values["basic_acoustic.nonspeech_ratio"], 0.6)

    def test_accepts_silence_ratio_prediction_alias(self):
        state, value = scorer.prediction_state(
            {"basic_acoustic": {"silence_ratio": 0.25}},
            "basic_acoustic.nonspeech_ratio",
            scorer.FIELD_SPECS["basic_acoustic.nonspeech_ratio"],
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, 0.25)

    def test_canonical_prediction_takes_precedence_over_alias(self):
        state, value = scorer.prediction_state(
            {"basic_acoustic": {"nonspeech_ratio": 0.2, "silence_ratio": 0.8}},
            "basic_acoustic.nonspeech_ratio",
            scorer.FIELD_SPECS["basic_acoustic.nonspeech_ratio"],
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, 0.2)

    def test_scores_absolute_ratio_without_tolerance_divisor(self):
        spec = scorer.FIELD_SPECS["basic_acoustic.nonspeech_ratio"]
        self.assertEqual(spec["metric"], "absolute_ratio")
        self.assertEqual(
            scorer.unit_score("absolute_ratio", "result_known", 1.0, 1.0, spec),
            2.0,
        )
        self.assertAlmostEqual(
            scorer.unit_score("absolute_ratio", "result_known", 0.2, 0.3, spec),
            1.8,
        )
        self.assertEqual(
            scorer.unit_score("absolute_ratio", "result_known", 0.0, 1.0, spec),
            0.0,
        )
        self.assertEqual(
            scorer.unit_score("absolute_ratio", "result_null", None, 0.3, spec),
            1.0,
        )

    def test_overlap_ratio_scores_without_tolerance_divisor(self):
        spec = scorer.FIELD_SPECS["speaker.overlap_ratio"]
        self.assertEqual(spec["metric"], "absolute_ratio")
        self.assertNotIn("tolerance", spec)
        self.assertAlmostEqual(
            scorer.unit_score("absolute_ratio", "result_known", 0.2, 0.3, spec),
            1.8,
        )

    def test_other_continuous_metrics_keep_their_tolerance(self):
        spec = scorer.FIELD_SPECS["deterministic_derived.filler_ratio"]
        self.assertEqual(spec["metric"], "continuous")
        self.assertEqual(spec["tolerance"], 0.05)


class NormalizedAbsoluteScoreTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"metric": "normalized_absolute"}

    def score(self, state, predicted, reference):
        return scorer.unit_score(
            "normalized_absolute", state, predicted, reference, self.spec
        )

    def test_exact_null_and_invalid_states(self):
        self.assertEqual(self.score("result_known", 12, 12), 2.0)
        self.assertEqual(self.score("result_known", 0, 0), 2.0)
        self.assertEqual(self.score("result_null", None, 12), 1.0)
        self.assertEqual(self.score("result_missing", None, 12), 0.0)

    def test_uses_symmetric_normalized_absolute_error(self):
        self.assertAlmostEqual(self.score("result_known", 8, 10), 1.6)
        self.assertAlmostEqual(self.score("result_known", 10, 8), 1.6)
        self.assertAlmostEqual(self.score("result_known", 1, 2), 1.0)
        self.assertEqual(self.score("result_known", 0, 3), 0.0)
        self.assertEqual(self.score("result_known", 3, 0), 0.0)

    def test_supports_nonnegative_rates(self):
        self.assertAlmostEqual(self.score("result_known", 7.5, 10.0), 1.5)
        self.assertAlmostEqual(self.score("result_known", 10.0, 7.5), 1.5)

    def test_reports_exact_and_absolute_error_diagnostics(self):
        path = "language_content.word_count"
        items = [
            {
                "gt": {"language": "en"},
                "state": "result_known",
                "predicted": 8,
                "reference": 10,
            },
            {
                "gt": {"language": "en"},
                "state": "result_known",
                "predicted": 12,
                "reference": 12,
            },
            {
                "gt": {"language": "en"},
                "state": "result_null",
                "predicted": None,
                "reference": 4,
            },
        ]
        result = scorer.raw_sample_result(items, path)
        self.assertAlmostEqual(result["score_rate"], 0.766667)
        self.assertEqual(result["auxiliary"]["exact_accuracy_known"], 0.5)
        self.assertAlmostEqual(result["auxiliary"]["mae_known"], 1.0)
        self.assertAlmostEqual(
            result["auxiliary"]["mean_normalized_absolute_error_known"], 0.1
        )


class LayerAggregationTest(unittest.TestCase):
    def test_excluded_lexical_labels_are_not_scored(self):
        excluded = {
            "language_content.filler",
            "language_content.punctuation.punctuation_count",
            "language_content.punctuation.has_terminal_punctuation",
        }
        self.assertTrue(excluded.isdisjoint(scorer.CAPABILITY_FIELDS))
        self.assertTrue(all(scorer.FIELD_SPECS[path]["role"] == "diagnostic" for path in excluded))
        self.assertTrue(
            excluded.isdisjoint(
                set(path for _, _, fields in scorer.LAYER_DEFINITIONS for path in fields)
            )
        )

    def test_averages_groups_per_label_before_averaging_labels(self):
        groups = {
            "group_a": {
                "fields": {
                    "speaker.asr_transcript": {
                        "score_sum_raw": 0.0,
                        "max_score_raw": 2.0,
                    },
                    "language_content.language": {
                        "score_sum_raw": 0.0,
                        "max_score_raw": 2.0,
                    },
                }
            },
            "group_b": {
                "fields": {
                    "speaker.asr_transcript": {
                        "score_sum_raw": 2.0,
                        "max_score_raw": 2.0,
                    }
                }
            },
        }

        layer = scorer.score_layers(groups)["layer_1"]

        self.assertEqual(layer["defined_label_count"], 2)
        self.assertEqual(
            layer["labels"]["speaker.asr_transcript"]["score_rate"], 0.5
        )
        self.assertEqual(
            layer["labels"]["language_content.language"]["score_rate"], 0.0
        )
        self.assertEqual(layer["score_rate"], 0.25)


class SpeakerGenderScoreTest(unittest.TestCase):
    path = scorer.SPEAKER_GENDER_FIELD

    def item(self, reference, state="result_known", predicted=None):
        if predicted is None and state == "result_known":
            predicted = reference
        return {
            "gt": {},
            "state": state,
            "predicted": predicted,
            "reference": tuple(reference),
        }

    def test_uses_reference_speaker_count_for_wer_normalization(self):
        result = scorer.gender_sequence_result(
            [self.item(("female", "male", "male"), predicted=("female", "male"))],
            self.path,
        )
        self.assertEqual(result["auxiliary"]["edit_distance_known"], 1)
        self.assertEqual(result["auxiliary"]["deletions_known"], 1)
        self.assertAlmostEqual(result["score_sum"], 4.0 / 3.0, places=6)
        self.assertAlmostEqual(result["score_rate"], 2.0 / 3.0, places=6)

    def test_reports_substitution_insertion_and_exact_sequence(self):
        result = scorer.gender_sequence_result(
            [
                self.item(("female",), predicted=("male",)),
                self.item(("male",), predicted=("male", "female")),
            ],
            self.path,
        )
        auxiliary = result["auxiliary"]
        self.assertEqual(auxiliary["substitutions_known"], 1)
        self.assertEqual(auxiliary["insertions_known"], 1)
        self.assertEqual(auxiliary["exact_sequence_accuracy_known"], 0.0)
        self.assertEqual(result["score_sum"], 0.0)

    def test_prediction_sequence_is_sorted_by_numeric_speaker_id(self):
        prediction = {
            "descriptive": {
                "speaker_descriptions": [
                    {"speaker_id": "speaker_2", "gender": "male"},
                    {"speaker_id": "speaker_1", "gender": "female"},
                ]
            }
        }
        state, value = scorer.prediction_state(
            prediction, self.path, scorer.FIELD_SPECS[self.path]
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, ("female", "male"))

    def test_missing_and_null_gender_are_distinct_states(self):
        missing = {"descriptive": {"speaker_descriptions": [{"speaker_id": "speaker_1"}]}}
        null = {"descriptive": {"speaker_descriptions": None}}
        self.assertEqual(
            scorer.prediction_state(missing, self.path, scorer.FIELD_SPECS[self.path])[0],
            "result_missing",
        )
        self.assertEqual(
            scorer.prediction_state(null, self.path, scorer.FIELD_SPECS[self.path])[0],
            "result_null",
        )

    def test_profile_gender_is_legacy_fallback_when_descriptive_is_absent(self):
        prediction = {
            "speaker": {
                "profiles": [
                    {"speaker_id": "speaker_2", "gender": "male"},
                    {"speaker_id": "speaker_1", "gender": "female"},
                ]
            }
        }
        state, value = scorer.prediction_state(
            prediction, self.path, scorer.FIELD_SPECS[self.path]
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, ("female", "male"))

    def test_profile_gender_precedes_generated_null_gender(self):
        prediction = {
            "speaker": {
                "profiles": [{"speaker_id": "speaker_1", "gender": "female"}]
            },
            "descriptive": {
                "speaker_descriptions": [
                    {"speaker_id": "speaker_1", "gender": None}
                ]
            },
        }
        state, value = scorer.prediction_state(
            prediction, self.path, scorer.FIELD_SPECS[self.path]
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, ("female",))

    def test_empty_profile_gender_sequence_is_preserved(self):
        prediction = {
            "speaker": {"profiles": []},
            "descriptive": {"speaker_descriptions": None},
        }
        state, value = scorer.prediction_state(
            prediction, self.path, scorer.FIELD_SPECS[self.path]
        )
        self.assertEqual(state, "result_known")
        self.assertEqual(value, ())

    def test_gender_is_a_layer_two_capability(self):
        self.assertIn(self.path, scorer.CAPABILITY_FIELDS)
        layer = dict((layer_id, fields) for layer_id, _, fields in scorer.LAYER_DEFINITIONS)
        self.assertIn(self.path, layer["layer_2"])


if __name__ == "__main__":
    unittest.main()
