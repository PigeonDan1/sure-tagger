import json
from pathlib import Path
import sys
import unittest


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import adapt_outputs
import build_ami_gt
import extract_captioner_deepseek_once
import score_structural


UNIVERSE_PATH = EVAL_DIR / "field_universe.structural-v0.1.json"


def gt_row(sample_id, path, value):
    return {
        "sample_id": sample_id,
        "ground_truth": {
            "values": {path: value},
            "gt_mask": {path: True},
        },
    }


def prediction_row(sample_id, path, value, provided=True):
    return {
        "sample_id": sample_id,
        "status": "ok",
        "schema_valid": True,
        "values": {path: value},
        "provided_fields": [path] if provided else [],
    }


class GroundTruthRulesTest(unittest.TestCase):
    def setUp(self):
        self.universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))

    def test_speaker_threshold_merge_change_and_overlap(self):
        utterances = [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "A", "start": 1.2, "end": 2.0},
            {"speaker": "B", "start": 2.5, "end": 3.0},
            {"speaker": "C", "start": 2.6, "end": 3.2},
            {"speaker": "D", "start": 4.0, "end": 4.05},
        ]
        result = build_ami_gt.derive_speaker_fields(
            utterances, 5.0, self.universe["speaker_rule"]
        )
        self.assertEqual(result["speaker.speaker_count"], 3)
        self.assertTrue(result["speaker.multi_speaker"])
        self.assertEqual(result["speaker.speaker_change_count"], 2)
        self.assertTrue(result["speaker.speaker_change"])
        self.assertTrue(result["speaker.speaker_overlap"])
        self.assertAlmostEqual(result["speaker.overlap_ratio"], 0.4 / 2.7, places=6)

    def test_text_rules(self):
        result = build_ami_gt.derive_text_fields(
            "Um, go go! Yeah.", self.universe["text_rule"]
        )
        self.assertEqual(result["language_content.word_count"], 4)
        self.assertEqual(
            result["language_content.punctuation.punctuation_count"], 3
        )
        self.assertTrue(
            result["language_content.punctuation.has_terminal_punctuation"]
        )
        self.assertTrue(result["language_content.repetition.has_repetition"])
        self.assertEqual(result["language_content.repetition.repetition_count"], 1)
        self.assertEqual(result["language_content.filler"], 2)


class DenominatorPolicyTest(unittest.TestCase):
    def test_independent_derived_field_scores_when_dependency_is_null(self):
        universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        fields = score_structural.field_specs(universe)
        path = "speaker.speaker_present"
        ground_truth = gt_row("a", path, True)
        prediction = prediction_row("a", path, True)
        prediction["values"]["speaker.speaker_count"] = None
        prediction["provided_fields"].append("speaker.speaker_count")

        result = score_structural.score_split(
            [(ground_truth, prediction)], universe, fields
        )

        self.assertIn(path, result["capabilities"])
        self.assertEqual(result["capabilities"][path]["result_known"], 1)
        self.assertEqual(result["capabilities"][path]["accuracy_e2e"], 1.0)
        self.assertEqual(
            score_structural.contract_outcome(
                universe["contract_rules"][0], prediction, fields
            ),
            "dependency_null",
        )

    def test_numeric_derived_field_is_not_contract_only(self):
        universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        fields = score_structural.field_specs(universe)
        path = "deterministic_derived.filler_ratio"
        items = [(gt_row("a", path, 0.25), prediction_row("a", path, 0.2))]
        result = score_structural.score_split(items, universe, fields)
        self.assertAlmostEqual(result["capabilities"][path]["mae"], 0.05)

    def test_boolean_null_and_missing_remain_eligible(self):
        path = "speaker.multi_speaker"
        spec = {"kind": "boolean", "metric": "boolean"}
        items = [
            (gt_row("a", path, True), prediction_row("a", path, True)),
            (gt_row("b", path, False), prediction_row("b", path, None, False)),
            (gt_row("c", path, True), prediction_row("c", path, None, True)),
        ]
        result = score_structural.boolean_summary(items, path, spec)
        self.assertEqual(result["eligible_gt"], 3)
        self.assertEqual(result["result_known"], 1)
        self.assertEqual(result["result_missing"], 1)
        self.assertEqual(result["result_null"], 1)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertEqual(result["specificity"], 0.0)
        self.assertAlmostEqual(result["accuracy_e2e"], 1.0 / 3.0)

    def test_missing_asr_is_scored_as_empty_hypothesis(self):
        path = "speaker.asr_transcript"
        spec = {"kind": "string", "metric": "text_wer"}
        items = [
            (gt_row("a", path, "One two."), prediction_row("a", path, "one two")),
            (gt_row("b", path, "Three"), prediction_row("b", path, None, False)),
        ]
        result = score_structural.text_wer_summary(items, path, spec)
        self.assertEqual(result["eligible_gt"], 2)
        self.assertEqual(result["result_missing"], 1)
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["word_errors"], 1)
        self.assertAlmostEqual(result["wer"], 1.0 / 3.0)

    def test_fully_missing_field_is_marked_interface_unavailable(self):
        path = "speaker.speaker_change_count"
        spec = {"kind": "integer", "metric": "integer_exact"}
        items = [
            (gt_row("a", path, 1), prediction_row("a", path, None, False)),
            (gt_row("b", path, 0), prediction_row("b", path, None, False)),
        ]
        result = score_structural.numeric_summary(
            items, path, spec, exact_integer=True
        )
        self.assertEqual(result["eligible_gt"], 2)
        self.assertEqual(result["result_missing"], 2)
        self.assertEqual(result["interface_provided_count"], 0)
        self.assertTrue(result["interface_unavailable"])
        self.assertEqual(result["exact_accuracy_e2e"], 0.0)


class AdapterTest(unittest.TestCase):
    def test_legacy_captioner_mapping_is_not_native_schema_valid(self):
        universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        fields = adapt_outputs.field_specs(universe)
        source = {
            "sample_id": "sample",
            "annotation": [
                {
                    "audio": {"tag": ["speech", "noise"]},
                    "speaker": {
                        "speaker_count": 2,
                        "multi_speaker": True,
                        "speaker_change": True,
                        "speaker_overlap": False,
                        "profiles": [],
                    },
                }
            ],
        }
        result = adapt_outputs.adapt_one(
            "captioner", "sample", source, "source.json", fields, universe
        )
        self.assertTrue(result["adapter_schema_valid"])
        self.assertFalse(result["native_current_schema_valid"])
        self.assertEqual(result["source_format"], "legacy_captioner_annotation")
        self.assertEqual(
            result["values"]["sound_field_scene.speech_music_events"], ["speech"]
        )
        self.assertEqual(result["values"]["speaker.speaker_count"], 2)


class IndependentCoarseFieldValidationTest(unittest.TestCase):
    def test_coarse_and_derived_values_can_be_known_with_null_dependencies(self):
        payload = {
            "sample_id": "sample",
            "schema_version": "structural-prediction-v0.1",
            "field_universe_version": "structural-v0.1",
            "basic_acoustic": {
                "duration_sec": None,
                "sample_rate_hz": None,
                "channels": None,
                "nonspeech_ratio": None,
                "silence_segments": None,
            },
            "audio_quality": {
                "snr_db": None,
                "dnsmos_sig": None,
                "dnsmos_bak": None,
                "dnsmos_ovrl": None,
                "dnsmos_p808": None,
            },
            "room_acoustic": {
                "far_field": None,
                "rt60_sec": None,
                "c50_db": None,
            },
            "sound_field_scene": {
                "speech_music_events": None,
                "music_present": True,
                "external_noise_type": None,
                "noise_composition": None,
            },
            "speaker": {
                "speaker_count": None,
                "speaker_present": True,
                "asr_transcript": None,
                "multi_speaker": None,
                "speaker_change_count": None,
                "speaker_change": True,
                "overlap_ratio": None,
                "speaker_overlap": False,
                "profiles": None,
            },
            "language_content": {
                "language": None,
                "word_count": None,
                "punctuation": {
                    "punctuation_count": None,
                    "has_terminal_punctuation": True,
                },
                "repetition": {
                    "has_repetition": True,
                    "repetition_count": None,
                },
                "filler": None,
            },
            "deterministic_derived": {
                "speaker_change_rate_per_min": 2.5,
                "filler_ratio": 0.1,
            },
            "descriptive": {
                "caption": None,
                "topic": None,
                "background_condition": None,
                "dominant_content_impression": None,
                "clarity_intelligibility_impression": None,
                "background_sound_summary": None,
                "descriptive_phrases": None,
                "facets": None,
                "speaker_descriptions": None,
                "scene_hypotheses": None,
                "usability_notes": None,
                "uncertainties": None,
            },
        }

        extract_captioner_deepseek_once.validate_payload(payload, "sample")


if __name__ == "__main__":
    unittest.main()
