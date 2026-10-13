"""Plus field migration and language feedback checks with no API calls."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import extract_captioner_deepseek_once as validator
import extract_sure_tagger_deepseek_once as sure
import run_deepseek_batch as batch


def structural():
    result = {
        group: {field: None for field in fields}
        for group, fields in sure.STRUCTURAL_FIELDS.items()
    }
    result["speaker"].update(
        speaker_count=2, speaker_present=True,
        asr_transcript="Hello this is a sufficiently long English transcript.",
        speaker_transcripts=[
            {"speaker_id": "speaker_1", "text": "Hello this is a sufficiently long"},
            {"speaker_id": "speaker_2", "text": None},
        ],
    )
    result["language_content"]["language"] = "fr"
    return result


def envelope(value):
    return {
        "sample_id": "sample", "schema_version": sure.SCHEMA_VERSION,
        "field_universe_version": sure.FIELD_UNIVERSE_VERSION,
        **sure.canonical_structural(value),
        "descriptive": {key: None for key in sure.DESCRIPTIVE_ORDER},
    }


class SpeakerTranscriptsMigrationTest(unittest.TestCase):
    def test_old_missing_field_is_unavailable_and_input_is_immutable(self):
        old = structural()
        del old["speaker"]["speaker_transcripts"]
        original = copy.deepcopy(old)
        normalized = batch._validate_and_canonicalize_structural(old, "sample")
        self.assertIsNone(normalized["speaker"]["speaker_transcripts"])
        self.assertEqual(old, original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tags.json"
            path.write_text(json.dumps(old))
            self.assertIsNone(sure.load_structural(path)["speaker"]["speaker_transcripts"])

    def test_new_array_survives_canonical_adapter_and_validation(self):
        value = structural()
        normalized = batch._validate_and_canonicalize_structural(value, "sample")
        self.assertEqual(normalized["speaker"]["speaker_transcripts"], value["speaker"]["speaker_transcripts"])
        validator.validate_payload(envelope(value), "sample")
        self.assertIn("speaker.speaker_transcripts", validator.STRUCTURAL_EVIDENCE_PATHS)
        self.assertNotIn("speaker.speaker_transcripts[0].text", validator.STRUCTURAL_EVIDENCE_PATHS)

    def test_invalid_keys_empty_text_duplicate_id_and_count_are_rejected(self):
        invalid = [
            [{"speaker_id": "speaker_1", "text": "hi", "confidence": 1}],
            [{"speaker_id": "speaker_1", "text": ""}, {"speaker_id": "speaker_2", "text": None}],
            [{"speaker_id": "speaker_1", "text": "hi"}] * 2,
            [{"speaker_id": "speaker_1", "text": "hi"}],
            [],
        ]
        for items in invalid:
            with self.subTest(items=items):
                value = structural()
                value["speaker"]["speaker_transcripts"] = items
                with self.assertRaises(validator.ValidationError):
                    validator.validate_payload(envelope(value), "sample")

    def test_language_conflict_is_feedback_and_descriptions_are_withheld(self):
        value = sure.canonical_structural(structural())
        raw = {
            "language_verification": {"language": "en", "confidence": 0.99, "reason_short": "English words"},
            "descriptive": {key: None for key in sure.DESCRIPTIVE_ORDER},
        }
        raw["descriptive"]["caption"] = "English speech."
        payload, row = batch._sure_payload(
            raw, value, "sample", require_language_dependency_review=True,
        )
        self.assertEqual(payload["language_content"]["language"], "fr")
        self.assertEqual(row["verified_language"], "en")
        self.assertEqual(row["verdict_state"], "pending_dependency_review")
        self.assertFalse(row["changed"])
        self.assertTrue(row["dependency_review_required"])
        self.assertTrue(all(item is None for item in payload["descriptive"].values()))
        self.assertEqual(payload["speaker"], value["speaker"])

    def test_after_upstream_language_audit_descriptions_can_publish(self):
        value = sure.canonical_structural(structural())
        value["language_content"]["language"] = "en"
        raw = {
            "language_verification": {"language": "en", "confidence": 0.99, "reason_short": "English words"},
            "descriptive": {key: None for key in sure.DESCRIPTIVE_ORDER},
        }
        raw["descriptive"]["caption"] = "English speech."
        payload, row = batch._sure_payload(raw, value, "sample", require_language_dependency_review=True)
        self.assertEqual(row["verdict_state"], "agreed")
        self.assertEqual(payload["descriptive"]["caption"], "English speech.")

    def test_legacy_language_override_is_retained_without_review_flag(self):
        value = structural()
        self.assertEqual(sure.resolve_language(value, {"language": "en"}), ("en", "changed", True, "en"))


if __name__ == "__main__":
    unittest.main()
