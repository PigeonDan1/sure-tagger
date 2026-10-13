import json
from pathlib import Path
import sys
import tempfile
import unittest


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_descriptive_semantic_scoring_batch as batch


def descriptive_payload():
    return {
        "caption": "speech in a quiet room",
        "topic": "planning",
        "background_condition": None,
        "dominant_content_impression": "speech",
        "clarity_intelligibility_impression": "clear",
        "background_sound_summary": None,
        "descriptive_phrases": ["clear speech"],
        "facets": [
            {
                "name": "interaction",
                "description": "two speakers",
                "confidence": 0.9,
                "evidence_tags": ["speaker.speaker_count"],
            }
        ],
        "speaker_descriptions": [
            {
                "speaker_id": "speaker_1",
                "gender": "female",
                "description": "steady voice",
                "descriptive_phrases": ["steady"],
                "confidence": 0.8,
                "evidence_tags": ["speaker.profiles"],
            }
        ],
        "scene_hypotheses": [
            {
                "description": "a meeting",
                "confidence": 0.7,
                "evidence_tags": ["speaker.asr_transcript"],
            }
        ],
        "uncertainties": ["exact room unknown"],
        "usability_notes": ["not for scoring"],
    }


class InputCleaningTest(unittest.TestCase):
    def test_clean_ground_truth_matches_prompt_contract(self):
        cleaned = batch.clean_descriptive(descriptive_payload(), True)
        self.assertEqual(set(cleaned), set(batch.SCORED_FIELDS) | {"uncertainties"})
        self.assertEqual(
            cleaned["facets"],
            [{"name": "interaction", "description": "two speakers"}],
        )
        self.assertNotIn("confidence", json.dumps(cleaned))
        self.assertNotIn("evidence_tags", json.dumps(cleaned))
        self.assertNotIn("usability_notes", json.dumps(cleaned))
        self.assertNotIn("gender", json.dumps(cleaned["speaker_descriptions"]))

        item = batch.ScoreItem(
            "sample-1",
            cleaned,
            cleaned,
            Path("gt.json"),
            Path("prediction.jsonl"),
            1,
        )
        self.assertNotIn("gender", item.user_text)

    def test_clean_prediction_drops_uncertainties(self):
        cleaned = batch.clean_descriptive(descriptive_payload(), False)
        self.assertEqual(set(cleaned), set(batch.SCORED_FIELDS))

    def test_prediction_loader_rejects_duplicate_ids(self):
        row = {"sample_id": "same", "descriptive": descriptive_payload()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(batch.BatchInputError):
                batch.load_predictions(path)


class CheckpointTest(unittest.TestCase):
    def test_reusable_success_requires_matching_input_and_config(self):
        payload = descriptive_payload()
        prediction = batch.clean_descriptive(payload, False)
        ground_truth = batch.clean_descriptive(payload, True)
        item = batch.ScoreItem(
            "sample-1",
            ground_truth,
            prediction,
            Path("gt.json"),
            Path("predictions.jsonl"),
            1,
        )
        scores = {field: 1.0 for field in batch.SCORED_FIELDS}
        record = {
            "status": "success",
            "sample_id": "sample-1",
            "prompt_sha256": "prompt",
            "config_fingerprint": "config",
            "input_sha256": item.input_sha256,
            "scores": scores,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample-1.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNotNone(
                batch.reusable_success(path, item, "prompt", "config")
            )
            self.assertIsNone(
                batch.reusable_success(path, item, "new-prompt", "config")
            )


class ApiConfigTest(unittest.TestCase):
    def test_load_newapi_json_requires_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grok.txt"
            path.write_text(
                '{"_type":"newapi_channel_conn","key":"sk-test","url":"https://example.com"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                batch.load_api_config(path)
            config = batch.load_api_config(path, "grok-4.7")
            self.assertEqual(config["api_key"], "sk-test")
            self.assertEqual(config["base_url"], "https://example.com")
            self.assertEqual(config["model"], "grok-4.7")


if __name__ == "__main__":
    unittest.main()
