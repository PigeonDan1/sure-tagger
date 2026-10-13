import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_deepseek_batch as batch


CAPTION_SMOKE = (
    EVAL_DIR.parent
    / "evaluation"
    / "captioner-v0"
    / "deepseek-text-single.json"
)


def _structural():
    tags = {
        group: {field: None for field in fields}
        for group, fields in batch.STRUCTURAL_FIELDS.items()
    }
    tags["basic_acoustic"].update(duration_sec=10.0, sample_rate_hz=16000, channels=1)
    tags["speaker"].update(
        speaker_count=2, speaker_present=True, multi_speaker=True,
        speaker_change_count=3, speaker_change=True,
        overlap_ratio=0.2, speaker_overlap=True,
        asr_transcript="Hello there, this is a test.",
        profiles=[
            {
                "speaker_id": "speaker_%d" % index,
                "speech_rate": {"band": None, "value": None, "unit": None},
                "pitch": None,
                "speaker_volume": None,
            }
            for index in (1, 2)
        ],
    )
    tags["language_content"].update(language="cy", word_count=6, filler=0)
    tags["deterministic_derived"].update(speaker_change_rate_per_min=18.0, filler_ratio=0.0)
    return tags


class DeepSeekBatchInputTest(unittest.TestCase):
    def test_sure_wire_text_drops_metadata(self):
        structural = _structural()
        row = {
            "sample_id": "sample-1",
            "audio_path": "/secret/audio.wav",
            "ground_truth": {"speaker_count": 99},
            "tags": structural,
        }
        text, _ = batch._sure_text_from_row(row, "sample-1", Path("."))
        value = json.loads(text)
        self.assertEqual(set(value), set(batch.STRUCTURAL_GROUPS))
        self.assertNotIn("sample_id", text)
        self.assertNotIn("/secret/audio.wav", text)
        self.assertNotIn("ground_truth", text)

    def test_sure_wire_text_drops_profile_gender(self):
        structural = _structural()
        structural["speaker"]["profiles"] = [
            {
                "speaker_id": "speaker_1",
                "gender": "female",
                "speech_rate": {"band": None, "value": None, "unit": None},
                "pitch": None,
                "speaker_volume": None,
            }
        ]
        text, _ = batch._sure_text_from_row(
            {"sample_id": "sample-1", "tags": structural}, "sample-1", Path(".")
        )
        self.assertNotIn('"gender"', text)

    def test_captioner_prefers_input_text_only(self):
        row = {
            "sample_id": "sample-1",
            "input_text": "one caption",
            "audio_path": "/secret/audio.wav",
            "ground_truth": "must not be sent",
        }
        text, _ = batch._caption_text_from_row(row, Path("."))
        self.assertEqual(text, "one caption")

    def test_load_items_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(
                json.dumps({"sample_id": "same", "input_text": "a"})
                + "\n"
                + json.dumps({"sample_id": "same", "input_text": "b"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(batch.BatchInputError):
                batch.load_items("captioner", path)


class DeepSeekBatchExecutionTest(unittest.TestCase):
    def test_sure_v1_rejects_speaker_correction_wrapper(self):
        raw = {
            "descriptive": {key: None for key in batch.sure_once.DESCRIPTIVE_ORDER},
            "language_verification": None,
            "speaker_correction": {"speaker_count": 2},
        }
        with self.assertRaisesRegex(batch.sure_once.ValidationError, "unsupported keys"):
            batch._sure_payload(raw, _structural(), "sample-1")

    def test_sure_v1_preserves_structure_and_only_records_language(self):
        source = _structural()
        wire_text, _ = batch._sure_text_from_row(
            {"sample_id": "sample-1", "tags": source}, "sample-1", Path(".")
        )
        structural = json.loads(wire_text)
        raw = {
            "descriptive": {key: None for key in batch.sure_once.DESCRIPTIVE_ORDER},
            "language_verification": {
                "language": "en", "confidence": 0.99, "reason_short": "English text"
            },
        }
        item = batch.BatchItem("sample-1", wire_text, source_hash="src")
        config = {"api_key": "secret", "base_url": "https://example/v1", "model": "m"}
        with mock.patch.object(batch.captioner_once, "call_api", return_value=(raw, {})):
            record = batch.process_item(
                item, "sure-tagger", config, "prompt", "p", "c",
                timeout=1, retries=0, max_tokens=100, thinking="disabled",
                backoff_base=0, backoff_max=0, jitter=False,
            )
        self.assertEqual(record["status"], "success", record.get("error"))
        payload = record["payload"]
        for group in batch.STRUCTURAL_GROUPS:
            if group != "language_content":
                self.assertEqual(payload[group], structural[group], group)
        self.assertEqual(payload["language_content"]["language"], "en")
        for field, value in structural["language_content"].items():
            if field != "language":
                self.assertEqual(payload["language_content"][field], value, field)
        self.assertEqual(record["language_verification"]["published_language"], "en")
        self.assertNotIn("speaker_correction", record)
        with tempfile.TemporaryDirectory() as directory:
            summary = batch._aggregate(
                [item], [record], directory, "sure-tagger", config, "p", "c", {}
            )
            self.assertEqual(summary["language_verification"]["changed_count"], 1)
            self.assertNotIn("speaker_correction", summary)
            self.assertTrue((Path(directory) / "language_verification.jsonl").is_file())
            self.assertFalse((Path(directory) / "speaker_correction.jsonl").exists())

    def test_network_classifier_respects_http_status(self):
        self.assertTrue(batch.is_network_failure(RuntimeError("HTTP 503: busy")))
        self.assertTrue(batch.is_network_failure(RuntimeError("HTTP 429: rate limit")))
        self.assertFalse(batch.is_network_failure(RuntimeError("HTTP 401: denied")))
        self.assertFalse(batch.is_network_failure(RuntimeError("HTTP 400: timeout is invalid")))

    def test_retry_and_local_schema_validation(self):
        payload = json.loads(CAPTION_SMOKE.read_text(encoding="utf-8"))
        calls = []
        original = batch.captioner_once.call_api

        def fake_call(*args):
            calls.append(args[4])
            if len(calls) == 1:
                raise RuntimeError("transient")
            return payload, {"total_tokens": 1}

        batch.captioner_once.call_api = fake_call
        try:
            item = batch.BatchItem("sample-1", "only caption", source_hash="src")
            record = batch.process_item(
                item,
                "captioner",
                {"api_key": "secret", "base_url": "https://example/v1", "model": "m"},
                "prompt",
                "prompt-hash",
                "config-hash",
                timeout=1,
                retries=1,
                max_tokens=100,
                thinking="disabled",
                backoff_base=0,
                backoff_max=0,
                jitter=False,
            )
        finally:
            batch.captioner_once.call_api = original
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(calls, ["only caption", "only caption"])

    def test_non_network_failure_is_not_retried(self):
        calls = []
        original = batch.captioner_once.call_api

        def fake_call(*args):
            calls.append(args[4])
            raise RuntimeError("HTTP 401: invalid API key")

        batch.captioner_once.call_api = fake_call
        try:
            item = batch.BatchItem("sample-1", "only caption", source_hash="src")
            record = batch.process_item(
                item,
                "captioner",
                {"api_key": "secret", "base_url": "https://example/v1", "model": "m"},
                "prompt",
                "prompt-hash",
                "config-hash",
                timeout=1,
                retries=4,
                max_tokens=100,
                thinking="disabled",
                backoff_base=0,
                backoff_max=0,
                jitter=False,
            )
        finally:
            batch.captioner_once.call_api = original
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "api_error")
        self.assertEqual(record["attempts"], 1)
        self.assertEqual(calls, ["only caption"])

    def test_failed_checkpoint_retries_only_network_errors(self):
        item = batch.BatchItem("sample-1", "only caption")
        common = {
            "sample_id": "sample-1",
            "status": "failed",
            "prompt_sha256": "p",
            "config_fingerprint": "c",
            "input_sha256": batch.sha256_text("only caption"),
        }
        deterministic = dict(common, error_type="api_error", error="HTTP 401: denied")
        network = dict(common, error_type="network_error", error="HTTP 503: busy")
        mixed = dict(
            common,
            error_type="api_error",
            error="HTTP 503: busy; HTTP 401: denied",
        )
        mixed_list = dict(
            common,
            error_type="api_error",
            errors=["HTTP 503: busy; HTTP 401: denied"],
        )
        typed_deterministic = dict(
            common,
            error_type="validation_error",
            error="timeout while validating a field",
        )
        self.assertTrue(
            batch._checkpoint_is_reusable(item=item, record=deterministic,
                                          prompt_hash="p", config_fingerprint="c",
                                          retry_failed=True)
        )
        self.assertFalse(
            batch._checkpoint_is_reusable(item=item, record=network,
                                          prompt_hash="p", config_fingerprint="c",
                                          retry_failed=True)
        )
        self.assertTrue(
            batch._checkpoint_is_reusable(item=item, record=mixed,
                                          prompt_hash="p", config_fingerprint="c",
                                          retry_failed=True)
        )
        self.assertTrue(
            batch._checkpoint_is_reusable(item=item, record=mixed_list,
                                          prompt_hash="p", config_fingerprint="c",
                                          retry_failed=True)
        )
        self.assertTrue(
            batch._checkpoint_is_reusable(item=item, record=typed_deterministic,
                                          prompt_hash="p", config_fingerprint="c",
                                          retry_failed=True)
        )

    def test_checkpoint_round_trip(self):
        payload = json.loads(CAPTION_SMOKE.read_text(encoding="utf-8"))
        payload["sample_id"] = "sample-1"
        batch.captioner_once.validate_payload(payload, "sample-1")
        item = batch.BatchItem("sample-1", "only caption")
        record = batch._success_record(item, payload, "p", "c", {}, 1, 0)
        with tempfile.TemporaryDirectory() as directory:
            batch._write_checkpoint(directory, record)
            loaded = batch._load_checkpoints(directory)
            self.assertEqual(set(loaded), {"sample-1"})
            self.assertTrue(
                batch._checkpoint_is_reusable(
                    loaded["sample-1"], item, "p", "c", retry_failed=True
                )
            )


class SureTaggerPromptVersionTest(unittest.TestCase):
    def test_default_is_v1(self):
        path, version = batch.resolve_prompt_path("sure-tagger")
        self.assertEqual(version, "v1")
        self.assertEqual(path, batch.SURE_TAGGER_PROMPT_VERSIONS["v1"].resolve())
        self.assertEqual(path, batch.DEFAULT_PROMPTS["sure-tagger"].resolve())
        self.assertTrue(path.is_file())

    def test_retired_versions_are_rejected(self):
        for version in ("v2", "v3", "v4", "v5", "v6"):
            with self.subTest(version=version):
                with self.assertRaises(ValueError):
                    batch.resolve_prompt_path("sure-tagger", prompt_version=version)

    def test_explicit_prompt_overrides_version(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom_prompt.txt"
            custom.write_text("custom prompt", encoding="utf-8")
            path, version = batch.resolve_prompt_path(
                "sure-tagger", prompt=custom, prompt_version="v1"
            )
            self.assertEqual(path, custom.resolve())
            self.assertEqual(version, "custom")

    def test_custom_prompt_is_labelled_custom(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("custom prompt")
            custom = Path(handle.name)
        try:
            path, version = batch.resolve_prompt_path("sure-tagger", prompt=custom)
            self.assertEqual(path, custom.resolve())
            self.assertEqual(version, "custom")
        finally:
            custom.unlink()

    def test_captioner_rejects_prompt_version(self):
        with self.assertRaises(ValueError):
            batch.resolve_prompt_path("captioner", prompt_version="v1")

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):
            batch.resolve_prompt_path("sure-tagger", prompt_version="v9")



if __name__ == "__main__":
    unittest.main()
