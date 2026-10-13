import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import summarize_public_phase0


UNIVERSE_PATH = EVAL_DIR / "field_universe.structural-v0.1.json"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_nested(payload, path, value):
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


class PublicPhase0SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = {
            "key": "demo",
            "name": "Demo",
            "manifest": "demo.manifest.jsonl",
            "metadata": "demo.metadata.jsonl",
            "sure": "demo.sure.jsonl",
            "captioner": "captioner_raw/demo/results.jsonl",
            "expected_count": 1,
        }
        self.sample_id = "demo_0001"
        self.audio_path = self.root / "audio" / "demo.wav"
        self.audio_path.parent.mkdir(parents=True)
        with wave.open(str(self.audio_path), "wb") as sink:
            sink.setnchannels(1)
            sink.setsampwidth(2)
            sink.setframerate(16000)
            sink.writeframes(b"\0\0" * 160)
        self.write_data(sha256_file(self.audio_path))

    def tearDown(self):
        self.temporary.cleanup()

    def write_data(self, digest, sample_rate_hz=16000):
        manifest = {
            "corpus": {
                "dataset_name": "Demo",
                "source_urls": {
                    "article": [],
                    "github": [],
                    "huggingface": [],
                    "dataset_card": [],
                },
                "native_metadata": {},
            },
            "sample": {
                "sample_id": self.sample_id,
                "audio": {"path": "audio/demo.wav"},
                "text": {"transcript": ""},
                "native_metadata": {},
            },
        }
        metadata = {
            "sample_id": self.sample_id,
            "system_input_audio": {
                "path": "audio/demo.wav",
                "sha256": digest,
                "header": {
                    "channels": 1,
                    "sample_rate_hz": sample_rate_hz,
                    "frame_count": 160,
                    "duration_sec": 0.01,
                    "sample_width_bytes": 2,
                },
            },
        }
        write_jsonl(self.root / self.spec["manifest"], [manifest])
        write_jsonl(self.root / self.spec["metadata"], [metadata])

    def current_schema_payload(self):
        universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        payload = {}
        for field in universe["fields"]:
            set_nested(payload, field["path"], None)
        payload["basic_acoustic"]["duration_sec"] = 0.01
        payload["basic_acoustic"]["sample_rate_hz"] = 16000
        payload["basic_acoustic"]["channels"] = 1
        return payload

    def write_system_outputs(self, sure_rows=None):
        if sure_rows is None:
            sure_rows = [self.current_schema_payload()]
        write_jsonl(self.root / self.spec["sure"], sure_rows)
        write_jsonl(
            self.root / self.spec["captioner"],
            [
                {
                    "sample_id": self.sample_id,
                    "audio_path": "audio/demo.wav",
                    "status": "success",
                    "caption": "A short demonstration sound.",
                }
            ],
        )

    def summarize(self):
        return summarize_public_phase0.summarize(
            self.root, [self.spec], UNIVERSE_PATH
        )

    def test_complete_summary_separates_raw_caption_from_schema(self):
        self.write_system_outputs()
        summary = self.summarize()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["issues"], [])
        dataset = summary["datasets"][0]
        self.assertEqual(dataset["data"]["audio"]["sha256_match_count"], 1)
        self.assertEqual(dataset["data"]["audio"]["wav_header_valid_count"], 1)
        sure = dataset["systems"]["sure-tagger"]
        self.assertEqual(sure["alignment"], "manifest_row_order_positional")
        for metric in sure["header_accuracy"].values():
            self.assertEqual(metric["accuracy"], 1.0)
        self.assertEqual(
            sure["schema"]["native_current_schema_valid_rate"], 1.0
        )
        captioner = dataset["systems"]["captioner"]
        self.assertEqual(captioner["runtime"]["success_rate"], 1.0)
        self.assertEqual(
            captioner["schema"]["native_current_schema_valid_rate"], 0.0
        )
        self.assertFalse(
            captioner["raw_caption_semantics"]["is_schema_prediction"]
        )
        markdown = summarize_public_phase0.render_markdown(summary)
        self.assertIn("not a schema prediction", markdown)

    def test_missing_system_outputs_are_incomplete(self):
        summary = self.summarize()
        self.assertEqual(summary["status"], "incomplete")
        codes = set(issue["code"] for issue in summary["issues"])
        self.assertEqual(
            codes, set(["missing_sure_output", "missing_captioner_output"])
        )
        dataset = summary["datasets"][0]
        self.assertEqual(dataset["systems"]["sure-tagger"]["status"], "missing")
        self.assertEqual(dataset["systems"]["captioner"]["status"], "missing")

    def test_audio_sha256_mismatch_is_invalid(self):
        self.write_data("0" * 64)
        self.write_system_outputs()
        summary = self.summarize()
        self.assertEqual(summary["status"], "invalid")
        codes = set(issue["code"] for issue in summary["issues"])
        self.assertIn("audio_sha256_mismatch", codes)
        self.assertEqual(
            summary["datasets"][0]["data"]["audio"]["sha256_match_rate"],
            0.0,
        )

    def test_metadata_wav_header_mismatch_is_invalid(self):
        self.write_data(sha256_file(self.audio_path), sample_rate_hz=8000)
        self.write_system_outputs()
        summary = self.summarize()
        self.assertEqual(summary["status"], "invalid")
        codes = set(issue["code"] for issue in summary["issues"])
        self.assertIn("metadata_wav_header_mismatch", codes)

    def test_unkeyed_sure_count_mismatch_is_invalid(self):
        payload = self.current_schema_payload()
        self.write_system_outputs([payload, payload])
        summary = self.summarize()
        self.assertEqual(summary["status"], "invalid")
        sure = summary["datasets"][0]["systems"]["sure-tagger"]
        self.assertEqual(sure["status"], "invalid")
        codes = set(issue["code"] for issue in summary["issues"])
        self.assertIn("invalid_sure_output", codes)


if __name__ == "__main__":
    unittest.main()
