"""Regression tests for the Qwen3-ASR-0.6B speaker-v2 adapter."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

from tagger.pipelines import speaker_evidence
from tagger.tools import subprocess_worker
from tagger.tools.base import ToolResult
from tagger.tools.speaker_v2 import qwen3_asr


def _write_wav(path, duration_sec=0.1):
    sample_rate = 16000
    frame_count = int(round(sample_rate * duration_sec))
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(b"\x00\x00" * frame_count)


class Qwen3AsrAdapterTest(unittest.TestCase):
    def test_config_and_worker_record_are_stable(self):
        config = qwen3_asr.Qwen3AsrConfig(
            model_dir="model",
            subprocess_python="runtime/bin/python",
            device="cuda:1",
            torch_dtype="float16",
            max_new_tokens=123,
            timeout_sec=17,
        )

        self.assertEqual(config.model_dir, "model")
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(config.max_new_tokens, 123)
        self.assertTrue(config.return_time_stamps)
        self.assertIn("forced_aligner_model_dir", config.to_record())
        self.assertTrue(config.to_record()["return_time_stamps"])
        record = qwen3_asr._subprocess_config(config)
        self.assertEqual(record["subprocess_python"], "")
        self.assertFalse(record["normalize_to_16k_mono_pcm"])
        self.assertNotIn("timeout_sec", record)

    def test_config_rejects_invalid_decode_settings(self):
        with self.assertRaises(ValueError):
            qwen3_asr.Qwen3AsrConfig(torch_dtype="int8")
        with self.assertRaises(ValueError):
            qwen3_asr.Qwen3AsrConfig(max_new_tokens=0)
        with self.assertRaises(ValueError):
            qwen3_asr.Qwen3AsrConfig(timeout_sec=0)

    def test_normalize_result_preserves_detected_language(self):
        result = qwen3_asr.normalize_result(
            {
                "text": "  bonjour\tle monde  ",
                "language": " French ",
                "runtime": {"device": "cuda:0"},
            },
            audio_path="clip.wav",
        )

        self.assertEqual(result["text"], "bonjour le monde")
        self.assertEqual(result["language"], "French")
        self.assertEqual(result["runtime"]["device"], "cuda:0")
        self.assertEqual(result["audio_path"], str(Path("clip.wav").resolve()))

    def test_normalize_result_preserves_native_timestamp_items(self):
        result = qwen3_asr.normalize_result(
            {
                "text": "hello world",
                "timestamps": [
                    {"text": "hello", "start_time": 0.12, "end_time": 0.48},
                    {"text": "world", "start_sec": 0.52, "end_sec": 0.91},
                    {"text": ".", "start_time": 0.91, "end_time": 0.91},
                ],
            }
        )

        self.assertEqual(len(result["timestamps"]), 2)
        self.assertEqual(result["timestamps"][0]["text"], "hello")
        self.assertEqual(result["timestamps"][0]["start_sec"], 0.12)
        self.assertEqual(result["timestamps"][1]["end_sec"], 0.91)

    def test_subprocess_client_uses_qwen_worker_and_cleans_prepared_audio(self):
        config = qwen3_asr.Qwen3AsrConfig(
            subprocess_python="runtime/bin/python", timeout_sec=23
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.wav"
            prepared = Path(tmpdir) / "prepared.wav"
            _write_wav(audio)
            _write_wav(prepared)
            cleanup = mock.Mock()
            with mock.patch.object(
                qwen3_asr,
                "_prepare_audio",
                return_value=(prepared, cleanup),
            ), mock.patch.object(
                qwen3_asr,
                "run_subprocess_tool",
                return_value={
                    "output": {"text": "bonjour", "language": "French"}
                },
            ) as runner:
                result = qwen3_asr.Qwen3AsrSubprocessClient(config).transcribe(
                    audio, context={}
                )

        self.assertEqual(result["text"], "bonjour")
        self.assertEqual(runner.call_args[0][1], "qwen3_asr_estimate")
        self.assertEqual(runner.call_args[1]["timeout_sec"], 23)
        cleanup.cleanup.assert_called_once_with()

    def test_worker_route_uses_cached_qwen_client(self):
        config = qwen3_asr._subprocess_config(
            qwen3_asr.Qwen3AsrConfig(model_dir="unused")
        )
        fake_client = mock.Mock()
        fake_client.transcribe.return_value = {
            "text": "bonjour",
            "language": "French",
        }
        with mock.patch.object(
            subprocess_worker, "_cached_client", return_value=fake_client
        ) as cached:
            result = subprocess_worker.dispatch(
                "qwen3_asr_estimate",
                {"audio_path": "audio.wav", "config": config},
            )

        self.assertEqual(result["output"]["text"], "bonjour")
        self.assertEqual(cached.call_args[0][0], "qwen3_asr_estimate")
        fake_client.transcribe.assert_called_once_with("audio.wav", context=None)

    def test_collector_builds_qwen_evidence(self):
        scope = {
            "sample_id": "sample",
            "audio_path": "clip.wav",
            "audio_sha256": "0" * 64,
            "duration_sec": 1.0,
        }
        raw = ToolResult(
            tag_path="speaker.asr_transcript",
            value={"text": "bonjour", "language": "French"},
            tool_name=qwen3_asr.TOOL_NAME,
            method=qwen3_asr.METHOD,
        )
        with mock.patch.object(
            speaker_evidence, "run_qwen3_asr", return_value=raw
        ):
            evidence = speaker_evidence.collect_qwen3_asr_evidence(
                scope,
                qwen3_asr.Qwen3AsrConfig(model_dir="unused"),
                verify_model_asset=False,
            )

        self.assertEqual(evidence["source"]["name"], "qwen3_asr_0_6b")
        self.assertEqual(evidence["payload"]["asr_transcript"], "bonjour")
        self.assertEqual(evidence["payload"]["language"], "French")
        self.assertTrue(evidence["quality"]["usable"])

    def test_collector_exposes_qwen_lexical_timeline(self):
        scope = {
            "sample_id": "sample",
            "audio_path": "clip.wav",
            "audio_sha256": "0" * 64,
            "duration_sec": 1.0,
        }
        raw = ToolResult(
            tag_path="speaker.asr_transcript",
            value={
                "text": "bonjour monde",
                "language": "French",
                "timestamps": [
                    {"text": "bonjour", "start_sec": 0.1, "end_sec": 0.4},
                    {"text": "monde", "start_sec": 0.5, "end_sec": 0.8},
                ],
            },
            tool_name=qwen3_asr.TOOL_NAME,
            method=qwen3_asr.METHOD,
        )
        with mock.patch.object(
            speaker_evidence, "run_qwen3_asr", return_value=raw
        ):
            evidence = speaker_evidence.collect_qwen3_asr_evidence(
                scope,
                qwen3_asr.Qwen3AsrConfig(model_dir="unused"),
                verify_model_asset=False,
            )

        self.assertIn("lexical_timeline", evidence["capabilities"])
        self.assertEqual(evidence["quality"]["timestamp_status"], "native")
        self.assertEqual(len(evidence["payload"]["lexical_units"]), 2)


if __name__ == "__main__":
    unittest.main()
