"""Regression tests for the FireRedASR2-AED speaker-v2 adapter.

These tests intentionally stop at the adapter boundary.  They do not load the
large AED checkpoint; model routing and inference are represented by a mocked
subprocess response so the tests remain deterministic and cheap.
"""

from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest import mock
import wave

from tagger.tools.base import ToolResult
from tagger.tools import subprocess_worker
from tagger.pipelines import speaker_evidence
from tagger.tools.speaker_v2 import firered_asr


def _write_wav(path, duration_sec=1.0, sample_rate=16000):
    frame_count = int(round(duration_sec * sample_rate))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


class FireRedAsrAdapterTest(unittest.TestCase):
    def test_config_aliases_and_worker_record_are_stable(self):
        config = firered_asr.FireRedAsrConfig(
            model_path="model",
            source_path="source",
            subprocess_python="runtime/bin/python",
            use_gpu=False,
            half=True,
            beam_size=5,
            timeout_sec=17,
            deployment_dir="deploy",
        )

        self.assertEqual(config.model_dir, "model")
        self.assertEqual(config.source_dir, "source")
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.use_half)
        self.assertEqual(config.beam_size, 5)
        record = config.to_record()
        self.assertEqual(record["model_dir"], "model")
        self.assertEqual(record["source_dir"], "source")
        self.assertEqual(record["beam_size"], 5)
        self.assertEqual(record["timeout_sec"], 17)

        worker_record = firered_asr._subprocess_config(config)
        self.assertEqual(worker_record["subprocess_python"], "")
        self.assertFalse(worker_record["normalize_to_16k_mono_pcm"])
        self.assertNotIn("timeout_sec", worker_record)

    def test_config_enables_lid_and_derives_lid_checkpoint_from_source(self):
        config = firered_asr.FireRedAsrConfig(
            model_dir="model",
            source_dir="source",
            subprocess_python="",
        )
        self.assertTrue(config.enable_lid)
        self.assertEqual(
            config.lid_model_dir,
            str(Path("source") / "pretrained_models" / "FireRedLID"),
        )
        record = firered_asr._subprocess_config(config)
        self.assertTrue(record["enable_lid"])
        self.assertEqual(record["lid_model_dir"], config.lid_model_dir)
        self.assertFalse(record["lid_use_half"])
        self.assertTrue(record["verify_lid_model_asset"])

    def test_lid_checkpoint_hash_is_verified_before_runtime_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            for name in firered_asr.REQUIRED_LID_MODEL_FILES:
                (model_dir / name).write_bytes(b"")
            checkpoint = model_dir / "model.pth.tar"
            checkpoint.write_bytes(b"pinned FireRedLID checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            with mock.patch.object(
                firered_asr, "LID_CHECKPOINT_SHA256", expected
            ):
                self.assertEqual(
                    firered_asr._validate_lid_model_files(model_dir), expected
                )
                checkpoint.write_bytes(b"replaced checkpoint")
                with self.assertRaisesRegex(
                    firered_asr.FireRedAsrError, "checkpoint SHA256 mismatch"
                ):
                    firered_asr._validate_lid_model_files(model_dir)
                self.assertIsNone(
                    firered_asr._validate_lid_model_files(
                        model_dir, verify_checkpoint=False
                    )
                )

            config = firered_asr.FireRedAsrConfig(
                model_dir=model_dir,
                source_dir=model_dir,
                subprocess_python="runtime/bin/python",
            )
            raw_result = mock.Mock(
                value={
                    "text": "hello",
                    "confidence": 0.9,
                    "duration_s": 1.0,
                    "timestamps": [],
                    "runtime": {"lid_status": "disabled"},
                }
            )
            scope = {
                "sample_id": "skip-verification",
                "audio_path": str(model_dir / "clip.wav"),
                "audio_sha256": "0" * 64,
                "duration_sec": 1.0,
            }
            with mock.patch.object(
                speaker_evidence,
                "run_firered_asr",
                return_value=raw_result,
            ) as runner:
                evidence = speaker_evidence.collect_firered_asr_evidence(
                    scope,
                    config,
                    context={},
                    verify_model_asset=False,
                )
            runtime_config = runner.call_args[1]["config"]
            self.assertFalse(runtime_config.verify_lid_model_asset)
            self.assertFalse(evidence["quality"]["model_asset_verified"])

    def test_config_rejects_invalid_decode_settings(self):
        with self.assertRaises(ValueError):
            firered_asr.FireRedAsrConfig(beam_size=0)
        with self.assertRaises(ValueError):
            firered_asr.FireRedAsrConfig(timeout_sec=0)
        with self.assertRaises(TypeError):
            firered_asr.FireRedAsrConfig(unknown_option=True)

    def test_normalize_result_accepts_upstream_spellings_and_preserves_provenance(self):
        result = firered_asr.normalize_result(
            {
                "schema_version": "upstream-v0",
                "transcript": "  hello\tworld  ",
                "confidence": "0.875",
                "dur_s": "2.5",
                "real_time_factor": "0.125",
                "timestamp": [
                    ["hello", 0, 1.0],
                    {"text": "world", "start": 1.0, "end": 2.0},
                ],
                "model": {"name": "FireRedASR2-AED"},
                "runtime": {"device": "cuda:0"},
            },
            audio_path="clip.wav",
        )

        self.assertEqual(result["schema_version"], "upstream-v0")
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["confidence"], 0.875)
        self.assertEqual(result["duration_s"], 2.5)
        self.assertEqual(result["rtf"], 0.125)
        self.assertEqual(
            result["timestamps"],
            [
                {"token": "hello", "start_s": 0.0, "end_s": 1.0},
                {"token": "world", "start_s": 1.0, "end_s": 2.0},
            ],
        )
        self.assertEqual(result["audio_path"], str(Path("clip.wav").resolve()))
        self.assertEqual(result["model"]["name"], "FireRedASR2-AED")
        self.assertEqual(result["runtime"]["device"], "cuda:0")

    def test_validate_result_allows_empty_prediction_but_checks_duration_and_timestamps(self):
        value = firered_asr.validate_result(
            {
                "text": "",
                "duration_s": 3,
                "timestamps": [],
            },
            duration_sec=3.2,
        )
        self.assertEqual(value["text"], "")
        self.assertEqual(value["duration_s"], 3.0)

        invalid_values = [
            {"text": "x", "confidence": 1.1},
            {"text": "x", "duration_s": -1},
            {
                "text": "x",
                "duration_s": 1,
                "timestamps": [["x", 0.5, 0.4]],
            },
            {
                "text": "x",
                "duration_s": 1,
                "timestamps": [["x", 0.2, 0.8], ["y", 0.1, 0.9]],
            },
            {
                "text": "x",
                "duration_s": 1,
                "timestamps": [["x", 0.0, 1.1]],
            },
        ]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(firered_asr.FireRedAsrError):
                    firered_asr.validate_result(invalid)

    def test_normalize_result_canonicalizes_language_and_rejects_conflicts(self):
        result = firered_asr.normalize_result(
            {
                "text": "bonjour",
                "language": " fr ",
                "language_confidence": "0.875",
            }
        )
        self.assertEqual(result["lang"], "fr")
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["language_confidence"], 0.875)

        with self.assertRaises(firered_asr.FireRedAsrError):
            firered_asr.normalize_result(
                {"text": "x", "lang": "en", "language": "fr"}
            )
        with self.assertRaises(firered_asr.FireRedAsrError):
            firered_asr.normalize_result(
                {"text": "x", "lang": "en", "language_confidence": 1.1}
            )

    def test_client_merges_successful_lid_without_changing_asr_text(self):
        class FakeRuntime(object):
            lid_load_error = None

            def transcribe(self, audio_path):
                return {
                    "text": "中文结果",
                    "confidence": 0.91,
                    "duration_s": 1.0,
                    "timestamps": [],
                    "runtime": {"device": "cpu"},
                }

            def detect_language(self, audio_path):
                return {
                    "lang": "zh mandarin",
                    "confidence": 0.996,
                    "dur_s": 1.0,
                    "rtf": "0.1250",
                    "runtime": {"device": "cpu", "inference_s": 0.01},
                }

        config = firered_asr.FireRedAsrConfig(
            subprocess_python="",
            normalize_to_16k_mono_pcm=False,
        )
        client = firered_asr.FireRedAsrClient(config)
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.wav"
            _write_wav(audio)
            with mock.patch.object(
                client, "_get_runtime", return_value=runtime
            ):
                result = client.transcribe(audio)

        self.assertEqual(result["text"], "中文结果")
        self.assertEqual(result["lang"], "zh mandarin")
        self.assertEqual(result["language"], "zh mandarin")
        self.assertEqual(result["language_confidence"], 0.996)
        self.assertEqual(result["language_duration_s"], 1.0)
        self.assertEqual(result["language_rtf"], 0.125)
        self.assertEqual(result["runtime"]["lid_status"], "ok")
        self.assertNotIn("language_error", result)

    def test_client_keeps_asr_text_when_lid_fails(self):
        class FakeRuntime(object):
            lid_load_error = "checkpoint unavailable"

            def transcribe(self, audio_path):
                return {
                    "text": "hello world",
                    "confidence": 0.8,
                    "duration_s": 1.0,
                    "timestamps": [],
                }

            def detect_language(self, audio_path):
                raise firered_asr.FireRedAsrError("LID inference failed")

        config = firered_asr.FireRedAsrConfig(
            subprocess_python="",
            normalize_to_16k_mono_pcm=False,
        )
        client = firered_asr.FireRedAsrClient(config)
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.wav"
            _write_wav(audio)
            with mock.patch.object(
                client, "_get_runtime", return_value=runtime
            ):
                result = client.transcribe(audio)

        self.assertEqual(result["text"], "hello world")
        self.assertIn("language_error", result)
        self.assertIn("LID inference failed", result["language_error"])
        self.assertEqual(result["runtime"]["lid_status"], "unavailable")
        self.assertIn("lid_error", result["runtime"])

    def test_subprocess_client_normalizes_result_and_sends_blind_request(self):
        config = firered_asr.FireRedAsrConfig(
            model_dir="model",
            source_dir="source",
            subprocess_python="runtime/bin/python",
            timeout_sec=23,
        )
        raw = {
            "text": "中文结果",
            "confidence": 0.91,
            "duration_s": 1.0,
            "timestamps": [["中", 0.0, 0.5], ["文", 0.5, 1.0]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.wav"
            prepared = Path(tmpdir) / "prepared.wav"
            _write_wav(audio)
            _write_wav(prepared)
            cleanup = mock.Mock()
            with mock.patch.object(
                firered_asr,
                "_prepare_audio_for_subprocess",
                return_value=(prepared, cleanup),
            ), mock.patch.object(
                firered_asr,
                "run_subprocess_tool",
                return_value={"output": raw},
            ) as runner:
                result = firered_asr.FireRedAsrSubprocessClient(config).transcribe(
                    audio, context={"request_id": "test"}
                )

        self.assertEqual(result["text"], "中文结果")
        self.assertEqual(result["audio_path"], str(audio.resolve()))
        self.assertEqual(result["runtime"]["prepared_audio_path"], str(prepared))
        self.assertTrue(result["runtime"]["audio_normalized"])
        self.assertEqual(runner.call_args[0][0], "runtime/bin/python")
        self.assertEqual(runner.call_args[0][1], "firered_asr_estimate")
        request = runner.call_args[0][2]
        self.assertEqual(set(request), {"audio_path", "config"})
        self.assertEqual(request["audio_path"], str(prepared))
        self.assertNotIn("timeout_sec", request["config"])
        self.assertEqual(runner.call_args[1]["timeout_sec"], 23)
        cleanup.cleanup.assert_called_once_with()

    def test_subprocess_client_rejects_missing_audio_before_worker_call(self):
        config = firered_asr.FireRedAsrConfig(subprocess_python="runtime/bin/python")
        with mock.patch.object(firered_asr, "run_subprocess_tool") as runner:
            with self.assertRaises(firered_asr.FireRedAsrError):
                firered_asr.FireRedAsrSubprocessClient(config).transcribe(
                    "/does/not/exist.wav"
                )
        runner.assert_not_called()

    def test_subprocess_client_preserves_worker_failure_detail(self):
        config = firered_asr.FireRedAsrConfig(
            subprocess_python="runtime/bin/python",
            normalize_to_16k_mono_pcm=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.wav"
            _write_wav(audio)
            with mock.patch.object(
                firered_asr,
                "_prepare_audio_for_subprocess",
                return_value=(audio, None),
            ), mock.patch.object(
                firered_asr,
                "run_subprocess_tool",
                side_effect=TimeoutError("worker timed out"),
            ):
                with self.assertRaisesRegex(
                    firered_asr.FireRedAsrError,
                    "TimeoutError: worker timed out",
                ):
                    firered_asr.FireRedAsrSubprocessClient(config).transcribe(
                        audio
                    )

    def test_worker_route_uses_cached_firered_client(self):
        config = firered_asr._subprocess_config(
            firered_asr.FireRedAsrConfig(model_dir="unused")
        )
        fake_client = mock.Mock()
        fake_client.transcribe.return_value = {
            "text": "中文结果",
            "confidence": 0.9,
            "duration_s": 1.0,
            "timestamps": [],
        }
        request = {
            "audio_path": "audio.wav",
            "config": config,
        }

        with mock.patch.object(
            subprocess_worker, "_cached_client", return_value=fake_client
        ) as cached:
            result = subprocess_worker.dispatch("firered_asr_estimate", request)

        self.assertEqual(result["output"]["text"], "中文结果")
        self.assertEqual(cached.call_args[0][0], "firered_asr_estimate")
        worker_config = cached.call_args[0][1]
        self.assertEqual(worker_config["subprocess_python"], "")
        self.assertNotIn("timeout_sec", worker_config)
        fake_client.transcribe.assert_called_once_with(
            "audio.wav", context=None
        )

    def test_run_returns_tool_result_with_asr_lineage(self):
        raw = {
            "text": "hello",
            "confidence": 0.8,
            "duration_s": 1.0,
            "rtf": 0.1,
            "timestamps": [{"token": "hello", "start_s": 0, "end_s": 1}],
            "model": {"name": "FireRedASR2-AED"},
            "runtime": {"device": "cpu"},
        }
        fake_client = mock.Mock()
        fake_client.transcribe.return_value = raw
        result = firered_asr.run("clip.wav", client=fake_client)

        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tag_path, "speaker.asr_transcript")
        self.assertEqual(result.tool_name, firered_asr.TOOL_NAME)
        self.assertEqual(result.value["text"], "hello")
        self.assertEqual(result.evidence["source_version"], firered_asr.SOURCE_VERSION)
        self.assertEqual(result.evidence["timestamps"][0]["token"], "hello")
        self.assertEqual(result.evidence["model"]["name"], "FireRedASR2-AED")
        fake_client.transcribe.assert_called_once_with("clip.wav", context=None)


if __name__ == "__main__":
    unittest.main()
