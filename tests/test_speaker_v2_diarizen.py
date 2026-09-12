"""Tests for the fixed OpenASR DiariZen plus FireRed adapter."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from tagger.pipelines import speaker_evidence
from tagger.tools.speaker_v2 import diarizen_timeline


SCOPE = {
    "sample_id": "sample",
    "audio_path": "/unused/audio.wav",
    "audio_sha256": "a" * 64,
    "duration_sec": 2.0,
    "sample_rate_hz": 16000,
    "channels": 1,
}


class Completed:
    def __init__(self, stdout, stderr, returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class DiariZenTimelineAdapterTest(unittest.TestCase):
    def test_config_record_is_reconstructable_and_has_fixed_fields(self):
        config = diarizen_timeline.DiariZenTimelineConfig(
            runtime_executable="runtime/openasr",
            diarizen_pack="models/diarizen.oasr",
            redimnet_pack="models/redimnet.oasr",
            firered_asr_pack="models/firered.oasr",
            forced_aligner_pack="models/aligner.oasr",
            timeout_sec=17,
        )
        record = config.to_record()

        self.assertEqual(
            set(record),
            {
                "runtime_executable",
                "diarizen_pack",
                "redimnet_pack",
                "firered_asr_pack",
                "forced_aligner_pack",
                "timeout_sec",
                "openasr_version",
                "runtime_executable_sha256",
            },
        )
        self.assertEqual(
            diarizen_timeline.DiariZenTimelineConfig(**record).to_record(),
            record,
        )
        self.assertNotIn("command_args", record)
        self.assertEqual(record["openasr_version"], "0.1.40")

    def test_default_paths_match_deployed_layout(self):
        config = diarizen_timeline.DiariZenTimelineConfig()

        self.assertTrue(
            config.runtime_executable.endswith(
                "/.runtime/openasr/0.1.40/openasr"
            )
        )
        self.assertTrue(
            config.diarizen_pack.endswith(
                "/models/diarizen-large-s80-v2/"
                "diarizen-large-s80-v2-fp16.oasr"
            )
        )
        self.assertTrue(
            config.redimnet_pack.endswith(
                "/models/redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr"
            )
        )
        self.assertTrue(
            config.firered_asr_pack.endswith(
                "/models/firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr"
            )
        )
        self.assertTrue(
            config.forced_aligner_pack.endswith(
                "/models/qwen3-forced-aligner-0.6b/"
                "qwen3-forced-aligner-0.6b-q4_k.oasr"
            )
        )

    def test_parser_normalizes_text_turns_and_overlap(self):
        result = diarizen_timeline.normalize_result(
            '{"text":" hello world "}',
            "log stage=batch turns=2 speakers=2\n"
            "log turn start=0 end=1.25 speaker=S0 overlap=false\n"
            "log turn start=1.0 end=2 speaker=S1 overlap=true details=x\n",
            ["openasr", "transcribe"],
        )

        self.assertEqual(result["text"], "hello world")
        self.assertEqual(
            result["segments"],
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.25,
                    "speaker_id": "S0",
                    "overlap": False,
                },
                {
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "speaker_id": "S1",
                    "overlap": True,
                },
            ],
        )
        self.assertEqual(result["runtime"]["debug_turn_count"], 2)
        self.assertEqual(result["runtime"]["debug_speaker_count"], 2)

    def test_zero_turn_silence_is_a_valid_timeline(self):
        result = diarizen_timeline.normalize_result(
            '{"text":""}', "stage=batch turns=0 speakers=0\n"
        )

        self.assertEqual(result["text"], "")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["runtime"]["asr_status"], "empty")
        self.assertEqual(result["runtime"]["timeline_status"], "usable")

    def test_stdout_and_trace_are_parsed_independently(self):
        trace_failure = diarizen_timeline.normalize_result(
            '{"text":"usable transcript"}', "trace unavailable"
        )
        stdout_failure = diarizen_timeline.normalize_result(
            "not-json", "stage=batch turns=0 speakers=0"
        )

        self.assertEqual(trace_failure["text"], "usable transcript")
        self.assertIsNone(trace_failure["segments"])
        self.assertEqual(
            trace_failure["runtime"]["timeline_status"], "invalid"
        )
        self.assertIsNone(stdout_failure["text"])
        self.assertEqual(stdout_failure["segments"], [])
        self.assertEqual(stdout_failure["runtime"]["asr_status"], "invalid")

    def test_trace_rejects_header_turn_speaker_and_interval_mismatches(self):
        cases = (
            (
                "stage=batch turns=2 speakers=1\n"
                "turn start=0 end=1 speaker=S0 overlap=0",
                "declares 2 turns",
            ),
            (
                "stage=batch turns=1 speakers=2\n"
                "turn start=0 end=1 speaker=S0 overlap=0",
                "declares 2 speakers",
            ),
            (
                "stage=batch turns=1 speakers=1\n"
                "turn start=1 end=1 speaker=S0 overlap=0",
                "invalid time bounds",
            ),
            (
                "stage=batch turns=0 speakers=0\n"
                "stage=batch turns=0 speakers=0",
                "exactly one batch header",
            ),
            (
                "stage=batch turns=1 speakers=1\n"
                "turn start=0 end=1 speaker=S0 overlap=true",
                "overlap=true without an overlapping different-speaker turn",
            ),
        )
        for trace, message in cases:
            with self.subTest(message=message):
                result = diarizen_timeline.normalize_result(
                    '{"text":"valid"}', trace
                )
                self.assertEqual(result["text"], "valid")
                self.assertIsNone(result["segments"])
                self.assertIn(message, result["runtime"]["timeline_error"])

    def test_client_uses_fixed_argv_and_explicit_pack_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = root / "openasr"
            audio = root / "audio.wav"
            diarizen = root / "diarizen.oasr"
            redimnet = root / "redimnet.oasr"
            firered = root / "firered.oasr"
            aligner = root / "aligner.oasr"
            for path in (
                runtime,
                audio,
                diarizen,
                redimnet,
                firered,
                aligner,
            ):
                path.write_bytes(b"x")
            runtime.chmod(0o755)
            config = diarizen_timeline.DiariZenTimelineConfig(
                runtime,
                diarizen,
                redimnet,
                firered,
                aligner,
                timeout_sec=23,
            )
            completed = Completed(
                '{"text":"hello"}', "stage=batch turns=0 speakers=0"
            )
            runtime_identity = {
                "engine_version": "0.1.40",
                "executable_sha256": diarizen_timeline.OPENASR_EXECUTABLE_SHA256,
                "executable_size_bytes": 42268384,
                "executable_version_output": "openasr 0.1.40",
                "executable_verified": True,
            }
            with mock.patch.object(
                diarizen_timeline,
                "_verify_openasr_runtime",
                return_value=runtime_identity,
            ), mock.patch.object(
                diarizen_timeline.subprocess, "run", return_value=completed
            ) as runner:
                result = diarizen_timeline.DiariZenTimelineClient(
                    config
                ).transcribe_and_diarize(audio, context={})

        args, kwargs = runner.call_args
        self.assertEqual(
            args[0],
            [
                str(runtime.resolve()),
                "transcribe",
                str(audio.resolve()),
                "--model-pack",
                str(firered.resolve()),
                "--diarize",
                "--offline",
                "--no-punctuate",
                "--format",
                "json",
            ],
        )
        self.assertEqual(
            kwargs["env"]["OPENASR_DIARIZEN_PACK"], str(diarizen.resolve())
        )
        self.assertEqual(
            kwargs["env"]["OPENASR_REDIMNET_PACK"], str(redimnet.resolve())
        )
        self.assertEqual(
            kwargs["env"]["OPENASR_FORCED_ALIGNER_PACK"],
            str(aligner.resolve()),
        )
        self.assertEqual(kwargs["env"]["OPENASR_DIARIZE_DEBUG"], "1")
        self.assertEqual(kwargs["timeout"], 23)
        self.assertTrue(kwargs["universal_newlines"])
        self.assertEqual(result["text"], "hello")
        self.assertTrue(result["runtime"]["executable_verified"])
        self.assertEqual(
            result["runtime"]["executable_sha256"],
            diarizen_timeline.OPENASR_EXECUTABLE_SHA256,
        )

    def test_runtime_verification_rejects_unpinned_executable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = Path(tmpdir) / "openasr"
            runtime.write_bytes(b"not the pinned runtime")
            runtime.chmod(0o755)
            with mock.patch.object(
                diarizen_timeline.subprocess, "run"
            ) as runner:
                with self.assertRaisesRegex(
                    diarizen_timeline.DiariZenTimelineError,
                    "runtime size mismatch",
                ):
                    diarizen_timeline._verify_openasr_runtime(
                        runtime,
                        diarizen_timeline.OPENASR_EXECUTABLE_SHA256,
                    )
            runner.assert_not_called()

    def test_client_rejects_missing_runtime_before_subprocess(self):
        config = diarizen_timeline.DiariZenTimelineConfig(
            runtime_executable="/definitely/missing/openasr",
        )
        with mock.patch.object(diarizen_timeline.subprocess, "run") as runner:
            with self.assertRaisesRegex(
                diarizen_timeline.DiariZenTimelineError,
                "runtime executable is missing",
            ):
                diarizen_timeline.DiariZenTimelineClient(
                    config
                ).transcribe_and_diarize("/missing/audio.wav")
        runner.assert_not_called()

    def test_client_rejects_missing_forced_aligner_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = root / "openasr"
            audio = root / "audio.wav"
            diarizen = root / "diarizen.oasr"
            redimnet = root / "redimnet.oasr"
            firered = root / "firered.oasr"
            for path in (runtime, audio, diarizen, redimnet, firered):
                path.write_bytes(b"x")
            runtime.chmod(0o755)
            config = diarizen_timeline.DiariZenTimelineConfig(
                runtime,
                diarizen,
                redimnet,
                firered,
                root / "missing-aligner.oasr",
            )
            with mock.patch.object(
                diarizen_timeline.subprocess, "run"
            ) as runner:
                with self.assertRaisesRegex(
                    diarizen_timeline.DiariZenTimelineError,
                    "forced aligner model pack is missing",
                ):
                    diarizen_timeline.DiariZenTimelineClient(
                        config
                    ).transcribe_and_diarize(audio)
        runner.assert_not_called()

    def test_shared_model_slot_uses_configured_capacity(self):
        context = {
            "_subprocess_worker_slots": {
                diarizen_timeline.MODEL_SLOT_KEY: 2,
            }
        }
        first = diarizen_timeline._model_limiter(context)
        second = diarizen_timeline._model_limiter(context)

        self.assertIs(first, second)
        self.assertTrue(first.acquire(False))
        self.assertTrue(first.acquire(False))
        self.assertFalse(first.acquire(False))
        first.release()
        first.release()

    def test_model_slot_creation_is_thread_safe(self):
        context = {
            "_subprocess_workers_lock": threading.Lock(),
            "_subprocess_worker_slots": {
                diarizen_timeline.MODEL_SLOT_KEY: 1,
            },
        }
        values = []

        def get_limiter():
            values.append(diarizen_timeline._model_limiter(context))

        threads = [threading.Thread(target=get_limiter) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len({id(item) for item in values}), 1)


class DiariZenEvidenceTest(unittest.TestCase):
    def _collect(self, result):
        config = diarizen_timeline.DiariZenTimelineConfig()
        with mock.patch.object(
            speaker_evidence, "diarize_diarizen", return_value=result
        ):
            return speaker_evidence.collect_diarizen_evidence(
                SCOPE, config, verify_model_asset=False
            )

    def test_valid_text_and_trace_create_two_usable_sources(self):
        asr, timeline = self._collect(
            {
                "text": "hello",
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                        "speaker_id": "S0",
                        "overlap": False,
                    }
                ],
                "runtime": {
                    "asr_status": "usable",
                    "timeline_status": "usable",
                },
            }
        )

        self.assertEqual(asr["status"], "estimated")
        self.assertEqual(asr["source"]["name"], "fireredasr2_aed")
        self.assertEqual(timeline["status"], "estimated")
        self.assertEqual(timeline["source"]["name"], "diarizen_large_s80_v2")
        self.assertEqual(timeline["payload"]["raw_turns"][0]["overlap"], False)

    def test_empty_text_does_not_invalidate_zero_turn_timeline(self):
        asr, timeline = self._collect(
            {
                "text": "",
                "segments": [],
                "runtime": {
                    "asr_status": "empty",
                    "timeline_status": "usable",
                },
            }
        )

        self.assertEqual(asr["status"], "missing")
        self.assertEqual(timeline["status"], "estimated")
        self.assertTrue(timeline["quality"]["empty_timeline"])
        self.assertEqual(
            timeline["payload"]["timeline_summary"]["observed_speaker_count"],
            0,
        )

    def test_component_parse_failures_are_independent(self):
        asr, timeline = self._collect(
            {
                "text": "valid text",
                "segments": None,
                "runtime": {
                    "asr_status": "usable",
                    "timeline_status": "invalid",
                    "timeline_error": "bad trace",
                },
            }
        )
        self.assertEqual(asr["status"], "estimated")
        self.assertEqual(timeline["status"], "missing")

        asr, timeline = self._collect(
            {
                "text": None,
                "segments": [],
                "runtime": {
                    "asr_status": "invalid",
                    "asr_error": "bad stdout",
                    "timeline_status": "usable",
                },
            }
        )
        self.assertEqual(asr["status"], "missing")
        self.assertEqual(timeline["status"], "estimated")

    def test_command_failure_marks_both_sources_missing(self):
        config = diarizen_timeline.DiariZenTimelineConfig()
        with mock.patch.object(
            speaker_evidence,
            "diarize_diarizen",
            side_effect=RuntimeError("runtime failed"),
        ):
            asr, timeline = speaker_evidence.collect_diarizen_evidence(
                SCOPE, config, verify_model_asset=False
            )

        self.assertEqual(asr["status"], "missing")
        self.assertEqual(timeline["status"], "missing")
        self.assertIn("runtime failed", asr["quality"]["reason"])
        self.assertIn("runtime failed", timeline["quality"]["reason"])

    def test_collector_verifies_all_four_pinned_packs(self):
        config = diarizen_timeline.DiariZenTimelineConfig(
            diarizen_pack="diarizen.oasr",
            redimnet_pack="redimnet.oasr",
            firered_asr_pack="firered.oasr",
            forced_aligner_pack="aligner.oasr",
        )
        output = {
            "text": "hello",
            "segments": [],
            "runtime": {
                "asr_status": "usable",
                "timeline_status": "usable",
            },
        }
        with mock.patch.object(
            speaker_evidence, "_verify_pack_asset"
        ) as verify, mock.patch.object(
            speaker_evidence, "diarize_diarizen", return_value=output
        ):
            asr, timeline = speaker_evidence.collect_diarizen_evidence(
                SCOPE, config, verify_model_asset=True
            )

        self.assertEqual(verify.call_count, 4)
        self.assertEqual(
            verify.call_args_list[0][0],
            (
                "diarizen.oasr",
                diarizen_timeline.MODEL_PACK_SHA256,
                diarizen_timeline.MODEL_PACK_SIZE_BYTES,
            ),
        )
        self.assertEqual(
            verify.call_args_list[3][0],
            (
                "aligner.oasr",
                diarizen_timeline.FORCED_ALIGNER_PACK_SHA256,
                diarizen_timeline.FORCED_ALIGNER_PACK_SIZE_BYTES,
            ),
        )
        self.assertEqual(asr["status"], "estimated")
        self.assertEqual(timeline["status"], "estimated")
        self.assertTrue(asr["quality"]["model_asset_verified"])
        self.assertTrue(timeline["quality"]["model_asset_verified"])

    def test_pack_size_is_checked_before_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pack.oasr"
            path.write_bytes(b"123")
            with mock.patch.object(
                speaker_evidence, "_verify_model_asset"
            ) as verify:
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    speaker_evidence._verify_pack_asset(path, "hash", 4)
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
