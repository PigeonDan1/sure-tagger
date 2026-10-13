"""Integration checks for the speaker-v2 language routing boundary."""

from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock
import wave

from tagger.pipelines import speaker_evidence
from tagger.pipelines.speaker_evidence import SpeakerEvidenceConfig, run_record
from tagger.tools.base import ToolResult
from tagger.tools.language_content.speaker_segment_lid import SpeakerSegmentLidConfig
from tagger.tools.speaker_v2.contracts import build_evidence, build_missing_evidence
from tagger.tools.speaker_v2.nemotron3_diarization import (
    Nemotron3DiarizationConfig,
)
from tagger.tools.speaker_v2.timeline import summarize_timeline


def _write_wav(path, duration_sec=1.0):
    sample_rate = 16000
    frame_count = int(round(sample_rate * duration_sec))
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(struct.pack("<%sh" % frame_count, *([0] * frame_count)))


def _record(audio_path, sample_id):
    return {
        "corpus": {
            "dataset_name": "speaker-v2-test",
            "source_urls": {
                "article": [],
                "github": [],
                "huggingface": [],
                "dataset_card": [],
            },
            "native_metadata": {},
        },
        "sample": {
            "sample_id": sample_id,
            "audio": {"path": str(audio_path)},
            "text": {"transcript": "native text is not an inference input"},
            "native_metadata": {},
        },
    }


def _lid_evidence(scope, language=None):
    if language is None:
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "language_identification",
            "firered_lid",
            "test-lid",
            "audio_language_identifier",
            ["language_route"],
            ["G_test_lid"],
            "LID failed",
        )
    return build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "language_identification",
        "firered_lid",
        "test-lid",
        "audio_language_identifier",
        ["language_route"],
        ["G_test_lid"],
        {"language": language, "language_confidence": 0.99},
        quality={"usable": True},
    )


def _segment_lid_evidence(scope, language):
    return build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "language_identification",
        "speaker_segment_firered_lid",
        "test-speaker-segment-lid",
        "speaker_aware_audio_language_identifier",
        ["language_route"],
        ["G_test_lid", "G_test_speaker_segment_lid"],
        {
            "language": language,
            "language_confidence": 0.88,
            "decision": "longest_clean_segment",
        },
        quality={"usable": True, "speaker_first": True},
    )


def _segments(scope, prefix, speaker_count, text=""):
    if not speaker_count:
        return []
    duration = scope["duration_sec"] / float(speaker_count)
    return [
        {
            "start_sec": index * duration,
            "end_sec": (index + 1) * duration,
            "speaker_id": "%s%s" % (prefix, index),
            "text": text if index == 0 else "",
            "overlap": False,
        }
        for index in range(speaker_count)
    ]


def _moss_evidence(scope, speaker_count=1):
    segments = _segments(scope, "M", speaker_count, text="中文候选")
    return build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "joint_speaker_timeline_and_text",
        "moss_transcribe_diarize",
        "test-moss",
        "joint_asr_diarizer",
        ["speaker_timeline", "joint_speaker_text"],
        ["G_test_moss"],
        {
            "timeline_summary": summarize_timeline(
                segments, scope["duration_sec"]
            ),
            "asr_transcript": "中文候选",
        },
        quality={"usable": True},
        applicability={"audio_sha256": scope["audio_sha256"]},
    )


def _firered_and_nemotron_evidence(scope, speaker_count=1):
    segments = _segments(scope, "N", speaker_count)
    asr = build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "asr_transcript",
        "fireredasr2_aed",
        "test-firered",
        "asr",
        ["asr_transcript"],
        ["G_test_firered"],
        {"text": "FireRed candidate", "asr_transcript": "FireRed candidate"},
        quality={"usable": True},
        applicability={"audio_sha256": scope["audio_sha256"]},
    )
    timeline = build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "speaker_timeline",
        "nemotron_3_diarization",
        "test-nemotron",
        "neural_diarization_model",
        ["speaker_timeline"],
        ["G_test_nemotron"],
        {
            "timeline_summary": summarize_timeline(
                segments, scope["duration_sec"]
            ),
            "raw_turns": segments,
        },
        quality={"usable": True},
        applicability={"audio_sha256": scope["audio_sha256"]},
    )
    return asr, timeline


def _firered_evidence(scope):
    return _firered_and_nemotron_evidence(scope)[0]


def _qwen3_evidence(scope):
    return build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "asr_transcript",
        "qwen3_asr_0_6b",
        "test-qwen3-asr",
        "asr",
        ["asr_transcript"],
        ["G_test_qwen3_asr"],
        {"text": "Qwen3 candidate", "asr_transcript": "Qwen3 candidate"},
        quality={"usable": True},
        applicability={"audio_sha256": scope["audio_sha256"]},
    )


def _nemotron_evidence(scope, speaker_count=1):
    return _firered_and_nemotron_evidence(scope, speaker_count=speaker_count)[1]


def _config(**overrides):
    values = {
        "moss_config": object(),
        "nemotron_config": Nemotron3DiarizationConfig(
            subprocess_python="", device="cuda"
        ),
        "enable_moss": True,
        "enable_vad": False,
        "enable_campplus": False,
        "enable_whisper": False,
        "enable_sortformer": False,
        "enable_pyannote": False,
        "enable_ecapa": False,
        "enable_brouhaha": False,
        "enable_speaker_profile": False,
        "verify_model_assets": False,
    }
    values.update(overrides)
    return SpeakerEvidenceConfig(**values)


class SpeakerV2LanguageBoundaryTest(unittest.TestCase):
    def test_artifact_sample_id_does_not_replace_evidence_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            sample_id = "business-sample"
            artifact_sample_id = "business-sample-sample-1"

            def collect_asr(source, scope, *_args, **_kwargs):
                if source == "moss":
                    return _moss_evidence(scope, speaker_count=2)
                if source == "qwen3_asr":
                    return _qwen3_evidence(scope)
                return _firered_evidence(scope)

            segment_result = ToolResult(
                tag_path="language_content.language",
                value="fr",
                tool_name="firered_lid_detector",
                method="speaker segment LID",
                status="estimated",
                confidence=0.9,
                tool_type="model",
                evidence={
                    "decision": "longest_clean_segment",
                    "fallback_reason": None,
                },
            )
            with mock.patch.object(
                speaker_evidence,
                "run_speaker_segment_lid",
                return_value=segment_result,
            ) as segment_lid, mock.patch.object(
                speaker_evidence,
                "_collect_asr_evidence",
                side_effect=collect_asr,
            ), mock.patch.object(
                speaker_evidence,
                "collect_nemotron_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _nemotron_evidence(
                    scope, speaker_count=2
                ),
            ):
                result = run_record(
                    _record(audio_path, sample_id),
                    root,
                    root / "output",
                    _config(
                        speaker_segment_lid_config=SpeakerSegmentLidConfig(
                            enabled=True,
                            keep_artifacts=True
                        )
                    ),
                    context={},
                    artifact_root=root / "artifacts",
                    artifact_sample_id=artifact_sample_id,
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sample_id"], sample_id)
        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(
            segment_lid.call_args[1]["sample_id"],
            artifact_sample_id,
        )

    def test_speaker_segment_lid_drives_route_before_full_audio_lid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)

            def collect_asr(source, scope, *_args, **_kwargs):
                if source == "moss":
                    return _moss_evidence(scope, speaker_count=1)
                if source == "qwen3_asr":
                    return _qwen3_evidence(scope)
                return _firered_evidence(scope)

            with mock.patch.object(
                speaker_evidence,
                "collect_firered_lid_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _lid_evidence(
                    scope, "zh"
                ),
            ) as full_lid, mock.patch.object(
                speaker_evidence,
                "collect_speaker_segment_lid_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _segment_lid_evidence(
                    scope, "fr"
                ),
            ), mock.patch.object(
                speaker_evidence,
                "_collect_asr_evidence",
                side_effect=collect_asr,
            ), mock.patch.object(
                speaker_evidence,
                "collect_nemotron_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _nemotron_evidence(
                    scope, speaker_count=1
                ),
            ):
                result = run_record(
                    _record(audio_path, "speaker-first-route"),
                    root,
                    root / "output",
                    _config(
                        speaker_segment_lid_config=SpeakerSegmentLidConfig(
                            enabled=True
                        )
                    ),
                    context={},
                )

        self.assertEqual(result["language_route"], "non_chinese_or_unknown")
        full_lid.assert_not_called()
        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(result["language_source"], "speaker_segment_firered_lid")
        self.assertEqual(result["asr_selected_source"], "qwen3_asr_0_6b")
        self.assertEqual(
            result["selected_timeline"]["source"],
            "nemotron_3_diarization",
        )

    def test_chinese_single_speaker_consensus_runs_both_and_uses_firered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            calls = []
            lid_finished = threading.Event()
            moss_started = threading.Event()
            nemotron_started = threading.Event()

            def collect_lid(scope, *_args, **_kwargs):
                self.assertTrue(moss_started.is_set())
                self.assertTrue(nemotron_started.is_set())
                calls.append("lid")
                lid_finished.set()
                return _lid_evidence(scope, "zh mandarin")

            def collect_asr(source, scope, *_args, **_kwargs):
                calls.append(source)
                if source == "moss":
                    moss_started.set()
                    self.assertTrue(nemotron_started.wait(1.0))
                    return _moss_evidence(scope, speaker_count=1)
                self.assertEqual(source, "firered_asr")
                self.assertTrue(moss_started.is_set())
                self.assertTrue(nemotron_started.is_set())
                return _firered_evidence(scope)

            def collect_nemotron(scope, *_args, **_kwargs):
                calls.append("nemotron")
                nemotron_started.set()
                self.assertTrue(moss_started.wait(1.0))
                return _nemotron_evidence(scope, speaker_count=1)

            with mock.patch.object(
                speaker_evidence,
                "collect_firered_lid_evidence",
                side_effect=collect_lid,
            ), mock.patch.object(
                speaker_evidence,
                "_collect_asr_evidence",
                side_effect=collect_asr,
            ), mock.patch.object(
                speaker_evidence,
                "collect_nemotron_evidence",
                side_effect=collect_nemotron,
            ) as nemotron:
                result = run_record(
                    _record(audio_path, "chinese"),
                    root,
                    root / "output",
                    _config(),
                    context={},
                )

        self.assertCountEqual(calls[0:2], ["moss", "nemotron"])
        self.assertEqual(calls[2], "lid")
        self.assertEqual(calls[3], "firered_asr")
        nemotron.assert_called_once()
        self.assertEqual(result["language_route"], "chinese")
        self.assertEqual(result["detected_language"], "zh mandarin")
        self.assertEqual(result["speaker_asr_transcript"], "FireRed candidate")
        self.assertEqual(result["speaker"]["speaker_count"], 1)
        self.assertEqual(
            result["asr_route"]["selection_basis"],
            "single_speaker_consensus",
        )
        self.assertTrue(result["asr_route"]["consensus_override_applied"])

    def test_chinese_multi_speaker_result_keeps_original_moss_asr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            asr_sources = []

            def collect_asr(source, scope, *_args, **_kwargs):
                asr_sources.append(source)
                self.assertEqual(source, "moss")
                return _moss_evidence(scope, speaker_count=2)

            with mock.patch.object(
                speaker_evidence,
                "collect_firered_lid_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _lid_evidence(
                    scope, "zh"
                ),
            ), mock.patch.object(
                speaker_evidence,
                "_collect_asr_evidence",
                side_effect=collect_asr,
            ) as moss, mock.patch.object(
                speaker_evidence,
                "collect_nemotron_evidence",
                side_effect=lambda scope, *_args, **_kwargs: (
                    _nemotron_evidence(scope, speaker_count=1)
                ),
            ) as nemotron:
                result = run_record(
                    _record(audio_path, "chinese-multi"),
                    root,
                    root / "output",
                    _config(),
                    context={},
                )

        moss.assert_called_once()
        nemotron.assert_called_once()
        self.assertEqual(asr_sources, ["moss"])
        self.assertEqual(result["language_route"], "chinese")
        self.assertEqual(result["speaker_asr_transcript"], "中文候选")
        self.assertEqual(result["speaker"]["speaker_count"], 2)
        self.assertEqual(result["asr_route"]["selection_basis"], "language_route")
        self.assertFalse(result["asr_route"]["consensus_override_applied"])

    def test_non_chinese_uses_qwen3_asr_and_nemotron_speaker_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "sample.wav"
            _write_wav(audio_path)
            asr_sources = []

            def collect_asr(source, scope, *_args, **_kwargs):
                asr_sources.append(source)
                if source == "moss":
                    return _moss_evidence(scope, speaker_count=1)
                self.assertEqual(source, "qwen3_asr")
                return _qwen3_evidence(scope)

            with mock.patch.object(
                speaker_evidence,
                "collect_firered_lid_evidence",
                side_effect=lambda scope, *_args, **_kwargs: _lid_evidence(
                    scope, "fr"
                ),
            ), mock.patch.object(
                speaker_evidence,
                "_collect_asr_evidence",
                side_effect=collect_asr,
            ) as asr, mock.patch.object(
                speaker_evidence,
                "collect_nemotron_evidence",
                side_effect=lambda scope, *_args, **_kwargs: (
                    _nemotron_evidence(scope, speaker_count=1)
                ),
            ) as nemotron:
                result = run_record(
                    _record(audio_path, "french"),
                    root,
                    root / "output",
                    _config(),
                    context={},
                )

        self.assertEqual(asr_sources, ["moss", "qwen3_asr"])
        self.assertEqual(asr.call_count, 2)
        nemotron.assert_called_once()
        self.assertEqual(result["language_route"], "non_chinese_or_unknown")
        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(result["speaker_asr_transcript"], "Qwen3 candidate")
        self.assertEqual(result["speaker"]["speaker_count"], 1)
        self.assertEqual(result["asr_route"]["selection_basis"], "language_route")
        self.assertTrue(
            result["asr_route"]["single_speaker_consensus"]["all_at_most_one"]
        )
        self.assertFalse(result["asr_route"]["consensus_override_applied"])

    def test_models_can_still_be_disabled_independently(self):
        for disabled in ("qwen3_asr", "nemotron"):
            with self.subTest(
                disabled=disabled
            ), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                audio_path = root / "sample.wav"
                _write_wav(audio_path)
                overrides = {"enable_%s" % disabled: False}

                def collect_asr(source, scope, *_args, **_kwargs):
                    if source == "moss":
                        return _moss_evidence(scope)
                    return _qwen3_evidence(scope)

                with mock.patch.object(
                    speaker_evidence,
                    "collect_firered_lid_evidence",
                    side_effect=lambda scope, *_args, **_kwargs: _lid_evidence(
                        scope, "fr"
                    ),
                ), mock.patch.object(
                    speaker_evidence,
                    "_collect_asr_evidence",
                    side_effect=collect_asr,
                ) as asr, mock.patch.object(
                    speaker_evidence,
                    "collect_nemotron_evidence",
                    side_effect=lambda scope, *_args, **_kwargs: (
                        _nemotron_evidence(scope)
                    ),
                ) as nemotron:
                    result = run_record(
                        _record(audio_path, "disabled-%s" % disabled),
                        root,
                        root / "output",
                        _config(**overrides),
                        context={},
                    )

                if disabled == "qwen3_asr":
                    self.assertEqual(asr.call_count, 1)
                    self.assertEqual(asr.call_args[0][0], "moss")
                    nemotron.assert_called_once()
                    self.assertEqual(result["speaker_asr_transcript"], "")
                    self.assertTrue(result["asr_route"]["failed_closed"])
                else:
                    self.assertEqual(asr.call_count, 2)
                    nemotron.assert_not_called()
                    self.assertEqual(
                        result["speaker_asr_transcript"], "Qwen3 candidate"
                    )
                    self.assertFalse(result["asr_route"]["failed_closed"])
                self.assertFalse(result["asr_route"]["fallback"])


if __name__ == "__main__":
    unittest.main()
