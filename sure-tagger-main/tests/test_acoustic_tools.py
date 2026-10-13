import gzip
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import wave

from scripts.run_c50_method_comparison import compare_record as compare_c50_record
from tagger.input_schema import InputSchemaError, validate_input_record
from tagger.tools.base import ToolResult
from tagger.pipelines.tagging import (
    FULL_STAGES,
    _merge_tags,
    _stages_for_tag_paths,
    audit_audio_quality,
    audit_basic_acoustic,
    audit_room_acoustic,
    audit_speaker,
    audit_sound_field_scene,
    build_arg_parser,
    empty_tags,
    resolve_audio_path,
    run_manifest as run_tagging_manifest,
    tag_record as tag_sample_record,
)
from tagger.tools.acoustic_io import probe_audio_info
from tagger.tools.audio_quality.brouhaha_signal_estimator import (
    BrouhahaConfig,
    run as run_brouhaha_signal_estimator,
)
from tagger.tools.basic_acoustic.brouhaha_vad_silence_detector import (
    run as run_brouhaha_vad_silence_detector,
)
from tagger.tools.audio_quality.dnsmos_quality_estimator import (
    DnsmosConfig,
    run as run_dnsmos_quality_estimator,
)
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    FireRedVadConfig,
    FireRedVadError,
    run as run_firered_vad_silence_detector,
    speech_segments_to_silence_segments,
    validate_silence_segments,
)
from tagger.tools.basic_acoustic.nonspeech_ratio_calculator import run as run_nonspeech_ratio
from tagger.tools.room_acoustic.c50_estimator import (
    run as run_c50_estimator,
)
from tagger.tools.sound_field_scene.dass_categories import (
    build_category_composition,
    classify_dass_label,
)
from tagger.tools.sound_field_scene.dass_noise_type_detector import (
    DassNoiseTypeConfig,
    DassNoiseTypeError,
    run as run_dass_noise_type_detector,
    select_noise_type_events,
    validate_dass_output,
)
from tagger.tools.sound_field_scene.firered_aed_detector import (
    FireRedAedConfig,
    FireRedAedError,
    run as run_firered_aed_detector,
    validate_aed_output,
)
from tagger.tools.room_acoustic.rir_estimator import (
    RecRirConfig,
    RecRirError,
    run as run_recrir_rir_estimator,
)
from tagger.tools.room_acoustic.rt60_estimator import (
    run as run_rt60_estimator,
)
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
)
from tagger.tools.speaker.metrics import (
    SpeakerMetricsConfig,
    build_metadata_from_timeline,
    public_results_from_metadata,
)
from tagger.tools.speaker.moss_diarizer import (
    MossDiarizeConfig,
    MossDiarizeSubprocessClient,
    parse_moss_text,
)


class AcousticToolsTest(unittest.TestCase):
    def setUp(self):
        self.speaker_v2_patcher = mock.patch(
            "tagger.pipelines.tagging.run_speaker_v2_record",
            return_value={
                "speaker": {
                    "speaker_count": 1,
                    "speaker_present": True,
                    "multi_speaker": False,
                    "speaker_change_count": 0,
                    "speaker_change": False,
                    "overlap_ratio": 0.0,
                    "speaker_overlap": False,
                },
                "run_profile": "quality-shadow",
                "policy_version": "policy-v1",
                "policy_hash": "hash",
                "fusion_artifact": "fusion.json.gz",
                "artifacts": {},
            },
        )
        self.speaker_v2_patcher.start()
        self.addCleanup(self.speaker_v2_patcher.stop)

    def test_probe_audio_info_reads_header_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=8000, channels=1, duration_sec=1.0)

            info = probe_audio_info(path)

            self.assertEqual(info.sample_rate_hz, 8000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.frame_count, 8000)
            self.assertAlmostEqual(info.duration_sec, 1.0)

    def test_closed_input_schema_accepts_raw_only_record(self):
        record = make_record("audio.wav")
        validate_input_record(record)

    def test_closed_input_schema_rejects_extra_fields(self):
        record = make_record("audio.wav")
        record["sample"]["audio"]["uri"] = "derived"
        with self.assertRaises(InputSchemaError):
            validate_input_record(record)

    def test_resolve_audio_path_prefers_existing_cwd_relative_path(self):
        sample = {"audio": {"path": "phase1_asr_samples/manifest.jsonl"}}
        resolved = resolve_audio_path(sample, "phase1_asr_samples")
        self.assertEqual(resolved, Path.cwd() / "phase1_asr_samples/manifest.jsonl")

    def test_tagging_record_returns_acoustic_and_sound_field_tags_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)
            config = FireRedVadConfig(
                model_dir=str(Path(tmpdir) / "missing_model"),
                subprocess_python="",
            )
            brouhaha_config = BrouhahaConfig(
                model_path=str(Path(tmpdir) / "missing_brouhaha.ckpt"),
                repo_dir=str(Path(tmpdir) / "missing_brouhaha_repo"),
                subprocess_python="",
            )
            recrir_config = RecRirConfig(
                repo_dir=str(Path(tmpdir) / "missing_recrir_repo"),
                config_path=str(Path(tmpdir) / "missing_recrir.toml"),
                checkpoint_path=str(Path(tmpdir) / "missing_recrir.tar"),
                subprocess_python="",
            )

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                config,
                brouhaha_config,
                recrir_config,
                dnsmos_config=DnsmosConfig(subprocess_python=""),
                dnsmos_client=FakeDnsmosClient(
                    {"sig": 4.1, "bak": 3.2, "ovrl": 3.5, "p808": 3.8}
                ),
                firered_aed_config=FireRedAedConfig(
                    model_dir=str(Path(tmpdir) / "missing_aed_model"),
                    subprocess_python="",
                ),
                firered_aed_client=FakeFireRedAedClient(
                    {
                        "event2timestamps": {
                            "speech": [[0.1, 0.9]],
                            "singing": [],
                            "music": [[0.2, 0.8]],
                        },
                        "event2ratio": {
                            "speech": 0.8,
                            "singing": 0.0,
                            "music": 0.6,
                        },
                    }
                ),
                dass_config=DassNoiseTypeConfig(
                    threshold=0.5,
                    subprocess_python="",
                ),
                dass_client=FakeDassNoiseTypeClient(
                    make_dass_output(0.8, "Traffic noise")
                ),
            )

            self.assertEqual(
                set(tags.keys()),
                set(
                    [
                        "basic_acoustic",
                        "audio_quality",
                        "room_acoustic",
                        "sound_field_scene",
                        "speaker",
                        "language_content",
                        "deterministic_derived",
                    ]
                ),
            )
            self.assertEqual(
                set(tags["basic_acoustic"].keys()),
                set(
                    [
                        "duration_sec",
                        "sample_rate_hz",
                        "channels",
                        "nonspeech_ratio",
                        "silence_segments",
                    ]
                ),
            )
            self.assertEqual(tags["basic_acoustic"]["duration_sec"], 1.0)
            self.assertEqual(tags["basic_acoustic"]["sample_rate_hz"], 16000)
            self.assertEqual(tags["basic_acoustic"]["channels"], 1)
            self.assertIsNone(tags["basic_acoustic"]["nonspeech_ratio"])
            self.assertIsNone(tags["basic_acoustic"]["silence_segments"])
            self.assertIsNone(tags["audio_quality"]["snr_db"])
            self.assertEqual(tags["audio_quality"]["dnsmos_sig"], 4.1)
            self.assertEqual(tags["audio_quality"]["dnsmos_bak"], 3.2)
            self.assertEqual(tags["audio_quality"]["dnsmos_ovrl"], 3.5)
            self.assertEqual(tags["audio_quality"]["dnsmos_p808"], 3.8)
            self.assertEqual(
                set(tags["room_acoustic"].keys()),
                set(
                    [
                        "far_field",
                        "rt60_sec",
                        "c50_db",
                    ]
                ),
            )
            self.assertIsNone(tags["room_acoustic"]["far_field"])
            self.assertIsNone(tags["room_acoustic"]["rt60_sec"])
            self.assertIsNone(tags["room_acoustic"]["c50_db"])
            self.assertEqual(
                set(tags["sound_field_scene"].keys()),
                set(
                    [
                        "speech_music_events",
                        "music_present",
                        "external_noise_type",
                        "noise_composition",
                    ]
                ),
            )
            self.assertEqual(
                tags["sound_field_scene"]["speech_music_events"],
                ["speech", "music"],
            )
            self.assertTrue(tags["sound_field_scene"]["music_present"])
            self.assertEqual(
                tags["sound_field_scene"]["external_noise_type"],
                ["mechanical"],
            )
            self.assertEqual(
                tags["sound_field_scene"]["noise_composition"],
                {
                    "music": [],
                    "animal": [],
                    "mechanical": ["Traffic noise"],
                    "nature": [],
                    "formless": [],
                    "channel_environment": [],
                },
            )

    def test_tagging_record_populates_deterministic_text_tags_without_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = make_record("missing.wav")
            record["sample"]["text"]["transcript"] = "Um hello hello."

            tags = tag_sample_record(record, tmpdir)

            self.assertEqual(
                set(tags["language_content"]),
                {"language", "word_count", "punctuation", "repetition", "filler"},
            )
            # language_content.language now comes from FireRed LID, an
            # audio-dependent stage; without audio it stays null.
            self.assertIsNone(tags["language_content"]["language"])
            self.assertEqual(tags["language_content"]["word_count"], 3)
            self.assertEqual(
                tags["language_content"]["punctuation"],
                {
                    "punctuation_count": 1,
                    "has_terminal_punctuation": True,
                },
            )
            self.assertEqual(
                tags["language_content"]["repetition"],
                {
                    "has_repetition": True,
                    "repetition_count": 1,
                },
            )
            self.assertEqual(tags["language_content"]["filler"], 1)

    def test_empty_transcript_reuses_speaker_lid_for_language_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = ""
            asr_text = "Um hello hello."
            public_speaker = {
                "speaker_count": 1,
                "speaker_present": True,
                "multi_speaker": False,
                "speaker_change_count": 0,
                "speaker_change": False,
                "overlap_ratio": 0.0,
                "speaker_overlap": False,
                "profiles": None,
                "asr_transcript": None,
            }
            lid_run = mock.Mock(
                side_effect=AssertionError(
                    "speaker-routed LID must be reused without a direct LID run"
                )
            )
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value={
                    "speaker": public_speaker,
                    "speaker_asr_transcript": asr_text,
                    "detected_language": "zh-mandarin",
                    "language_confidence": 0.98,
                    "language_error": None,
                    "language_evidence_id": "lid-evidence",
                    "language_route": "chinese",
                    "run_profile": "quality-shadow",
                    "policy_version": "policy-v1",
                    "policy_hash": "hash",
                    "fusion_artifact": "fusion.json.gz",
                    "artifacts": {},
                },
            ) as run_speaker, mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": lid_run},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    speaker_config=object(),
                    selected_tag_paths=["language_content"],
                )

            self.assertEqual(run_speaker.call_count, 1)
            lid_run.assert_not_called()
            self.assertEqual(
                tags["language_content"]["language"], "zh-mandarin"
            )
            self.assertEqual(tags["language_content"]["word_count"], 3)
            self.assertEqual(tags["language_content"]["filler"], 1)
            self.assertEqual(
                tags["language_content"]["punctuation"],
                {"punctuation_count": 1, "has_terminal_punctuation": True},
            )
            self.assertTrue(tags["language_content"]["repetition"]["has_repetition"])
            self.assertEqual(tags["speaker"]["asr_transcript"], asr_text)

    def test_empty_transcript_language_tag_starts_speaker_and_reuses_full_lid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = ""

            lid_run = mock.Mock(
                side_effect=AssertionError(
                    "speaker-routed full-audio LID must be reused"
                )
            )
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value={
                    "speaker": {
                        "speaker_count": 1,
                        "speaker_present": True,
                        "multi_speaker": False,
                        "speaker_change_count": 0,
                        "speaker_change": False,
                        "overlap_ratio": 0.0,
                        "speaker_overlap": False,
                    },
                    "detected_language": "de",
                    "language_confidence": 0.91,
                    "language_error": None,
                    "language_evidence_id": "lid-evidence",
                    "language_route": "non_chinese_or_unknown",
                    "selected_timeline": {
                        "duration_sec": 2.0,
                        "activity_segments": [
                            {
                                "speaker_id": "speaker_1",
                                "start_sec": 0.0,
                                "end_sec": 2.0,
                            }
                        ],
                    },
                },
            ) as run_speaker, mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": lid_run},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    selected_tag_paths=["language_content.language"],
                    speaker_segment_lid_config=SpeakerSegmentLidConfig(
                        enabled=True
                    ),
                )

            run_speaker.assert_called_once()
            lid_run.assert_not_called()
            self.assertEqual(tags["language_content"]["language"], "de")
            self.assertIsNone(tags["language_content"]["word_count"])
            self.assertTrue(all(value is None for value in tags["speaker"].values()))

    def test_empty_transcript_speaker_and_language_reuse_speaker_lid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = ""
            lid_run = mock.Mock(
                side_effect=AssertionError(
                    "speaker-routed LID must be reused without a direct LID run"
                )
            )
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value={
                    "speaker": {
                        "speaker_count": 1,
                        "speaker_present": True,
                        "multi_speaker": False,
                        "speaker_change_count": 0,
                        "speaker_change": False,
                        "overlap_ratio": 0.0,
                        "speaker_overlap": False,
                    },
                    "speaker_asr_transcript": "Hello world.",
                    "detected_language": "de",
                    "language_confidence": 0.91,
                    "language_error": None,
                    "language_evidence_id": "lid-evidence",
                    "language_route": "non_chinese_or_unknown",
                },
            ) as run_speaker, mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": lid_run},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    speaker_config=object(),
                    selected_tag_paths=["speaker", "language_content.language"],
                )

            run_speaker.assert_called_once()
            lid_run.assert_not_called()
            self.assertEqual(tags["language_content"]["language"], "de")
            self.assertEqual(tags["speaker"]["speaker_count"], 1)

    def test_multi_speaker_language_uses_segment_lid_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=5.0)
            record = make_record(str(path))
            selected_timeline = {
                "source": "diarizen_official",
                "evidence_id": "timeline-evidence",
                "duration_sec": 5.0,
                "activity_segments": [
                    {"speaker_id": "A", "start_sec": 0.0, "end_sec": 3.0},
                    {"speaker_id": "B", "start_sec": 3.0, "end_sec": 5.0},
                ],
            }
            speaker_result = {
                "speaker": {
                    "speaker_count": 2,
                    "speaker_present": True,
                    "multi_speaker": True,
                    "speaker_change_count": 1,
                    "speaker_change": True,
                    "overlap_ratio": 0.0,
                    "speaker_overlap": False,
                },
                "detected_language": "mn",
                "language_confidence": 0.71,
                "language_error": None,
                "language_evidence_id": "lid-evidence",
                "language_route": "non_chinese_or_unknown",
                "selected_timeline": selected_timeline,
            }
            segment_result = ToolResult(
                tag_path="language_content.language",
                value="en",
                tool_name="firered_lid_detector",
                method="FireRed LID (longest clean speaker speech segment)",
                status="estimated",
                confidence=0.95,
                tool_type="model",
                evidence={"decision": "longest_clean_segment", "fallback_reason": None},
            )
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value=speaker_result,
            ), mock.patch(
                "tagger.pipelines.tagging.run_speaker_segment_lid",
                return_value=segment_result,
            ) as segment_lid:
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    speaker_config=object(),
                    selected_tag_paths=["speaker", "language_content.language"],
                    speaker_segment_lid_config=SpeakerSegmentLidConfig(
                        enabled=True
                    ),
                )

            self.assertEqual(tags["language_content"]["language"], "en")
            self.assertEqual(segment_lid.call_args[0][1], selected_timeline)
            self.assertEqual(
                segment_lid.call_args[0][2]["language"],
                "mn",
            )

    def test_disabled_segment_lid_keeps_language_only_standalone_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = ""
            lid_run = mock.Mock(
                return_value=ToolResult(
                    tag_path="language_content.language",
                    value="de",
                    tool_name="firered_lid_detector",
                    method="FireRed LID",
                    status="estimated",
                    confidence=0.91,
                    tool_type="model",
                )
            )
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
            ) as run_speaker, mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": lid_run},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    selected_tag_paths=["language_content.language"],
                    speaker_segment_lid_config=SpeakerSegmentLidConfig(
                        enabled=False
                    ),
                )

            run_speaker.assert_not_called()
            lid_run.assert_called_once()
            self.assertEqual(tags["language_content"]["language"], "de")

    def test_segment_lid_can_disable_full_audio_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=5.0)
            record = make_record(str(path))
            speaker_result = {
                "speaker": {"speaker_count": 2, "multi_speaker": True},
                "detected_language": "mn",
                "language_confidence": 0.71,
                "selected_timeline": {
                    "duration_sec": 5.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 3.0},
                        {"speaker_id": "B", "start_sec": 3.0, "end_sec": 5.0},
                    ],
                },
            }
            failed_segment_result = ToolResult(
                tag_path="language_content.language",
                value=None,
                tool_name="firered_lid_detector",
                method="FireRed LID (longest clean speaker speech segment)",
                status="failed",
                tool_type="model",
                evidence={"fallback_reason": "speaker_lid_failed"},
            )
            full_audio_lid = mock.Mock()
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value=speaker_result,
            ), mock.patch(
                "tagger.pipelines.tagging.run_speaker_segment_lid",
                return_value=failed_segment_result,
            ), mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": full_audio_lid},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    speaker_config=object(),
                    selected_tag_paths=["speaker", "language_content.language"],
                    speaker_segment_lid_config=SpeakerSegmentLidConfig(
                        enabled=True,
                        fallback_to_full_audio=False
                    ),
                )

            full_audio_lid.assert_not_called()
            self.assertIsNone(tags["language_content"]["language"])

    def test_tagging_pipeline_uses_speaker_v2_public_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meeting.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=5.0)
            record = make_record(str(path))
            record["sample"]["native_metadata"] = {
                "speaker_segments": [
                    {"speaker": "native-only", "start": 0.0, "end": 5.0}
                ]
            }
            speaker_config = object()
            artifact_dir = Path(tmpdir) / "custom-artifacts"
            speaker_v2_output = {
                "speaker_count": 2,
                "multi_speaker": True,
                "speaker_change_count": 3,
                "speaker_change": True,
                "overlap_ratio": 0.25,
                "speaker_overlap": True,
                "profiles": [
                    {
                        "speaker_id": "speaker_1",
                        "speech_rate": {
                            "band": "normal",
                            "value": 4.2,
                            "unit": "zh_char_per_sec",
                        },
                        "gender": "female",
                        "pitch": "mid",
                        "speaker_volume": "normal",
                    }
                ],
            }
            result = {
                "speaker": speaker_v2_output,
                "speaker_asr_transcript": "  MOSS full sample transcript.  ",
                "run_profile": "quality-shadow",
                "policy_version": "policy-v1",
                "policy_hash": "hash",
                "fusion_artifact": "fusion.json.gz",
                "artifacts": {"sample_dir": "speaker_v2/sample"},
            }

            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value=result,
            ) as run_speaker:
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    speaker_config=speaker_config,
                    artifact_dir=artifact_dir,
                    selected_tag_paths=["speaker"],
                )

            expected_speaker = dict(
                speaker_v2_output,
                speaker_present=True,
                asr_transcript="MOSS full sample transcript.",
            )
            self.assertEqual(tags["speaker"], expected_speaker)
            args, kwargs = run_speaker.call_args
            self.assertIs(args[0], record)
            self.assertEqual(Path(args[1]), Path(tmpdir))
            self.assertEqual(Path(args[2]), Path(tmpdir))
            self.assertIs(args[3], speaker_config)
            self.assertIsInstance(kwargs["context"], dict)
            self.assertEqual(kwargs["artifact_root"], artifact_dir)
            self.assertTrue(kwargs["artifact_sample_id"].endswith("sample-1"))

    def test_tagging_pipeline_skips_only_language_stages_without_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noise.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = ""
            record["sample"]["native_metadata"]["speech_segments"] = [
                {"start_sec": 0.5, "end_sec": 1.5}
            ]
            dnsmos_client = FakeDnsmosClient(
                {
                    "sig": 3.0,
                    "bak": 3.0,
                    "ovrl": 3.0,
                    "p808": 3.0,
                }
            )
            recrir_client = FakeRecRirClient(
                {
                    "sample_rate_hz": 16000,
                    "samples": [1.0, 0.0],
                }
            )
            lid_run = mock.Mock(
                return_value=ToolResult(
                    tag_path="language_content.language",
                    value="en",
                    tool_name="firered_lid_detector",
                    method="FireRed LID",
                    status="estimated",
                    confidence=0.9,
                    tool_type="model",
                )
            )

            with mock.patch.dict(
                "tagger.pipelines.tagging.FIRERED_LID_LANGUAGE_CONTENT_TOOL",
                {"run": lid_run},
            ):
                tags = tag_sample_record(
                    record,
                    tmpdir,
                    dnsmos_client=dnsmos_client,
                    recrir_client=recrir_client,
                    selected_tag_paths=[
                        "language_content",
                        "basic_acoustic.nonspeech_ratio",
                        "audio_quality.dnsmos_ovrl",
                        "speaker",
                        "room_acoustic.rt60_sec",
                    ],
                    **missing_external_model_configs(tmpdir)
                )

            self.assertEqual(dnsmos_client.call_count, 1)
            self.assertEqual(recrir_client.call_count, 1)
            lid_run.assert_called_once()
            self.assertEqual(tags["language_content"]["language"], "en")
            self.assertTrue(
                all(
                    tags["language_content"][field] is None
                    for field in ("word_count", "punctuation", "repetition", "filler")
                )
            )
            self.assertEqual(
                tags["basic_acoustic"]["silence_segments"],
                [
                    {"start_sec": 0.0, "end_sec": 0.5},
                    {"start_sec": 1.5, "end_sec": 2.0},
                ],
            )
            self.assertEqual(tags["basic_acoustic"]["nonspeech_ratio"], 0.5)
            self.assertEqual(tags["audio_quality"]["dnsmos_ovrl"], 3.0)
            self.assertEqual(tags["speaker"]["speaker_count"], 1)
            self.assertTrue(tags["speaker"]["speaker_present"])
            self.assertFalse(tags["speaker"]["multi_speaker"])

    def test_fire_red_speech_segments_convert_to_silence_segments(self):
        segments = speech_segments_to_silence_segments(
            [[0.44, 1.82]],
            duration_sec=2.32,
        )

        self.assertEqual(
            segments,
            [
                {"start_sec": 0.0, "end_sec": 0.44},
                {"start_sec": 1.82, "end_sec": 2.32},
            ],
        )

    def test_fire_red_silence_detector_accepts_injected_fire_red_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)

            result = run_firered_vad_silence_detector(
                path,
                duration_sec=2.0,
                context={},
                config=FireRedVadConfig(model_dir=str(Path(tmpdir) / "model")),
                client=FakeFireRedClient([[0.5, 1.25], [1.5, 2.0]]),
            )

            self.assertEqual(result.tag_path, "basic_acoustic.silence_segments")
            self.assertEqual(result.tool_name, "firered_vad_silence_detector")
            self.assertEqual(
                result.value,
                [
                    {"start_sec": 0.0, "end_sec": 0.5},
                    {"start_sec": 1.25, "end_sec": 1.5},
                ],
            )

    def test_fire_red_aed_maps_event_names_and_music_to_public_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)

            results = run_firered_aed_detector(
                path,
                duration_sec=2.0,
                context={},
                config=FireRedAedConfig(
                    model_dir=str(Path(tmpdir) / "model"),
                    subprocess_python="",
                ),
                client=FakeFireRedAedClient(
                    {
                        "event2timestamps": {
                            "speech": [[0.1, 1.8]],
                            "singing": [[0.5, 1.0]],
                            "music": [],
                        },
                        "event2ratio": {
                            "speech": 0.85,
                            "singing": 0.25,
                            "music": 0.0,
                        },
                    }
                ),
            )

            by_path = {result.tag_path: result for result in results}
            self.assertEqual(
                set(by_path),
                set(
                    [
                        "sound_field_scene.speech_music_events",
                        "sound_field_scene.music_present",
                    ]
                ),
            )
            self.assertEqual(
                by_path["sound_field_scene.speech_music_events"].value,
                ["speech", "singing"],
            )
            self.assertFalse(by_path["sound_field_scene.music_present"].value)
            self.assertEqual(
                by_path["sound_field_scene.music_present"].evidence[
                    "event_ratios"
                ]["speech"],
                0.85,
            )

    def test_fire_red_aed_gates_music_present_by_min_music_ratio(self):
        def run_with_ratio(ratio):
            results = run_firered_aed_detector(
                "audio.wav",
                duration_sec=10.0,
                config=FireRedAedConfig(subprocess_python=""),
                client=FakeFireRedAedClient(
                    {
                        "event2timestamps": {
                            "speech": [[0.1, 9.0]],
                            "singing": [],
                            "music": [[0.5, 1.0]],
                        },
                        "event2ratio": {
                            "speech": 0.89,
                            "singing": 0.0,
                            "music": ratio,
                        },
                    }
                ),
            )
            return {r.tag_path: r for r in results}

        # Music segments exist but the ratio stays below the default 0.10
        # floor: music is gated out of the public events, music_present is
        # False, and the gate is recorded in the internal evidence.
        by_path = run_with_ratio(0.05)
        self.assertFalse(by_path["sound_field_scene.music_present"].value)
        self.assertEqual(
            by_path["sound_field_scene.speech_music_events"].value, ["speech"]
        )
        self.assertEqual(
            by_path["sound_field_scene.music_present"].evidence["event_gates"],
            {
                "singing": {"min_ratio": 0.10, "gated": False},
                "music": {"min_ratio": 0.10, "gated": True},
            },
        )

        # Substantive music passes the gate.
        by_path = run_with_ratio(0.25)
        self.assertTrue(by_path["sound_field_scene.music_present"].value)
        self.assertEqual(
            by_path["sound_field_scene.speech_music_events"].value, ["speech", "music"]
        )
        self.assertEqual(
            by_path["sound_field_scene.music_present"].evidence["event_gates"],
            {
                "singing": {"min_ratio": 0.10, "gated": False},
                "music": {"min_ratio": 0.10, "gated": False},
            },
        )

    def test_fire_red_aed_gates_singing_by_min_singing_ratio(self):
        results = run_firered_aed_detector(
            "audio.wav",
            duration_sec=10.0,
            config=FireRedAedConfig(subprocess_python=""),
            client=FakeFireRedAedClient(
                {
                    "event2timestamps": {
                        "speech": [[0.1, 9.0]],
                        "singing": [[0.5, 0.9]],
                        "music": [],
                    },
                    "event2ratio": {
                        "speech": 0.89,
                        "singing": 0.04,
                        "music": 0.0,
                    },
                }
            ),
        )
        by_path = {r.tag_path: r for r in results}

        # Short singing segments below the 0.10 floor are gated out of the
        # public events while raw segments stay in the evidence.
        self.assertEqual(
            by_path["sound_field_scene.speech_music_events"].value, ["speech"]
        )
        self.assertFalse(by_path["sound_field_scene.music_present"].value)
        self.assertEqual(
            by_path["sound_field_scene.speech_music_events"].evidence["event_gates"],
            {
                "singing": {"min_ratio": 0.10, "gated": True},
                "music": {"min_ratio": 0.10, "gated": False},
            },
        )

    def test_fire_red_aed_min_ratio_floors_must_be_probabilities(self):
        with self.assertRaises(FireRedAedError):
            FireRedAedConfig(min_music_ratio=1.5)
        with self.assertRaises(FireRedAedError):
            FireRedAedConfig(min_music_ratio=-0.1)
        with self.assertRaises(FireRedAedError):
            FireRedAedConfig(min_singing_ratio=1.5)
        with self.assertRaises(FireRedAedError):
            FireRedAedConfig(min_singing_ratio=-0.1)

    def test_fire_red_aed_rejects_invalid_event_output(self):
        with self.assertRaises(FireRedAedError):
            validate_aed_output(
                {
                    "event2timestamps": {
                        "speech": [[0.0, 1.1]],
                        "singing": [],
                        "music": [],
                    },
                    "event2ratio": {
                        "speech": 1.1,
                        "singing": 0.0,
                        "music": 0.0,
                    },
                },
                duration_sec=1.0,
            )

    def test_fire_red_aed_returns_empty_event_list_when_nothing_is_detected(self):
        output = {
            "event2timestamps": {"speech": [], "singing": [], "music": []},
            "event2ratio": {"speech": 0.0, "singing": 0.0, "music": 0.0},
        }
        results = run_firered_aed_detector(
            "audio.wav",
            duration_sec=1.0,
            config=FireRedAedConfig(subprocess_python=""),
            client=FakeFireRedAedClient(output),
        )
        by_path = {result.tag_path: result.value for result in results}

        self.assertEqual(by_path["sound_field_scene.speech_music_events"], [])
        self.assertFalse(by_path["sound_field_scene.music_present"])

    def test_dass_noise_type_detector_uses_inclusive_threshold(self):
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.5, subprocess_python=""),
            client=FakeDassNoiseTypeClient(make_dass_output(0.5, "Traffic noise")),
        )
        by_path = {result.tag_path: result for result in results}
        result = by_path["sound_field_scene.external_noise_type"]

        self.assertEqual(result.tag_path, "sound_field_scene.external_noise_type")
        self.assertEqual(result.value, ["mechanical"])
        self.assertEqual(result.evidence["max_noise_score"], 0.5)
        self.assertEqual(result.evidence["winning_event"]["display_name"], "Traffic noise")

    def test_dass_noise_type_detector_default_threshold_is_calibrated(self):
        # Regressed on 2026-08-24: the default was lowered from the AudioSet
        # convention 0.50 to 0.25 because DASS-medium noise scores are soft.
        self.assertEqual(DassNoiseTypeConfig().threshold, 0.25)
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(subprocess_python=""),
            client=FakeDassNoiseTypeClient(make_dass_output(0.3, "Traffic noise")),
        )
        by_path = {result.tag_path: result for result in results}

        self.assertEqual(
            by_path["sound_field_scene.external_noise_type"].value, ["mechanical"]
        )

    def test_dass_composition_threshold_default_aligns_with_categories(self):
        # Regressed on 2026-08-25: the composition default was 0.30 while the
        # category default was 0.25, so labels scoring in the 0.25-0.30 band
        # (e.g. CHiME4 / TUT mechanical labels) produced a present category
        # with an empty composition bucket. The defaults are now aligned so a
        # present category always has a non-empty composition bucket.
        self.assertEqual(DassNoiseTypeConfig().composition_threshold, 0.25)
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(subprocess_python=""),
            client=FakeDassNoiseTypeClient(make_dass_output(0.3, "Traffic noise")),
        )
        by_path = {result.tag_path: result for result in results}

        self.assertEqual(
            by_path["sound_field_scene.noise_composition"].value["mechanical"],
            ["Traffic noise"],
        )

    def test_dass_noise_type_detector_returns_empty_list_below_threshold(self):
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.5, subprocess_python=""),
            client=FakeDassNoiseTypeClient(make_dass_output(0.499999, "Traffic noise")),
        )
        by_path = {result.tag_path: result for result in results}

        # External categories derive from the full vector at the (higher)
        # threshold: 0.499999 stays below 0.5, so no category is present.
        self.assertEqual(by_path["sound_field_scene.external_noise_type"].value, [])
        # Composition has its own (lower) threshold, so the same event still
        # enters the composition bucket.
        self.assertEqual(
            by_path["sound_field_scene.noise_composition"].value["mechanical"],
            ["Traffic noise"],
        )

    def test_dass_noise_type_detector_returns_ranked_classes_above_threshold(self):
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.5, subprocess_python=""),
            client=FakeDassNoiseTypeClient(
                make_dass_output(
                    0.7,
                    "Traffic noise",
                    extra_events=[(0.6, "Vehicle"), (0.2, "Noise")],
                )
            ),
        )
        by_path = {result.tag_path: result for result in results}

        # Both events share the mechanical category; "Noise" (0.2) stays
        # below the threshold so channel_environment is absent.
        self.assertEqual(
            by_path["sound_field_scene.external_noise_type"].value,
            ["mechanical"],
        )

    def test_dass_detector_orders_external_categories_by_best_score(self):
        # Categories are derived from the full 527-class vector (the
        # exclusion policy does not apply) and ordered by each category's
        # best label score.
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.4, subprocess_python=""),
            client=FakeDassNoiseTypeClient(
                make_dass_output(
                    0.8,
                    "Traffic noise",
                    extra_vector_events=[
                        (0.9, "Rain"),
                        (0.6, "Dog"),
                        (0.4, "Music"),
                    ],
                )
            ),
        )
        by_path = {result.tag_path: result for result in results}

        self.assertEqual(
            by_path["sound_field_scene.external_noise_type"].value,
            ["nature", "mechanical", "animal", "music"],
        )

    def test_dass_detector_ignores_excluded_labels_in_external_categories(self):
        # Silence (formless) is excluded by policy and must not drive the
        # external category list, while Traffic noise (mechanical) does.
        # Regressed on 2026-08-24: deriving categories from the raw full
        # vector flagged clean-speech samples as formless via Silence.
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.25, subprocess_python=""),
            client=FakeDassNoiseTypeClient(
                make_dass_output(
                    0.6,
                    "Traffic noise",
                    extra_vector_events=[(0.9, "Silence")],
                )
            ),
        )
        by_path = {result.tag_path: result for result in results}

        self.assertEqual(
            by_path["sound_field_scene.external_noise_type"].value,
            ["mechanical"],
        )
        # Composition reads the full vector and is unaffected by the policy.
        self.assertEqual(
            by_path["sound_field_scene.noise_composition"].value["formless"],
            ["Silence"],
        )

    def test_dass_noise_type_selection_excludes_primary_speech_and_scene(self):
        summary = select_noise_type_events(
            [
                {"index": 0, "display_name": "Speech"},
                {"index": 1, "display_name": "Inside, small room"},
                {"index": 2, "display_name": "Music"},
            ],
            [0.99, 0.95, 0.51],
        )

        self.assertEqual(summary["max_noise_score"], 0.51)
        self.assertEqual(summary["winning_event"]["display_name"], "Music")

    def test_dass_noise_type_output_rejects_excluded_winner(self):
        with self.assertRaises(DassNoiseTypeError):
            validate_dass_output(make_dass_output(0.9, "Speech"))

    def test_dass_noise_type_selection_keeps_all_classes_without_exclusion(self):
        summary = select_noise_type_events(
            [
                {"index": 0, "display_name": "Speech"},
                {"index": 1, "display_name": "Silence"},
                {"index": 2, "display_name": "Music"},
            ],
            [0.99, 0.95, 0.51],
            exclude_classes=False,
        )

        self.assertEqual(summary["max_noise_score"], 0.99)
        self.assertEqual(summary["winning_event"]["display_name"], "Speech")
        self.assertEqual(
            [item["display_name"] for item in summary["top_noise_events"]],
            ["Speech", "Silence", "Music"],
        )

    def test_dass_noise_type_output_accepts_silence_winner_without_exclusion(self):
        summary = validate_dass_output(
            make_dass_output(0.9, "Silence"), exclude_classes=False
        )

        self.assertEqual(summary["winning_event"]["display_name"], "Silence")

    def test_dass_noise_type_output_accepts_speech_winner_without_exclusion(self):
        summary = validate_dass_output(
            make_dass_output(0.9, "Speech"), exclude_classes=False
        )

        self.assertEqual(summary["winning_event"]["display_name"], "Speech")

    def test_dass_noise_type_output_accepts_scene_winner_without_exclusion(self):
        summary = validate_dass_output(
            make_dass_output(0.9, "Inside, small room"), exclude_classes=False
        )

        self.assertEqual(summary["winning_event"]["display_name"], "Inside, small room")

    def test_dass_noise_type_detector_publishes_all_classes_without_exclusion(self):
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(
                threshold=0.5, exclude_classes=False, subprocess_python=""
            ),
            client=FakeDassNoiseTypeClient(make_dass_output(0.8, "Speech")),
        )
        by_path = {result.tag_path: result for result in results}
        result = by_path["sound_field_scene.external_noise_type"]

        self.assertEqual(result.tag_path, "sound_field_scene.external_noise_type")
        # Speech classifies into the human category, which is evidence-only:
        # it never enters the public external_noise_type value.
        self.assertEqual(result.value, [])
        self.assertEqual(result.evidence["winning_event"]["display_name"], "Speech")
        self.assertFalse(result.evidence["config"]["exclude_classes"])
        # Human labels never enter the public composition either, even with
        # exclusion disabled.
        self.assertEqual(
            by_path["sound_field_scene.noise_composition"].value["formless"],
            [],
        )

    def test_dass_tag_paths_are_not_selectable_stages(self):
        # Deprecated on 2026-08-25: the PANNs stage and its public
        # ``sound_field_scene.sound`` field were removed (noise_composition
        # supersedes them). Selecting "panns" or "sound_field_scene.sound"
        # must now raise instead of silently no-opping.
        self.assertNotIn("panns", FULL_STAGES)
        self.assertNotIn("panns", _stages_for_tag_paths(None))
        with self.assertRaises(ValueError):
            _stages_for_tag_paths(["panns"])
        with self.assertRaises(ValueError):
            _stages_for_tag_paths(["sound_field_scene.sound"])

    def test_tagging_pipeline_publishes_dass_all_classes_without_exclusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)
            configs = missing_external_model_configs(tmpdir)
            configs["dass_config"] = DassNoiseTypeConfig(
                threshold=0.5, exclude_classes=False, subprocess_python=""
            )

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                dass_client=FakeDassNoiseTypeClient(
                    make_dass_output(0.9, "Silence")
                ),
                **configs
            )

            self.assertEqual(
                tags["sound_field_scene"]["external_noise_type"], ["formless"]
            )
            self.assertEqual(
                tags["sound_field_scene"]["noise_composition"],
                {
                    "music": [],
                    "animal": [],
                    "mechanical": [],
                    "nature": [],
                    "formless": ["Silence"],
                    "channel_environment": [],
                },
            )

    def test_tagging_pipeline_rejects_deprecated_panns_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            with self.assertRaises(ValueError):
                tag_sample_record(
                    make_record(str(path)),
                    tmpdir,
                    selected_tag_paths=["panns"],
                )
            with self.assertRaises(ValueError):
                tag_sample_record(
                    make_record(str(path)),
                    tmpdir,
                    selected_tag_paths=["sound_field_scene.sound"],
                )

    def test_tagging_pipeline_nulls_dass_noise_type_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                **missing_external_model_configs(tmpdir),
            )

            self.assertIsNone(tags["sound_field_scene"]["external_noise_type"])
            self.assertIsNone(tags["sound_field_scene"]["noise_composition"])

    def test_classify_dass_label_assigns_expected_categories(self):
        self.assertEqual(classify_dass_label("Speech"), "human")
        self.assertEqual(
            classify_dass_label("Traffic noise, roadway noise"), "mechanical"
        )
        self.assertEqual(classify_dass_label("Music"), "music")
        self.assertEqual(classify_dass_label("Rain"), "nature")
        self.assertEqual(classify_dass_label("Silence"), "formless")
        self.assertEqual(classify_dass_label("Reverberation"), "channel_environment")
        self.assertEqual(classify_dass_label("class_123"), "other")

    def test_classify_dass_label_covers_deployed_checkpoint(self):
        import json

        checkpoint_config = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "DASS"
            / "saurabhati__DASS_medium_AudioSet_48.9"
            / "config.json"
        )
        if not checkpoint_config.is_file():
            self.skipTest("DASS checkpoint is not deployed in this source checkout")
        with open(str(checkpoint_config), encoding="utf-8") as source:
            id2label = json.load(source)["id2label"]
        for index in range(527):
            category = classify_dass_label(id2label[str(index)])
            self.assertIn(
                category,
                ("music", "animal", "mechanical", "nature", "formless",
                 "channel_environment", "human", "other"),
            )
        categories = {
            classify_dass_label(id2label[str(index)]) for index in range(527)
        }
        self.assertNotIn("other", categories)

    def test_dass_category_composition_ranks_filters_and_truncates(self):
        labels = [
            {"index": index, "display_name": "class_%03d" % index}
            for index in range(4)
        ]
        labels[1] = {"index": 1, "display_name": "Car"}
        labels[2] = {"index": 2, "display_name": "Truck"}
        labels[3] = {"index": 3, "display_name": "Rain"}
        scores = [0.0, 0.8, 0.6, 0.4]
        composition, category_events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=None
        )

        self.assertEqual(composition["mechanical"], ["Car", "Truck"])
        self.assertEqual(composition["nature"], [])
        self.assertEqual(
            category_events["mechanical"],
            [
                {"index": 1, "display_name": "Car", "score": 0.8},
                {"index": 2, "display_name": "Truck", "score": 0.6},
            ],
        )

    def test_dass_category_composition_truncates_per_category(self):
        labels = [
            {"index": index, "display_name": "class_%03d" % index}
            for index in range(6)
        ]
        names = ["Car", "Bus", "Train", "Bicycle", "Radio"]
        for offset, name in enumerate(names, start=1):
            labels[offset] = {"index": offset, "display_name": name}
        scores = [0.0, 0.9, 0.85, 0.8, 0.75, 0.7]
        composition, category_events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=None
        )

        self.assertEqual(composition["mechanical"], ["Car", "Bus", "Train"])
        self.assertEqual(
            [item["display_name"] for item in category_events["mechanical"]],
            ["Car", "Bus", "Train"],
        )

    def test_dass_category_composition_applies_music_gate(self):
        labels = [
            {"index": index, "display_name": "class_%03d" % index}
            for index in range(2)
        ]
        labels[1] = {"index": 1, "display_name": "Music"}
        scores = [0.0, 0.9]

        kept, _events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=True
        )
        gated, _events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=False
        )
        unguarded, events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=None
        )

        self.assertEqual(kept["music"], ["Music"])
        self.assertEqual(gated["music"], [])
        self.assertEqual(unguarded["music"], ["Music"])
        self.assertEqual(events["music"][0]["score"], 0.9)

    def test_dass_category_composition_keeps_human_in_evidence_only(self):
        labels = [
            {"index": index, "display_name": "class_%03d" % index}
            for index in range(2)
        ]
        labels[1] = {"index": 1, "display_name": "Laughter"}
        scores = [0.0, 0.8]
        composition, category_events = build_category_composition(
            labels, scores, threshold=0.5, top_k=3, music_present=None
        )

        for values in composition.values():
            self.assertEqual(values, [])
        self.assertEqual(
            category_events["human"],
            [{"index": 1, "display_name": "Laughter", "score": 0.8}],
        )

    def test_dass_detector_publishes_composition_with_music_gate(self):
        results = run_dass_noise_type_detector(
            "audio.wav",
            config=DassNoiseTypeConfig(threshold=0.5, subprocess_python=""),
            client=FakeDassNoiseTypeClient(
                make_dass_output(
                    0.8,
                    "Traffic noise",
                    extra_vector_events=[(0.6, "Music"), (0.7, "Laughter")],
                )
            ),
            music_present=False,
        )
        by_path = {result.tag_path: result for result in results}
        composition_result = by_path["sound_field_scene.noise_composition"]

        self.assertEqual(
            composition_result.value,
            {
                "music": [],
                "animal": [],
                "mechanical": ["Traffic noise"],
                "nature": [],
                "formless": [],
                "channel_environment": [],
            },
        )
        self.assertEqual(
            composition_result.evidence["music_gate"],
            {"aed_music_present": False, "gated": True},
        )
        self.assertEqual(
            composition_result.evidence["category_events"]["human"][0][
                "display_name"
            ],
            "Laughter",
        )

    def test_tagging_pipeline_gates_dass_music_by_firered_aed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)
            configs = missing_external_model_configs(tmpdir)
            configs["firered_aed_config"] = FireRedAedConfig(
                model_dir=str(Path(tmpdir) / "missing_aed"),
                subprocess_python="",
            )
            configs["firered_aed_client"] = FakeFireRedAedClient(
                {
                    "event2timestamps": {
                        "speech": [[0.1, 0.9]],
                        "singing": [],
                        "music": [],
                    },
                    "event2ratio": {
                        "speech": 0.8,
                        "singing": 0.0,
                        "music": 0.0,
                    },
                }
            )
            configs["dass_client"] = FakeDassNoiseTypeClient(
                make_dass_output(
                    0.8,
                    "Traffic noise",
                    extra_vector_events=[(0.6, "Music")],
                )
            )

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                **configs
            )

            self.assertFalse(tags["sound_field_scene"]["music_present"])
            self.assertEqual(
                tags["sound_field_scene"]["noise_composition"]["music"], []
            )
            self.assertEqual(
                tags["sound_field_scene"]["noise_composition"]["mechanical"],
                ["Traffic noise"],
            )

    def test_tagging_pipeline_keeps_dass_music_without_aed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)
            configs = missing_external_model_configs(tmpdir)
            configs["dass_config"] = DassNoiseTypeConfig(
                threshold=0.5, subprocess_python=""
            )
            configs["dass_client"] = FakeDassNoiseTypeClient(
                make_dass_output(
                    0.8,
                    "Traffic noise",
                    extra_vector_events=[(0.6, "Music")],
                )
            )

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                selected_tag_paths=["sound_field_scene.noise_composition"],
                **configs
            )

            self.assertIsNone(tags["sound_field_scene"]["music_present"])
            self.assertEqual(
                tags["sound_field_scene"]["noise_composition"]["music"], ["Music"]
            )
            # One DASS inference publishes both tag paths.
            self.assertEqual(
                tags["sound_field_scene"]["external_noise_type"],
                ["mechanical", "music"],
            )

    def test_nonspeech_ratio_is_calculated_from_segments_and_duration(self):
        result = run_nonspeech_ratio(
            [
                {"start_sec": 0.0, "end_sec": 0.5},
                {"start_sec": 1.5, "end_sec": 2.0},
            ],
            duration_sec=2.0,
        )

        self.assertEqual(result.tag_path, "basic_acoustic.nonspeech_ratio")
        self.assertEqual(result.value, 0.5)

    def test_silence_segments_reject_overlap(self):
        with self.assertRaises(FireRedVadError):
            validate_silence_segments(
                [
                    {"start_sec": 0.0, "end_sec": 0.7},
                    {"start_sec": 0.6, "end_sec": 1.0},
                ],
                duration_sec=1.0,
            )

    def test_silence_segments_reject_out_of_bounds(self):
        with self.assertRaises(FireRedVadError):
            validate_silence_segments(
                [{"start_sec": 0.0, "end_sec": 1.1}],
                duration_sec=1.0,
            )

    def test_brouhaha_estimator_aggregates_snr_and_c50(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            results = run_brouhaha_signal_estimator(
                path,
                context={},
                config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                ),
                client=FakeBrouhahaClient({"snr": [10.0, 14.0], "c50": [2.0, 4.0]}),
            )

            values = {result.tag_path: result.value for result in results}
            self.assertEqual(values["audio_quality.snr_db"], 12.0)
            self.assertEqual(values["internal.brouhaha_c50_db"], 3.0)

    def test_brouhaha_estimator_allows_partial_missing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            results = run_brouhaha_signal_estimator(
                path,
                context={},
                config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                ),
                client=FakeBrouhahaClient({"c50": [5.0]}),
            )

            by_path = {result.tag_path: result for result in results}
            self.assertIsNone(by_path["audio_quality.snr_db"].value)
            self.assertEqual(by_path["audio_quality.snr_db"].status, "failed")
            self.assertEqual(by_path["internal.brouhaha_c50_db"].value, 5.0)
            self.assertEqual(by_path["internal.brouhaha_c50_db"].status, "estimated")

    def test_brouhaha_estimator_rejects_nan_and_inf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            results = run_brouhaha_signal_estimator(
                path,
                context={},
                config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                ),
                client=FakeBrouhahaClient(
                    {"snr": [10.0, float("inf")], "c50": [float("nan")]}
                ),
            )

            values = {result.tag_path: result.value for result in results}
            self.assertIsNone(values["audio_quality.snr_db"])
            self.assertIsNone(values["internal.brouhaha_c50_db"])

    def test_dnsmos_estimator_maps_all_official_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            results = run_dnsmos_quality_estimator(
                path,
                context={},
                config=DnsmosConfig(subprocess_python=""),
                client=FakeDnsmosClient(
                    {
                        "sig": 4.1256789,
                        "bak": 3.25,
                        "ovrl": 3.75,
                        "p808": 4.0,
                        "num_hops": 1,
                        "audio_length_sec": 1.0,
                    }
                ),
            )

            by_path = {result.tag_path: result for result in results}
            self.assertEqual(by_path["audio_quality.dnsmos_sig"].value, 4.125679)
            self.assertEqual(by_path["audio_quality.dnsmos_bak"].value, 3.25)
            self.assertEqual(by_path["audio_quality.dnsmos_ovrl"].value, 3.75)
            self.assertEqual(by_path["audio_quality.dnsmos_p808"].value, 4.0)
            self.assertTrue(all(result.status == "estimated" for result in results))

    def test_dnsmos_estimator_nulls_missing_and_out_of_range_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            results = run_dnsmos_quality_estimator(
                path,
                config=DnsmosConfig(subprocess_python=""),
                client=FakeDnsmosClient(
                    {"sig": 5.1, "bak": float("nan"), "ovrl": 3.0}
                ),
            )

            by_path = {result.tag_path: result for result in results}
            self.assertIsNone(by_path["audio_quality.dnsmos_sig"].value)
            self.assertIsNone(by_path["audio_quality.dnsmos_bak"].value)
            self.assertEqual(by_path["audio_quality.dnsmos_ovrl"].value, 3.0)
            self.assertIsNone(by_path["audio_quality.dnsmos_p808"].value)
            self.assertEqual(by_path["audio_quality.dnsmos_p808"].status, "failed")

    def test_brouhaha_vad_silence_detector_converts_annotation_to_silence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)

            result = run_brouhaha_vad_silence_detector(
                path,
                duration_sec=2.0,
                context={},
                config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                ),
                client=FakeBrouhahaClient(
                    {
                        "annotation": [
                            {"start_sec": 0.4, "end_sec": 1.0},
                            {"start_sec": 1.25, "end_sec": 2.0},
                        ]
                    }
                ),
            )

            self.assertEqual(result.tag_path, "diagnostic.brouhaha_silence_segments")
            self.assertEqual(
                result.value,
                [
                    {"start_sec": 0.0, "end_sec": 0.4},
                    {"start_sec": 1.0, "end_sec": 1.25},
                ],
            )

    def test_brouhaha_vad_silence_detector_clips_annotation_to_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=2.0)

            result = run_brouhaha_vad_silence_detector(
                path,
                duration_sec=2.0,
                context={},
                config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                ),
                client=FakeBrouhahaClient(
                    {"annotation": [{"start_sec": -0.05, "end_sec": 2.2}]}
                ),
            )

            self.assertEqual(result.value, [])

    def test_recrir_rir_estimator_accepts_injected_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            result = run_recrir_rir_estimator(
                path,
                context={},
                config=RecRirConfig(max_rir_seconds=1.0),
                client=FakeRecRirClient(
                    {"sample_rate_hz": 16000, "samples": [0.0, 2.0, -1.0]}
                ),
            )

            self.assertEqual(result.tag_path, "room_acoustic.rir")
            self.assertEqual(result.tool_name, "rir_estimator")
            self.assertEqual(result.value["sample_rate_hz"], 16000)
            self.assertEqual(result.value["samples"], [0.0, 1.0, -0.5])

    def test_recrir_rir_estimator_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            with self.assertRaises(RecRirError):
                run_recrir_rir_estimator(
                    path,
                    context={},
                    config=RecRirConfig(max_rir_seconds=1.0),
                    client=FakeRecRirClient(
                        {"sample_rate_hz": 16000, "samples": [float("nan")]}
                    ),
                )

    def test_rt60_estimator_derives_value_from_rir(self):
        sample_rate_hz = 1000
        tau = 0.2
        samples = [
            math.exp(-float(index) / (sample_rate_hz * tau))
            for index in range(2000)
        ]

        result = run_rt60_estimator(
            {"sample_rate_hz": sample_rate_hz, "samples": samples}
        )

        self.assertEqual(result.tag_path, "room_acoustic.rt60_sec")
        self.assertGreater(result.value, 1.0)
        self.assertLess(result.value, 1.7)

    def test_c50_estimator_derives_value_from_rir(self):
        result = run_c50_estimator(
            {"sample_rate_hz": 1000, "samples": [1.0] * 50 + [0.5] * 50}
        )

        self.assertEqual(result.tag_path, "room_acoustic.c50_db")
        self.assertAlmostEqual(result.value, 6.0206, places=4)

    def test_room_acoustic_auditor_rejects_invalid_rt60_and_c50(self):
        room_acoustic = {
            "far_field": None,
            "rt60_sec": -1.0,
            "c50_db": float("nan"),
        }

        warnings = audit_room_acoustic(room_acoustic)

        self.assertIsNone(room_acoustic["rt60_sec"])
        self.assertIsNone(room_acoustic["c50_db"])
        self.assertEqual(
            warnings,
            [
                {"type": "invalid_room_acoustic_value", "field": "rt60_sec"},
                {"type": "invalid_room_acoustic_value", "field": "c50_db"},
            ],
        )

    def test_sound_field_auditor_rejects_invalid_event_label_lists(self):
        sound_field_scene = {
            "speech_music_events": ["music", "speech"],
            "music_present": True,
            "external_noise_type": ["mechanical", "human"],
            "noise_composition": {
                "music": [],
                "animal": [],
                "mechanical": ["Music"],
                "nature": [],
                "formless": [],
                "channel_environment": [],
            },
        }

        warnings = audit_sound_field_scene(sound_field_scene)

        self.assertIsNone(sound_field_scene["speech_music_events"])
        self.assertIsNone(sound_field_scene["external_noise_type"])
        self.assertIsNone(sound_field_scene["noise_composition"])
        self.assertEqual(
            [warning["field"] for warning in warnings],
            [
                "speech_music_events",
                "external_noise_type",
                "noise_composition",
            ],
        )

    def test_sound_field_auditor_drops_deprecated_sound_key(self):
        # Deprecated on 2026-08-25: the PANNs stage and its public
        # ``sound_field_scene.sound`` field were removed (noise_composition
        # supersedes them). Stale inputs carrying the key get it deleted; a
        # non-null stale value additionally yields a deprecation warning.
        sound_field_scene = {
            "speech_music_events": ["speech"],
            "music_present": False,
            "sound": ["Noise"],
            "external_noise_type": [],
            "noise_composition": {
                "music": [],
                "animal": [],
                "mechanical": [],
                "nature": [],
                "formless": [],
                "channel_environment": [],
            },
        }

        warnings = audit_sound_field_scene(sound_field_scene)

        self.assertNotIn("sound", sound_field_scene)
        self.assertEqual(
            warnings,
            [{"type": "deprecated_sound_field_scene_value", "field": "sound"}],
        )

        stale_null = {"sound": None}
        self.assertEqual(audit_sound_field_scene(stale_null), [])
        self.assertNotIn("sound", stale_null)

    def test_sound_field_auditor_accepts_valid_noise_composition(self):
        sound_field_scene = {
            "speech_music_events": ["speech"],
            "music_present": False,
            "external_noise_type": [],
            "noise_composition": {
                "music": [],
                "animal": ["Dog"],
                "mechanical": ["Traffic noise, roadway noise"],
                "nature": ["Rain"],
                "formless": ["Silence"],
                "channel_environment": ["Reverberation"],
            },
        }

        warnings = audit_sound_field_scene(sound_field_scene)

        self.assertEqual(warnings, [])
        self.assertEqual(
            sound_field_scene["noise_composition"]["mechanical"],
            ["Traffic noise, roadway noise"],
        )

    def test_audio_quality_auditor_rejects_invalid_dnsmos_scores(self):
        audio_quality = {
            "dnsmos_sig": 0.9,
            "dnsmos_bak": 5.1,
            "dnsmos_ovrl": float("nan"),
            "dnsmos_p808": 4.0,
        }

        warnings = audit_audio_quality(audio_quality)

        self.assertIsNone(audio_quality["dnsmos_sig"])
        self.assertIsNone(audio_quality["dnsmos_bak"])
        self.assertIsNone(audio_quality["dnsmos_ovrl"])
        self.assertEqual(audio_quality["dnsmos_p808"], 4.0)
        self.assertEqual(
            [warning["field"] for warning in warnings],
            ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"],
        )

    def test_tagging_pipeline_uses_injected_recrir_client_for_non_model_input_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=8000, channels=2, duration_sec=1.0)
            rir_samples = [
                math.exp(-float(index) / (16000 * 0.2))
                for index in range(16000)
            ]
            artifact_dir = Path(tmpdir) / "artifacts"

            tags = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                firered_vad_config=FireRedVadConfig(
                    model_dir=str(Path(tmpdir) / "missing_model"),
                    subprocess_python="",
                ),
                brouhaha_config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "missing_brouhaha.ckpt"),
                    repo_dir=str(Path(tmpdir) / "missing_brouhaha_repo"),
                    subprocess_python="",
                ),
                recrir_config=RecRirConfig(max_rir_seconds=1.0),
                recrir_client=FakeRecRirClient(
                    {"sample_rate_hz": 16000, "samples": rir_samples}
                ),
                artifact_dir=artifact_dir,
                dnsmos_config=DnsmosConfig(subprocess_python=""),
                dnsmos_client=FakeDnsmosClient(
                    {"sig": 4.0, "bak": 3.0, "ovrl": 3.5, "p808": 3.75}
                ),
                firered_aed_config=FireRedAedConfig(
                    model_dir=str(Path(tmpdir) / "missing_aed_model"),
                    subprocess_python="",
                ),
                firered_aed_client=FakeFireRedAedClient(
                    {
                        "event2timestamps": {
                            "speech": [[0.1, 0.9]],
                            "singing": [],
                            "music": [],
                        },
                        "event2ratio": {
                            "speech": 0.8,
                            "singing": 0.0,
                            "music": 0.0,
                        },
                    }
                ),
                selected_tag_paths=[
                    "recrir",
                    "firered_aed",
                ],
            )

            self.assertEqual(tags["basic_acoustic"]["sample_rate_hz"], 8000)
            self.assertEqual(tags["basic_acoustic"]["channels"], 2)
            self.assertNotIn("rir", tags["room_acoustic"])
            self.assertIsNotNone(tags["room_acoustic"]["rt60_sec"])
            self.assertIsNotNone(tags["room_acoustic"]["c50_db"])
            self.assertNotIn("snr_db", tags["basic_acoustic"])
            self.assertEqual(
                tags["sound_field_scene"]["speech_music_events"], ["speech"]
            )
            self.assertFalse(tags["sound_field_scene"]["music_present"])
            self.assertNotIn("sound", tags["sound_field_scene"])

            artifacts = list((artifact_dir / "rir").glob("*.rir.json.gz"))
            self.assertEqual(len(artifacts), 1)
            with gzip.open(str(artifacts[0]), "rt", encoding="utf-8") as source:
                artifact = json.load(source)
            self.assertEqual(artifact["sample_rate_hz"], 16000)
            self.assertEqual(len(artifact["samples"]), len(rir_samples))

    def test_tagging_pipeline_isolates_music_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)
            common = {
                "firered_vad_config": FireRedVadConfig(
                    model_dir=str(Path(tmpdir) / "missing_vad"),
                    subprocess_python="",
                ),
                "brouhaha_config": BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "missing_brouhaha.ckpt"),
                    repo_dir=str(Path(tmpdir) / "missing_brouhaha"),
                    subprocess_python="",
                ),
                "recrir_config": RecRirConfig(
                    repo_dir=str(Path(tmpdir) / "missing_recrir"),
                    config_path=str(Path(tmpdir) / "missing.toml"),
                    checkpoint_path=str(Path(tmpdir) / "missing.tar"),
                    subprocess_python="",
                ),
                "dnsmos_config": DnsmosConfig(subprocess_python=""),
                "firered_aed_config": FireRedAedConfig(subprocess_python=""),
            }

            fire_failure = tag_sample_record(
                make_record(str(path)),
                tmpdir,
                selected_tag_paths=["firered_aed"],
                firered_aed_client=FakeFireRedAedClient({}),
                **common
            )

            self.assertIsNone(fire_failure["sound_field_scene"]["music_present"])
            self.assertIsNone(fire_failure["sound_field_scene"]["speech_music_events"])
            self.assertNotIn("sound", fire_failure["sound_field_scene"])

    def test_c50_comparison_reports_brouhaha_and_recrir_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tone.wav"
            write_test_wav(path, sample_rate=16000, channels=1, duration_sec=1.0)

            result = compare_c50_record(
                make_record(str(path)),
                manifest_dir=tmpdir,
                context={},
                brouhaha_config=BrouhahaConfig(
                    model_path=str(Path(tmpdir) / "best.ckpt"),
                    repo_dir=str(Path(tmpdir) / "repo"),
                    subprocess_python="",
                ),
                recrir_config=RecRirConfig(max_rir_seconds=1.0),
                brouhaha_client=FakeBrouhahaClient(
                    {"snr": [10.0], "c50": [3.0]}
                ),
                recrir_client=FakeRecRirClient(
                    {
                        "sample_rate_hz": 1000,
                        "samples": [1.0] * 50 + [0.5] * 50,
                    }
                ),
            )

            self.assertEqual(result["brouhaha"]["status"], "ok")
            self.assertEqual(result["brouhaha"]["c50_db"], 3.0)
            self.assertEqual(result["recrir"]["status"], "ok")
            self.assertAlmostEqual(result["recrir"]["c50_db"], 6.0206, places=4)
            self.assertTrue(result["comparison"]["paired"])
            self.assertAlmostEqual(
                result["comparison"]["c50_delta_recrir_minus_brouhaha_db"],
                3.0206,
                places=4,
            )

    def test_speaker_metrics_derive_internal_summary_and_public_utterance_tags(self):
        metadata = build_metadata_from_timeline(
            [
                {"start_sec": 0.0, "end_sec": 1.2, "speaker_id": "S01"},
                {"start_sec": 0.8, "end_sec": 2.0, "speaker_id": "S02"},
            ],
            duration_sec=2.0,
            sample_id="u1",
            recording_id="meet1",
            target_units=[
                {"unit_id": "u0", "start_sec": 0.0, "end_sec": 0.75},
                {"unit_id": "u1", "start_sec": 0.7, "end_sec": 1.1},
            ],
            config=SpeakerMetricsConfig(min_speech_duration_sec=0.1),
        )

        summary = metadata["recording_summary"]
        self.assertEqual(summary["speaker_count"], 2)
        self.assertTrue(summary["multi_speaker"])
        self.assertEqual(summary["turn_count"], 2)
        self.assertEqual(summary["speaker_change_rate_per_min"], 30.0)
        self.assertEqual(summary["overlap_ratio_speech"], 0.2)
        self.assertEqual(summary["crosstalk_level"], "medium")
        self.assertEqual(summary["dominant_speaker_ratio"], 0.6)

        utterances = metadata["utterances"]
        self.assertEqual(utterances[0]["primary_speaker_id"], "spk_001")
        self.assertEqual(utterances[0]["active_speaker_count"], 1)
        self.assertFalse(utterances[0]["is_overlapped"])
        self.assertEqual(utterances[1]["active_speaker_count"], 2)
        self.assertEqual(utterances[1]["speaker_change_count"], 1)
        self.assertTrue(utterances[1]["speaker_change"])
        self.assertTrue(utterances[1]["is_overlapped"])
        self.assertEqual(utterances[1]["overlap_ratio"], 0.75)

        public_values = {
            result.tag_path: result.value
            for result in public_results_from_metadata(metadata)
        }
        self.assertEqual(public_values["speaker.speaker_count"], 2)
        self.assertTrue(public_values["speaker.multi_speaker"])
        self.assertEqual(public_values["speaker.speaker_change_count"], 1)
        self.assertTrue(public_values["speaker.speaker_change"])
        self.assertEqual(public_values["speaker.overlap_ratio"], 0.75)
        self.assertTrue(public_values["speaker.speaker_overlap"])
        self.assertEqual(
            set(public_values),
            {
                "speaker.speaker_count",
                "speaker.multi_speaker",
                "speaker.speaker_change_count",
                "speaker.speaker_change",
                "speaker.overlap_ratio",
                "speaker.speaker_overlap",
            },
        )

    def test_utterance_overlap_ratio_uses_speech_union_not_utterance_duration(self):
        metadata = build_metadata_from_timeline(
            [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "S01"},
                {"start_sec": 0.8, "end_sec": 1.2, "speaker_id": "S02"},
            ],
            duration_sec=10.0,
            sample_id="u_silence_padded",
            recording_id="meet1",
            target_units=[
                {"unit_id": "u_silence_padded", "start_sec": 0.0, "end_sec": 10.0},
            ],
            config=SpeakerMetricsConfig(min_speech_duration_sec=0.1),
        )

        utterance = metadata["utterances"][0]
        self.assertEqual(utterance["overlap_duration_sec"], 0.2)
        self.assertEqual(utterance["speech_union_duration_sec"], 1.2)
        self.assertEqual(utterance["overlap_ratio"], 0.166667)
        self.assertTrue(utterance["is_overlapped"])

        public_values = {
            result.tag_path: result.value
            for result in public_results_from_metadata(metadata)
        }
        self.assertEqual(public_values["speaker.overlap_ratio"], 0.166667)
        self.assertTrue(public_values["speaker.speaker_overlap"])

    def test_moss_text_parser_reads_speaker_timestamp_segments(self):
        segments = parse_moss_text(
            "[0.00][S01] hello there [1.00]\n"
            "[0.80][S02] hi [2.00]"
        )

        self.assertEqual(
            segments,
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_id": "S01",
                    "text": "hello there",
                },
                {
                    "start_sec": 0.8,
                    "end_sec": 2.0,
                    "speaker_id": "S02",
                    "text": "hi",
                },
            ],
        )

    def test_moss_subprocess_client_calls_local_worker(self):
        config = MossDiarizeConfig(
            model="local-moss",
            subprocess_python="/tmp/moss-python",
            device="cpu",
            torch_dtype="float32",
            max_new_tokens=2048,
        )
        client = MossDiarizeSubprocessClient(config)

        with mock.patch(
            "tagger.tools.speaker.moss_diarizer.run_subprocess_tool",
            return_value={"output": {"text": "[0.00][S01] hi [1.00]"}},
        ) as run_tool:
            output = client.diarize("audio.wav", context={"cache": True})

        self.assertEqual(output["text"], "[0.00][S01] hi [1.00]")
        run_tool.assert_called_once_with(
            "/tmp/moss-python",
            "moss_diarize_estimate",
            {
                "audio_path": "audio.wav",
                "config": {
                    "endpoint": "",
                    "model": "local-moss",
                    "timeout_sec": 900,
                    "max_new_tokens": 2048,
                    "api_key": "",
                    "subprocess_python": "",
                    "device": "cpu",
                    "torch_dtype": "float32",
                    "trust_remote_code": True,
                    "prompt": "",
                },
            },
            context={"cache": True},
            timeout_sec=900,
        )

    def test_speaker_v2_cli_profile_options(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.speaker_profile, "quality-shadow")
        self.assertFalse(args.speaker_v2_skip_model_verification)
        self.assertFalse(args.speaker_segment_lid_enabled)
        self.assertTrue(args.language_only_start_speaker_stage)
        self.assertEqual(args.speaker_segment_lid_min_speaker_sec, 2.0)
        self.assertEqual(args.speaker_segment_lid_min_total_sec, 3.0)
        self.assertFalse(args.speaker_segment_lid_shadow_only)
        self.assertFalse(args.speaker_segment_lid_keep_artifacts)
        args = parser.parse_args(["--speaker-segment-lid"])
        self.assertTrue(args.speaker_segment_lid_enabled)
        self.assertTrue(
            args.qwen3_asr_model.endswith(
                "/models/Qwen3-ASR-0.6B"
            )
        )
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--nemotron-model-dir", options)
        self.assertIn("--qwen3-asr-python", options)
        self.assertIn("--qwen3-asr-model", options)
        self.assertIn("--firered-asr-python", options)
        self.assertIn("--firered-asr-model", options)
        self.assertIn("--speaker-segment-lid", options)
        self.assertIn("--no-speaker-segment-lid", options)
        self.assertIn("--no-language-only-speaker-stage", options)

        args = parser.parse_args(
            [
                "--speaker-profile",
                "lean-shadow",
                "--speaker-v2-skip-model-verification",
                "--sample-id",
                "utt1",
                "--input-tags",
                "old.jsonl",
                "--only-tags",
                "speaker.asr_transcript,language_content.word_count",
                "--missing-only",
                "--no-speaker-segment-lid",
                "--no-language-only-speaker-stage",
                "--speaker-segment-lid-shadow-only",
                "--speaker-segment-lid-keep-artifacts",
            ]
        )
        self.assertEqual(args.speaker_profile, "lean-shadow")
        self.assertTrue(args.speaker_v2_skip_model_verification)
        self.assertEqual(args.sample_id, ["utt1"])
        self.assertEqual(args.input_tags, "old.jsonl")
        self.assertEqual(
            args.only_tags,
            "speaker.asr_transcript,language_content.word_count",
        )
        self.assertTrue(args.missing_only)
        self.assertFalse(args.speaker_segment_lid_enabled)
        self.assertFalse(args.language_only_start_speaker_stage)
        self.assertTrue(args.speaker_segment_lid_shadow_only)
        self.assertTrue(args.speaker_segment_lid_keep_artifacts)

    def test_public_schema_replaces_topic_with_speaker_asr_transcript(self):
        tags = empty_tags()

        self.assertEqual(sum(len(fields) for fields in tags.values()), 33)
        self.assertIn("asr_transcript", tags["speaker"])
        self.assertEqual(
            set(tags["deterministic_derived"]),
            {"speaker_change_rate_per_min", "filler_ratio"},
        )
        self.assertNotIn("topic", tags["language_content"])

    def test_speaker_asr_tag_selects_speaker_and_audio_probe_stages(self):
        self.assertEqual(
            _stages_for_tag_paths(["speaker.asr_transcript"]),
            {"speaker", "audio_probe"},
        )

    def test_merge_tags_drops_removed_topic_field(self):
        merged = _merge_tags(
            {
                "language_content": {
                    "topic": "technology/artificial_intelligence",
                    "word_count": 12,
                }
            }
        )

        self.assertNotIn("topic", merged["language_content"])
        self.assertEqual(merged["language_content"]["word_count"], 12)

    def test_tagging_manifest_reuses_context_and_isolates_speaker_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            write_test_wav(path, sample_rate=8000, channels=1, duration_sec=2.0)
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            output_path = Path(tmpdir) / "tags.jsonl"
            records = [make_record(str(path)), make_record(str(path))]
            with manifest_path.open("w", encoding="utf-8") as sink:
                for record in records:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

            public_speaker = {
                "speaker_count": 1,
                "speaker_present": True,
                "multi_speaker": False,
                "speaker_change_count": 0,
                "speaker_change": False,
                "overlap_ratio": 0.0,
                "speaker_overlap": False,
                "profiles": None,
            }
            result = {
                "speaker": public_speaker,
                "run_profile": "quality-shadow",
                "policy_version": "policy-v1",
                "policy_hash": "hash",
                "fusion_artifact": "fusion.json.gz",
                "artifacts": {},
            }
            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                return_value=result,
            ) as run_speaker:
                summary = run_tagging_manifest(
                    manifest_path,
                    output_path,
                    speaker_config=object(),
                    selected_tag_paths=["speaker"],
                )

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(run_speaker.call_count, 2)
            first = run_speaker.call_args_list[0][1]
            second = run_speaker.call_args_list[1][1]
            self.assertIs(first["context"], second["context"])
            self.assertNotEqual(
                first["artifact_sample_id"],
                second["artifact_sample_id"],
            )
            self.assertEqual(
                first["artifact_root"],
                Path(tmpdir) / "artifacts",
            )
            with output_path.open("r", encoding="utf-8") as source_file:
                rows = [json.loads(line) for line in source_file if line.strip()]
            expected_speaker = dict(public_speaker, asr_transcript=None)
            self.assertEqual(
                [row["speaker"] for row in rows],
                [expected_speaker] * 2,
            )

    def test_tagging_manifest_supplements_selected_sample_tags_from_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            input_tags_path = Path(tmpdir) / "existing_tags.jsonl"
            output_path = Path(tmpdir) / "updated_tags.jsonl"
            first = make_record("missing_first.wav")
            first["sample"]["sample_id"] = "first"
            first["sample"]["text"]["transcript"] = "Um hello."
            second = make_record("missing_second.wav")
            second["sample"]["sample_id"] = "second"
            second["sample"]["text"]["transcript"] = "Keep this row."
            with manifest_path.open("w", encoding="utf-8") as sink:
                sink.write(json.dumps(first, ensure_ascii=False) + "\n")
                sink.write(json.dumps(second, ensure_ascii=False) + "\n")

            first_tags = empty_tags()
            second_tags = empty_tags()
            second_tags["language_content"]["filler"] = 99
            with input_tags_path.open("w", encoding="utf-8") as sink:
                sink.write(json.dumps(first_tags, ensure_ascii=False) + "\n")
                sink.write(json.dumps(second_tags, ensure_ascii=False) + "\n")

            summary = run_tagging_manifest(
                manifest_path,
                output_path,
                sample_ids=["first"],
                existing_tags_path=input_tags_path,
                selected_tag_paths=["language_content.filler"],
            )

            with output_path.open("r", encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["processed_sample_count"], 1)
            self.assertEqual(rows[0]["language_content"]["filler"], 1)
            self.assertEqual(rows[1]["language_content"]["filler"], 99)
            self.assertIsNone(rows[0]["basic_acoustic"]["duration_sec"])

    def test_tagging_pipeline_nulls_speaker_fields_when_v2_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            write_test_wav(path, sample_rate=8000, channels=1, duration_sec=2.0)
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            output_path = Path(tmpdir) / "tags.jsonl"
            with manifest_path.open("w", encoding="utf-8") as sink:
                sink.write(json.dumps(make_record(str(path)), ensure_ascii=False) + "\n")

            with mock.patch(
                "tagger.pipelines.tagging.run_speaker_v2_record",
                side_effect=RuntimeError("speaker failed"),
            ):
                summary = run_tagging_manifest(
                    manifest_path,
                    output_path,
                    speaker_config=object(),
                    selected_tag_paths=["speaker"],
                )

            self.assertEqual(summary["internal_warning_count"], 1)
            with output_path.open("r", encoding="utf-8") as source_file:
                tags = json.loads(source_file.readline())
            self.assertTrue(all(value is None for value in tags["speaker"].values()))

    def test_input_transcript_does_not_populate_speaker_asr_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            write_test_wav(path, sample_rate=8000, channels=1, duration_sec=2.0)
            record = make_record(str(path))
            record["sample"]["text"]["transcript"] = "Reference transcript."

            tags = tag_sample_record(
                record,
                tmpdir,
                speaker_config=object(),
                selected_tag_paths=["speaker.asr_transcript"],
            )

            self.assertIsNone(tags["speaker"]["asr_transcript"])

    def test_speaker_auditor_rejects_invalid_public_values(self):
        speaker = {
            "multi_speaker": "yes",
            "speaker_change": "yes",
            "speaker_overlap": 1,
            "speaker_count": -1,
            "speaker_change_count": 1.5,
            "overlap_ratio": 1.1,
        }

        warnings = audit_speaker(speaker)

        self.assertEqual(
            warnings,
            [
                {"type": "invalid_speaker_value", "field": "multi_speaker"},
                {"type": "invalid_speaker_value", "field": "speaker_change"},
                {"type": "invalid_speaker_value", "field": "speaker_overlap"},
                {"type": "invalid_speaker_value", "field": "speaker_count"},
                {"type": "invalid_speaker_value", "field": "speaker_change_count"},
                {"type": "invalid_speaker_value", "field": "overlap_ratio"},
            ],
        )
        self.assertTrue(all(value is None for value in speaker.values()))

    def test_speaker_present_is_derived_from_speaker_count(self):
        for speaker_count, expected in (
            (None, None),
            (0, False),
            (1, True),
            (2, True),
        ):
            speaker = {"speaker_count": speaker_count}

            warnings = audit_speaker(speaker)

            self.assertEqual(warnings, [])
            self.assertIs(speaker["speaker_present"], expected)

    def test_speaker_present_overrides_inconsistent_existing_value(self):
        speaker = {"speaker_count": 0, "speaker_present": True}

        warnings = audit_speaker(speaker)

        self.assertEqual(
            warnings,
            [{"type": "invalid_speaker_value", "field": "speaker_present"}],
        )
        self.assertFalse(speaker["speaker_present"])

    def test_speaker_auditor_normalizes_asr_transcript(self):
        speaker = {"asr_transcript": "  Complete MOSS transcript.  "}

        warnings = audit_speaker(speaker)

        self.assertEqual(warnings, [])
        self.assertEqual(speaker["asr_transcript"], "Complete MOSS transcript.")

    def test_speaker_auditor_nulls_invalid_asr_transcript(self):
        for value in ("", "   ", 42, ["not", "text"]):
            speaker = {"asr_transcript": value}

            warnings = audit_speaker(speaker)

            self.assertEqual(
                warnings,
                [{"type": "invalid_speaker_value", "field": "asr_transcript"}],
            )
            self.assertIsNone(speaker["asr_transcript"])

    def test_speaker_auditor_accepts_valid_profiles(self):
        speaker = {
            "speaker_count": 2,
            "multi_speaker": True,
            "speaker_change_count": 1,
            "speaker_change": True,
            "overlap_ratio": 0.125,
            "speaker_overlap": True,
            "profiles": [
                {
                    "speaker_id": "speaker_1",
                    "speech_rate": {
                        "band": "normal",
                        "value": 4.2,
                        "unit": "zh_char_per_sec",
                    },
                    "gender": "female",
                    "pitch": "mid",
                    "speaker_volume": "normal",
                },
                {
                    "speaker_id": "speaker_2",
                    "speech_rate": {
                        "band": "fast",
                        "value": 165.0,
                        "unit": "word_per_min",
                    },
                    "gender": "male",
                    "pitch": "high",
                    "speaker_volume": "low",
                },
                {
                    "speaker_id": "speaker_3",
                    "speech_rate": {"band": None, "value": None, "unit": None},
                    "gender": None,
                    "pitch": None,
                    "speaker_volume": None,
                },
            ],
        }

        warnings = audit_speaker(speaker)

        self.assertEqual(warnings, [])
        self.assertTrue(speaker["speaker_present"])
        self.assertEqual(len(speaker["profiles"]), 3)

    def test_speaker_auditor_rejects_invalid_profiles(self):
        valid_profile = {
            "speaker_id": "speaker_1",
            "speech_rate": {
                "band": "normal",
                "value": 4.2,
                "unit": "zh_char_per_sec",
            },
            "gender": "female",
            "pitch": "mid",
            "speaker_volume": "normal",
        }
        bad_cases = [
            "not-a-list",
            [valid_profile, dict(valid_profile)],  # duplicate speaker ids
            [dict(valid_profile, speaker_id="1")],
            [
                dict(
                    valid_profile,
                    speech_rate={
                        "band": "instant",
                        "value": 4.2,
                        "unit": "zh_char_per_sec",
                    },
                )
            ],
            [
                dict(
                    valid_profile,
                    speech_rate={"band": "normal", "value": 4.2, "unit": None},
                )
            ],
            [dict(valid_profile, pitch="tenor")],
            [dict(valid_profile, speaker_volume="screaming")],
            [dict(valid_profile, gender="unknown")],
        ]
        for value in bad_cases:
            speaker = {
                "speaker_count": 1,
                "multi_speaker": False,
                "speaker_change_count": 0,
                "speaker_change": False,
                "overlap_ratio": 0.0,
                "speaker_overlap": False,
                "profiles": value,
            }

            warnings = audit_speaker(speaker)

            self.assertEqual(
                warnings,
                [{"type": "invalid_speaker_value", "field": "profiles"}],
            )
            self.assertIsNone(speaker["profiles"])

    def test_speaker_auditor_accepts_empty_profiles_list(self):
        speaker = {
            "speaker_count": 0,
            "multi_speaker": False,
            "speaker_change_count": 0,
            "speaker_change": False,
            "overlap_ratio": 0.0,
            "speaker_overlap": False,
            "profiles": [],
        }

        warnings = audit_speaker(speaker)

        self.assertEqual(warnings, [])
        self.assertFalse(speaker["speaker_present"])
        self.assertEqual(speaker["profiles"], [])


def make_record(audio_path):
    return {
        "corpus": {
            "dataset_name": "unit_test",
            "source_urls": {
                "article": [],
                "github": [],
                "huggingface": [],
                "dataset_card": [],
            },
            "native_metadata": {},
        },
        "sample": {
            "sample_id": "utt1",
            "audio": {"path": audio_path},
            "text": {"transcript": "HELLO WORLD"},
            "native_metadata": {"utt2spk": "spk1"},
        },
    }


class FakeFireRedClient:
    def __init__(self, speech_segments):
        self.speech_segments = speech_segments

    def detect_speech_segments(self, audio_path, context=None):
        return self.speech_segments


class FakeFireRedAedClient:
    def __init__(self, output):
        self.output = output

    def detect_audio_events(self, audio_path, context=None):
        return self.output


class FakeDassNoiseTypeClient:
    def __init__(self, output):
        self.output = output

    def estimate(self, audio_path, context=None):
        return self.output


class FakeBrouhahaClient:
    def __init__(self, output):
        self.output = output

    def estimate(self, audio_path, context=None):
        return self.output


class FakeDnsmosClient:
    def __init__(self, output):
        self.output = output
        self.call_count = 0

    def estimate(self, audio_path, context=None):
        self.call_count += 1
        return self.output


class FakeRecRirClient:
    def __init__(self, output):
        self.output = output
        self.call_count = 0

    def estimate_rir(self, audio_path, context=None):
        self.call_count += 1
        return self.output


def make_dass_output(score, display_name, extra_events=None, extra_vector_events=None):
    """Build a full fake DASS estimate.

    ``extra_events`` are ``(score, display_name)`` pairs appended to the
    public top-noise list (scores must not exceed the winning score).
    ``extra_vector_events`` are injected only into the full 527-class vector,
    which is where the composition layer reads from.
    """
    labels = [
        {"index": index, "display_name": "class_%03d" % index}
        for index in range(527)
    ]
    scores = [0.0] * 527
    events = [(score, display_name)] + list(extra_events or [])
    top_noise_events = []
    for event_index, (event_score, event_name) in enumerate(events, start=1):
        labels[event_index] = {"index": event_index, "display_name": event_name}
        scores[event_index] = event_score
        top_noise_events.append(
            {"index": event_index, "display_name": event_name, "score": event_score}
        )
    for event_index, (event_score, event_name) in enumerate(
        extra_vector_events or [], start=100
    ):
        labels[event_index] = {"index": event_index, "display_name": event_name}
        scores[event_index] = event_score
    return {
        "chunk_count": 1,
        "max_noise_score": score,
        "winning_event": dict(top_noise_events[0]),
        "top_noise_events": top_noise_events,
        "labels": labels,
        "scores": scores,
    }


def missing_external_model_configs(tmpdir):
    root = Path(tmpdir)
    return {
        "firered_vad_config": FireRedVadConfig(
            model_dir=str(root / "missing_vad"),
            subprocess_python="",
        ),
        "brouhaha_config": BrouhahaConfig(
            model_path=str(root / "missing_brouhaha.ckpt"),
            repo_dir=str(root / "missing_brouhaha_repo"),
            subprocess_python="",
        ),
        "recrir_config": RecRirConfig(
            repo_dir=str(root / "missing_recrir_repo"),
            config_path=str(root / "missing_recrir.toml"),
            checkpoint_path=str(root / "missing_recrir.tar"),
            subprocess_python="",
        ),
        "dnsmos_config": DnsmosConfig(
            primary_model_path=str(root / "missing_dnsmos.onnx"),
            p808_model_path=str(root / "missing_p808.onnx"),
            personalized_model_path=str(root / "missing_pdnsmos.onnx"),
            subprocess_python="",
        ),
        "firered_aed_config": FireRedAedConfig(
            model_dir=str(root / "missing_aed"),
            subprocess_python="",
        ),
        "dass_config": DassNoiseTypeConfig(
            model_dir=str(root / "missing_dass"),
            subprocess_python="",
        ),
    }


def write_test_wav(
    path: Path,
    sample_rate: int,
    channels: int,
    duration_sec: float,
):
    frame_count = int(sample_rate * duration_sec)
    frames = []
    for index in range(frame_count):
        sample = int(10000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend([sample] * channels)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<" + "h" * len(frames), *frames))


if __name__ == "__main__":
    unittest.main()
