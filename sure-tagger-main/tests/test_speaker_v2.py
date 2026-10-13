import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import wave

from tagger.pipelines.speaker_evidence import (
    SpeakerEvidenceConfig,
    _speaker_asr_transcript,
    collect_moss_evidence,
    collect_pyannote_evidence,
    collect_sortformer_evidence,
    collect_whisper_evidence,
    run_manifest,
    run_record,
)
from tagger.tools.base import ToolResult
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    FireRedVadConfig,
)
from tagger.tools.speaker.moss_diarizer import MossDiarizeConfig
from tagger.tools.speaker_v2.artifacts import safe_stem, write_json_gz_atomic
from tagger.tools.speaker_v2.campplus_identity import (
    CampPlusIdentityConfig,
    select_candidate_regions,
)
from tagger.tools.speaker_v2.contracts import (
    EvidenceContractError,
    build_evidence,
    independent,
    validate_evidence,
)
from tagger.tools.speaker_v2.hypotheses import (
    build_count_hypothesis_case,
    evaluate_count_hypothesis_case,
)
from tagger.tools.speaker_v2.pyannote_community1 import (
    PyannoteCommunity1Config,
    PyannoteCommunity1SubprocessClient,
    annotation_segments,
)
from tagger.tools.speaker_v2.lexical import (
    project_asr_track,
    speaker_assignment_comparison,
    speaker_text_track,
)
from tagger.tools.speaker_v2.resolver import resolve
from tagger.tools.speaker_v2.timeline import (
    interval_set_iou,
    summarize_timeline,
    timeline_comparison,
)
from tagger.tools.speaker_v2.sortformer_timeline import (
    SortformerTimelineConfig,
    _parse_segments,
)
from tagger.tools.speaker_v2.whisper_lexical import WhisperLexicalConfig


class SpeakerV2Test(unittest.TestCase):
    def test_speaker_asr_transcript_ignores_timestamp_only_output(self):
        lid = {
            "source": {"name": "firered_lid"},
            "status": "estimated",
            "quality": {"usable": True},
            "payload": {"language": "zh", "language_confidence": 1.0},
        }
        evidence = {
            "source": {"name": "moss_transcribe_diarize"},
            "payload": {
                "asr_transcript": "",
                "timeline_summary": {
                    "segments": [
                        {
                            "start_sec": 0.0,
                            "end_sec": 10.88,
                            "speaker_id": "S01",
                        }
                    ]
                },
                "model_output_text": "[0.00][S01][10.88]",
            },
        }

        self.assertEqual(_speaker_asr_transcript([lid, evidence]), "")

    def test_speaker_asr_transcript_joins_moss_segments_in_time_order(self):
        lid = {
            "source": {"name": "firered_lid"},
            "status": "estimated",
            "quality": {"usable": True},
            "payload": {"language": "zh-mandarin", "language_confidence": 1.0},
        }
        evidence = {
            "source": {"name": "moss_transcribe_diarize"},
            "payload": {
                "asr_transcript": "first second third",
                "timeline_summary": {
                    "segments": [
                        {"start_sec": 0.0, "text": "merged speaker timeline"},
                    ]
                },
            },
        }

        self.assertEqual(
            _speaker_asr_transcript([lid, evidence]),
            "first second third",
        )

    def test_collect_moss_evidence_builds_asr_in_segment_time_order(self):
        moss_result = ToolResult(
            "speaker.diarization_timeline",
            {
                "segments": [
                    {
                        "start_sec": 2.0,
                        "end_sec": 3.0,
                        "speaker_id": "S01",
                        "text": "third",
                    },
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "speaker_id": "S01",
                        "text": " first ",
                    },
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "speaker_id": "S02",
                        "text": "second",
                    },
                    {
                        "start_sec": 0.5,
                        "end_sec": 0.75,
                        "speaker_id": "S03",
                        "text": None,
                    },
                    {
                        "start_sec": 2.5,
                        "end_sec": 2.75,
                        "speaker_id": "S03",
                        "text": 42,
                    },
                    {
                        "end_sec": 2.75,
                        "speaker_id": "S03",
                        "text": "missing start",
                    },
                ],
                "raw_text": "",
            },
            "moss",
            "fake",
        )
        scope = {
            "sample_id": "sample",
            "duration_sec": 3.0,
            "audio_path": "/audio.wav",
            "audio_sha256": "audio",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.run_moss_diarizer",
            return_value=moss_result,
        ):
            evidence = collect_moss_evidence(
                scope,
                MossDiarizeConfig(model="fake"),
                verify_model_asset=False,
            )

        self.assertEqual(
            evidence["payload"]["asr_transcript"],
            "first second third",
        )

    def test_speaker_asr_transcript_rejects_unparsed_raw_moss_output(self):
        evidence = {
            "source": {"name": "moss_transcribe_diarize"},
            "payload": {
                "asr_transcript": "",
                "timeline_summary": {"segments": []},
                "model_output_text": "malformed response without a timeline",
            },
        }

        self.assertEqual(_speaker_asr_transcript([evidence]), "")

    def test_evidence_id_is_deterministic_and_excludes_runtime(self):
        first = make_coverage_evidence(runtime={"elapsed_sec": 1.0})
        second = make_coverage_evidence(runtime={"elapsed_sec": 99.0})
        self.assertEqual(first["evidence_id"], second["evidence_id"])
        validate_evidence(first)

    def test_evidence_id_changes_with_audio_fingerprint(self):
        first = make_coverage_evidence(applicability={"audio_sha256": "a"})
        second = make_coverage_evidence(applicability={"audio_sha256": "b"})
        self.assertNotEqual(first["evidence_id"], second["evidence_id"])

    def test_evidence_id_changes_with_certification_quality(self):
        uncalibrated = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=False
        )
        calibrated = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        self.assertNotEqual(
            uncalibrated["evidence_id"], calibrated["evidence_id"]
        )

    def test_evidence_rejects_non_finite_payload(self):
        with self.assertRaises(EvidenceContractError):
            build_evidence(
                "sample",
                2.0,
                "coverage",
                "vad",
                "v1",
                "vad",
                ["speech_coverage"],
                ["G_vad"],
                {"ratio": float("nan")},
            )

    def test_timeline_summary_keeps_text_count_and_overlap(self):
        summary = summarize_timeline(
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_id": "S01",
                    "text": "hello",
                },
                {
                    "start_sec": 0.7,
                    "end_sec": 1.5,
                    "speaker_id": "S02",
                    "text": "yes",
                },
            ],
            2.0,
        )
        self.assertEqual(summary["observed_speaker_count"], 2)
        self.assertTrue(summary["overlap_observed"])
        self.assertEqual(summary["segments"][0]["text"], "hello")

    def test_same_speaker_gap_smoothing_does_not_create_overlap(self):
        summary = summarize_timeline(
            [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"},
                {"start_sec": 1.04, "end_sec": 1.16, "speaker_id": "B"},
                {"start_sec": 1.2, "end_sec": 2.0, "speaker_id": "A"},
            ],
            2.0,
        )

        self.assertEqual(
            [
                (item["speaker_id"], item["start_sec"], item["end_sec"])
                for item in summary["segments"]
            ],
            [("a", 0.0, 2.0), ("b", 1.04, 1.16)],
        )
        self.assertEqual(summary["overlap_activity_segments"], [])
        self.assertEqual(summary["overlap_segments"], [])
        self.assertFalse(summary["overlap_observed"])
        self.assertEqual(summary["speech_union_duration_sec"], 1.92)
        regions = select_candidate_regions(
            summary,
            CampPlusIdentityConfig(
                model_dir="unused",
                min_region_duration_sec=0.10,
                max_regions_per_speaker=2,
            ),
        )
        self.assertEqual(
            [
                (item["speaker_id"], item["start_sec"], item["end_sec"])
                for item in regions
            ],
            [("a", 0.0, 1.0), ("b", 1.04, 1.16), ("a", 1.2, 2.0)],
        )

    def test_campplus_candidates_exclude_all_overlap_audio(self):
        summary = summarize_timeline(
            [
                {"start_sec": 0.0, "end_sec": 2.0, "speaker_id": "A"},
                {"start_sec": 0.75, "end_sec": 1.25, "speaker_id": "B"},
            ],
            2.0,
        )
        config = CampPlusIdentityConfig(
            model_dir="unused",
            min_region_duration_sec=0.70,
            max_regions_per_speaker=2,
        )

        regions = select_candidate_regions(summary, config)

        self.assertEqual(
            [(item["start_sec"], item["end_sec"]) for item in regions],
            [(0.0, 0.75), (1.25, 2.0)],
        )
        self.assertTrue(
            all(
                item["end_sec"] <= 0.75 or item["start_sec"] >= 1.25
                for item in regions
            )
        )
        abstain_config = CampPlusIdentityConfig(
            model_dir="unused",
            min_region_duration_sec=0.80,
            max_regions_per_speaker=2,
        )
        self.assertEqual(select_candidate_regions(summary, abstain_config), [])

    def test_timeline_comparison_uses_event_local_alignment(self):
        base = summarize_timeline(
            [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"},
                {"start_sec": 0.5, "end_sec": 1.5, "speaker_id": "B"},
            ],
            3.0,
        )
        shifted = summarize_timeline(
            [
                {"start_sec": 2.0, "end_sec": 2.8, "speaker_id": "X"},
                {"start_sec": 2.3, "end_sec": 3.0, "speaker_id": "Y"},
            ],
            3.0,
        )
        comparison = timeline_comparison(base, shifted)
        self.assertTrue(comparison["overlap_bool_equal"])
        self.assertEqual(comparison["overlap_event_iou"], 0.0)

    def test_same_positive_bool_with_disjoint_events_is_conflicted(self):
        first = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        second = build_evidence(
            "sample",
            2.0,
            "speaker_timeline",
            "source_b",
            "v1",
            "diarizer",
            ["speaker_timeline"],
            ["G_b"],
            {
                "timeline_summary": summarize_timeline(
                    [
                        {"start_sec": 0.0, "end_sec": 0.35, "speaker_id": "X"},
                        {"start_sec": 0.1, "end_sec": 0.45, "speaker_id": "Y"},
                    ],
                    2.0,
                )
            },
            quality={"usable": True},
        )
        fusion = resolve(
            "sample", 2.0, [first, second, make_coverage_evidence()]
        )
        self.assertEqual(fusion["claims"]["speaker_overlap"]["status"], "conflicted")

    def test_empty_interval_iou_is_one_for_joint_negative_comparison(self):
        self.assertEqual(interval_set_iou([], []), 1.0)

    def test_resolver_abstains_when_evidence_is_missing(self):
        fusion = resolve("sample", 2.0, [])
        self.assertEqual(fusion["claims"]["speaker_count"]["status"], "insufficient")
        self.assertIsNone(fusion["claims"]["multi_speaker"]["public_value"])

    def test_resolver_rejects_cross_sample_evidence(self):
        timeline = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True
        )
        with self.assertRaisesRegex(ValueError, "cross-sample"):
            resolve("other-sample", 2.0, [timeline])

    def test_single_timeline_is_supported_but_never_public(self):
        timeline = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        fusion = resolve("sample", 2.0, [timeline, make_coverage_evidence()])
        self.assertEqual(fusion["claims"]["multi_speaker"]["status"], "supported")
        self.assertTrue(fusion["claims"]["multi_speaker"]["candidate_value"])
        self.assertIsNone(fusion["claims"]["multi_speaker"]["public_value"])
        self.assertEqual(
            fusion["claims"]["speaker_count"]["observed_lower_bound_candidate"],
            2,
        )
        self.assertIsNone(fusion["claims"]["speaker_count"]["exact"])

    def test_independent_count_disagreement_is_conflicted(self):
        one = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        two = make_timeline_evidence("source_b", "G_b", two_speakers=False)
        fusion = resolve("sample", 2.0, [one, two, make_coverage_evidence()])
        self.assertEqual(fusion["claims"]["speaker_count"]["status"], "conflicted")
        self.assertEqual(fusion["claims"]["multi_speaker"]["status"], "conflicted")
        self.assertEqual(
            fusion["claims"]["speaker_count"]["supported_lower_bound"], 1
        )

    def test_exact_count_conflict_retains_shared_lower_bound(self):
        two = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        three = build_evidence(
            "sample",
            2.0,
            "speaker_timeline",
            "source_b",
            "v1",
            "diarizer",
            ["speaker_timeline"],
            ["G_b"],
            {
                "timeline_summary": summarize_timeline(
                    [
                        {"start_sec": 0.0, "end_sec": 0.8, "speaker_id": "A"},
                        {"start_sec": 0.6, "end_sec": 1.4, "speaker_id": "B"},
                        {"start_sec": 1.2, "end_sec": 2.0, "speaker_id": "C"},
                    ],
                    2.0,
                )
            },
            quality={"usable": True},
        )
        fusion = resolve("sample", 2.0, [two, three])
        count = fusion["claims"]["speaker_count"]
        self.assertEqual(count["status"], "conflicted")
        self.assertEqual(count["supported_lower_bound"], 2)
        self.assertEqual(count["observed_lower_bound_candidate"], 3)
        self.assertIsNone(count["exact"])

    def test_same_dependency_group_is_not_an_independent_vote(self):
        one = make_timeline_evidence("wrapper_a", "G_shared", two_speakers=True)
        two = make_timeline_evidence("wrapper_b", "G_shared", two_speakers=True)
        self.assertFalse(independent(one, two))
        fusion = resolve("sample", 2.0, [one, two, make_coverage_evidence()])
        roles = fusion["claims"]["multi_speaker"]["roles"]
        self.assertFalse(roles["independent_event_sources"])

    def test_uncalibrated_identity_cannot_complete_certification_roles(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=True, calibrated=True
        )
        identity = make_identity_evidence("different", calibrated=False)
        fusion = resolve(
            "sample", 2.0, [one, two, identity, make_coverage_evidence()]
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertTrue(claim["roles"]["identity_guard_observed"])
        self.assertFalse(claim["roles"]["identity_guard_calibrated"])

    def test_positive_claim_can_certify_only_when_all_roles_are_present(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=True, calibrated=True
        )
        identity = make_identity_evidence("different", calibrated=True)
        fusion = resolve(
            "sample", 2.0, [one, two, identity, make_coverage_evidence()]
        )
        self.assertEqual(fusion["claims"]["speaker_overlap"]["status"], "certified")
        self.assertTrue(fusion["claims"]["speaker_overlap"]["public_value"])

    def test_license_pending_timeline_cannot_certify(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "pyannote_community_1",
            "G_pyannote",
            two_speakers=True,
            calibrated=True,
            counts_for_certification=False,
        )
        identity = make_identity_evidence("different", calibrated=True)
        fusion = resolve(
            "sample", 2.0, [one, two, identity, make_coverage_evidence()]
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertFalse(claim["roles"]["calibration_ready"])
        self.assertIsNone(claim["public_value"])

    def test_empty_speech_coverage_cannot_certify_positive_claim(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=True, calibrated=True
        )
        identity = make_identity_evidence("different", calibrated=True)
        fusion = resolve(
            "sample",
            2.0,
            [one, two, identity, make_coverage_evidence(speech=False)],
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertFalse(claim["roles"]["coverage_guard"])

    def test_negative_claim_requires_joint_negative_profile(self):
        one = make_timeline_evidence(
            "source_a",
            "G_a",
            two_speakers=False,
            calibrated=True,
            joint_negative=False,
        )
        two = make_timeline_evidence(
            "source_b",
            "G_b",
            two_speakers=False,
            calibrated=True,
            joint_negative=False,
        )
        fusion = resolve("sample", 2.0, [one, two, make_coverage_evidence()])
        self.assertEqual(fusion["claims"]["speaker_overlap"]["status"], "supported")
        self.assertIsNone(fusion["claims"]["speaker_overlap"]["public_value"])

    def test_joint_negative_profile_must_cover_both_timelines(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=False, calibrated=True,
            joint_negative=True,
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=False, calibrated=True,
            joint_negative=False,
        )
        fusion = resolve("sample", 2.0, [one, two, make_coverage_evidence()])
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertFalse(claim["roles"]["joint_negative_ready"])

    def test_derived_identity_cannot_certify_its_parent_timelines(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=True, calibrated=True
        )
        identity = make_identity_evidence(
            "different", calibrated=True, parent=one,
            candidate_selection_independent=False,
        )
        fusion = resolve(
            "sample", 2.0, [one, two, identity, make_coverage_evidence()]
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertTrue(claim["roles"]["identity_guard_observed"])
        self.assertFalse(claim["roles"]["identity_guard_calibrated"])

    def test_within_cluster_difference_is_not_an_identity_guard(self):
        one = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True, calibrated=True
        )
        two = make_timeline_evidence(
            "source_b", "G_b", two_speakers=True, calibrated=True
        )
        identity = make_identity_evidence(
            "different",
            calibrated=True,
            comparison_kind="within_source_cluster",
        )
        fusion = resolve(
            "sample", 2.0, [one, two, identity, make_coverage_evidence()]
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertEqual(claim["status"], "supported")
        self.assertFalse(claim["roles"]["identity_guard_observed"])
        self.assertFalse(claim["roles"]["identity_guard_calibrated"])

    def test_within_cluster_same_result_does_not_falsify_count_hypothesis(self):
        higher = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        lower = make_timeline_evidence("source_b", "G_b", two_speakers=False)
        case = build_count_hypothesis_case("sample", [higher, lower])
        identity = make_identity_evidence(
            "same", calibrated=True, comparison_kind="within_source_cluster"
        )
        evaluated = evaluate_count_hypothesis_case(case, [identity])
        statuses = {
            item["hypothesis_id"]: item["status"]
            for item in evaluated["branches"]
        }
        self.assertEqual(statuses["H1"], "untested")
        self.assertEqual(
            evaluated["termination_reason"],
            "identity_evidence_missing_or_dependent",
        )

    def test_hypotheses_are_frozen_for_independent_count_mismatch(self):
        higher = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        lower = make_timeline_evidence("source_b", "G_b", two_speakers=False)
        case = build_count_hypothesis_case("sample", [higher, lower])
        self.assertIsNotNone(case)
        self.assertEqual(
            [item["hypothesis_id"] for item in case["branches"]],
            ["H1", "H2", "H_other"],
        )
        self.assertTrue(case["acquisition_plan"][0]["frozen_before_acquisition"])
        self.assertEqual(case["certification_effect"], "none")

    def test_uncalibrated_identity_is_diagnostic_not_a_falsifier(self):
        higher = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        lower = make_timeline_evidence("source_b", "G_b", two_speakers=False)
        case = build_count_hypothesis_case("sample", [higher, lower])
        evaluated = evaluate_count_hypothesis_case(
            case, [make_identity_evidence("same", calibrated=False)]
        )
        statuses = {
            item["hypothesis_id"]: item["status"]
            for item in evaluated["branches"]
        }
        self.assertEqual(statuses["H1"], "viable")
        self.assertEqual(evaluated["termination_reason"], "uncalibrated_identity_diagnostic")

    def test_calibrated_same_identity_falsifies_higher_count_branch_only(self):
        higher = make_timeline_evidence("source_a", "G_a", two_speakers=True)
        lower = make_timeline_evidence("source_b", "G_b", two_speakers=False)
        case = build_count_hypothesis_case("sample", [higher, lower])
        evaluated = evaluate_count_hypothesis_case(
            case, [make_identity_evidence("same", calibrated=True)]
        )
        statuses = {
            item["hypothesis_id"]: item["status"]
            for item in evaluated["branches"]
        }
        self.assertEqual(statuses["H1"], "falsified")
        self.assertEqual(statuses["H2"], "viable")
        self.assertEqual(statuses["H_other"], "viable")

    def test_speaker_text_is_not_promoted_to_an_independent_vote(self):
        timeline = make_timeline_evidence("moss", "G_moss", two_speakers=True)
        track = speaker_text_track(timeline)
        self.assertEqual(track["dependency_groups"], ["G_moss"])
        self.assertEqual(track["source_evidence_id"], timeline["evidence_id"])

    def test_independent_asr_projection_inherits_both_dependency_groups(self):
        timeline = make_timeline_evidence("moss", "G_moss", two_speakers=True)
        asr = build_evidence(
            "sample",
            2.0,
            "lexical_timeline",
            "whisper",
            "v1",
            "asr",
            ["lexical_timeline"],
            ["G_whisper"],
            {
                "lexical_units": [
                    {
                        "unit_id": "w0",
                        "start_sec": 0.85,
                        "end_sec": 1.05,
                        "text": "yes",
                        "timestamp_method": "asr_segment_interval",
                    }
                ]
            },
        )
        track = project_asr_track(asr, timeline)
        self.assertEqual(track["dependency_groups"], ["G_moss", "G_whisper"])
        self.assertEqual(track["units"][0]["assignment"], "ambiguous")
        self.assertFalse(track["speaker_event_vote"])

    def test_whisper_evidence_is_lexical_only(self):
        config = WhisperLexicalConfig(
            model_path="fake.pt", device="cpu", word_timestamps=True
        )
        scope = {
            "sample_id": "sample",
            "audio_path": "sample.wav",
            "audio_sha256": "audio-hash",
            "duration_sec": 2.0,
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        result = {
            "text": "hello",
            "language": "en",
            "lexical_units": [
                {
                    "unit_id": "word_000000_0000",
                    "start_sec": 0.1,
                    "end_sec": 0.5,
                    "text": "hello",
                    "timestamp_method": "attention_dtw_word_interval",
                }
            ],
            "runtime": {"elapsed_sec": 0.1},
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.transcribe_whisper",
            return_value=result,
        ):
            evidence = collect_whisper_evidence(
                scope, config, verify_model_asset=False
            )
        self.assertEqual(evidence["status"], "estimated")
        self.assertIn("lexical_timeline", evidence["capabilities"])
        self.assertNotIn("speaker_timeline", evidence["capabilities"])
        self.assertTrue(evidence["payload"]["not_a_speaker_event_vote"])

    def test_sortformer_parses_overlap_aware_segments(self):
        segments = _parse_segments(
            [[
                "0.000 1.000 speaker_0",
                "0.800 1.200 speaker_1",
            ]]
        )
        summary = summarize_timeline(segments, 2.0)
        self.assertEqual(summary["observed_speaker_count"], 2)
        self.assertTrue(summary["overlap_observed"])

    def test_sortformer_evidence_cannot_claim_global_upper_bound(self):
        config = SortformerTimelineConfig(
            model_path="fake.nemo", device="cpu"
        )
        scope = {
            "sample_id": "sample",
            "audio_path": "sample.wav",
            "audio_sha256": "audio-hash",
            "duration_sec": 2.0,
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        result = {
            "segments": [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "speaker_0"},
                {"start_sec": 0.8, "end_sec": 1.2, "speaker_id": "speaker_1"},
            ],
            "probabilities": {"shape": [25, 4]},
            "runtime": {"elapsed_sec": 0.1},
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.diarize_sortformer",
            return_value=result,
        ):
            evidence = collect_sortformer_evidence(
                scope, config, verify_model_asset=False
            )
        self.assertEqual(
            evidence["payload"]["timeline_summary"]["observed_speaker_count"],
            2,
        )
        self.assertTrue(evidence["payload"]["not_a_global_speaker_upper_bound"])
        self.assertFalse(evidence["quality"]["negative_capability"])
        self.assertEqual(evidence["applicability"]["maximum_speakers"], 4)

    def test_pyannote_evidence_is_a_gated_full_timeline(self):
        config = PyannoteCommunity1Config(
            "/model",
            calibration_profile_id="py-cal-v1",
            license_review_status="pending",
        )
        result = {
            "raw_segments": [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"},
                {"start_sec": 0.4, "end_sec": 1.2, "speaker_id": "B"},
            ],
            "exclusive_segments": [
                {"start_sec": 0.0, "end_sec": 0.6, "speaker_id": "A"},
                {"start_sec": 0.6, "end_sec": 1.2, "speaker_id": "B"},
            ],
            "runtime": {"torchcodec_bypassed": True},
        }
        scope = {
            "sample_id": "sample",
            "duration_sec": 2.0,
            "audio_path": "/audio.wav",
            "audio_sha256": "audio",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.diarize_pyannote",
            return_value=result,
        ):
            evidence = collect_pyannote_evidence(
                scope, config, verify_model_asset=False
            )
        self.assertIn("speaker_timeline", evidence["capabilities"])
        self.assertTrue(
            evidence["payload"]["timeline_summary"]["overlap_observed"]
        )
        self.assertFalse(
            evidence["payload"]["exclusive_timeline_summary"][
                "overlap_observed"
            ]
        )
        self.assertFalse(evidence["quality"]["counts_for_certification"])

    def test_pyannote_subprocess_config_never_contains_credentials(self):
        config = PyannoteCommunity1Config(
            "/model", subprocess_python="/runtime/python"
        )
        expected = {
            "raw_segments": [],
            "exclusive_segments": [],
            "runtime": {},
        }
        with mock.patch(
            "tagger.tools.speaker_v2.pyannote_community1.run_subprocess_tool",
            return_value={"output": expected},
        ) as runner:
            result = PyannoteCommunity1SubprocessClient(config).diarize(
                "/audio.wav", context={}
            )
        self.assertEqual(result, expected)
        request = runner.call_args[0][2]
        self.assertNotIn("token", request["config"])
        self.assertNotIn("timeout_sec", request["config"])

    def test_pyannote_annotation_parser_keeps_speaker_labels(self):
        class Segment:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        class Annotation:
            def itertracks(self, yield_label=False):
                self.yield_label = yield_label
                return iter([(Segment(0.1, 0.8), "track", "SPEAKER_00")])

        annotation = Annotation()
        self.assertEqual(
            annotation_segments(annotation),
            [
                {
                    "start_sec": 0.1,
                    "end_sec": 0.8,
                    "speaker_id": "SPEAKER_00",
                }
            ],
        )
        self.assertTrue(annotation.yield_label)

    def test_assignment_comparison_is_diagnostic_only(self):
        timeline_a = make_timeline_evidence(
            "source_a", "G_a", two_speakers=True
        )
        timeline_b = make_timeline_evidence(
            "source_b", "G_b", two_speakers=False
        )
        asr = build_evidence(
            "sample",
            2.0,
            "lexical_timeline",
            "whisper",
            "v1",
            "asr",
            ["lexical_timeline"],
            ["G_whisper"],
            {
                "lexical_units": [
                    {
                        "unit_id": "w0",
                        "start_sec": 0.85,
                        "end_sec": 1.05,
                        "text": "yes",
                    }
                ]
            },
        )
        comparison = speaker_assignment_comparison(
            project_asr_track(asr, timeline_a),
            project_asr_track(asr, timeline_b),
        )
        self.assertEqual(comparison["common_unit_count"], 1)
        self.assertFalse(comparison["speaker_event_vote"])
        self.assertFalse(comparison["speaker_ids_compared"])

    def test_atomic_gzip_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json.gz"
            write_json_gz_atomic(path, {"sample": "value"})
            with gzip.open(str(path), "rt", encoding="utf-8") as source:
                self.assertEqual(json.load(source), {"sample": "value"})
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

    def test_artifact_stem_does_not_alias_distinct_sample_ids(self):
        self.assertNotEqual(safe_stem("a/b"), safe_stem("a?b"))
        self.assertEqual(safe_stem("sample_001"), "sample_001")

    def test_pipeline_hides_native_metadata_until_post_inference_scoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audio_path = tmpdir / "sample.wav"
            write_wav(audio_path, 2.0)
            record = make_record(str(audio_path))
            output_dir = tmpdir / "output"
            moss_result = ToolResult(
                "speaker.diarization_timeline",
                {
                    "segments": [
                        {
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "speaker_id": "S01",
                            "text": "hello",
                        }
                    ],
                    "raw_text": "[0][S01]hello[1]",
                },
                "moss",
                "fake",
            )
            vad_result = ToolResult(
                "basic_acoustic.silence_segments",
                [{"start_sec": 1.0, "end_sec": 2.0}],
                "vad",
                "fake",
                evidence={
                    "speech_segments": [{"start_sec": 0.0, "end_sec": 1.0}]
                },
            )
            lid_result = ToolResult(
                "language_content.language",
                "zh-mandarin",
                "firered_lid_detector",
                "fake",
                confidence=1.0,
            )
            config = SpeakerEvidenceConfig(
                moss_config=MossDiarizeConfig(
                    model="fake", subprocess_python="fake-python"
                ),
                vad_config=FireRedVadConfig(
                    model_dir=str(tmpdir), subprocess_python="fake-python"
                ),
                score_native=True,
                verify_model_assets=False,
            )
            with mock.patch(
                "tagger.pipelines.speaker_evidence.run_firered_lid",
                return_value=lid_result,
            ), mock.patch(
                "tagger.pipelines.speaker_evidence.run_moss_diarizer",
                return_value=moss_result,
            ) as moss_call, mock.patch(
                "tagger.pipelines.speaker_evidence.run_firered_vad",
                return_value=vad_result,
            ):
                result = run_record(record, tmpdir, output_dir, config, context={})

            self.assertEqual(result["status"], "ok")
            call_args = moss_call.call_args
            self.assertEqual(call_args[0][0], str(audio_path.resolve()))
            self.assertNotIn("native_metadata", call_args[1])
            with gzip.open(result["fusion_artifact"], "rt", encoding="utf-8") as source:
                fusion = json.load(source)
            self.assertFalse(fusion["input_provenance"]["native_metadata_entered_inference"])
            self.assertFalse(fusion["evaluation_only"]["entered_resolver"])
            self.assertEqual(fusion["evaluation_only"]["reference"]["observed_speaker_count"], 2)
            self.assertEqual(
                fusion["public_adapter"]["speaker"],
                result["evaluation_output"]["speaker"],
            )
            artifacts = result["artifacts"]
            self.assertTrue(Path(artifacts["certification_artifact"]).is_file())
            self.assertTrue(Path(artifacts["compat_metadata"]).is_file())
            self.assertTrue(artifacts["speaker_text_artifacts"])

    def test_run_manifest_records_models_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audio_path = tmpdir / "sample.wav"
            write_wav(audio_path, 2.0)
            manifest_path = tmpdir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(make_record(str(audio_path))) + "\n",
                encoding="utf-8",
            )
            output_dir = tmpdir / "output"
            config = SpeakerEvidenceConfig(
                moss_config=MossDiarizeConfig(),
                vad_config=FireRedVadConfig(),
                enable_moss=False,
                enable_vad=False,
                enable_campplus=False,
                enable_whisper=False,
                enable_sortformer=False,
                enable_pyannote=False,
                verify_model_assets=False,
            )
            summary = run_manifest(manifest_path, output_dir, config)
            run_manifest_path = Path(summary["run_manifest"])
            self.assertTrue(run_manifest_path.is_file())
            text = run_manifest_path.read_text(encoding="utf-8")
            value = json.loads(text)
            self.assertEqual(value["result"]["success_count"], 1)
            self.assertTrue(value["public_adapter_enabled"])
            self.assertNotIn('"token"', text)
            self.assertNotIn('"access_token"', text)


def make_timeline_evidence(
    source,
    group,
    two_speakers,
    calibrated=False,
    joint_negative=False,
    counts_for_certification=True,
):
    if two_speakers:
        segments = [
            {
                "start_sec": 0.0,
                "end_sec": 1.2,
                "speaker_id": "S01",
                "text": "hello",
            },
            {
                "start_sec": 0.8,
                "end_sec": 2.0,
                "speaker_id": "S02",
                "text": "yes",
            },
        ]
    else:
        segments = [
            {
                "start_sec": 0.0,
                "end_sec": 2.0,
                "speaker_id": "S01",
                "text": "hello yes",
            }
        ]
    return build_evidence(
        "sample",
        2.0,
        "speaker_timeline",
        source,
        "v1",
        "diarizer",
        ["speaker_timeline"],
        [group],
        {"timeline_summary": summarize_timeline(segments, 2.0)},
        quality={
            "usable": True,
            "calibration_profile_id": "cal-v1" if calibrated else None,
            "counts_for_certification": bool(counts_for_certification),
            "joint_negative_profile_id": "joint-neg-v1" if joint_negative else None,
            "joint_negative_claims": (
                ["multi_speaker", "speaker_overlap", "speaker_change"]
                if joint_negative
                else []
            ),
        },
    )


def make_identity_evidence(
    decision,
    calibrated,
    parent=None,
    candidate_selection_independent=True,
    comparison_kind="between_source_clusters",
):
    return build_evidence(
        "sample",
        2.0,
        "speaker_identity_matrix",
        "campplus",
        "v1",
        "speaker_embedding",
        ["speaker_identity_comparison"],
        ["G_identity_independent"],
        {
            "comparisons": [
                {
                    "speaker_pair": ["spk_001", "spk_002"],
                    "comparison_kind": comparison_kind,
                    "decision": decision,
                    "score": 0.9 if decision == "same" else 0.1,
                }
            ]
        },
        quality={
            "usable": True,
            "calibration_profile_id": "campplus-cal-v1" if calibrated else None,
            "candidate_selection_independent": bool(
                candidate_selection_independent
            ),
            "validator_dependency_closure_independent": bool(calibrated),
        },
        lineage={
            "parent_evidence_ids": [parent["evidence_id"]] if parent else []
        },
    )


def make_coverage_evidence(runtime=None, applicability=None, speech=True):
    return build_evidence(
        "sample",
        2.0,
        "speech_coverage",
        "vad",
        "v1",
        "vad",
        ["speech_coverage"],
        ["G_vad"],
        {
            "speech_segments": (
                [{"start_sec": 0.0, "end_sec": 2.0}] if speech else []
            )
        },
        quality={"usable": True, "full_scope_invocation": True},
        runtime=runtime,
        applicability=applicability,
    )


def make_record(audio_path):
    return {
        "corpus": {
            "dataset_name": "test",
            "source_urls": {
                "article": [],
                "github": [],
                "huggingface": [],
                "dataset_card": [],
            },
            "native_metadata": {},
        },
        "sample": {
            "sample_id": "sample",
            "audio": {"path": audio_path},
            "text": {"transcript": "oracle words must stay out of inference"},
            "native_metadata": {
                "utterances": [
                    {
                        "speaker": "A",
                        "start": 0.0,
                        "end": 1.2,
                        "text": "oracle",
                    },
                    {
                        "speaker": "B",
                        "start": 0.8,
                        "end": 2.0,
                        "text": "words",
                    },
                ]
            },
        },
    }


def write_wav(path, duration_sec):
    sample_rate = 16000
    frame_count = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(struct.pack("<%sh" % frame_count, *([0] * frame_count)))


if __name__ == "__main__":
    unittest.main()
