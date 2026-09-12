import copy
import unittest
from unittest import mock

from scripts import run_speaker_evidence_v2 as speaker_v2_cli
from tagger.pipelines.speaker_evidence import (
    SpeakerEvidenceConfig,
    _select_identity_candidate_timeline,
    collect_brouhaha_evidence,
    collect_ecapa_evidence,
)
from tagger.tools.basic_acoustic.brouhaha_signal_estimator import BrouhahaConfig
from tagger.tools.speaker_v2.contracts import build_evidence
from tagger.tools.speaker_v2.ecapa_identity import EcapaIdentityConfig
from tagger.tools.speaker_v2.profiles import (
    ClaimPolicyError,
    claim_policy_hash,
    expand_profile,
    validate_claim_policy,
)
from tagger.tools.speaker_v2.resolver import resolve
from tagger.tools.speaker_v2.timeline import summarize_timeline


SAMPLE_ID = "sample"
DURATION_SEC = 4.0


def timeline(source, group, segments):
    return build_evidence(
        SAMPLE_ID,
        DURATION_SEC,
        "speaker_timeline",
        source,
        "v1",
        "diarizer",
        ["speaker_timeline"],
        [group],
        {"timeline_summary": summarize_timeline(segments, DURATION_SEC)},
        quality={"usable": True},
        applicability={"audio_sha256": "audio"},
    )


def source_timelines():
    moss = timeline(
        "moss_transcribe_diarize",
        "G_moss",
        [{"start_sec": 0.0, "end_sec": 4.0, "speaker_id": "m"}],
    )
    sortformer = timeline(
        "nvidia_streaming_sortformer_4spk_v2",
        "G_sortformer",
        [
            {"start_sec": 0.0, "end_sec": 2.0, "speaker_id": "a"},
            {"start_sec": 2.0, "end_sec": 4.0, "speaker_id": "b"},
        ],
    )
    pyannote = timeline(
        "pyannote_community_1",
        "G_pyannote",
        [
            {"start_sec": 0.0, "end_sec": 2.2, "speaker_id": "x"},
            {"start_sec": 1.0, "end_sec": 3.0, "speaker_id": "y"},
            {"start_sec": 3.0, "end_sec": 4.0, "speaker_id": "z"},
        ],
    )
    return moss, sortformer, pyannote


def diarizen_timeline():
    return timeline(
        "diarizen_large_s80_v2_official",
        "G_diarizen_large_s80_v2_official",
        [
            {"start_sec": 0.0, "end_sec": 2.5, "speaker_id": "a"},
            {"start_sec": 1.0, "end_sec": 4.0, "speaker_id": "b"},
        ],
    )


class SpeakerV2ProfilesTest(unittest.TestCase):
    def test_profile_model_sets_are_frozen(self):
        legacy = expand_profile("legacy-shadow")
        quality = expand_profile("quality-shadow")
        lean = expand_profile("lean-shadow")

        for profile in (legacy, quality, lean):
            self.assertTrue(profile["models"]["moss"])
            self.assertTrue(profile["models"]["firered_asr"])
            self.assertTrue(profile["models"]["qwen3_asr"])
            self.assertTrue(profile["models"]["diarizen"])
            self.assertTrue(profile["models"]["gender"])
            self.assertFalse(profile["models"]["sortformer"])
            self.assertFalse(profile["models"]["pyannote"])
        self.assertTrue(legacy["models"]["whisper"])
        self.assertFalse(legacy["models"]["ecapa"])
        self.assertFalse(quality["models"]["whisper"])
        self.assertTrue(quality["models"]["ecapa"])
        self.assertTrue(quality["models"]["brouhaha"])
        self.assertFalse(lean["models"]["vad"])

    def test_policy_hash_is_stable_and_validation_fails_closed(self):
        policy = expand_profile("quality-shadow")["claim_policy"]
        self.assertEqual(claim_policy_hash(policy), policy["policy_hash"])
        self.assertEqual(
            claim_policy_hash(copy.deepcopy(policy)),
            policy["policy_hash"],
        )

        unknown = copy.deepcopy(policy)
        unknown.pop("policy_hash")
        unknown["claims"]["speaker_count"]["primary_sources"] = ["unknown"]
        with self.assertRaises(ClaimPolicyError):
            validate_claim_policy(unknown)

        conflict = copy.deepcopy(policy)
        conflict.pop("policy_hash")
        conflict["claims"]["speaker_count"]["excluded_sources"].append(
            "nvidia_streaming_sortformer_4spk_v2"
        )
        with self.assertRaises(ClaimPolicyError):
            validate_claim_policy(conflict)

        with self.assertRaises(ClaimPolicyError):
            resolve(SAMPLE_ID, DURATION_SEC, [], claim_policy=None)
        with self.assertRaisesRegex(
            ClaimPolicyError, "requires a language_route object"
        ):
            resolve(SAMPLE_ID, DURATION_SEC, [], claim_policy=policy)

    def test_all_profiles_publish_only_language_routed_timelines(self):
        for profile_id in ("legacy-shadow", "quality-shadow", "lean-shadow"):
            claims = expand_profile(profile_id)["claim_policy"]["claims"]
            for rule in claims.values():
                self.assertEqual(
                    rule["primary_sources"],
                    [
                        "moss_transcribe_diarize",
                        "diarizen_large_s80_v2_official",
                    ],
                )
                self.assertEqual(rule["guard_sources"], [])
                self.assertEqual(rule["fallback_sources"], [])
                self.assertEqual(rule["selection"], "language_routed")
                self.assertEqual(
                    rule["excluded_sources"],
                    [
                        "nvidia_streaming_sortformer_4spk_v2",
                        "pyannote_community_1",
                    ],
                )

    def test_refresh_shadow_keeps_original_language_routed_claim_chain(self):
        profile = expand_profile("refresh-shadow")
        models = profile["models"]
        self.assertTrue(models["moss"])
        self.assertTrue(models["firered_asr"])
        self.assertTrue(models["qwen3_asr"])
        self.assertTrue(models["diarizen"])
        for model_name in (
            "sortformer",
            "pyannote",
            "ecapa",
            "brouhaha",
        ):
            self.assertFalse(models[model_name])

        for rule in profile["claim_policy"]["claims"].values():
            self.assertEqual(
                rule["primary_sources"],
                [
                    "moss_transcribe_diarize",
                    "diarizen_large_s80_v2_official",
                ],
            )
            self.assertEqual(rule["fallback_sources"], [])
            self.assertEqual(rule["selection"], "language_routed")

    def test_quality_policy_routes_every_claim_to_diarizen_when_moss_missing(self):
        _moss, sortformer, pyannote = source_timelines()
        diarizen = diarizen_timeline()
        profile = expand_profile("quality-shadow")
        fusion = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [diarizen, sortformer, pyannote],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "non_chinese_or_unknown"},
        )

        self.assertEqual(
            fusion["claims"]["speaker_count"]["route"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            fusion["claims"]["multi_speaker"]["route"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            fusion["claims"]["speaker_overlap"]["route"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            fusion["claims"]["speaker_change"]["route"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            fusion["claims"]["speaker_count"]["observed_values"][0]["value"],
            2,
        )
        self.assertTrue(
            fusion["claims"]["speaker_overlap"]["candidate_value"]
        )
        self.assertTrue(
            fusion["claims"]["speaker_change"]["candidate_value"]
        )
        self.assertIn(
            pyannote["evidence_id"],
            fusion["claims"]["speaker_count"]["route"][
                "diagnostic_evidence_ids"
            ],
        )

    def test_language_route_cannot_fall_back_to_the_other_branch(self):
        moss, _sortformer, _pyannote = source_timelines()
        diarizen = diarizen_timeline()
        profile = expand_profile("quality-shadow")

        chinese = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [moss, diarizen],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "chinese"},
        )
        non_chinese = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [moss, diarizen],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "non_chinese_or_unknown"},
        )
        english = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [moss, diarizen],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "english"},
        )
        failed_chinese = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [diarizen],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "chinese"},
        )

        self.assertEqual(
            chinese["claims"]["speaker_count"]["route"]["decision_sources"],
            ["moss_transcribe_diarize"],
        )
        self.assertEqual(
            non_chinese["claims"]["speaker_count"]["route"][
                "decision_sources"
            ],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            english["claims"]["speaker_count"]["route"]["decision_sources"],
            ["moss_transcribe_diarize"],
        )
        failed_route = failed_chinese["claims"]["speaker_count"]["route"]
        self.assertEqual(failed_route["selection"], "none")
        self.assertEqual(failed_route["decision_sources"], [])
        self.assertEqual(
            failed_route["language_selected_source"],
            "moss_transcribe_diarize",
        )
        self.assertEqual(
            failed_route["fallback_reason"],
            "selected language-route source is unavailable",
        )

    def test_resolver_publishes_direct_speaker_output(self):
        _moss, sortformer, pyannote = source_timelines()
        diarizen = diarizen_timeline()
        profile = expand_profile("quality-shadow")
        fusion = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [diarizen, sortformer, pyannote],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "non_chinese_or_unknown"},
        )

        output = fusion["evaluation_output"]
        self.assertEqual(output["mode"], "direct")
        self.assertTrue(output["production_eligible"])
        self.assertTrue(output["public_metadata_published"])
        self.assertEqual(output["speaker_count"], 2)
        self.assertEqual(output["speaker_change_count"], 1)
        self.assertEqual(output["overlap_ratio"], 0.375)
        self.assertEqual(
            output["speaker"],
            {
                "speaker_count": 2,
                "multi_speaker": True,
                "speaker_change_count": 1,
                "speaker_change": True,
                "overlap_ratio": 0.375,
                "speaker_overlap": True,
            },
        )
        self.assertEqual(
            {
                name: claim["value"]
                for name, claim in output["claims"].items()
            },
            {
                "speaker_count": 2,
                "multi_speaker": True,
                "speaker_overlap": True,
                "speaker_change": True,
            },
        )
        self.assertEqual(output["metrics"]["speaker_change_count"]["value"], 1)
        self.assertEqual(output["metrics"]["overlap_ratio"]["value"], 0.375)
        self.assertEqual(
            output["metrics"]["speaker_change_count"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertEqual(
            output["metrics"]["overlap_ratio"]["decision_sources"],
            ["diarizen_large_s80_v2_official"],
        )
        self.assertTrue(fusion["public_adapter"]["enabled"])
        self.assertEqual(
            fusion["public_adapter"]["speaker"], output["speaker"]
        )

    def test_excluded_diagnostics_cannot_conflict_with_public_claims(self):
        moss, sortformer, pyannote = source_timelines()
        profile = expand_profile("legacy-shadow")
        fusion = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [moss, sortformer, pyannote],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "chinese"},
        )

        output = fusion["evaluation_output"]
        self.assertEqual(output["mode"], "direct")
        self.assertEqual(output["speaker_count"], 1)
        self.assertEqual(output["speaker_change_count"], 0)
        self.assertEqual(output["overlap_ratio"], 0.0)
        self.assertEqual(
            {
                claim["availability"] for claim in output["claims"].values()
            },
            {"available"},
        )
        self.assertEqual(
            {
                metric["availability"]
                for metric in output["metrics"].values()
            },
            {"available"},
        )
        self.assertEqual(
            fusion["public_adapter"]["speaker"], output["speaker"]
        )

    def test_diarizen_is_selected_when_moss_is_unavailable(self):
        _moss, _sortformer, pyannote = source_timelines()
        diarizen = diarizen_timeline()
        profile = expand_profile("quality-shadow")
        fusion = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [diarizen, pyannote],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "non_chinese_or_unknown"},
        )
        route = fusion["claims"]["speaker_count"]["route"]
        self.assertEqual(route["selection"], "language_route")
        self.assertEqual(
            route["decision_sources"], ["diarizen_large_s80_v2_official"]
        )

    def test_diagnostic_timelines_are_excluded_not_guards(self):
        moss, sortformer, pyannote = source_timelines()
        profile = expand_profile("quality-shadow")
        fusion = resolve(
            SAMPLE_ID,
            DURATION_SEC,
            [moss, sortformer, pyannote],
            claim_policy=profile["claim_policy"],
            profile_id=profile["profile_id"],
            language_route={"language_route": "chinese"},
        )
        claim = fusion["claims"]["speaker_overlap"]
        self.assertFalse(claim["candidate_value"])
        self.assertEqual(
            claim["route"]["decision_sources"], ["moss_transcribe_diarize"]
        )
        self.assertEqual(claim["route"]["guard_observations"], [])
        self.assertIn(
            pyannote["evidence_id"], claim["route"]["diagnostic_evidence_ids"]
        )

    def test_config_expands_profiles_and_explicit_overrides(self):
        quality = SpeakerEvidenceConfig(
            profile_id="quality-shadow",
            sortformer_config=object(),
            pyannote_config=object(),
            ecapa_config=EcapaIdentityConfig(model_dir="ecapa"),
            brouhaha_config=BrouhahaConfig(
                model_path="checkpoint",
                repo_dir="repo",
                subprocess_python="python",
            ),
            enable_whisper=True,
            whisper_config=object(),
            enable_ecapa=False,
        )
        self.assertTrue(quality.enable_whisper)
        self.assertFalse(quality.enable_ecapa)
        self.assertTrue(quality.enable_brouhaha)
        self.assertEqual(quality.profile_id, "quality-shadow")
        self.assertEqual(
            quality.expanded_run_profile["models"]["ecapa"], False
        )
        self.assertTrue(quality.profile_model_defaults["ecapa"])
        self.assertFalse(quality.profile_model_defaults["whisper"])
        self.assertEqual(
            quality.expanded_run_profile["profile_model_defaults"],
            quality.profile_model_defaults,
        )

    def test_legacy_openasr_toggle_and_independent_model_toggles(self):
        disabled = SpeakerEvidenceConfig(enable_openasr=False)
        self.assertFalse(disabled.enable_openasr)
        self.assertFalse(disabled.enable_firered_asr)
        self.assertFalse(disabled.enable_qwen3_asr)
        self.assertFalse(disabled.enable_diarizen)
        self.assertFalse(
            disabled.expanded_run_profile["models"]["firered_asr"]
        )
        self.assertFalse(
            disabled.expanded_run_profile["models"]["qwen3_asr"]
        )
        self.assertFalse(disabled.expanded_run_profile["models"]["diarizen"])

        no_diarizen = SpeakerEvidenceConfig(enable_diarizen=False)
        self.assertFalse(no_diarizen.enable_openasr)
        self.assertTrue(no_diarizen.enable_firered_asr)
        self.assertFalse(no_diarizen.enable_diarizen)

        no_firered = SpeakerEvidenceConfig(enable_firered_asr=False)
        self.assertFalse(no_firered.enable_openasr)
        self.assertFalse(no_firered.enable_firered_asr)
        self.assertTrue(no_firered.enable_diarizen)

        no_qwen3 = SpeakerEvidenceConfig(enable_qwen3_asr=False)
        self.assertFalse(no_qwen3.enable_openasr)
        self.assertTrue(no_qwen3.enable_firered_asr)
        self.assertFalse(no_qwen3.enable_qwen3_asr)
        self.assertTrue(no_qwen3.enable_diarizen)

    def test_cli_preserves_profile_defaults_before_overrides(self):
        captured = {}

        def fake_run_manifest(_manifest, _output_dir, config, **_kwargs):
            captured["config"] = config
            return {"failure_count": 0, "results": []}

        with mock.patch.object(
            speaker_v2_cli, "run_manifest", side_effect=fake_run_manifest
        ), mock.patch("builtins.print"):
            status = speaker_v2_cli.main(
                [
                    "--manifest",
                    "input.jsonl",
                    "--output-dir",
                    "output",
                    "--profile",
                    "quality-shadow",
                    "--whisper-enable",
                    "--ecapa-disable",
                ]
            )

        config = captured["config"]
        self.assertEqual(status, 0)
        self.assertFalse(config.profile_model_defaults["whisper"])
        self.assertTrue(config.profile_model_defaults["ecapa"])
        self.assertTrue(config.expanded_run_profile["models"]["whisper"])
        self.assertFalse(config.expanded_run_profile["models"]["ecapa"])
        self.assertEqual(
            config.expanded_run_profile["profile_model_defaults"],
            config.profile_model_defaults,
        )

    def test_cli_no_longer_exposes_a_certification_gate(self):
        parser = speaker_v2_cli.build_arg_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--certification-gate-enable", options)
        self.assertIn("--openasr-enable", options)
        self.assertIn("--openasr-disable", options)
        self.assertIn("--firered-asr-enable", options)
        self.assertIn("--firered-asr-disable", options)
        self.assertIn("--qwen3-asr-enable", options)
        self.assertIn("--qwen3-asr-disable", options)
        self.assertIn("--diarizen-enable", options)
        self.assertIn("--diarizen-disable", options)
        self.assertIn("--firered-asr-workers", options)
        self.assertIn("--qwen3-asr-workers", options)
        self.assertIn("--diarizen-workers", options)

    def test_cli_exposes_independent_asr_diarizen_and_lid_controls(self):
        captured = {}

        def fake_run_manifest(_manifest, _output_dir, config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs
            return {"failure_count": 0, "results": []}

        with mock.patch.object(
            speaker_v2_cli, "run_manifest", side_effect=fake_run_manifest
        ), mock.patch("builtins.print"):
            status = speaker_v2_cli.main(
                [
                    "--manifest",
                    "input.jsonl",
                    "--output-dir",
                    "output",
                    "--firered-asr-disable",
                    "--diarizen-enable",
                    "--firered-asr-workers",
                    "2",
                    "--qwen3-asr-workers",
                    "4",
                    "--diarizen-workers",
                    "3",
                    "--firered-lid-half",
                    "--firered-lid-model",
                    "/models/FireRedLID",
                    "--diarizen-model-dir",
                    "/models/diarizen",
                    "--diarizen-embedding-model",
                    "/models/pyannote-wespeaker/pytorch_model.bin",
                    "--diarizen-source-dir",
                    "/src/DiariZen",
                    "--diarizen-python",
                    "/runtime/diarizen-python",
                ]
            )

        self.assertEqual(status, 0)
        config = captured["config"]
        self.assertFalse(config.enable_openasr)
        self.assertFalse(config.enable_firered_asr)
        self.assertTrue(config.enable_qwen3_asr)
        self.assertTrue(config.enable_diarizen)
        self.assertTrue(config.firered_lid_config.use_half)
        self.assertEqual(
            config.firered_lid_config.model_dir, "/models/FireRedLID"
        )
        self.assertEqual(
            config.diarizen_config.model_dir, "/models/diarizen"
        )
        self.assertEqual(
            config.diarizen_config.embedding_model,
            "/models/pyannote-wespeaker/pytorch_model.bin",
        )
        self.assertEqual(
            config.diarizen_config.source_dir, "/src/DiariZen"
        )
        self.assertEqual(
            config.diarizen_config.subprocess_python,
            "/runtime/diarizen-python",
        )
        worker_overrides = captured["kwargs"]["model_worker_overrides"]
        self.assertEqual(worker_overrides["firered_asr_estimate"], 2)
        self.assertEqual(worker_overrides["qwen3_asr_estimate"], 4)
        self.assertEqual(worker_overrides["diarizen_official_estimate"], 3)

    def test_identity_candidate_uses_count_policy_not_evidence_order(self):
        moss, sortformer, pyannote = source_timelines()
        policy = expand_profile("quality-shadow")["claim_policy"]
        selected = _select_identity_candidate_timeline(
            [moss, pyannote, sortformer], policy
        )
        self.assertEqual(
            selected["source"]["name"],
            "moss_transcribe_diarize",
        )


class SpeakerV2CollectorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "sample_id": SAMPLE_ID,
            "audio_path": "/blind/audio.wav",
            "audio_sha256": "audio",
            "duration_sec": DURATION_SEC,
            "sample_rate_hz": 16000,
            "channels": 1,
        }

    def test_brouhaha_collector_is_coverage_only(self):
        config = BrouhahaConfig(
            model_path="checkpoint",
            repo_dir="repo",
            subprocess_python="python",
            model_version="brouhaha@revision",
        )
        output = {
            "raw_speech_segments": [{"start_sec": 0.0, "end_sec": 2.0}],
            "speech_segments": [{"start_sec": 0.0, "end_sec": 2.0}],
            "silence_segments": [{"start_sec": 2.0, "end_sec": 4.0}],
            "speech_duration_sec": 2.0,
            "speech_coverage_ratio": 0.5,
            "binarization": {"onset": 0.78, "offset": 0.78},
            "boundary_postprocess": "clip_sort_merge_to_audio_duration",
            "runtime": {"execution": "shared_subprocess_worker"},
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.estimate_brouhaha_coverage",
            return_value=output,
        ):
            evidence = collect_brouhaha_evidence(
                self.scope, config, verify_model_asset=False
            )

        self.assertEqual(evidence["capabilities"], ["speech_coverage"])
        self.assertEqual(evidence["source"]["name"], "brouhaha_vad")
        self.assertTrue(evidence["payload"]["not_a_speaker_event_vote"])
        self.assertNotIn("snr", evidence["payload"])
        self.assertNotIn("c50", evidence["payload"])

    def test_ecapa_collector_records_predicted_timeline_lineage(self):
        timeline_evidence = source_timelines()[1]
        config = EcapaIdentityConfig(
            model_dir="ecapa",
            subprocess_python="python",
        )
        output = {
            "regions": [
                {
                    "region_id": "r1",
                    "speaker_id": "a",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                },
                {
                    "region_id": "r2",
                    "speaker_id": "b",
                    "start_sec": 2.0,
                    "end_sec": 3.0,
                },
            ],
            "comparisons": [
                {
                    "comparison_kind": "cross_source_cluster",
                    "region_ids": ["r1", "r2"],
                    "speaker_pair": ["a", "b"],
                    "score": 0.1,
                    "decision": "different",
                    "threshold": config.threshold,
                }
            ],
            "score_kind": "cosine_similarity",
        }
        with mock.patch(
            "tagger.pipelines.speaker_evidence.compare_ecapa_regions",
            return_value=output,
        ) as compare:
            evidence = collect_ecapa_evidence(
                self.scope,
                timeline_evidence,
                config,
                verify_model_asset=False,
            )

        self.assertEqual(
            evidence["lineage"]["parent_evidence_ids"],
            [timeline_evidence["evidence_id"]],
        )
        self.assertEqual(
            evidence["source"]["name"], "speechbrain_ecapa_voxceleb"
        )
        self.assertFalse(evidence["quality"]["counts_for_certification"])
        self.assertFalse(
            evidence["quality"]["candidate_selection_independent"]
        )
        passed_summary = compare.call_args[0][1]
        self.assertNotIn("native_metadata", passed_summary)


if __name__ == "__main__":
    unittest.main()
