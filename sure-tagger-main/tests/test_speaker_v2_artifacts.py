import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tagger.pipelines.speaker_evidence import SpeakerEvidenceConfig
from tagger.tools.basic_acoustic.brouhaha_signal_estimator import BrouhahaConfig
from tagger.tools.speaker_v2.artifacts import (
    write_run_manifest,
    write_sample_artifacts,
)
from tagger.tools.speaker_v2.ecapa_identity import EcapaIdentityConfig
from tagger.tools.speaker_v2.nemotron3_diarization import (
    Nemotron3DiarizationConfig,
)
from tagger.tools.speaker_v2.profiles import expand_profile
from tagger.tools.speaker_v2.resolver import build_evaluation_output


class StubConfig:
    pass


class SpeakerV2ArtifactsTest(unittest.TestCase):
    def test_legacy_sample_call_persists_expanded_policy(self):
        fusion = {
            "schema_version": "speaker_fusion_artifact_v2.0-shadow.1",
            "sample_id": "sample",
            "fusion_id": "fusion_legacy",
            "profile": "v2-shadow",
            "claims": {"speaker_count": {"exact": None}},
            "public_adapter": {
                "speaker": {
                    "multi_speaker": None,
                    "speaker_change": None,
                    "speaker_overlap": None,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_sample_artifacts(tmpdir, "sample", [], fusion)
            with gzip.open(
                paths["fusion_artifact"], "rt", encoding="utf-8"
            ) as source:
                written_fusion = json.load(source)
            with gzip.open(
                paths["certification_artifact"], "rt", encoding="utf-8"
            ) as source:
                certification = json.load(source)

        self.assertEqual(written_fusion["run_profile"], "legacy-shadow")
        self.assertEqual(
            certification["policy_hash"], written_fusion["policy_hash"]
        )
        self.assertEqual(
            certification["claim_policy"], written_fusion["claim_policy"]
        )
        self.assertEqual(
            set(certification["claim_policy"]["claims"]),
            {
                "speaker_count",
                "multi_speaker",
                "speaker_overlap",
                "speaker_change",
            },
        )
        self.assertNotIn("certification_gate_enabled", certification)

    def test_sample_artifacts_persist_candidate_evaluation_output(self):
        claims = {
            "speaker_count": {
                "status": "supported",
                "observed_values": [{"value": 2}],
            },
            "multi_speaker": {
                "status": "supported",
                "candidate_value": True,
            },
            "speaker_overlap": {
                "status": "supported",
                "candidate_value": False,
            },
            "speaker_change": {
                "status": "supported",
                "candidate_value": True,
            },
        }
        evaluation_output = build_evaluation_output(claims)
        fusion = {
            "schema_version": "speaker_fusion_artifact_v2.0-shadow.1",
            "sample_id": "sample",
            "fusion_id": "fusion_candidate",
            "profile": "v2-shadow",
            "claims": claims,
            "evaluation_output": evaluation_output,
            "public_adapter": {
                "enabled": True,
                "speaker": {
                    "speaker_count": 2,
                    "multi_speaker": True,
                    "speaker_change_count": None,
                    "speaker_change": True,
                    "overlap_ratio": None,
                    "speaker_overlap": False,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_sample_artifacts(tmpdir, "sample", [], fusion)
            with gzip.open(
                paths["fusion_artifact"], "rt", encoding="utf-8"
            ) as source:
                written_fusion = json.load(source)
            with gzip.open(
                paths["certification_artifact"], "rt", encoding="utf-8"
            ) as source:
                certification = json.load(source)
            with gzip.open(
                paths["evaluation_output_artifact"], "rt", encoding="utf-8"
            ) as source:
                evaluation_artifact = json.load(source)
            with gzip.open(
                paths["compat_metadata"], "rt", encoding="utf-8"
            ) as source:
                compat_metadata = json.load(source)

        self.assertNotIn("certification_gate_enabled", written_fusion)
        self.assertEqual(written_fusion["evaluation_output"], evaluation_output)
        self.assertNotIn("certification_gate_enabled", certification)
        self.assertEqual(evaluation_artifact["output"], evaluation_output)
        self.assertEqual(evaluation_artifact["fusion_id"], "fusion_candidate")
        self.assertTrue(compat_metadata["published"])
        self.assertEqual(
            compat_metadata["speaker"], evaluation_output["speaker"]
        )

    def test_run_manifest_records_policy_and_all_models(self):
        config = StubConfig()
        config.run_profile = "quality-shadow"
        config.claim_policy = expand_profile("quality-shadow")["claim_policy"]
        config.ecapa_config = EcapaIdentityConfig(model_dir="ecapa-model")
        config.brouhaha_config = BrouhahaConfig(
            model_path="brouhaha.ckpt",
            repo_dir="brouhaha-repo",
            subprocess_python="brouhaha-python",
        )
        for name in (
            "moss",
            "firered_asr",
            "nemotron",
            "vad",
            "whisper",
            "sortformer",
            "pyannote",
            "campplus",
        ):
            setattr(config, "%s_config" % name, None)
        nemotron_config = Nemotron3DiarizationConfig(
            model_dir="nemotron-model",
            subprocess_python="nemotron-python",
            device="cuda:1",
            timeout_sec=87,
        )
        config.firered_asr_config = StubConfig()
        config.nemotron_config = nemotron_config
        enabled = {
            "moss": True,
            "firered_asr": True,
            "nemotron": True,
            "vad": True,
            "campplus": False,
            "whisper": False,
            "sortformer": True,
            "pyannote": True,
            "ecapa": True,
            "brouhaha": True,
        }
        for name, value in enabled.items():
            setattr(config, "enable_%s" % name, value)
        config.whisper_disabled_reason = "lexical audit disabled"
        summary = {
            "profile": "v2-shadow",
            "workers": 1,
            "model_workers": 1,
            "model_worker_overrides": {},
            "result_path": "results.jsonl",
            "processed_sample_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = tmpdir / "input.jsonl"
            manifest.write_text("", encoding="utf-8")
            path = write_run_manifest(
                tmpdir,
                manifest,
                "manifest-sha",
                config,
                summary,
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(value["run_profile"], "quality-shadow")
        self.assertEqual(
            value["policy_hash"], value["claim_policy"]["policy_hash"]
        )
        self.assertEqual(
            set(value["models"]),
            {
                "moss",
                "firered_asr",
                "qwen3_asr",
                "nemotron",
                "vad",
                "campplus",
                "whisper",
                "sortformer",
                "pyannote",
                "ecapa",
                "brouhaha",
                "gender",
            },
        )
        self.assertEqual(
            value["models"]["whisper"]["disabled_reason"],
            "lexical audit disabled",
        )
        self.assertEqual(
            value["models"]["ecapa"]["config"]["model_dir"],
            "ecapa-model",
        )
        self.assertEqual(
            value["models"]["brouhaha"]["config"]["subprocess_python"],
            "brouhaha-python",
        )
        self.assertEqual(
            value["models"]["nemotron"]["config"]["model_dir"],
            "nemotron-model",
        )
        self.assertEqual(
            value["models"]["nemotron"]["config"]["subprocess_python"],
            "nemotron-python",
        )
        self.assertEqual(value["models"]["nemotron"]["config"]["device"], "cuda:1")
        self.assertEqual(value["models"]["nemotron"]["config"]["timeout_sec"], 87)
        self.assertNotIn("certification_gate_enabled", value)
        self.assertEqual(value["evaluation_output"]["mode"], "direct")
        self.assertTrue(value["evaluation_output"]["production_eligible"])
        self.assertTrue(value["public_adapter_enabled"])

    def test_run_manifest_distinguishes_profile_defaults_from_overrides(self):
        config = SpeakerEvidenceConfig(
            profile_id="quality-shadow",
            whisper_config=StubConfig(),
            sortformer_config=StubConfig(),
            pyannote_config=StubConfig(),
            ecapa_config=EcapaIdentityConfig(model_dir="ecapa-model"),
            brouhaha_config=BrouhahaConfig(
                model_path="brouhaha.ckpt",
                repo_dir="brouhaha-repo",
                subprocess_python="brouhaha-python",
            ),
            enable_whisper=True,
            enable_ecapa=False,
        )
        summary = {
            "profile": "v2-shadow",
            "workers": 1,
            "model_workers": 1,
            "model_worker_overrides": {},
            "result_path": "results.jsonl",
            "processed_sample_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest = tmpdir / "input.jsonl"
            manifest.write_text("", encoding="utf-8")
            path = write_run_manifest(
                tmpdir,
                manifest,
                "manifest-sha",
                config,
                summary,
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertTrue(value["models"]["whisper"]["enabled"])
        self.assertFalse(
            value["models"]["whisper"]["profile_default_enabled"]
        )
        self.assertFalse(value["models"]["ecapa"]["enabled"])
        self.assertTrue(
            value["models"]["ecapa"]["profile_default_enabled"]
        )
        self.assertFalse(value["profile_model_defaults"]["whisper"])
        self.assertTrue(value["profile_model_defaults"]["ecapa"])
        self.assertNotIn("certification_gate_enabled", value)
        self.assertEqual(value["evaluation_output"]["mode"], "direct")
        self.assertTrue(value["evaluation_output"]["production_eligible"])


if __name__ == "__main__":
    unittest.main()
