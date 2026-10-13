import unittest
from unittest import mock

from tagger.tools import subprocess_worker
from tagger.tools.speaker_v2.ecapa_identity import (
    DEFAULT_CALIBRATION_PROFILE_ID,
    DEFAULT_THRESHOLD,
    EcapaIdentityClient,
    EcapaIdentityConfig,
    EcapaIdentityError,
    EcapaIdentitySubprocessClient,
    select_candidate_regions,
    validate_subprocess_request,
)


def make_regions():
    return [
        {
            "region_id": "r1",
            "speaker_id": "predicted_a",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "selection": "source_timeline_nonoverlap_candidate",
        },
        {
            "region_id": "r2",
            "speaker_id": "predicted_a",
            "start_sec": 1.0,
            "end_sec": 2.0,
            "selection": "source_timeline_nonoverlap_candidate",
        },
        {
            "region_id": "r3",
            "speaker_id": "predicted_b",
            "start_sec": 2.0,
            "end_sec": 3.0,
            "selection": "source_timeline_nonoverlap_candidate",
        },
    ]


class EcapaIdentityTest(unittest.TestCase):
    def test_default_config_uses_frozen_atomic_profile(self):
        config = EcapaIdentityConfig(model_dir="unused")

        self.assertEqual(config.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(
            config.calibration_profile_id, DEFAULT_CALIBRATION_PROFILE_ID
        )
        self.assertEqual(config.to_record()["model_dir"], "unused")

    def test_candidate_selection_reuses_campplus_nonoverlap_contract(self):
        timeline = {
            "activity_segments": [
                {"start_sec": 0.0, "end_sec": 2.0, "speaker_id": "A"},
                {"start_sec": 0.75, "end_sec": 1.25, "speaker_id": "B"},
            ],
            "overlap_activity_segments": [
                {"start_sec": 0.75, "end_sec": 1.25}
            ],
        }
        config = EcapaIdentityConfig(
            model_dir="unused",
            min_region_duration_sec=0.70,
            max_regions_per_speaker=2,
        )

        regions = select_candidate_regions(timeline, config)

        self.assertEqual(
            [(item["start_sec"], item["end_sec"]) for item in regions],
            [(0.0, 0.75), (1.25, 2.0)],
        )
        self.assertTrue(
            all(
                item["selection"] == "source_timeline_nonoverlap_candidate"
                for item in regions
            )
        )

    def test_candidate_selection_passes_only_predicted_region_fields(self):
        timeline = {
            "segments": [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "speaker_id": "predicted_a",
                    "text": "ignored predicted transcript",
                }
            ],
            "overlap_segments": [],
            "observed_speaker_count": 1,
        }
        config = EcapaIdentityConfig(model_dir="unused")
        sentinel = [{"region_id": "safe"}]
        with mock.patch(
            "tagger.tools.speaker_v2.ecapa_identity._campplus_select_candidate_regions",
            return_value=sentinel,
        ) as selector:
            result = select_candidate_regions(timeline, config)

        self.assertIs(result, sentinel)
        passed_timeline = selector.call_args[0][0]
        self.assertEqual(
            passed_timeline,
            {
                "activity_segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "speaker_id": "predicted_a",
                    }
                ],
                "overlap_activity_segments": [],
            },
        )

    def test_native_metadata_and_gold_fields_are_rejected(self):
        config = EcapaIdentityConfig(model_dir="unused")
        timeline = {
            "segments": [
                {"start_sec": 0.0, "end_sec": 1.0, "speaker_id": "A"}
            ],
            "native_metadata": {"utterances": []},
        }
        with self.assertRaises(EcapaIdentityError):
            select_candidate_regions(timeline, config)

        regions = make_regions()
        regions[0]["gold_same"] = True
        client = EcapaIdentityClient(config)
        with self.assertRaises(EcapaIdentityError):
            client.compare_regions("unused.wav", regions)

    def test_comparison_output_is_campplus_compatible(self):
        config = EcapaIdentityConfig(model_dir="unused", threshold=0.5)
        client = EcapaIdentityClient(config)
        client._embeddings_for_regions = mock.Mock(
            return_value={
                "r1": [1.0, 0.0],
                "r2": [1.0, 0.0],
                "r3": [-1.0, 0.0],
            }
        )

        result = client.compare_regions("unused.wav", make_regions())

        self.assertEqual(len(result["comparisons"]), 3)
        within = result["comparisons"][0]
        cross = result["comparisons"][1]
        compatibility_fields = {
            "comparison_kind",
            "region_ids",
            "speaker_pair",
            "score",
            "decision",
            "threshold",
            "model_output_text",
        }
        self.assertTrue(compatibility_fields <= set(within))
        self.assertEqual(within["comparison_kind"], "within_source_cluster")
        self.assertEqual(within["decision"], "same")
        self.assertEqual(within["score"], 1.0)
        self.assertEqual(cross["comparison_kind"], "cross_source_cluster")
        self.assertEqual(cross["decision"], "different")
        self.assertEqual(cross["score"], -1.0)
        self.assertGreaterEqual(within["probability_same"], 0.0)
        self.assertLessEqual(within["probability_same"], 1.0)

    def test_embedding_dimensions_must_match(self):
        config = EcapaIdentityConfig(model_dir="unused")
        client = EcapaIdentityClient(config)
        client._embeddings_for_regions = mock.Mock(
            return_value={"r1": [1.0], "r2": [1.0, 0.0], "r3": [1.0]}
        )

        with self.assertRaises(EcapaIdentityError):
            client.compare_regions("unused.wav", make_regions())

    def test_uncalibrated_output_does_not_publish_probability(self):
        config = EcapaIdentityConfig(
            model_dir="unused", calibration_profile_id=""
        )
        client = EcapaIdentityClient(config)
        client._embeddings_for_regions = mock.Mock(
            return_value={
                "r1": [1.0, 0.0],
                "r2": [1.0, 0.0],
                "r3": [-1.0, 0.0],
            }
        )

        result = client.compare_regions("unused.wav", make_regions())

        self.assertIsNone(result["calibration_profile_id"])
        self.assertTrue(
            all(
                item["probability_same"] is None
                for item in result["comparisons"]
            )
        )

    def test_subprocess_client_sends_only_blind_inference_payload(self):
        config = EcapaIdentityConfig(
            model_dir="model",
            subprocess_python="runtime/bin/python",
            timeout_sec=17,
        )
        expected = {"regions": [], "comparisons": []}
        with mock.patch(
            "tagger.tools.speaker_v2.ecapa_identity.run_subprocess_tool",
            return_value={"output": expected},
        ) as runner:
            result = EcapaIdentitySubprocessClient(config).compare_regions(
                "audio.wav", make_regions()
            )

        self.assertIs(result, expected)
        self.assertEqual(runner.call_args[0][1], "ecapa_identity_estimate")
        request = runner.call_args[0][2]
        self.assertEqual(set(request), {"audio_path", "regions", "config"})
        self.assertNotIn("timeout_sec", request["config"])
        self.assertEqual(request["config"]["subprocess_python"], "")
        self.assertEqual(runner.call_args[1]["timeout_sec"], 17)

    def test_worker_route_uses_cached_ecapa_client(self):
        config = EcapaIdentityConfig(model_dir="unused").to_record()
        fake_client = mock.Mock()
        fake_client.compare_regions.return_value = {
            "regions": make_regions(),
            "comparisons": [],
        }
        request = {
            "audio_path": "audio.wav",
            "regions": make_regions(),
            "config": config,
        }
        with mock.patch.object(
            subprocess_worker, "_cached_client", return_value=fake_client
        ) as cached:
            result = subprocess_worker.dispatch(
                "ecapa_identity_estimate", request
            )

        self.assertEqual(result["output"]["comparisons"], [])
        self.assertEqual(cached.call_args[0][0], "ecapa_identity_estimate")
        fake_client.compare_regions.assert_called_once_with(
            "audio.wav", request["regions"]
        )

    def test_worker_request_rejects_extra_metadata(self):
        request = {
            "audio_path": "audio.wav",
            "regions": make_regions(),
            "config": EcapaIdentityConfig(model_dir="unused").to_record(),
            "native_metadata": {},
        }

        with self.assertRaises(EcapaIdentityError):
            validate_subprocess_request(request)


if __name__ == "__main__":
    unittest.main()
