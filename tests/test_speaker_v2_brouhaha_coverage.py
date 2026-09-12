import tempfile
import unittest
from unittest import mock

from tagger.tools.basic_acoustic.brouhaha_signal_estimator import BrouhahaConfig
from tagger.tools.speaker_v2 import brouhaha_coverage


FORBIDDEN_SPEAKER_CAPABILITIES = {
    "speaker_count_candidate",
    "multi_speaker_candidate",
    "overlap_candidate",
    "change_candidate",
}


class FakeBrouhahaClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def estimate(self, audio_path, context=None):
        self.calls.append((str(audio_path), context))
        return self.output


class BrouhahaCoverageTest(unittest.TestCase):
    def test_only_declares_speech_coverage_and_normalizes_segments(self):
        client = FakeBrouhahaClient(
            {
                "annotation": [
                    {"start_sec": 3.0, "end_sec": 5.0},
                    {"start_sec": -0.5, "end_sec": 1.0},
                    {"start_sec": 0.5, "end_sec": 2.0},
                ],
                "snr": [12.0],
                "c50": [1.0],
            }
        )
        config = BrouhahaConfig(
            model_path="unused.ckpt",
            repo_dir="",
            subprocess_python="",
        )

        result = brouhaha_coverage.estimate_coverage(
            "sample.wav",
            4.0,
            config=config,
            context={"request_id": "test"},
            client=client,
        )

        self.assertEqual(result["evidence_type"], "speech_coverage")
        self.assertEqual(result["capabilities"], ["speech_coverage"])
        self.assertFalse(
            FORBIDDEN_SPEAKER_CAPABILITIES.intersection(result["capabilities"])
        )
        self.assertEqual(
            result["speech_segments"],
            [
                {"start_sec": 0.0, "end_sec": 2.0},
                {"start_sec": 3.0, "end_sec": 4.0},
            ],
        )
        self.assertEqual(
            result["silence_segments"],
            [{"start_sec": 2.0, "end_sec": 3.0}],
        )
        self.assertEqual(result["speech_duration_sec"], 3.0)
        self.assertEqual(result["speech_coverage_ratio"], 0.75)
        self.assertNotIn("snr", result)
        self.assertNotIn("c50", result)
        self.assertEqual(
            client.calls,
            [("sample.wav", {"request_id": "test"})],
        )

    def test_empty_prediction_is_valid_zero_coverage(self):
        result = brouhaha_coverage.estimate_coverage(
            "sample.wav",
            2.5,
            config=BrouhahaConfig(
                model_path="unused.ckpt",
                repo_dir="",
                subprocess_python="",
            ),
            client=FakeBrouhahaClient({"annotation": []}),
        )

        self.assertEqual(result["speech_segments"], [])
        self.assertEqual(
            result["silence_segments"],
            [{"start_sec": 0.0, "end_sec": 2.5}],
        )
        self.assertEqual(result["speech_coverage_ratio"], 0.0)

    def test_configured_subprocess_uses_existing_brouhaha_worker_client(self):
        config = BrouhahaConfig(
            model_path="unused.ckpt",
            repo_dir="",
            subprocess_python="/runtime/python",
        )
        subprocess_client = mock.Mock()
        subprocess_client.estimate.return_value = {"annotation": []}
        with mock.patch.object(
            brouhaha_coverage,
            "BrouhahaSubprocessClient",
            return_value=subprocess_client,
        ) as client_class:
            result = brouhaha_coverage.estimate_coverage(
                "sample.wav",
                1.0,
                config=config,
            )

        client_class.assert_called_once_with(config)
        subprocess_client.estimate.assert_called_once_with(
            "sample.wav",
            context=None,
        )
        self.assertEqual(result["runtime"]["execution"], "shared_subprocess_worker")

    def test_invalid_duration_fails_before_inference(self):
        client = FakeBrouhahaClient({"annotation": []})
        with self.assertRaises(brouhaha_coverage.BrouhahaCoverageError):
            brouhaha_coverage.estimate_coverage(
                "sample.wav",
                0.0,
                client=client,
            )
        self.assertEqual(client.calls, [])

    def test_model_verifier_rejects_unpinned_checkpoint(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint.write(b"not the pinned checkpoint")
            checkpoint.flush()
            with self.assertRaises(brouhaha_coverage.BrouhahaCoverageError):
                brouhaha_coverage.verify_model_asset(checkpoint.name)


if __name__ == "__main__":
    unittest.main()
