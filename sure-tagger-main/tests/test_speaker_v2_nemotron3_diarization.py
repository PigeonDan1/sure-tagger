import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import wave

from tagger.tools.speaker_v2.nemotron3_diarization import (
    MAX_SPEAKERS,
    MODEL_SLOT_KEY,
    Nemotron3DiarizationConfig,
    Nemotron3DiarizationError,
    _check_model_files,
    _read_audio,
    native_segments_to_timeline,
    normalize_result,
)
from tagger.tools.subprocess_worker import dispatch


class Nemotron3DiarizationAdapterTest(unittest.TestCase):
    def test_cuda_is_required_by_default(self):
        config = Nemotron3DiarizationConfig()
        self.assertEqual(config.device, "cuda:0")
        with self.assertRaisesRegex(ValueError, "requires an explicit CUDA"):
            Nemotron3DiarizationConfig(device="cpu")
        self.assertEqual(
            Nemotron3DiarizationConfig(device="cuda:1").device, "cuda:1"
        )

    def test_native_segments_preserve_speakers_and_overlap(self):
        native = [
            {"Start": 2.0, "End": 3.0, "Speaker": 7},
            {"Start": 0.0, "End": 2.5, "Speaker": 0},
            {"Start": 3.9, "End": 4.2, "Speaker": 0},
        ]
        self.assertEqual(
            native_segments_to_timeline(native, 4.0),
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 2.5,
                    "speaker_id": "speaker_1",
                    "overlap": True,
                },
                {
                    "start_sec": 2.0,
                    "end_sec": 3.0,
                    "speaker_id": "speaker_8",
                    "overlap": True,
                },
                {
                    "start_sec": 3.9,
                    "end_sec": 4.0,
                    "speaker_id": "speaker_1",
                    "overlap": False,
                },
            ],
        )
        self.assertEqual(MAX_SPEAKERS, 8)

    def test_invalid_native_segments_are_rejected(self):
        invalid_rows = [
            {"Start": 1, "End": 2, "Speaker": 8},
            {"Start": 1, "End": 2, "Speaker": True},
            {"Start": 2, "End": 1, "Speaker": 0},
            {"Start": float("nan"), "End": 1, "Speaker": 0},
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(Nemotron3DiarizationError):
                native_segments_to_timeline([row], 4.0)

    def test_subprocess_result_contract_is_validated(self):
        with self.assertRaises(Nemotron3DiarizationError):
            normalize_result({"segments": "invalid"})
        with self.assertRaises(Nemotron3DiarizationError):
            normalize_result(
                {
                    "segments": [
                        {"start_sec": 2, "end_sec": 1, "speaker_id": "speaker_1"}
                    ]
                }
            )

    def test_model_asset_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = {
                "model.safetensors": b"checkpoint",
                "config.json": b"config",
                "processor_config.json": b"processor",
            }
            for name, content in assets.items():
                (root / name).write_bytes(content)
            hashes = [hashlib.sha256(content).hexdigest() for content in assets.values()]
            with mock.patch(
                "tagger.tools.speaker_v2.nemotron3_diarization.MODEL_SHA256",
                hashes[0],
            ), mock.patch(
                "tagger.tools.speaker_v2.nemotron3_diarization.CONFIG_SHA256",
                hashes[1],
            ), mock.patch(
                "tagger.tools.speaker_v2.nemotron3_diarization.PROCESSOR_CONFIG_SHA256",
                hashes[2],
            ):
                _check_model_files(root, True)
                (root / "config.json").write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    Nemotron3DiarizationError, "SHA-256 mismatch"
                ):
                    _check_model_files(root, True)

    def test_worker_dispatch_uses_new_slot(self):
        class FakeClient:
            def __init__(self, config):
                self.config = config

            def diarize(self, audio_path, context=None):
                return {"segments": [], "runtime": {"device": self.config.device}}

        request = {
            "audio_path": "/audio.wav",
            "config": {"device": "cuda:1", "subprocess_python": "/python"},
        }
        with mock.patch(
            "tagger.tools.speaker_v2.nemotron3_diarization.Nemotron3DiarizationClient",
            FakeClient,
        ), mock.patch("tagger.tools.subprocess_worker._CLIENTS", {}):
            self.assertEqual(
                dispatch(MODEL_SLOT_KEY, request),
                {"output": {"segments": [], "runtime": {"device": "cuda:1"}}},
            )

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("numpy", "soundfile", "scipy")),
        "audio preprocessing packages are not installed",
    )
    def test_stereo_audio_uses_channel_zero_and_resamples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo_8k.wav"
            with wave.open(str(path), "wb") as sink:
                sink.setnchannels(2)
                sink.setsampwidth(2)
                sink.setframerate(8000)
                sink.writeframes(b"".join(struct.pack("<hh", 1000, -1000) for _ in range(800)))
            samples, details = _read_audio(path)
        self.assertEqual(len(samples), 1600)
        self.assertEqual(details["source_channels"], 2)
        self.assertEqual(details["source_sample_rate_hz"], 8000)
        self.assertEqual(details["channel_selection"], 0)
        self.assertGreater(float(samples.mean()), 0)


if __name__ == "__main__":
    unittest.main()
