import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from tagger.tools.speaker_v2.contracts import build_evidence
from tagger.tools.speaker_v2 import speaker_gender
from tagger.tools.speaker_v2.speaker_gender import (
    AFRISPEECH_MODEL,
    ALICE_MODEL,
    GRIKO_MODEL,
    SpeakerGenderConfig,
    gender_model_for_language,
    infer_speaker_genders,
)
from tagger.tools.speaker_v2.timeline import summarize_timeline
from tagger.pipelines.speaker_evidence import _merge_profile_genders


def timeline(segments, duration=6.0):
    return build_evidence(
        "sample",
        duration,
        "speaker_timeline",
        "moss_transcribe_diarize",
        "v1",
        "diarizer",
        ["speaker_timeline"],
        ["G_timeline"],
        {"timeline_summary": summarize_timeline(segments, duration)},
        quality={"usable": True},
    )


def config():
    return SpeakerGenderConfig(
        alice_python="",
        afrispeech_python="",
        griko_python="",
        verify_model_assets=False,
    )


class SpeakerGenderTest(unittest.TestCase):
    def test_alice_rejects_onnx_cpu_fallback_without_caching_session(self):
        cpu_session = mock.Mock()
        cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
        ort = mock.Mock()
        ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        ort.InferenceSession.return_value = cpu_session
        client = speaker_gender.SpeakerGenderClient(
            SpeakerGenderConfig(device="cuda:0", verify_model_assets=False)
        )
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}), mock.patch.object(
            speaker_gender, "_required_file", return_value=Path("/alice.onnx")
        ), mock.patch.object(speaker_gender, "_verify_hash"):
            with self.assertRaisesRegex(
                speaker_gender.SpeakerGenderError, "refusing CPU fallback"
            ):
                client._predict_alice(np.zeros(16000, dtype=np.float32))
        self.assertIsNone(client._alice_session)

    def test_afrispeech_rejects_cpu_head_without_caching_either_session(self):
        gpu_session = mock.Mock()
        gpu_session.get_providers.return_value = ["CUDAExecutionProvider"]
        cpu_session = mock.Mock()
        cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
        ort = mock.Mock()
        ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        ort.InferenceSession.side_effect = [gpu_session, cpu_session]
        client = speaker_gender.SpeakerGenderClient(
            SpeakerGenderConfig(device="cuda:0", verify_model_assets=False)
        )
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}), mock.patch.object(
            speaker_gender, "_required_file", side_effect=lambda path, label: Path(path)
        ), mock.patch.object(speaker_gender, "_verify_hash"):
            with self.assertRaisesRegex(
                speaker_gender.SpeakerGenderError, "refusing CPU fallback"
            ):
                client._predict_afrispeech(np.zeros(16000, dtype=np.float32))
        self.assertIsNone(client._afrispeech_embedding_session)
        self.assertIsNone(client._afrispeech_session)

    def test_language_model_routes(self):
        for language in ("zh", "zh-mandarin", "en", "en-us", "english"):
            self.assertEqual(gender_model_for_language(language), ALICE_MODEL)
        for language in ("sw", "swa", "am", "amh", "yo", "yor", "so"):
            self.assertEqual(
                gender_model_for_language(language), AFRISPEECH_MODEL
            )
        for language in ("ar", "es", "fr", "ja", "unknown", None):
            self.assertEqual(gender_model_for_language(language), GRIKO_MODEL)

    def test_speakers_are_numbered_by_first_speech_not_cluster_label(self):
        evidence = timeline(
            [
                {"start_sec": 3.0, "end_sec": 4.5, "speaker_id": "A"},
                {"start_sec": 0.2, "end_sec": 2.0, "speaker_id": "Z"},
            ]
        )
        with mock.patch.object(
            speaker_gender,
            "predict_gender",
            return_value={"gender": "female", "score": 0.9},
        ):
            result = infer_speaker_genders(
                evidence, "/audio.wav", "en", config=config()
            )

        self.assertEqual(
            [item["source_speaker_id"] for item in result["speakers"]],
            ["z", "a"],
        )
        self.assertEqual(
            [item["speaker_id"] for item in result["speakers"]],
            ["speaker_1", "speaker_2"],
        )
        self.assertEqual(
            evidence["payload"]["timeline_summary"]["speaker_ids"],
            ["z", "a"],
        )

    def test_multi_speaker_removes_overlap_and_concatenates_clean_intervals(self):
        evidence = timeline(
            [
                {"start_sec": 0.0, "end_sec": 2.0, "speaker_id": "late-label"},
                {"start_sec": 1.0, "end_sec": 3.0, "speaker_id": "early-label"},
                {"start_sec": 4.0, "end_sec": 4.5, "speaker_id": "late-label"},
            ]
        )
        with mock.patch.object(
            speaker_gender,
            "predict_gender",
            side_effect=[
                {"gender": "male", "score": 0.8},
                {"gender": "female", "score": 0.9},
            ],
        ) as predict:
            result = infer_speaker_genders(
                evidence, "/audio.wav", "en", config=config()
            )

        first_intervals = predict.call_args_list[0].kwargs["intervals"]
        second_intervals = predict.call_args_list[1].kwargs["intervals"]
        self.assertEqual(first_intervals, [(0.0, 1.0), (4.0, 4.5)])
        self.assertEqual(second_intervals, [(2.0, 3.0)])
        self.assertEqual(
            [item["clean_speech_duration_sec"] for item in result["speakers"]],
            [1.5, 1.0],
        )
        self.assertTrue(
            all(
                item["input_mode"] == "clean_non_overlap_concatenation"
                for item in result["speakers"]
            )
        )

    def test_multi_speaker_under_one_second_is_null_without_model_call(self):
        evidence = timeline(
            [
                {"start_sec": 0.0, "end_sec": 0.99, "speaker_id": "A"},
                {"start_sec": 2.0, "end_sec": 3.2, "speaker_id": "B"},
            ],
            duration=4.0,
        )
        with mock.patch.object(
            speaker_gender,
            "predict_gender",
            return_value={"gender": "male", "score": 0.8},
        ) as predict:
            result = infer_speaker_genders(
                evidence, "/audio.wav", "sw", config=config()
            )

        self.assertIsNone(result["speakers"][0]["gender"])
        self.assertEqual(
            result["speakers"][0]["reason"], "insufficient_clean_speech"
        )
        self.assertEqual(result["speakers"][1]["gender"], "male")
        predict.assert_called_once()

    def test_single_speaker_passes_full_audio_without_intervals(self):
        evidence = timeline(
            [{"start_sec": 1.0, "end_sec": 2.0, "speaker_id": "only"}]
        )
        with mock.patch.object(
            speaker_gender,
            "predict_gender",
            return_value={"gender": "female", "score": 0.9},
        ) as predict:
            result = infer_speaker_genders(
                evidence, "/audio.wav", "zh", config=config()
            )

        self.assertEqual(result["speakers"][0]["input_mode"], "full_audio")
        self.assertEqual(result["speakers"][0]["gender"], "female")
        self.assertIsNone(predict.call_args.kwargs["intervals"])

    def test_audio_preparation_concatenates_only_requested_regions(self):
        rate = 16000
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            values = []
            for frequency in (100.0, 200.0, 300.0):
                values.extend(
                    0.1 * math.sin(2 * math.pi * frequency * index / rate)
                    for index in range(rate)
                )
            pcm = np.asarray(values, dtype=np.float32)
            with wave.open(str(path), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(rate)
                sink.writeframes(
                    np.asarray(pcm * 32767, dtype="<i2").tobytes()
                )
            prepared = speaker_gender._load_audio(
                path, intervals=[(0.0, 0.5), (2.0, 2.5)]
            )

        self.assertEqual(len(prepared), rate)
        self.assertTrue(np.allclose(prepared[:100], pcm[:100], atol=4e-5))
        self.assertTrue(
            np.allclose(prepared[rate // 2 : rate // 2 + 100], pcm[2 * rate : 2 * rate + 100], atol=4e-5)
        )

    def test_gender_projection_keeps_scores_internal(self):
        profiles = {
            "profiles": [
                {"speaker_id": "speaker_1"},
                {"speaker_id": "speaker_2"},
            ],
            "details": {},
        }
        gender = {
            "schema_version": "v1",
            "status": "observed",
            "selected_model": "alice",
            "speakers": [
                {
                    "speaker_id": "speaker_1",
                    "gender": "female",
                    "score": 0.99,
                },
                {
                    "speaker_id": "speaker_2",
                    "gender": None,
                    "score": None,
                },
            ],
        }

        _merge_profile_genders(profiles, gender)

        self.assertEqual(profiles["profiles"][0]["gender"], "female")
        self.assertIsNone(profiles["profiles"][1]["gender"])
        self.assertNotIn("score", profiles["profiles"][0])


if __name__ == "__main__":
    unittest.main()
