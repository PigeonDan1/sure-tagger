"""Tests for the FireRed LID language-content tool."""

from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import wave

from tagger.tools.language_content.firered_lid_detector import (
    FireRedLidClient,
    FireRedLidConfig,
    FireRedLidError,
    run,
    validate_lid_output,
)


class FireRedLidToolTest(unittest.TestCase):
    def test_client_converts_stereo_audio_to_16k_mono_and_cleans_up(self):
        class InspectingLid:
            def process(self, uttids, paths):
                self.path = Path(paths[0])
                self.uttids = uttids
                with wave.open(str(self.path), "rb") as audio:
                    self.channels = audio.getnchannels()
                    self.sample_rate = audio.getframerate()
                    self.sample_width = audio.getsampwidth()
                return [{"lang": "en", "confidence": 0.95}]

        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "stereo.wav"
            with wave.open(str(source_path), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(
                    b"".join(struct.pack("<hh", 1000, -1000) for _ in range(800))
                )
            lid = InspectingLid()
            client = FireRedLidClient()
            with mock.patch.object(client, "_get_lid", return_value=lid):
                result = client.detect_language(source_path)

            self.assertEqual(result["lang"], "en")
            self.assertEqual(lid.uttids, ["sample"])
            self.assertEqual((lid.channels, lid.sample_rate, lid.sample_width), (1, 16000, 2))
            self.assertNotEqual(lid.path, source_path)
            self.assertFalse(lid.path.exists())

    def test_client_cleans_prepared_audio_when_lid_fails(self):
        class FailingLid:
            def process(self, _uttids, paths):
                self.path = Path(paths[0])
                raise RuntimeError("model failed")

        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "stereo.wav"
            with wave.open(str(source_path), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(struct.pack("<hh", 1000, -1000) * 1600)
            lid = FailingLid()
            client = FireRedLidClient()
            with mock.patch.object(client, "_get_lid", return_value=lid):
                with self.assertRaisesRegex(FireRedLidError, "inference failed"):
                    client.detect_language(source_path)

            self.assertFalse(lid.path.exists())

    def test_run_reuses_one_audio_result_for_the_same_config(self):
        class CountingClient:
            calls = 0

            def detect_language(self, _audio_path, _context):
                self.calls += 1
                return {"lang": "en", "confidence": 0.95}

        client = CountingClient()
        config = FireRedLidConfig(
            model_dir="lid-model",
            repo_dir="lid-repo",
            subprocess_python="lid-python",
        )
        context = {}

        first = run("audio.wav", config=config, context=context, client=client)
        second = run("audio.wav", config=config, context=context, client=client)

        self.assertEqual(client.calls, 1)
        self.assertEqual(first.value, "en")
        self.assertEqual(second.value, "en")

    def test_validate_accepts_upstream_output(self):
        summary = validate_lid_output(
            {
                "uttid": "sample",
                "lang": "en",
                "confidence": 0.996,
                "dur_s": 18.436,
                "rtf": "0.0860",  # upstream formats rtf as a display string
                "wav": "audio.wav",
            }
        )
        self.assertEqual(summary["lang"], "en")
        self.assertEqual(summary["confidence"], 0.996)
        self.assertEqual(summary["rtf"], 0.086)

    def test_validate_accepts_region_code(self):
        summary = validate_lid_output({"lang": "zh-xinan", "confidence": 0.9})
        self.assertEqual(summary["lang"], "zh-xinan")

    def test_validate_canonicalizes_upstream_two_token_chinese_code(self):
        summary = validate_lid_output(
            {"lang": "zh mandarin", "confidence": 1.0}
        )
        self.assertEqual(summary["lang"], "zh-mandarin")
        self.assertEqual(summary["raw_lang"], "zh mandarin")

    def test_validate_rejects_empty_lang(self):
        with self.assertRaises(FireRedLidError):
            validate_lid_output({"lang": "", "confidence": 0.9})

    def test_validate_rejects_out_of_range_confidence(self):
        with self.assertRaises(FireRedLidError):
            validate_lid_output({"lang": "en", "confidence": 1.5})

    def test_validate_rejects_non_numeric_dur(self):
        with self.assertRaises(FireRedLidError):
            validate_lid_output({"lang": "en", "confidence": 0.9, "dur_s": "long"})


if __name__ == "__main__":
    unittest.main()
