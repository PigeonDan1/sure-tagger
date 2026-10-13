import math
import struct
import subprocess
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

from tagger.tools.speaker_v2.contracts import build_evidence
from tagger.tools.speaker_v2 import speaker_profile
from tagger.tools.speaker_v2.speaker_profile import compute_speaker_profiles
from tagger.tools.speaker_v2.timeline import summarize_timeline


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


def coverage(segments, duration=6.0):
    return build_evidence(
        "sample",
        duration,
        "speech_coverage",
        "firered_vad",
        "v1",
        "vad",
        ["speech_coverage"],
        ["G_vad"],
        {"speech_segments": segments},
        quality={"usable": True},
    )


def write_sine_wav(path, blocks, rate=16000, frequency=220.0):
    """Write a mono 16-bit PCM WAV from (seconds, amplitude) sine blocks."""
    frames = []
    for seconds, amplitude in blocks:
        count = int(seconds * rate)
        frames.extend(
            int(amplitude * 32767 * math.sin(2.0 * math.pi * frequency * index / rate))
            for index in range(count)
        )
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(rate)
        sink.writeframes(struct.pack("<%dh" % len(frames), *frames))


class SpeakerProfileTest(unittest.TestCase):
    def test_cjk_rate_uses_character_per_second(self):
        evidence = timeline(
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 6.0,
                    "speaker_id": "S01",
                    "text": "这是一个用于测试语速的中文句子这是一个用于测试语速的中文句子",
                }
            ]
        )
        result = compute_speaker_profiles(evidence)
        rate = result["profiles"][0]["speech_rate"]
        self.assertEqual(rate["unit"], "zh_char_per_sec")
        self.assertEqual(rate["band"], "normal")
        self.assertGreater(rate["value"], 0)

    def test_latin_rate_uses_words_per_minute(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = compute_speaker_profiles(
            timeline([{"start_sec": 0.0, "end_sec": 6.0, "speaker_id": "S01", "text": text}])
        )
        rate = result["profiles"][0]["speech_rate"]
        self.assertEqual(rate["unit"], "word_per_min")
        self.assertEqual(rate["value"], 120.0)

    def test_overlap_is_excluded_from_rate_duration(self):
        evidence = timeline(
            [
                {"start_sec": 0.0, "end_sec": 6.0, "speaker_id": "S01", "text": "这是一个用于测试语速的中文句子"},
                {"start_sec": 2.0, "end_sec": 4.0, "speaker_id": "S02", "text": "这是另一个说话人的重叠片段"},
            ]
        )
        result = compute_speaker_profiles(evidence)
        self.assertEqual(result["details"]["speakers"][0]["clean_speech_duration_sec"], 4.0)
        self.assertEqual(result["details"]["speakers"][1]["clean_speech_duration_sec"], 0.0)

    def test_short_text_abstains_and_missing_timeline_is_null(self):
        short = compute_speaker_profiles(
            timeline([{"start_sec": 0.0, "end_sec": 6.0, "speaker_id": "S01", "text": "hello"}])
        )
        self.assertIsNone(short["profiles"][0]["speech_rate"]["value"])
        self.assertEqual(compute_speaker_profiles(None)["profiles"], None)

    def test_valid_timeline_with_no_vad_speech_returns_empty(self):
        result = compute_speaker_profiles(
            timeline(
                [
                    {
                        "start_sec": 0.0,
                        "end_sec": 6.0,
                        "speaker_id": "S01",
                        "text": "hello world this is text",
                    }
                ]
            ),
            coverage([], 6.0),
        )
        self.assertEqual(result["profiles"], [])

    def test_speaker_volume_variable_band_for_large_within_speaker_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "variable.wav"
            write_sine_wav(audio, [(0.5, 0.002), (0.5, 0.008)])
            result = compute_speaker_profiles(
                timeline(
                    [
                        {
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "speaker_id": "S01",
                            "text": "",
                        }
                    ],
                    duration=1.0,
                ),
                audio_path=str(audio),
                sample_rate_hz=16000,
            )
        self.assertEqual(result["profiles"][0]["speaker_volume"], "variable")

    def test_speaker_volume_normal_band_for_constant_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "constant.wav"
            write_sine_wav(audio, [(1.0, 0.004)])
            result = compute_speaker_profiles(
                timeline(
                    [
                        {
                            "start_sec": 0.0,
                            "end_sec": 1.0,
                            "speaker_id": "S01",
                            "text": "",
                        }
                    ],
                    duration=1.0,
                ),
                audio_path=str(audio),
                sample_rate_hz=16000,
            )
        self.assertEqual(result["profiles"][0]["speaker_volume"], "normal")

    def test_speaker_volume_relative_bands_compare_within_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "relative.wav"
            write_sine_wav(audio, [(0.8, 0.002), (0.8, 0.008)])
            result = compute_speaker_profiles(
                timeline(
                    [
                        {
                            "start_sec": 0.0,
                            "end_sec": 0.8,
                            "speaker_id": "S01",
                            "text": "",
                        },
                        {
                            "start_sec": 0.8,
                            "end_sec": 1.6,
                            "speaker_id": "S02",
                            "text": "",
                        },
                    ],
                    duration=1.6,
                ),
                audio_path=str(audio),
                sample_rate_hz=16000,
            )
        volumes = {
            profile["speaker_id"]: profile["speaker_volume"]
            for profile in result["profiles"]
        }
        self.assertEqual(volumes, {"speaker_1": "low", "speaker_2": "loud"})

    def test_mp3_fallback_decodes_mono_float_samples(self):
        decoded = array("f", [0.25]) * 16000
        completed = subprocess.CompletedProcess(
            ["ffmpeg"], 0, stdout=decoded.tobytes(), stderr=b""
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"not a wav")
            audio.flush()
            with mock.patch.object(speaker_profile.shutil, "which", return_value="/usr/bin/ffmpeg"):
                with mock.patch.object(
                    speaker_profile.subprocess, "run", return_value=completed
                ) as run:
                    samples, rate = speaker_profile._read_interval_samples(
                        audio.name, [(0.25, 0.75)], sample_rate_hz=16000
                    )
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 8000)
        self.assertAlmostEqual(samples[0], 0.25, places=5)
        command = run.call_args[0][0]
        self.assertEqual(command[0], "/usr/bin/ffmpeg")
        self.assertIn("-f", command)
        self.assertEqual(command[-1], "pipe:1")


if __name__ == "__main__":
    unittest.main()
