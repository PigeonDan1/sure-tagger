"""Tests for speaker-aware FireRed LID aggregation."""

import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from tagger.pipelines.speaker_evidence import _selected_timeline_output
from tagger.tools.language_content.firered_lid_detector import FireRedLidConfig
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
    aggregate_language_votes,
    normalize_timeline,
    run,
)


class QueueLidClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def detect_language(self, audio_path, _context):
        path = Path(audio_path)
        with wave.open(str(path), "rb") as source:
            duration = source.getnframes() / float(source.getframerate())
            sample_rate_hz = source.getframerate()
        self.calls.append(
            {
                "path": path,
                "duration_sec": duration,
                "sample_rate_hz": sample_rate_hz,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class SpeakerSegmentLidTest(unittest.TestCase):
    def test_config_rejects_nonpositive_duration_thresholds(self):
        with self.assertRaises(ValueError):
            SpeakerSegmentLidConfig(min_speaker_duration_sec=0)
        with self.assertRaises(ValueError):
            SpeakerSegmentLidConfig(min_total_duration_sec=-1)

    def test_selected_speaker_timeline_exposes_activity_and_overlap(self):
        value = _selected_timeline_output(
            {
                "evidence_id": "timeline-1",
                "source": {"name": "diarizen_official"},
                "payload": {
                    "timeline_summary": {
                        "activity_segments": [
                            {
                                "speaker_id": "S1",
                                "start_sec": 1.0,
                                "end_sec": 5.0,
                                "text": "must stay internal",
                            },
                            {
                                "speaker_id": "S2",
                                "start_sec": 4.0,
                                "end_sec": 9.0,
                            },
                        ],
                        "overlap_activity_segments": [
                            {"start_sec": 4.0, "end_sec": 5.0},
                            {"start_sec": 5.5, "end_sec": 8.0},
                        ],
                    }
                },
            },
            duration_sec=6.0,
        )

        self.assertEqual(value["source"], "diarizen_official")
        self.assertEqual(value["evidence_id"], "timeline-1")
        self.assertEqual(
            value["activity_segments"],
            [
                {"speaker_id": "S1", "start_sec": 1.0, "end_sec": 5.0},
                {"speaker_id": "S2", "start_sec": 4.0, "end_sec": 6.0},
            ],
        )
        self.assertEqual(
            value["overlap_activity_segments"],
            [
                {"start_sec": 4.0, "end_sec": 5.0},
                {"start_sec": 5.5, "end_sec": 6.0},
            ],
        )

    def test_normalize_timeline_filters_clamps_and_merges(self):
        timeline = {
            "duration_sec": 8.0,
            "activity_segments": [
                {"speaker_id": "B", "start_sec": 5.0, "end_sec": 9.0},
                {"speaker_id": "A", "start_sec": -1.0, "end_sec": 1.0},
                {"speaker_id": "A", "start_sec": 1.0, "end_sec": 2.0},
                {"speaker_id": "A", "start_sec": 3.0, "end_sec": 3.0},
                {"speaker_id": "", "start_sec": 3.0, "end_sec": 4.0},
                {"speaker_id": "C", "start_sec": math.nan, "end_sec": 4.0},
            ],
        }

        self.assertEqual(
            normalize_timeline(timeline),
            {"A": [(0.0, 2.0)], "B": [(5.0, 8.0)]},
        )

    def test_normalize_timeline_removes_overlap_and_selects_longest_segment(self):
        timeline = {
            "duration_sec": 10.0,
            "activity_segments": [
                {"speaker_id": "A", "start_sec": 0.0, "end_sec": 4.0},
                {"speaker_id": "A", "start_sec": 6.0, "end_sec": 9.0},
                {"speaker_id": "B", "start_sec": 3.0, "end_sec": 7.0},
                {"speaker_id": "B", "start_sec": 8.0, "end_sec": 10.0},
            ],
            "overlap_activity_segments": [
                {"start_sec": 3.0, "end_sec": 4.0},
                {"start_sec": 6.0, "end_sec": 7.0},
                {"start_sec": 8.0, "end_sec": 9.0},
            ],
        }

        normalized, overlaps = normalize_timeline(timeline, return_overlap=True)

        self.assertEqual(normalized, {"A": [(0.0, 3.0)], "B": [(4.0, 6.0)]})
        self.assertEqual(overlaps, [(3.0, 4.0), (6.0, 7.0), (8.0, 9.0)])

    def test_normalize_timeline_derives_overlap_when_not_supplied(self):
        timeline = {
            "duration_sec": 6.0,
            "activity_segments": [
                {"speaker_id": "A", "start_sec": 0.0, "end_sec": 4.0},
                {"speaker_id": "B", "start_sec": 3.0, "end_sec": 6.0},
            ],
        }

        normalized, overlaps = normalize_timeline(timeline, return_overlap=True)

        self.assertEqual(normalized, {"A": [(0.0, 3.0)], "B": [(4.0, 6.0)]})
        self.assertEqual(overlaps, [(3.0, 4.0)])

    def test_one_eligible_speaker_falls_back_without_lid_call(self):
        client = QueueLidClient([{"lang": "en", "confidence": 0.9}])
        result = run(
            "unused.wav",
            {
                "duration_sec": 4.0,
                "activity_segments": [
                    {"speaker_id": "A", "start_sec": 0.0, "end_sec": 4.0}
                ],
            },
            {"language": "de", "confidence": 0.8},
            config=SpeakerSegmentLidConfig(enabled=True),
            lid_config=self._lid_config(),
            client=client,
        )

        self.assertEqual(result.value, "de")
        self.assertEqual(
            result.evidence["fallback_reason"],
            "fewer_than_two_eligible_speakers",
        )
        self.assertEqual(client.calls, [])

    def test_insufficient_total_duration_falls_back_without_lid_call(self):
        client = QueueLidClient([{"lang": "en", "confidence": 0.9}])
        result = run(
            "unused.wav",
            {
                "duration_sec": 3.0,
                "activity_segments": [
                    {"speaker_id": "A", "start_sec": 0.0, "end_sec": 1.2},
                    {"speaker_id": "B", "start_sec": 1.5, "end_sec": 2.7},
                ],
            },
            {"language": "de", "confidence": 0.8},
            config=SpeakerSegmentLidConfig(
                enabled=True,
                min_speaker_duration_sec=1.0,
                min_total_duration_sec=3.0,
            ),
            lid_config=self._lid_config(),
            client=client,
        )

        self.assertEqual(result.value, "de")
        self.assertEqual(
            result.evidence["fallback_reason"],
            "insufficient_total_speaker_audio",
        )
        self.assertEqual(client.calls, [])

    def test_multi_speaker_vote_uses_aggregate_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source.wav"
            write_test_wav(audio_path, duration_sec=8.0)
            client = QueueLidClient(
                [
                    {"lang": "en", "confidence": 0.8},
                    {"lang": "fr", "confidence": 0.95},
                    {"lang": "en", "confidence": 1.0},
                ]
            )
            result = run(
                audio_path,
                {
                    "source": "diarizen_official",
                    "evidence_id": "timeline-1",
                    "duration_sec": 8.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 3.0},
                        {"speaker_id": "B", "start_sec": 3.0, "end_sec": 5.0},
                        {"speaker_id": "C", "start_sec": 5.0, "end_sec": 8.0},
                    ],
                },
                {"language": "mn", "confidence": 0.7},
                context={},
                config=SpeakerSegmentLidConfig(enabled=True),
                lid_config=self._lid_config(),
                client=client,
            )

        self.assertEqual(result.value, "en")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.evidence["decision"], "longest_clean_segment")
        self.assertEqual(
            result.evidence["language_duration_sec"],
            {"en": 6.0, "fr": 2.0},
        )
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(
            all(
                abs(item["duration_sec"] - expected) < 1e-6
                for item, expected in zip(client.calls, (3.0, 2.0, 3.0))
            )
        )
        self.assertTrue(all(not item["path"].exists() for item in client.calls))

    def test_multi_speaker_lid_uses_only_longest_clean_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source.wav"
            write_test_wav(audio_path, duration_sec=10.0)
            client = QueueLidClient(
                [
                    {"lang": "en", "confidence": 0.9},
                    {"lang": "en", "confidence": 0.8},
                ]
            )
            result = run(
                audio_path,
                {
                    "duration_sec": 10.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 3.0},
                        {"speaker_id": "A", "start_sec": 5.0, "end_sec": 7.0},
                        {"speaker_id": "B", "start_sec": 2.0, "end_sec": 5.0},
                        {"speaker_id": "B", "start_sec": 8.0, "end_sec": 10.0},
                    ],
                    "overlap_activity_segments": [
                        {"start_sec": 2.0, "end_sec": 3.0},
                    ],
                },
                {"language": "de", "confidence": 0.7},
                context={},
                config=SpeakerSegmentLidConfig(
                    enabled=True,
                    min_speaker_duration_sec=1.5,
                    min_total_duration_sec=3.0,
                ),
                lid_config=self._lid_config(),
                client=client,
            )

        self.assertEqual(result.value, "en")
        self.assertEqual(
            [item["duration_sec"] for item in client.calls],
            [2.0, 2.0],
        )
        self.assertEqual(result.evidence["speaker_activity_sec"], {"A": 2.0, "B": 2.0})
        self.assertEqual(
            result.evidence["overlap_exclusion"],
            {
                "applied": True,
                "intervals": [{"start_sec": 2.0, "end_sec": 3.0}],
                "policy": "remove_overlap_then_select_longest_speech_segment",
            },
        )

    def test_speaker_audio_is_normalized_to_16khz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source.wav"
            write_test_wav(audio_path, duration_sec=4.0, sample_rate=24000)
            client = QueueLidClient(
                [
                    {"lang": "en", "confidence": 0.9},
                    {"lang": "en", "confidence": 0.8},
                ]
            )
            result = run(
                audio_path,
                {
                    "duration_sec": 4.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 2.0},
                        {"speaker_id": "B", "start_sec": 2.0, "end_sec": 4.0},
                    ],
                },
                {"language": "de", "confidence": 0.7},
                context={},
                config=SpeakerSegmentLidConfig(enabled=True),
                lid_config=self._lid_config(),
                client=client,
            )

        self.assertEqual(result.value, "en")
        self.assertEqual(
            [item["sample_rate_hz"] for item in client.calls],
            [16000, 16000],
        )

    def test_duration_tie_falls_back_to_full_audio(self):
        vote = aggregate_language_votes(
            [
                {"language": "en", "confidence": 0.9, "activity_sec": 2.0},
                {"language": "fr", "confidence": 0.8, "activity_sec": 2.0},
            ]
        )

        self.assertTrue(vote["tie"])
        self.assertIsNone(vote["language"])

    def test_speaker_lid_failure_falls_back_to_full_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source.wav"
            write_test_wav(audio_path, duration_sec=5.0)
            client = QueueLidClient(
                [
                    {"lang": "en", "confidence": 0.9},
                    RuntimeError("lid failed"),
                ]
            )
            result = run(
                audio_path,
                {
                    "duration_sec": 5.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 2.5},
                        {"speaker_id": "B", "start_sec": 2.5, "end_sec": 5.0},
                    ],
                },
                {"language": "de", "confidence": 0.75},
                context={},
                config=SpeakerSegmentLidConfig(enabled=True),
                lid_config=self._lid_config(),
                client=client,
            )

        self.assertEqual(result.value, "de")
        self.assertEqual(result.evidence["fallback_reason"], "speaker_lid_failed")
        self.assertEqual(len(client.calls), 2)

    def test_shadow_mode_records_proposal_but_publishes_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "source.wav"
            write_test_wav(audio_path, duration_sec=4.0)
            result = run(
                audio_path,
                {
                    "duration_sec": 4.0,
                    "activity_segments": [
                        {"speaker_id": "A", "start_sec": 0.0, "end_sec": 2.0},
                        {"speaker_id": "B", "start_sec": 2.0, "end_sec": 4.0},
                    ],
                },
                {"language": "de", "confidence": 0.7},
                context={},
                config=SpeakerSegmentLidConfig(enabled=True, shadow_only=True),
                lid_config=self._lid_config(),
                client=QueueLidClient(
                    [
                        {"lang": "en", "confidence": 0.9},
                        {"lang": "en", "confidence": 0.8},
                    ]
                ),
            )

        self.assertEqual(result.value, "de")
        self.assertEqual(result.evidence["proposed_language"], "en")
        self.assertEqual(result.evidence["decision"], "shadow_full_audio")

    @staticmethod
    def _lid_config():
        return FireRedLidConfig(
            model_dir="unused-model",
            repo_dir="unused-repo",
            subprocess_python="",
        )


def write_test_wav(path, duration_sec, sample_rate=16000):
    frame_count = int(duration_sec * sample_rate)
    frames = struct.pack("<" + "h" * frame_count, *([0] * frame_count))
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(frames)


if __name__ == "__main__":
    unittest.main()
