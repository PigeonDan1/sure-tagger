"""Tests for the standalone-LID speaker ASR routing policy."""

import unittest

from tagger.pipelines.speaker_evidence import (
    _classify_asr_language,
    _language_route_from_evidence,
    _select_speaker_asr,
)


def _candidate(
    source, text, status="estimated", usable=True, speaker_count=None
):
    payload = {"text": text, "asr_transcript": text}
    if speaker_count is not None:
        payload["timeline_summary"] = {
            "observed_speaker_count": speaker_count,
            "segments": [],
        }
    return {
        "source": {"name": source},
        "status": status,
        "quality": {"usable": usable},
        "payload": payload,
    }


def _lid(language=None, confidence=0.9, error=None):
    if language is None:
        return {
            "evidence_id": "ev_lid_missing",
            "source": {"name": "firered_lid"},
            "status": "missing",
            "quality": {"usable": False, "reason": error or "LID failed"},
            "payload": {},
        }
    return {
        "evidence_id": "ev_lid",
        "source": {"name": "firered_lid"},
        "status": "estimated",
        "quality": {"usable": True},
        "payload": {
            "language": language,
            "language_confidence": confidence,
        },
    }


class SpeakerV2AsrRoutingTest(unittest.TestCase):
    def test_both_single_speaker_counts_select_firered_for_chinese(self):
        result = _select_speaker_asr(
            [
                _lid("zh"),
                _candidate(
                    "moss_transcribe_diarize",
                    "MOSS text",
                    speaker_count=1,
                ),
                _candidate(
                    "diarizen_large_s80_v2_official",
                    "",
                    speaker_count=1,
                ),
                _candidate("fireredasr2_aed", "FireRed text"),
            ]
        )

        self.assertEqual(result["text"], "FireRed text")
        self.assertEqual(result["route"]["selected_model"], "firered")
        self.assertEqual(
            result["route"]["selection_basis"],
            "single_speaker_consensus",
        )
        self.assertTrue(result["route"]["consensus_override_applied"])

    def test_either_multi_speaker_count_keeps_original_route(self):
        result = _select_speaker_asr(
            [
                _lid("zh"),
                _candidate(
                    "moss_transcribe_diarize",
                    "MOSS text",
                    speaker_count=1,
                ),
                _candidate(
                    "diarizen_large_s80_v2_official",
                    "",
                    speaker_count=2,
                ),
                _candidate("fireredasr2_aed", "FireRed text"),
            ]
        )

        self.assertEqual(result["text"], "MOSS text")
        self.assertEqual(result["route"]["selected_model"], "moss")
        self.assertEqual(result["route"]["selection_basis"], "language_route")
        self.assertFalse(result["route"]["consensus_override_applied"])

    def test_missing_count_does_not_form_single_speaker_consensus(self):
        result = _select_speaker_asr(
            [
                _lid("en"),
                _candidate(
                    "moss_transcribe_diarize",
                    "MOSS text",
                    speaker_count=1,
                ),
                _candidate(
                    "diarizen_large_s80_v2_official",
                    "",
                    status="missing",
                    usable=False,
                ),
                _candidate("fireredasr2_aed", "FireRed text"),
            ]
        )

        self.assertEqual(result["text"], "MOSS text")
        self.assertFalse(
            result["route"]["single_speaker_consensus"]["all_at_most_one"]
        )

    def test_single_speaker_consensus_does_not_fallback_when_firered_fails(self):
        result = _select_speaker_asr(
            [
                _lid("zh"),
                _candidate(
                    "moss_transcribe_diarize",
                    "MOSS text",
                    speaker_count=1,
                ),
                _candidate(
                    "diarizen_large_s80_v2_official",
                    "",
                    speaker_count=0,
                ),
                _candidate(
                    "fireredasr2_aed",
                    "",
                    status="missing",
                    usable=False,
                ),
            ]
        )

        self.assertEqual(result["text"], "")
        self.assertIsNone(result["route"]["selected_source"])
        self.assertEqual(result["route"]["required_model"], "firered")
        self.assertTrue(result["route"]["failed_closed"])
        self.assertFalse(result["route"]["fallback"])

    def test_chinese_lid_aliases_select_moss(self):
        for language in ("zh", "zh-cn", "zh_CN", "zh mandarin"):
            with self.subTest(language=language):
                evidence = [
                    _lid(language),
                    _candidate("moss_transcribe_diarize", "MOSS 中文"),
                    _candidate("fireredasr2_aed", "ineligible FireRed"),
                ]
                result = _select_speaker_asr(evidence)

                self.assertEqual(result["text"], "MOSS 中文")
                self.assertEqual(result["route"]["language_route"], "chinese")
                self.assertEqual(result["route"]["selected_model"], "moss")
                self.assertEqual(
                    result["route"]["selected_source"],
                    "moss_transcribe_diarize",
                )
                self.assertFalse(result["route"]["fallback"])

    def test_english_lid_aliases_select_moss(self):
        for language in ("en", "en-us", "en_US", "en english", "english"):
            with self.subTest(language=language):
                evidence = [
                    _lid(language),
                    _candidate("moss_transcribe_diarize", "MOSS English"),
                    _candidate("fireredasr2_aed", "ineligible FireRed"),
                ]
                result = _select_speaker_asr(evidence)

                self.assertEqual(result["text"], "MOSS English")
                self.assertEqual(result["route"]["language_route"], "english")
                self.assertEqual(result["route"]["selected_model"], "moss")
                self.assertFalse(result["route"]["fallback"])

    def test_other_and_mixed_lid_select_qwen3(self):
        for language in ("fr", "ru", "mul", "mixed", "en zh"):
            with self.subTest(language=language):
                evidence = [
                    _lid(language),
                    _candidate("moss_transcribe_diarize", "ineligible MOSS"),
                    _candidate("fireredasr2_aed", "ineligible FireRed"),
                    _candidate("qwen3_asr_0_6b", "Qwen3 text"),
                ]
                result = _select_speaker_asr(evidence)

                self.assertEqual(result["text"], "Qwen3 text")
                self.assertEqual(
                    result["route"]["language_route"],
                    "non_chinese_or_unknown",
                )
                self.assertEqual(
                    result["route"]["selected_model"], "qwen3_asr"
                )
                self.assertFalse(result["route"]["fallback"])

    def test_missing_lid_routes_unknown_to_qwen3(self):
        evidence = [
            _lid(error="standalone LID unavailable"),
            _candidate("moss_transcribe_diarize", "ineligible MOSS"),
            _candidate("fireredasr2_aed", "ineligible FireRed"),
            _candidate("qwen3_asr_0_6b", "unknown-language text"),
        ]
        result = _select_speaker_asr(evidence)

        self.assertEqual(result["text"], "unknown-language text")
        self.assertEqual(
            result["route"]["language_route"],
            "non_chinese_or_unknown",
        )
        self.assertEqual(
            result["route"]["language_source"], "firered_lid_unavailable"
        )
        self.assertEqual(
            result["route"]["language_error"], "standalone LID unavailable"
        )
        self.assertEqual(result["route"]["selected_model"], "qwen3_asr")

    def test_chinese_moss_failure_does_not_fallback_to_firered(self):
        result = _select_speaker_asr(
            [
                _lid("zh"),
                _candidate(
                    "moss_transcribe_diarize",
                    "",
                    status="missing",
                    usable=False,
                ),
                _candidate("fireredasr2_aed", "must not cross route"),
            ]
        )

        self.assertEqual(result["text"], "")
        self.assertIsNone(result["route"]["selected_source"])
        self.assertEqual(result["route"]["required_model"], "moss")
        self.assertTrue(result["route"]["failed_closed"])
        self.assertFalse(result["route"]["fallback"])

    def test_non_chinese_qwen3_failure_does_not_fallback(self):
        result = _select_speaker_asr(
            [
                _lid("fr"),
                _candidate("moss_transcribe_diarize", "must not cross route"),
                _candidate(
                    "fireredasr2_aed",
                    "must not cross route",
                ),
                _candidate(
                    "qwen3_asr_0_6b", "", status="missing", usable=False
                ),
            ]
        )

        self.assertEqual(result["text"], "")
        self.assertIsNone(result["route"]["selected_source"])
        self.assertEqual(result["route"]["required_model"], "qwen3_asr")
        self.assertTrue(result["route"]["failed_closed"])
        self.assertFalse(result["route"]["fallback"])

    def test_route_uses_lid_only_not_transcript_script(self):
        route = _language_route_from_evidence([_lid("en")])
        result = _select_speaker_asr(
            [
                _lid("en"),
                _candidate("moss_transcribe_diarize", "MOSS"),
                _candidate("fireredasr2_aed", "这是转写文本"),
            ],
            language_route=route,
        )

        self.assertEqual(result["text"], "MOSS")
        self.assertEqual(result["route"]["language_source"], "firered_lid")
        self.assertEqual(result["route"]["detected_language"], "en")

    def test_classifier_has_moss_and_official_routes(self):
        chinese = _classify_asr_language(language="zh mandarin")
        english = _classify_asr_language(language="en")
        french = _classify_asr_language(language="fr")
        unknown = _classify_asr_language(language=None)

        self.assertEqual(chinese[0], "chinese")
        self.assertEqual(english[0], "english")
        self.assertEqual(french[0], "non_chinese_or_unknown")
        self.assertEqual(unknown[0], "non_chinese_or_unknown")
        self.assertEqual(unknown[2], "firered_lid_unavailable")


if __name__ == "__main__":
    unittest.main()
