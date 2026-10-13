"""The one sanctioned structural override must stay gated.

The merge step is the anti-tamper boundary between the DeepSeek batch and the
published result: everything except ``language_content.language`` has to come
through byte-identical to the pipeline's own tags.  These tests pin both halves
of that contract -- the exception works, and nothing else sneaks through with
it.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import merge_sure_tagger_descriptive as merge


def _structural(language="cy", word_count=12):
    return {
        "basic_acoustic": {
            "duration_sec": 3.0,
            "sample_rate_hz": 16000,
            "channels": 1,
            "nonspeech_ratio": 0.1,
            "silence_segments": None,
        },
        "audio_quality": {
            "snr_db": None,
            "dnsmos_sig": None,
            "dnsmos_bak": None,
            "dnsmos_ovrl": None,
            "dnsmos_p808": None,
        },
        "room_acoustic": {"far_field": None, "rt60_sec": None, "c50_db": None},
        "sound_field_scene": {
            "speech_music_events": None,
            "music_present": False,
            "external_noise_type": None,
            "noise_composition": None,
        },
        "speaker": {
            "speaker_count": 1,
            "speaker_present": True,
            "asr_transcript": "hello there this is a test",
            "multi_speaker": False,
            "speaker_change_count": 0,
            "speaker_change": False,
            "overlap_ratio": 0.0,
            "speaker_overlap": False,
            "profiles": [],
        },
        "language_content": {
            "language": language,
            "word_count": word_count,
            "punctuation": None,
            "repetition": None,
            "filler": None,
        },
        "deterministic_derived": {
            "speaker_change_rate_per_min": None,
            "filler_ratio": None,
        },
    }


def _descriptive():
    return {
        "caption": None,
        "topic": None,
        "background_condition": None,
        "dominant_content_impression": None,
        "clarity_intelligibility_impression": None,
        "background_sound_summary": None,
        "descriptive_phrases": None,
        "facets": None,
        "speaker_descriptions": None,
        "scene_hypotheses": None,
        "usability_notes": None,
        "uncertainties": None,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class LanguageOverrideMergeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.structural_path = self.tmp / "result_with_id.jsonl"
        self.deepseek_path = self.tmp / "results.jsonl"
        self.verification_path = self.tmp / "language_verification.jsonl"
        self.out = self.tmp / "out"
        _write_jsonl(
            self.structural_path,
            [{"sample_id": "s1", "tags": _structural()}],
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _candidate(self, language, word_count=12):
        # Mirror the batch runner: the DeepSeek-side structural groups are the
        # canonicalized ones (null punctuation/repetition become null-objects),
        # so a faithful candidate differs from the source in nothing but the
        # field under test.
        source = _structural(language=language, word_count=word_count)
        row = {
            group: merge._canonical_structural(source[group], group)
            for group in merge.STRUCTURAL_GROUPS
        }
        row["sample_id"] = "s1"
        row["descriptive"] = _descriptive()
        return row

    def _write_sidecar(self, changed=True, published="en", lid="cy"):
        _write_jsonl(
            self.verification_path,
            [
                {
                    "sample_id": "s1",
                    "lid_language": lid,
                    "published_language": published,
                    "changed": changed,
                    "verdict_state": "changed" if changed else "agreed",
                }
            ],
        )

    def test_sanctioned_language_change_is_applied_and_is_the_only_diff(self):
        _write_jsonl(self.deepseek_path, [self._candidate("en")])
        self._write_sidecar()
        merge.merge(
            self.structural_path,
            self.deepseek_path,
            self.out,
            verification_path=self.verification_path,
        )
        published = json.loads(
            (self.out / "result_with_id_descriptive.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(published["language_content"]["language"], "en")
        source = _structural()
        for group in merge.STRUCTURAL_GROUPS:
            if group == "language_content":
                continue
            self.assertEqual(published[group], source[group], group)
        differing = [
            key
            for key in published["language_content"]
            if published["language_content"][key] != source["language_content"][key]
        ]
        self.assertEqual(differing, ["language"])

    def test_language_change_without_a_sidecar_is_rejected(self):
        _write_jsonl(self.deepseek_path, [self._candidate("en")])
        with self.assertRaises(ValueError) as caught:
            merge.merge(self.structural_path, self.deepseek_path, self.out)
        self.assertIn("without an authorizing verification record", str(caught.exception))

    def test_sidecar_that_did_not_authorize_the_change_is_rejected(self):
        _write_jsonl(self.deepseek_path, [self._candidate("en")])
        self._write_sidecar(changed=False, published="en", lid="cy")
        with self.assertRaises(ValueError):
            merge.merge(
                self.structural_path,
                self.deepseek_path,
                self.out,
                verification_path=self.verification_path,
            )

    def test_sidecar_disagreeing_with_the_result_is_rejected(self):
        _write_jsonl(self.deepseek_path, [self._candidate("fr")])
        self._write_sidecar(published="en", lid="cy")
        with self.assertRaises(ValueError) as caught:
            merge.merge(
                self.structural_path,
                self.deepseek_path,
                self.out,
                verification_path=self.verification_path,
            )
        self.assertIn("disagrees with the DeepSeek result", str(caught.exception))

    def test_sidecar_disagreeing_with_the_source_lid_is_rejected(self):
        _write_jsonl(self.deepseek_path, [self._candidate("en")])
        self._write_sidecar(published="en", lid="mi")
        with self.assertRaises(ValueError) as caught:
            merge.merge(
                self.structural_path,
                self.deepseek_path,
                self.out,
                verification_path=self.verification_path,
            )
        self.assertIn("disagrees with the source LID value", str(caught.exception))

    def test_other_group_changes_still_raise_even_with_a_sidecar(self):
        _write_jsonl(self.deepseek_path, [self._candidate("cy", word_count=99)])
        self._write_sidecar(changed=False, published="cy", lid="cy")
        with self.assertRaises(ValueError) as caught:
            merge.merge(
                self.structural_path,
                self.deepseek_path,
                self.out,
                verification_path=self.verification_path,
            )
        self.assertIn("language_content", str(caught.exception))

    def test_unchanged_language_needs_no_sidecar(self):
        _write_jsonl(self.deepseek_path, [self._candidate("cy")])
        summary = merge.merge(self.structural_path, self.deepseek_path, self.out)
        self.assertEqual(summary["structural_overrides"], {"language_content.language": 0})
        self.assertIsNone(summary["language_verification"])

    def test_summary_counts_the_override(self):
        _write_jsonl(self.deepseek_path, [self._candidate("en")])
        self._write_sidecar()
        summary = merge.merge(
            self.structural_path,
            self.deepseek_path,
            self.out,
            verification_path=self.verification_path,
        )
        self.assertEqual(summary["structural_overrides"], {"language_content.language": 1})
        self.assertEqual(summary["language_override_counts"], {"cy -> en": 1})
        self.assertTrue(summary["language_verification_sha256"])


class SplitOutputTest(unittest.TestCase):
    def test_wrapped_output_yields_descriptive_and_verdict(self):
        import extract_sure_tagger_deepseek_once as sure_once

        descriptive = _descriptive()
        descriptive["caption"] = "a caption"
        value = {
            "language_verification": {
                "language": " EN ",
                "confidence": 1.5,
                "reason_short": "english function words",
            },
            "descriptive": descriptive,
        }
        parsed, verification = sure_once.split_output(copy.deepcopy(value))
        self.assertEqual(parsed["caption"], "a caption")
        self.assertEqual(verification["language"], "en")
        self.assertEqual(verification["confidence"], 1.0)

    def test_bare_descriptive_output_has_no_verdict(self):
        import extract_sure_tagger_deepseek_once as sure_once

        parsed, verification = sure_once.split_output(_descriptive())
        self.assertIsNone(verification)
        self.assertIsNone(parsed["caption"])

    def test_malformed_verdict_degrades_to_abstention(self):
        import extract_sure_tagger_deepseek_once as sure_once

        parsed, verification = sure_once.split_output(
            {"language_verification": "en", "descriptive": _descriptive()}
        )
        self.assertIsNone(verification)
        self.assertTrue(parsed)

    def test_descriptive_keys_are_still_strict(self):
        import extract_sure_tagger_deepseek_once as sure_once

        broken = _descriptive()
        broken["extra"] = 1
        with self.assertRaises(Exception):
            sure_once.split_output({"descriptive": broken})


class ResolveLanguageTest(unittest.TestCase):
    def _structural_with(self, language, transcript):
        value = _structural(language=language)
        value["speaker"]["asr_transcript"] = transcript
        return value

    def test_verdict_overrides_when_it_disagrees(self):
        import extract_sure_tagger_deepseek_once as sure_once

        structural = self._structural_with("cy", "hello there this is a test")
        published, state, changed, verified = sure_once.resolve_language(
            structural, {"language": "en"}
        )
        self.assertEqual((published, state, changed, verified), ("en", "changed", True, "en"))

    def test_agreement_is_not_a_change(self):
        import extract_sure_tagger_deepseek_once as sure_once

        structural = self._structural_with("en", "hello there this is a test")
        published, state, changed, _ = sure_once.resolve_language(structural, {"language": "en"})
        self.assertEqual((published, state, changed), ("en", "agreed", False))

    def test_null_verdict_keeps_the_lid_value(self):
        import extract_sure_tagger_deepseek_once as sure_once

        structural = self._structural_with("cy", "hello there this is a test")
        published, state, changed, _ = sure_once.resolve_language(structural, {"language": None})
        self.assertEqual((published, state, changed), ("cy", "abstained", False))

    def test_missing_verdict_keeps_the_lid_value(self):
        import extract_sure_tagger_deepseek_once as sure_once

        structural = self._structural_with("cy", "hello there this is a test")
        published, state, changed, _ = sure_once.resolve_language(structural, None)
        self.assertEqual((published, state, changed), ("cy", "absent", False))

    def test_empty_transcript_forces_abstention_in_code(self):
        import extract_sure_tagger_deepseek_once as sure_once

        for transcript in (None, "", "   "):
            structural = self._structural_with("cy", transcript)
            published, state, changed, _ = sure_once.resolve_language(
                structural, {"language": "en"}
            )
            self.assertEqual(published, "cy", transcript)
            self.assertEqual(state, "no_transcript", transcript)
            self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
