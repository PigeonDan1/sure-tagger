import copy
import json
import sys
import tempfile
from pathlib import Path
import unittest


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import merge_sure_tagger_descriptive as merge


class ProfileGenderMergeTest(unittest.TestCase):
    def test_restores_gender_only_for_matching_descriptions(self):
        tags = {
            "speaker": {
                "profiles": [
                    {"speaker_id": "speaker_1", "gender": "female"},
                    {"speaker_id": "speaker_2", "gender": None},
                ]
            }
        }
        descriptive = {
            "speaker_descriptions": [
                {"speaker_id": "speaker_1", "gender": None, "description": "fast"}
            ]
        }
        restored = merge._restore_profile_genders(tags, descriptive)
        self.assertEqual(
            restored["speaker_descriptions"][0]["gender"], "female"
        )
        self.assertIsNone(descriptive["speaker_descriptions"][0]["gender"])


class StructuralMergeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.structural_path = self.tmp / "structural.jsonl"
        self.deepseek_path = self.tmp / "results.jsonl"
        self.output_dir = self.tmp / "merged"
        self.tags = {group: {} for group in merge.STRUCTURAL_GROUPS}
        self.tags["speaker"] = {
            "speaker_count": 1,
            "speaker_present": True,
            "asr_transcript": "Hello there.",
            "multi_speaker": False,
            "speaker_change_count": 0,
            "speaker_change": False,
            "overlap_ratio": 0.0,
            "speaker_overlap": False,
            "profiles": [{"speaker_id": "speaker_1", "gender": "female"}],
        }
        self.tags["language_content"] = {
            "language": "en", "punctuation": None, "repetition": None,
        }
        self.tags["deterministic_derived"] = {
            "speaker_change_rate_per_min": 0.0,
            "filler_ratio": 0.0,
        }
        self._write(self.structural_path, {"sample_id": "s1", "tags": self.tags})

    def _write(self, path, row):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def _candidate(self):
        candidate = {
            group: merge._canonical_structural(copy.deepcopy(self.tags[group]), group)
            for group in merge.STRUCTURAL_GROUPS
        }
        candidate["sample_id"] = "s1"
        candidate["descriptive"] = {key: None for key in merge.DESCRIPTIVE_KEYS}
        candidate["descriptive"]["speaker_descriptions"] = [
            {"speaker_id": "speaker_1", "gender": "male", "description": "Steady speech."}
        ]
        return candidate

    def test_speaker_and_derived_changes_are_rejected_even_with_old_sidecar(self):
        self._write(
            self.tmp / "speaker_correction.jsonl",
            {"sample_id": "s1", "changed": True},
        )
        changes = (
            ("speaker", "speaker_count", 2),
            ("speaker", "profiles", []),
            ("deterministic_derived", "speaker_change_rate_per_min", 10.0),
        )
        for group, field, value in changes:
            with self.subTest(group=group, field=field):
                candidate = self._candidate()
                candidate[group][field] = value
                self._write(self.deepseek_path, candidate)
                with self.assertRaisesRegex(ValueError, "structural field changed.*" + group):
                    merge.merge(self.structural_path, self.deepseek_path, self.output_dir)
                self.assertFalse(self.output_dir.exists())

    def test_source_structure_and_gender_are_preserved_without_reading_old_sidecar(self):
        (self.tmp / "speaker_correction.jsonl").write_text("invalid JSON\n", encoding="utf-8")
        self._write(self.deepseek_path, self._candidate())
        merge.merge(self.structural_path, self.deepseek_path, self.output_dir)
        published = merge.read_jsonl(self.output_dir / "result_with_id_descriptive.jsonl")[0]
        for group in merge.STRUCTURAL_GROUPS:
            self.assertEqual(published[group], self.tags[group], group)
        self.assertEqual(published["descriptive"]["speaker_descriptions"][0]["gender"], "female")


if __name__ == "__main__":
    unittest.main()
