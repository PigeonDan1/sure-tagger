from pathlib import Path
import sys
import unittest


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_qwen3_5_omni_plus_descriptive_gt_official_assisted_demo as assisted


class AssistedSanitizerTest(unittest.TestCase):
    def test_removes_demographic_wording_recursively(self):
        payload = {
            "descriptive": {
                "caption": "A male voice reads clearly.",
                "speaker_descriptions": [
                    {
                        "speaker_id": "speaker_1",
                        "description": "The narrator sounds male.",
                        "descriptive_phrases": ["male voice"],
                        "evidence_tags": ["speaker.asr_transcript"],
                        "confidence": 0.8,
                    }
                ],
            }
        }
        sanitized, changes = assisted.sanitize_assisted_descriptive(
            payload, {"speaker.asr_transcript": "words"}
        )
        text = str(sanitized).lower()
        self.assertNotIn("male voice", text)
        self.assertNotIn("sounds male", text)
        self.assertEqual(sanitized["descriptive"]["caption"], "A voice reads clearly.")
        self.assertEqual(len(changes), 3)

    def test_filters_only_unavailable_and_duplicate_evidence_tags(self):
        payload = {
            "descriptive": {
                "facets": [
                    {
                        "name": "room",
                        "description": "quiet",
                        "evidence_tags": [
                            "basic_acoustic.channels",
                            "background_condition",
                            "basic_acoustic.channels",
                        ],
                        "confidence": 0.8,
                    }
                ]
            }
        }
        sanitized, changes = assisted.sanitize_assisted_descriptive(
            payload, {"basic_acoustic.channels": 1}
        )
        self.assertEqual(
            sanitized["descriptive"]["facets"][0]["evidence_tags"],
            ["basic_acoustic.channels"],
        )
        self.assertEqual(changes[0]["removed_count"], 2)


if __name__ == "__main__":
    unittest.main()
