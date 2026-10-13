import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_ami_speaker_gender_gt.py"
SPEC = importlib.util.spec_from_file_location("run_ami_speaker_gender_gt", SCRIPT_PATH)
gender_gt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gender_gt)


class ParseGenderResponseTest(unittest.TestCase):
    def test_parses_exact_json(self):
        self.assertEqual(gender_gt.parse_gender_response('{"gender":"female"}'), "female")

    def test_parses_fenced_json(self):
        self.assertEqual(
            gender_gt.parse_gender_response('```json\n{"gender": "MALE"}\n```'),
            "male",
        )

    def test_rejects_extra_keys(self):
        with self.assertRaises(ValueError):
            gender_gt.parse_gender_response('{"gender":"male","confidence":0.9}')


class SpeakerOrderTest(unittest.TestCase):
    def test_orders_by_first_start_then_source_id(self):
        utterances = [
            {"speaker": "C", "start": 3.0},
            {"speaker": "B", "start": 1.0},
            {"speaker": "A", "start": 1.0},
            {"speaker": "B", "start": 0.5},
        ]
        self.assertEqual(gender_gt.speaker_order(utterances), ["B", "A", "C"])


class RecoverOriginTest(unittest.TestCase):
    def test_recovers_shared_absolute_origin(self):
        local = [
            {"utt_id": "u1", "speaker": "A", "start": 1.2, "end": 2.0},
            {"utt_id": "u2", "speaker": "B", "start": 3.0, "end": 4.0},
        ]
        absolute = {
            "u1": {"speaker": "A", "start": 11.2, "end": 12.0},
            "u2": {"speaker": "B", "start": 13.0, "end": 14.0},
        }
        self.assertEqual(gender_gt.recover_origin(local, absolute), 10.0)


if __name__ == "__main__":
    unittest.main()
