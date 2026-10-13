from __future__ import print_function

import unittest

from tagger.tools.speaker_v2.diarizen_official import (
    DEFAULT_EMBEDDING_MODEL,
    OfficialDiariZenError,
    _is_no_active_speaker_error,
    annotation_to_segments,
    normalize_result,
)


class _Turn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, rows):
        self.rows = rows

    def itertracks(self, yield_label=False):
        self.assert_yield_label = yield_label
        for row in self.rows:
            yield row


class OfficialDiariZenAdapterTest(unittest.TestCase):
    def test_annotation_is_sorted_clipped_and_speaker_ids_are_stable(self):
        annotation = _Annotation(
            [
                (_Turn(2.0, 5.5), "track-c", "upstream-b"),
                (_Turn(-1.0, 2.5), "track-a", "upstream-a"),
                (_Turn(5.0, 7.0), "track-b", "upstream-a"),
            ]
        )

        self.assertEqual(
            annotation_to_segments(annotation, 6.0),
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 2.5,
                    "speaker_id": "speaker_1",
                    "overlap": True,
                },
                {
                    "start_sec": 2.0,
                    "end_sec": 5.5,
                    "speaker_id": "speaker_2",
                    "overlap": True,
                },
                {
                    "start_sec": 5.0,
                    "end_sec": 6.0,
                    "speaker_id": "speaker_1",
                    "overlap": True,
                },
            ],
        )
        self.assertTrue(annotation.assert_yield_label)

    def test_zero_length_segments_after_clipping_are_dropped(self):
        annotation = _Annotation(
            [
                (_Turn(9.0, 10.0), "track-a", "late"),
                (_Turn(0.25, 0.75), "track-b", "kept"),
            ]
        )
        self.assertEqual(
            annotation_to_segments(annotation, 1.0),
            [
                {
                    "start_sec": 0.25,
                    "end_sec": 0.75,
                    "speaker_id": "speaker_1",
                    "overlap": False,
                }
            ],
        )

    def test_normalize_result_rejects_invalid_segments(self):
        with self.assertRaises(OfficialDiariZenError):
            normalize_result(
                {
                    "segments": [
                        {
                            "start_sec": 2.0,
                            "end_sec": 1.0,
                            "speaker_id": "speaker_1",
                        }
                    ],
                    "runtime": {},
                }
            )
        with self.assertRaises(OfficialDiariZenError):
            normalize_result({"segments": "invalid", "runtime": {}})
        with self.assertRaises(OfficialDiariZenError):
            normalize_result({"segments": [], "runtime": []})

    def test_default_wespeaker_path_uses_canonical_pyannote_name(self):
        self.assertIn(
            "pyannote-wespeaker-voxceleb-resnet34-LM",
            str(DEFAULT_EMBEDDING_MODEL),
        )

    def test_only_exact_upstream_empty_cluster_error_is_normalized(self):
        self.assertTrue(
            _is_no_active_speaker_error(
                ValueError("negative dimensions are not allowed")
            )
        )
        self.assertFalse(
            _is_no_active_speaker_error(ValueError("another value error"))
        )
        self.assertFalse(
            _is_no_active_speaker_error(
                RuntimeError("negative dimensions are not allowed")
            )
        )


if __name__ == "__main__":
    unittest.main()
