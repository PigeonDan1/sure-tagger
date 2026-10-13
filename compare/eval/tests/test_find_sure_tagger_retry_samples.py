import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from find_sure_tagger_retry_samples import find_retry_samples  # noqa: E402


class FindSureTaggerRetrySamplesTest(unittest.TestCase):
    def test_required_source_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "nemotron_3_diarization"
            expected = {
                ("shard_00", "healthy"),
                ("shard_00", "unusable"),
                ("shard_00", "missing"),
            }
            for sample_id, status, usable in (
                ("healthy", "estimated", True),
                ("unusable", "missing", False),
            ):
                evidence_dir = root / "shard_00" / "speaker_v2" / sample_id / "evidence"
                evidence_dir.mkdir(parents=True)
                with gzip.open(str(evidence_dir / "sample.json.gz"), "wt", encoding="utf-8") as stream:
                    json.dump({
                        "scope": {"sample_id": sample_id},
                        "source": {"name": source},
                        "status": status,
                        "quality": {"usable": usable, "reason": "inference failed"},
                    }, stream)

            failures = find_retry_samples(root, source, expected)
            self.assertEqual(
                [(shard, sample_id) for shard, sample_id, _reason in failures],
                [("shard_00", "missing"), ("shard_00", "unusable")],
            )


if __name__ == "__main__":
    unittest.main()
