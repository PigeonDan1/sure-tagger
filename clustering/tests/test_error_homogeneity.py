#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "error_homogeneity.py"
PRIMARY_MODEL = "Qwen__Qwen3-ASR-1.7B"
MISSING_MODEL = "FunAudioLLM__SenseVoiceSmall"


class ErrorHomogeneityCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assignments_path = self.root / "assignments.jsonl"
        self.scores_path = self.root / "scores.jsonl"
        self.summary_path = self.root / "cluster_summary.json"
        self.output_path = self.root / "diagnostics.json"
        self.assignments = [
            {"sample_id": "sample_%s" % i, "global_index": i,
             "cluster_id": "C%02d" % (i // 2), "dataset": "source_%s" % (i % 2)}
            for i in range(4)
        ]
        self.assignments_path.write_text(self.jsonl(self.assignments), encoding="utf-8")
        self.summary_path.write_text(json.dumps({"clusters": ["C00", "C01"]}), encoding="utf-8")

    @staticmethod
    def jsonl(rows):
        return "".join(json.dumps(row) + "\n" for row in rows)

    def score_rows(self, valid_count):
        rows = []
        for i, assignment in enumerate(self.assignments):
            valid = i < valid_count
            rows.append({
                "sample_id": assignment["sample_id"],
                "global_index": i,
                "language": "en",
                "scores": [{
                    "model_id": PRIMARY_MODEL,
                    "metric_name": "wer",
                    "error_rate": (0.1, 0.2, 0.5, 0.6)[i] if valid else None,
                    "wer_valid": valid,
                    "cer_valid": False,
                    "counts": {"all": 10} if valid else None,
                }],
            })
        return rows

    def run_cli(self, rows):
        self.scores_path.write_text(self.jsonl(rows), encoding="utf-8")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--assignments", str(self.assignments_path),
             "--wer-cer", str(self.scores_path),
             "--cluster-summary", str(self.summary_path),
             "--no-update-summary", "--output", str(self.output_path)],
            cwd=str(self.root), env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )

    def assert_valid_output(self, valid_count):
        original_assignments = self.assignments_path.read_bytes()
        original_summary = self.summary_path.read_bytes()
        completed = self.run_cli(self.score_rows(valid_count))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(self.output_path.is_file())

        def reject_nonfinite(value):
            self.fail("diagnostic JSON contains %s" % value)

        payload = json.loads(self.output_path.read_text(), parse_constant=reject_nonfinite)
        primary = payload["models"][PRIMARY_MODEL]
        missing = payload["models"][MISSING_MODEL]
        self.assertEqual(primary["n_valid"], valid_count)
        self.assertEqual(missing["n_valid"], 0)
        self.assertIsNone(missing["cluster"]["eta2"])
        self.assertIsNone(missing["random_same_sizes"])
        self.assertFalse(payload["used_for_assignment"])
        self.assertFalse(primary["used_for_assignment"])
        self.assertEqual(self.assignments_path.read_bytes(), original_assignments)
        self.assertEqual(self.summary_path.read_bytes(), original_summary)
        return primary, completed

    def test_zero_valid_scores_produce_null_diagnostics(self):
        primary, completed = self.assert_valid_output(0)
        self.assertIsNone(primary["cluster"]["eta2"])
        self.assertIsNone(primary["optimal_k"])
        self.assertIn("eta2=null", completed.stdout)

    def test_one_valid_score_keeps_undefined_metrics_null(self):
        primary, _completed = self.assert_valid_output(1)
        self.assertIsNone(primary["cluster"]["eta2"])
        self.assertIsNone(primary["cluster"]["sigma_total"])
        self.assertIsNone(primary["random_same_sizes"])

    def test_normal_scores_produce_numeric_metrics_without_changing_partition(self):
        primary, completed = self.assert_valid_output(4)
        self.assertAlmostEqual(primary["cluster"]["eta2"], 16.0 / 17.0)
        self.assertEqual(set(primary["cluster"]["per_group"]), {"C00", "C01"})
        self.assertIn("eta2=0.9412", completed.stdout)

    def test_duplicate_sample_id_is_rejected(self):
        rows = self.score_rows(4)
        rows.append(dict(rows[0]))
        completed = self.run_cli(rows)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate sample_id sample_0", completed.stderr)
        self.assertFalse(self.output_path.exists())

    def test_duplicate_model_id_is_rejected(self):
        rows = self.score_rows(4)
        rows[0]["scores"].append(dict(rows[0]["scores"][0]))
        completed = self.run_cli(rows)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate model_id " + PRIMARY_MODEL, completed.stderr)
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
