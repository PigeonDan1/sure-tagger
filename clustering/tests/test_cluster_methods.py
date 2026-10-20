#!/usr/bin/env python3

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import cluster_methods
import cluster_scenes


def blob_distance(n_per=4, n_blobs=3, inner=0.05, outer=1.0):
    n = n_per * n_blobs
    distance = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = inner if (i // n_per) == (j // n_per) else outer
            distance[i][j] = distance[j][i] = value
    return distance


class KMedoidsTest(unittest.TestCase):
    def test_kmedoids_recovers_separated_blobs_and_is_reproducible(self):
        distance = blob_distance()
        labels, medoids = cluster_scenes.kmedoids(distance, k=3, seed=0)
        repeated = cluster_scenes.kmedoids(distance, k=3, seed=0)

        self.assertEqual((labels, medoids), repeated)
        groups = {frozenset(i for i, value in enumerate(labels) if value == label)
                  for label in set(labels)}
        self.assertEqual(groups, {frozenset(range(start, start + 4))
                                  for start in (0, 4, 8)})
        self.assertEqual(len(set(medoids)), 3)
        for label, medoid in enumerate(medoids):
            self.assertEqual(labels[medoid], label)

    def test_partition_generator_preserves_global_indices_and_inactive_rows(self):
        distance = blob_distance()
        active = [1, 2, 5, 6, 9, 10]
        args = SimpleNamespace(k_min=3, k_max=3, seed_values=[0])
        partitions = list(cluster_methods.generate_method_partitions(distance, active, args))

        self.assertEqual(len(partitions), 1)
        labels = partitions[0]["labels"]
        self.assertEqual(len(labels), len(distance))
        self.assertEqual({i for i, label in enumerate(labels) if label >= 0}, set(active))
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[5], labels[6])
        self.assertEqual(labels[9], labels[10])
        self.assertEqual(len({labels[1], labels[5], labels[9]}), 3)

    def test_partition_generator_searches_requested_k_and_seeds(self):
        distance = blob_distance()
        args = SimpleNamespace(k_min=2, k_max=3, seed_values=[0, 1])
        partitions = list(cluster_methods.generate_method_partitions(
            distance, list(range(len(distance))), args
        ))

        self.assertEqual({(item["k"], item["seed"]) for item in partitions},
                         {(2, 0), (2, 1), (3, 0), (3, 1)})
        self.assertTrue(all(item["method"] == "kmedoids" for item in partitions))

    def test_partition_generator_rejects_unbundled_methods(self):
        args = SimpleNamespace(k_min=2, k_max=2, seed_values=[0])
        distance = blob_distance()
        with self.assertRaisesRegex(ValueError, "only kmedoids is packaged"):
            list(cluster_methods.generate_method_partitions(
                distance, list(range(len(distance))), args, ["kmedoids", "dbscan"]
            ))

    def test_cli_defaults_to_kmedoids_and_rejects_unbundled_methods(self):
        with patch.object(sys, "argv", ["cluster_scenes.py"]):
            self.assertEqual(cluster_scenes.parse_args().cluster_methods, "kmedoids")
        with patch.object(sys, "argv", ["cluster_scenes.py", "--cluster-methods", "dbscan"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    cluster_scenes.parse_args()
        self.assertEqual(failure.exception.code, 2)

    def test_snk_penalizes_extra_k_and_handles_missing_scores(self):
        self.assertAlmostEqual(cluster_methods.snk_score(0.5, 6), 0.5)
        self.assertLess(cluster_methods.snk_score(0.5, 12), 0.5)
        self.assertIsNone(cluster_methods.snk_score(None, 6))

    def test_select_candidate_can_filter_kmedoids_seeds_by_posthoc_snk(self):
        def fake(seed, j, snk, k=6):
            return {
                "method": "kmedoids",
                "seed": seed,
                "k": k,
                "metrics": {
                    "passed": True,
                    "j": j,
                    "k": k,
                    "q_tag": 0.5,
                    "source_driven": False,
                },
                "posthoc": {"mean_snk": snk},
            }

        chosen, publishable = cluster_scenes.select_candidate(
            [fake(0, 0.20, 0.05), fake(1, 0.21, 0.40), fake(2, 0.50, 0.90)],
            wer_filter=True,
        )
        self.assertTrue(publishable)
        self.assertEqual(chosen["seed"], 1)


if __name__ == "__main__":
    unittest.main()
