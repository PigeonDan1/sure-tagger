#!/usr/bin/env python3
"""K-medoids candidates on a precomputed tag distance matrix.

This release packages one clustering method. WER/CER diagnostics stay in the
caller and do not change the distance used by this partition generator.
"""

import math

DEFAULT_METHODS = ("kmedoids",)


def snk_score(eta2, k, k_ref=6, lam=0.3):
    if eta2 is None or k is None or k < 1:
        return None
    ratio = max(float(k), float(k_ref)) / float(k_ref)
    cost = lam * (math.log(ratio) / math.log(2.0))
    return eta2 - cost


def generate_method_partitions(distance, active, args, methods=None):
    """Yield raw partitions. Labels cover the full distance matrix length."""
    methods = tuple(methods or getattr(args, "cluster_method_list", DEFAULT_METHODS))
    unsupported = set(methods) - set(DEFAULT_METHODS)
    if unsupported:
        raise ValueError("unsupported clustering method(s): %s; only kmedoids is packaged" % ",".join(sorted(unsupported)))
    k_min = int(getattr(args, "k_min", 6))
    k_max = int(getattr(args, "k_max", 12))
    seeds = list(getattr(args, "seed_values", [0, 1, 2]))
    ks = list(range(k_min, k_max + 1))
    n = len(distance)

    if "kmedoids" in methods:
        from cluster_scenes import kmedoids, submatrix

        active_dist = submatrix(distance, active)
        for k in ks:
            for seed in seeds:
                local_labels, local_medoids = kmedoids(active_dist, k, seed)
                if local_labels is None:
                    continue
                labels = [-1] * n
                for local_index, label in enumerate(local_labels):
                    labels[active[local_index]] = label
                yield {
                    "method": "kmedoids",
                    "k": k,
                    "seed": seed,
                    "labels": labels,
                }
