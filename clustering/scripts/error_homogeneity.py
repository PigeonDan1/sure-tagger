#!/usr/bin/env python3
"""Post-hoc WER/CER homogeneity for a frozen tags-only partition.

Does not change cluster_id. Missing error_rate stays null and is excluded.
Python 3.6 compatible.
"""

from __future__ import print_function

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
HERE = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
DEFAULT_ASSIGN = HERE / "result" / "scene-clusters" / "cluster_assignments.jsonl"
DEFAULT_WER = HERE / "sampled-100-per-source" / "sample_wer_cer.jsonl"
DEFAULT_SUMMARY = HERE / "result" / "scene-clusters" / "cluster_summary.json"
DEFAULT_OUTPUT = HERE / "result" / "scene-clusters" / "error_homogeneity.json"
PRIMARY_MODEL = "Qwen__Qwen3-ASR-1.7B"
DIAGNOSTIC_MODELS = (
    "Qwen__Qwen3-ASR-1.7B",
    "FunAudioLLM__SenseVoiceSmall",
    "openai__whisper-large-v3-turbo",
    "fun-asr-flash-2026-06-15",
)
N_RANDOM = 80
RANDOM_SEED = 0
EPS = 1e-12
K_REF = 6
LAMBDA_K = 0.3


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def iqr(values):
    if len(values) < 4:
        return None
    return quantile(values, 0.75) - quantile(values, 0.25)


def mad(values):
    if not values:
        return None
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def mean(values):
    if not values:
        return None
    return sum(values) / float(len(values))


def eta_k_score(eta2, k, k_ref=K_REF, lam=LAMBDA_K):
    """Post-hoc score: eta2 is gain, extra K beyond k_ref is cost.

    Never used to choose or change cluster_id.
    """
    if eta2 is None or k is None or k < 1:
        return None
    # cost = 0 when K <= K_ref; each doubling of K beyond K_ref costs lambda
    ratio = max(float(k), float(k_ref)) / float(k_ref)
    cost_units = math.log(ratio) / math.log(2.0)
    cost = lam * cost_units
    return {
        "eta2": eta2,
        "k": int(k),
        "k_ref": int(k_ref),
        "lambda": lam,
        "gain": eta2,
        "cost": cost,
        "score": eta2 - cost,
        "formula": "S = eta2 - lambda * log2(max(K, K_ref) / K_ref)",
        "used_for_assignment": False,
    }


def optimal_partition_stats(values, k):
    """Exact 1-D k-means on sorted error_rate. Diagnostic upper bound only."""
    xs = sorted(float(value) for value in values)
    n = len(xs)
    if n < 2 or k < 1:
        return None
    k = min(int(k), n)
    inf = 1e100
    ps = [0.0]
    ps2 = [0.0]
    for x in xs:
        ps.append(ps[-1] + x)
        ps2.append(ps2[-1] + x * x)
    mu = ps[-1] / float(n)
    ss_total = ps2[-1] - n * mu * mu

    def cost(i, j):
        count = j - i
        s = ps[j] - ps[i]
        s2 = ps2[j] - ps2[i]
        return s2 - (s * s) / float(count)

    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    prev = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for t in range(1, k + 1):
        for j in range(t, n + 1):
            best = inf
            best_i = -1
            for i in range(t - 1, j):
                base = dp[t - 1][i]
                if base >= inf / 10.0:
                    continue
                val = base + cost(i, j)
                if val < best:
                    best = val
                    best_i = i
            dp[t][j] = best
            prev[t][j] = best_i
    ss_within = dp[k][n]
    eta2 = 0.0 if ss_total <= EPS else 1.0 - ss_within / ss_total
    return {
        "k": k,
        "n_valid": n,
        "ss_total": ss_total,
        "ss_within": ss_within,
        "ss_between": ss_total - ss_within,
        "eta2": eta2,
        "eta_k_score": eta_k_score(eta2, k),
        "used_for_assignment": False,
    }


def is_valid_score(score):
    if not isinstance(score, dict):
        return False
    error_rate = score.get("error_rate")
    if not isinstance(error_rate, (int, float)) or isinstance(error_rate, bool):
        return False
    if error_rate < 0 or math.isnan(float(error_rate)):
        return False
    counts = score.get("counts") if isinstance(score.get("counts"), dict) else None
    n_ref = counts.get("all") if counts else None
    valid_flag = bool(score.get("wer_valid")) or bool(score.get("cer_valid"))
    return (
        score.get("metric_name") in ("wer", "cer")
        and valid_flag
        and isinstance(n_ref, (int, float))
        and n_ref > 0
    )


def load_assignments(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "cluster_id": row["cluster_id"],
                    "dataset": row.get("dataset"),
                }
            )
    return rows


def load_scores(path):
    mapping = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            by_model = {}
            for score in row.get("scores") or []:
                by_model[score.get("model_id")] = score
            mapping[row["sample_id"]] = {
                "language": row.get("language"),
                "scores": by_model,
            }
    return mapping


def collect_items(assignments, score_map, model_id):
    items = []
    for row in assignments:
        packed = score_map.get(row["sample_id"]) or {}
        score = (packed.get("scores") or {}).get(model_id)
        if not is_valid_score(score):
            continue
        items.append(
            {
                "sample_id": row["sample_id"],
                "cluster_id": row["cluster_id"],
                "dataset": row.get("dataset"),
                "language": packed.get("language"),
                "metric_name": score.get("metric_name"),
                "error_rate": float(score["error_rate"]),
            }
        )
    return items


def grouping_stats(items, key):
    values = [item["error_rate"] for item in items]
    n = len(values)
    empty = {
        "n_valid": n,
        "k": 0,
        "eta2": None,
        "ss_total": None,
        "ss_within": None,
        "ss_between": None,
        "sigma_total": None,
        "sigma_within": None,
        "sigma_ratio": None,
        "h_mad": None,
        "mad_global": None,
        "mad_within_weighted": None,
        "iqr_global": None,
        "iqr_within_weighted": None,
        "mean": None,
        "median": None,
        "per_group": {},
    }
    if n < 2:
        return empty
    mu = mean(values)
    ss_total = sum((value - mu) ** 2 for value in values)
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item["error_rate"])
    ss_within = 0.0
    ss_between = 0.0
    per_group = {}
    mad_global = mad(values)
    iqr_global = iqr(values)
    weighted_mad = 0.0
    weighted_iqr = 0.0
    for group_id, group_values in groups.items():
        group_mean = mean(group_values)
        ss_within += sum((value - group_mean) ** 2 for value in group_values)
        ss_between += len(group_values) * ((group_mean - mu) ** 2)
        group_mad = mad(group_values)
        group_iqr = iqr(group_values)
        weighted_mad += (len(group_values) / float(n)) * (group_mad or 0.0)
        if group_iqr is not None:
            weighted_iqr += (len(group_values) / float(n)) * group_iqr
        per_group[str(group_id)] = {
            "n_valid": len(group_values),
            "mean": group_mean,
            "median": statistics.median(group_values),
            "mad": group_mad,
            "iqr": group_iqr,
            "mad_ratio_vs_global": None if not mad_global else group_mad / mad_global,
            "iqr_ratio_vs_global": None if not iqr_global or group_iqr is None else group_iqr / iqr_global,
        }
    eta2 = 0.0 if ss_total <= EPS else ss_between / ss_total
    sigma_total = math.sqrt(ss_total / float(n))
    sigma_within = math.sqrt(ss_within / float(n))
    score = eta_k_score(eta2, len(groups))
    return {
        "n_valid": n,
        "k": len(groups),
        "eta2": eta2,
        "eta_k_score": score,
        "ss_total": ss_total,
        "ss_within": ss_within,
        "ss_between": ss_between,
        "sigma_total": sigma_total,
        "sigma_within": sigma_within,
        "sigma_ratio": None if sigma_total <= EPS else sigma_within / sigma_total,
        "h_mad": None if not mad_global else 1.0 - weighted_mad / mad_global,
        "mad_global": mad_global,
        "mad_within_weighted": weighted_mad,
        "iqr_global": iqr_global,
        "iqr_within_weighted": weighted_iqr,
        "mean": mu,
        "median": statistics.median(values),
        "per_group": per_group,
    }


def random_eta2(items, n_rep, seed):
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    sizes = list(Counter(item["cluster_id"] for item in items).values())
    values = []
    for _ in range(n_rep):
        labels = []
        for index, size in enumerate(sizes):
            labels.extend([index] * size)
        rng.shuffle(labels)
        tmp = []
        for item, label in zip(items, labels):
            cloned = dict(item)
            cloned["random_id"] = label
            tmp.append(cloned)
        values.append(grouping_stats(tmp, "random_id")["eta2"])
    eta_mean = mean(values)
    k_random = len(sizes)
    return {
        "n_rep": n_rep,
        "seed": seed,
        "k": k_random,
        "mean": eta_mean,
        "min": min(values),
        "max": max(values),
        "eta_k_score": eta_k_score(eta_mean, k_random),
    }


def evaluate_model(items):
    cluster = grouping_stats(items, "cluster_id")
    source = grouping_stats(items, "dataset")
    language = grouping_stats(items, "language")
    no_c05 = [item for item in items if item["cluster_id"] != "C05"]
    optimal = optimal_partition_stats([item["error_rate"] for item in items], K_REF)
    return {
        "n_valid": len(items),
        "used_for_assignment": False,
        "cluster": cluster,
        "source": source,
        "language": language,
        "cluster_exclude_C05": grouping_stats(no_c05, "cluster_id"),
        "random_same_sizes": random_eta2(items, N_RANDOM, RANDOM_SEED),
        "optimal_k": optimal,
        "formula": {
            "eta2": "1 - SS_within / SS_total = SS_between / SS_total",
            "ss_total": "sum_i (error_i - mean_*)^2",
            "ss_within": "sum_g sum_{i in g} (error_i - mean_g)^2",
            "sigma_ratio": "sigma_within / sigma_total = sqrt(1 - eta2)",
            "h_mad": "1 - weighted_MAD_g / MAD_*",
            "eta_k_score": "S = eta2 - lambda * log2(max(K, K_ref) / K_ref); lambda=%s, K_ref=%s; not used for assignment"
            % (LAMBDA_K, K_REF),
            "missing": "error_rate is null or invalid -> excluded, not filled with 0",
        },
    }


def compact_for_summary(result):
    cluster = result["cluster"]
    random_stats = result["random_same_sizes"] or {}
    source = result["source"]
    exclude = result["cluster_exclude_C05"]
    return {
        "metric": "eta2_error_rate",
        "n_valid": result["n_valid"],
        "eta2": cluster["eta2"],
        "sigma_within": cluster["sigma_within"],
        "sigma_total": cluster["sigma_total"],
        "sigma_ratio": cluster["sigma_ratio"],
        "h_mad": cluster["h_mad"],
        "mad_global": cluster["mad_global"],
        "mad_within_weighted": cluster["mad_within_weighted"],
        "eta2_random_same_sizes": random_stats.get("mean"),
        "eta2_by_source": source["eta2"],
        "eta2_exclude_C05": exclude["eta2"],
        "eta_k_score": cluster.get("eta_k_score"),
        "eta_k_score_random": random_stats.get("eta_k_score"),
        "eta_k_score_source": source.get("eta_k_score"),
        "eta_k_score_optimal": (result.get("optimal_k") or {}).get("eta_k_score"),
        "used_for_assignment": False,
        "per_cluster": cluster["per_group"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc WER/CER homogeneity on frozen clusters")
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGN)
    parser.add_argument("--wer-cer", type=Path, default=DEFAULT_WER)
    parser.add_argument("--cluster-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--update-summary", action="store_true", default=True)
    parser.add_argument("--no-update-summary", action="store_false", dest="update_summary")
    return parser.parse_args()


def main():
    args = parse_args()
    assignments = load_assignments(args.assignments)
    score_map = load_scores(args.wer_cer)
    payload = {
        "assignment_basis": "audio_tags_only",
        "used_for_assignment": False,
        "primary_model": PRIMARY_MODEL,
        "n_random": N_RANDOM,
        "models": {},
    }
    for model_id in DIAGNOSTIC_MODELS:
        items = collect_items(assignments, score_map, model_id)
        payload["models"][model_id] = evaluate_model(items)
        cluster = payload["models"][model_id]["cluster"]
        random_stats = payload["models"][model_id]["random_same_sizes"]
        print(
            "%s n=%s eta2=%.4f S=%.4f random=%.4f source=%.4f H_MAD=%s"
            % (
                model_id,
                len(items),
                cluster["eta2"],
                (cluster.get("eta_k_score") or {}).get("score"),
                random_stats["mean"] if random_stats else float("nan"),
                payload["models"][model_id]["source"]["eta2"],
                None if cluster["h_mad"] is None else round(cluster["h_mad"], 4),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.update_summary and args.cluster_summary.is_file():
        summary = json.loads(args.cluster_summary.read_text())
        by_model = payload["models"]
        for item in summary.get("dispersion_evaluations") or []:
            model_id = item.get("eval_asr_system_id")
            if model_id in by_model:
                item["error_homogeneity"] = compact_for_summary(by_model[model_id])
        summary["error_homogeneity_primary"] = compact_for_summary(by_model[PRIMARY_MODEL])
        args.cluster_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print("updated %s" % args.cluster_summary)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
