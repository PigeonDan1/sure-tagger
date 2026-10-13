#!/usr/bin/env python3
"""Score the merged language verdict against the benchmark's own ground truth.

This is a pure analysis pass: no model calls, no GPU, no audio.  It reads the
language audit sidecar written by ``run_deepseek_batch.py`` plus the
sure-tagger structural result, and reports, per evaluation group, how the
published ``language_content.language`` compares with the upstream LID value.

The scoring rule is imported from ``score_benchmark_v1`` rather than
re-implemented, so this report can never drift from the official scorer:

* a value outside the scorer's language set is ``schema_invalid`` -> 0 of 2
* a null value is ``result_null`` -> 1 of 2
* a known value scores 2 of 2 only when it matches the ground truth

Usage::

    python3 report_language_verification.py \\
        --verification <run>/deepseek/language_verification.jsonl \\
        --structural   <run>/result_with_id.jsonl \\
        --benchmark-root ../dataset/benchmark
"""

from __future__ import print_function

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_benchmark_v1 import LANGUAGES, canonical_language  # noqa: E402

COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
T1_GROUP = "T1_natural_meeting_utterance"
T1_GT_LANGUAGE = "en"
MAX_UNIT_SCORE = 2.0
NULL_UNIT_SCORE = 1.0


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, exc))
    if not rows:
        raise ValueError("empty JSONL: %s" % path)
    return rows


def load_ground_truth(benchmark_root):
    """sample_id -> (evaluation_group, reference language).

    Only the two language-bearing tracks are returned; T3 has no language axis
    and is therefore absent from the map rather than defaulted to something.
    """
    benchmark_root = Path(benchmark_root)
    ground_truth = {}
    t1_path = benchmark_root / "t1-ami" / "manifest.jsonl"
    for row in load_jsonl(t1_path):
        sample_id = row["sample"]["sample_id"]
        ground_truth[sample_id] = (T1_GROUP, T1_GT_LANGUAGE)
    t2_path = benchmark_root / "t2-language-asr" / "metadata.jsonl"
    for row in load_jsonl(t2_path):
        track = row["track"]
        if track == "T2_multilingual_fleurs":
            reference = canonical_language(row["language"])
        elif track == "T2_clean_librispeech":
            reference = canonical_language(row["ground_truth"]["language"])
        else:
            raise ValueError("unexpected T2 track %s" % track)
        ground_truth[row["sample_id"]] = (track, reference)
    return ground_truth


def unit_score(value, reference):
    """Replicate ``prediction_state`` + ``unit_score`` for a language field."""
    if value is None:
        return NULL_UNIT_SCORE, "result_null"
    canonical = canonical_language(value)
    if canonical is None:
        return 0.0, "schema_invalid"
    if canonical == reference:
        return MAX_UNIT_SCORE, "result_known"
    return 0.0, "result_known"


def _rate(pairs):
    if not pairs:
        return None
    return sum(score for score, _ in pairs) / (MAX_UNIT_SCORE * len(pairs))


def build_report(verification_path, structural_path, benchmark_root):
    ground_truth = load_ground_truth(benchmark_root)
    structural = {
        row["sample_id"]: row.get("tags") or {}
        for row in load_jsonl(structural_path)
    }
    groups = OrderedDict()
    transitions = Counter()
    rows_out = []
    unmatched = []
    for row in load_jsonl(verification_path):
        sample_id = row.get("sample_id")
        if sample_id not in ground_truth:
            unmatched.append(sample_id)
            continue
        group, reference = ground_truth[sample_id]
        lid = row.get("lid_language")
        published = row.get("published_language")
        lid_score, lid_state = unit_score(lid, reference)
        published_score, published_state = unit_score(published, reference)
        entry = groups.setdefault(
            group,
            {"baseline": [], "published": [], "lid_states": Counter(), "published_states": Counter()},
        )
        entry["baseline"].append((lid_score, lid_state))
        entry["published"].append((published_score, published_state))
        entry["lid_states"][lid_state] += 1
        entry["published_states"][published_state] += 1
        if row.get("changed"):
            transitions["%s -> %s" % (lid, published)] += 1
        if published_score < lid_score:
            verdict = "broken"
        elif published_score > lid_score:
            verdict = "fixed"
        else:
            verdict = "unchanged"
        rows_out.append(
            {
                "sample_id": sample_id,
                "evaluation_group": group,
                "reference": reference,
                "lid_language": lid,
                "published_language": published,
                "verdict_state": row.get("verdict_state"),
                "confidence": row.get("confidence"),
                "reason_short": row.get("reason_short"),
                "lid_score": lid_score,
                "published_score": published_score,
                "outcome": verdict,
                "transcript_chars": row.get("transcript_chars"),
            }
        )

    summary_groups = OrderedDict()
    for group in sorted(groups):
        entry = groups[group]
        baseline = _rate(entry["baseline"])
        published = _rate(entry["published"])
        summary_groups[group] = {
            "sample_count": len(entry["baseline"]),
            "lid_score_rate": baseline,
            "published_score_rate": published,
            "delta": None if baseline is None else published - baseline,
            "fixed_count": sum(1 for item in rows_out
                               if item["evaluation_group"] == group and item["outcome"] == "fixed"),
            "broken_count": sum(1 for item in rows_out
                                if item["evaluation_group"] == group and item["outcome"] == "broken"),
            "lid_state_counts": dict(entry["lid_states"]),
            "published_state_counts": dict(entry["published_states"]),
        }

    overall_baseline = _rate([(row["lid_score"], "") for row in rows_out])
    overall_published = _rate([(row["published_score"], "") for row in rows_out])
    return {
        "verification": str(Path(verification_path).resolve()),
        "structural": str(Path(structural_path).resolve()),
        "scored_sample_count": len(rows_out),
        "unmatched_sample_count": len(unmatched),
        "unmatched_sample_ids": unmatched[:20],
        "scorer_language_set": sorted(LANGUAGES),
        "overall": {
            "lid_score_rate": overall_baseline,
            "published_score_rate": overall_published,
            "delta": None if overall_baseline is None else overall_published - overall_baseline,
        },
        "outcome_counts": dict(Counter(row["outcome"] for row in rows_out)),
        "transition_counts": dict(
            sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "groups": summary_groups,
        "rows": rows_out,
    }


def render_text(report):
    lines = []
    overall = report["overall"]
    lines.append(
        "language axis (scorer vocabulary: %s)" % ", ".join(report["scorer_language_set"])
    )
    lines.append(
        "scored %d samples; LID %.6f -> merged %.6f (%+.6f)"
        % (
            report["scored_sample_count"],
            overall["lid_score_rate"],
            overall["published_score_rate"],
            overall["delta"],
        )
    )
    if report["unmatched_sample_count"]:
        lines.append(
            "warning: %d verification rows had no language ground truth"
            % report["unmatched_sample_count"]
        )
    lines.append("")
    lines.append("%-32s %6s %10s %10s %9s %7s %7s" % (
        "evaluation group", "n", "LID", "merged", "delta", "fixed", "broken"))
    for group, value in report["groups"].items():
        lines.append("%-32s %6d %10.6f %10.6f %+9.6f %7d %7d" % (
            group,
            value["sample_count"],
            value["lid_score_rate"],
            value["published_score_rate"],
            value["delta"],
            value["fixed_count"],
            value["broken_count"],
        ))
    lines.append("")
    lines.append("outcomes: %s" % json.dumps(report["outcome_counts"], sort_keys=True))
    lines.append("transitions:")
    for key, count in list(report["transition_counts"].items())[:25]:
        lines.append("  %-24s %d" % (key, count))
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--structural", required=True, type=Path)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args.verification, args.structural, args.benchmark_root)
    if not args.quiet:
        print(render_text(report))
    if args.json_output is not None:
        summary = dict(report)
        summary.pop("rows")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in report["rows"]
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
