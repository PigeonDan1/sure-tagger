#!/usr/bin/env python3
"""Prepare and score the 10-sample single-speaker ASR routing demo."""

import argparse
import copy
import gzip
import json
from pathlib import Path
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = PROJECT_ROOT / "demo" / "data.js"
DATASETS = (("LibriSpeech", 5, "en"), ("AISHELL-1", 5, "zh"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_demo_data():
    source = DEMO_DATA.read_text(encoding="utf-8").strip()
    prefix = "window.DEMO_DATA = "
    if not source.startswith(prefix) or not source.endswith(";"):
        raise ValueError("demo/data.js does not contain the expected JSON wrapper")
    return json.loads(source[len(prefix) : -1])


def prepare(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo = _load_demo_data()
    selected = []
    for dataset, count, language in DATASETS:
        rows = [row for row in demo["samples"] if row["dataset"] == dataset]
        if len(rows) < count:
            raise ValueError("not enough %s demo samples" % dataset)
        for row in rows[:count]:
            speaker_count = row.get("tags", {}).get("speaker", {}).get(
                "speaker_count"
            )
            if speaker_count != 1:
                raise ValueError(
                    "%s is not a confirmed single-speaker demo sample"
                    % row["sampleId"]
                )
            selected.append((row, language))

    manifests = []
    references = []
    baseline_tags = []
    for row, language in selected:
        audio_path = (PROJECT_ROOT / "demo" / row["audio"]).resolve()
        if not audio_path.is_file():
            raise ValueError("demo audio is missing: %s" % audio_path)
        manifests.append(
            {
                "corpus": {
                    "dataset_name": row["dataset"],
                    "native_metadata": {},
                    "source_urls": {
                        "article": [],
                        "dataset_card": [],
                        "github": [],
                        "huggingface": [],
                    },
                },
                "sample": {
                    "audio": {"path": str(audio_path)},
                    "native_metadata": {},
                    "sample_id": row["sampleId"],
                    # Reference text stays in references.jsonl only.
                    "text": {"transcript": ""},
                },
            }
        )
        references.append(
            {
                "sample_id": row["sampleId"],
                "dataset": row["dataset"],
                "language": language,
                "reference": row["transcript"],
            }
        )
        baseline_tags.append(copy.deepcopy(row["tags"]))

    _write_jsonl(output_dir / "manifest.jsonl", manifests)
    _write_jsonl(output_dir / "references.jsonl", references)
    _write_jsonl(output_dir / "baseline_tags.jsonl", baseline_tags)
    _write_json(
        output_dir / "selection.json",
        {
            "source": str(DEMO_DATA),
            "selection": "first five LibriSpeech and first five AISHELL-1 demo samples",
            "sample_count": len(selected),
            "sample_ids": [row["sampleId"] for row, _language in selected],
            "reference_isolation": "manifest transcript is empty",
        },
    )
    print("prepared %d samples in %s" % (len(selected), output_dir))


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _units(text, language):
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    if language == "en":
        return re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)?", normalized.upper())
    return [
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    ]


def _metric(reference, hypothesis, language):
    reference_units = _units(reference, language)
    hypothesis_units = _units(hypothesis, language)
    errors = _edit_distance(reference_units, hypothesis_units)
    return {
        "errors": errors,
        "reference_units": len(reference_units),
        "hypothesis_units": len(hypothesis_units),
        "error_rate": (
            round(errors / float(len(reference_units)), 6)
            if reference_units
            else None
        ),
        "exact_match": reference_units == hypothesis_units,
    }


def _aggregate(rows):
    output = {}
    for label, dataset in (("all", None),) + tuple(
        (name, name) for name, _count, _language in DATASETS
    ):
        selected = [row for row in rows if dataset is None or row["dataset"] == dataset]
        reference_units = sum(row["old_metric"]["reference_units"] for row in selected)
        old_errors = sum(row["old_metric"]["errors"] for row in selected)
        new_errors = sum(row["new_metric"]["errors"] for row in selected)
        old_rate = old_errors / float(reference_units) if reference_units else None
        new_rate = new_errors / float(reference_units) if reference_units else None
        output[label] = {
            "sample_count": len(selected),
            "unit": "mixed" if dataset is None else ("word" if dataset == "LibriSpeech" else "character"),
            "reference_units": reference_units,
            "old_moss_errors": old_errors,
            "new_route_errors": new_errors,
            "old_moss_error_rate": round(old_rate, 6) if old_rate is not None else None,
            "new_route_error_rate": round(new_rate, 6) if new_rate is not None else None,
            "absolute_error_rate_reduction": (
                round(old_rate - new_rate, 6)
                if old_rate is not None and new_rate is not None
                else None
            ),
            "relative_error_reduction": (
                round((old_rate - new_rate) / old_rate, 6)
                if old_rate
                else None
            ),
            "old_exact_match_count": sum(row["old_metric"]["exact_match"] for row in selected),
            "new_exact_match_count": sum(row["new_metric"]["exact_match"] for row in selected),
        }
    return output


def _percent(value):
    return "N/A" if value is None else "%.2f%%" % (100.0 * value)


def score(output_dir):
    output_dir = Path(output_dir)
    references = _read_jsonl(output_dir / "references.jsonl")
    baseline_tags = _read_jsonl(output_dir / "baseline_tags.jsonl")
    results = []
    for path in sorted((output_dir / "shards").glob("shard_*/speaker_v2_shadow_results.jsonl")):
        results.extend(_read_jsonl(path))
    by_id = {row["sample_id"]: row for row in results}
    expected_ids = [row["sample_id"] for row in references]
    if len(results) != len(by_id) or sorted(by_id) != sorted(expected_ids):
        raise ValueError(
            "speaker results do not match selection: expected %s, got %s"
            % (sorted(expected_ids), sorted(row["sample_id"] for row in results))
        )
    if len(baseline_tags) != len(references):
        raise ValueError("baseline tags and references have different lengths")

    comparisons = []
    merged_tags = []
    ordered_results = []
    for reference, base_tags in zip(references, baseline_tags):
        result = by_id[reference["sample_id"]]
        if result.get("status") != "ok":
            raise ValueError("speaker run failed for %s" % reference["sample_id"])
        fusion = _read_json(result["fusion_artifact"])
        moss_text = fusion["asr_candidates"]["moss"]["text"]
        new_text = result["speaker_asr_transcript"]
        route = result["asr_route"]
        count_candidates = route["single_speaker_consensus"]["candidates"]
        comparisons.append(
            {
                "sample_id": reference["sample_id"],
                "dataset": reference["dataset"],
                "language": reference["language"],
                "reference": reference["reference"],
                "old_moss_text": moss_text,
                "new_route_text": new_text,
                "old_metric": _metric(reference["reference"], moss_text, reference["language"]),
                "new_metric": _metric(reference["reference"], new_text, reference["language"]),
                "selected_source": route.get("selected_source"),
                "selection_basis": route.get("selection_basis"),
                "consensus_override_applied": route.get("consensus_override_applied"),
                "single_speaker_consensus": route["single_speaker_consensus"]["all_at_most_one"],
                "moss_speaker_count": count_candidates["moss"]["speaker_count"],
                "diarizen_speaker_count": count_candidates["diarizen"]["speaker_count"],
                "public_speaker_count": result["speaker"].get("speaker_count"),
            }
        )
        merged = copy.deepcopy(base_tags)
        merged.setdefault("speaker", {})["asr_transcript"] = new_text or None
        merged_tags.append(merged)
        ordered_results.append(result)

    aggregate = _aggregate(comparisons)
    summary = {
        "sample_count": len(comparisons),
        "successful_sample_count": len(comparisons),
        "consensus_count": sum(row["single_speaker_consensus"] for row in comparisons),
        "route_changed_to_firered_count": sum(
            row["consensus_override_applied"] for row in comparisons
        ),
        "moss_count_correct": sum(row["moss_speaker_count"] == 1 for row in comparisons),
        "diarizen_count_correct": sum(row["diarizen_speaker_count"] == 1 for row in comparisons),
        "public_count_correct": sum(row["public_speaker_count"] == 1 for row in comparisons),
        "metrics": aggregate,
    }
    _write_jsonl(output_dir / "speaker_results.jsonl", ordered_results)
    _write_jsonl(output_dir / "asr_comparison.jsonl", comparisons)
    _write_jsonl(output_dir / "merged_tags.jsonl", merged_tags)
    _write_json(output_dir / "score.json", summary)

    lines = [
        "# Single-speaker ASR routing demo",
        "",
        "- Samples: %d (5 LibriSpeech + 5 AISHELL-1)" % len(comparisons),
        "- Single-speaker consensus: %d/%d" % (summary["consensus_count"], len(comparisons)),
        "- ASR route changed from MOSS to FireRed: %d/%d" % (summary["route_changed_to_firered_count"], len(comparisons)),
        "- Count accuracy: MOSS %d/%d, DiariZen %d/%d, public route %d/%d"
        % (
            summary["moss_count_correct"],
            len(comparisons),
            summary["diarizen_count_correct"],
            len(comparisons),
            summary["public_count_correct"],
            len(comparisons),
        ),
        "",
        "| Slice | Metric | Old MOSS | New route | Absolute reduction | Exact match old -> new |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, metric_name in (("LibriSpeech", "WER"), ("AISHELL-1", "CER"), ("all", "mixed unit error")):
        item = aggregate[label]
        lines.append(
            "| %s | %s | %s | %s | %s | %d -> %d |"
            % (
                label,
                metric_name,
                _percent(item["old_moss_error_rate"]),
                _percent(item["new_route_error_rate"]),
                _percent(item["absolute_error_rate_reduction"]),
                item["old_exact_match_count"],
                item["new_exact_match_count"],
            )
        )
    lines.extend(
        [
            "",
            "Positive absolute reduction means the new route improved over the old MOSS route.",
            "The 10-sample demo is directional only and is not a statistically powered evaluation.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
    elif args.command == "score":
        score(args.output_dir)
    else:
        parser.error("a command is required")


if __name__ == "__main__":
    main()
