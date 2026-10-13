#!/usr/bin/env python3
"""Prepare and score the local speaker-v2 multi-speaker evaluation set."""

import argparse
import csv
import json
import math
from pathlib import Path


BOOLEAN_FIELDS = ("multi_speaker", "speaker_change", "speaker_overlap")
INTEGER_FIELDS = ("speaker_count", "speaker_change_count")
FLOAT_FIELDS = ("overlap_ratio",)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--audio-dir", default=None)
    prepare.add_argument("--audio-extension", default=".mp3")
    prepare.add_argument("--manifest", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--data-dir", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.error("a command is required")
    if args.command == "prepare":
        summary = prepare_manifest(
            Path(args.data_dir),
            Path(args.manifest),
            audio_dir=Path(args.audio_dir) if args.audio_dir else None,
            audio_extension=args.audio_extension,
        )
    else:
        summary = score_results(
            Path(args.data_dir), Path(args.results), Path(args.output_dir)
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def prepare_manifest(
    data_dir, manifest_path, audio_dir=None, audio_extension=".mp3"
):
    data_dir = data_dir.resolve()
    audio_dir = (audio_dir or data_dir).resolve()
    manifest_path = manifest_path.resolve()
    audio_extension = str(audio_extension)
    if not audio_extension.startswith("."):
        audio_extension = "." + audio_extension
    records = []
    for tag_path in sorted(data_dir.glob("*.json")):
        audio_path = audio_dir / (tag_path.stem + audio_extension)
        if not audio_path.is_file():
            raise FileNotFoundError(f"paired audio missing for {tag_path.name}")
        tag = read_json(tag_path)
        sample_id = str(tag.get("sample_id") or tag_path.stem)
        if sample_id != tag_path.stem:
            raise ValueError(f"sample_id mismatch in {tag_path.name}")
        extract_speaker_tag(tag, tag_path)
        records.append(
            {
                "corpus": {
                    "dataset_name": "caption_pairs_3000_speaker_v2_eval_100",
                    "source_urls": {
                        "article": [],
                        "github": [],
                        "huggingface": [],
                        "dataset_card": [],
                    },
                    "native_metadata": {},
                },
                "sample": {
                    "sample_id": sample_id,
                    "audio": {"path": str(audio_path.resolve())},
                    "text": {"transcript": ""},
                    "native_metadata": {},
                },
            }
        )
    if not records:
        raise ValueError(f"no JSON tags found in {data_dir}")
    write_jsonl(manifest_path, records)
    return {
        "manifest": str(manifest_path),
        "sample_count": len(records),
        "audio_dir": str(audio_dir),
        "audio_extension": audio_extension,
        "ground_truth_entered_manifest": False,
    }


def score_results(data_dir, results_path, output_dir):
    data_dir = data_dir.resolve()
    results_path = results_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    for tag_path in sorted(data_dir.glob("*.json")):
        tag = read_json(tag_path)
        ground_truth[tag_path.stem] = extract_speaker_tag(tag, tag_path)

    results = {}
    with results_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"result line {line_number} has no sample_id")
            if sample_id in results:
                raise ValueError(f"duplicate result for {sample_id}")
            results[sample_id] = item

    rows = []
    for sample_id, gt in sorted(ground_truth.items()):
        result = results.get(sample_id)
        status = result.get("status") if result else "missing"
        speaker_output = {}
        if result and status == "ok":
            speaker_output = (
                result.get("evaluation_output", {}).get("speaker", {}) or {}
            )
        row = {"sample_id": sample_id, "status": status}
        for field in BOOLEAN_FIELDS + INTEGER_FIELDS + FLOAT_FIELDS:
            row[f"gt_{field}"] = gt.get(field)
            row[f"pred_{field}"] = speaker_output.get(field)
        rows.append(row)

    metrics = {
        "dataset": {
            "sample_count": len(ground_truth),
            "result_count": len(results),
            "successful_inference_count": sum(
                row["status"] == "ok" for row in rows
            ),
            "failed_or_missing_count": sum(
                row["status"] != "ok" for row in rows
            ),
            "unexpected_result_ids": sorted(set(results) - set(ground_truth)),
        },
        "boolean": {
            field: boolean_metrics(rows, field) for field in BOOLEAN_FIELDS
        },
        "integer": {
            field: numeric_metrics(rows, field, exact=True)
            for field in INTEGER_FIELDS
        },
        "float": {
            field: numeric_metrics(rows, field, exact=False)
            for field in FLOAT_FIELDS
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_predictions(output_dir / "predictions.csv", rows)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    return metrics


def boolean_metrics(rows, field):
    eligible = [row for row in rows if isinstance(row[f"gt_{field}"], bool)]
    covered = [row for row in eligible if isinstance(row[f"pred_{field}"], bool)]
    tp = sum(row[f"gt_{field}"] and row[f"pred_{field}"] for row in covered)
    tn = sum(
        not row[f"gt_{field}"] and not row[f"pred_{field}"] for row in covered
    )
    fp = sum(
        not row[f"gt_{field}"] and row[f"pred_{field}"] for row in covered
    )
    fn = sum(
        row[f"gt_{field}"] and not row[f"pred_{field}"] for row in covered
    )
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    accuracy = divide(tp + tn, len(covered))
    return {
        "eligible_count": len(eligible),
        "null_ground_truth_excluded_count": len(rows) - len(eligible),
        "predicted_count": len(covered),
        "abstained_or_failed_count": len(eligible) - len(covered),
        "coverage": divide(len(covered), len(eligible)),
        "positive_ground_truth_count": sum(row[f"gt_{field}"] for row in eligible),
        "negative_ground_truth_count": sum(
            not row[f"gt_{field}"] for row in eligible
        ),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy_on_covered": accuracy,
        "end_to_end_accuracy": divide(tp + tn, len(eligible)),
        "precision": precision,
        "recall": recall,
        "f1": harmonic_mean(precision, recall),
        "specificity": specificity,
        "balanced_accuracy": mean_defined(recall, specificity),
    }


def numeric_metrics(rows, field, exact):
    expected_type = int if exact else (int, float)
    eligible = [
        row
        for row in rows
        if isinstance(row[f"gt_{field}"], expected_type)
        and not isinstance(row[f"gt_{field}"], bool)
    ]
    covered = [
        row
        for row in eligible
        if isinstance(row[f"pred_{field}"], expected_type)
        and not isinstance(row[f"pred_{field}"], bool)
    ]
    errors = [
        float(row[f"pred_{field}"]) - float(row[f"gt_{field}"])
        for row in covered
    ]
    output = {
        "eligible_count": len(eligible),
        "null_ground_truth_excluded_count": len(rows) - len(eligible),
        "predicted_count": len(covered),
        "abstained_or_failed_count": len(eligible) - len(covered),
        "coverage": divide(len(covered), len(eligible)),
        "mae": divide(sum(abs(error) for error in errors), len(errors)),
        "mean_error": divide(sum(errors), len(errors)),
        "rmse": (
            math.sqrt(sum(error * error for error in errors) / len(errors))
            if errors
            else None
        ),
    }
    if exact:
        correct = sum(error == 0 for error in errors)
        output["exact_match_count"] = correct
        output["exact_accuracy_on_covered"] = divide(correct, len(covered))
        output["end_to_end_exact_accuracy"] = divide(correct, len(eligible))
    return output


def extract_speaker_tag(tag, path):
    annotations = tag.get("annotation")
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ValueError(f"expected one annotation in {path.name}")
    speaker = annotations[0].get("speaker")
    if not isinstance(speaker, dict):
        raise ValueError(f"speaker tag missing in {path.name}")
    return speaker


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def harmonic_mean(left, right):
    if left is None or right is None or left + right == 0:
        return None if left is None or right is None else 0.0
    return 2 * left * right / (left + right)


def mean_defined(left, right):
    values = [value for value in (left, right) if value is not None]
    return sum(values) / len(values) if values else None


def read_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        json.dump(value, sink, ensure_ascii=False, indent=2, sort_keys=True)
        sink.write("\n")
    temporary.replace(path)


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for value in values:
            sink.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            sink.write("\n")
    temporary.replace(path)


def write_predictions(path, rows):
    fieldnames = ["sample_id", "status"]
    for field in BOOLEAN_FIELDS + INTEGER_FIELDS + FLOAT_FIELDS:
        fieldnames.extend((f"gt_{field}", f"pred_{field}"))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
