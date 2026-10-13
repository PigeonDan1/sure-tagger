#!/usr/bin/env python3
"""Rescore saved descriptive demo inputs with the current semantic prompt."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from extract_captioner_deepseek_once import call_api, endpoint, read_api_file


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT = SCRIPT_DIR / "deepseek_descriptive_semantic_scoring_prompt.txt"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "deepseek-api.txt"
DEFAULT_SOURCE_DIRS = (
    COMPARE_ROOT
    / "evaluation"
    / "descriptive-semantic-scoring-official-assisted-demo",
    COMPARE_ROOT
    / "evaluation"
    / "descriptive-semantic-scoring-official-assisted-demo-qwen-captioner",
)
SCORED_FIELDS = (
    "caption",
    "topic",
    "background_condition",
    "dominant_content_impression",
    "clarity_intelligibility_impression",
    "background_sound_summary",
    "descriptive_phrases",
    "facets",
    "speaker_descriptions",
    "scene_hypotheses",
)


def semantic_input(value):
    """Keep the historical semantic payload while excluding speaker gender."""
    if not isinstance(value, dict):
        return value
    cleaned = dict(value)
    descriptions = value.get("speaker_descriptions")
    if isinstance(descriptions, list):
        cleaned["speaker_descriptions"] = [
            dict((key, child) for key, child in item.items() if key != "gender")
            if isinstance(item, dict) else item
            for item in descriptions
        ]
    return cleaned


class Item:
    def __init__(
        self,
        sample_id,
        source_dir,
        output_dir,
        ground_truth_path,
        prediction_path,
        old_result_path,
        ground_truth,
        prediction,
        old_result,
        user_text,
    ):
        self.sample_id = sample_id
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.ground_truth_path = ground_truth_path
        self.prediction_path = prediction_path
        self.old_result_path = old_result_path
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.old_result = old_result
        self.user_text = user_text


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def validate_input(ground_truth, prediction, path):
    allowed_gt = set(SCORED_FIELDS) | {"uncertainties"}
    if set(ground_truth) != allowed_gt:
        raise ValueError(f"{path}: ground-truth fields do not match the contract")
    if set(prediction) != set(SCORED_FIELDS):
        raise ValueError(f"{path}: prediction fields do not match the contract")


def validate_scores(response, prediction):
    if set(response) != {"scores"} or not isinstance(response["scores"], dict):
        raise ValueError("model response must contain exactly the scores object")
    scores = response["scores"]
    if set(scores) != set(SCORED_FIELDS):
        raise ValueError("model response contains an invalid score field set")
    normalized = {}
    for field in SCORED_FIELDS:
        score = scores[field]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 2
        ):
            raise ValueError(f"invalid score for {field}: {score!r}")
        score = round(float(score), 2)
        if prediction[field] is None and score != 1.0:
            raise ValueError(f"null prediction {field} must receive 1.00, got {score}")
        normalized[field] = score
    return normalized


def discover_items(source_dirs, output_suffix):
    items = []
    for source_dir in source_dirs:
        source_dir = source_dir.resolve()
        if not source_dir.is_dir():
            raise ValueError(f"source directory not found: {source_dir}")
        output_dir = source_dir.with_name(source_dir.name + output_suffix)
        gt_paths = sorted(source_dir.glob("*_ground_truth_input.json"))
        if not gt_paths:
            raise ValueError(f"no ground-truth inputs found in {source_dir}")
        for gt_path in gt_paths:
            sample_id = gt_path.name[: -len("_ground_truth_input.json")]
            prediction_paths = sorted(
                path
                for path in source_dir.glob(sample_id + "_*_input.json")
                if path != gt_path
            )
            if len(prediction_paths) != 1:
                raise ValueError(
                    f"expected one prediction input for {sample_id}, found "
                    f"{len(prediction_paths)}"
                )
            old_result_path = source_dir / f"{sample_id}.json"
            ground_truth = load_json(gt_path)
            prediction = load_json(prediction_paths[0])
            old_result = load_json(old_result_path)
            validate_input(ground_truth, prediction, gt_path)
            payload = {
                "ground_truth": semantic_input(ground_truth),
                "prediction": semantic_input(prediction),
            }
            user_text = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            items.append(
                Item(
                    sample_id=sample_id,
                    source_dir=source_dir,
                    output_dir=output_dir,
                    ground_truth_path=gt_path,
                    prediction_path=prediction_paths[0],
                    old_result_path=old_result_path,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    old_result=old_result,
                    user_text=user_text,
                )
            )
    return items


def system_name(item, model):
    existing = item.old_result.get("system")
    if existing:
        return existing
    if "sure_tagger" in item.prediction_path.name:
        return "sure-tagger+" + model
    if "qwen_captioner" in item.prediction_path.name:
        return "qwen-captioner+" + model
    return "descriptive-prediction+" + model


def score_item(
    item,
    config,
    prompt,
    prompt_path,
    timeout,
    max_tokens,
):
    started = time.time()
    response, usage = call_api(
        config["api_key"],
        endpoint(config["base_url"]),
        config["model"],
        prompt,
        item.user_text,
        timeout,
        max_tokens,
        "enabled",
    )
    scores = validate_scores(response, item.prediction)
    score_sum = round(sum(scores.values()), 6)
    old_scores = item.old_result.get("scores") or {}
    score_deltas = {
        field: round(scores[field] - float(old_scores[field]), 2)
        for field in SCORED_FIELDS
    }
    old_percent = float(item.old_result["normalized_percent"])
    normalized_percent = round(score_sum / (2 * len(SCORED_FIELDS)) * 100, 4)
    return {
        "status": "success",
        "sample_id": item.sample_id,
        "system": system_name(item, config["model"]),
        "model": config["model"],
        "endpoint": endpoint(config["base_url"]),
        "temperature": 0,
        "thinking": "enabled",
        "elapsed_sec": round(time.time() - started, 3),
        "usage": usage,
        "scoring_prompt": str(prompt_path),
        "prompt_sha256": sha256_text(prompt),
        "input_sha256": sha256_text(item.user_text),
        "input_contract": item.old_result.get("input_contract"),
        "ground_truth_source": item.old_result.get("ground_truth_source"),
        "prediction_source": item.old_result.get("prediction_source"),
        "prediction_source_line": item.old_result.get("prediction_source_line"),
        "score_mean": round(score_sum / len(SCORED_FIELDS), 6),
        "score_sum": score_sum,
        "normalized_percent": normalized_percent,
        "scores": scores,
        "previous_result": str(item.old_result_path),
        "previous_prompt_sha256": item.old_result.get("prompt_sha256"),
        "previous_normalized_percent": old_percent,
        "normalized_percent_delta": round(normalized_percent - old_percent, 4),
        "score_deltas": score_deltas,
    }


def write_input_copies(item):
    atomic_write_json(item.output_dir / item.ground_truth_path.name, item.ground_truth)
    atomic_write_json(item.output_dir / item.prediction_path.name, item.prediction)


def write_summaries(
    items,
    records,
    prompt,
    prompt_path,
    config,
):
    for output_dir in sorted({item.output_dir for item in items}):
        source_dir = next(
            item.source_dir for item in items if item.output_dir == output_dir
        )
        selected = sorted(
            (
                record
                for record in records
                if Path(record["previous_result"]).parent == source_dir
            ),
            key=lambda record: record["sample_id"],
        )
        if not selected:
            continue
        old_mean = sum(record["previous_normalized_percent"] for record in selected) / len(selected)
        new_mean = sum(record["normalized_percent"] for record in selected) / len(selected)
        field_means = {
            field: round(sum(record["scores"][field] for record in selected) / len(selected), 6)
            for field in SCORED_FIELDS
        }
        summary = {
            "status": "success",
            "sample_count": len(selected),
            "success_count": len(selected),
            "sample_ids": [record["sample_id"] for record in selected],
            "system": selected[0]["system"],
            "model": config["model"],
            "temperature": 0,
            "thinking": "enabled",
            "scoring_prompt": str(prompt_path),
            "prompt_sha256": sha256_text(prompt),
            "previous_prompt_sha256": selected[0]["previous_prompt_sha256"],
            "score_mean": round(new_mean / 50, 6),
            "normalized_percent": round(new_mean, 4),
            "previous_normalized_percent": round(old_mean, 4),
            "normalized_percent_delta": round(new_mean - old_mean, 4),
            "field_mean_scores": field_means,
            "samples": [
                {
                    "sample_id": record["sample_id"],
                    "score_sum": record["score_sum"],
                    "score_mean": record["score_mean"],
                    "normalized_percent": record["normalized_percent"],
                    "previous_normalized_percent": record["previous_normalized_percent"],
                    "normalized_percent_delta": record["normalized_percent_delta"],
                }
                for record in selected
            ],
        }
        atomic_write_json(output_dir / "summary.json", summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-suffix", default="-prompt-v2")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dirs = args.source_dir or list(DEFAULT_SOURCE_DIRS)
    if not args.output_suffix or "/" in args.output_suffix:
        raise SystemExit("output suffix must be a non-empty filename suffix")
    if args.workers < 1 or args.timeout < 1 or args.max_tokens < 1:
        raise SystemExit("workers, timeout, and max-tokens must be positive")
    items = discover_items(source_dirs, args.output_suffix)
    prompt_path = args.prompt.resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    config = read_api_file(args.api_file.resolve())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "item_count": len(items),
                    "prompt_sha256": sha256_text(prompt),
                    "output_dirs": sorted({str(item.output_dir) for item in items}),
                },
                indent=2,
            )
        )
        return 0

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                score_item,
                item,
                config,
                prompt,
                prompt_path,
                args.timeout,
                args.max_tokens,
            ): item
            for item in items
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            record = future.result()
            write_input_copies(item)
            atomic_write_json(item.output_dir / f"{item.sample_id}.json", record)
            records.append(record)
            print(
                f"{item.source_dir.name}/{item.sample_id}: "
                f"{record['previous_normalized_percent']:.2f}% -> "
                f"{record['normalized_percent']:.2f}%"
            )
    write_summaries(items, records, prompt, prompt_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
