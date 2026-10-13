#!/usr/bin/env python3
"""Score captioner descriptions as official-assisted GT checkpoints arrive."""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from extract_captioner_deepseek_once import call_api, endpoint, read_api_file


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT = SCRIPT_DIR / "deepseek_descriptive_semantic_scoring_prompt.txt"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "deepseek-api.txt"
DEFAULT_GT_DIR = (
    COMPARE_ROOT
    / "evaluation"
    / "qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3"
)
DEFAULT_PREDICTIONS = (
    COMPARE_ROOT / "docs" / "report" / "result" / "qwen-captioner" / "results_t1_t3.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    COMPARE_ROOT
    / "evaluation"
    / "descriptive-semantic-scoring-official-assisted-benchmark-qwen-captioner"
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
SCALAR_FIELDS = set(SCORED_FIELDS[:6])
INPUT_CONTRACT = {
    "excluded_everywhere": ["confidence", "evidence_tags", "usability_notes"],
    "speaker_gender_excluded": True,
    "ground_truth_uncertainties_included": True,
    "prediction_uncertainties_included": False,
    "scored_field_count": len(SCORED_FIELDS),
}


class BatchInputError(ValueError):
    pass


class ScoreItem:
    def __init__(
        self,
        sample_id,
        ground_truth,
        prediction,
        ground_truth_path,
        prediction_path,
        prediction_line,
    ):
        self.sample_id = sample_id
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.ground_truth_path = ground_truth_path
        self.prediction_path = prediction_path
        self.prediction_line = prediction_line
        self.user_text = json.dumps(
            {"ground_truth": ground_truth, "prediction": prediction},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.input_sha256 = sha256_text(self.user_text)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchInputError("%s must contain a JSON object" % path)
    return value


def _optional_text(value, path):
    if value is not None and not isinstance(value, str):
        raise BatchInputError("%s must be a string or null" % path)
    return value


def _optional_text_list(value, path):
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise BatchInputError("%s must be an array of strings or null" % path)
    return list(value)


def _clean_facets(value, path):
    if value is None:
        return None
    if not isinstance(value, list):
        raise BatchInputError("%s must be an array or null" % path)
    cleaned = []
    for index, item in enumerate(value):
        item_path = "%s[%d]" % (path, index)
        if not isinstance(item, dict):
            raise BatchInputError("%s must be an object" % item_path)
        cleaned.append(
            {
                "name": _optional_text(item.get("name"), item_path + ".name"),
                "description": _optional_text(
                    item.get("description"), item_path + ".description"
                ),
            }
        )
    return cleaned


def _clean_speaker_descriptions(value, path):
    if value is None:
        return None
    if not isinstance(value, list):
        raise BatchInputError("%s must be an array or null" % path)
    cleaned = []
    for index, item in enumerate(value):
        item_path = "%s[%d]" % (path, index)
        if not isinstance(item, dict):
            raise BatchInputError("%s must be an object" % item_path)
        cleaned.append(
            {
                "speaker_id": _optional_text(
                    item.get("speaker_id"), item_path + ".speaker_id"
                ),
                # Gender is a separate structural score and is intentionally
                # excluded from the semantic evaluator input.
                "description": _optional_text(
                    item.get("description"), item_path + ".description"
                ),
                "descriptive_phrases": _optional_text_list(
                    item.get("descriptive_phrases"),
                    item_path + ".descriptive_phrases",
                ),
            }
        )
    return cleaned


def _clean_scene_hypotheses(value, path):
    if value is None:
        return None
    if not isinstance(value, list):
        raise BatchInputError("%s must be an array or null" % path)
    cleaned = []
    for index, item in enumerate(value):
        item_path = "%s[%d]" % (path, index)
        if not isinstance(item, dict):
            raise BatchInputError("%s must be an object" % item_path)
        cleaned.append(
            {
                "description": _optional_text(
                    item.get("description"), item_path + ".description"
                )
            }
        )
    return cleaned


def clean_descriptive(descriptive, include_uncertainties, path="descriptive"):
    if not isinstance(descriptive, dict):
        raise BatchInputError("%s must be an object" % path)
    result = {}
    for field in SCORED_FIELDS:
        field_path = path + "." + field
        value = descriptive.get(field)
        if field in SCALAR_FIELDS:
            result[field] = _optional_text(value, field_path)
        elif field == "descriptive_phrases":
            result[field] = _optional_text_list(value, field_path)
        elif field == "facets":
            result[field] = _clean_facets(value, field_path)
        elif field == "speaker_descriptions":
            result[field] = _clean_speaker_descriptions(value, field_path)
        elif field == "scene_hypotheses":
            result[field] = _clean_scene_hypotheses(value, field_path)
    if include_uncertainties:
        result["uncertainties"] = _optional_text_list(
            descriptive.get("uncertainties"), path + ".uncertainties"
        )
    return result


def load_predictions(path):
    path = Path(path).resolve()
    predictions = {}
    order = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise BatchInputError("%s:%d: %s" % (path, line_number, error))
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
                raise BatchInputError("%s:%d has no valid sample_id" % (path, line_number))
            sample_id = row["sample_id"]
            if sample_id in predictions:
                raise BatchInputError("duplicate prediction sample_id: %s" % sample_id)
            predictions[sample_id] = {
                "descriptive": clean_descriptive(
                    row.get("descriptive"),
                    include_uncertainties=False,
                    path="%s:%d.descriptive" % (path, line_number),
                ),
                "line": line_number,
            }
            order.append(sample_id)
    if not order:
        raise BatchInputError("prediction JSONL is empty: %s" % path)
    return predictions, order


def load_ready_ground_truth(items_dir, expected_ids):
    ready = {}
    failed = {}
    for path in sorted(Path(items_dir).glob("*.json")):
        try:
            record = load_json(path)
        except (OSError, ValueError):
            continue
        sample_id = record.get("sample_id")
        if sample_id not in expected_ids:
            continue
        if record.get("status") == "success":
            try:
                descriptive = clean_descriptive(
                    record.get("descriptive"),
                    include_uncertainties=True,
                    path=str(path) + ".descriptive",
                )
            except BatchInputError as error:
                failed[sample_id] = "invalid successful GT: %s" % error
                continue
            ready[sample_id] = (path.resolve(), descriptive)
        else:
            failed[sample_id] = str(record.get("error") or "GT status is not success")
    return ready, failed


def make_score_item(sample_id, gt_entry, prediction_entry, prediction_path):
    ground_truth_path, ground_truth = gt_entry
    return ScoreItem(
        sample_id=sample_id,
        ground_truth=ground_truth,
        prediction=prediction_entry["descriptive"],
        ground_truth_path=ground_truth_path,
        prediction_path=Path(prediction_path).resolve(),
        prediction_line=prediction_entry["line"],
    )


def validate_scores(response, prediction):
    if not isinstance(response, dict) or set(response) != {"scores"}:
        raise ValueError("model response must contain exactly the scores object")
    scores = response["scores"]
    if not isinstance(scores, dict) or set(scores) != set(SCORED_FIELDS):
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
            raise ValueError("invalid score for %s: %r" % (field, score))
        score = round(float(score), 2)
        if prediction[field] is None and score != 1.0:
            raise ValueError(
                "null prediction %s must receive 1.00, got %s" % (field, score)
            )
        normalized[field] = score
    return normalized


def config_fingerprint(config, timeout, max_tokens, thinking):
    value = {
        "endpoint": endpoint(config["base_url"]),
        "model": config["model"],
        "timeout": timeout,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "temperature": 0,
    }
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def reusable_success(path, item, prompt_hash, fingerprint):
    try:
        record = load_json(path)
        if (
            record.get("status") != "success"
            or record.get("sample_id") != item.sample_id
            or record.get("prompt_sha256") != prompt_hash
            or record.get("config_fingerprint") != fingerprint
            or record.get("input_sha256") != item.input_sha256
        ):
            return None
        validate_scores({"scores": record.get("scores")}, item.prediction)
        return record
    except (OSError, ValueError):
        return None


def redact_error(error, api_key):
    return str(error).replace(api_key, "<redacted>")[:4000]


def is_rate_limit_error(error):
    text = str(error).lower()
    return (
        "429" in text
        or "rate_limit" in text
        or "gateway_queue_full" in text
        or "too many pending" in text
    )


def default_system_name(model):
    return "qwen-captioner+" + model


def load_api_config(path, model_override=None):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("API file is empty")
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("API JSON must be an object: %s" % path)
        api_key = payload.get("key") or payload.get("api_key")
        base_url = payload.get("url") or payload.get("base_url")
        model = payload.get("model")
        if not api_key:
            raise ValueError("API JSON has no key: %s" % path)
        if not base_url:
            raise ValueError("API JSON has no url: %s" % path)
        config = {
            "api_key": str(api_key),
            "base_url": str(base_url),
            "model": str(model) if model else None,
        }
    else:
        config = read_api_file(path)
    if model_override:
        config["model"] = model_override
    if not config.get("model"):
        raise ValueError("API file has no model; pass --model")
    return config


def score_item(
    item,
    config,
    prompt,
    prompt_path,
    prompt_hash,
    fingerprint,
    timeout,
    max_tokens,
    thinking,
    retries,
    system_name,
):
    started = time.time()
    errors = []
    attempt = 0
    hard_failures = 0
    rate_limit_tries = 0
    while True:
        attempt += 1
        try:
            response, usage = call_api(
                config["api_key"],
                endpoint(config["base_url"]),
                config["model"],
                prompt,
                item.user_text,
                timeout,
                max_tokens,
                thinking,
            )
            scores = validate_scores(response, item.prediction)
            score_sum = round(sum(scores.values()), 6)
            return {
                "status": "success",
                "sample_id": item.sample_id,
                "system": system_name,
                "model": config["model"],
                "endpoint": endpoint(config["base_url"]),
                "temperature": 0,
                "thinking": thinking,
                "attempts": attempt,
                "elapsed_sec": round(time.time() - started, 3),
                "usage": usage,
                "scoring_prompt": str(prompt_path),
                "prompt_sha256": prompt_hash,
                "config_fingerprint": fingerprint,
                "input_sha256": item.input_sha256,
                "input_contract": INPUT_CONTRACT,
                "ground_truth_source": str(item.ground_truth_path),
                "prediction_source": str(item.prediction_path),
                "prediction_source_line": item.prediction_line,
                "score_mean": round(score_sum / len(SCORED_FIELDS), 6),
                "score_sum": score_sum,
                "normalized_percent": round(
                    score_sum / (2 * len(SCORED_FIELDS)) * 100, 4
                ),
                "scores": scores,
            }
        except Exception as error:
            errors.append(redact_error(error, config["api_key"]))
            if is_rate_limit_error(error) and rate_limit_tries < 40:
                rate_limit_tries += 1
                time.sleep(min(8 * rate_limit_tries, 45) + random.random() * 3)
                continue
            hard_failures += 1
            if hard_failures <= retries:
                time.sleep(min(2 ** (hard_failures - 1), 8))
                continue
            break
    return {
        "status": "failed",
        "sample_id": item.sample_id,
        "system": system_name,
        "model": config["model"],
        "attempts": attempt,
        "elapsed_sec": round(time.time() - started, 3),
        "errors": errors,
        "scoring_prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "config_fingerprint": fingerprint,
        "input_sha256": item.input_sha256,
        "input_contract": INPUT_CONTRACT,
        "ground_truth_source": str(item.ground_truth_path),
        "prediction_source": str(item.prediction_path),
        "prediction_source_line": item.prediction_line,
    }


def write_aggregates(
    output_dir,
    order,
    records,
    gt_ready,
    gt_failed,
    running_ids,
    attempted,
    config,
):
    selected = [records[sample_id] for sample_id in order if sample_id in records]
    results_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in selected
    )
    atomic_write_text(output_dir / "results.jsonl", results_text)
    successful = [record for record in selected if record.get("status") == "success"]
    failed = [record for record in selected if record.get("status") != "success"]
    field_means = {}
    normalized_percent = None
    if successful:
        field_means = {
            field: round(
                sum(record["scores"][field] for record in successful) / len(successful),
                6,
            )
            for field in SCORED_FIELDS
        }
        normalized_percent = round(
            sum(record["normalized_percent"] for record in successful) / len(successful),
            4,
        )
    if len(successful) == len(order):
        status = "complete"
    elif running_ids:
        status = "running"
    elif len(gt_ready) < len(order) and not failed:
        status = "complete_with_missing_ground_truth"
    elif len(gt_ready) < len(order):
        status = "waiting_for_ground_truth"
    else:
        status = "scoring_failed"
    summary = {
        "status": status,
        "sample_count": len(order),
        "evaluated_sample_count": len(successful),
        "score_denominator_sample_count": len(successful),
        "score_aggregation": "macro_mean_over_scoring_successes",
        "ground_truth_ready_count": len(gt_ready),
        "ground_truth_failed_count": len(gt_failed),
        "ground_truth_failed_sample_ids": sorted(gt_failed),
        "excluded_missing_ground_truth_count": len(order) - len(gt_ready),
        "excluded_missing_ground_truth_sample_ids": sorted(
            set(order) - set(gt_ready)
        ),
        "scoring_success_count": len(successful),
        "scoring_failed_count": len(failed),
        "scoring_failed_sample_ids": sorted(
            record["sample_id"] for record in failed
        ),
        "scoring_attempted_count": len(attempted),
        "running_count": len(running_ids),
        "running_sample_ids": sorted(running_ids),
        "waiting_for_ground_truth_count": len(order) - len(gt_ready),
        "normalized_percent_on_successes": normalized_percent,
        "field_mean_scores_on_successes": field_means,
        "workers": config["workers"],
        "model": config["model"],
        "prompt": config["prompt"],
        "prompt_sha256": config["prompt_sha256"],
        "ground_truth_directory": config["ground_truth_directory"],
        "prediction_source": config["prediction_source"],
        "results": str((output_dir / "results.jsonl").resolve()),
        "updated_at_unix": round(time.time(), 3),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--model", help="override model name from the API file")
    parser.add_argument(
        "--system",
        help="system label written into run_config and per-sample records",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (
        args.workers < 1
        or args.timeout < 1
        or args.max_tokens < 1
        or args.retries < 0
        or args.poll_interval <= 0
    ):
        raise SystemExit("workers/timeouts/token/retry/poll values are invalid")

    gt_dir = args.ground_truth_dir.resolve()
    gt_items_dir = gt_dir / "items"
    prediction_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    prompt_path = args.prompt.resolve()
    api_file = args.api_file.resolve()
    if not gt_items_dir.is_dir():
        raise SystemExit("ground-truth items directory not found: %s" % gt_items_dir)
    for path in (prediction_path, prompt_path, api_file):
        if not path.is_file():
            raise SystemExit("required file not found: %s" % path)

    predictions, order = load_predictions(prediction_path)
    expected_ids = set(order)
    gt_run_config_path = gt_dir / "run_config.json"
    if gt_run_config_path.is_file():
        gt_run_config = load_json(gt_run_config_path)
        if gt_run_config.get("sample_count") != len(order):
            raise SystemExit("GT and captioner sample counts differ")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit("scoring prompt is empty")
    prompt_hash = sha256_text(prompt)
    try:
        config = load_api_config(api_file, args.model)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
    system_name = args.system or default_system_name(config["model"])
    fingerprint = config_fingerprint(
        config, args.timeout, args.max_tokens, args.thinking
    )
    run_config = {
        "ground_truth_directory": str(gt_dir),
        "prediction_source": str(prediction_path),
        "prediction_source_sha256": sha256_file(prediction_path),
        "sample_count": len(order),
        "system": system_name,
        "model": config["model"],
        "endpoint": endpoint(config["base_url"]),
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "config_fingerprint": fingerprint,
        "input_contract": INPUT_CONTRACT,
        "workers": args.workers,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "retries": args.retries,
        "poll_interval": args.poll_interval,
    }
    if args.dry_run:
        gt_ready, gt_failed = load_ready_ground_truth(gt_items_dir, expected_ids)
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "sample_count": len(order),
                    "ground_truth_ready_count": len(gt_ready),
                    "ground_truth_failed_count": len(gt_failed),
                    "prompt_sha256": prompt_hash,
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_items_dir = output_dir / "items"
    output_items_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run_config.json", run_config)

    records = {}
    attempted = set()
    running = {}
    last_report = 0
    last_state = None
    print(
        "samples=%d workers=%d gt_items=%s output=%s"
        % (len(order), args.workers, gt_items_dir, output_dir),
        file=sys.stderr,
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            gt_ready, gt_failed = load_ready_ground_truth(gt_items_dir, expected_ids)

            candidates = []
            running_ids = {item.sample_id for item in running.values()}
            for sample_id in order:
                if sample_id not in gt_ready or sample_id in running_ids:
                    continue
                item = make_score_item(
                    sample_id,
                    gt_ready[sample_id],
                    predictions[sample_id],
                    prediction_path,
                )
                checkpoint_path = output_items_dir / (sample_id + ".json")
                existing = reusable_success(
                    checkpoint_path, item, prompt_hash, fingerprint
                )
                if existing is not None:
                    records[sample_id] = existing
                    continue
                records.pop(sample_id, None)
                if (sample_id, item.input_sha256) not in attempted:
                    candidates.append(item)

            free_slots = args.workers - len(running)
            for item in candidates[:free_slots]:
                future = pool.submit(
                    score_item,
                    item,
                    config,
                    prompt,
                    prompt_path,
                    prompt_hash,
                    fingerprint,
                    args.timeout,
                    args.max_tokens,
                    args.thinking,
                    args.retries,
                    system_name,
                )
                running[future] = item

            running_ids = {item.sample_id for item in running.values()}
            summary_config = dict(run_config)
            summary_config["workers"] = args.workers
            summary = write_aggregates(
                output_dir,
                order,
                records,
                gt_ready,
                gt_failed,
                running_ids,
                attempted,
                summary_config,
            )
            state = (
                len(gt_ready),
                len(gt_failed),
                summary["scoring_success_count"],
                summary["scoring_failed_count"],
                len(running),
            )
            now = time.time()
            if state != last_state or now - last_report >= 60:
                print(
                    "gt_ready=%d/%d gt_failed=%d score_success=%d "
                    "score_failed=%d running=%d"
                    % (state[0], len(order), state[1], state[2], state[3], state[4]),
                    file=sys.stderr,
                    flush=True,
                )
                last_state = state
                last_report = now

            if summary["status"] in ("complete", "complete_with_missing_ground_truth"):
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                return 0
            if not running:
                if len(gt_ready) == len(order) and not candidates:
                    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                    return 2
                time.sleep(args.poll_interval)
                continue

            done, _ = concurrent.futures.wait(
                tuple(running),
                timeout=args.poll_interval,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                item = running.pop(future)
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "status": "failed",
                        "sample_id": item.sample_id,
                        "error": redact_error(error, config["api_key"]),
                        "prompt_sha256": prompt_hash,
                        "config_fingerprint": fingerprint,
                        "input_sha256": item.input_sha256,
                    }
                errors = record.get("errors") or []
                rate_limited_only = bool(errors) and all(
                    is_rate_limit_error(error) for error in errors
                )
                if not rate_limited_only:
                    attempted.add((item.sample_id, item.input_sha256))
                records[item.sample_id] = record
                atomic_write_json(
                    output_items_dir / (item.sample_id + ".json"), record
                )
                print(
                    "sample_id=%s status=%s attempts=%s score=%s"
                    % (
                        item.sample_id,
                        record.get("status"),
                        record.get("attempts"),
                        record.get("normalized_percent"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main())
