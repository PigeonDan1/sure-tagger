#!/usr/bin/env python3
"""Generate official-GT-assisted descriptive references for all T1-T3 audio."""

from __future__ import print_function

import argparse
import concurrent.futures
import json
import os
import sys
from collections import Counter
from pathlib import Path

from run_qwen3_5_omni_plus_descriptive_gt_batch import (
    atomic_write,
    atomic_write_json,
    load_samples,
)
from run_qwen3_5_omni_plus_descriptive_gt_demo import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    read_api_key,
    redact,
    sha256_file,
    sha256_text,
)
from run_qwen3_5_omni_plus_descriptive_gt_official_assisted_demo import (
    DEFAULT_API_FILE,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_PROMPT,
    REFERENCE_CONTRACT_VERSION,
    process_sample,
    reference_payload,
    validate_assisted_descriptive,
)
from score_benchmark_v1 import BENCHMARK_VERSION, SCORING_VERSION, build_ground_truth


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = (
    COMPARE_ROOT
    / "evaluation"
    / "qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3"
)


def canonical_reference_text(payload):
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def read_success_item(path, sample, gt_record, prompt_hash, model):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("status") != "success":
        return None
    if value.get("sample_id") != sample["sample_id"]:
        return None
    if value.get("audio_sha256") != sha256_file(sample["audio_path"]):
        return None
    if value.get("prompt_sha256") != prompt_hash or value.get("model") != model:
        return None
    expected_payload = reference_payload(gt_record)
    expected_hash = sha256_text(canonical_reference_text(expected_payload))
    if value.get("reference_facts_sha256") != expected_hash:
        return None
    if value.get("reference_facts") != expected_payload:
        return None
    try:
        validate_assisted_descriptive(
            {"descriptive": value["descriptive"]}, expected_payload["facts"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sanitize-invalid-output", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 0 or args.max_tokens < 1:
        parser.error("workers/retries/max-tokens values are invalid")

    benchmark_root = args.benchmark_root.resolve()
    manifest_path = args.manifest.resolve()
    prompt_path = args.prompt.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest_path.is_file():
        parser.error("manifest does not exist: %s" % manifest_path)
    if not prompt_path.is_file():
        parser.error("prompt does not exist: %s" % prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        parser.error("prompt file is empty")
    prompt_hash = sha256_text(prompt_text)

    samples = load_samples(manifest_path, benchmark_root)
    ground_truth = build_ground_truth(benchmark_root)
    sample_ids = set(sample["sample_id"] for sample in samples)
    if sample_ids != set(ground_truth):
        raise ValueError("manifest and canonical GT sample IDs differ")
    api_key = os.getenv("DASHSCOPE_API_KEY") or read_api_key(args.api_file.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    track_counts = Counter(sample["track"] for sample in samples)
    config = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_root": str(benchmark_root),
        "benchmark_version": BENCHMARK_VERSION,
        "scoring_version": SCORING_VERSION,
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "retries": args.retries,
        "max_tokens": args.max_tokens,
        "sanitize_invalid_output": args.sanitize_invalid_output,
        "input_contract": (
            "system_prompt_plus_official_reference_facts_plus_one_audio"
        ),
        "reference_contract_version": REFERENCE_CONTRACT_VERSION,
        "reference_builder": "score_benchmark_v1.build_ground_truth",
        "sample_count": len(samples),
        "track_counts": dict(sorted(track_counts.items())),
    }
    atomic_write_json(output_dir / "run_config.json", config)

    records = {}
    pending = []
    skipped = 0
    for sample in samples:
        item_path = items_dir / (sample["sample_id"] + ".json")
        existing = None
        if not args.overwrite:
            existing = read_success_item(
                item_path,
                sample,
                ground_truth[sample["sample_id"]],
                prompt_hash,
                args.model,
            )
        if existing is None:
            pending.append((sample, item_path))
        else:
            records[sample["sample_id"]] = existing
            skipped += 1

    print(
        "samples=%d pending=%d skipped_success=%d workers=%d"
        % (len(samples), len(pending), skipped, args.workers),
        file=sys.stderr,
    )
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_sample,
                    sample,
                    ground_truth[sample["sample_id"]],
                    api_key,
                    prompt_text,
                    prompt_hash,
                    args,
                ): (sample, item_path)
                for sample, item_path in pending
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                sample, item_path = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "failed",
                        "sample_id": sample["sample_id"],
                        "track": sample["track"],
                        "manifest_line": sample["manifest_line"],
                        "audio_path": str(sample["audio_path"]),
                        "error": redact(error, api_key),
                    }
                records[sample["sample_id"]] = result
                atomic_write_json(item_path, result)
                completed += 1
                if completed % 5 == 0 or result.get("status") != "success":
                    print(
                        "progress=%d/%d sample_id=%s status=%s"
                        % (
                            completed,
                            len(pending),
                            sample["sample_id"],
                            result.get("status"),
                        ),
                        file=sys.stderr,
                    )

    ordered = [records[sample["sample_id"]] for sample in samples]
    results_text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in ordered
    )
    atomic_write(output_dir / "results.jsonl", results_text)
    success_count = sum(1 for item in ordered if item.get("status") == "success")
    failed_ids = [
        item["sample_id"] for item in ordered if item.get("status") != "success"
    ]
    summary = {
        "sample_count": len(samples),
        "success_count": success_count,
        "failed_count": len(failed_ids),
        "failed_sample_ids": failed_ids,
        "skipped_success_count": skipped,
        "pending_count_at_start": len(pending),
        "workers": args.workers,
        "model": args.model,
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "reference_contract_version": REFERENCE_CONTRACT_VERSION,
        "manifest": str(manifest_path),
        "results": str(output_dir / "results.jsonl"),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not failed_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
