#!/usr/bin/env python3
"""Generate audio-only descriptive GT for the T1-T3 benchmark manifest."""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from run_qwen3_5_omni_plus_descriptive_gt_demo import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    parse_json_object,
    read_api_key,
    redact,
    sha256_file,
    sha256_text,
    stream_with_urllib,
    validate_descriptive,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = COMPARE_ROOT / "dataset" / "benchmark" / "benchmark.manifest.jsonl"
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"
DEFAULT_OUTPUT = (
    COMPARE_ROOT / "evaluation" / "qwen3_5_omni_plus_descriptive_gt_benchmark_t1_t3"
)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_samples(manifest_path, benchmark_root):
    manifest_path = Path(manifest_path).resolve()
    benchmark_root = Path(benchmark_root).resolve()
    samples = []
    seen_ids = set()
    seen_paths = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError("invalid JSON at manifest line %d: %s" % (line_number, error))
            sample = row.get("sample") if isinstance(row, dict) else None
            if not isinstance(sample, dict):
                raise ValueError("manifest line %d lacks sample object" % line_number)
            sample_id = sample.get("sample_id")
            audio = sample.get("audio")
            audio_rel = audio.get("path") if isinstance(audio, dict) else None
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("manifest line %d has invalid sample_id" % line_number)
            if sample_id in seen_ids:
                raise ValueError("duplicate sample_id: %s" % sample_id)
            if not isinstance(audio_rel, str) or not audio_rel:
                raise ValueError("manifest line %d has invalid audio path" % line_number)
            candidate = Path(audio_rel)
            audio_path = (candidate if candidate.is_absolute() else benchmark_root / candidate).resolve()
            if audio_path != benchmark_root and benchmark_root not in audio_path.parents:
                raise ValueError("audio path escapes benchmark root: %s" % audio_path)
            if not audio_path.is_file():
                raise ValueError("audio file missing: %s" % audio_path)
            if audio_path in seen_paths:
                raise ValueError("duplicate audio path: %s" % audio_path)
            benchmark = row.get("benchmark") if isinstance(row, dict) else None
            track = benchmark.get("track") if isinstance(benchmark, dict) else None
            samples.append(
                {
                    "sample_id": sample_id,
                    "audio_path": audio_path,
                    "track": track,
                    "manifest_line": line_number,
                }
            )
            seen_ids.add(sample_id)
            seen_paths.add(audio_path)
    if not samples:
        raise ValueError("manifest contains no samples")
    return samples


def read_success_item(path, sample, prompt_hash, model):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("status") != "success":
        return None
    if value.get("sample_id") != sample["sample_id"]:
        return None
    if value.get("audio_sha256") != sha256_file(sample["audio_path"]):
        return None
    if value.get("prompt_sha256") != prompt_hash or value.get("model") != model:
        return None
    try:
        validate_descriptive({"descriptive": value["descriptive"]})
    except (KeyError, ValueError, TypeError):
        return None
    return value


def process_sample(sample, api_key, prompt_text, prompt_hash, args):
    audio_path = sample["audio_path"]
    audio_sha256 = sha256_file(audio_path)
    last_error = "unknown error"
    total_attempts = args.retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            raw_text, usage = stream_with_urllib(
                args.base_url,
                api_key,
                prompt_text,
                audio_path.read_bytes(),
                args.model,
                args.max_tokens,
            )
            raw_text = raw_text.strip()
            if not raw_text:
                raise ValueError("empty model response")
            parsed = parse_json_object(raw_text)
            validate_descriptive(parsed)
            return {
                "status": "success",
                "sample_id": sample["sample_id"],
                "track": sample["track"],
                "manifest_line": sample["manifest_line"],
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha256,
                "model": args.model,
                "base_url": args.base_url,
                "transport": "urllib-fallback",
                "prompt_sha256": prompt_hash,
                "attempts": attempt,
                "usage": usage,
                "raw_text": raw_text,
                "descriptive": parsed["descriptive"],
            }
        except Exception as error:
            last_error = redact(error, api_key)
            if attempt < total_attempts:
                delay = min(60, 2 ** min(attempt - 1, 5)) + random.random()
                time.sleep(delay)
    return {
        "status": "failed",
        "sample_id": sample["sample_id"],
        "track": sample["track"],
        "manifest_line": sample["manifest_line"],
        "audio_path": str(audio_path),
        "audio_sha256": audio_sha256,
        "model": args.model,
        "base_url": args.base_url,
        "transport": "urllib-fallback",
        "prompt_sha256": prompt_hash,
        "attempts": total_attempts,
        "error": last_error,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 0 or args.max_tokens < 1:
        parser.error("workers/retries/max-tokens values are invalid")

    manifest_path = args.manifest.resolve()
    benchmark_root = args.benchmark_root.resolve()
    prompt_path = args.prompt.resolve()
    output_dir = args.output_dir.resolve()
    if not manifest_path.is_file():
        parser.error("manifest does not exist: %s" % manifest_path)
    if not prompt_path.is_file():
        parser.error("prompt does not exist: %s" % prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = sha256_text(prompt_text)
    samples = load_samples(manifest_path, benchmark_root)
    api_key = os.getenv("DASHSCOPE_API_KEY") or read_api_key(args.api_file.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_root": str(benchmark_root),
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "retries": args.retries,
        "max_tokens": args.max_tokens,
        "input_contract": "system_prompt_plus_one_audio",
        "sample_count": len(samples),
    }
    atomic_write_json(output_dir / "run_config.json", config)

    records = {}
    pending = []
    skipped = 0
    for sample in samples:
        item_path = items_dir / (sample["sample_id"] + ".json")
        existing = None if args.overwrite else read_success_item(item_path, sample, prompt_hash, args.model)
        if existing is not None:
            records[sample["sample_id"]] = existing
            skipped += 1
        else:
            pending.append((sample, item_path))

    print(
        "samples=%d pending=%d skipped_success=%d workers=%d"
        % (len(samples), len(pending), skipped, args.workers),
        file=sys.stderr,
    )
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_sample, sample, api_key, prompt_text, prompt_hash, args): (
                    sample,
                    item_path,
                )
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
                        % (completed, len(pending), sample["sample_id"], result.get("status")),
                        file=sys.stderr,
                    )

    ordered = [records[sample["sample_id"]] for sample in samples if sample["sample_id"] in records]
    results_text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in ordered
    )
    atomic_write(output_dir / "results.jsonl", results_text)
    success_count = sum(1 for item in ordered if item.get("status") == "success")
    failed_count = len(ordered) - success_count
    summary = {
        "sample_count": len(samples),
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_success_count": skipped,
        "pending_count_at_start": len(pending),
        "workers": args.workers,
        "model": args.model,
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "manifest": str(manifest_path),
        "results": str(output_dir / "results.jsonl"),
        "failed_sample_ids": [item["sample_id"] for item in ordered if item.get("status") != "success"],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
