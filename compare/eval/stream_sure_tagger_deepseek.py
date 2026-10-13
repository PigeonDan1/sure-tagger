#!/usr/bin/env python3
"""Stream completed Sure-Tagger speaker artifacts into DeepSeek."""

from __future__ import print_function

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = COMPARE_ROOT.parent
SURE_TAGGER_ROOT = WORKSPACE_ROOT / "sure-tagger-main"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SURE_TAGGER_ROOT) not in sys.path:
    sys.path.insert(0, str(SURE_TAGGER_ROOT))

import merge_sure_tagger_descriptive as merge_descriptive  # noqa: E402
import run_deepseek_batch as batch  # noqa: E402
import score_benchmark_v1 as benchmark_score  # noqa: E402
from tagger.tools.language_content import deterministic  # noqa: E402


DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "evaluation" / "benchmark-v1"
DEFAULT_RUN_ROOT = (
    DEFAULT_BENCHMARK_ROOT / "sure_tagger_asr_consensus_original_chain_20260908"
)
DEFAULT_FINAL_ROOT = (
    DEFAULT_BENCHMARK_ROOT
    / "sure_tagger_asr_consensus_original_chain_final_20260908"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_BENCHMARK_ROOT
    / "sure_tagger_asr_consensus_refresh_20260908_deepseek"
)
DEFAULT_RUNTIME_MANIFEST = DEFAULT_BENCHMARK_ROOT / "runtime.manifest.jsonl"
DEFAULT_PROMPT = SCRIPT_DIR / "sure_tagger_deepseek_descriptive_prompt.txt"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "deepseek-api.txt"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank line at %s:%d" % (path, line_number))
            rows.append(json.loads(line))
    return rows


def read_json_gz(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def load_inputs(run_root, runtime_manifest, gpu_count):
    runtime_rows = read_jsonl(runtime_manifest)
    runtime_ids = [row["sample"]["sample_id"] for row in runtime_rows]
    if len(runtime_ids) != len(set(runtime_ids)):
        raise ValueError("runtime manifest contains duplicate sample IDs")

    baseline_by_id = {}
    for gpu in range(gpu_count):
        name = "%02d" % gpu
        manifests = read_jsonl(run_root / "manifests" / ("manifest_%s.jsonl" % name))
        baselines = read_jsonl(run_root / "baseline" / ("tags_%s.jsonl" % name))
        if len(manifests) != len(baselines):
            raise ValueError("manifest/baseline count mismatch for shard %s" % name)
        for manifest, tags in zip(manifests, baselines):
            sample_id = manifest["sample"]["sample_id"]
            if sample_id in baseline_by_id:
                raise ValueError("duplicate prepared sample ID: %s" % sample_id)
            baseline_by_id[sample_id] = tags
    if set(runtime_ids) != set(baseline_by_id):
        raise ValueError("prepared shards do not match the runtime manifest")
    return runtime_ids, baseline_by_id


def speaker_present(speaker_count):
    if isinstance(speaker_count, bool) or not isinstance(speaker_count, int):
        return None
    return speaker_count > 0


def derive_tags(tags):
    change_count = tags["speaker"].get("speaker_change_count")
    duration_sec = tags["basic_acoustic"].get("duration_sec")
    if (
        isinstance(change_count, int)
        and not isinstance(change_count, bool)
        and change_count >= 0
        and isinstance(duration_sec, (int, float))
        and not isinstance(duration_sec, bool)
        and duration_sec > 0
    ):
        change_rate = round(float(change_count) * 60.0 / float(duration_sec), 6)
    else:
        change_rate = None
    filler = tags["language_content"].get("filler")
    word_count = tags["language_content"].get("word_count")
    if (
        isinstance(filler, int)
        and not isinstance(filler, bool)
        and filler >= 0
        and isinstance(word_count, int)
        and not isinstance(word_count, bool)
        and word_count > 0
    ):
        filler_ratio = round(float(filler) / float(word_count), 6)
    else:
        filler_ratio = None
    tags["deterministic_derived"]["speaker_change_rate_per_min"] = change_rate
    tags["deterministic_derived"]["filler_ratio"] = filler_ratio


def reconstruct_tags(evaluation_path, baseline_tags):
    evaluation = read_json_gz(evaluation_path)
    fusion = read_json_gz(evaluation_path.parent / "fusion_artifact_v2.json.gz")
    sample_id = evaluation["sample_id"]
    if fusion.get("sample_id") != sample_id:
        raise ValueError("evaluation/fusion sample ID mismatch for %s" % sample_id)

    tags = copy.deepcopy(baseline_tags)
    observed_speaker = evaluation.get("output", {}).get("speaker")
    if not isinstance(observed_speaker, dict):
        raise ValueError("evaluation output has no speaker object for %s" % sample_id)
    # This refresh changes the ASR text only.  The public speaker timeline
    # fields must remain owned by the latest original-chain result: the
    # MOSS/DiariZen consensus is an ASR routing signal, not a new speaker
    # count/overlap/change/profile decision.
    speaker = tags["speaker"]

    route = fusion.get("asr_route", {})
    selected_model = route.get("selected_model")
    candidate = fusion.get("asr_candidates", {}).get(selected_model, {})
    transcript = candidate.get("text") if route.get("selected_source") else ""
    if not isinstance(transcript, str):
        transcript = ""
    transcript = transcript.strip()
    speaker["asr_transcript"] = transcript or None

    language = route.get("detected_language")
    tags["language_content"]["language"] = (
        language.strip() if isinstance(language, str) and language.strip() else None
    )
    if transcript:
        for result in deterministic.run_all(transcript):
            field = result.tag_path.split(".", 1)[1]
            tags["language_content"][field] = result.value
    else:
        for field in ("word_count", "punctuation", "repetition", "filler"):
            tags["language_content"][field] = None
    derive_tags(tags)

    canonical = batch._validate_and_canonicalize_structural(  # noqa: SLF001
        {"tags": tags}, sample_id
    )
    return sample_id, tags, canonical


def make_item(sample_id, canonical, source_index):
    input_text = batch.sure_once.descriptive_input(canonical)
    return batch.BatchItem(
        sample_id,
        input_text=input_text,
        source_index=source_index,
        source_hash=batch.sha256_text(input_text),
    )


def load_final_rows(path):
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        tags = row.get("tags")
        if not isinstance(sample_id, str) or not isinstance(tags, dict):
            raise ValueError("invalid finalized structural row")
        if sample_id in result:
            raise ValueError("duplicate finalized sample ID: %s" % sample_id)
        result[sample_id] = tags
    return result


def process(args):
    run_root = args.run_root.resolve()
    final_root = args.final_root.resolve()
    output_root = args.output_dir.resolve()
    runtime_ids, baseline_by_id = load_inputs(
        run_root, args.runtime_manifest.resolve(), args.gpu_count
    )
    source_indices = {
        sample_id: index for index, sample_id in enumerate(runtime_ids, 1)
    }

    prompt = args.prompt.resolve().read_text(encoding="utf-8")
    prompt_hash = batch.sha256_text(prompt)
    config = batch.captioner_once.read_api_file(args.api_file.resolve())
    fingerprint_value = {
        "system": "sure-tagger",
        "model": config["model"],
        "endpoint": batch.captioner_once.endpoint(config["base_url"]),
        "prompt_sha256": prompt_hash,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    config_fingerprint = batch.sha256_text(
        json.dumps(fingerprint_value, sort_keys=True, separators=(",", ":"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    previous_config_path = output_root / "run_config.json"
    if previous_config_path.is_file():
        previous = json.loads(previous_config_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != config_fingerprint:
            raise ValueError("existing DeepSeek run uses a different request fingerprint")

    run_meta = {
        "input_manifest": str((final_root / "result_with_id.jsonl").resolve()),
        "input_contract": "one_upstream_text_value_per_request",
        "workers": args.workers,
        "retries": args.retries,
        "timeout_sec": args.timeout,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "source_count": len(runtime_ids),
        "api_request_count": 0,
        "resume_skipped_count": 0,
        "streamed_from_speaker_artifacts": True,
    }
    batch.atomic_write_json(
        previous_config_path,
        dict(run_meta, **fingerprint_value, config_fingerprint=config_fingerprint),
    )

    checkpoints = batch._load_checkpoints(output_root)  # noqa: SLF001
    items = {}
    native_tags = {}
    records = {}
    futures = {}
    seen_artifacts = set()
    last_status = 0.0
    final_path = final_root / "result_with_id.jsonl"
    # The finalized structural result is the authoritative public input for
    # descriptive tagging.  Speaker artifacts can contain an earlier
    # baseline/profile snapshot (the ASR routing artifact is not itself the
    # finalized structural output), so prefer final rows whenever they are
    # already available.
    authoritative_final_rows = None
    if final_path.is_file():
        authoritative_final_rows = load_final_rows(final_path)
        if set(authoritative_final_rows) != set(runtime_ids):
            raise ValueError("finalized structural output has incomplete coverage")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        while len(records) < len(runtime_ids):
            discovered = 0
            artifact_pattern = "shard_*/speaker_v2/*/evaluation_output.json.gz"
            for evaluation_path in sorted((run_root / "artifacts").glob(artifact_pattern)):
                path_key = str(evaluation_path)
                if path_key in seen_artifacts:
                    continue
                try:
                    evaluation = read_json_gz(evaluation_path)
                    sample_id = evaluation.get("sample_id")
                    if sample_id not in baseline_by_id:
                        raise ValueError("unexpected artifact sample ID: %s" % sample_id)
                    sample_id, tags, canonical = reconstruct_tags(
                        evaluation_path, baseline_by_id[sample_id]
                    )
                    if authoritative_final_rows is not None:
                        tags = authoritative_final_rows[sample_id]
                        canonical = batch._validate_and_canonicalize_structural(  # noqa: SLF001
                            {"tags": tags}, sample_id
                        )
                except (EOFError, OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                seen_artifacts.add(path_key)
                if sample_id in items:
                    continue
                item = make_item(sample_id, canonical, source_indices[sample_id])
                items[sample_id] = item
                native_tags[sample_id] = tags
                old = checkpoints.get(sample_id)
                if old is not None and batch._checkpoint_is_reusable(  # noqa: SLF001
                    old, item, prompt_hash, config_fingerprint, True
                ):
                    records[sample_id] = old
                    run_meta["resume_skipped_count"] += 1
                else:
                    future = executor.submit(
                        batch.process_item,
                        item,
                        "sure-tagger",
                        config,
                        prompt,
                        prompt_hash,
                        config_fingerprint,
                        args.timeout,
                        args.retries,
                        args.max_tokens,
                        args.thinking,
                        args.backoff_base,
                        args.backoff_max,
                        True,
                    )
                    futures[future] = sample_id
                batch.atomic_write_json(
                    output_root / "structural_items" / (batch._item_filename(sample_id)),  # noqa: SLF001
                    {"sample_id": sample_id, "tags": tags},
                )
                discovered += 1

            if final_path.is_file() and len(items) < len(runtime_ids):
                final_rows = authoritative_final_rows or load_final_rows(final_path)
                if set(final_rows) != set(runtime_ids):
                    raise ValueError("finalized structural output has incomplete coverage")
                authoritative_final_rows = final_rows
                for sample_id in runtime_ids:
                    if sample_id in items:
                        continue
                    tags = final_rows[sample_id]
                    canonical = batch._validate_and_canonicalize_structural(  # noqa: SLF001
                        {"tags": tags}, sample_id
                    )
                    item = make_item(sample_id, canonical, source_indices[sample_id])
                    items[sample_id] = item
                    native_tags[sample_id] = tags
                    old = checkpoints.get(sample_id)
                    if old is not None and batch._checkpoint_is_reusable(  # noqa: SLF001
                        old, item, prompt_hash, config_fingerprint, True
                    ):
                        records[sample_id] = old
                        run_meta["resume_skipped_count"] += 1
                    else:
                        future = executor.submit(
                            batch.process_item,
                            item,
                            "sure-tagger",
                            config,
                            prompt,
                            prompt_hash,
                            config_fingerprint,
                            args.timeout,
                            args.retries,
                            args.max_tokens,
                            args.thinking,
                            args.backoff_base,
                            args.backoff_max,
                            True,
                        )
                        futures[future] = sample_id
                    discovered += 1

            for future in list(futures):
                if not future.done():
                    continue
                sample_id = futures.pop(future)
                record = future.result()
                batch._write_checkpoint(output_root, record)  # noqa: SLF001
                records[sample_id] = record
                run_meta["api_request_count"] += record.get("attempts", 0)
                print(
                    "deepseek %d/%d %s %s"
                    % (len(records), len(runtime_ids), sample_id, record["status"]),
                    flush=True,
                )

            now = time.time()
            if discovered or now - last_status >= args.status_interval:
                print(
                    "stream artifacts=%d submitted=%d completed=%d inflight=%d workers=%d"
                    % (
                        len(seen_artifacts),
                        len(items),
                        len(records),
                        len(futures),
                        args.workers,
                    ),
                    flush=True,
                )
                last_status = now
            if len(records) < len(runtime_ids):
                time.sleep(args.poll_interval)
    finally:
        executor.shutdown(wait=True)

    while not final_path.is_file():
        print("waiting for finalized structural output", flush=True)
        time.sleep(args.poll_interval)
    final_rows = load_final_rows(final_path)
    if set(final_rows) != set(runtime_ids):
        raise ValueError("finalized structural output does not cover the runtime")
    mismatches = [
        sample_id
        for sample_id in runtime_ids
        if native_tags.get(sample_id) != final_rows[sample_id]
    ]
    if mismatches:
        raise ValueError(
            "stream reconstruction differs from final output for %d samples"
            % len(mismatches)
        )

    input_lines = [
        json.dumps(
            {"sample_id": sample_id, "tags": final_rows[sample_id]},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for sample_id in runtime_ids
    ]
    input_path = output_root / "stream_input.jsonl"
    batch.atomic_write_text(input_path, "".join(input_lines))
    run_meta["input_manifest_sha256"] = hashlib.sha256(
        final_path.read_bytes()
    ).hexdigest()
    ordered_items = [items[sample_id] for sample_id in runtime_ids]
    ordered_records = [records[sample_id] for sample_id in runtime_ids]
    summary = batch._aggregate(  # noqa: SLF001
        ordered_items,
        ordered_records,
        output_root,
        "sure-tagger",
        config,
        prompt_hash,
        config_fingerprint,
        run_meta,
    )
    batch.atomic_write_json(
        output_root / "run_config.json",
        dict(
            run_meta,
            **fingerprint_value,
            config_fingerprint=config_fingerprint,
            success_count=summary["success_count"],
            failed_count=summary["failed_count"],
            reconstruction_matches_final=True,
        ),
    )
    if summary["failed_count"]:
        raise ValueError("DeepSeek failed for %d samples" % summary["failed_count"])

    merge_summary = merge_descriptive.merge(
        final_path, output_root / "results.jsonl", output_root
    )
    merged_path = Path(merge_summary["result_with_id_descriptive"])
    score = benchmark_score.score(
        args.benchmark_root.resolve(),
        merged_path,
        system="sure-tagger-asr-consensus-refresh+deepseek-v4-pro",
        structural_prediction_owner="sure-tagger",
    )
    benchmark_score.atomic_write(
        output_root / "score.json",
        json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    benchmark_score.atomic_write(
        output_root / "score.md", benchmark_score.render_report(score)
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST
    )
    parser.add_argument("--benchmark-root", type=Path, default=benchmark_score.BENCHMARK_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--gpu-count", type=int, default=7)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--backoff-max", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.workers != 20:
        raise SystemExit("this run requires exactly 20 DeepSeek workers")
    return process(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("ERROR: %s" % batch.redact(exc), file=sys.stderr)
        raise SystemExit(1)
