#!/usr/bin/env python3
"""Prepare the AMI-only runtime manifest and audit a fresh language run."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tagger.pipelines.speaker_evidence import (
    _selected_timeline_output,
    collect_firered_lid_evidence,
    collect_official_diarizen_evidence,
    collect_speaker_segment_lid_evidence,
)
from tagger.tools.language_content.firered_lid_detector import FireRedLidConfig
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
    normalize_timeline,
)
from tagger.tools.speaker_v2.diarizen_official import OfficialDiariZenConfig
from tagger.tools.subprocess_runner import close_subprocess_workers


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_field(rows, field):
    """Count a result field using a JSON-sortable label for missing values."""

    return dict(Counter(
        "<none>" if row.get(field) is None else str(row.get(field))
        for row in rows
    ))


def prepare(args):
    manifest = read_jsonl(args.runtime_manifest)
    index = json.loads(Path(args.staging_index).read_text(encoding="utf-8"))
    samples = index["samples"]
    if len(manifest) != len(samples):
        raise ValueError("benchmark manifest and staging index differ in length")
    selected = []
    for record, sample in zip(manifest, samples):
        if record["sample"]["sample_id"] != sample["sample_id"]:
            raise ValueError("benchmark manifest and staging index order differs")
        if "/t1-ami/" not in sample["source_audio"]:
            continue
        path = Path(record["sample"]["audio"]["path"])
        with path.open("rb") as audio:
            if not audio.read(1):
                raise ValueError("empty AMI audio: %s" % path)
        selected.append(record)
    if len(selected) != 350 or len({row["sample"]["sample_id"] for row in selected}) != 350:
        raise ValueError("expected 350 unique, readable AMI samples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite manifest: %s" % output)
    write_jsonl(output, selected)
    print("AMI manifest: %s (350 unique, readable samples)" % output)


def run(args):
    manifest = read_jsonl(args.manifest)
    ids = [record["sample"]["sample_id"] for record in manifest]
    if len(ids) != 350 or len(set(ids)) != 350:
        raise ValueError("expected 350 unique AMI samples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_jsonl(output) if output.exists() else []
    done = {row["sample_id"] for row in completed}
    if len(done) != len(completed) or not done.issubset(ids):
        raise ValueError("invalid or duplicate checkpoint rows")

    diarizen_config = OfficialDiariZenConfig(batch_size=args.diarizen_batch_size)
    lid_config = FireRedLidConfig(use_gpu=args.firered_lid_use_gpu)
    segment_config = SpeakerSegmentLidConfig()
    context = {}

    def classify(record):
        sample = record["sample"]
        audio_path = Path(sample["audio"]["path"])
        with audio_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with wave.open(str(audio_path), "rb") as source:
            duration = source.getnframes() / source.getframerate()
        scope = {
            "sample_id": sample["sample_id"],
            "audio_path": str(audio_path),
            "audio_sha256": digest,
            "duration_sec": duration,
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        diarizen = collect_official_diarizen_evidence(
            scope, diarizen_config, context=context
        )
        segment = collect_speaker_segment_lid_evidence(
            scope, diarizen, None, None,
            context=context,
            config=segment_config,
            lid_config=lid_config,
        )
        # This independent full-audio call is an actual baseline, never a
        # language_route value inferred from speaker-v2 fusion.
        full = collect_firered_lid_evidence(scope, lid_config, context=context)
        payload = segment.get("payload", {})
        segment_detail = payload.get("speaker_segment_lid", {})
        selected_timeline = _selected_timeline_output(diarizen, duration)
        clean, overlap = normalize_timeline(selected_timeline, return_overlap=True)
        longest_active = bool(segment.get("quality", {}).get("usable"))
        full_language = full.get("payload", {}).get("language")
        selected_language = payload.get("language") if longest_active else None
        return {
            "sample_id": sample["sample_id"],
            "audio_sha256": digest,
            "ground_truth": "en",
            "language": selected_language if longest_active else full_language,
            "full_audio_language": full_language,
            "segment_language": selected_language,
            "decision": payload.get("decision") or "timeline_unavailable",
            "fallback_reason": payload.get("fallback_reason"),
            "segment_status": segment.get("status"),
            "segment_reason": segment.get("quality", {}).get("reason"),
            "timeline_source": (selected_timeline or {}).get("source"),
            "diarizen_status": diarizen.get("status"),
            "diarizen_reason": diarizen.get("quality", {}).get("reason"),
            "diarizen_batch_size": diarizen.get("runtime", {}).get("inference_batch_size"),
            "full_audio_status": full.get("status"),
            "full_audio_reason": full.get("quality", {}).get("reason"),
            "selected_clean_segments": {
                speaker: [{"start_sec": start, "end_sec": end} for start, end in intervals]
                for speaker, intervals in clean.items()
            },
            "excluded_overlap_intervals": [
                {"start_sec": start, "end_sec": end} for start, end in overlap
            ],
            "speaker_results": segment_detail.get("speaker_results", []),
        }

    pending = [record for record in manifest if record["sample"]["sample_id"] not in done]
    try:
        with output.open("a", encoding="utf-8") as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for index, row in enumerate(executor.map(classify, pending), start=1):
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "[%d/%d] %s: %s (full=%s, decision=%s, diarizen=%s)" % (
                            len(done) + index,
                            len(manifest),
                            row["sample_id"],
                            row["language"],
                            row["full_audio_language"],
                            row["decision"],
                            row["diarizen_status"],
                        ),
                        flush=True,
                    )
    finally:
        close_subprocess_workers(context)
    rows = read_jsonl(output)
    if {row["sample_id"] for row in rows} != set(ids) or len(rows) != 350:
        raise ValueError("AMI rerun did not cover all 350 unique samples")
    counters = Counter(row["decision"] for row in rows)
    correct = sum(row["language"] == "en" for row in rows)
    known = sum(row["language"] is not None for row in rows)
    full_correct = sum(row["full_audio_language"] == "en" for row in rows)
    full_known = sum(row["full_audio_language"] is not None for row in rows)
    active = [row for row in rows if row["decision"] == "longest_clean_segment"]
    summary = {
        "sample_count": 350,
        "correct": correct,
        "accuracy_e2e": correct / 350,
        "known_count": known,
        "accuracy_known": correct / known if known else None,
        "full_audio_correct": full_correct,
        "full_audio_accuracy_e2e": full_correct / 350,
        "full_audio_known_count": full_known,
        "full_audio_accuracy_known": full_correct / full_known if full_known else None,
        "longest_segment_count": len(active),
        "longest_segment_correct": sum(row["language"] == "en" for row in active),
        "longest_segment_accuracy": (
            sum(row["language"] == "en" for row in active) / len(active)
            if active else None
        ),
        "longest_segment_full_audio_correct": sum(
            row["full_audio_language"] == "en" for row in active
        ),
        "longest_segment_improved_over_full_audio": sum(
            row["language"] == "en" and row["full_audio_language"] != "en"
            for row in active
        ),
        "longest_segment_degraded_from_full_audio": sum(
            row["language"] != "en" and row["full_audio_language"] == "en"
            for row in active
        ),
        "decision_counts": dict(counters),
        "diarizen_status_counts": dict(Counter(row["diarizen_status"] for row in rows)),
        "fallback_reason_counts": count_field(rows, "fallback_reason"),
        "segment_reason_counts": count_field(rows, "segment_reason"),
        "diarizen_batch_size": args.diarizen_batch_size,
        "workers": args.workers,
        "firered_lid_use_gpu": args.firered_lid_use_gpu,
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_staged(args):
    """Run DiariZen and FireRed LID in separate GPU-resident stages."""

    manifest = read_jsonl(args.manifest)
    ids = [record["sample"]["sample_id"] for record in manifest]
    if len(ids) != 350 or len(set(ids)) != 350:
        raise ValueError("expected 350 unique AMI samples")
    output = Path(args.output)
    cache_path = Path(args.diarization_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    completed = read_jsonl(output) if output.exists() else []
    done = {row["sample_id"] for row in completed}
    cached = read_jsonl(cache_path) if cache_path.exists() else []
    cache_by_id = {row["sample_id"]: row for row in cached}
    if len(done) != len(completed) or not done.issubset(ids):
        raise ValueError("invalid or duplicate result checkpoint rows")
    if len(cache_by_id) != len(cached) or not set(cache_by_id).issubset(ids):
        raise ValueError("invalid or duplicate diarization checkpoint rows")

    diarizen_config = OfficialDiariZenConfig(batch_size=args.diarizen_batch_size)
    context = {}

    def build_scope(record):
        sample = record["sample"]
        audio_path = Path(sample["audio"]["path"])
        with audio_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        with wave.open(str(audio_path), "rb") as source:
            duration = source.getnframes() / source.getframerate()
        return {
            "sample_id": sample["sample_id"],
            "audio_path": str(audio_path),
            "audio_sha256": digest,
            "duration_sec": duration,
            "sample_rate_hz": 16000,
            "channels": 1,
        }

    def diarize(record):
        scope = build_scope(record)
        evidence = collect_official_diarizen_evidence(
            scope, diarizen_config, context=context
        )
        return {
            "sample_id": scope["sample_id"],
            "scope": scope,
            "diarizen": evidence,
        }

    pending_diarization = [
        record for record in manifest
        if record["sample"]["sample_id"] not in cache_by_id
    ]
    try:
        with cache_path.open("a", encoding="utf-8") as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for index, row in enumerate(
                    executor.map(diarize, pending_diarization), start=1
                ):
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    cache_by_id[row["sample_id"]] = row
                    print(
                        "[diarization %d/%d] %s: %s" % (
                            len(cached) + index,
                            len(manifest),
                            row["sample_id"],
                            row["diarizen"].get("status"),
                        ),
                        flush=True,
                    )
    finally:
        close_subprocess_workers(context)

    if len(cache_by_id) != 350:
        raise ValueError("diarization stage did not cover all 350 samples")

    # Start a new subprocess context only after DiariZen has exited, so FireRed
    # LID can use the same GPU without the two models competing for memory.
    context = {}
    lid_config = FireRedLidConfig(use_gpu=True)
    segment_config = SpeakerSegmentLidConfig()

    def classify(record):
        sample_id = record["sample"]["sample_id"]
        cached_row = cache_by_id[sample_id]
        scope = cached_row["scope"]
        diarizen = cached_row["diarizen"]
        segment = collect_speaker_segment_lid_evidence(
            scope, diarizen, None, None,
            context=context,
            config=segment_config,
            lid_config=lid_config,
        )
        full = collect_firered_lid_evidence(scope, lid_config, context=context)
        payload = segment.get("payload", {})
        segment_detail = payload.get("speaker_segment_lid", {})
        selected_timeline = _selected_timeline_output(
            diarizen, scope["duration_sec"]
        )
        clean, overlap = normalize_timeline(selected_timeline, return_overlap=True)
        longest_active = bool(segment.get("quality", {}).get("usable"))
        full_language = full.get("payload", {}).get("language")
        selected_language = payload.get("language") if longest_active else None
        return {
            "sample_id": sample_id,
            "audio_sha256": scope["audio_sha256"],
            "ground_truth": "en",
            "language": selected_language if longest_active else full_language,
            "full_audio_language": full_language,
            "segment_language": selected_language,
            "decision": payload.get("decision") or "timeline_unavailable",
            "fallback_reason": payload.get("fallback_reason"),
            "segment_status": segment.get("status"),
            "segment_reason": segment.get("quality", {}).get("reason"),
            "timeline_source": (selected_timeline or {}).get("source"),
            "diarizen_status": diarizen.get("status"),
            "diarizen_reason": diarizen.get("quality", {}).get("reason"),
            "diarizen_batch_size": diarizen.get("runtime", {}).get("inference_batch_size"),
            "full_audio_status": full.get("status"),
            "full_audio_reason": full.get("quality", {}).get("reason"),
            "selected_clean_segments": {
                speaker: [{"start_sec": start, "end_sec": end} for start, end in intervals]
                for speaker, intervals in clean.items()
            },
            "excluded_overlap_intervals": [
                {"start_sec": start, "end_sec": end} for start, end in overlap
            ],
            "speaker_results": segment_detail.get("speaker_results", []),
        }

    pending = [
        record for record in manifest
        if record["sample"]["sample_id"] not in done
    ]
    try:
        with output.open("a", encoding="utf-8") as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for index, row in enumerate(executor.map(classify, pending), start=1):
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        "[language %d/%d] %s: %s (full=%s, decision=%s)" % (
                            len(done) + index,
                            len(manifest),
                            row["sample_id"],
                            row["language"],
                            row["full_audio_language"],
                            row["decision"],
                        ),
                        flush=True,
                    )
    finally:
        close_subprocess_workers(context)

    rows = read_jsonl(output)
    if len(rows) != 350 or {row["sample_id"] for row in rows} != set(ids):
        raise ValueError("AMI staged rerun did not cover all 350 unique samples")
    correct = sum(row["language"] == "en" for row in rows)
    known = sum(row["language"] is not None for row in rows)
    full_correct = sum(row["full_audio_language"] == "en" for row in rows)
    full_known = sum(row["full_audio_language"] is not None for row in rows)
    active = [row for row in rows if row["decision"] == "longest_clean_segment"]
    summary = {
        "sample_count": 350,
        "correct": correct,
        "accuracy_e2e": correct / 350,
        "known_count": known,
        "accuracy_known": correct / known if known else None,
        "full_audio_correct": full_correct,
        "full_audio_accuracy_e2e": full_correct / 350,
        "full_audio_known_count": full_known,
        "full_audio_accuracy_known": full_correct / full_known if full_known else None,
        "longest_segment_count": len(active),
        "longest_segment_correct": sum(row["language"] == "en" for row in active),
        "longest_segment_accuracy": (
            sum(row["language"] == "en" for row in active) / len(active)
            if active else None
        ),
        "longest_segment_full_audio_correct": sum(
            row["full_audio_language"] == "en" for row in active
        ),
        "longest_segment_improved_over_full_audio": sum(
            row["language"] == "en" and row["full_audio_language"] != "en"
            for row in active
        ),
        "longest_segment_degraded_from_full_audio": sum(
            row["language"] != "en" and row["full_audio_language"] == "en"
            for row in active
        ),
        "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "diarizen_status_counts": dict(Counter(row["diarizen_status"] for row in rows)),
        "fallback_reason_counts": count_field(rows, "fallback_reason"),
        "segment_reason_counts": count_field(rows, "segment_reason"),
        "diarizen_batch_size": args.diarizen_batch_size,
        "workers": args.workers,
        "firered_lid_use_gpu": True,
        "staged_model_residency": True,
        "diarization_cache": str(cache_path),
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def report(args):
    manifest = read_jsonl(args.manifest)
    tags = read_jsonl(args.tags)
    if len(manifest) != 350 or len(tags) != 350:
        raise ValueError("expected 350 manifest and prediction rows")
    artifacts = {}
    for path in Path(args.artifact_dir).glob("speaker_v2/*/fusion_artifact_v2.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            fusion = json.load(stream)
        sample_id = fusion["sample_id"]
        if sample_id in artifacts:
            raise ValueError("duplicate fusion artifact: %s" % sample_id)
        artifacts[sample_id] = (path, fusion)

    rows = []
    for record, tags_row in zip(manifest, tags):
        sample_id = record["sample"]["sample_id"]
        if sample_id not in artifacts:
            raise ValueError("missing fusion artifact: %s" % sample_id)
        fusion_path, fusion = artifacts[sample_id]
        segment_evidence = []
        for path in (fusion_path.parent / "evidence").glob("*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                evidence = json.load(stream)
            if evidence.get("source", {}).get("name") == "speaker_segment_firered_lid":
                segment_evidence.append(evidence)
        if len(segment_evidence) != 1:
            raise ValueError("expected one speaker-segment evidence for %s, got %d" % (
                sample_id, len(segment_evidence)))
        evidence = segment_evidence[0]
        payload = evidence.get("payload", {})
        lid = payload.get("speaker_segment_lid", {})
        route = fusion.get("language_route", {})
        language = tags_row["language_content"]["language"]
        rows.append({
            "sample_id": sample_id,
            "ground_truth": "en",
            "language": language,
            "correct": language == "en",
            "route_language": route.get("detected_language"),
            "route_source": route.get("language_source"),
            "segment_evidence_status": evidence.get("status"),
            "decision": payload.get("decision"),
            "fallback_reason": payload.get("fallback_reason"),
            "timeline_source": (payload.get("timeline") or {}).get("source"),
            "speaker_results": lid.get("speaker_results", []),
            "overlap_exclusion": lid.get("overlap_exclusion", {}),
        })

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite report: %s" % output)
    write_jsonl(output, rows)
    counts = Counter(row["decision"] or "missing" for row in rows)
    failures = Counter(row["fallback_reason"] or "none" for row in rows)
    correct = sum(row["correct"] for row in rows)
    known = sum(row["language"] is not None for row in rows)
    active = [row for row in rows if row["decision"] == "longest_clean_segment"]
    summary = {
        "samples": len(rows),
        "correct": correct,
        "accuracy_e2e": correct / len(rows),
        "non_null": known,
        "accuracy_non_null": correct / known if known else None,
        "decisions": dict(counts),
        "fallback_reasons": dict(failures),
        "longest_segment_correct": sum(row["correct"] for row in active),
        "longest_segment_count": len(active),
        "route_public_disagreements": sum(
            row["route_language"] != row["language"] for row in rows
        ),
        "per_sample": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--runtime-manifest", required=True)
    prep.add_argument("--staging-index", required=True)
    prep.add_argument("--output", required=True)
    prep.set_defaults(func=prepare)
    rerun = subparsers.add_parser("run")
    rerun.add_argument("--manifest", required=True)
    rerun.add_argument("--output", required=True)
    rerun.add_argument("--workers", type=int, default=2)
    rerun.add_argument("--diarizen-batch-size", type=int, default=8)
    rerun.add_argument("--firered-lid-use-gpu", action="store_true")
    rerun.set_defaults(func=run)
    staged = subparsers.add_parser("run-staged")
    staged.add_argument("--manifest", required=True)
    staged.add_argument("--output", required=True)
    staged.add_argument("--diarization-cache", required=True)
    staged.add_argument("--workers", type=int, default=2)
    staged.add_argument("--diarizen-batch-size", type=int, default=8)
    staged.set_defaults(func=run_staged)
    audit = subparsers.add_parser("report")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--tags", required=True)
    audit.add_argument("--artifact-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
