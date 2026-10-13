#!/usr/bin/env python3
"""Replay speaker-aware FireRed LID from saved speaker-v2 artifacts."""

import argparse
import gzip
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tagger.tools.language_content.firered_lid_detector import FireRedLidConfig
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
    run as run_speaker_segment_lid,
)
from tagger.tools.subprocess_runner import close_subprocess_workers


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved speaker-v2 timelines and rerun the current "
            "speaker-aware FireRed LID adapter."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-tags", required=True)
    parser.add_argument("--speaker-artifact-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--firered-lid-model", default=None)
    parser.add_argument("--firered-lid-repo", default=None)
    parser.add_argument("--firered-lid-python", default=None)
    parser.add_argument("--firered-lid-use-gpu", action="store_true")
    parser.add_argument("--firered-lid-half", action="store_true")
    parser.add_argument("--min-speaker-sec", type=float, default=2.0)
    parser.add_argument("--min-total-sec", type=float, default=3.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N manifest rows for a smoke run.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Replace existing output files instead of resuming them.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(
                    "%s:%d is not valid JSON" % (path, line_number)
                ) from exc
    return rows


def keyed(rows, label):
    result = {}
    for row in rows:
        sample_id = (
            row.get("id")
            or row.get("sample_id")
            or row.get("sample", {}).get("sample_id")
        )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("%s row has no id/sample_id" % label)
        if sample_id in result:
            raise ValueError("duplicate %s sample id: %s" % (label, sample_id))
        result[sample_id] = row
    return result


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def index_speaker_artifacts(root):
    result = {}
    for path in Path(root).glob("shard_*/speaker_v2/*/fusion_artifact_v2.json.gz"):
        fusion = load_gzip_json(path)
        sample_id = fusion.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("speaker fusion artifact has no sample_id: %s" % path)
        if sample_id in result:
            raise ValueError("duplicate speaker fusion artifact: %s" % sample_id)
        result[sample_id] = (path, fusion)
    return result


def selected_timeline(fusion_path, fusion):
    route = fusion.get("language_route", {})
    source_name = route.get("timeline_source")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("fusion artifact has no selected timeline source")
    candidates = []
    for path in (fusion_path.parent / "evidence").glob("*.json.gz"):
        evidence = load_gzip_json(path)
        if evidence.get("source", {}).get("name") != source_name:
            continue
        if "speaker_timeline" not in evidence.get("capabilities", []):
            continue
        if evidence.get("status") not in ("observed", "estimated"):
            continue
        if evidence.get("quality", {}).get("usable", True) is False:
            continue
        candidates.append(evidence)
    if len(candidates) != 1:
        raise ValueError(
            "expected one usable %s timeline, found %d"
            % (source_name, len(candidates))
        )
    evidence = candidates[0]
    summary = evidence.get("payload", {}).get("timeline_summary", {})
    segments = summary.get("activity_segments") or summary.get("segments") or []
    return {
        "source": source_name,
        "evidence_id": evidence.get("evidence_id"),
        "duration_sec": fusion.get("input_provenance", {}).get("duration_sec"),
        "activity_segments": segments,
    }


def baseline_result(row, fusion):
    tags = row.get("tags", row)
    language = tags.get("language_content", {}).get("language")
    route = fusion.get("language_route", {})
    return {
        "language": language,
        "confidence": route.get("language_confidence"),
        "error": route.get("language_error"),
        "evidence_id": route.get("lid_evidence_id"),
        "language_route": route.get("language_route"),
    }


def existing_ids(path):
    path = Path(path)
    return set(keyed(load_jsonl(path), "existing output")) if path.exists() else set()


def evidence_record(sample_id, result):
    evidence = result.evidence
    return {
        "sample_id": sample_id,
        "language": result.value,
        "confidence": result.confidence,
        "status": result.status,
        "decision": evidence.get("decision"),
        "fallback_reason": evidence.get("fallback_reason"),
        "timeline": evidence.get("timeline"),
        "full_audio_result": evidence.get("full_audio_result"),
        "eligible_speaker_ids": evidence.get("eligible_speaker_ids", []),
        "speaker_activity_sec": evidence.get("speaker_activity_sec", {}),
        "eligible_total_activity_sec": evidence.get(
            "eligible_total_activity_sec"
        ),
        "speaker_results": evidence.get("speaker_results", []),
        "language_duration_sec": evidence.get("language_duration_sec", {}),
        "proposed_language": evidence.get("proposed_language"),
        "segment_error": evidence.get("segment_error"),
        "config": evidence.get("config"),
    }


def main():
    args = parse_args()
    manifest_rows = load_jsonl(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        manifest_rows = manifest_rows[: args.limit]
    manifest = keyed(manifest_rows, "manifest")
    baseline = keyed(load_jsonl(args.baseline_tags), "baseline tags")
    artifacts = index_speaker_artifacts(args.speaker_artifact_root)
    expected_ids = list(manifest)
    for label, values in (("baseline", baseline), ("artifacts", artifacts)):
        missing = sorted(set(expected_ids) - set(values))
        if missing:
            raise ValueError("%s misses sample ids: %s" % (label, missing[:10]))

    output_path = Path(args.output)
    evidence_path = Path(args.evidence_output)
    summary_path = Path(args.summary)
    for path in (output_path, evidence_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for path in (output_path, evidence_path, summary_path):
            if path.exists():
                path.unlink()
    output_done = existing_ids(output_path)
    evidence_done = existing_ids(evidence_path)
    if output_done != evidence_done:
        raise ValueError("output and evidence checkpoints contain different sample IDs")

    lid_config = FireRedLidConfig(
        model_dir=args.firered_lid_model,
        repo_dir=args.firered_lid_repo,
        use_gpu=args.firered_lid_use_gpu,
        use_half=args.firered_lid_half,
        subprocess_python=args.firered_lid_python,
    )
    policy = SpeakerSegmentLidConfig(
        enabled=True,
        min_speaker_duration_sec=args.min_speaker_sec,
        min_total_duration_sec=args.min_total_sec,
    )
    context = {}
    counts = {
        "expected_sample_count": len(expected_ids),
        "resumed_sample_count": len(output_done),
        "processed_sample_count": 0,
        "segment_aggregate_count": 0,
        "full_audio_fallback_count": 0,
        "changed_from_baseline_count": 0,
        "speaker_lid_call_count": 0,
    }
    try:
        with output_path.open("a", encoding="utf-8") as output_sink, evidence_path.open(
            "a", encoding="utf-8"
        ) as evidence_sink:
            for index, sample_id in enumerate(expected_ids, start=1):
                if sample_id in output_done:
                    continue
                manifest_row = manifest[sample_id]
                baseline_row = baseline[sample_id]
                fusion_path, fusion = artifacts[sample_id]
                timeline = selected_timeline(fusion_path, fusion)
                audio_path = manifest_row.get("sample", {}).get("audio", {}).get("path")
                if not isinstance(audio_path, str) or not Path(audio_path).is_file():
                    raise ValueError("audio is missing for %s: %s" % (sample_id, audio_path))
                full_audio_result = baseline_result(baseline_row, fusion)
                result = run_speaker_segment_lid(
                    audio_path,
                    timeline,
                    full_audio_result,
                    context=context,
                    config=policy,
                    lid_config=lid_config,
                    sample_id=sample_id,
                )

                output_row = json.loads(json.dumps(baseline_row))
                output_row["id"] = sample_id
                output_row["audio_path"] = audio_path
                output_row.setdefault("tags", {}).setdefault(
                    "language_content", {}
                )["language"] = result.value
                output_sink.write(
                    json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                evidence_sink.write(
                    json.dumps(
                        evidence_record(sample_id, result),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                output_sink.flush()
                evidence_sink.flush()

                counts["processed_sample_count"] += 1
                decision = result.evidence.get("decision")
                if decision == "segment_aggregate":
                    counts["segment_aggregate_count"] += 1
                else:
                    counts["full_audio_fallback_count"] += 1
                if result.value != full_audio_result.get("language"):
                    counts["changed_from_baseline_count"] += 1
                counts["speaker_lid_call_count"] += len(
                    result.evidence.get("speaker_results", [])
                )
                print(
                    "[%d/%d] %s: %s -> %s (%s)"
                    % (
                        index,
                        len(expected_ids),
                        sample_id,
                        full_audio_result.get("language"),
                        result.value,
                        decision,
                    ),
                    flush=True,
                )
    finally:
        close_subprocess_workers(context)

    completed = existing_ids(output_path)
    all_evidence = load_jsonl(evidence_path)
    counts["segment_aggregate_count"] = sum(
        row.get("decision") == "segment_aggregate" for row in all_evidence
    )
    counts["full_audio_fallback_count"] = sum(
        row.get("decision") != "segment_aggregate" for row in all_evidence
    )
    counts["speaker_lid_call_count"] = sum(
        len(row.get("speaker_results", [])) for row in all_evidence
    )
    counts["changed_from_baseline_count"] = sum(
        row.get("language") != row.get("full_audio_result", {}).get("language")
        for row in all_evidence
    )
    counts["completed_sample_count"] = len(completed)
    counts["complete"] = len(completed) == len(expected_ids)
    counts["inputs"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "baseline_tags": str(Path(args.baseline_tags).resolve()),
        "speaker_artifact_root": str(Path(args.speaker_artifact_root).resolve()),
    }
    counts["outputs"] = {
        "predictions": str(output_path.resolve()),
        "evidence": str(evidence_path.resolve()),
    }
    counts["firered_lid"] = lid_config.to_record()
    counts["policy"] = policy.to_record()
    summary_path.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    if not counts["complete"]:
        raise RuntimeError("rerun is incomplete")


if __name__ == "__main__":
    main()
