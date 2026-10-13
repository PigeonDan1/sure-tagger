import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys

import yaml

from sure_tagger.datasets.ami import build_manifest
from sure_tagger.datasets.ami_utterance import build_utterance_manifest
from sure_tagger.io.jsonl import JsonlWriter, read_jsonl
from sure_tagger.meeting_manifest import build_meeting_manifest
from sure_tagger.report import update_distribution, write_report
from sure_tagger.schemas import make_error, now_iso, validate_manifest_record
from sure_tagger.tags import filler, language, punctuation, repetition, topic, word_count
from sure_tagger.text.context import ContextBuilder


TAGGERS = {
    "language": language.tag,
    "word_count": word_count.tag,
    "punctuation": punctuation.tag,
    "filler": filler.tag,
    "repetition": repetition.tag,
    "topic": topic.tag,
}


DETERMINISTIC_TAGS = ["language", "word_count", "punctuation", "filler", "repetition"]


def parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value):
    value = value or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def load_yaml(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def selected_tags(args, config):
    if args.tags:
        return [t.strip() for t in args.tags.split(",") if t.strip()]
    tags_conf = config.get("tags", {})
    if tags_conf:
        return [name for name, conf in tags_conf.items() if conf.get("enabled", True)]
    return ["language", "word_count", "punctuation", "filler", "repetition", "topic"]


def cmd_build_manifest(args):
    meetings = parse_csv(args.meetings)
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    bad_path = args.bad_samples or os.path.join(parent, "bad_samples.build_manifest.jsonl")
    with JsonlWriter(args.output) as out, JsonlWriter(bad_path) as bad:
        if args.dataset != "ami":
            raise ValueError("Unsupported dataset: %s" % args.dataset)
        stats = build_manifest(args.root, out, bad_writer=bad, limit=args.limit, meetings=meetings)
    report_path = args.report or os.path.join(parent, "build_manifest.report.json")
    stats["dataset"] = args.dataset
    stats["root"] = args.root
    stats["created_at"] = now_iso()
    write_report(report_path, stats)
    print("wrote %s records to %s" % (stats["records"], args.output))
    print("report: %s" % report_path)


def cmd_build_meeting_manifest(args):
    meetings = parse_csv(args.meetings)
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    bad_path = args.bad_samples or os.path.join(parent, "bad_samples.build_meeting_manifest.jsonl")
    with JsonlWriter(args.output) as out, JsonlWriter(bad_path) as bad:
        stats = build_meeting_manifest(args.manifest, out, bad_writer=bad, limit=args.limit, meetings=meetings)
    report_path = args.report or os.path.join(parent, "build_meeting_manifest.report.json")
    stats["created_at"] = now_iso()
    write_report(report_path, stats)
    print("wrote %s meeting records to %s" % (stats["records"], args.output))
    print("report: %s" % report_path)


def cmd_build_utterance_manifest(args):
    meetings = parse_csv(args.meetings)
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    bad_path = args.bad_samples or os.path.join(parent, "bad_samples.build_utterance_manifest.jsonl")
    with JsonlWriter(args.output) as out, JsonlWriter(bad_path) as bad:
        if args.dataset not in ("ami_utterance", "ami"):
            raise ValueError("Unsupported utterance dataset: %s" % args.dataset)
        stats = build_utterance_manifest(
            args.root,
            out,
            bad_writer=bad,
            limit=args.limit,
            meetings=meetings,
        )
    report_path = args.report or os.path.join(parent, "build_utterance_manifest.report.json")
    stats["dataset"] = args.dataset
    stats["root"] = args.root
    stats["created_at"] = now_iso()
    write_report(report_path, stats)
    print("wrote %s utterance records to %s" % (stats["records"], args.output))
    print("report: %s" % report_path)


def build_topic_context(record, context_builder, conf):
    sample = record["sample"]
    meta = sample.get("native_metadata", {})
    if meta.get("granularity") == "meeting":
        audio = sample.get("audio", {})
        start = audio.get("start_sec")
        end = audio.get("end_sec")
        duration = None
        if start is not None and end is not None:
            try:
                duration = float(end) - float(start)
            except (TypeError, ValueError):
                duration = None
        return {
            "meeting_id": meta.get("meeting_id", ""),
            "speaker_ids": meta.get("speaker_ids", []),
            "target_granularity": "meeting",
            "evidence_scope": "meeting",
            "evidence_sample_count": meta.get("source_sample_count", 1),
            "text_segment_count": meta.get("text_segment_count", 0),
            "meeting_duration_sec": duration,
            "transcript_format": "speaker_labeled_transcript_if_available",
        }

    ctx_conf = conf.get("context", {})
    return context_builder.build(
        record,
        meeting_window_sec=ctx_conf.get("meeting_window_sec", 120),
        speaker_neighbor_segments=ctx_conf.get("speaker_neighbor_segments", 3),
        max_context_chars=ctx_conf.get("max_context_chars", 6000),
    )


def _tag_one_record(record, tag_names, config, context_builder, run_id, config_path):
    sample_id = record["sample"]["sample_id"]
    result_tags = {}
    errors = []
    for name in tag_names:
        conf = config.get("tags", {}).get(name, {})
        try:
            if name == "topic":
                ctx = build_topic_context(record, context_builder, conf)
                tag_result = TAGGERS[name](record, conf, context=ctx)
            else:
                tag_result = TAGGERS[name](record, conf)
            result_tags[name] = tag_result
        except Exception as exc:
            errors.append(make_error(
                sample_id,
                "tag:%s" % name,
                exc.__class__.__name__,
                exc,
                record["sample"].get("provenance", {}).get("source_path"),
            ))
    return {
        "record": {
            "sample_id": sample_id,
            "tags": result_tags,
            "pipeline": {
                "run_id": run_id or "sure_tagger_run",
                "config": config_path or "",
                "created_at": now_iso(),
            },
        },
        "errors": errors,
    }


def _write_tagged_record(result, out, bad, report):
    for name, tag_result in result["record"]["tags"].items():
        update_distribution(report["distributions"], name, tag_result)
    for error in result["errors"]:
        report["bad_records"] += 1
        bad.write(error)
    out.write(result["record"])
    report["records_written"] += 1


def cmd_tag(args):
    config = load_yaml(args.config)
    tag_names = selected_tags(args, config)
    for name in tag_names:
        if name not in TAGGERS:
            raise ValueError("Unknown tag: %s" % name)
    if args.topic_provider:
        config.setdefault("tags", {}).setdefault("topic", {}).setdefault("model", {})["provider"] = args.topic_provider
    if args.dry_run:
        config.setdefault("tags", {}).setdefault("topic", {}).setdefault("model", {})["provider"] = "dry_run"

    records = []
    for idx, rec in enumerate(read_jsonl(args.manifest), 1):
        validate_manifest_record(rec)
        records.append(rec)
        if args.limit and len(records) >= int(args.limit):
            break

    context_builder = ContextBuilder(records) if "topic" in tag_names else None
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    bad_path = args.bad_samples or os.path.join(parent, "bad_samples.tag.jsonl")
    report = {
        "created_at": now_iso(),
        "manifest": args.manifest,
        "output": args.output,
        "tag_names": tag_names,
        "workers": max(1, int(getattr(args, "workers", 1) or 1)),
        "records_seen": len(records),
        "records_written": 0,
        "bad_records": 0,
        "distributions": {},
    }

    workers = report["workers"]
    if workers > 1:
        print("tag workers: %s" % workers)

    with JsonlWriter(args.output) as out, JsonlWriter(bad_path) as bad:
        if workers <= 1 or len(records) <= 1:
            for rec in records:
                result = _tag_one_record(
                    rec,
                    tag_names,
                    config,
                    context_builder,
                    args.run_id,
                    args.config,
                )
                _write_tagged_record(result, out, bad, report)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _tag_one_record,
                        rec,
                        tag_names,
                        config,
                        context_builder,
                        args.run_id,
                        args.config,
                    )
                    for rec in records
                ]
                for future in futures:
                    _write_tagged_record(future.result(), out, bad, report)
    report_path = args.report or os.path.join(parent, "tag.report.json")
    write_report(report_path, report)
    print("tagged %s records -> %s" % (report["records_written"], args.output))
    print("bad tag records: %s" % report["bad_records"])
    print("report: %s" % report_path)


def cmd_inspect(args):
    manifest = {}
    for rec in read_jsonl(args.manifest):
        manifest[rec["sample"]["sample_id"]] = rec
        if args.sample_size and len(manifest) >= int(args.sample_size):
            break
    tags = {}
    for rec in read_jsonl(args.tags_file):
        sid = rec.get("sample_id")
        if sid in manifest:
            tags[sid] = rec
    parent = os.path.dirname(args.output)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with JsonlWriter(args.output) as out:
        for sid, rec in manifest.items():
            out.write({"manifest": rec, "tags": tags.get(sid, {})})
    print("wrote inspection samples: %s" % args.output)


def pipeline_tag_names(args):
    if args.tags:
        tag_names = parse_csv(args.tags)
    else:
        tag_names = list(DETERMINISTIC_TAGS)
    if args.include_topic and "topic" not in tag_names:
        tag_names.append("topic")
    return tag_names


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_run_meeting_pipeline(args):
    if args.dataset != "ami":
        raise ValueError("Unsupported dataset: %s" % args.dataset)

    meetings = parse_csv(args.meetings)
    run_name = args.run_name
    if not run_name:
        run_name = "full" if not meetings else "meetings.%s" % "_".join(meetings)
    run_name = safe_name(run_name)

    output_dir = args.output_dir or os.path.join("outputs", args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    segment_manifest = os.path.join(output_dir, "segment_manifest.%s.jsonl" % run_name)
    segment_bad = os.path.join(output_dir, "bad_samples.build_segment_manifest.%s.jsonl" % run_name)
    segment_report = os.path.join(output_dir, "build_segment_manifest.%s.report.json" % run_name)

    meeting_manifest = os.path.join(output_dir, "meeting_manifest.%s.jsonl" % run_name)
    meeting_bad = os.path.join(output_dir, "bad_samples.build_meeting_manifest.%s.jsonl" % run_name)
    meeting_report = os.path.join(output_dir, "build_meeting_manifest.%s.report.json" % run_name)

    tag_output = args.output or os.path.join(output_dir, "meeting_tags.%s.jsonl" % run_name)
    tag_bad = os.path.join(output_dir, "bad_samples.tag.meeting.%s.jsonl" % run_name)
    tag_report = os.path.join(output_dir, "tag.meeting.%s.report.json" % run_name)

    qa_output = os.path.join(output_dir, "meeting_qa_samples.%s.jsonl" % run_name)
    pipeline_report = os.path.join(output_dir, "pipeline.%s.report.json" % run_name)

    tag_names = pipeline_tag_names(args)
    for name in tag_names:
        if name not in TAGGERS:
            raise ValueError("Unknown tag: %s" % name)

    if args.segment_limit:
        print("warning: --segment-limit truncates raw segments and should only be used for parser debugging, not final meeting tags")

    print("[1/4] build segment manifest")
    cmd_build_manifest(argparse.Namespace(
        dataset=args.dataset,
        root=args.root,
        output=segment_manifest,
        bad_samples=segment_bad,
        report=segment_report,
        limit=args.segment_limit,
        meetings=args.meetings,
    ))

    print("[2/4] build meeting manifest")
    cmd_build_meeting_manifest(argparse.Namespace(
        manifest=segment_manifest,
        output=meeting_manifest,
        bad_samples=meeting_bad,
        report=meeting_report,
        limit=args.meeting_limit,
        meetings=args.meetings,
    ))

    print("[3/4] tag meetings")
    cmd_tag(argparse.Namespace(
        manifest=meeting_manifest,
        config=args.config,
        tags=",".join(tag_names),
        output=tag_output,
        bad_samples=tag_bad,
        report=tag_report,
        limit=None,
        run_id=args.run_id or run_name,
        topic_provider=args.topic_provider,
        dry_run=args.dry_run,
        workers=args.workers,
    ))

    qa_path = ""
    if not args.skip_qa:
        print("[4/4] build QA samples")
        cmd_inspect(argparse.Namespace(
            manifest=meeting_manifest,
            tags_file=tag_output,
            sample_size=args.qa_sample_size,
            output=qa_output,
        ))
        qa_path = qa_output
    else:
        print("[4/4] skip QA samples")

    report = {
        "created_at": now_iso(),
        "dataset": args.dataset,
        "root": args.root,
        "run_name": run_name,
        "meetings": meetings,
        "tag_names": tag_names,
        "include_topic": bool("topic" in tag_names),
        "topic_provider_override": args.topic_provider or "",
        "workers": max(1, int(getattr(args, "workers", 1) or 1)),
        "paths": {
            "segment_manifest": segment_manifest,
            "segment_report": segment_report,
            "segment_bad_samples": segment_bad,
            "meeting_manifest": meeting_manifest,
            "meeting_report": meeting_report,
            "meeting_bad_samples": meeting_bad,
            "tag_output": tag_output,
            "tag_report": tag_report,
            "tag_bad_samples": tag_bad,
            "qa_output": qa_path,
        },
        "segment_stats": load_json(segment_report),
        "meeting_stats": load_json(meeting_report),
        "tag_stats": load_json(tag_report),
    }
    write_report(pipeline_report, report)
    print("pipeline report: %s" % pipeline_report)
    print("final meeting tags: %s" % tag_output)


def cmd_run_utterance_pipeline(args):
    if args.dataset not in ("ami_utterance", "ami"):
        raise ValueError("Unsupported utterance dataset: %s" % args.dataset)

    meetings = parse_csv(args.meetings)
    run_name = args.run_name
    if not run_name:
        run_name = "full" if not meetings else "meetings.%s" % "_".join(meetings)
    run_name = safe_name(run_name)

    output_dir = args.output_dir or os.path.join("outputs", "ami_utterance")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utterance_manifest = os.path.join(output_dir, "utterance_manifest.%s.jsonl" % run_name)
    utterance_bad = os.path.join(output_dir, "bad_samples.build_utterance_manifest.%s.jsonl" % run_name)
    utterance_report = os.path.join(output_dir, "build_utterance_manifest.%s.report.json" % run_name)

    tag_output = args.output or os.path.join(output_dir, "utterance_tags.%s.jsonl" % run_name)
    tag_bad = os.path.join(output_dir, "bad_samples.tag.utterance.%s.jsonl" % run_name)
    tag_report = os.path.join(output_dir, "tag.utterance.%s.report.json" % run_name)

    qa_output = os.path.join(output_dir, "utterance_qa_samples.%s.jsonl" % run_name)
    pipeline_report = os.path.join(output_dir, "pipeline.utterance.%s.report.json" % run_name)

    tag_names = pipeline_tag_names(args)
    for name in tag_names:
        if name not in TAGGERS:
            raise ValueError("Unknown tag: %s" % name)

    print("[1/3] build utterance manifest")
    cmd_build_utterance_manifest(argparse.Namespace(
        dataset=args.dataset,
        root=args.root,
        output=utterance_manifest,
        bad_samples=utterance_bad,
        report=utterance_report,
        limit=args.limit,
        meetings=args.meetings,
    ))

    print("[2/3] tag utterances")
    cmd_tag(argparse.Namespace(
        manifest=utterance_manifest,
        config=args.config,
        tags=",".join(tag_names),
        output=tag_output,
        bad_samples=tag_bad,
        report=tag_report,
        limit=None,
        run_id=args.run_id or run_name,
        topic_provider=args.topic_provider,
        dry_run=args.dry_run,
        workers=args.workers,
    ))

    qa_path = ""
    if not args.skip_qa:
        print("[3/3] build QA samples")
        cmd_inspect(argparse.Namespace(
            manifest=utterance_manifest,
            tags_file=tag_output,
            sample_size=args.qa_sample_size,
            output=qa_output,
        ))
        qa_path = qa_output
    else:
        print("[3/3] skip QA samples")

    report = {
        "created_at": now_iso(),
        "dataset": args.dataset,
        "root": args.root,
        "run_name": run_name,
        "meetings": meetings,
        "tag_names": tag_names,
        "include_topic": bool("topic" in tag_names),
        "topic_provider_override": args.topic_provider or "",
        "workers": max(1, int(getattr(args, "workers", 1) or 1)),
        "paths": {
            "utterance_manifest": utterance_manifest,
            "utterance_report": utterance_report,
            "utterance_bad_samples": utterance_bad,
            "tag_output": tag_output,
            "tag_report": tag_report,
            "tag_bad_samples": tag_bad,
            "qa_output": qa_path,
        },
        "utterance_stats": load_json(utterance_report),
        "tag_stats": load_json(tag_report),
    }
    write_report(pipeline_report, report)
    print("pipeline report: %s" % pipeline_report)
    print("final utterance tags: %s" % tag_output)


def build_parser():
    parser = argparse.ArgumentParser(prog="sure_tagger")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build-manifest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bad-samples")
    p.add_argument("--report")
    p.add_argument("--limit", type=int)
    p.add_argument("--meetings")
    p.set_defaults(func=cmd_build_manifest)

    p = sub.add_parser("build-meeting-manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bad-samples")
    p.add_argument("--report")
    p.add_argument("--limit", type=int)
    p.add_argument("--meetings")
    p.set_defaults(func=cmd_build_meeting_manifest)

    p = sub.add_parser("build-utterance-manifest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bad-samples")
    p.add_argument("--report")
    p.add_argument("--limit", type=int)
    p.add_argument("--meetings")
    p.set_defaults(func=cmd_build_utterance_manifest)

    p = sub.add_parser("tag")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config")
    p.add_argument("--tags")
    p.add_argument("--output", required=True)
    p.add_argument("--bad-samples")
    p.add_argument("--report")
    p.add_argument("--limit", type=int)
    p.add_argument("--run-id")
    p.add_argument("--topic-provider")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("inspect")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tags-file", required=True)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("run-meeting-pipeline")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", default=os.path.join("outputs", "ami"))
    p.add_argument("--output")
    p.add_argument("--config", default=os.path.join("configs", "tags_language_mvp.yaml"))
    p.add_argument("--meetings")
    p.add_argument("--run-name")
    p.add_argument("--tags")
    p.add_argument("--include-topic", action="store_true")
    p.add_argument("--topic-provider")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--segment-limit", type=int)
    p.add_argument("--meeting-limit", type=int)
    p.add_argument("--qa-sample-size", type=int, default=20)
    p.add_argument("--skip-qa", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_run_meeting_pipeline)

    p = sub.add_parser("run-utterance-pipeline")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", default=os.path.join("outputs", "ami_utterance"))
    p.add_argument("--output")
    p.add_argument("--config", default=os.path.join("configs", "tags_language_mvp.yaml"))
    p.add_argument("--meetings")
    p.add_argument("--run-name")
    p.add_argument("--tags")
    p.add_argument("--include-topic", action="store_true")
    p.add_argument("--topic-provider")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--qa-sample-size", type=int, default=20)
    p.add_argument("--skip-qa", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_run_utterance_pipeline)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
