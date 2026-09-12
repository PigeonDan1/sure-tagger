#!/usr/bin/env python3
"""Run speaker evidence evaluation over every manifest in a dataset tree."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "scripts" / "run_speaker_evidence_v2.py"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-workers", type=int, default=1)
    parser.add_argument("--moss-workers", type=int, default=None)
    parser.add_argument("--whisper-workers", type=int, default=None)
    parser.add_argument("--sortformer-workers", type=int, default=None)
    parser.add_argument("--firered-vad-workers", type=int, default=None)
    parser.add_argument("--campplus-workers", type=int, default=None)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-native", action="store_true")
    parser.add_argument("--max-manifests", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260812)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifests = sorted(dataset_root.glob("utterance/*/manifest.jsonl"))
    if args.max_manifests is not None:
        manifests = manifests[: args.max_manifests]
    if not manifests:
        raise SystemExit("no manifests found under %s/utterance/*" % dataset_root)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "status": "running",
        "started_at_unix": time.time(),
        "dataset_root": str(dataset_root),
        "dataset_manifest_count": len(manifests),
        "dataset_sample_count": sum(_line_count(path) for path in manifests),
        "workers": args.workers,
        "model_workers": args.model_workers,
        "score_native": bool(args.score_native),
        "resume": bool(args.resume),
        "runner": str(Path(args.runner).resolve()),
        "runner_sha256": _sha256(Path(args.runner)),
        "manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path in manifests
        ],
    }
    run_manifest_path = output_dir / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)

    combined_manifest = output_dir / "combined_manifest.jsonl"
    sampled_count = _write_combined_manifest(
        combined_manifest,
        manifests,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
    )
    run_manifest["combined_manifest"] = str(combined_manifest)
    run_manifest["combined_manifest_sha256"] = _sha256(combined_manifest)
    run_manifest["selected_sample_count"] = sampled_count
    run_manifest["sample_seed"] = args.sample_seed if args.sample_count else None
    _write_json(run_manifest_path, run_manifest)

    inference_output = output_dir / "inference"
    command = [
        args.python,
        args.runner,
        "--manifest",
        str(combined_manifest),
        "--output-dir",
        str(inference_output),
        "--workers",
        str(args.workers),
        "--model-workers",
        str(args.model_workers),
    ]
    for option, value in (
        ("--moss-workers", args.moss_workers),
        ("--whisper-workers", args.whisper_workers),
        ("--sortformer-workers", args.sortformer_workers),
        ("--firered-vad-workers", args.firered_vad_workers),
        ("--campplus-workers", args.campplus_workers),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.resume:
        command.append("--resume")
    if args.score_native:
        command.append("--score-native")
    runner_args = list(args.runner_args)
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    command.extend(runner_args)
    log_path = output_dir / "run.log"
    print(
        "running %d samples from %d manifests with %d workers"
        % (run_manifest["dataset_sample_count"], len(manifests), args.workers),
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    failures = []
    if completed.returncode != 0:
        failures.append(
            {
                "returncode": completed.returncode,
                "log": str(log_path),
            }
        )

    run_manifest.update(
        {
            "status": "completed" if not failures else "completed_with_failures",
            "completed_at_unix": time.time(),
            "run_failure_count": len(failures),
            "run_failures": failures,
            "result_path": str(
                inference_output / "speaker_v2_shadow_results.jsonl"
            ),
        }
    )
    _write_json(run_manifest_path, run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _line_count(path):
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _write_combined_manifest(path, manifests, sample_count=None, sample_seed=0):
    records = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                audio_path = Path(record["sample"]["audio"]["path"])
                if not audio_path.is_absolute():
                    audio_path = (manifest.parent / audio_path).resolve()
                record["sample"]["audio"]["path"] = str(audio_path)
                records.append(record)
    if sample_count is not None:
        sample_count = int(sample_count)
        if sample_count < 1:
            raise SystemExit("--sample-count must be positive")
        if sample_count > len(records):
            raise SystemExit(
                "--sample-count=%d exceeds available samples=%d"
                % (sample_count, len(records))
            )
        rng = random.Random(int(sample_seed))
        records = [records[index] for index in sorted(rng.sample(range(len(records)), sample_count))]
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            sink.write("\n")
    temporary.replace(path)
    return len(records)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("w", encoding="utf-8") as sink:
        json.dump(value, sink, ensure_ascii=False, indent=2, sort_keys=True)
        sink.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
