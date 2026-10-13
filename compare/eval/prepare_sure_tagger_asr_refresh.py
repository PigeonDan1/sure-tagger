#!/usr/bin/env python3
"""Prepare aligned T1-T3 manifests and baseline tags for an ASR refresh."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_runtime(path):
    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank runtime row at line %d" % line_number)
            value = json.loads(line)
            sample_id = value.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("runtime row %d has no sample_id" % line_number)
            if sample_id in seen:
                raise ValueError("duplicate runtime sample_id: %s" % sample_id)
            seen.add(sample_id)
            rows.append((sample_id, json.dumps(value, ensure_ascii=False, sort_keys=True)))
    if not rows:
        raise ValueError("runtime manifest is empty")
    return rows


def load_baseline(path):
    by_id = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank baseline row at line %d" % line_number)
            value = json.loads(line)
            sample_id = value.get("sample_id")
            tags = value.get("tags")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("baseline row %d has no sample_id" % line_number)
            if not isinstance(tags, dict):
                raise ValueError("baseline row %d has no tags object" % line_number)
            if sample_id in by_id:
                raise ValueError("duplicate baseline sample_id: %s" % sample_id)
            by_id[sample_id] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
    return by_id


def prepare(runtime_manifest, baseline, output_dir, gpu_count, force=False):
    runtime_manifest = Path(runtime_manifest).resolve()
    baseline = Path(baseline).resolve()
    output_dir = Path(output_dir).resolve()
    rows = load_runtime(runtime_manifest)
    baseline_by_id = load_baseline(baseline)
    missing = sorted(sample_id for sample_id, _ in rows if sample_id not in baseline_by_id)
    if missing:
        raise ValueError("baseline is missing %d runtime IDs" % len(missing))
    gpu_count = int(gpu_count)
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")

    partitions = [[] for _ in range(gpu_count)]
    for index, row in enumerate(rows):
        partitions[index % gpu_count].append(row)

    for gpu, partition in enumerate(partitions):
        name = "%02d" % gpu
        manifest_path = output_dir / "manifests" / ("manifest_%s.jsonl" % name)
        baseline_path = output_dir / "baseline" / ("tags_%s.jsonl" % name)
        if not force and (manifest_path.exists() or baseline_path.exists()):
            raise ValueError("refusing to overwrite prepared shard %s" % name)
        atomic_write(
            manifest_path,
            "".join(line + "\n" for _sample_id, line in partition),
        )
        atomic_write(
            baseline_path,
            "".join(baseline_by_id[sample_id] + "\n" for sample_id, _line in partition),
        )

    summary = {
        "schema_version": "sure-tagger-asr-refresh-preparation.v1",
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256_file(runtime_manifest),
        "baseline": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "sample_count": len(rows),
        "gpu_count": gpu_count,
        "workers_per_gpu": 1,
        "partition_counts": [len(partition) for partition in partitions],
        "selected_stages": [
            "speaker",
            "firered_lid",
            "language_deterministic",
            "deterministic_derived",
        ],
        "preserved_groups": [
            "basic_acoustic",
            "audio_quality",
            "room_acoustic",
            "sound_field_scene",
        ],
    }
    atomic_write(
        output_dir / "prepare.summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-count", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare(
        args.runtime_manifest,
        args.baseline,
        args.output_dir,
        args.gpu_count,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
