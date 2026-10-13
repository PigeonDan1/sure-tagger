#!/usr/bin/env python3
"""Prepare partitioned sure-tagger stage manifests.

The full sure-tagger profile is memory-heavy.  This helper partitions the
runtime audio-only manifest into one manifest per GPU.  The ``core`` and
``aux`` trees are retained for compatibility with older launchers, while the
current launcher runs only the core manifest once on each GPU.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile
import wave
from pathlib import Path


EXPECTED_TOP = set(("corpus", "sample"))
EXPECTED_CORPUS = set(("dataset_name", "native_metadata", "source_urls"))
EXPECTED_SAMPLE = set(("audio", "native_metadata", "sample_id", "text"))
EXPECTED_AUDIO = set(("path",))
EXPECTED_TEXT = set(("transcript",))
EXPECTED_URLS = set(("article", "dataset_card", "github", "huggingface"))
ID_RE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(path, allowed_root):
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve()
    rows = []
    seen_ids = set()
    seen_audio = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid JSON at line %d: %s" % (line_number, exc))
            _require(isinstance(record, dict), "line %d is not an object" % line_number)
            _require(set(record) == EXPECTED_TOP, "line %d top-level schema mismatch" % line_number)
            corpus = record["corpus"]
            sample = record["sample"]
            _require(isinstance(corpus, dict) and set(corpus) == EXPECTED_CORPUS,
                     "line %d corpus schema mismatch" % line_number)
            _require(isinstance(sample, dict) and set(sample) == EXPECTED_SAMPLE,
                     "line %d sample schema mismatch" % line_number)
            _require(corpus["dataset_name"] == "", "line %d leaks dataset_name" % line_number)
            _require(corpus["native_metadata"] == {}, "line %d leaks corpus metadata" % line_number)
            urls = corpus["source_urls"]
            _require(isinstance(urls, dict) and set(urls) == EXPECTED_URLS,
                     "line %d source_urls schema mismatch" % line_number)
            _require(all(value == [] for value in urls.values()),
                     "line %d leaks source URLs" % line_number)
            _require(sample["native_metadata"] == {}, "line %d leaks sample metadata" % line_number)
            sample_id = sample["sample_id"]
            _require(isinstance(sample_id, str) and ID_RE.match(sample_id),
                     "line %d has invalid sample_id" % line_number)
            _require(sample_id not in seen_ids, "duplicate sample_id: %s" % sample_id)
            audio = sample["audio"]
            _require(isinstance(audio, dict) and set(audio) == EXPECTED_AUDIO,
                     "line %d audio schema mismatch" % line_number)
            raw_audio = audio["path"]
            _require(isinstance(raw_audio, str) and raw_audio,
                     "line %d has empty audio path" % line_number)
            audio_path = Path(raw_audio)
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            # Validate the lexical staging path before resolving symlinks.
            # The runtime bundle intentionally uses flat symlinks whose real
            # targets live under the benchmark source tree.
            lexical_audio_path = Path(os.path.abspath(str(audio_path)))
            try:
                lexical_audio_path.relative_to(allowed_root)
            except ValueError:
                raise ValueError("line %d audio path escapes allowed root: %s" % (line_number, lexical_audio_path))
            audio_path = lexical_audio_path.resolve()
            _require(audio_path.is_file(), "line %d audio file missing: %s" % (line_number, audio_path))
            _require(audio_path not in seen_audio, "duplicate audio path: %s" % audio_path)
            text = sample["text"]
            _require(isinstance(text, dict) and set(text) == EXPECTED_TEXT,
                     "line %d text schema mismatch" % line_number)
            _require(text["transcript"] == "", "line %d contains transcript" % line_number)
            # Keep the original closed-schema record.  The sure-tagger pipeline
            # needs the sample ID/path for local dispatch, but no GT fields are
            # present and no record is serialized into an API request.
            rows.append((sample_id, line))
            seen_ids.add(sample_id)
            seen_audio.add(audio_path)
    _require(rows, "runtime manifest is empty")
    return rows


def _duration_sec(row, manifest_dir):
    record = json.loads(row[1])
    path = Path(record["sample"]["audio"]["path"])
    if not path.is_absolute():
        path = manifest_dir / path
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / float(source.getframerate())


def _duration_balanced_partitions(rows, gpu_count, manifest_dir):
    weighted = [
        (_duration_sec(row, manifest_dir), index, row)
        for index, row in enumerate(rows)
    ]
    partitions = [[] for _ in range(gpu_count)]
    duration_totals = [0.0] * gpu_count
    for duration, index, row in sorted(
        weighted, key=lambda item: (-item[0], item[1])
    ):
        target = min(
            range(gpu_count),
            key=lambda gpu: (
                duration_totals[gpu],
                len(partitions[gpu]),
                gpu,
            ),
        )
        partitions[target].append((index, row))
        duration_totals[target] += duration
    return (
        [
            [row for _index, row in sorted(partition)]
            for partition in partitions
        ],
        duration_totals,
    )


def prepare(
    runtime_manifest,
    output_dir,
    audio_root,
    gpu_count,
    force=False,
    balance_by_duration=False,
    workers_per_gpu=1,
):
    runtime_manifest = Path(runtime_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    rows = load_rows(runtime_manifest, audio_root)
    gpu_count = int(gpu_count)
    _require(gpu_count > 0, "gpu_count must be positive")
    workers_per_gpu = int(workers_per_gpu)
    _require(workers_per_gpu > 0, "workers_per_gpu must be positive")
    if balance_by_duration:
        partitions, partition_durations = _duration_balanced_partitions(
            rows, gpu_count, runtime_manifest.parent
        )
    else:
        partitions = [[] for _ in range(gpu_count)]
        for index, row in enumerate(rows):
            partitions[index % gpu_count].append(row)
        partition_durations = [
            sum(_duration_sec(row, runtime_manifest.parent) for row in partition)
            for partition in partitions
        ]

    for stage in ("core", "aux"):
        (output_dir / stage).mkdir(parents=True, exist_ok=True)
        for gpu, partition in enumerate(partitions):
            target = output_dir / stage / ("manifest_%02d.jsonl" % gpu)
            if target.exists() and not force:
                existing = target.read_text(encoding="utf-8")
                expected = "".join(line if line.endswith("\n") else line + "\n" for _, line in partition)
                if existing != expected:
                    raise ValueError("refusing to overwrite staged manifest: %s" % target)
                continue
            text = "".join(line if line.endswith("\n") else line + "\n" for _, line in partition)
            atomic_write(target, text)

    manifest_hash = sha256_file(runtime_manifest)
    summary = {
        "audio_only": True,
        "gpu_count": gpu_count,
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": manifest_hash,
        "sample_count": len(rows),
        "partition_counts": [len(partition) for partition in partitions],
        "partition_duration_sec": [
            round(duration, 6) for duration in partition_durations
        ],
        "partition_strategy": (
            "longest_processing_time_by_audio_duration"
            if balance_by_duration
            else "round_robin_by_sample_count"
        ),
        "stage_manifest_dir": str(output_dir),
        "stages": ["core", "aux"],
        "workers_per_gpu": workers_per_gpu,
    }
    atomic_write(
        output_dir / "prepare.summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--gpu-count", type=int, default=7)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--balance-by-duration", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare(
        args.runtime_manifest,
        args.output_dir,
        args.audio_root,
        args.gpu_count,
        force=args.force,
        balance_by_duration=args.balance_by_duration,
        workers_per_gpu=args.workers_per_gpu,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
