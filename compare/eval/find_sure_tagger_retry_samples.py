#!/usr/bin/env python3
"""Find samples whose speaker evidence contains a retryable model failure."""

from __future__ import print_function

import argparse
import gzip
import json
from pathlib import Path


RETRYABLE_MARKERS = (
    "out of memory",
    "cuda oom",
    "memoryerror",
    "embedding inference failed",
    "model loading failed",
    "subprocess worker exited",
)


def _expected_samples(manifest_dir=None, retry_input=None):
    expected = set()
    if manifest_dir is not None:
        for manifest in sorted(Path(manifest_dir).glob("manifest_*.jsonl")):
            shard = "shard_%s" % manifest.stem.split("_")[-1]
            with manifest.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        expected.add((shard, row["sample"]["sample_id"]))
    if retry_input is not None:
        with Path(retry_input).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    shard, sample_id = line.rstrip("\n").split("\t", 2)[:2]
                    expected.add((shard, sample_id))
    return expected


def find_retry_samples(artifact_root, required_source=None, expected_samples=None):
    artifact_root = Path(artifact_root).resolve()
    failures = {}
    observed = set()
    pattern = "shard_*/speaker_v2/*/evidence/*.json.gz"
    for path in sorted(artifact_root.glob(pattern)):
        try:
            with gzip.open(str(path), "rt", encoding="utf-8") as stream:
                evidence = json.load(stream)
        except (EOFError, OSError, ValueError):
            continue
        quality = evidence.get("quality") or {}
        reason = str(quality.get("reason") or "").lower()
        scope = evidence.get("scope", {})
        sample_id = scope.get("sample_id")
        shard = next(
            (part for part in path.parts if part.startswith("shard_")),
            None,
        )
        if not isinstance(sample_id, str) or not sample_id or not shard:
            continue
        key = (shard, sample_id)
        if expected_samples is not None and key not in expected_samples:
            continue
        source = (evidence.get("source") or {}).get("name")
        if source == required_source:
            observed.add(key)
            if evidence.get("status") not in ("estimated", "observed") or quality.get("usable") is False:
                failures[key] = reason or "required source is unusable"
        elif any(marker in reason for marker in RETRYABLE_MARKERS):
            failures.setdefault(key, reason)
    if required_source and expected_samples is not None:
        for key in expected_samples - observed:
            failures[key] = "required source evidence is missing: %s" % required_source
    return sorted((shard, sample_id, reason) for (shard, sample_id), reason in failures.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--required-source", default=None)
    parser.add_argument("--expected-manifest-dir", type=Path)
    parser.add_argument("--expected-samples-tsv", type=Path)
    args = parser.parse_args(argv)
    if args.expected_manifest_dir and args.expected_samples_tsv:
        parser.error("choose one expected-sample input")
    expected = None
    if args.expected_manifest_dir or args.expected_samples_tsv:
        expected = _expected_samples(args.expected_manifest_dir, args.expected_samples_tsv)
        if not expected:
            parser.error("expected-sample input is empty")
    failures = find_retry_samples(args.artifact_root, args.required_source, expected)
    for shard, sample_id, reason in failures:
        print("%s\t%s\t%s" % (shard, sample_id, reason.replace("\t", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
