#!/usr/bin/env python3
"""Attach external sample IDs to tags-only sure-tagger shard outputs.

The sure-tagger public JSONL deliberately has no sample identifier.  This
utility joins each line to its input manifest by position, verifies complete
coverage and schema, then writes a tags-only file and an ID-keyed sidecar for
downstream extraction.  No benchmark metadata is copied into either output.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


GROUP_FIELDS = {
    "basic_acoustic": {
        "duration_sec", "sample_rate_hz", "channels", "nonspeech_ratio",
        "silence_segments",
    },
    "audio_quality": {
        "snr_db", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808",
    },
    "room_acoustic": {"far_field", "rt60_sec", "c50_db"},
    "sound_field_scene": {
        "speech_music_events", "music_present", "external_noise_type",
        "noise_composition",
    },
    "speaker": {
        "speaker_count", "speaker_present", "asr_transcript", "multi_speaker",
        "speaker_change_count", "speaker_change", "overlap_ratio",
        "speaker_overlap", "profiles", "speaker_transcripts",
    },
    "language_content": {
        "language", "word_count", "punctuation", "repetition", "filler",
    },
    "deterministic_derived": {
        "speaker_change_rate_per_min", "filler_ratio",
    },
}
GROUP_ORDER = (
    "basic_acoustic", "audio_quality", "room_acoustic", "sound_field_scene",
    "speaker", "language_content", "deterministic_derived",
)


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


def _load_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid manifest JSON at %s:%d: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("manifest row %s:%d is not an object" % (path, line_number))
            sample = value.get("sample")
            if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
                raise ValueError("manifest row %s:%d lacks sample_id" % (path, line_number))
            rows.append(sample["sample_id"])
    if not rows:
        raise ValueError("manifest is empty: %s" % path)
    return rows


def _validate_tags(value, location):
    if not isinstance(value, dict) or set(value) != set(GROUP_ORDER):
        raise ValueError("tags schema mismatch at %s" % location)
    for group in GROUP_ORDER:
        fields = value.get(group)
        expected = GROUP_FIELDS[group]
        if group == "speaker" and isinstance(fields, dict) and "speaker_transcripts" not in fields:
            expected = expected - {"speaker_transcripts"}
        if not isinstance(fields, dict) or set(fields) != expected:
            raise ValueError("tags fields mismatch at %s (%s)" % (location, group))
    if value["speaker"].get("speaker_transcripts") is not None:
        from extract_captioner_deepseek_once import validate_speaker_transcripts
        validate_speaker_transcripts(value["speaker"])
    # The public output must remain tags-only.  In particular, reject a future
    # accidental sample_id/audio/GT field instead of silently dropping it.
    return value


def _read_tags(path, expected_count):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank tags line at %s:%d" % (path, line_number))
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid tags JSON at %s:%d: %s" % (path, line_number, exc))
            rows.append(_validate_tags(value, "%s:%d" % (path, line_number)))
    if len(rows) != expected_count:
        raise ValueError(
            "line count mismatch for %s: expected %d, got %d"
            % (path, expected_count, len(rows))
        )
    return rows


def finalize(runtime_manifest, manifest_dir, tags_dir, output_dir, gpu_count):
    runtime_manifest = Path(runtime_manifest).resolve()
    manifest_dir = Path(manifest_dir).resolve()
    tags_dir = Path(tags_dir).resolve()
    output_dir = Path(output_dir).resolve()
    gpu_count = int(gpu_count)
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")

    runtime_ids = _load_manifest(runtime_manifest)
    if len(set(runtime_ids)) != len(runtime_ids):
        raise ValueError("runtime manifest contains duplicate sample IDs")
    by_id = {}
    shard_counts = []
    for gpu in range(gpu_count):
        name = "%02d" % gpu
        manifest_path = manifest_dir / ("manifest_%s.jsonl" % name)
        tags_path = tags_dir / ("tags_%s.jsonl" % name)
        if not manifest_path.is_file():
            raise ValueError("missing stage manifest: %s" % manifest_path)
        if not tags_path.is_file():
            raise ValueError("missing tags output: %s" % tags_path)
        ids = _load_manifest(manifest_path)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample IDs in %s" % manifest_path)
        tags = _read_tags(tags_path, len(ids))
        for sample_id, value in zip(ids, tags):
            if sample_id in by_id:
                raise ValueError("duplicate sample ID across shards: %s" % sample_id)
            by_id[sample_id] = value
        shard_counts.append(len(ids))

    expected = set(runtime_ids)
    actual = set(by_id)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "coverage mismatch (missing=%d, extra=%d)" % (len(missing), len(extra))
        )

    tags_lines = []
    keyed_lines = []
    for sample_id in runtime_ids:
        value = by_id[sample_id]
        tags_lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        keyed_lines.append(
            json.dumps(
                {"sample_id": sample_id, "tags": value},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "tags.jsonl", "".join(tags_lines))
    atomic_write(output_dir / "result_with_id.jsonl", "".join(keyed_lines))
    summary = {
        "audio_only": True,
        "gpu_count": gpu_count,
        "manifest_dir": str(manifest_dir),
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256_file(runtime_manifest),
        "sample_count": len(runtime_ids),
        "shard_counts": shard_counts,
        "tags_path": str(output_dir / "tags.jsonl"),
        "result_with_id_path": str(output_dir / "result_with_id.jsonl"),
        "keyed_input_contract": "sample_id_outside_api_tags_only_inside_api",
    }
    atomic_write(
        output_dir / "finalize.summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--tags-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-count", type=int, default=7)
    args = parser.parse_args(argv)
    summary = finalize(
        args.runtime_manifest,
        args.manifest_dir,
        args.tags_dir,
        args.output_dir,
        args.gpu_count,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
