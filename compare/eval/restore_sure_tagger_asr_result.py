#!/usr/bin/env python3
"""Restore the original speaker labels while keeping refreshed ASR fields."""

from __future__ import print_function

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path


SPEAKER_FIELDS = (
    "speaker_count",
    "speaker_present",
    "asr_transcript",
    "multi_speaker",
    "speaker_change_count",
    "speaker_change",
    "overlap_ratio",
    "speaker_overlap",
    "profiles",
)


def _read(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank line at %s:%d" % (path, line_number))
            row = json.loads(line)
            sample_id = row.get("sample_id")
            tags = row.get("tags")
            if not isinstance(sample_id, str) or not isinstance(tags, dict):
                raise ValueError("invalid row at %s:%d" % (path, line_number))
            if sample_id in rows:
                raise ValueError("duplicate sample_id: %s" % sample_id)
            rows[sample_id] = tags
    return rows


def _atomic_write(path, text):
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


def _speaker_change_rate(tags):
    count = tags["speaker"].get("speaker_change_count")
    duration = tags["basic_acoustic"].get("duration_sec")
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration > 0
    ):
        return round(float(count) * 60.0 / float(duration), 6)
    return None


def restore(baseline_path, refreshed_path, output_dir):
    baseline = _read(baseline_path)
    refreshed = _read(refreshed_path)
    if set(baseline) != set(refreshed):
        raise ValueError("baseline/refreshed sample coverage differs")

    keyed_lines = []
    tags_lines = []
    changed_transcripts = 0
    restored_speaker_rows = 0
    for sample_id in baseline:
        old = baseline[sample_id]
        new = refreshed[sample_id]
        tags = copy.deepcopy(old)
        old_speaker = tags["speaker"]
        new_speaker = new["speaker"]
        for field in SPEAKER_FIELDS:
            if field not in old_speaker or field not in new_speaker:
                raise ValueError("missing speaker field %s for %s" % (field, sample_id))

        transcript = new_speaker.get("asr_transcript")
        if transcript != old_speaker.get("asr_transcript"):
            changed_transcripts += 1
        old_speaker["asr_transcript"] = transcript

        # Language and text-derived fields belong to the refreshed ASR output.
        tags["language_content"] = copy.deepcopy(new["language_content"])
        tags["deterministic_derived"] = copy.deepcopy(new["deterministic_derived"])
        # Recompute this field from the restored original speaker timeline.
        tags["deterministic_derived"]["speaker_change_rate_per_min"] = (
            _speaker_change_rate(tags)
        )

        keyed_lines.append(
            json.dumps(
                {"sample_id": sample_id, "tags": tags},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        tags_lines.append(
            json.dumps(tags, ensure_ascii=False, sort_keys=True) + "\n"
        )
        restored_speaker_rows += 1

    output_dir = Path(output_dir)
    _atomic_write(output_dir / "tags.jsonl", "".join(tags_lines))
    _atomic_write(output_dir / "result_with_id.jsonl", "".join(keyed_lines))
    summary = {
        "schema_version": "sure-tagger-asr-result-restore.v1",
        "baseline": str(Path(baseline_path).resolve()),
        "refreshed": str(Path(refreshed_path).resolve()),
        "sample_count": len(keyed_lines),
        "restored_speaker_rows": restored_speaker_rows,
        "changed_asr_transcripts": changed_transcripts,
        "preserved_speaker_fields": [
            field for field in SPEAKER_FIELDS if field != "asr_transcript"
        ],
        "refreshed_fields": [
            "speaker.asr_transcript",
            "language_content",
            "deterministic_derived.filler_ratio",
            "deterministic_derived.speaker_change_rate_per_min",
        ],
    }
    _atomic_write(
        output_dir / "restore.summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--refreshed", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            restore(args.baseline, args.refreshed, args.output_dir),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
