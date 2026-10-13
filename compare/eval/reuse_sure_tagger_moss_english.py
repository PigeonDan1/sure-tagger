#!/usr/bin/env python3
"""Reuse historical English-MOSS tags with the current T1-T3 result set.

The current official refresh already contains the correct acoustic groups and
the restored ``basic_acoustic.nonspeech_ratio`` values.  The staged core run
contains a complete English-MOSS run for the same samples.  This utility joins
only the ASR-owned groups from that run and leaves all other current fields
unchanged; no model is started.
"""

from __future__ import print_function

import argparse
import copy
import gzip
import json
from pathlib import Path


REUSED_GROUPS = ("speaker", "language_content", "deterministic_derived")
GROUPS = (
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
)


def _is_chinese(value):
    value = str(value or "").strip().lower()
    return value == "zh" or value.startswith(("zh-", "zh_", "zh "))


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row %s:%d is not an object" % (path, line_number))
            rows.append(value)
    if not rows:
        raise ValueError("empty JSONL: %s" % path)
    return rows


def _load_current(path):
    result = {}
    order = []
    for row in _load_jsonl(path):
        sample_id = row.get("sample_id")
        tags = row.get("tags")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("current result row has no sample_id")
        if sample_id in result:
            raise ValueError("duplicate current sample_id: %s" % sample_id)
        if not isinstance(tags, dict) or set(tags) != set(GROUPS):
            raise ValueError("current tags schema mismatch: %s" % sample_id)
        if sample_id.lower().startswith("t4_"):
            raise ValueError("T4 row is not allowed: %s" % sample_id)
        result[sample_id] = tags
        order.append(sample_id)
    return result, order


def _load_staged_core(root):
    root = Path(root)
    tags_by_id = {}
    manifest_root = root.parent / "manifests" / root.name
    for manifest_path in sorted(manifest_root.glob("manifest_*.jsonl")):
        shard = manifest_path.stem.split("_", 1)[1]
        tags_path = root / ("tags_%s.jsonl" % shard)
        if not tags_path.is_file():
            raise ValueError("missing staged tags for %s" % manifest_path)
        manifests = _load_jsonl(manifest_path)
        tags = _load_jsonl(tags_path)
        if len(manifests) != len(tags):
            raise ValueError("manifest/tags mismatch for shard %s" % shard)
        for manifest, value in zip(manifests, tags):
            sample_id = manifest.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("staged manifest has no sample_id")
            if sample_id in tags_by_id:
                raise ValueError("duplicate staged sample_id: %s" % sample_id)
            if not isinstance(value, dict):
                raise ValueError("staged tags are not an object: %s" % sample_id)
            for group in REUSED_GROUPS:
                if group not in value or not isinstance(value[group], dict):
                    raise ValueError("staged group missing for %s: %s" % (sample_id, group))
            tags_by_id[sample_id] = value

    route_by_id = {}
    artifact_root = root / "artifacts"
    for artifact in artifact_root.rglob("fusion_artifact_v2.json.gz"):
        with gzip.open(str(artifact), "rt", encoding="utf-8") as stream:
            value = json.load(stream)
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            sample_id = artifact.parent.name.rsplit("-sample-", 1)[0]
        route_by_id[sample_id] = value.get("asr_route", {})
    return tags_by_id, route_by_id


def reuse(current_result, staged_core, output_dir):
    current, order = _load_current(current_result)
    staged, routes = _load_staged_core(staged_core)
    missing = sorted(set(order) - set(staged))
    if missing:
        raise ValueError("staged core is missing %d current IDs" % len(missing))

    english_ids = {
        sample_id
        for sample_id in order
        if routes.get(sample_id, {}).get("language_route") == "pure_english"
        and routes.get(sample_id, {}).get("selected_source") == "moss_transcribe_diarize"
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_lines = []
    tags_lines = []
    assignment_lines = []
    route_counts = {"moss_english_reused": 0, "moss_chinese_existing": 0, "official_other_existing": 0}
    for sample_id in order:
        tags = copy.deepcopy(current[sample_id])
        if sample_id in english_ids:
            route_name = "english_moss_reused"
            for group in REUSED_GROUPS:
                tags[group] = copy.deepcopy(staged[sample_id][group])
            route_counts["moss_english_reused"] += 1
        elif _is_chinese(tags.get("language_content", {}).get("language")):
            route_name = "chinese_moss_existing"
            route_counts["moss_chinese_existing"] += 1
        else:
            route_name = "official_diarizen_existing"
            route_counts["official_other_existing"] += 1
        tags_lines.append(json.dumps(tags, ensure_ascii=False, sort_keys=True) + "\n")
        result_lines.append(json.dumps({"sample_id": sample_id, "tags": tags}, ensure_ascii=False, sort_keys=True) + "\n")
        assignment_lines.append(json.dumps({"sample_id": sample_id, "route": route_name}, ensure_ascii=False, sort_keys=True) + "\n")

    (output_dir / "tags.jsonl").write_text("".join(tags_lines), encoding="utf-8")
    (output_dir / "result_with_id.jsonl").write_text("".join(result_lines), encoding="utf-8")
    (output_dir / "route_assignments.jsonl").write_text("".join(assignment_lines), encoding="utf-8")
    summary = {
        "schema_version": "sure-tagger-route-reuse.v1",
        "sample_count": len(order),
        "t4_excluded": True,
        "current_result": str(Path(current_result).resolve()),
        "staged_core": str(Path(staged_core).resolve()),
        "reused_groups": list(REUSED_GROUPS),
        "route_counts": route_counts,
        "route_assignments": str((output_dir / "route_assignments.jsonl").resolve()),
        "deepseek_rerun": False,
        "sure_tagger_rerun": False,
    }
    (output_dir / "reuse.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-result", required=True)
    parser.add_argument("--staged-core", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(reuse(args.current_result, args.staged_core, args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
