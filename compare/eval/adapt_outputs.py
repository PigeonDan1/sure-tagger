#!/usr/bin/env python3
"""Adapt keyed system outputs to the structural benchmark prediction format."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_UNIVERSE = SCRIPT_DIR / "field_universe.structural-v0.1.json"
DEFAULT_GT = COMPARE_ROOT / "evaluation" / "ami-v0" / "ground_truth.jsonl"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("%s:%d is invalid JSON: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("%s:%d must contain an object" % (path, line_number))
            rows.append(value)
    return rows


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_jsonl(path, rows):
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_specs(universe):
    output = {}
    for field in universe.get("fields", []):
        path = field.get("path") if isinstance(field, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError("field universe contains an invalid path")
        if path in output:
            raise ValueError("field universe contains duplicate path %s" % path)
        output[path] = field
    if not output:
        raise ValueError("field universe has no fields")
    return output


def nested_value(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def valid_kind(value, kind):
    if value is None:
        return True
    if kind == "number":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
    if kind == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string_array":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if kind == "object":
        return isinstance(value, dict)
    return False


def _target_payload(source):
    if isinstance(source.get("tags"), dict):
        return source["tags"]
    groups = (
        "basic_acoustic",
        "audio_quality",
        "room_acoustic",
        "sound_field_scene",
        "speaker",
        "language_content",
        "deterministic_derived",
    )
    if any(group in source for group in groups):
        return source
    return None


def adapt_target_payload(source, fields):
    payload = _target_payload(source)
    if payload is None:
        return None
    values = {path: None for path in fields}
    provided = []
    for path in fields:
        exists, value = nested_value(payload, path)
        if exists:
            values[path] = value
            provided.append(path)
    return {
        "values": values,
        "provided_fields": sorted(provided),
        "mapped_fields": [],
        "source_format": "target_structural_schema",
    }


def adapt_legacy_captioner(source, fields, universe):
    annotations = source.get("annotation")
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ValueError("legacy captioner output must contain one annotation")
    annotation = annotations[0]
    if not isinstance(annotation, dict):
        raise ValueError("legacy captioner annotation must be an object")
    values = {path: None for path in fields}
    provided = []
    mapped = []

    speaker = annotation.get("speaker")
    if isinstance(speaker, dict):
        for field in (
            "speaker_count",
            "multi_speaker",
            "speaker_change",
            "speaker_overlap",
            "profiles",
        ):
            if field in speaker:
                path = "speaker.%s" % field
                if path in values:
                    values[path] = speaker[field]
                    provided.append(path)
                    mapped.append(path)

    audio = annotation.get("audio")
    if isinstance(audio, dict) and "tag" in audio:
        raw_tags = audio.get("tag")
        path = "sound_field_scene.speech_music_events"
        if raw_tags is None:
            values[path] = None
        elif isinstance(raw_tags, list):
            allowed = universe.get("multilabel_classes", {}).get(path, [])
            values[path] = [label for label in allowed if label in raw_tags]
        else:
            values[path] = raw_tags
        provided.append(path)
        mapped.append(path)

    return {
        "values": values,
        "provided_fields": sorted(set(provided)),
        "mapped_fields": sorted(set(mapped)),
        "source_format": "legacy_captioner_annotation",
    }


def source_sample_id(source, fallback=None):
    for key in ("sample_id", "id", "key"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def load_sources(input_path):
    input_path = Path(input_path).resolve()
    sources = {}
    source_paths = {}
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.json"), key=lambda path: path.name)
        paths = [path for path in paths if path.name != "_summary.json"]
        if not paths:
            raise ValueError("no JSON outputs found in %s" % input_path)
        for path in paths:
            try:
                value = load_json(path)
            except (OSError, ValueError) as exc:
                value = {"sample_id": path.stem, "_adapter_parse_error": str(exc)}
            sample_id = source_sample_id(value, path.stem)
            if sample_id in sources:
                raise ValueError("duplicate source sample_id %s" % sample_id)
            sources[sample_id] = value
            source_paths[sample_id] = str(path)
    elif input_path.is_file():
        for line_number, value in enumerate(load_jsonl(input_path), 1):
            sample_id = source_sample_id(value)
            if not sample_id:
                raise ValueError(
                    "%s:%d has no sample_id; line-order adaptation is forbidden"
                    % (input_path, line_number)
                )
            if sample_id in sources:
                raise ValueError("duplicate source sample_id %s" % sample_id)
            sources[sample_id] = value
            source_paths[sample_id] = "%s:%d" % (input_path, line_number)
    else:
        raise FileNotFoundError(str(input_path))
    return sources, source_paths, input_path


def _source_failed(source):
    status = source.get("status")
    return status in ("failed", "error", "runtime_failed")


def _schema_errors(adapted, fields, universe):
    errors = []
    provided = set(adapted["provided_fields"])
    for path in provided:
        value = adapted["values"].get(path)
        if not valid_kind(value, fields[path].get("kind")):
            errors.append("%s has invalid %s value" % (path, fields[path].get("kind")))
    events_path = "sound_field_scene.speech_music_events"
    if events_path in provided and isinstance(adapted["values"].get(events_path), list):
        allowed = set(universe.get("multilabel_classes", {}).get(events_path, []))
        unexpected = sorted(set(adapted["values"][events_path]) - allowed)
        if unexpected:
            errors.append("%s has unknown labels: %s" % (events_path, unexpected))
        if len(adapted["values"][events_path]) != len(
            set(adapted["values"][events_path])
        ):
            errors.append("%s has duplicate labels" % events_path)
    return errors


def adapt_one(system, sample_id, source, source_path, fields, universe):
    empty = {path: None for path in fields}
    if source is None:
        return {
            "schema_version": "structural-prediction-v0.1",
            "field_universe_version": universe["universe_version"],
            "system": system,
            "sample_id": sample_id,
            "status": "runtime_failed",
            "schema_valid": True,
            "adapter_schema_valid": True,
            "native_current_schema_valid": False,
            "native_schema_errors": ["source record is missing"],
            "schema_errors": [],
            "values": empty,
            "provided_fields": [],
            "mapped_fields": [],
            "source_format": "missing",
            "source_path": None,
        }
    if source.get("_adapter_parse_error"):
        return {
            "schema_version": "structural-prediction-v0.1",
            "field_universe_version": universe["universe_version"],
            "system": system,
            "sample_id": sample_id,
            "status": "schema_invalid",
            "schema_valid": False,
            "adapter_schema_valid": False,
            "native_current_schema_valid": False,
            "native_schema_errors": [source["_adapter_parse_error"]],
            "schema_errors": [source["_adapter_parse_error"]],
            "values": empty,
            "provided_fields": [],
            "mapped_fields": [],
            "source_format": "invalid_json",
            "source_path": source_path,
        }
    if _source_failed(source):
        return {
            "schema_version": "structural-prediction-v0.1",
            "field_universe_version": universe["universe_version"],
            "system": system,
            "sample_id": sample_id,
            "status": "runtime_failed",
            "schema_valid": True,
            "adapter_schema_valid": True,
            "native_current_schema_valid": False,
            "native_schema_errors": ["source runtime failed"],
            "schema_errors": [],
            "values": empty,
            "provided_fields": [],
            "mapped_fields": [],
            "source_format": "failed_source_record",
            "source_path": source_path,
        }
    adapted = adapt_target_payload(source, fields)
    if adapted is None and system == "captioner":
        try:
            adapted = adapt_legacy_captioner(source, fields, universe)
        except ValueError as exc:
            adapted = {
                "values": empty,
                "provided_fields": [],
                "mapped_fields": [],
                "source_format": "invalid_captioner_annotation",
                "adapter_error": str(exc),
            }
    if adapted is None:
        adapted = {
            "values": empty,
            "provided_fields": [],
            "mapped_fields": [],
            "source_format": "unrecognized",
            "adapter_error": "unrecognized structural output format",
        }
    errors = _schema_errors(adapted, fields, universe)
    if adapted.get("adapter_error"):
        errors.append(adapted["adapter_error"])
    provided = set(adapted["provided_fields"])
    missing_native_fields = sorted(set(fields) - provided)
    native_errors = []
    if adapted["source_format"] == "legacy_captioner_annotation":
        native_errors.append("legacy captioner annotation is not the current structural schema")
    if missing_native_fields:
        native_errors.append(
            "missing current structural fields: %s" % ", ".join(missing_native_fields)
        )
    if errors:
        native_errors.extend(errors)
    native_valid = not native_errors
    source_format = adapted["source_format"]
    if system == "sure-tagger" and source_format == "target_structural_schema" and not native_valid:
        source_format = "legacy_sure_tags_six_group"
    return {
        "schema_version": "structural-prediction-v0.1",
        "field_universe_version": universe["universe_version"],
        "system": system,
        "sample_id": sample_id,
        "status": "schema_invalid" if errors else "ok",
        "schema_valid": not errors,
        "adapter_schema_valid": not errors,
        "native_current_schema_valid": native_valid,
        "native_schema_errors": native_errors,
        "schema_errors": errors,
        "values": adapted["values"],
        "provided_fields": adapted["provided_fields"],
        "mapped_fields": adapted["mapped_fields"],
        "source_format": source_format,
        "source_path": source_path,
    }


def adapt(system, input_path, ground_truth_path, universe_path, output_path):
    if system not in ("sure-tagger", "captioner"):
        raise ValueError("unsupported system: %s" % system)
    universe = load_json(universe_path)
    fields = field_specs(universe)
    ground_truth = load_jsonl(ground_truth_path)
    gt_ids = [row.get("sample_id") for row in ground_truth]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in gt_ids):
        raise ValueError("ground truth contains an invalid sample_id")
    if len(gt_ids) != len(set(gt_ids)):
        raise ValueError("ground truth contains duplicate sample IDs")
    sources, source_paths, resolved_input = load_sources(input_path)
    unexpected = sorted(set(sources) - set(gt_ids))
    if unexpected:
        raise ValueError("source output has unexpected sample IDs: %s" % unexpected[:10])
    records = [
        adapt_one(
            system,
            sample_id,
            sources.get(sample_id),
            source_paths.get(sample_id),
            fields,
            universe,
        )
        for sample_id in gt_ids
    ]
    write_jsonl(output_path, records)
    status_counts = {}
    format_counts = {}
    native_valid_count = 0
    provided_total = 0
    for row in records:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        source_format = row["source_format"]
        format_counts[source_format] = format_counts.get(source_format, 0) + 1
        native_valid_count += row["native_current_schema_valid"] is True
        provided_total += len(row["provided_fields"])
    summary = {
        "schema_version": "structural-adapter-summary-v0.1",
        "field_universe_version": universe["universe_version"],
        "system": system,
        "input": str(resolved_input),
        "input_sha256": sha256_file(resolved_input) if resolved_input.is_file() else None,
        "ground_truth": str(Path(ground_truth_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "sample_count": len(records),
        "source_record_count": len(sources),
        "missing_source_count": len(records) - len(sources),
        "status_counts": status_counts,
        "source_format_counts": format_counts,
        "adapter_schema_valid_count": sum(row["adapter_schema_valid"] for row in records),
        "native_current_schema_valid_count": native_valid_count,
        "native_current_schema_valid_rate": (
            float(native_valid_count) / float(len(records)) if records else None
        ),
        "mean_provided_field_fraction": (
            float(provided_total) / float(len(records) * len(fields)) if records else None
        ),
    }
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return summary


def default_input(system):
    if system == "sure-tagger":
        return COMPARE_ROOT / "dataset" / "Ami" / "sure-tagger_result" / "result_with_audio.jsonl"
    return COMPARE_ROOT / "dataset" / "Ami" / "captioner_result" / "tag_extracted"


def default_output(system):
    return COMPARE_ROOT / "evaluation" / "ami-v0" / (system + ".predictions.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=("sure-tagger", "captioner"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--field-universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_path = args.input or str(default_input(args.system))
    output_path = args.output or str(default_output(args.system))
    summary = adapt(
        args.system,
        input_path,
        args.ground_truth,
        args.field_universe,
        output_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
