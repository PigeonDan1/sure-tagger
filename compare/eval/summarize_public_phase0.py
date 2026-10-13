#!/usr/bin/env python3
"""Audit public Phase 0 artifacts and render JSON and Markdown summaries.

The captioner ``caption`` field is a raw runtime artifact. This program never
parses caption prose into tags or credits it as a current-schema prediction.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import wave

import adapt_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = COMPARE_ROOT / "evaluation" / "public-phase0"
DEFAULT_UNIVERSE = SCRIPT_DIR / "field_universe.structural-v0.1.json"
SUMMARY_SCHEMA_VERSION = "public-phase0-summary-v0.1"
EXPECTED_SAMPLE_COUNT = 10

DATASET_SPECS = (
    {
        "key": "esc50",
        "name": "ESC-50",
        "manifest": "esc50_pilot.manifest.jsonl",
        "metadata": "esc50_pilot.metadata.jsonl",
        "sure": "esc50_sure_header.tags.jsonl",
        "captioner": "captioner_raw/esc50/results.jsonl",
        "captioner_summary": "captioner_raw/esc50/summary.json",
        "captioner_dataset_names": ["esc50", "ESC-50", "ESC-50_phase0"],
        "expected_count": EXPECTED_SAMPLE_COUNT,
    },
    {
        "key": "librispeech",
        "name": "LibriSpeech",
        "manifest": "librispeech_manifest.jsonl",
        "metadata": "librispeech_pilot_metadata.jsonl",
        "sure": "librispeech_sure_header.tags.jsonl",
        "captioner": "captioner_raw/librispeech/results.jsonl",
        "captioner_summary": "captioner_raw/librispeech/summary.json",
        "captioner_dataset_names": [
            "librispeech",
            "LibriSpeech",
            "LibriSpeech-test-clean-public-phase0",
        ],
        "expected_count": EXPECTED_SAMPLE_COUNT,
    },
    {
        "key": "fleurs",
        "name": "FLEURS",
        "manifest": "fleurs_pilot.manifest.jsonl",
        "metadata": "fleurs_pilot.metadata.jsonl",
        "sure": "fleurs_sure_header.tags.jsonl",
        "captioner": "captioner_raw/fleurs/results.jsonl",
        "captioner_summary": "captioner_raw/fleurs/summary.json",
        "captioner_dataset_names": ["fleurs", "FLEURS"],
        "expected_count": EXPECTED_SAMPLE_COUNT,
    },
)

MANIFEST_TOP_KEYS = set(["corpus", "sample"])
MANIFEST_CORPUS_KEYS = set(["dataset_name", "source_urls", "native_metadata"])
MANIFEST_SOURCE_URL_KEYS = set(
    ["article", "github", "huggingface", "dataset_card"]
)
MANIFEST_SAMPLE_KEYS = set(["sample_id", "audio", "text", "native_metadata"])


class AuditError(ValueError):
    """Raised for malformed artifacts that cannot be audited safely."""


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def add_issue(issues, severity, code, message, dataset=None, component=None):
    issue = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if dataset is not None:
        issue["dataset"] = dataset
    if component is not None:
        issue["component"] = component
    issues.append(issue)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, ValueError) as exc:
        raise AuditError("cannot read JSON %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise AuditError("%s must contain a JSON object" % path)
    return value


def load_jsonl(path):
    rows = []
    physical_lines = 0
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                physical_lines = line_number
                if not line.strip():
                    raise AuditError("%s:%d is blank" % (path, line_number))
                try:
                    value = json.loads(line)
                except ValueError as exc:
                    raise AuditError(
                        "%s:%d contains invalid JSON: %s" % (path, line_number, exc)
                    )
                if not isinstance(value, dict):
                    raise AuditError("%s:%d must contain an object" % (path, line_number))
                rows.append(value)
    except OSError as exc:
        raise AuditError("cannot read JSONL %s: %s" % (path, exc))
    return rows, physical_lines


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(content)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_audio_path(root, raw_path):
    if not isinstance(raw_path, str) or not raw_path:
        raise AuditError("audio path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        common = os.path.commonpath([str(root_resolved), str(resolved)])
    except ValueError:
        common = None
    if common != str(root_resolved):
        raise AuditError("audio path escapes evaluation root: %s" % raw_path)
    return resolved


def exact_keys(value, expected, label):
    if not isinstance(value, dict):
        return ["%s must be an object" % label]
    actual = set(value)
    errors = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append("%s missing keys: %s" % (label, ", ".join(missing)))
    if extra:
        errors.append("%s has extra keys: %s" % (label, ", ".join(extra)))
    return errors


def manifest_errors(record):
    errors = exact_keys(record, MANIFEST_TOP_KEYS, "record")
    if errors:
        return errors
    corpus = record["corpus"]
    sample = record["sample"]
    errors.extend(exact_keys(corpus, MANIFEST_CORPUS_KEYS, "corpus"))
    errors.extend(exact_keys(sample, MANIFEST_SAMPLE_KEYS, "sample"))
    if errors:
        return errors
    if not isinstance(corpus["dataset_name"], str):
        errors.append("corpus.dataset_name must be a string")
    errors.extend(
        exact_keys(corpus["source_urls"], MANIFEST_SOURCE_URL_KEYS, "source_urls")
    )
    if isinstance(corpus.get("source_urls"), dict):
        for key, value in corpus["source_urls"].items():
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                errors.append("source_urls.%s must be an array of strings" % key)
    if not isinstance(corpus["native_metadata"], dict):
        errors.append("corpus.native_metadata must be an object")
    if corpus["native_metadata"] != {}:
        errors.append("corpus.native_metadata must be empty for raw-only input")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        errors.append("sample.sample_id must be a non-empty string")
    errors.extend(exact_keys(sample.get("audio"), set(["path"]), "sample.audio"))
    errors.extend(
        exact_keys(sample.get("text"), set(["transcript"]), "sample.text")
    )
    if isinstance(sample.get("audio"), dict) and not isinstance(
        sample["audio"].get("path"), str
    ):
        errors.append("sample.audio.path must be a string")
    if isinstance(sample.get("text"), dict):
        transcript = sample["text"].get("transcript")
        if not isinstance(transcript, str):
            errors.append("sample.text.transcript must be a string")
        elif transcript != "":
            errors.append("sample.text.transcript must be empty for raw-only input")
    if not isinstance(sample.get("native_metadata"), dict):
        errors.append("sample.native_metadata must be an object")
    elif sample["native_metadata"] != {}:
        errors.append("sample.native_metadata must be empty for raw-only input")
    return errors


def keyed_rows(rows, sample_id_getter, label):
    keyed = {}
    ordered_ids = []
    for index, row in enumerate(rows, 1):
        sample_id = sample_id_getter(row)
        if not isinstance(sample_id, str) or not sample_id:
            raise AuditError("%s row %d has no valid sample_id" % (label, index))
        if sample_id in keyed:
            raise AuditError("%s has duplicate sample_id %s" % (label, sample_id))
        keyed[sample_id] = row
        ordered_ids.append(sample_id)
    return keyed, ordered_ids


def read_wav_header(path):
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise AuditError("invalid WAV %s: %s" % (path, exc))
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frame_count": frame_count,
        "duration_sec": float(frame_count) / float(sample_rate),
        "compression": compression,
    }


def metadata_audio(record):
    audio = record.get("system_input_audio")
    if not isinstance(audio, dict):
        raise AuditError("metadata system_input_audio must be an object")
    path = audio.get("path")
    digest = audio.get("sha256")
    if not isinstance(path, str) or not path:
        raise AuditError("metadata system_input_audio.path is missing")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AuditError("metadata system_input_audio.sha256 is invalid")
    try:
        int(digest, 16)
    except ValueError:
        raise AuditError("metadata system_input_audio.sha256 is invalid")
    header = audio.get("header")
    if header is None:
        header = audio
    if not isinstance(header, dict):
        raise AuditError("metadata system_input_audio header is invalid")
    normalized = {
        "channels": header.get("channels"),
        "sample_rate_hz": header.get("sample_rate_hz"),
        "frame_count": header.get("frame_count", header.get("frames")),
        "duration_sec": header.get("duration_sec"),
        "sample_width_bytes": header.get("sample_width_bytes"),
    }
    return path, digest.lower(), normalized


def empty_field_metric(tolerance=None):
    value = {
        "eligible_count": 0,
        "evaluated_count": 0,
        "correct_count": 0,
        "accuracy": None,
    }
    if tolerance is not None:
        value["tolerance"] = tolerance
    return value


def finish_field_metric(metric):
    metric["accuracy"] = safe_rate(metric["correct_count"], metric["evaluated_count"])
    return metric


def values_match(field, expected, actual, tolerance=None):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if field == "duration_sec":
        if not isinstance(expected, (int, float)) or not isinstance(
            actual, (int, float)
        ):
            return False
        if not math.isfinite(float(expected)) or not math.isfinite(float(actual)):
            return False
        return abs(float(expected) - float(actual)) <= float(tolerance)
    return expected == actual


def artifact_summary(root, relative_path, path, rows=None, physical_lines=None):
    output = {
        "path": relative_path,
        "present": path.is_file(),
    }
    if path.is_file():
        output["sha256"] = sha256_file(path)
    if rows is not None:
        output["record_count"] = len(rows)
        output["physical_line_count"] = physical_lines
    return output


def audit_dataset_inputs(root, spec, issues):
    key = spec["key"]
    expected_count = spec["expected_count"]
    manifest_path = root / spec["manifest"]
    metadata_path = root / spec["metadata"]
    result = {
        "status": "ready",
        "expected_sample_count": expected_count,
        "manifest": artifact_summary(root, spec["manifest"], manifest_path),
        "metadata": artifact_summary(root, spec["metadata"], metadata_path),
        "sample_ids": {},
        "audio": {
            "referenced_count": 0,
            "existing_count": 0,
            "existing_rate": None,
            "path_match_count": 0,
            "path_match_rate": None,
            "sha256_match_count": 0,
            "sha256_match_rate": None,
            "wav_header_valid_count": 0,
            "wav_header_valid_rate": None,
            "canonical_pcm16_16khz_mono_count": 0,
            "canonical_pcm16_16khz_mono_rate": None,
            "metadata_header_fields": {
                "channels": empty_field_metric(),
                "sample_rate_hz": empty_field_metric(),
                "frame_count": empty_field_metric(),
                "duration_sec": empty_field_metric(),
                "sample_width_bytes": empty_field_metric(),
            },
        },
    }
    context = {
        "sample_ids": [],
        "manifest_by_id": {},
        "metadata_by_id": {},
        "audio_paths": {},
        "audio_headers": {},
    }
    missing = []
    for label, path in (("manifest", manifest_path), ("metadata", metadata_path)):
        if not path.is_file():
            missing.append(label)
            add_issue(
                issues,
                "incomplete",
                "missing_%s" % label,
                "%s is missing" % path,
                key,
                "data",
            )
    if missing:
        result["status"] = "missing"
        return result, context

    try:
        manifest_rows, manifest_lines = load_jsonl(manifest_path)
        metadata_rows, metadata_lines = load_jsonl(metadata_path)
        result["manifest"] = artifact_summary(
            root, spec["manifest"], manifest_path, manifest_rows, manifest_lines
        )
        result["metadata"] = artifact_summary(
            root, spec["metadata"], metadata_path, metadata_rows, metadata_lines
        )
        manifest_by_id, manifest_ids = keyed_rows(
            manifest_rows,
            lambda row: row.get("sample", {}).get("sample_id")
            if isinstance(row.get("sample"), dict)
            else None,
            "%s manifest" % key,
        )
        metadata_by_id, metadata_ids = keyed_rows(
            metadata_rows, lambda row: row.get("sample_id"), "%s metadata" % key
        )
    except AuditError as exc:
        add_issue(issues, "error", "invalid_data_jsonl", str(exc), key, "data")
        result["status"] = "invalid"
        return result, context

    schema_error_rows = []
    for index, row in enumerate(manifest_rows, 1):
        errors = manifest_errors(row)
        if errors:
            schema_error_rows.append({"line": index, "errors": errors})
    result["manifest"]["schema_valid_count"] = len(manifest_rows) - len(
        schema_error_rows
    )
    result["manifest"]["schema_valid_rate"] = safe_rate(
        result["manifest"]["schema_valid_count"], len(manifest_rows)
    )
    result["manifest"]["schema_error_rows"] = schema_error_rows[:20]
    if schema_error_rows:
        add_issue(
            issues,
            "error",
            "invalid_manifest_schema",
            "%s manifest has %d invalid records"
            % (key, len(schema_error_rows)),
            key,
            "data",
        )

    if len(manifest_rows) != expected_count or len(metadata_rows) != expected_count:
        add_issue(
            issues,
            "error",
            "unexpected_data_line_count",
            "%s expected %d records, found manifest=%d metadata=%d"
            % (key, expected_count, len(manifest_rows), len(metadata_rows)),
            key,
            "data",
        )
    manifest_set = set(manifest_ids)
    metadata_set = set(metadata_ids)
    missing_metadata = sorted(manifest_set - metadata_set)
    unexpected_metadata = sorted(metadata_set - manifest_set)
    result["sample_ids"] = {
        "manifest_unique_count": len(manifest_set),
        "metadata_unique_count": len(metadata_set),
        "sets_match": not missing_metadata and not unexpected_metadata,
        "order_matches": manifest_ids == metadata_ids,
        "missing_in_metadata": missing_metadata,
        "unexpected_in_metadata": unexpected_metadata,
    }
    if missing_metadata or unexpected_metadata:
        add_issue(
            issues,
            "error",
            "data_sample_id_mismatch",
            "%s manifest/metadata sample IDs differ" % key,
            key,
            "data",
        )

    context.update(
        {
            "sample_ids": manifest_ids,
            "manifest_by_id": manifest_by_id,
            "metadata_by_id": metadata_by_id,
        }
    )
    audio = result["audio"]
    for sample_id in manifest_ids:
        manifest = manifest_by_id[sample_id]
        metadata = metadata_by_id.get(sample_id)
        if metadata is None:
            continue
        try:
            manifest_raw_path = manifest["sample"]["audio"]["path"]
            manifest_audio_path = resolve_audio_path(root, manifest_raw_path)
            metadata_raw_path, expected_hash, expected_header = metadata_audio(metadata)
            metadata_audio_path = resolve_audio_path(root, metadata_raw_path)
        except (KeyError, TypeError, AuditError) as exc:
            add_issue(
                issues,
                "error",
                "invalid_audio_reference",
                "%s/%s: %s" % (key, sample_id, exc),
                key,
                "data",
            )
            continue
        audio["referenced_count"] += 1
        if manifest_audio_path == metadata_audio_path:
            audio["path_match_count"] += 1
        else:
            add_issue(
                issues,
                "error",
                "audio_path_mismatch",
                "%s/%s manifest and metadata point to different audio"
                % (key, sample_id),
                key,
                "data",
            )
            continue
        context["audio_paths"][sample_id] = manifest_audio_path
        if not manifest_audio_path.is_file():
            add_issue(
                issues,
                "error",
                "missing_audio",
                "%s/%s audio is missing: %s"
                % (key, sample_id, manifest_audio_path),
                key,
                "data",
            )
            continue
        audio["existing_count"] += 1
        actual_hash = sha256_file(manifest_audio_path)
        if actual_hash == expected_hash:
            audio["sha256_match_count"] += 1
        else:
            add_issue(
                issues,
                "error",
                "audio_sha256_mismatch",
                "%s/%s audio SHA256 does not match metadata" % (key, sample_id),
                key,
                "data",
            )
        try:
            actual_header = read_wav_header(manifest_audio_path)
        except AuditError as exc:
            add_issue(
                issues,
                "error",
                "invalid_wav",
                "%s/%s: %s" % (key, sample_id, exc),
                key,
                "data",
            )
            continue
        context["audio_headers"][sample_id] = actual_header
        audio["wav_header_valid_count"] += 1
        if (
            actual_header["compression"] == "NONE"
            and actual_header["sample_width_bytes"] == 2
            and actual_header["sample_rate_hz"] == 16000
            and actual_header["channels"] == 1
        ):
            audio["canonical_pcm16_16khz_mono_count"] += 1
        for field, metric in audio["metadata_header_fields"].items():
            expected = expected_header.get(field)
            if expected is None:
                continue
            metric["eligible_count"] += 1
            metric["evaluated_count"] += 1
            tolerance = (
                1.0 / float(actual_header["sample_rate_hz"])
                if field == "duration_sec"
                else None
            )
            if values_match(field, expected, actual_header[field], tolerance):
                metric["correct_count"] += 1

    audio["sha256_match_rate"] = safe_rate(
        audio["sha256_match_count"], audio["referenced_count"]
    )
    audio["existing_rate"] = safe_rate(
        audio["existing_count"], audio["referenced_count"]
    )
    audio["path_match_rate"] = safe_rate(
        audio["path_match_count"], audio["referenced_count"]
    )
    audio["wav_header_valid_rate"] = safe_rate(
        audio["wav_header_valid_count"], audio["referenced_count"]
    )
    audio["canonical_pcm16_16khz_mono_rate"] = safe_rate(
        audio["canonical_pcm16_16khz_mono_count"], audio["referenced_count"]
    )
    for metric in audio["metadata_header_fields"].values():
        finish_field_metric(metric)
    if audio["canonical_pcm16_16khz_mono_count"] != audio["referenced_count"]:
        add_issue(
            issues,
            "error",
            "noncanonical_wav_header",
            "%s has %d/%d PCM16 16 kHz mono WAV inputs"
            % (
                key,
                audio["canonical_pcm16_16khz_mono_count"],
                audio["referenced_count"],
            ),
            key,
            "data",
        )
    for field in ("channels", "sample_rate_hz", "frame_count", "duration_sec"):
        metric = audio["metadata_header_fields"][field]
        if (
            metric["eligible_count"] != audio["referenced_count"]
            or metric["correct_count"] != metric["eligible_count"]
        ):
            add_issue(
                issues,
                "error",
                "metadata_wav_header_mismatch",
                "%s metadata field %s matches %d/%d referenced WAV files"
                % (
                    key,
                    field,
                    metric["correct_count"],
                    audio["referenced_count"],
                ),
                key,
                "data",
            )

    if any(
        issue.get("dataset") == key
        and issue.get("component") == "data"
        and issue["severity"] == "error"
        for issue in issues
    ):
        result["status"] = "invalid"
    return result, context


def alignment_sources(rows, expected_ids, label):
    ids = [adapt_outputs.source_sample_id(row) for row in rows]
    if all(sample_id is None for sample_id in ids):
        if len(rows) != len(expected_ids):
            raise AuditError(
                "%s has no sample IDs and record count %d does not match manifest %d"
                % (label, len(rows), len(expected_ids))
            )
        return (
            dict(zip(expected_ids, rows)),
            "manifest_row_order_positional",
            [],
            [],
        )
    if any(sample_id is None for sample_id in ids):
        raise AuditError("%s mixes keyed and unkeyed records" % label)
    sources, _ = keyed_rows(rows, adapt_outputs.source_sample_id, label)
    expected = set(expected_ids)
    actual = set(sources)
    return (
        sources,
        "sample_id",
        sorted(expected - actual),
        sorted(actual - expected),
    )


def schema_metrics(system, expected_ids, sources, fields, universe, source_path):
    native_valid = 0
    adapter_valid = 0
    evaluated = 0
    status_counts = {}
    source_format_counts = {}
    native_error_counts = {}
    for sample_id in expected_ids:
        source = sources.get(sample_id)
        row = adapt_outputs.adapt_one(
            system, sample_id, source, source_path, fields, universe
        )
        evaluated += 1
        native_valid += row["native_current_schema_valid"] is True
        adapter_valid += row["adapter_schema_valid"] is True
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        source_format = row["source_format"]
        source_format_counts[source_format] = (
            source_format_counts.get(source_format, 0) + 1
        )
        for error in row.get("native_schema_errors", []):
            native_error_counts[error] = native_error_counts.get(error, 0) + 1
    return {
        "denominator": len(expected_ids),
        "evaluated_count": evaluated,
        "native_current_schema_valid_count": native_valid,
        "native_current_schema_valid_rate": safe_rate(native_valid, evaluated),
        "adapter_schema_valid_count": adapter_valid,
        "adapter_schema_valid_rate": safe_rate(adapter_valid, evaluated),
        "status_counts": status_counts,
        "source_format_counts": source_format_counts,
        "native_schema_error_counts": native_error_counts,
    }


def sure_payload(source):
    tags = source.get("tags") if isinstance(source, dict) else None
    return tags if isinstance(tags, dict) else source


def audit_sure(root, spec, context, fields, universe, issues):
    key = spec["key"]
    path = root / spec["sure"]
    result = {
        "status": "ready",
        "artifact": artifact_summary(root, spec["sure"], path),
        "alignment": None,
        "missing_sample_ids": [],
        "unexpected_sample_ids": [],
        "header_accuracy": {
            "duration_sec": empty_field_metric(
                fields["basic_acoustic.duration_sec"].get("tolerance", 0.001)
            ),
            "sample_rate_hz": empty_field_metric(),
            "channels": empty_field_metric(),
        },
        "schema": None,
    }
    if not path.is_file():
        result["status"] = "missing"
        add_issue(
            issues,
            "incomplete",
            "missing_sure_output",
            "%s is missing" % path,
            key,
            "sure-tagger",
        )
        return result
    try:
        rows, physical_lines = load_jsonl(path)
        result["artifact"] = artifact_summary(
            root, spec["sure"], path, rows, physical_lines
        )
        sources, alignment, missing, unexpected = alignment_sources(
            rows, context["sample_ids"], "%s sure-tagger" % key
        )
    except AuditError as exc:
        result["status"] = "invalid"
        add_issue(
            issues,
            "error",
            "invalid_sure_output",
            str(exc),
            key,
            "sure-tagger",
        )
        return result
    result["alignment"] = alignment
    result["missing_sample_ids"] = missing
    result["unexpected_sample_ids"] = unexpected
    if missing or unexpected:
        result["status"] = "invalid"
        add_issue(
            issues,
            "error",
            "sure_sample_id_mismatch",
            "%s sure-tagger sample IDs do not match the manifest" % key,
            key,
            "sure-tagger",
        )
    result["schema"] = schema_metrics(
        "sure-tagger", context["sample_ids"], sources, fields, universe, str(path)
    )
    for sample_id in context["sample_ids"]:
        source = sources.get(sample_id)
        expected_header = context["audio_headers"].get(sample_id)
        if source is None or expected_header is None:
            continue
        payload = sure_payload(source)
        basic = payload.get("basic_acoustic") if isinstance(payload, dict) else None
        if not isinstance(basic, dict):
            continue
        for field, metric in result["header_accuracy"].items():
            metric["eligible_count"] += 1
            if field not in basic:
                continue
            metric["evaluated_count"] += 1
            if values_match(
                field,
                expected_header[field],
                basic[field],
                metric.get("tolerance"),
            ):
                metric["correct_count"] += 1
    for metric in result["header_accuracy"].values():
        finish_field_metric(metric)
    return result


def caption_runtime_success(source):
    return isinstance(source, dict) and source.get("status") == "success"


def caption_audio_path_matches(root, source, expected_path):
    raw_path = source.get("audio_path") if isinstance(source, dict) else None
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        return resolve_audio_path(root, raw_path) == expected_path
    except AuditError:
        return False


def audit_captioner(root, spec, context, fields, universe, issues):
    key = spec["key"]
    path = root / spec["captioner"]
    result = {
        "status": "ready",
        "artifact": artifact_summary(root, spec["captioner"], path),
        "alignment": "sample_id",
        "missing_sample_ids": [],
        "unexpected_sample_ids": [],
        "runtime": {
            "denominator": len(context["sample_ids"]),
            "record_count": 0,
            "success_count": 0,
            "success_rate": None,
            "status_counts": {},
            "nonempty_caption_count": 0,
            "audio_path_match_count": 0,
        },
        "schema": None,
        "raw_caption_semantics": {
            "is_schema_prediction": False,
            "credited_as_schema_prediction": False,
            "note": "Raw caption text is a runtime artifact and is never parsed into schema fields.",
        },
        "runner_summary": {"path": spec.get("captioner_summary"), "present": False},
    }
    if not path.is_file():
        result["status"] = "missing"
        add_issue(
            issues,
            "incomplete",
            "missing_captioner_output",
            "%s is missing" % path,
            key,
            "captioner",
        )
        return result
    try:
        rows, physical_lines = load_jsonl(path)
        result["artifact"] = artifact_summary(
            root, spec["captioner"], path, rows, physical_lines
        )
        sources, _ = keyed_rows(
            rows, adapt_outputs.source_sample_id, "%s captioner" % key
        )
    except AuditError as exc:
        result["status"] = "invalid"
        add_issue(
            issues,
            "error",
            "invalid_captioner_output",
            str(exc),
            key,
            "captioner",
        )
        return result
    expected = set(context["sample_ids"])
    actual = set(sources)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    result["missing_sample_ids"] = missing
    result["unexpected_sample_ids"] = unexpected
    if missing:
        result["status"] = "incomplete"
        add_issue(
            issues,
            "incomplete",
            "captioner_missing_samples",
            "%s captioner is missing %d expected sample records" % (key, len(missing)),
            key,
            "captioner",
        )
    if unexpected:
        result["status"] = "invalid"
        add_issue(
            issues,
            "error",
            "captioner_unexpected_samples",
            "%s captioner has %d unexpected sample records"
            % (key, len(unexpected)),
            key,
            "captioner",
        )

    runtime = result["runtime"]
    runtime["record_count"] = len(rows)
    for sample_id in context["sample_ids"]:
        source = sources.get(sample_id)
        status = source.get("status") if isinstance(source, dict) else "missing"
        if not isinstance(status, str) or not status:
            status = "missing_status"
        runtime["status_counts"][status] = runtime["status_counts"].get(status, 0) + 1
        if caption_runtime_success(source):
            runtime["success_count"] += 1
        if isinstance(source, dict) and isinstance(source.get("caption"), str) and source[
            "caption"
        ].strip():
            runtime["nonempty_caption_count"] += 1
        expected_path = context["audio_paths"].get(sample_id)
        if expected_path is not None and caption_audio_path_matches(
            root, source or {}, expected_path
        ):
            runtime["audio_path_match_count"] += 1
    runtime["success_rate"] = safe_rate(
        runtime["success_count"], runtime["denominator"]
    )
    expected_present_count = len(context["sample_ids"]) - len(missing)
    if runtime["audio_path_match_count"] != expected_present_count:
        result["status"] = "invalid"
        add_issue(
            issues,
            "error",
            "captioner_audio_path_mismatch",
            "%s captioner audio paths match %d/%d present expected records"
            % (key, runtime["audio_path_match_count"], expected_present_count),
            key,
            "captioner",
        )
    result["schema"] = schema_metrics(
        "captioner", context["sample_ids"], sources, fields, universe, str(path)
    )

    runner_path_value = spec.get("captioner_summary")
    if runner_path_value:
        runner_path = root / runner_path_value
        result["runner_summary"] = {
            "path": runner_path_value,
            "present": runner_path.is_file(),
        }
        if runner_path.is_file():
            try:
                runner = load_json(runner_path)
                declared_dataset = runner.get("dataset")
                declared_count = runner.get("sample_count")
                declared_success = runner.get("success_count")
                result["runner_summary"].update(
                    {
                        "declared_dataset": declared_dataset,
                        "declared_sample_count": declared_count,
                        "declared_success_count": declared_success,
                        "count_matches": declared_count == len(rows),
                        "success_count_matches": declared_success
                        == runtime["success_count"],
                    }
                )
                accepted_names = set(
                    value.lower()
                    for value in spec.get(
                        "captioner_dataset_names", [spec["key"], spec["name"]]
                    )
                )
                dataset_matches = (
                    isinstance(declared_dataset, str)
                    and declared_dataset.lower() in accepted_names
                )
                result["runner_summary"]["dataset_matches"] = dataset_matches
                if not dataset_matches:
                    add_issue(
                        issues,
                        "warning",
                        "captioner_runner_dataset_mismatch",
                        "%s captioner runner summary declares dataset %r"
                        % (key, declared_dataset),
                        key,
                        "captioner",
                    )
                if declared_count != len(rows) or declared_success != runtime["success_count"]:
                    add_issue(
                        issues,
                        "warning",
                        "captioner_runner_count_mismatch",
                        "%s captioner runner summary counts disagree with results.jsonl"
                        % key,
                        key,
                        "captioner",
                    )
            except AuditError as exc:
                add_issue(
                    issues,
                    "warning",
                    "invalid_captioner_runner_summary",
                    str(exc),
                    key,
                    "captioner",
                )
    return result


def status_from_issues(issues):
    if any(issue["severity"] == "error" for issue in issues):
        return "invalid"
    if any(issue["severity"] == "incomplete" for issue in issues):
        return "incomplete"
    return "complete"


def dataset_status(data, sure, captioner):
    statuses = set([data["status"], sure["status"], captioner["status"]])
    if "invalid" in statuses:
        return "invalid"
    if statuses & set(["missing", "incomplete"]):
        return "incomplete"
    return "complete"


def aggregate_summary(datasets):
    totals = {
        "dataset_count": len(datasets),
        "complete_dataset_count": sum(
            dataset["status"] == "complete" for dataset in datasets
        ),
        "expected_sample_count": sum(
            dataset["data"]["expected_sample_count"] for dataset in datasets
        ),
        "manifest_record_count": 0,
        "metadata_record_count": 0,
        "audio_sha256_match_count": 0,
        "wav_header_valid_count": 0,
        "canonical_pcm16_16khz_mono_count": 0,
        "sure_record_count": 0,
        "captioner_record_count": 0,
        "captioner_runtime_success_count": 0,
        "sure_native_current_schema_valid_count": 0,
        "captioner_native_current_schema_valid_count": 0,
    }
    sure_native_evaluated = 0
    caption_native_evaluated = 0
    for dataset in datasets:
        data = dataset["data"]
        totals["manifest_record_count"] += data["manifest"].get("record_count", 0)
        totals["metadata_record_count"] += data["metadata"].get("record_count", 0)
        totals["audio_sha256_match_count"] += data["audio"]["sha256_match_count"]
        totals["wav_header_valid_count"] += data["audio"]["wav_header_valid_count"]
        totals["canonical_pcm16_16khz_mono_count"] += data["audio"][
            "canonical_pcm16_16khz_mono_count"
        ]
        sure = dataset["systems"]["sure-tagger"]
        captioner = dataset["systems"]["captioner"]
        totals["sure_record_count"] += sure["artifact"].get("record_count", 0)
        totals["captioner_record_count"] += captioner["runtime"]["record_count"]
        totals["captioner_runtime_success_count"] += captioner["runtime"][
            "success_count"
        ]
        if sure.get("schema"):
            totals["sure_native_current_schema_valid_count"] += sure["schema"][
                "native_current_schema_valid_count"
            ]
            sure_native_evaluated += sure["schema"]["evaluated_count"]
        if captioner.get("schema"):
            totals["captioner_native_current_schema_valid_count"] += captioner[
                "schema"
            ]["native_current_schema_valid_count"]
            caption_native_evaluated += captioner["schema"]["evaluated_count"]
    totals["audio_sha256_match_rate"] = safe_rate(
        totals["audio_sha256_match_count"], totals["expected_sample_count"]
    )
    totals["wav_header_valid_rate"] = safe_rate(
        totals["wav_header_valid_count"], totals["expected_sample_count"]
    )
    totals["canonical_pcm16_16khz_mono_rate"] = safe_rate(
        totals["canonical_pcm16_16khz_mono_count"], totals["expected_sample_count"]
    )
    totals["captioner_runtime_evaluated_count"] = totals["captioner_record_count"]
    totals["captioner_runtime_success_rate"] = safe_rate(
        totals["captioner_runtime_success_count"],
        totals["captioner_runtime_evaluated_count"],
    )
    totals["captioner_runtime_success_rate_e2e"] = safe_rate(
        totals["captioner_runtime_success_count"], totals["expected_sample_count"]
    )
    totals["sure_native_current_schema_evaluated_count"] = sure_native_evaluated
    totals["sure_native_current_schema_valid_rate"] = safe_rate(
        totals["sure_native_current_schema_valid_count"], sure_native_evaluated
    )
    totals[
        "captioner_native_current_schema_evaluated_count"
    ] = caption_native_evaluated
    totals["captioner_native_current_schema_valid_rate"] = safe_rate(
        totals["captioner_native_current_schema_valid_count"],
        caption_native_evaluated,
    )
    return totals


def summarize(root=DEFAULT_ROOT, dataset_specs=DATASET_SPECS, universe_path=DEFAULT_UNIVERSE):
    root = Path(root).resolve()
    issues = []
    try:
        universe = load_json(universe_path)
        fields = adapt_outputs.field_specs(universe)
    except (AuditError, ValueError) as exc:
        add_issue(
            issues,
            "error",
            "invalid_field_universe",
            str(exc),
            component="summary",
        )
        universe = {"universe_version": None, "fields": []}
        fields = {}
    datasets = []
    for spec in dataset_specs:
        data, context = audit_dataset_inputs(root, spec, issues)
        if fields:
            sure = audit_sure(root, spec, context, fields, universe, issues)
            captioner = audit_captioner(
                root, spec, context, fields, universe, issues
            )
        else:
            sure = {
                "status": "invalid",
                "artifact": artifact_summary(root, spec["sure"], root / spec["sure"]),
                "alignment": None,
                "header_accuracy": {},
                "schema": None,
            }
            captioner = {
                "status": "invalid",
                "artifact": artifact_summary(
                    root, spec["captioner"], root / spec["captioner"]
                ),
                "alignment": "sample_id",
                "runtime": {
                    "denominator": len(context["sample_ids"]),
                    "record_count": 0,
                    "success_count": 0,
                    "success_rate": None,
                },
                "schema": None,
                "raw_caption_semantics": {
                    "is_schema_prediction": False,
                    "credited_as_schema_prediction": False,
                },
            }
        row = {
            "key": spec["key"],
            "name": spec["name"],
            "status": dataset_status(data, sure, captioner),
            "data": data,
            "systems": {"sure-tagger": sure, "captioner": captioner},
        }
        datasets.append(row)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "root": str(root),
        "field_universe_version": universe.get("universe_version"),
        "status": status_from_issues(issues),
        "caption_policy": {
            "raw_caption_is_schema_prediction": False,
            "raw_caption_is_parsed_into_schema": False,
        },
        "datasets": datasets,
        "totals": aggregate_summary(datasets),
        "issues": issues,
    }
    return summary


def percent(value):
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def metric_cell(metric):
    if not metric:
        return "n/a"
    rate = metric.get("accuracy")
    if rate is None:
        rate = metric.get("success_rate")
    if rate is None:
        rate = metric.get("native_current_schema_valid_rate")
    if rate is None:
        return "n/a"
    count = metric.get("correct_count")
    denominator = metric.get("evaluated_count")
    if count is None:
        count = metric.get("success_count")
        denominator = metric.get("denominator")
    if count is None:
        count = metric.get("native_current_schema_valid_count")
    return "%d/%d (%s)" % (count, denominator, percent(rate))


def render_markdown(summary):
    lines = [
        "# Public Phase 0 Audit",
        "",
        "Status: **%s**" % summary["status"],
        "",
        "| Dataset | Data | Samples | Audio SHA | WAV | sure header (dur/rate/ch) | sure native schema | caption runtime | caption native schema |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for dataset in summary["datasets"]:
        data = dataset["data"]
        sure = dataset["systems"]["sure-tagger"]
        captioner = dataset["systems"]["captioner"]
        expected = data["expected_sample_count"]
        manifest_count = data["manifest"].get("record_count", 0)
        sha_cell = "%d/%d" % (data["audio"]["sha256_match_count"], expected)
        wav_cell = "%d/%d" % (data["audio"]["wav_header_valid_count"], expected)
        header = sure.get("header_accuracy", {})
        header_cell = "/".join(
            percent(header.get(field, {}).get("accuracy"))
            for field in ("duration_sec", "sample_rate_hz", "channels")
        )
        sure_schema = sure.get("schema")
        caption_schema = captioner.get("schema")
        lines.append(
            "| %s | %s | %d/%d | %s | %s | %s | %s | %s | %s |"
            % (
                dataset["name"],
                data["status"],
                manifest_count,
                expected,
                sha_cell,
                wav_cell,
                header_cell,
                metric_cell(sure_schema),
                metric_cell(captioner["runtime"]),
                metric_cell(caption_schema),
            )
        )
    lines.extend(
        [
            "",
            "Unkeyed sure-tagger tags-only rows are aligned positionally only when their count exactly matches the manifest; the alignment mode is retained in JSON.",
            "",
            "Native current-schema validity uses the frozen structural-v0.1 adapter contract; raw six-group tags are not silently upgraded to the derived schema.",
            "",
            "Raw caption text is a runtime artifact, not a schema prediction; it is not parsed or credited as one.",
            "",
        ]
    )
    if summary["issues"]:
        lines.append("## Issues")
        lines.append("")
        for issue in summary["issues"]:
            scope = "/".join(
                value
                for value in (issue.get("dataset"), issue.get("component"))
                if value
            )
            prefix = "[%s]" % issue["severity"]
            if scope:
                prefix += " %s" % scope
            lines.append("- %s: %s" % (prefix, issue["message"]))
        lines.append("")
    else:
        lines.extend(["No audit issues detected.", ""])
    return "\n".join(lines)


def write_outputs(summary, json_path, markdown_path):
    atomic_write_text(
        json_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(markdown_path, render_markdown(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--field-universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    json_path = args.json_output or root / "public_phase0.summary.json"
    markdown_path = args.markdown_output or root / "public_phase0.summary.md"
    summary = summarize(root, DATASET_SPECS, args.field_universe)
    write_outputs(summary, json_path, markdown_path)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(json_path).resolve()),
                "markdown": str(Path(markdown_path).resolve()),
                "issue_count": len(summary["issues"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary["status"] == "complete":
        return 0
    if summary["status"] == "incomplete":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
