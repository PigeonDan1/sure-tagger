"""Atomic artifact I/O for the speaker-v2 shadow pipeline."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def write_json_gz_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ".%s." % path.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary_name, "wt", encoding="utf-8") as sink:
            json.dump(
                value,
                sink,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        with open(temporary_name, "rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary_name, str(path))
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def write_sample_artifacts(
    output_dir,
    sample_id,
    evidence,
    fusion,
    artifact_root=None,
):
    root = (
        Path(output_dir) / "artifacts"
        if artifact_root is None
        else Path(artifact_root)
    )
    sample_dir = root / "speaker_v2" / safe_stem(sample_id)
    evidence_dir = sample_dir / "evidence"
    paths = []
    for item in evidence:
        paths.append(
            write_json_gz_atomic(
                evidence_dir / (safe_stem(item["evidence_id"]) + ".json.gz"),
                item,
            )
        )
    alignment_paths = []
    for item in fusion.get("timeline_comparisons", []):
        comparison_id = item.get("comparison_id") or _artifact_id(
            "alignment", item
        )
        alignment_paths.append(
            write_json_gz_atomic(
                sample_dir / "alignments" / (safe_stem(comparison_id) + ".json.gz"),
                item,
            )
        )
    speaker_text_paths = []
    for item in fusion.get("speaker_text_tracks", []):
        speaker_text_paths.append(
            write_json_gz_atomic(
                sample_dir / "speaker_text" / (safe_stem(item["track_id"]) + ".json.gz"),
                item,
            )
        )
    speaker_text_comparison_paths = []
    for item in fusion.get("speaker_text_comparisons", []):
        speaker_text_comparison_paths.append(
            write_json_gz_atomic(
                sample_dir
                / "speaker_text_comparisons"
                / (safe_stem(item["comparison_id"]) + ".json.gz"),
                item,
            )
        )
    hypothesis_paths = []
    for item in fusion.get("hypotheses", []):
        hypothesis_paths.append(
            write_json_gz_atomic(
                sample_dir / "hypotheses" / (safe_stem(item["case_id"]) + ".json.gz"),
                item,
            )
        )
    certification = {
        "schema_version": fusion["schema_version"],
        "sample_id": fusion["sample_id"],
        "fusion_id": fusion["fusion_id"],
        "profile": fusion["profile"],
        "claims": fusion["claims"],
        "public_adapter": fusion["public_adapter"],
    }
    certification_path = write_json_gz_atomic(
        sample_dir / "certifications" / (safe_stem(sample_id) + ".json.gz"),
        certification,
    )
    compat_metadata = {
        "schema_version": "speaker_v2.compat_metadata.shadow.1",
        "sample_id": str(sample_id),
        "profile": fusion["profile"],
        "speaker": dict(fusion["public_adapter"]["speaker"]),
        "speaker_count": fusion["claims"]["speaker_count"].get("exact"),
        "fusion_id": fusion["fusion_id"],
        "published": False,
    }
    compat_path = write_json_gz_atomic(
        sample_dir / "compat_metadata.json.gz",
        compat_metadata,
    )
    fusion_path = write_json_gz_atomic(
        sample_dir / "fusion_artifact_v2.json.gz",
        fusion,
    )
    return {
        "sample_dir": str(sample_dir),
        "fusion_artifact": str(fusion_path),
        "evidence_artifacts": [str(item) for item in paths],
        "alignment_artifacts": [str(item) for item in alignment_paths],
        "speaker_text_artifacts": [str(item) for item in speaker_text_paths],
        "speaker_text_comparison_artifacts": [
            str(item) for item in speaker_text_comparison_paths
        ],
        "hypothesis_artifacts": [str(item) for item in hypothesis_paths],
        "certification_artifact": str(certification_path),
        "compat_metadata": str(compat_path),
    }


def write_run_manifest(
    output_dir,
    input_manifest,
    input_manifest_sha256,
    config,
    summary,
    sample_ids=None,
    max_samples=None,
    fail_fast=False,
    resume=False,
):
    model_configs = {}
    for name in (
        "moss",
        "vad",
        "campplus",
        "whisper",
        "sortformer",
        "pyannote",
    ):
        model_config = getattr(config, "%s_config" % name, None)
        enabled = bool(getattr(config, "enable_%s" % name, False))
        model_configs[name] = {
            "enabled": enabled,
            "config": _config_record(model_config),
        }
    value = {
        "schema_version": "speaker_v2.run_manifest.shadow.1",
        "profile": "v2-shadow",
        "input": {
            "manifest": str(Path(input_manifest).resolve()),
            "manifest_sha256": str(input_manifest_sha256),
            "selected_sample_ids": sorted(str(item) for item in (sample_ids or [])),
            "max_samples": max_samples,
            "native_metadata_entered_inference": False,
            "input_transcript_entered_resolver": False,
        },
        "execution": {
            "fail_fast": bool(fail_fast),
            "resume": bool(resume),
            "workers": summary["workers"],
            "model_workers": summary["model_workers"],
            "model_worker_overrides": summary["model_worker_overrides"],
        },
        "models": model_configs,
        "result": {
            "result_path": summary["result_path"],
            "processed_sample_count": summary["processed_sample_count"],
            "success_count": summary["success_count"],
            "failure_count": summary["failure_count"],
        },
        "public_adapter_enabled": False,
    }
    path = Path(output_dir) / "run_manifest.json"
    _write_json_atomic(path, value)
    return path


def safe_stem(value):
    raw = str(value)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    stem = stem or "sample"
    if stem == raw and len(stem) <= 160:
        return stem
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return "%s__%s" % (stem[:146], suffix)


def _artifact_id(prefix, value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "%s_%s" % (prefix, hashlib.sha256(encoded).hexdigest()[:24])


def _config_record(config):
    if config is None:
        return None
    if hasattr(config, "to_record"):
        value = config.to_record()
    else:
        value = dict(vars(config))
    value.pop("token", None)
    value.pop("access_token", None)
    return value


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(
                value,
                sink,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary_name, str(path))
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _fsync_directory(directory):
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
