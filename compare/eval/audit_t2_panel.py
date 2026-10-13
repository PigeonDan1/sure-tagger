#!/usr/bin/env python3
"""Audit the frozen T2 panel without optional Python dependencies."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import wave


EXPECTED = {
    "fleurs": {
        "manifest": "fleurs_t2_panel.manifest.jsonl",
        "metadata": "fleurs_t2_panel.metadata.jsonl",
        "expected": 400,
        "per_language": 40,
    },
    "librispeech": {
        "manifest": "librispeech_t2_panel.manifest.jsonl",
        "metadata": "librispeech_t2_panel.metadata.jsonl",
        "expected": 100,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError("%s:%d is blank" % (path, line_number))
            rows.append(json.loads(line))
    return rows


def wav_header(path):
    with wave.open(str(path), "rb") as source:
        return {
            "channels": source.getnchannels(),
            "sample_rate_hz": source.getframerate(),
            "sample_width_bytes": source.getsampwidth(),
            "frame_count": source.getnframes(),
            "duration_sec": round(source.getnframes() / float(source.getframerate()), 6),
            "compression": source.getcomptype(),
        }


def audit_one(root, name, spec):
    issues = []
    result = {"expected_count": spec["expected"], "issues": issues}
    manifest_path = root / spec["manifest"]
    metadata_path = root / spec["metadata"]
    result["manifest"] = {"path": spec["manifest"], "present": manifest_path.is_file()}
    result["metadata"] = {"path": spec["metadata"], "present": metadata_path.is_file()}
    if not manifest_path.is_file() or not metadata_path.is_file():
        issues.append("missing manifest or metadata")
        result["status"] = "fail"
        return result
    try:
        manifests = load_jsonl(manifest_path)
        metadata = load_jsonl(metadata_path)
    except Exception as error:
        issues.append("invalid JSONL: %s" % error)
        result["status"] = "fail"
        return result
    result["manifest"].update({"records": len(manifests), "sha256": sha256_file(manifest_path)})
    result["metadata"].update({"records": len(metadata), "sha256": sha256_file(metadata_path)})
    if len(manifests) != spec["expected"] or len(metadata) != spec["expected"]:
        issues.append("record count mismatch")
    mids = [row.get("sample", {}).get("sample_id") for row in manifests]
    dids = [row.get("sample_id") for row in metadata]
    if any(not isinstance(value, str) or not value for value in mids + dids):
        issues.append("missing sample ID")
    if len(set(mids)) != len(mids) or len(set(dids)) != len(dids):
        issues.append("duplicate sample ID")
    if mids != dids:
        issues.append("manifest/metadata ID order mismatch")
    manifest_by_id = dict(zip(mids, manifests))
    metadata_by_id = dict(zip(dids, metadata))
    raw_violations = []
    audio = {"referenced": 0, "existing": 0, "hash_match": 0, "wav_valid": 0, "canonical": 0}
    frame_mismatch = []
    for sample_id in dids:
        manifest = manifest_by_id.get(sample_id)
        record = metadata_by_id.get(sample_id)
        if not isinstance(manifest, dict) or not isinstance(record, dict):
            continue
        corpus = manifest.get("corpus", {})
        sample = manifest.get("sample", {})
        if set(manifest) != {"corpus", "sample"}:
            issues.append("manifest top-level keys are not closed: %s" % sample_id)
        if set(corpus) != {"dataset_name", "source_urls", "native_metadata"}:
            issues.append("manifest corpus keys are not closed: %s" % sample_id)
        if set(sample) != {"sample_id", "audio", "text", "native_metadata"}:
            issues.append("manifest sample keys are not closed: %s" % sample_id)
        if corpus.get("native_metadata") != {} or sample.get("native_metadata") != {}:
            raw_violations.append((sample_id, "native_metadata"))
        if sample.get("text", {}).get("transcript") != "":
            raw_violations.append((sample_id, "transcript"))
        manifest_audio = sample.get("audio", {}).get("path")
        metadata_audio = record.get("system_input_audio", {})
        metadata_path_value = metadata_audio.get("path")
        for path_value in (manifest_audio, metadata_path_value):
            if not isinstance(path_value, str) or Path(path_value).is_absolute() or ".." in Path(path_value).parts:
                issues.append("unsafe audio path: %s" % sample_id)
        if manifest_audio != metadata_path_value:
            issues.append("audio path mismatch: %s" % sample_id)
            continue
        audio["referenced"] += 1
        path = root / metadata_path_value
        if not path.is_file():
            issues.append("missing audio: %s" % sample_id)
            continue
        audio["existing"] += 1
        digest = metadata_audio.get("sha256")
        if isinstance(digest, str) and sha256_file(path) == digest.lower():
            audio["hash_match"] += 1
        else:
            issues.append("audio hash mismatch: %s" % sample_id)
        try:
            header = wav_header(path)
            audio["wav_valid"] += 1
            canonical = (header["channels"], header["sample_rate_hz"], header["sample_width_bytes"], header["compression"]) == (1, 16000, 2, "NONE")
            if canonical:
                audio["canonical"] += 1
            else:
                issues.append("noncanonical WAV: %s" % sample_id)
            declared = metadata_audio.get("header", {})
            for key in ("channels", "sample_rate_hz", "sample_width_bytes", "frame_count"):
                if declared.get(key) != header.get(key):
                    issues.append("metadata header mismatch %s/%s" % (sample_id, key))
            source_header = record.get("source", {}).get("header", {})
            if source_header and source_header.get("frame_count") != header.get("frame_count"):
                frame_mismatch.append({"sample_id": sample_id, "source": source_header.get("frame_count"), "output": header.get("frame_count")})
        except Exception as error:
            issues.append("invalid WAV %s: %s" % (sample_id, error))
    result["audio"] = audio
    result["raw_only_violation_count"] = len(raw_violations)
    result["raw_only_violations"] = raw_violations[:20]
    result["source_output_frame_mismatch"] = frame_mismatch
    if name == "fleurs":
        languages = collections.Counter(row.get("fleurs_config") for row in metadata)
        result["per_language"] = dict(sorted(languages.items()))
        if set(languages.values()) != {spec["per_language"]} or len(languages) != 10:
            issues.append("FLEURS per-language quota mismatch")
        rejection_path = root / "fleurs_t2_panel.rejections.jsonl"
        if rejection_path.is_file():
            result["rejections"] = {"path": rejection_path.name, "records": len(load_jsonl(rejection_path)), "sha256": sha256_file(rejection_path)}
    else:
        revisions = sorted({row.get("dataset", {}).get("revision") for row in metadata})
        result["source_revisions"] = revisions
        if revisions != ["71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"]:
            issues.append("LibriSpeech revision mismatch")
    result["status"] = "pass" if not issues and not raw_violations and not frame_mismatch else "fail"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "evaluation" / "t2-panel")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = {"schema_version": "t2-panel-audit-v0.1", "root": str(root), "datasets": {name: audit_one(root, name, spec) for name, spec in EXPECTED.items()}}
    report["status"] = "pass" if all(item["status"] == "pass" for item in report["datasets"].values()) else "fail"
    output = args.output.resolve() if args.output else root / "t2_panel.audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
