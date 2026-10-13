#!/usr/bin/env python3
"""Merge validated DeepSeek descriptive tags onto Sure-Tagger results."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


STRUCTURAL_GROUPS = (
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
)
DESCRIPTIVE_KEYS = (
    "caption",
    "topic",
    "background_condition",
    "dominant_content_impression",
    "clarity_intelligibility_impression",
    "background_sound_summary",
    "descriptive_phrases",
    "facets",
    "speaker_descriptions",
    "scene_hypotheses",
    "usability_notes",
    "uncertainties",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("blank line at %s:%d" % (path, line_number))
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("row at %s:%d is not an object" % (path, line_number))
            rows.append(value)
    if not rows:
        raise ValueError("empty JSONL: %s" % path)
    return rows


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


def _by_id(rows, source_name):
    result = {}
    for index, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("%s row %d lacks sample_id" % (source_name, index))
        if sample_id in result:
            raise ValueError("duplicate sample_id in %s: %s" % (source_name, sample_id))
        result[sample_id] = row
    return result


def _canonical_structural(tags, group):
    """Match the batch runner's null-object normalization for comparison."""
    value = dict(tags)
    if group == "speaker":
        value.setdefault("speaker_transcripts", None)
    if group == "language_content" and value.get("punctuation") is None:
        value["punctuation"] = {
            "punctuation_count": None,
            "has_terminal_punctuation": None,
        }
    if group == "language_content" and value.get("repetition") is None:
        value["repetition"] = {
            "has_repetition": None,
            "repetition_count": None,
        }
    profiles = value.get("profiles") if group == "speaker" else None
    if isinstance(profiles, list):
        # The DeepSeek wire schema intentionally excludes profile gender, while
        # finalized source rows may retain it for structural compatibility.
        value["profiles"] = []
        for profile in profiles:
            if isinstance(profile, dict):
                normalized = dict(profile)
                normalized.pop("gender", None)
                value["profiles"].append(normalized)
            else:
                value["profiles"].append(profile)
    return value


def _diff_keys(left, right):
    """Field names whose values differ between two structural groups."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))

def _sanctioned_language(sample_id, verification_by_id):
    """Return the audited language override for a sample, or None.

    The DeepSeek call is the only place allowed to change a structural value,
    and only for ``language_content.language``.  This function is the gate that
    decides whether a change seen at merge time was actually authorized, so a
    tampered or stale results file cannot quietly rewrite the published label.
    """
    row = verification_by_id.get(sample_id)
    if not isinstance(row, dict):
        return None
    if not row.get("changed"):
        return None
    published = row.get("published_language")
    if not isinstance(published, str) or not published:
        return None
    return row


def _resolve_verification_path(deepseek_path, verification_path):
    """Locate the language verdict sidecar written next to the batch results.

    Returns None when there is none, which keeps the historical strict merge
    (no structural change tolerated at all) as the default for older runs.
    """
    if verification_path is not None:
        path = Path(verification_path)
        if not path.is_file():
            raise ValueError("language verification file not found: %s" % path)
        return path
    candidate = Path(deepseek_path).parent / "language_verification.jsonl"
    return candidate if candidate.is_file() else None


def _restore_profile_genders(tags, descriptive):
    """Copy source profile genders onto matching generated descriptions."""
    profiles = tags.get("speaker", {}).get("profiles")
    descriptions = descriptive.get("speaker_descriptions")
    if not isinstance(profiles, list) or not isinstance(descriptions, list):
        return descriptive
    genders = {
        profile.get("speaker_id"): profile.get("gender")
        for profile in profiles
        if isinstance(profile, dict) and "gender" in profile
    }
    if not genders:
        return descriptive
    result = dict(descriptive)
    result["speaker_descriptions"] = []
    for description in descriptions:
        value = dict(description)
        if value.get("speaker_id") in genders:
            value["gender"] = genders[value["speaker_id"]]
        result["speaker_descriptions"].append(value)
    return result


def merge(structural_path, deepseek_path, output_dir, verification_path=None):
    structural_rows = read_jsonl(structural_path)
    deepseek_rows = read_jsonl(deepseek_path)
    structural_by_id = _by_id(structural_rows, "structural input")
    deepseek_by_id = _by_id(deepseek_rows, "DeepSeek results")
    if set(structural_by_id) != set(deepseek_by_id):
        missing = sorted(set(structural_by_id) - set(deepseek_by_id))
        extra = sorted(set(deepseek_by_id) - set(structural_by_id))
        raise ValueError("sample coverage mismatch (missing=%d, extra=%d)" % (len(missing), len(extra)))

    verification_path = _resolve_verification_path(deepseek_path, verification_path)
    verification_by_id = {}
    if verification_path is not None:
        verification_by_id = _by_id(read_jsonl(verification_path), "language verification")
    merged = []
    override_counts = {}
    for source in structural_rows:
        sample_id = source["sample_id"]
        tags = source.get("tags")
        if not isinstance(tags, dict) or set(tags) != set(STRUCTURAL_GROUPS):
            raise ValueError("structural tags schema mismatch for %s" % sample_id)
        candidate = deepseek_by_id[sample_id]
        if set(STRUCTURAL_GROUPS) - set(candidate):
            raise ValueError("DeepSeek result lacks structural groups for %s" % sample_id)
        override = None
        for group in STRUCTURAL_GROUPS:
            source_group = _canonical_structural(tags[group], group)
            if candidate[group] == source_group:
                continue
            differing = _diff_keys(source_group, candidate[group])
            if group == "language_content" and differing == ["language"]:
                row = _sanctioned_language(sample_id, verification_by_id)
                if row is None:
                    raise ValueError(
                        "language changed without an authorizing verification record for %s"
                        % sample_id
                    )
                if source_group.get("language") != row.get("lid_language"):
                    raise ValueError(
                        "language verification record disagrees with the source LID value for %s"
                        % sample_id
                    )
                if candidate[group].get("language") != row.get("published_language"):
                    raise ValueError(
                        "language verification record disagrees with the DeepSeek result for %s"
                        % sample_id
                    )
                override = row["published_language"]
                key = "%s -> %s" % (source_group.get("language"), override)
                override_counts[key] = override_counts.get(key, 0) + 1
                continue
            raise ValueError(
                "structural field changed during DeepSeek merge for %s (%s)" % (sample_id, group)
            )
        descriptive = candidate.get("descriptive")
        if not isinstance(descriptive, dict) or set(descriptive) != set(DESCRIPTIVE_KEYS):
            raise ValueError("descriptive schema mismatch for %s" % sample_id)
        merged_row = dict(candidate)
        merged_row["descriptive"] = _restore_profile_genders(tags, descriptive)
        merged_row.update(tags)
        if override is not None:
            language_group = dict(merged_row["language_content"])
            language_group["language"] = override
            merged_row["language_content"] = language_group
        merged.append(merged_row)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged)
    tags_with_descriptive = []
    for row in merged:
        value = dict((group, row[group]) for group in STRUCTURAL_GROUPS)
        value["descriptive"] = row["descriptive"]
        tags_with_descriptive.append(value)
    tags_text = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        for value in tags_with_descriptive
    )
    atomic_write(output_dir / "result_with_id_descriptive.jsonl", merged_text)
    atomic_write(output_dir / "tags_with_descriptive.jsonl", tags_text)
    summary = {
        "sample_count": len(merged),
        "structural_input": str(Path(structural_path).resolve()),
        "structural_input_sha256": sha256_file(structural_path),
        "deepseek_results": str(Path(deepseek_path).resolve()),
        "deepseek_results_sha256": sha256_file(deepseek_path),
        "result_with_id_descriptive": str(output_dir / "result_with_id_descriptive.jsonl"),
        "tags_with_descriptive": str(output_dir / "tags_with_descriptive.jsonl"),
        # Every structural group is preserved except the one leaf that the
        # language verdict is authorized to move; ``structural_overrides``
        # records how many samples actually moved.
        "structural_groups_preserved": True,
        "structural_overrides": {"language_content.language": sum(override_counts.values())},
        "language_override_counts": dict(
            sorted(override_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "language_verification": None if verification_path is None else str(verification_path.resolve()),
        "language_verification_sha256": (
            None if verification_path is None else sha256_file(verification_path)
        ),
        "descriptive_key_count": len(DESCRIPTIVE_KEYS),
    }
    atomic_write(output_dir / "merge.summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-input", required=True, type=Path)
    parser.add_argument("--deepseek-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--language-verification",
        type=Path,
        default=None,
        help=(
            "language verdict sidecar authorizing language_content.language "
            "changes (default: language_verification.jsonl next to the results)"
        ),
    )
    args = parser.parse_args(argv)
    summary = merge(
        args.structural_input,
        args.deepseek_results,
        args.output_dir,
        verification_path=args.language_verification,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
