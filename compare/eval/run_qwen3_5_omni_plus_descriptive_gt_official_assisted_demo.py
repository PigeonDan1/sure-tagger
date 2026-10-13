#!/usr/bin/env python3
"""Run one official-GT-assisted descriptive demo for each T1-T3 track."""

from __future__ import print_function

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from run_qwen3_5_omni_plus_descriptive_gt_batch import (
    atomic_write_json,
    load_samples,
)
from run_qwen3_5_omni_plus_descriptive_gt_demo import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DESCRIPTIVE_KEYS,
    parse_json_object,
    read_api_key,
    redact,
    sha256_file,
    sha256_text,
    stream_with_urllib,
)
from score_benchmark_v1 import build_ground_truth


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_MANIFEST = DEFAULT_BENCHMARK_ROOT / "benchmark.manifest.jsonl"
DEFAULT_PROMPT = (
    SCRIPT_DIR / "qwen3_5_omni_plus_descriptive_gt_official_assisted_prompt.txt"
)
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"
DEFAULT_OUTPUT = (
    COMPARE_ROOT
    / "evaluation"
    / "qwen3_5_omni_plus_descriptive_gt_official_assisted_demo"
)
REFERENCE_CONTRACT_VERSION = "official-reference-facts-v1"
SPEAKER_GENDERS = frozenset(("male", "female"))
SPEAKER_ID_RE = re.compile(r"^speaker_[1-9][0-9]*$")
SPEAKER_DEMOGRAPHIC_RE = re.compile(
    r"\b(?:male|female)\s+(?:voice|speaker|narrator|talker|speech)\b"
    r"|\b(?:voice|speaker|narrator|talker)\s+(?:is|sounds?)\s+(?:male|female)\b"
    r"|\b(?:man|woman|boy|girl)\s+(?:speaks?|says?|talks?)\b",
    re.IGNORECASE,
)
SPEAKER_DEMOGRAPHIC_SUBSTITUTIONS = (
    (
        re.compile(
            r"\b(?:male|female)\s+(voice|speaker|narrator|talker|speech)\b",
            re.IGNORECASE,
        ),
        r"\1",
    ),
    (
        re.compile(
            r"\b(voice|speaker|narrator|talker)\s+(?:is|sounds?)\s+"
            r"(?:male|female)\b",
            re.IGNORECASE,
        ),
        r"\1",
    ),
    (
        re.compile(
            r"\b(?:man|woman|boy|girl)\s+(speaks?|says?|talks?)\b",
            re.IGNORECASE,
        ),
        r"speaker \1",
    ),
)


def track_group(track):
    if isinstance(track, str):
        for group in ("T1", "T2", "T3"):
            if track.startswith(group + "_"):
                return group
    raise ValueError("unsupported benchmark track: %r" % track)


def select_demo_samples(samples):
    selected = {}
    for sample in samples:
        group = track_group(sample["track"])
        if group not in selected:
            selected[group] = sample
    missing = [group for group in ("T1", "T2", "T3") if group not in selected]
    if missing:
        raise ValueError("manifest lacks demo tracks: %s" % ", ".join(missing))
    return [selected[group] for group in ("T1", "T2", "T3")]


def reference_payload(gt_record):
    facts = {}
    for path in sorted(gt_record["gt_mask"]):
        # Gender is a structural sequence score, not part of the descriptive
        # semantic reference prompt. Keep the semantic input contract stable.
        if path == "speaker.gender":
            continue
        if path in gt_record["values"] and gt_record["values"][path] is not None:
            facts[path] = gt_record["values"][path]
    semantics = {
        "listed_fields": "authoritative official ground truth",
        "unlisted_fields": "unknown",
    }
    if track_group(gt_record["track"]) == "T3":
        semantics["event_noise_negatives"] = (
            "native-label closed-world negatives, not human-confirmed audible absences"
        )
    return {
        "contract_version": REFERENCE_CONTRACT_VERSION,
        "facts": facts,
        "annotation_semantics": semantics,
    }


def user_text(payload):
    return "REFERENCE_FACTS_JSON\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def confidence(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % path)
    if value < 0 or value > 1:
        raise ValueError("%s must be in [0, 1]" % path)


def text_or_null(value, path):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("%s must be a non-empty string or null" % path)


def string_array_or_null(value, path):
    if value is None:
        return
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("%s must be an array of non-empty strings or null" % path)


def evidence_tags(value, path, allowed_paths):
    if not isinstance(value, list):
        raise ValueError("%s must be an array" % path)
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("%s contains an invalid path" % path)
    if len(value) != len(set(value)):
        raise ValueError("%s contains duplicate paths" % path)
    unknown = sorted(set(value) - allowed_paths)
    if unknown:
        raise ValueError("%s contains unavailable paths: %s" % (path, unknown))


def sanitize_assisted_descriptive(value, reference_facts):
    """Remove only locally prohibited wording and unavailable evidence paths."""
    sanitized = copy.deepcopy(value)
    changes = []
    descriptive = sanitized.get("descriptive") if isinstance(sanitized, dict) else None
    if not isinstance(descriptive, dict):
        return sanitized, changes

    def sanitize_strings(item, path):
        if isinstance(item, str):
            updated = item
            for pattern, replacement in SPEAKER_DEMOGRAPHIC_SUBSTITUTIONS:
                updated = pattern.sub(replacement, updated)
            updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
            updated = re.sub(r"\s{2,}", " ", updated).strip()
            if updated != item:
                changes.append({
                    "type": "speaker_demographic_wording_removed",
                    "path": path,
                })
            return updated
        if isinstance(item, list):
            return [
                sanitize_strings(child, "%s[%d]" % (path, index))
                for index, child in enumerate(item)
            ]
        if isinstance(item, dict):
            return {
                key: sanitize_strings(child, path + "." + key)
                for key, child in item.items()
            }
        return item

    sanitized["descriptive"] = sanitize_strings(descriptive, "descriptive")
    descriptive = sanitized["descriptive"]
    allowed_paths = set(
        path for path, item in reference_facts.items() if item is not None
    )
    for field in ("facets", "speaker_descriptions", "scene_hypotheses"):
        items = descriptive.get(field)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("evidence_tags"), list):
                continue
            original = item["evidence_tags"]
            filtered = []
            for tag in original:
                if tag in allowed_paths and tag not in filtered:
                    filtered.append(tag)
            removed_count = len(original) - len(filtered)
            if removed_count:
                item["evidence_tags"] = filtered
                changes.append({
                    "type": "unavailable_evidence_tags_removed",
                    "path": "descriptive.%s[%d].evidence_tags" % (field, index),
                    "removed_count": removed_count,
                })
    return sanitized, changes


def validate_assisted_descriptive(value, reference_facts):
    if set(value) != {"descriptive"}:
        raise ValueError("root keys must be exactly ['descriptive']")
    descriptive = value["descriptive"]
    if not isinstance(descriptive, dict) or set(descriptive) != set(DESCRIPTIVE_KEYS):
        raise ValueError("descriptive keys do not match the 12-key contract")
    for field in (
        "caption",
        "topic",
        "background_condition",
        "dominant_content_impression",
        "clarity_intelligibility_impression",
        "background_sound_summary",
    ):
        text_or_null(descriptive[field], "descriptive.%s" % field)
    for field in ("descriptive_phrases", "usability_notes", "uncertainties"):
        string_array_or_null(descriptive[field], "descriptive.%s" % field)

    serialized_descriptive = json.dumps(descriptive, ensure_ascii=False, sort_keys=True)
    match = SPEAKER_DEMOGRAPHIC_RE.search(serialized_descriptive)
    if match:
        raise ValueError(
            "descriptive output contains prohibited speaker demographic wording: %s"
            % match.group(0)
        )

    allowed_paths = set(
        path for path, item in reference_facts.items() if item is not None
    )
    facets = descriptive["facets"]
    if facets is not None:
        if not isinstance(facets, list) or len(facets) > 6:
            raise ValueError("descriptive.facets must contain at most 6 items or be null")
        names = []
        for index, item in enumerate(facets):
            path = "descriptive.facets[%d]" % index
            if not isinstance(item, dict) or set(item) != {
                "name", "description", "evidence_tags", "confidence"
            }:
                raise ValueError("%s has invalid keys" % path)
            text_or_null(item["name"], path + ".name")
            text_or_null(item["description"], path + ".description")
            if item["name"] is None or item["description"] is None:
                raise ValueError("%s name/description cannot be null" % path)
            names.append(item["name"])
            evidence_tags(item["evidence_tags"], path + ".evidence_tags", allowed_paths)
            confidence(item["confidence"], path + ".confidence")
        if len(names) != len(set(names)):
            raise ValueError("descriptive.facets names must be unique")

    profiles = descriptive["speaker_descriptions"]
    if profiles is not None:
        if not isinstance(profiles, list):
            raise ValueError("descriptive.speaker_descriptions must be an array or null")
        ids = []
        for index, item in enumerate(profiles):
            path = "descriptive.speaker_descriptions[%d]" % index
            if not isinstance(item, dict) or set(item) != {
                "speaker_id", "gender", "description", "descriptive_phrases",
                "evidence_tags", "confidence"
            }:
                raise ValueError("%s has invalid keys" % path)
            if not isinstance(item["speaker_id"], str) or not SPEAKER_ID_RE.match(
                item["speaker_id"]
            ):
                raise ValueError("%s.speaker_id is invalid" % path)
            ids.append(item["speaker_id"])
            if item["gender"] is not None and item["gender"] not in SPEAKER_GENDERS:
                raise ValueError("%s.gender must be male, female, or null" % path)
            text_or_null(item["description"], path + ".description")
            if item["description"] is None:
                raise ValueError("%s.description cannot be null" % path)
            string_array_or_null(
                item["descriptive_phrases"], path + ".descriptive_phrases"
            )
            evidence_tags(item["evidence_tags"], path + ".evidence_tags", allowed_paths)
            confidence(item["confidence"], path + ".confidence")
        if len(ids) != len(set(ids)):
            raise ValueError("speaker_descriptions speaker_id values must be unique")

    hypotheses = descriptive["scene_hypotheses"]
    if hypotheses is not None:
        if not isinstance(hypotheses, list) or len(hypotheses) > 3:
            raise ValueError(
                "descriptive.scene_hypotheses must contain at most 3 items or be null"
            )
        for index, item in enumerate(hypotheses):
            path = "descriptive.scene_hypotheses[%d]" % index
            if not isinstance(item, dict) or set(item) != {
                "description", "evidence_tags", "confidence"
            }:
                raise ValueError("%s has invalid keys" % path)
            text_or_null(item["description"], path + ".description")
            if item["description"] is None:
                raise ValueError("%s.description cannot be null" % path)
            evidence_tags(item["evidence_tags"], path + ".evidence_tags", allowed_paths)
            confidence(item["confidence"], path + ".confidence")


def process_sample(sample, gt_record, api_key, prompt_text, prompt_hash, args):
    payload = reference_payload(gt_record)
    request_text = user_text(payload)
    reference_hash = sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    result = {
        "sample_id": sample["sample_id"],
        "track": sample["track"],
        "manifest_line": sample["manifest_line"],
        "audio_path": str(sample["audio_path"]),
        "audio_sha256": sha256_file(sample["audio_path"]),
        "model": args.model,
        "base_url": args.base_url,
        "transport": "urllib-fallback",
        "input_contract": "system_prompt_plus_official_reference_facts_plus_one_audio",
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": prompt_hash,
        "reference_contract_version": REFERENCE_CONTRACT_VERSION,
        "reference_facts_sha256": reference_hash,
        "reference_facts": payload,
    }
    last_error = "unknown error"
    for attempt in range(1, args.retries + 2):
        try:
            raw_text, usage = stream_with_urllib(
                args.base_url,
                api_key,
                prompt_text,
                sample["audio_path"].read_bytes(),
                args.model,
                args.max_tokens,
                user_text=request_text,
            )
            raw_text = raw_text.strip()
            if not raw_text:
                raise ValueError("empty model response")
            parsed = parse_json_object(raw_text)
            sanitizations = []
            if getattr(args, "sanitize_invalid_output", False):
                parsed, sanitizations = sanitize_assisted_descriptive(
                    parsed, payload["facts"]
                )
            validate_assisted_descriptive(parsed, payload["facts"])
            stored_raw_text = raw_text
            if sanitizations:
                stored_raw_text = json.dumps(
                    parsed, ensure_ascii=False, indent=2, sort_keys=True
                )
            result.update({
                "status": "success",
                "attempts": attempt,
                "usage": usage,
                "raw_text": stored_raw_text,
                "descriptive": parsed["descriptive"],
            })
            if sanitizations:
                result.update({
                    "raw_text_sanitized": True,
                    "original_raw_text_sha256": sha256_text(raw_text),
                    "sanitizations": sanitizations,
                })
            return result
        except Exception as error:
            last_error = redact(error, api_key)
            if attempt <= args.retries:
                time.sleep(min(30, 2 ** (attempt - 1)) + random.random())
    result.update({
        "status": "failed",
        "attempts": args.retries + 1,
        "error": last_error,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sanitize-invalid-output", action="store_true")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.retries < 0:
        parser.error("max-tokens/retries values are invalid")

    benchmark_root = args.benchmark_root.resolve()
    manifest_path = args.manifest.resolve()
    prompt_path = args.prompt.resolve()
    if not manifest_path.is_file() or not prompt_path.is_file():
        parser.error("manifest or prompt does not exist")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = sha256_text(prompt_text)
    samples = select_demo_samples(load_samples(manifest_path, benchmark_root))
    ground_truth = build_ground_truth(benchmark_root)
    api_key = os.getenv("DASHSCOPE_API_KEY") or read_api_key(args.api_file.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample in samples:
        print("running %s %s" % (track_group(sample["track"]), sample["sample_id"]), file=sys.stderr)
        result = process_sample(
            sample,
            ground_truth[sample["sample_id"]],
            api_key,
            prompt_text,
            prompt_hash,
            args,
        )
        atomic_write_json(args.output_dir / (sample["sample_id"] + ".json"), result)
        results.append(result)
        print("%s: %s" % (sample["sample_id"], result["status"]), file=sys.stderr)

    summary = {
        "sample_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "failed_count": sum(item["status"] != "success" for item in results),
        "sample_ids": [item["sample_id"] for item in results],
        "input_contract": "system_prompt_plus_official_reference_facts_plus_one_audio",
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "reference_contract_version": REFERENCE_CONTRACT_VERSION,
        "model": args.model,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
