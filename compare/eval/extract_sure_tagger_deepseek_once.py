#!/usr/bin/env python3
"""Generate descriptive tags from one sure-tagger structural output."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from extract_captioner_deepseek_once import (
    DESCRIPTIVE_KEYS,
    ValidationError,
    call_api,
    endpoint,
    read_api_file,
    strict_json,
    validate_payload,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = COMPARE_ROOT.parent
DEFAULT_PROMPT = SCRIPT_DIR / "sure_tagger_deepseek_language_descriptive_prompt.txt"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "deepseek-api.txt"
DEFAULT_INPUT = (
    WORKSPACE_ROOT
    / "sure-tagger-main"
    / "outputs"
    / "smoke_structural_complete.jsonl"
)
DEFAULT_OUTPUT = (
    COMPARE_ROOT / "evaluation" / "sure-tagger-v0" / "descriptive-single.json"
)
SCHEMA_VERSION = "structural-prediction-v0.1"
FIELD_UNIVERSE_VERSION = "structural-v0.1"

STRUCTURAL_GROUPS = (
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
)
STRUCTURAL_FIELDS = {
    "basic_acoustic": {
        "duration_sec",
        "sample_rate_hz",
        "channels",
        "nonspeech_ratio",
        "silence_segments",
    },
    "audio_quality": {
        "snr_db",
        "dnsmos_sig",
        "dnsmos_bak",
        "dnsmos_ovrl",
        "dnsmos_p808",
    },
    "room_acoustic": {"far_field", "rt60_sec", "c50_db"},
    "sound_field_scene": {
        "speech_music_events",
        "music_present",
        "external_noise_type",
        "noise_composition",
    },
    "speaker": {
        "speaker_count",
        "speaker_present",
        "asr_transcript",
        "speaker_transcripts",
        "multi_speaker",
        "speaker_change_count",
        "speaker_change",
        "overlap_ratio",
        "speaker_overlap",
        "profiles",
    },
    "language_content": {
        "language",
        "word_count",
        "punctuation",
        "repetition",
        "filler",
    },
    "deterministic_derived": {
        "speaker_change_rate_per_min",
        "filler_ratio",
    },
}
FIELD_ORDER = {
    "basic_acoustic": (
        "duration_sec",
        "sample_rate_hz",
        "channels",
        "nonspeech_ratio",
        "silence_segments",
    ),
    "audio_quality": (
        "snr_db",
        "dnsmos_sig",
        "dnsmos_bak",
        "dnsmos_ovrl",
        "dnsmos_p808",
    ),
    "room_acoustic": ("far_field", "rt60_sec", "c50_db"),
    "sound_field_scene": (
        "speech_music_events",
        "music_present",
        "external_noise_type",
        "noise_composition",
    ),
    "speaker": (
        "speaker_count",
        "speaker_present",
        "asr_transcript",
        "speaker_transcripts",
        "multi_speaker",
        "speaker_change_count",
        "speaker_change",
        "overlap_ratio",
        "speaker_overlap",
        "profiles",
    ),
    "language_content": (
        "language",
        "word_count",
        "punctuation",
        "repetition",
        "filler",
    ),
    "deterministic_derived": (
        "speaker_change_rate_per_min",
        "filler_ratio",
    ),
}
DESCRIPTIVE_ORDER = (
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


def load_structural(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError("input JSONL must contain exactly one non-empty row")
        value = strict_json(lines[0])
    else:
        value = strict_json(text)
    if not isinstance(value, dict):
        raise ValueError("sure-tagger input must be a JSON object")
    if set(value) != set(STRUCTURAL_GROUPS):
        raise ValueError(
            "sure-tagger structural groups mismatch: %s"
            % sorted(set(value) ^ set(STRUCTURAL_GROUPS))
        )
    for group in STRUCTURAL_GROUPS:
        fields = value[group]
        if group == "speaker" and isinstance(fields, dict):
            fields.setdefault("speaker_transcripts", None)
        if not isinstance(fields, dict) or set(fields) != STRUCTURAL_FIELDS[group]:
            raise ValueError("sure-tagger fields mismatch in %s" % group)
    return value


def descriptive_input(structural):
    return json.dumps(structural, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_descriptive(value):
    """Return only the 12-field descriptive object from a model response.

    Kept for callers that do not care about the language verdict; use
    :func:`split_output` when the verdict is needed.
    """
    return split_output(value)[0]


def normalize_language_verification(value):
    """Coerce the ``language_verification`` object into a fixed shape.

    The verdict is advisory evidence: a malformed value degrades field by
    field to null (meaning "abstain, keep the upstream LID value") instead of
    raising, because losing the whole sample over an optional side field would
    be far worse than keeping the LID guess.
    """
    if not isinstance(value, dict):
        return None
    language = value.get("language")
    if isinstance(language, str):
        language = language.strip().lower() or None
    elif language is not None:
        language = None
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    else:
        confidence = float(confidence)
        confidence = min(1.0, max(0.0, confidence))
    reason = value.get("reason_short")
    if reason is not None and not isinstance(reason, str):
        reason = None
    return {"language": language, "confidence": confidence, "reason_short": reason}


def split_output(value):
    """Split a model response into ``(descriptive, language_verification)``."""
    return split_sure_tagger_output(value)


def split_sure_tagger_output(value):
    """Split the v1 response into descriptive tags and a language verdict.

    Older descriptive-only objects remain valid and have no language verdict.
    """
    if not isinstance(value, dict):
        raise ValidationError("DeepSeek output must be an object")
    verification = None
    inner = value.get("descriptive")
    if isinstance(inner, dict):
        extra = set(value) - {"descriptive", "language_verification"}
        if extra:
            raise ValidationError("DeepSeek response has unsupported keys: %s" % sorted(extra))
        verification = normalize_language_verification(value.get("language_verification"))
        value = inner
    if set(value) != DESCRIPTIVE_KEYS:
        raise ValidationError(
            "DeepSeek descriptive keys mismatch: %s"
            % sorted(set(value) ^ set(DESCRIPTIVE_KEYS))
        )
    for key in ("usability_notes", "uncertainties"):
        if isinstance(value.get(key), str):
            value[key] = [value[key]]
    return {key: value[key] for key in DESCRIPTIVE_ORDER}, verification


def resolve_language(structural, verification, require_dependency_review=False):
    """Decide the published language for one sample.

    Returns ``(published_language, verdict_state, changed, verified_language)``.
    The single sanctioned structural override in this pipeline lives here, and
    both the single-sample driver and the batch runner call it so the rule
    cannot drift between the two paths.
    """
    language = structural.get("language_content")
    lid = language.get("language") if isinstance(language, dict) else None
    speaker = structural.get("speaker")
    transcript = speaker.get("asr_transcript") if isinstance(speaker, dict) else None
    has_transcript = isinstance(transcript, str) and bool(transcript.strip())
    verified = verification.get("language") if verification else None
    if not has_transcript:
        # Nothing to read: a verdict here would be unverifiable, so refuse the
        # override in code rather than trusting the prompt to abstain.
        return lid, "no_transcript", False, verified
    if verification is None:
        return lid, "absent", False, None
    if verified is None:
        return lid, "abstained", False, None
    if verified == lid:
        return lid, "agreed", False, verified
    if require_dependency_review:
        return lid, "pending_dependency_review", False, verified
    return verified, "changed", True, verified


def language_verification_row(
    structural, verification, sample_id=None, require_dependency_review=False
):
    """Audit record for one language verdict; never sent to the model."""
    published, state, changed, verified = resolve_language(
        structural, verification, require_dependency_review=require_dependency_review
    )
    speaker = structural.get("speaker")
    transcript = speaker.get("asr_transcript") if isinstance(speaker, dict) else None
    transcript = transcript if isinstance(transcript, str) else ""
    language = structural.get("language_content")
    row = {
        "sample_id": sample_id,
        "lid_language": language.get("language") if isinstance(language, dict) else None,
        "published_language": published,
        "verified_language": verified,
        "confidence": verification.get("confidence") if verification else None,
        "reason_short": verification.get("reason_short") if verification else None,
        "changed": changed,
        "verdict_state": state,
        "transcript_chars": len(transcript),
        "transcript_sha256": (
            hashlib.sha256(transcript.encode("utf-8")).hexdigest() if transcript else None
        ),
    }
    if state == "pending_dependency_review":
        row["dependency_review_required"] = True
        row["dependency_issue"] = {
            "issue_id": "language_verification_conflict",
            "target_fields": ["language_content.language", "speaker.speaker_transcripts"],
            "question": "Recheck ASR, alignment, speaker gender and text dependencies before publishing the verified language.",
        }
    return row


def canonical_structural(structural):
    """Represent native sure-tagger tags in the frozen prediction envelope."""
    result = {
        group: {
            field: copy.deepcopy(
                structural[group].get(field)
                if group == "speaker" and field == "speaker_transcripts"
                else structural[group][field]
            )
            for field in FIELD_ORDER[group]
        }
        for group in STRUCTURAL_GROUPS
    }
    # Gender is scored from descriptive.speaker_descriptions and must not be
    # included in the structural evidence sent to DeepSeek.  A few finalized
    # runs also persisted it on profiles, so remove that compatibility field
    # from the wire representation while leaving the source result untouched.
    profiles = result["speaker"].get("profiles")
    if isinstance(profiles, list):
        for profile in profiles:
            if isinstance(profile, dict):
                profile.pop("gender", None)
    language = result["language_content"]
    if language["punctuation"] is None:
        language["punctuation"] = {
            "punctuation_count": None,
            "has_terminal_punctuation": None,
        }
    if language["repetition"] is None:
        language["repetition"] = {
            "has_repetition": None,
            "repetition_count": None,
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tags", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="external sample ID injected into the output envelope, never sent to DeepSeek",
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--language-verification-output",
        type=Path,
        default=None,
        help="optional path for the single-sample language verdict audit record",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument(
        "--require-language-dependency-review", action="store_true",
        help="Defer language changes to the plus evidence loop; withhold descriptions while review is pending.",
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="enabled",
        help="DeepSeek hidden reasoning mode; enabled is the quality-first default",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    structural = canonical_structural(load_structural(args.input_tags))
    prompt = args.prompt.read_text(encoding="utf-8")
    config = read_api_file(args.api_file)
    descriptive, usage = call_api(
        config["api_key"],
        endpoint(config["base_url"]),
        config["model"],
        prompt,
        descriptive_input(structural),
        args.timeout,
        args.max_tokens,
        args.thinking,
    )
    descriptive, verification = split_sure_tagger_output(descriptive)
    candidate = {
        "sample_id": args.sample_id,
        "schema_version": SCHEMA_VERSION,
        "field_universe_version": FIELD_UNIVERSE_VERSION,
        "basic_acoustic": copy.deepcopy(structural["basic_acoustic"]),
        "audio_quality": copy.deepcopy(structural["audio_quality"]),
        "room_acoustic": copy.deepcopy(structural["room_acoustic"]),
        "sound_field_scene": copy.deepcopy(structural["sound_field_scene"]),
        "speaker": copy.deepcopy(structural["speaker"]),
        "language_content": copy.deepcopy(structural["language_content"]),
        "deterministic_derived": copy.deepcopy(structural["deterministic_derived"]),
        "descriptive": descriptive,
    }
    row = language_verification_row(
        structural, verification, args.sample_id,
        require_dependency_review=args.require_language_dependency_review,
    )
    if row["verdict_state"] == "pending_dependency_review":
        candidate["descriptive"] = {key: None for key in DESCRIPTIVE_ORDER}
    candidate["language_content"]["language"] = row["published_language"]
    validate_payload(candidate, args.sample_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.language_verification_output is not None:
        args.language_verification_output.parent.mkdir(parents=True, exist_ok=True)
        args.language_verification_output.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    input_bytes = args.input_tags.read_bytes()
    prompt_bytes = prompt.encode("utf-8")
    print(
        json.dumps(
            {
                "status": "succeeded",
                "sample_id": args.sample_id,
                "input_tags": str(args.input_tags.resolve()),
                "output": str(args.output.resolve()),
                "model": config["model"],
                "endpoint": endpoint(config["base_url"]),
                "thinking": args.thinking,
                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "language_verification": row,
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
