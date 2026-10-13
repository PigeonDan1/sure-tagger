#!/usr/bin/env python3
"""Extract one raw caption text into the complete schema with DeepSeek."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "eval" / "captioner_deepseek_extraction_prompt.txt"
# API credentials live outside the documentation tree.  Keep this default in
# sync with the batch runner and the user-provided compare/api location.
DEFAULT_API_FILE = ROOT / "api" / "deepseek-api.txt"
DEFAULT_INPUT = Path(
    "/hpc_stor03/sjtu_home/weihan.chen/share/caption_pairs_3000/"
    "0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2.txt"
)
DEFAULT_OUTPUT = ROOT / "evaluation" / "captioner-v0" / "deepseek-text-single.json"

TOP_LEVEL_KEYS = {
    "sample_id",
    "schema_version",
    "field_universe_version",
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
    "descriptive",
}
EVENTS = ("speech", "singing", "music")
NOISE_CLASSES = (
    "music",
    "animal",
    "mechanical",
    "nature",
    "formless",
    "channel_environment",
)
PROFILE_KEYS = {"speaker_id", "speech_rate", "pitch", "speaker_volume"}
SPEECH_RATE_KEYS = {"band", "value", "unit"}
DESCRIPTIVE_KEYS = {
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
}
FACET_KEYS = {"name", "description", "evidence_tags", "confidence"}
SPEAKER_DESCRIPTION_KEYS = {
    "speaker_id",
    "gender",
    "description",
    "descriptive_phrases",
    "evidence_tags",
    "confidence",
}
HYPOTHESIS_KEYS = {"description", "evidence_tags", "confidence"}
STRUCTURAL_EVIDENCE_PATHS = {
    "basic_acoustic.duration_sec",
    "basic_acoustic.sample_rate_hz",
    "basic_acoustic.channels",
    "basic_acoustic.nonspeech_ratio",
    "basic_acoustic.silence_segments",
    "audio_quality.snr_db",
    "audio_quality.dnsmos_sig",
    "audio_quality.dnsmos_bak",
    "audio_quality.dnsmos_ovrl",
    "audio_quality.dnsmos_p808",
    "room_acoustic.far_field",
    "room_acoustic.rt60_sec",
    "room_acoustic.c50_db",
    "sound_field_scene.speech_music_events",
    "sound_field_scene.music_present",
    "sound_field_scene.external_noise_type",
    "sound_field_scene.noise_composition",
    "speaker.speaker_count",
    "speaker.speaker_present",
    "speaker.asr_transcript",
    "speaker.speaker_transcripts",
    "speaker.multi_speaker",
    "speaker.speaker_change_count",
    "speaker.speaker_change",
    "speaker.overlap_ratio",
    "speaker.speaker_overlap",
    "speaker.profiles",
    "language_content.language",
    "language_content.word_count",
    "language_content.punctuation.punctuation_count",
    "language_content.punctuation.has_terminal_punctuation",
    "language_content.repetition.has_repetition",
    "language_content.repetition.repetition_count",
    "language_content.filler",
    "deterministic_derived.speaker_change_rate_per_min",
    "deterministic_derived.filler_ratio",
}
STRUCTURAL_GROUPS = tuple(
    sorted(
        {
            path.split(".", 1)[0]
            for path in STRUCTURAL_EVIDENCE_PATHS
        }
    )
)
COMPOSITE_EVIDENCE_PARENTS = (
    "basic_acoustic.silence_segments",
    "sound_field_scene.noise_composition",
    "speaker.profiles",
    "speaker.speaker_transcripts",
)


class ValidationError(ValueError):
    pass


def strict_json(text):
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: %s" % value)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: %s" % key)
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def read_api_file(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("API file is empty")
    values = {"api_key": None, "base_url": None, "model": None}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if key in values:
                values[key] = value
        elif values["api_key"] is None:
            values["api_key"] = line
    if not values["api_key"]:
        raise ValueError("API file has no API key")
    if not values["base_url"] or not values["model"]:
        raise ValueError("API file must define base_url and model")
    return values


def endpoint(base_url):
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def content_text(message):
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return ""


def call_api(
    api_key,
    endpoint_url,
    model,
    prompt,
    input_text,
    timeout,
    max_tokens,
    thinking,
):
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("input text is empty")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                # The user message is deliberately the raw caption text only.
                "content": input_text,
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": int(max_tokens),
        "stream": False,
        # This gateway supports the generic JSON-object contract. The local
        # validator below remains authoritative for the exact field universe.
        "response_format": {"type": "json_object"},
        # The gateway honors this top-level switch; chat_template_kwargs does not.
        "thinking": {"type": thinking},
    }
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "User-Agent": "captioner-structural-extractor/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError("HTTP %s: %s" % (exc.code, detail))
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("request failed: %s" % exc)
    response_json = strict_json(raw)
    if status != 200:
        raise RuntimeError("HTTP %s: %s" % (status, response_json.get("error")))
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("response has no choices")
    choice = choices[0]
    if choice.get("finish_reason") not in (None, "stop"):
        raise RuntimeError("finish_reason=%s" % choice.get("finish_reason"))
    text = content_text(choice.get("message") or {})
    if not text:
        raise RuntimeError("response content is empty")
    return strict_json(text), response_json.get("usage") or {}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def require_keys(value, keys, path):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValidationError("%s must contain exactly %s" % (path, sorted(keys)))


def dotted_value(payload, path):
    value = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def canonical_evidence_paths(tag, payload):
    """Map common group/composite aliases to exact, non-null field paths."""
    if not isinstance(tag, str):
        return []
    tag = tag.strip()
    if tag.startswith("structural."):
        tag = tag[len("structural.") :]
    if tag in STRUCTURAL_EVIDENCE_PATHS:
        return [tag] if dotted_value(payload, tag) is not None else []

    # A group-level reference expands only to exact paths with known values.
    if tag in STRUCTURAL_GROUPS:
        paths = []
        for path in sorted(STRUCTURAL_EVIDENCE_PATHS):
            if path.split(".", 1)[0] == tag and dotted_value(payload, path) is not None:
                paths.append(path)
        return paths

    # Composite children (including array-index notation) are represented by
    # their field-universe parent because the universe does not score leaves.
    for parent in COMPOSITE_EVIDENCE_PARENTS:
        if tag.startswith(parent + ".") or tag.startswith(parent + "["):
            return [parent] if dotted_value(payload, parent) is not None else []
    return []


def canonicalize_descriptive_evidence(payload):
    descriptive = payload.get("descriptive") if isinstance(payload, dict) else None
    if not isinstance(descriptive, dict):
        return

    def rewrite(item):
        if not isinstance(item, dict) or not isinstance(item.get("evidence_tags"), list):
            return
        rewritten = []
        for tag in item["evidence_tags"]:
            for path in canonical_evidence_paths(tag, payload):
                if path not in rewritten:
                    rewritten.append(path)
        item["evidence_tags"] = rewritten

    for key in ("facets", "speaker_descriptions", "scene_hypotheses"):
        values = descriptive.get(key)
        if isinstance(values, list):
            for item in values:
                rewrite(item)


def validate_string_array(value, path, unique=False):
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValidationError("%s must be a non-empty string array or null" % path)
    if unique and len(value) != len(set(value)):
        raise ValidationError("%s must not contain duplicates" % path)


def validate_confidence(value, path, nullable=True):
    if value is None and nullable:
        return
    if not finite_number(value) or not 0 <= value <= 1:
        raise ValidationError("%s must be in [0,1]%s" % (path, " or null" if nullable else ""))


def validate_evidence_tags(value, path, payload):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValidationError("%s must be a string array" % path)
    if len(value) != len(set(value)):
        raise ValidationError("%s must not contain duplicates" % path)
    for tag in value:
        if tag not in STRUCTURAL_EVIDENCE_PATHS:
            raise ValidationError("%s contains non-structural tag %s" % (path, tag))
        if dotted_value(payload, tag) is None:
            raise ValidationError("%s references unknown structural tag %s" % (path, tag))


def validate_descriptive(descriptive, payload):
    require_keys(descriptive, DESCRIPTIVE_KEYS, "descriptive")
    for key in (
        "caption",
        "topic",
        "background_condition",
        "dominant_content_impression",
        "clarity_intelligibility_impression",
        "background_sound_summary",
    ):
        value = descriptive[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValidationError("descriptive.%s must be a non-empty string or null" % key)

    validate_string_array(descriptive["descriptive_phrases"], "descriptive.descriptive_phrases", unique=True)

    facets = descriptive["facets"]
    if facets is not None:
        if not isinstance(facets, list):
            raise ValidationError("descriptive.facets must be an array or null")
        for index, facet in enumerate(facets):
            path = "descriptive.facets[%s]" % index
            require_keys(facet, FACET_KEYS, path)
            for key in ("name", "description"):
                if not isinstance(facet[key], str) or not facet[key].strip():
                    raise ValidationError("%s.%s must be a non-empty string" % (path, key))
            validate_evidence_tags(facet["evidence_tags"], path + ".evidence_tags", payload)
            validate_confidence(facet["confidence"], path + ".confidence", nullable=True)

    profiles = payload["speaker"]["profiles"]
    profile_ids = set()
    if isinstance(profiles, list):
        profile_ids = set(profile["speaker_id"] for profile in profiles)
    speaker_descriptions = descriptive["speaker_descriptions"]
    if speaker_descriptions is not None:
        if not isinstance(speaker_descriptions, list):
            raise ValidationError("descriptive.speaker_descriptions must be an array or null")
        for index, item in enumerate(speaker_descriptions):
            path = "descriptive.speaker_descriptions[%s]" % index
            require_keys(item, SPEAKER_DESCRIPTION_KEYS, path)
            if not isinstance(item["speaker_id"], str) or item["speaker_id"] not in profile_ids:
                raise ValidationError("%s.speaker_id must reference speaker.profiles" % path)
            if item["gender"] is not None and item["gender"] not in {"male", "female"}:
                raise ValidationError("%s.gender must be male, female, or null" % path)
            if not isinstance(item["description"], str) or not item["description"].strip():
                raise ValidationError("%s.description must be a non-empty string" % path)
            validate_string_array(item["descriptive_phrases"], path + ".descriptive_phrases", unique=True)
            validate_evidence_tags(item["evidence_tags"], path + ".evidence_tags", payload)
            validate_confidence(item["confidence"], path + ".confidence", nullable=True)

    hypotheses = descriptive["scene_hypotheses"]
    if hypotheses is not None:
        if not isinstance(hypotheses, list):
            raise ValidationError("descriptive.scene_hypotheses must be an array or null")
        for index, item in enumerate(hypotheses):
            path = "descriptive.scene_hypotheses[%s]" % index
            require_keys(item, HYPOTHESIS_KEYS, path)
            if not isinstance(item["description"], str) or not item["description"].strip():
                raise ValidationError("%s.description must be a non-empty string" % path)
            validate_evidence_tags(item["evidence_tags"], path + ".evidence_tags", payload)
            validate_confidence(item["confidence"], path + ".confidence", nullable=False)

    validate_string_array(descriptive["usability_notes"], "descriptive.usability_notes")
    validate_string_array(descriptive["uncertainties"], "descriptive.uncertainties")


def validate_speaker_transcripts(speaker):
    """Validate the optional plus field without discarding unrelated evidence."""
    count = speaker.get("speaker_count")
    profiles = speaker.get("profiles")
    transcripts = speaker.get("speaker_transcripts")
    if transcripts is not None:
        if not isinstance(transcripts, list):
            raise ValidationError("speaker_transcripts must be array or null")
        transcript_ids = []
        for index, item in enumerate(transcripts):
            path = "speaker_transcripts[%s]" % index
            require_keys(item, {"speaker_id", "text"}, path)
            if not isinstance(item["speaker_id"], str) or not re.fullmatch(
                r"speaker_[1-9][0-9]*", item["speaker_id"]
            ):
                raise ValidationError("invalid transcript speaker_id")
            if item["text"] is not None and (
                not isinstance(item["text"], str) or not item["text"].strip()
            ):
                raise ValidationError("transcript text must be nonempty string or null")
            transcript_ids.append(item["speaker_id"])
        expected_ids = ["speaker_%d" % index for index in range(1, len(transcripts) + 1)]
        if transcript_ids != expected_ids:
            raise ValidationError("speaker_transcripts IDs must be unique and in first-onset order")
        if count is not None and count != len(transcripts):
            raise ValidationError("speaker_transcripts must match speaker_count")
        if not transcripts and count != 0:
            raise ValidationError("empty speaker_transcripts requires confirmed zero speakers")
        if isinstance(profiles, list) and transcript_ids != [item["speaker_id"] for item in profiles]:
            raise ValidationError("speaker_transcripts must match profiles identities and order")


def validate_payload(payload, sample_id):
    if not isinstance(payload, dict):
        raise ValidationError("output must be an object")
    if set(payload) != TOP_LEVEL_KEYS:
        raise ValidationError("top-level keys mismatch: %s" % sorted(set(payload) ^ TOP_LEVEL_KEYS))
    if payload["sample_id"] != sample_id:
        raise ValidationError("sample_id does not match input")
    if payload["schema_version"] != "structural-prediction-v0.1":
        raise ValidationError("unexpected schema_version")
    if payload["field_universe_version"] != "structural-v0.1":
        raise ValidationError("unexpected field_universe_version")

    basic = payload["basic_acoustic"]
    require_keys(basic, {"duration_sec", "sample_rate_hz", "channels", "nonspeech_ratio", "silence_segments"}, "basic_acoustic")
    for key in ("duration_sec", "nonspeech_ratio"):
        if basic[key] is not None and not finite_number(basic[key]):
            raise ValidationError("basic_acoustic.%s must be finite number or null" % key)
    if basic["duration_sec"] is not None and basic["duration_sec"] < 0:
        raise ValidationError("duration_sec must be non-negative")
    for key in ("sample_rate_hz", "channels"):
        if basic[key] is not None and (not isinstance(basic[key], int) or isinstance(basic[key], bool) or basic[key] < 1):
            raise ValidationError("basic_acoustic.%s must be positive integer or null" % key)
    if basic["nonspeech_ratio"] is not None and not 0 <= basic["nonspeech_ratio"] <= 1:
        raise ValidationError("nonspeech_ratio outside [0,1]")
    segments = basic["silence_segments"]
    if segments is not None:
        if not isinstance(segments, list):
            raise ValidationError("silence_segments must be array or null")
        previous_end = None
        for index, segment in enumerate(segments):
            require_keys(segment, {"start_sec", "end_sec"}, "silence_segments[%s]" % index)
            start, end = segment["start_sec"], segment["end_sec"]
            if not finite_number(start) or not finite_number(end) or start < 0 or start >= end:
                raise ValidationError("invalid silence segment %s" % index)
            if previous_end is not None and start < previous_end:
                raise ValidationError("silence_segments are not sorted/non-overlapping")
            if basic["duration_sec"] is not None and end > basic["duration_sec"] + 1e-6:
                raise ValidationError("silence segment exceeds duration")
            previous_end = end

    quality = payload["audio_quality"]
    require_keys(quality, {"snr_db", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"}, "audio_quality")
    if quality["snr_db"] is not None and not finite_number(quality["snr_db"]):
        raise ValidationError("snr_db must be finite number or null")
    for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        value = quality[key]
        if value is not None and (not finite_number(value) or not 1 <= value <= 5):
            raise ValidationError("%s outside [1,5]" % key)

    room = payload["room_acoustic"]
    require_keys(room, {"far_field", "rt60_sec", "c50_db"}, "room_acoustic")
    if room["far_field"] is not None and not isinstance(room["far_field"], bool):
        raise ValidationError("far_field must be boolean or null")
    for key in ("rt60_sec", "c50_db"):
        if room[key] is not None and not finite_number(room[key]):
            raise ValidationError("%s must be finite number or null" % key)
    if room["rt60_sec"] is not None and room["rt60_sec"] < 0:
        raise ValidationError("rt60_sec must be non-negative")

    scene = payload["sound_field_scene"]
    require_keys(scene, {"speech_music_events", "music_present", "external_noise_type", "noise_composition"}, "sound_field_scene")
    events = scene["speech_music_events"]
    if events is not None:
        if not isinstance(events, list) or any(item not in EVENTS for item in events) or len(events) != len(set(events)):
            raise ValidationError("invalid speech_music_events")
        if events != [item for item in EVENTS if item in events]:
            raise ValidationError("speech_music_events has wrong order")
    if scene["music_present"] is not None and not isinstance(scene["music_present"], bool):
        raise ValidationError("music_present must be boolean or null")
    noise = scene["external_noise_type"]
    if noise is not None:
        if not isinstance(noise, list) or any(item not in NOISE_CLASSES for item in noise) or len(noise) != len(set(noise)):
            raise ValidationError("invalid external_noise_type")
    composition = scene["noise_composition"]
    if composition is not None:
        require_keys(composition, set(NOISE_CLASSES), "noise_composition")
        for key in NOISE_CLASSES:
            if not isinstance(composition[key], list) or any(not isinstance(item, str) or not item for item in composition[key]):
                raise ValidationError("invalid noise_composition.%s" % key)

    speaker = payload["speaker"]
    if isinstance(speaker, dict):
        # Historical outputs predate the plus field; missing means unavailable.
        speaker = dict(speaker)
        speaker.setdefault("speaker_transcripts", None)
    require_keys(speaker, {"speaker_count", "speaker_present", "asr_transcript", "speaker_transcripts", "multi_speaker", "speaker_change_count", "speaker_change", "overlap_ratio", "speaker_overlap", "profiles"}, "speaker")
    if speaker["asr_transcript"] is not None and not isinstance(speaker["asr_transcript"], str):
        raise ValidationError("asr_transcript must be string or null")
    count = speaker["speaker_count"]
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
        raise ValidationError("invalid speaker_count")
    if speaker["speaker_present"] is not None and not isinstance(speaker["speaker_present"], bool):
        raise ValidationError("invalid speaker_present")
    for key in ("multi_speaker", "speaker_change", "speaker_overlap"):
        if speaker[key] is not None and not isinstance(speaker[key], bool):
            raise ValidationError("invalid %s" % key)
    change_count = speaker["speaker_change_count"]
    if change_count is not None and (not isinstance(change_count, int) or isinstance(change_count, bool) or change_count < 0):
        raise ValidationError("invalid speaker_change_count")
    overlap = speaker["overlap_ratio"]
    if overlap is not None and (not finite_number(overlap) or not 0 <= overlap <= 1):
        raise ValidationError("invalid overlap_ratio")
    profiles = speaker["profiles"]
    if profiles is not None:
        if not isinstance(profiles, list):
            raise ValidationError("profiles must be array or null")
        ids = []
        for index, profile in enumerate(profiles):
            require_keys(profile, PROFILE_KEYS, "profiles[%s]" % index)
            if not isinstance(profile["speaker_id"], str) or not profile["speaker_id"]:
                raise ValidationError("invalid profile speaker_id")
            ids.append(profile["speaker_id"])
            rate = profile["speech_rate"]
            require_keys(rate, SPEECH_RATE_KEYS, "profiles[%s].speech_rate" % index)
            if rate["band"] is not None and rate["band"] not in ("slow", "normal", "fast", "variable"):
                raise ValidationError("invalid speech rate band")
            if rate["value"] is not None and not finite_number(rate["value"]):
                raise ValidationError("invalid speech rate value")
            if rate["unit"] is not None and rate["unit"] not in ("zh_char_per_sec", "word_per_min"):
                raise ValidationError("invalid speech rate unit")
            if rate["value"] is not None and rate["unit"] is None:
                raise ValidationError("speech rate value requires unit")
            if profile["pitch"] is not None and profile["pitch"] not in ("low", "mid", "high", "variable"):
                raise ValidationError("invalid profile pitch")
            if profile["speaker_volume"] is not None and profile["speaker_volume"] not in ("low", "normal", "loud", "variable"):
                raise ValidationError("invalid profile speaker_volume")
        if len(ids) != len(set(ids)):
            raise ValidationError("duplicate profile speaker_id")

    validate_speaker_transcripts(speaker)

    language = payload["language_content"]
    require_keys(language, {"language", "word_count", "punctuation", "repetition", "filler"}, "language_content")
    if language["language"] is not None and not isinstance(language["language"], str):
        raise ValidationError("language must be string or null")
    for key in ("word_count", "filler"):
        value = language[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValidationError("invalid language_content.%s" % key)
    punctuation = language["punctuation"]
    require_keys(punctuation, {"punctuation_count", "has_terminal_punctuation"}, "language_content.punctuation")
    if punctuation["punctuation_count"] is not None and (not isinstance(punctuation["punctuation_count"], int) or isinstance(punctuation["punctuation_count"], bool) or punctuation["punctuation_count"] < 0):
        raise ValidationError("invalid punctuation_count")
    if punctuation["has_terminal_punctuation"] is not None and not isinstance(punctuation["has_terminal_punctuation"], bool):
        raise ValidationError("invalid has_terminal_punctuation")
    repetition = language["repetition"]
    require_keys(repetition, {"has_repetition", "repetition_count"}, "language_content.repetition")
    if repetition["has_repetition"] is not None and not isinstance(repetition["has_repetition"], bool):
        raise ValidationError("invalid has_repetition")
    if repetition["repetition_count"] is not None and (not isinstance(repetition["repetition_count"], int) or isinstance(repetition["repetition_count"], bool) or repetition["repetition_count"] < 0):
        raise ValidationError("invalid repetition_count")
    derived = payload["deterministic_derived"]
    require_keys(derived, {"speaker_change_rate_per_min", "filler_ratio"}, "deterministic_derived")
    rate = derived["speaker_change_rate_per_min"]
    if rate is not None and (not finite_number(rate) or rate < 0):
        raise ValidationError("speaker_change_rate_per_min must be non-negative number or null")
    ratio = derived["filler_ratio"]
    if ratio is not None and (not finite_number(ratio) or not 0 <= ratio <= 1):
        raise ValidationError("filler_ratio must be in [0,1] or null")
    validate_descriptive(payload["descriptive"], payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        "--input-text",
        dest="input_file",
        type=Path,
        default=DEFAULT_INPUT,
        help="UTF-8 caption text; its contents are the entire user message",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="optional external result ID; injected after the API call, never sent as input",
    )
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="enabled",
        help="DeepSeek hidden reasoning mode; enabled is the quality-first default",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = read_api_file(args.api_file)
    input_bytes = args.input_file.read_bytes()
    input_text = input_bytes.decode("utf-8")
    if not input_text.strip():
        raise SystemExit("input text is empty")
    sample_id = args.sample_id if args.sample_id else None
    prompt = args.prompt.read_text(encoding="utf-8")
    payload, usage = call_api(
        config["api_key"],
        endpoint(config["base_url"]),
        config["model"],
        prompt,
        input_text,
        args.timeout,
        args.max_tokens,
        args.thinking,
    )
    # The identifier is caller metadata, not evidence extracted from text.
    if isinstance(payload, dict):
        payload["sample_id"] = sample_id
    canonicalize_descriptive_evidence(payload)
    validate_payload(payload, sample_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "succeeded",
        "sample_id": sample_id,
        "output": str(args.output.resolve()),
        "source_mode": "raw_caption_text",
        "input_file": str(args.input_file.resolve()),
        "input_bytes": len(input_bytes),
        "input_chars": len(input_text),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "model": config["model"],
        "endpoint": endpoint(config["base_url"]),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "thinking": args.thinking,
        "usage": usage,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
