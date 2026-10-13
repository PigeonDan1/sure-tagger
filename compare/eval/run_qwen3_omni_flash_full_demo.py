#!/usr/bin/env python3
"""Run the documented Qwen3-Omni-Flash -> DeepSeek full-schema demo."""

from __future__ import print_function

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import extract_captioner_deepseek_once as deepseek_api
import run_qwen3_5_omni_plus_descriptive_gt_demo as qwen_api


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_AUDIO = (
    COMPARE_ROOT / "dataset" / "Ami" / "data" / "audio" / "IS1006c_utterance_00027.wav"
)
DEFAULT_QWEN_PROMPT = SCRIPT_DIR / "qwen3_omni_flash_full_schema_prompt.txt"
DEFAULT_DEEPSEEK_PROMPT = SCRIPT_DIR / "deepseek_full_schema_repair_prompt.txt"
DEFAULT_QWEN_API = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"
DEFAULT_DEEPSEEK_API = COMPARE_ROOT / "api" / "deepseek-api.txt"
DEFAULT_OUTPUT_DIR = COMPARE_ROOT / "evaluation" / "qwen3_omni_flash_full_demo"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-omni-flash"

TOP_LEVEL_KEYS = {
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
    "descriptive",
}
GROUP_KEYS = {
    "basic_acoustic": {
        "duration_sec",
        "sample_rate_hz",
        "channels",
        "nonspeech_ratio",
        "silence_segments",
    },
    "audio_quality": {"snr_db", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"},
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
    "language_content": {"language", "word_count", "punctuation", "repetition", "filler"},
    "deterministic_derived": {"speaker_change_rate_per_min", "filler_ratio"},
    "descriptive": {
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
    },
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
LANGUAGES = {"ar", "zh", "en", "es", "fr", "hi", "ja", "ko", "ru", "sw"}
PROFILE_ID_RE = re.compile(r"^speaker_[1-9][0-9]*$")
FACET_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
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
EVIDENCE_PATHS = deepseek_api.STRUCTURAL_EVIDENCE_PATHS


class ValidationError(ValueError):
    pass


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def qwen_messages(system_prompt, user_prompt, audio_bytes):
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64," + encoded,
                        "format": "wav",
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def qwen_stream(base_url, api_key, model, messages, max_tokens, timeout):
    body = {
        "model": model,
        "messages": messages,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_p": 1,
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "qwen3-omni-flash-full-schema-demo/1.0",
        },
        method="POST",
    )
    text_parts = []
    usage = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError("Qwen HTTP %s: %s" % (error.code, qwen_api.redact(detail, api_key)))
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError("Qwen request failed: %s" % error)
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                text_parts.append(qwen_api.json_delta_text(choices[0].get("delta")))
    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError("Qwen returned empty text")
    return text, usage


def require_keys(value, expected, path):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValidationError("%s keys mismatch: %s" % (path, actual))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def optional_number(value, path, low=None, high=None):
    if value is None:
        return
    if not finite_number(value):
        raise ValidationError("%s must be a finite number or null" % path)
    if low is not None and value < low:
        raise ValidationError("%s is below %s" % (path, low))
    if high is not None and value > high:
        raise ValidationError("%s is above %s" % (path, high))


def optional_integer(value, path, minimum=0):
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValidationError("%s must be an integer >= %s or null" % (path, minimum))


def optional_boolean(value, path):
    if value is not None and not isinstance(value, bool):
        raise ValidationError("%s must be boolean or null" % path)


def optional_string(value, path):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValidationError("%s must be a non-empty string or null" % path)


def optional_string_array(value, path, unique=False):
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValidationError("%s must be a string array or null" % path)
    if unique and len(value) != len(set(value)):
        raise ValidationError("%s contains duplicate strings" % path)


def optional_short_phrase_array(value, path):
    optional_string_array(value, path, unique=True)
    if value is None:
        return
    for item in value:
        if len(item) > 40 or any(mark in item for mark in ("。", "！", "？", ".", "!", "?")):
            raise ValidationError("%s contains a sentence instead of a short phrase" % path)


def confidence(value, path, nullable=True):
    if value is None and nullable:
        return
    optional_number(value, path, 0, 1)
    if value is None:
        raise ValidationError("%s cannot be null" % path)


def dotted_value(payload, path):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_evidence(value, path, payload):
    optional_string_array(value, path, unique=True)
    if not isinstance(value, list):
        raise ValidationError("%s must be an array" % path)
    for item in value:
        if item not in EVIDENCE_PATHS:
            raise ValidationError("%s contains unknown path %s" % (path, item))
        if dotted_value(payload, item) is None:
            raise ValidationError("%s references null path %s" % (path, item))


def validate_full_schema(payload):
    require_keys(payload, TOP_LEVEL_KEYS, "root")
    for group, keys in GROUP_KEYS.items():
        fields = payload[group]
        if group == "speaker" and isinstance(fields, dict):
            fields = dict(fields)
            fields.setdefault("speaker_transcripts", None)
        require_keys(fields, keys, group)

    basic = payload["basic_acoustic"]
    optional_number(basic["duration_sec"], "basic_acoustic.duration_sec", 0)
    optional_integer(basic["sample_rate_hz"], "basic_acoustic.sample_rate_hz", 1)
    optional_integer(basic["channels"], "basic_acoustic.channels", 1)
    optional_number(basic["nonspeech_ratio"], "basic_acoustic.nonspeech_ratio", 0, 1)
    segments = basic["silence_segments"]
    if segments is not None:
        if not isinstance(segments, list):
            raise ValidationError("basic_acoustic.silence_segments must be an array or null")
        previous_end = None
        for index, item in enumerate(segments):
            path = "basic_acoustic.silence_segments[%s]" % index
            require_keys(item, {"start_sec", "end_sec"}, path)
            optional_number(item["start_sec"], path + ".start_sec", 0)
            optional_number(item["end_sec"], path + ".end_sec", 0)
            if item["start_sec"] is None or item["end_sec"] is None or item["start_sec"] >= item["end_sec"]:
                raise ValidationError("%s has invalid bounds" % path)
            if previous_end is not None and item["start_sec"] < previous_end:
                raise ValidationError("silence segments overlap or are unsorted")
            if basic["duration_sec"] is not None and item["end_sec"] > basic["duration_sec"] + 1e-6:
                raise ValidationError("%s exceeds duration" % path)
            previous_end = item["end_sec"]

    quality = payload["audio_quality"]
    optional_number(quality["snr_db"], "audio_quality.snr_db")
    for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        optional_number(quality[key], "audio_quality." + key, 1, 5)

    room = payload["room_acoustic"]
    optional_boolean(room["far_field"], "room_acoustic.far_field")
    optional_number(room["rt60_sec"], "room_acoustic.rt60_sec", 0)
    optional_number(room["c50_db"], "room_acoustic.c50_db")

    scene = payload["sound_field_scene"]
    events = scene["speech_music_events"]
    if events is not None:
        if not isinstance(events, list) or events != [item for item in EVENTS if item in events]:
            raise ValidationError("sound_field_scene.speech_music_events has invalid values/order")
    optional_boolean(scene["music_present"], "sound_field_scene.music_present")
    if events is not None and scene["music_present"] is not None:
        if scene["music_present"] != ("music" in events):
            raise ValidationError("music_present conflicts with speech_music_events")
    noise = scene["external_noise_type"]
    if noise is not None:
        if not isinstance(noise, list) or any(item not in NOISE_CLASSES for item in noise) or len(noise) != len(set(noise)):
            raise ValidationError("sound_field_scene.external_noise_type is invalid")
    composition = scene["noise_composition"]
    if composition is not None:
        require_keys(composition, set(NOISE_CLASSES), "sound_field_scene.noise_composition")
        for key in NOISE_CLASSES:
            optional_string_array(composition[key], "sound_field_scene.noise_composition." + key)
            if composition[key] is None:
                raise ValidationError("noise_composition.%s must be an array" % key)

    speaker = payload["speaker"]
    optional_integer(speaker["speaker_count"], "speaker.speaker_count")
    optional_boolean(speaker["speaker_present"], "speaker.speaker_present")
    optional_string(speaker["asr_transcript"], "speaker.asr_transcript")
    optional_boolean(speaker["multi_speaker"], "speaker.multi_speaker")
    optional_integer(speaker["speaker_change_count"], "speaker.speaker_change_count")
    optional_boolean(speaker["speaker_change"], "speaker.speaker_change")
    optional_number(speaker["overlap_ratio"], "speaker.overlap_ratio", 0, 1)
    optional_boolean(speaker["speaker_overlap"], "speaker.speaker_overlap")
    count = speaker["speaker_count"]
    if count is not None:
        if speaker["speaker_present"] is not None and speaker["speaker_present"] != (count > 0):
            raise ValidationError("speaker_present conflicts with speaker_count")
        if speaker["multi_speaker"] is not None and speaker["multi_speaker"] != (count >= 2):
            raise ValidationError("multi_speaker conflicts with speaker_count")
        if count < 2:
            if speaker["speaker_change_count"] not in (None, 0) or speaker["speaker_change"] not in (None, False):
                raise ValidationError("speaker change conflicts with speaker_count")
            if speaker["overlap_ratio"] not in (None, 0) or speaker["speaker_overlap"] not in (None, False):
                raise ValidationError("speaker overlap conflicts with speaker_count")
    if speaker["speaker_change_count"] is not None and speaker["speaker_change"] is not None:
        if speaker["speaker_change"] != (speaker["speaker_change_count"] > 0):
            raise ValidationError("speaker_change conflicts with speaker_change_count")
    if speaker["overlap_ratio"] is not None and speaker["speaker_overlap"] is not None:
        if speaker["speaker_overlap"] != (speaker["overlap_ratio"] > 0):
            raise ValidationError("speaker_overlap conflicts with overlap_ratio")

    try:
        deepseek_api.validate_speaker_transcripts(speaker)
    except deepseek_api.ValidationError as exc:
        raise ValidationError(str(exc))

    profiles = speaker["profiles"]
    profile_ids = []
    if profiles is not None:
        if not isinstance(profiles, list):
            raise ValidationError("speaker.profiles must be an array or null")
        for index, profile in enumerate(profiles):
            path = "speaker.profiles[%s]" % index
            require_keys(profile, {"speaker_id", "speech_rate", "pitch", "speaker_volume"}, path)
            if not isinstance(profile["speaker_id"], str) or not PROFILE_ID_RE.match(profile["speaker_id"]):
                raise ValidationError("%s.speaker_id is invalid" % path)
            profile_ids.append(profile["speaker_id"])
            require_keys(profile["speech_rate"], {"band", "value", "unit"}, path + ".speech_rate")
            rate = profile["speech_rate"]
            if rate["band"] not in (None, "slow", "normal", "fast", "variable"):
                raise ValidationError("%s.speech_rate.band is invalid" % path)
            optional_number(rate["value"], path + ".speech_rate.value", 0)
            if rate["unit"] not in (None, "zh_char_per_sec", "word_per_min"):
                raise ValidationError("%s.speech_rate.unit is invalid" % path)
            if rate["value"] is not None and rate["unit"] is None:
                raise ValidationError("%s speech rate value has no unit" % path)
            if profile["pitch"] not in (None, "low", "mid", "high", "variable"):
                raise ValidationError("%s.pitch is invalid" % path)
            if profile["speaker_volume"] not in (None, "low", "normal", "loud", "variable"):
                raise ValidationError("%s.speaker_volume is invalid" % path)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValidationError("speaker.profiles has duplicate speaker_id")
        if count is not None and len(profiles) != count:
            raise ValidationError("speaker.profiles length conflicts with speaker_count")

    language = payload["language_content"]
    value = language["language"]
    if value is not None and (not isinstance(value, str) or (value not in LANGUAGES and not value.startswith("zh-"))):
        raise ValidationError("language_content.language is invalid")
    optional_integer(language["word_count"], "language_content.word_count")
    optional_integer(language["filler"], "language_content.filler")
    punctuation = language["punctuation"]
    if punctuation is not None:
        require_keys(punctuation, {"punctuation_count", "has_terminal_punctuation"}, "language_content.punctuation")
        optional_integer(punctuation["punctuation_count"], "language_content.punctuation.punctuation_count")
        optional_boolean(punctuation["has_terminal_punctuation"], "language_content.punctuation.has_terminal_punctuation")
        if None in punctuation.values():
            raise ValidationError("language_content.punctuation cannot be partial")
    repetition = language["repetition"]
    if repetition is not None:
        require_keys(repetition, {"has_repetition", "repetition_count"}, "language_content.repetition")
        optional_boolean(repetition["has_repetition"], "language_content.repetition.has_repetition")
        optional_integer(repetition["repetition_count"], "language_content.repetition.repetition_count")
        if None in repetition.values():
            raise ValidationError("language_content.repetition cannot be partial")

    derived = payload["deterministic_derived"]
    optional_number(derived["speaker_change_rate_per_min"], "deterministic_derived.speaker_change_rate_per_min", 0)
    optional_number(derived["filler_ratio"], "deterministic_derived.filler_ratio", 0, 1)
    if speaker["speaker_change_count"] is not None and basic["duration_sec"] is not None and basic["duration_sec"] > 0:
        expected = round(speaker["speaker_change_count"] * 60.0 / basic["duration_sec"], 6)
        if derived["speaker_change_rate_per_min"] is None or abs(derived["speaker_change_rate_per_min"] - expected) > 1e-6:
            raise ValidationError("speaker_change_rate_per_min violates its formula")
        if derived["speaker_change_rate_per_min"] != round(derived["speaker_change_rate_per_min"], 6):
            raise ValidationError("speaker_change_rate_per_min must be rounded to six decimals")
    elif derived["speaker_change_rate_per_min"] is not None:
        raise ValidationError("speaker_change_rate_per_min has missing dependencies")
    if language["filler"] is not None and language["word_count"] is not None and language["word_count"] > 0:
        expected = round(float(language["filler"]) / language["word_count"], 6)
        if derived["filler_ratio"] is None or abs(derived["filler_ratio"] - expected) > 1e-6:
            raise ValidationError("filler_ratio violates its formula")
        if derived["filler_ratio"] != round(derived["filler_ratio"], 6):
            raise ValidationError("filler_ratio must be rounded to six decimals")
    elif derived["filler_ratio"] is not None:
        raise ValidationError("filler_ratio has missing dependencies")

    descriptive = payload["descriptive"]
    for key in (
        "caption",
        "topic",
        "background_condition",
        "dominant_content_impression",
        "clarity_intelligibility_impression",
        "background_sound_summary",
    ):
        optional_string(descriptive[key], "descriptive." + key)
    optional_short_phrase_array(descriptive["descriptive_phrases"], "descriptive.descriptive_phrases")
    for key in ("usability_notes", "uncertainties"):
        optional_string_array(descriptive[key], "descriptive." + key, unique=True)
    if descriptive["topic"] is not None and speaker["asr_transcript"] is None:
        raise ValidationError("descriptive.topic requires speaker.asr_transcript")

    facets = descriptive["facets"]
    if facets is not None:
        if not isinstance(facets, list) or len(facets) > 6:
            raise ValidationError("descriptive.facets must be an array of at most 6 items or null")
        names = []
        for index, item in enumerate(facets):
            path = "descriptive.facets[%s]" % index
            require_keys(item, FACET_KEYS, path)
            optional_string(item["name"], path + ".name")
            optional_string(item["description"], path + ".description")
            if item["name"] is None or item["description"] is None:
                raise ValidationError("%s has an empty required string" % path)
            if not FACET_NAME_RE.match(item["name"]):
                raise ValidationError("%s.name must be lower snake_case" % path)
            names.append(item["name"])
            validate_evidence(item["evidence_tags"], path + ".evidence_tags", payload)
            confidence(item["confidence"], path + ".confidence")
        if len(names) != len(set(names)):
            raise ValidationError("descriptive.facets has duplicate names")

    descriptions = descriptive["speaker_descriptions"]
    if descriptions is not None:
        if not isinstance(descriptions, list):
            raise ValidationError("descriptive.speaker_descriptions must be an array or null")
        ids = []
        for index, item in enumerate(descriptions):
            path = "descriptive.speaker_descriptions[%s]" % index
            require_keys(item, SPEAKER_DESCRIPTION_KEYS, path)
            if item["speaker_id"] not in profile_ids:
                raise ValidationError("%s.speaker_id does not reference speaker.profiles" % path)
            if item["gender"] is not None and item["gender"] not in {"male", "female"}:
                raise ValidationError("%s.gender must be male, female, or null" % path)
            ids.append(item["speaker_id"])
            optional_string(item["description"], path + ".description")
            if item["description"] is None:
                raise ValidationError("%s.description is required" % path)
            optional_short_phrase_array(item["descriptive_phrases"], path + ".descriptive_phrases")
            validate_evidence(item["evidence_tags"], path + ".evidence_tags", payload)
            confidence(item["confidence"], path + ".confidence")
        if len(ids) != len(set(ids)):
            raise ValidationError("speaker_descriptions has duplicate speaker_id")

    hypotheses = descriptive["scene_hypotheses"]
    if hypotheses is not None:
        if not isinstance(hypotheses, list) or len(hypotheses) > 3:
            raise ValidationError("descriptive.scene_hypotheses must be an array of at most 3 items or null")
        for index, item in enumerate(hypotheses):
            path = "descriptive.scene_hypotheses[%s]" % index
            require_keys(item, HYPOTHESIS_KEYS, path)
            optional_string(item["description"], path + ".description")
            if item["description"] is None:
                raise ValidationError("%s.description is required" % path)
            validate_evidence(item["evidence_tags"], path + ".evidence_tags", payload)
            confidence(item["confidence"], path + ".confidence", nullable=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--qwen-prompt", type=Path, default=DEFAULT_QWEN_PROMPT)
    parser.add_argument("--deepseek-prompt", type=Path, default=DEFAULT_DEEPSEEK_PROMPT)
    parser.add_argument("--qwen-api-file", type=Path, default=DEFAULT_QWEN_API)
    parser.add_argument("--deepseek-api-file", type=Path, default=DEFAULT_DEEPSEEK_API)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-max-tokens", type=int, default=8000)
    parser.add_argument("--deepseek-max-tokens", type=int, default=16000)
    parser.add_argument("--deepseek-attempts", type=int, default=2)
    parser.add_argument(
        "--omni-input",
        type=Path,
        default=None,
        help="reuse a saved Omni JSON and skip the Qwen request",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        parser.error("audio file does not exist: %s" % audio_path)
    qwen_system = args.qwen_prompt.resolve().read_text(encoding="utf-8").strip()
    deepseek_system = args.deepseek_prompt.resolve().read_text(encoding="utf-8").strip()
    qwen_user = (
        "Listen to the attached audio according to the system message and return only the required "
        "full-schema JSON.\nThis is an audio-only candidate annotation: every value must come from "
        "the audio or metadata explicitly exposed by the audio part; use null when uncertain."
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    omni_raw_path = output_dir / (audio_path.stem + ".omni.raw.txt")
    omni_path = output_dir / (audio_path.stem + ".omni.json")
    final_path = output_dir / (audio_path.stem + ".json")
    audit_path = output_dir / (audio_path.stem + ".audit.json")

    audio_bytes = audio_path.read_bytes()
    if args.omni_input is None:
        qwen_key = qwen_api.read_api_key(args.qwen_api_file.resolve())
        raw_text, qwen_usage = qwen_stream(
            args.qwen_base_url,
            qwen_key,
            args.qwen_model,
            qwen_messages(qwen_system, qwen_user, audio_bytes),
            args.qwen_max_tokens,
            args.timeout,
        )
        omni_raw_path.write_text(raw_text + "\n", encoding="utf-8")
        omni = qwen_api.parse_json_object(raw_text)
        omni_path.write_text(json.dumps(omni, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        raw_text = args.omni_input.resolve().read_text(encoding="utf-8").strip()
        omni = qwen_api.parse_json_object(raw_text)
        qwen_usage = None

    deepseek_config = deepseek_api.read_api_file(args.deepseek_api_file.resolve())
    deepseek_attempts = []
    deepseek_input = json.dumps(omni, ensure_ascii=False, separators=(",", ":"))
    final = None
    for attempt in range(1, max(1, args.deepseek_attempts) + 1):
        attempt_prompt = deepseek_system
        if attempt > 1:
            attempt_prompt += (
                "\n\n这是校验失败后的修复重试。用户消息包含 omni_candidate、previous_repair "
                "和 validation_error。请以 omni_candidate 为证据，在 previous_repair 上只修复"
                "校验错误及其连带冲突，仍只输出完整 schema JSON。"
            )
        candidate, usage = deepseek_api.call_api(
            deepseek_config["api_key"],
            deepseek_api.endpoint(deepseek_config["base_url"]),
            deepseek_config["model"],
            attempt_prompt,
            deepseek_input,
            args.timeout,
            args.deepseek_max_tokens,
            "enabled",
        )
        attempt_path = output_dir / (audio_path.stem + ".deepseek.attempt_%02d.json" % attempt)
        attempt_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        attempt_record = {"attempt": attempt, "output": str(attempt_path), "usage": usage}
        try:
            validate_full_schema(candidate)
        except ValidationError as error:
            attempt_record["validation_error"] = str(error)
            deepseek_attempts.append(attempt_record)
            if attempt >= max(1, args.deepseek_attempts):
                raise
            deepseek_input = json.dumps(
                {
                    "omni_candidate": omni,
                    "previous_repair": candidate,
                    "validation_error": str(error),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            continue
        deepseek_attempts.append(attempt_record)
        final = candidate
        break
    if final is None:
        raise ValidationError("DeepSeek did not produce a valid final result")

    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "sample_id": audio_path.stem,
        "audio_path": str(audio_path),
        "audio_sha256": sha256_bytes(audio_bytes),
        "qwen_prompt_path": str(args.qwen_prompt.resolve()),
        "deepseek_prompt_path": str(args.deepseek_prompt.resolve()),
        "qwen": {
            "model": args.qwen_model,
            "base_url": args.qwen_base_url,
            "system_prompt_sha256": sha256_bytes(qwen_system.encode("utf-8")),
            "user_prompt_sha256": sha256_bytes(qwen_user.encode("utf-8")),
            "usage": qwen_usage,
            "raw_text": raw_text,
            "parsed": omni,
        },
        "deepseek": {
            "model": deepseek_config["model"],
            "base_url": deepseek_config["base_url"],
            "system_prompt_sha256": sha256_bytes(deepseek_system.encode("utf-8")),
            "attempts": deepseek_attempts,
            "final": final,
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "succeeded",
        "sample_id": audio_path.stem,
        "qwen_model": args.qwen_model,
        "deepseek_model": deepseek_config["model"],
        "omni_output": str(omni_path),
        "final_output": str(final_path),
        "audit_output": str(audit_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ValidationError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        raise SystemExit(1)
