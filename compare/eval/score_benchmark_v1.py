#!/usr/bin/env python3
"""Score the T1-T3 structural benchmark labels using benchmark-scoring-v1.0.

This scorer canonicalizes the three active benchmark tracks directly from their
frozen sidecars.  It accepts the finalized Sure-Tagger JSONL as well as the
Sure-Tagger + DeepSeek merged JSONL; the latter has the same structural values
and adds an unscored ``descriptive`` group.
"""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import unicodedata
import wave


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_PREDICTIONS = (
    COMPARE_ROOT
    / "evaluation"
    / "benchmark-v1"
    / "sure-tagger-deepseek-full"
    / "results.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_PREDICTIONS.parent / "score.json"
DEFAULT_REPORT = DEFAULT_PREDICTIONS.parent / "score.md"

SCORING_VERSION = "benchmark-scoring-v1.0"
SERIALIZATION_VERSION = "numeric-round-half-even-v1"
BENCHMARK_VERSION = "structural-benchmark-subset-v1.0"
FIELD_UNIVERSE_VERSION = "benchmark-v1.1-inline"
NORMALIZATION_VERSION = "text-normalization-v1"
LEXICAL_RULE_VERSION = "text-lexical-rules-v1"
EVENT_ENUM_VERSION = "event-noise-enum-v1"
TEXT_EDIT_VERSION = "text-edit-v1"
NORMALIZED_ABSOLUTE_ERROR_VERSION = "symmetric-normalized-absolute-error-v1"
LAYER_AGGREGATION_VERSION = "label-first-layer-macro-v1"
SILENCE_RATIO_GT_VERSION = "ami-turn-complement-v1"
SPEAKER_GENDER_GT_VERSION = "ami-speaker-gender-gt-v1"
GENDER_EDIT_VERSION = "speaker-gender-edit-v1"
SPEAKER_GENDER_FIELD = "speaker.gender"

PREDICTION_PATH_ALIASES = {
    "basic_acoustic.nonspeech_ratio": ("basic_acoustic.silence_ratio",),
}

RESULT_STATES = (
    "result_known",
    "result_null",
    "result_missing",
    "schema_invalid",
    "runtime_failed",
)

CAPABILITY_FIELDS = (
    "basic_acoustic.nonspeech_ratio",
    "audio_quality.snr_db",
    "room_acoustic.rt60_sec",
    "room_acoustic.c50_db",
    "sound_field_scene.speech_music_events",
    "sound_field_scene.music_present",
    "sound_field_scene.external_noise_type",
    "speaker.speaker_count",
    "speaker.speaker_present",
    "speaker.multi_speaker",
    "speaker.speaker_change_count",
    "speaker.speaker_change",
    "speaker.overlap_ratio",
    "speaker.speaker_overlap",
    SPEAKER_GENDER_FIELD,
    "speaker.asr_transcript",
    "language_content.language",
    "language_content.word_count",
    "language_content.repetition.has_repetition",
    "language_content.repetition.repetition_count",
    "deterministic_derived.speaker_change_rate_per_min",
    "deterministic_derived.filler_ratio",
)

CONTRACT_FIELDS = (
    "basic_acoustic.duration_sec",
    "basic_acoustic.sample_rate_hz",
    "basic_acoustic.channels",
)

LAYER_DEFINITIONS = (
    (
        "layer_1",
        "Layer 1: Lexical Content",
        (
            "speaker.asr_transcript",
            "language_content.language",
            "language_content.word_count",
            "language_content.repetition.has_repetition",
            "language_content.repetition.repetition_count",
            "deterministic_derived.filler_ratio",
        ),
    ),
    (
        "layer_2",
        "Layer 2: Speaker",
        (
            "speaker.speaker_count",
            "speaker.speaker_present",
            "speaker.multi_speaker",
            "speaker.speaker_overlap",
            SPEAKER_GENDER_FIELD,
        ),
    ),
    (
        "layer_3",
        "Layer 3: Temporal",
        (
            "speaker.speaker_change_count",
            "speaker.speaker_change",
            "speaker.overlap_ratio",
            "deterministic_derived.speaker_change_rate_per_min",
        ),
    ),
    ("layer_4", "Layer 4: Paralinguistic & Non-verbal", ()),
    (
        "layer_5",
        "Layer 5: Acoustic Scene & Events",
        (
            "basic_acoustic.nonspeech_ratio",
            "audio_quality.snr_db",
            "room_acoustic.rt60_sec",
            "room_acoustic.c50_db",
            "sound_field_scene.speech_music_events",
            "sound_field_scene.music_present",
            "sound_field_scene.external_noise_type",
        ),
    ),
)

MULTILABEL_CLASSES = {
    "sound_field_scene.speech_music_events": ("speech", "music", "singing"),
    "sound_field_scene.external_noise_type": (
        "music",
        "animal",
        "mechanical",
        "nature",
        "formless",
        "channel_environment",
    ),
}

FIELD_SPECS = {
    "basic_acoustic.duration_sec": {"kind": "number_nonnegative", "metric": "tolerance", "tolerance": 0.001, "role": "contract"},
    "basic_acoustic.sample_rate_hz": {"kind": "integer_positive", "metric": "integer", "role": "contract"},
    "basic_acoustic.channels": {"kind": "integer_positive", "metric": "integer", "role": "contract"},
    "basic_acoustic.nonspeech_ratio": {
        "kind": "ratio",
        "metric": "absolute_ratio",
        "role": "capability",
        "reference_name": "silence_ratio",
        "gt_rule_version": SILENCE_RATIO_GT_VERSION,
    },
    "audio_quality.snr_db": {"kind": "number", "metric": "continuous", "tolerance": 5.0, "role": "capability"},
    "room_acoustic.rt60_sec": {"kind": "number_nonnegative", "metric": "continuous", "tolerance": 0.5, "role": "capability"},
    "room_acoustic.c50_db": {"kind": "number", "metric": "continuous", "tolerance": 3.0, "role": "capability"},
    "sound_field_scene.speech_music_events": {"kind": "event_array", "metric": "multilabel", "role": "capability"},
    "sound_field_scene.music_present": {"kind": "boolean", "metric": "boolean", "role": "capability"},
    "sound_field_scene.external_noise_type": {"kind": "noise_array", "metric": "multilabel", "role": "capability"},
    "speaker.speaker_count": {"kind": "integer_nonnegative", "metric": "integer", "role": "capability"},
    "speaker.speaker_present": {"kind": "boolean", "metric": "boolean", "role": "capability"},
    "speaker.multi_speaker": {"kind": "boolean", "metric": "boolean", "role": "capability"},
    "speaker.speaker_change_count": {"kind": "integer_nonnegative", "metric": "normalized_absolute", "role": "capability"},
    "speaker.speaker_change": {"kind": "boolean", "metric": "boolean", "role": "capability"},
    "speaker.overlap_ratio": {"kind": "ratio", "metric": "absolute_ratio", "role": "capability"},
    "speaker.speaker_overlap": {"kind": "boolean", "metric": "boolean", "role": "capability"},
    SPEAKER_GENDER_FIELD: {
        "kind": "gender_sequence",
        "metric": "gender_sequence",
        "role": "capability",
        "gt_rule_version": SPEAKER_GENDER_GT_VERSION,
        "edit_version": GENDER_EDIT_VERSION,
    },
    "speaker.asr_transcript": {"kind": "string", "metric": "text", "role": "capability"},
    "language_content.language": {"kind": "language", "metric": "language", "role": "capability"},
    "language_content.word_count": {"kind": "integer_nonnegative", "metric": "normalized_absolute", "role": "capability", "lexical_grouped": True},
    "language_content.punctuation.punctuation_count": {"kind": "integer_nonnegative", "metric": "normalized_absolute", "role": "diagnostic", "lexical_grouped": True},
    "language_content.punctuation.has_terminal_punctuation": {"kind": "boolean", "metric": "boolean", "role": "diagnostic", "lexical_grouped": True},
    "language_content.repetition.has_repetition": {"kind": "boolean", "metric": "boolean", "role": "capability", "lexical_grouped": True},
    "language_content.repetition.repetition_count": {"kind": "integer_nonnegative", "metric": "integer", "role": "capability", "lexical_grouped": True},
    "language_content.filler": {"kind": "integer_nonnegative", "metric": "integer", "role": "diagnostic", "lexical_grouped": True},
    "deterministic_derived.speaker_change_rate_per_min": {"kind": "number_nonnegative", "metric": "normalized_absolute", "role": "capability"},
    "deterministic_derived.filler_ratio": {"kind": "ratio", "metric": "continuous", "tolerance": 0.05, "role": "capability", "lexical_grouped": True},
}

LANGUAGE_ALIASES = {
    "ami": "en",
    "librispeech": "en",
    "en": "en",
    "en_us": "en",
    "ar_eg": "ar",
    "cmn_hans_cn": "zh",
    "zh_cn": "zh",
    "es_419": "es",
    "fr_fr": "fr",
    "hi_in": "hi",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "ru_ru": "ru",
    "sw_ke": "sw",
}
LANGUAGES = frozenset(("ar", "zh", "en", "es", "fr", "hi", "ja", "ko", "ru", "sw"))
CJK_LANGUAGES = frozenset(("zh", "ja", "ko"))
ENGLISH_FILLERS = frozenset(("ah", "er", "erm", "hmm", "mm", "oh", "uh", "um", "yeah"))
TERMINAL_PUNCTUATION = frozenset((".", "?", "!", "\u3002", "\u061f", "\u0964", "\u0965"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError("blank JSONL line at %s:%d" % (path, line_number))
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, exc))
            if not isinstance(row, dict):
                raise ValueError("%s:%d is not an object" % (path, line_number))
            rows.append(row)
    return rows


def keyed(rows, id_getter, label):
    result = {}
    for index, row in enumerate(rows, 1):
        sample_id = id_getter(row)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("%s row %d has no sample_id" % (label, index))
        if sample_id in result:
            raise ValueError("%s has duplicate sample_id %s" % (label, sample_id))
        result[sample_id] = row
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def q6(value):
    if value is None:
        return None
    quantized = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    number = float(quantized)
    return 0.0 if number == 0 else number


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def mean_defined(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def unquantized_rate(result):
    pairs = (
        ("score_sum_raw", "max_score_raw"),
        ("score_sum_bal_raw", "max_score_bal_raw"),
        ("score_sum_bal_field_raw", "max_score_bal_field_raw"),
        ("score_sum_macro_raw", "max_score_macro_raw"),
        ("score_sum_macro_asr_raw", "max_score_macro_asr_raw"),
    )
    for numerator_key, denominator_key in pairs:
        if numerator_key in result and denominator_key in result:
            return safe_divide(result[numerator_key], result[denominator_key])
    return None


def get_path(value, path):
    current = value
    for part in path.split("."):
        if current is None:
            # A null parent object is an explicit null for every enabled leaf.
            return True, None
        if not isinstance(current, dict):
            return None, None
        if part not in current:
            return False, None
        current = current[part]
    return True, current


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def canonical_language(value):
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("-", "_")
    canonical = LANGUAGE_ALIASES.get(normalized, normalized)
    return canonical if canonical in LANGUAGES else None


def canonical_array(value, allowed):
    if not isinstance(value, list):
        return None
    normalized = []
    for item in value:
        if not isinstance(item, str):
            return None
        item = item.strip(" \t\r\n\f\v").casefold()
        if item not in allowed or item in normalized:
            return None
        normalized.append(item)
    return frozenset(normalized)


def canonical_gender(value):
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in {"male", "female"} else None


def speaker_id_number(value):
    if not isinstance(value, str) or not value.startswith("speaker_"):
        return None
    suffix = value[len("speaker_"):]
    if not suffix.isdigit() or not suffix.startswith(tuple("123456789")):
        return None
    return int(suffix)


def _gender_sequence_entries(raw_entries, require_gender=True):
    """Canonicalize speaker gender entries in numeric speaker-ID order."""
    entries = []
    seen_ids = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            return "schema_invalid", None
        speaker_id = item.get("speaker_id")
        number = speaker_id_number(speaker_id)
        if number is None or speaker_id in seen_ids:
            return "schema_invalid", None
        seen_ids.add(speaker_id)
        if require_gender and "gender" not in item:
            return "result_missing", None
        gender = item.get("gender")
        if gender is None:
            return "result_null", None
        gender = canonical_gender(gender)
        if gender is None:
            return "schema_invalid", None
        entries.append((number, gender))
    entries.sort(key=lambda item: item[0])
    return "result_known", tuple(gender for _, gender in entries)


def gender_sequence_state(prediction):
    """Read genders from the canonical field or a profile compatibility field.

    The canonical prediction surface is ``descriptive.speaker_descriptions``.
    Some finalized Sure-Tagger runs stored the same labels in
    ``speaker.profiles[].gender`` before the descriptive merge.  When that
    explicit surface exists it remains authoritative after a descriptive merge,
    whose model intentionally never receives gender as input.
    """
    if prediction is None:
        return "runtime_failed", None

    profile_present, profiles = get_path(prediction, "speaker.profiles")
    if profile_present is None:
        return "schema_invalid", None
    if isinstance(profiles, list) and all(
        isinstance(profile, dict) and "gender" in profile for profile in profiles
    ):
        return _gender_sequence_entries(profiles)

    present, descriptions = get_path(prediction, "descriptive.speaker_descriptions")
    if present is None:
        return "schema_invalid", None
    if present is not False:
        if descriptions is None:
            return "result_null", None
        if not isinstance(descriptions, list):
            return "schema_invalid", None
        return _gender_sequence_entries(descriptions)

    # Legacy predictions may have neither descriptive labels nor profile
    # gender. Preserve the ordinary missing/null states for those rows.
    if profile_present is False:
        return "result_missing", None
    if profiles is None:
        return "result_null", None
    if not isinstance(profiles, list):
        return "schema_invalid", None
    return _gender_sequence_entries(profiles)


def canonical_prediction(value, spec):
    kind = spec["kind"]
    if kind == "number":
        return float(value) if finite_number(value) else None
    if kind == "number_nonnegative":
        return float(value) if finite_number(value) and float(value) >= 0 else None
    if kind == "ratio":
        return float(value) if finite_number(value) and 0 <= float(value) <= 1 else None
    if kind == "integer_nonnegative":
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    if kind == "integer_positive":
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
    if kind == "boolean":
        return value if isinstance(value, bool) else None
    if kind == "string":
        return value if isinstance(value, str) else None
    if kind == "language":
        return canonical_language(value)
    if kind == "event_array":
        return canonical_array(value, frozenset(MULTILABEL_CLASSES["sound_field_scene.speech_music_events"]))
    if kind == "noise_array":
        return canonical_array(value, frozenset(MULTILABEL_CLASSES["sound_field_scene.external_noise_type"]))
    raise ValueError("unknown kind %s" % kind)


def prediction_state(prediction, path, spec):
    if path == SPEAKER_GENDER_FIELD:
        return gender_sequence_state(prediction)
    if prediction is None:
        return "runtime_failed", None
    present, value = get_path(prediction, path)
    if present is False:
        for alias in PREDICTION_PATH_ALIASES.get(path, ()):
            alias_present, alias_value = get_path(prediction, alias)
            if alias_present is not False:
                present, value = alias_present, alias_value
                break
    if present is None:
        return "schema_invalid", None
    if present is False:
        return "result_missing", None
    if value is None:
        return "result_null", None
    canonical = canonical_prediction(value, spec)
    if canonical is None:
        return "schema_invalid", None
    return "result_known", canonical


def wav_header(path):
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError("compressed WAV is unsupported: %s" % path)
        frames = source.getnframes()
        sample_rate = source.getframerate()
        channels = source.getnchannels()
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("invalid WAV header: %s" % path)
    return {
        "basic_acoustic.duration_sec": frames / float(sample_rate),
        "basic_acoustic.sample_rate_hz": sample_rate,
        "basic_acoustic.channels": channels,
    }


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def text_units(text, language):
    normalized = normalize_text(text)
    if language in CJK_LANGUAGES:
        return [char for char in normalized if not char.isspace()]
    return normalized.split() if normalized else []


def repetition_count(tokens):
    count = 0
    for size in (3, 2, 1):
        index = 0
        while index + 2 * size <= len(tokens):
            if tokens[index:index + size] == tokens[index + size:index + 2 * size]:
                count += 1
                index += 2 * size
            else:
                index += 1
    return count


def lexical_values(text, language):
    normalized_raw = unicodedata.normalize("NFKC", text)
    tokens = text_units(text, language)
    repetitions = repetition_count(tokens)
    output = {
        "language_content.word_count": len(tokens),
        "language_content.punctuation.punctuation_count": sum(
            unicodedata.category(char).startswith("P") for char in normalized_raw
        ),
        "language_content.punctuation.has_terminal_punctuation": bool(
            normalized_raw.rstrip() and normalized_raw.rstrip()[-1] in TERMINAL_PUNCTUATION
        ),
        "language_content.repetition.has_repetition": repetitions > 0,
        "language_content.repetition.repetition_count": repetitions,
    }
    if language == "en":
        filler_count = sum(token in ENGLISH_FILLERS for token in tokens)
        output["language_content.filler"] = filler_count
        output["deterministic_derived.filler_ratio"] = (
            q6(filler_count / len(tokens)) if tokens else None
        )
    return output


def merge_intervals_by_speaker(utterances, duration):
    grouped = defaultdict(list)
    for item in utterances:
        if not isinstance(item, dict) or not isinstance(item.get("speaker"), str):
            raise ValueError("invalid AMI utterance")
        start = max(0.0, float(item["start"]))
        end = min(duration, float(item["end"]))
        if end <= start or end - start < 0.1:
            continue
        grouped[item["speaker"].strip().lower()].append((start, end))
    merged = {}
    for speaker, intervals in grouped.items():
        result = []
        for start, end in sorted(intervals):
            if not result or start - result[-1][1] > 0.3:
                result.append([start, end])
            else:
                result[-1][1] = max(result[-1][1], end)
        if sum(end - start for start, end in result) >= 0.1:
            merged[speaker] = result
    return merged


def union_duration(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def overlap_duration(merged_by_speaker, collar):
    events = defaultdict(list)
    for speaker, intervals in merged_by_speaker.items():
        for start, end in intervals:
            events[start].append((speaker, True))
            events[end].append((speaker, False))
    active = set()
    previous = None
    overlaps = []
    for moment in sorted(events):
        if previous is not None and moment > previous and len(active) >= 2:
            if moment - previous >= collar:
                overlaps.append((previous, moment))
        for speaker, starts in events[moment]:
            if not starts:
                active.discard(speaker)
        for speaker, starts in events[moment]:
            if starts:
                active.add(speaker)
        previous = moment
    return sum(end - start for start, end in overlaps)


def ami_speaker_values(utterances, duration):
    merged = merge_intervals_by_speaker(utterances, duration)
    intervals = [(start, end) for values in merged.values() for start, end in values]
    speech_union = union_duration(intervals)
    overlap = overlap_duration(merged, 0.1)
    ordered = sorted(
        (start, end, speaker)
        for speaker, values in merged.items()
        for start, end in values
    )
    changes = []
    previous = None
    for start, end, speaker in ordered:
        if previous is not None and speaker != previous[2] and start - previous[1] <= 1.0:
            if not changes or abs(start - changes[-1]) > 1e-6:
                changes.append(start)
        previous = (start, end, speaker)
    speaker_count = len(merged)
    change_count = len(changes)
    return {
        "basic_acoustic.nonspeech_ratio": q6(
            max(0.0, min(1.0, 1.0 - speech_union / duration))
        ),
        "speaker.speaker_count": speaker_count,
        "speaker.speaker_present": speaker_count > 0,
        "speaker.multi_speaker": speaker_count > 1,
        "speaker.speaker_change_count": change_count,
        "speaker.speaker_change": change_count > 0,
        "speaker.overlap_ratio": q6(overlap / speech_union) if speech_union > 0 else None,
        "speaker.speaker_overlap": overlap >= 0.1,
        "deterministic_derived.speaker_change_rate_per_min": q6(change_count * 60.0 / duration),
    }


def empty_gt(sample_id, track, evaluation_group, language=None, source_group=None):
    return {
        "sample_id": sample_id,
        "track": track,
        "evaluation_group": evaluation_group,
        "source_group": source_group,
        "language": language,
        "values": {},
        "gt_mask": set(),
        "label_mask_by_field": {},
    }


def add_header(record, header):
    record["values"].update(header)
    record["gt_mask"].update(CONTRACT_FIELDS)


def load_speaker_gender_gt(benchmark_root, expected_sample_ids):
    path = Path(benchmark_root) / "t1-ami" / "speaker_gender_gt" / "speaker_gender_gt.jsonl"
    if not path.is_file():
        raise ValueError("speaker gender GT is missing: %s" % path)
    rows = keyed(load_jsonl(path), lambda row: row.get("sample_id"), "speaker gender GT")
    expected_sample_ids = set(expected_sample_ids)
    if set(rows) != expected_sample_ids:
        raise ValueError(
            "speaker gender GT/sample ID mismatch: missing=%d extra=%d" % (
                len(expected_sample_ids - set(rows)),
                len(set(rows) - expected_sample_ids),
            )
        )

    sequences = {}
    for sample_id, row in rows.items():
        speakers = row.get("speakers")
        if not isinstance(speakers, list):
            raise ValueError("speaker gender GT speakers must be an array: %s" % sample_id)
        sequence = []
        expected_number = 1
        for speaker in speakers:
            if not isinstance(speaker, dict):
                raise ValueError("invalid speaker gender GT item: %s" % sample_id)
            speaker_id = speaker.get("speaker_id")
            if speaker_id_number(speaker_id) != expected_number:
                raise ValueError(
                    "speaker gender GT IDs must be speaker_1..N in first-appearance order: %s"
                    % sample_id
                )
            gender = canonical_gender(speaker.get("gender"))
            if gender is None:
                raise ValueError("invalid speaker gender GT value: %s" % sample_id)
            sequence.append(gender)
            expected_number += 1
        if not sequence:
            raise ValueError("speaker gender GT cannot be empty: %s" % sample_id)
        sequences[sample_id] = tuple(sequence)
    return sequences


def build_ground_truth(benchmark_root):
    benchmark_root = Path(benchmark_root).resolve()
    manifest_rows = load_jsonl(benchmark_root / "benchmark.manifest.jsonl")
    manifests = keyed(manifest_rows, lambda row: row.get("sample", {}).get("sample_id"), "benchmark manifest")
    headers = {}
    for sample_id, row in manifests.items():
        audio_path = benchmark_root / row["sample"]["audio"]["path"]
        headers[sample_id] = wav_header(audio_path)

    records = {}
    t1_rows = load_jsonl(benchmark_root / "t1-ami" / "manifest.jsonl")
    t1_sample_ids = [row.get("sample", {}).get("sample_id") for row in t1_rows]
    gender_sequences = load_speaker_gender_gt(benchmark_root, t1_sample_ids)
    for row in t1_rows:
        sample = row["sample"]
        sample_id = sample["sample_id"]
        record = empty_gt(sample_id, "T1_natural_meeting_utterance", "T1_natural_meeting_utterance", "en")
        add_header(record, headers[sample_id])
        transcript = sample["text"]["transcript"]
        duration = headers[sample_id]["basic_acoustic.duration_sec"]
        record["values"].update(ami_speaker_values(sample["native_metadata"]["utterances"], duration))
        record["values"][SPEAKER_GENDER_FIELD] = gender_sequences[sample_id]
        record["values"]["speaker.asr_transcript"] = transcript
        record["values"]["language_content.language"] = "en"
        record["values"].update(lexical_values(transcript, "en"))
        record["gt_mask"].update(
            path for path in CAPABILITY_FIELDS
            if path.startswith("speaker.") or path.startswith("language_content.")
            or path.startswith("deterministic_derived.")
        )
        record["gt_mask"].add("basic_acoustic.nonspeech_ratio")
        record["gt_mask"].add(SPEAKER_GENDER_FIELD)
        records[sample_id] = record

    t2_rows = load_jsonl(benchmark_root / "t2-language-asr" / "metadata.jsonl")
    for row in t2_rows:
        sample_id = row["sample_id"]
        track = row["track"]
        if track == "T2_multilingual_fleurs":
            language = canonical_language(row["language"])
            transcript = row["reference_text_raw"]
        elif track == "T2_clean_librispeech":
            language = canonical_language(row["ground_truth"]["language"])
            transcript = row["ground_truth"]["text"]
        else:
            raise ValueError("unexpected T2 track %s" % track)
        record = empty_gt(sample_id, track, track, language)
        add_header(record, headers[sample_id])
        record["values"].update({
            "speaker.asr_transcript": transcript,
            "language_content.language": language,
        })
        record["values"].update(lexical_values(transcript, language))
        record["gt_mask"].update((
            "speaker.asr_transcript",
            "language_content.language",
            "language_content.word_count",
            "language_content.punctuation.punctuation_count",
            "language_content.punctuation.has_terminal_punctuation",
            "language_content.repetition.has_repetition",
            "language_content.repetition.repetition_count",
        ))
        if language == "en":
            record["gt_mask"].add("language_content.filler")
            if record["values"].get("deterministic_derived.filler_ratio") is not None:
                record["gt_mask"].add("deterministic_derived.filler_ratio")
        records[sample_id] = record

    t3_rows = load_jsonl(benchmark_root / "t3-event-noise" / "closed_world_gt.jsonl")
    for row in t3_rows:
        sample_id = row["sample_id"]
        source = row["source_dataset"]
        track = "T3_events_noise"
        record = empty_gt(sample_id, track, "%s@%s" % (track, source), source_group=source)
        add_header(record, headers[sample_id])
        for path, enabled in row["gt_mask"].items():
            if enabled:
                record["values"][path] = row["ground_truth"][path]
                record["gt_mask"].add(path)
                if path in MULTILABEL_CLASSES:
                    record["label_mask_by_field"][path] = {
                        label: True for label in MULTILABEL_CLASSES[path]
                    }
                    expected_state = {
                        label: label in set(row["ground_truth"][path])
                        for label in MULTILABEL_CLASSES[path]
                    }
                    if row["label_state_by_field"][path] != expected_state:
                        raise ValueError("T3 label-state mismatch for %s %s" % (sample_id, path))
        records[sample_id] = record

    if set(records) != set(manifests):
        raise ValueError("GT/manifest ID mismatch: missing=%d extra=%d" % (
            len(set(manifests) - set(records)), len(set(records) - set(manifests))
        ))
    return records


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for ref_token in reference:
        current = [previous[0] + 1]
        for index, hyp_token in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[index] + 1,
                previous[index - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def gender_sequence_alignment(reference, hypothesis):
    """Return WER-style edit distance diagnostics for gender sequences."""
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        distances[row][0] = row
    for column in range(1, columns):
        distances[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
            )

    substitutions = deletions = insertions = matches = 0
    row, column = len(reference), len(hypothesis)
    while row or column:
        if row and column and reference[row - 1] == hypothesis[column - 1] \
                and distances[row][column] == distances[row - 1][column - 1]:
            matches += 1
            row -= 1
            column -= 1
            continue
        candidates = []
        if row and column:
            candidates.append((
                distances[row - 1][column - 1],
                0,
                "substitution",
            ))
        if row:
            candidates.append((distances[row - 1][column], 1, "deletion"))
        if column:
            candidates.append((distances[row][column - 1], 2, "insertion"))
        best_distance, _, operation = min(candidates)
        if best_distance + 1 != distances[row][column]:
            raise AssertionError("invalid gender sequence alignment")
        if operation == "substitution":
            substitutions += 1
            row -= 1
            column -= 1
        elif operation == "deletion":
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return {
        "distance": distances[-1][-1],
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def gender_sequence_result(items, path):
    output = base_result(items, path)
    known = [item for item in items if item["state"] == "result_known"]
    scores = []
    distance_total = 0
    reference_units = 0
    hypothesis_units = 0
    exact_known = 0
    operation_totals = Counter()
    alignments = []
    for item in items:
        reference = tuple(item["reference"])
        if item["state"] == "result_known":
            hypothesis = tuple(item["predicted"])
            alignment = gender_sequence_alignment(reference, hypothesis)
            distance = alignment["distance"]
            denominator = len(reference)
            score = (
                2.0 * max(0.0, 1.0 - distance / float(denominator))
                if denominator
                else (2.0 if not hypothesis else 0.0)
            )
            distance_total += distance
            reference_units += denominator
            hypothesis_units += len(hypothesis)
            exact_known += reference == hypothesis
            operation_totals.update({
                key: value for key, value in alignment.items()
                if key != "distance"
            })
            alignments.append(alignment)
        elif item["state"] == "result_null":
            # Keep the scorer's established partial credit for an explicit
            # abstention; missing and malformed fields remain zero.
            score = 1.0
        else:
            score = 0.0
        scores.append(score)
    score_sum = sum(scores)
    output.update({
        "score_aggregation": "sample_micro",
        "score_sum_raw": score_sum,
        "max_score_raw": 2.0 * len(items),
        "score_sum": q6(score_sum),
        "max_score": q6(2.0 * len(items)),
        "score_rate": q6(safe_divide(score_sum, 2.0 * len(items))),
        "gender_edit_version": GENDER_EDIT_VERSION,
        "score_formula": "2*max(0,1-edit_distance(reference,hypothesis)/reference_speaker_count)",
        "auxiliary": {
            "exact_sequence_accuracy_known": q6(safe_divide(exact_known, len(known))),
            "edit_distance_known": distance_total,
            "reference_speaker_count_known": reference_units,
            "predicted_speaker_count_known": hypothesis_units,
            "substitutions_known": operation_totals["substitutions"],
            "deletions_known": operation_totals["deletions"],
            "insertions_known": operation_totals["insertions"],
            "matches_known": operation_totals["matches"],
            "edit_error_rate_known": q6(safe_divide(distance_total, reference_units)),
            "alignments_known": alignments,
        },
    })
    return output


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for offset in range(index, end):
            ranks[indexed[offset][0]] = rank
        index = end
    return ranks


def pearson(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def unit_score(metric, state, predicted, reference, spec, language=None):
    if state == "result_null":
        return 1.0
    if state != "result_known":
        return 0.0
    if metric in ("integer", "boolean", "language"):
        return 2.0 if predicted == reference else 0.0
    if metric == "normalized_absolute":
        error = abs(predicted - reference) / max(predicted, reference, 1)
        return 2.0 * (1.0 - error)
    if metric == "tolerance":
        return 2.0 if abs(float(predicted) - float(reference)) <= spec["tolerance"] else 0.0
    if metric == "continuous":
        error = abs(float(predicted) - float(reference))
        return 2.0 * max(0.0, 1.0 - error / spec["tolerance"])
    if metric == "absolute_ratio":
        return 2.0 * (1.0 - abs(float(predicted) - float(reference)))
    if metric == "text":
        ref_units = text_units(reference, language)
        hyp_units = text_units(predicted, language)
        error = min(edit_distance(ref_units, hyp_units) / max(len(ref_units), 1), 1.0)
        return 2.0 * (1.0 - error)
    raise ValueError("unsupported unit metric %s" % metric)


def state_items(gt_rows, predictions, path):
    spec = FIELD_SPECS[path]
    items = []
    for ground_truth in gt_rows:
        state, predicted = prediction_state(predictions.get(ground_truth["sample_id"]), path, spec)
        items.append({
            "gt": ground_truth,
            "state": state,
            "predicted": predicted,
            "reference": ground_truth["values"][path],
        })
    return items


def base_result(items, path):
    counts = Counter(item["state"] for item in items)
    output = {
        "field": path,
        "metric_role": FIELD_SPECS[path]["role"],
        "input_sample_count": len(items),
        "eligible_samples": len(items),
        "eligible_units": len(items),
        "gt_excluded_samples": 0,
        "leakage_excluded_units": 0,
        "state_counts_scope": "eligible_samples",
    }
    spec = FIELD_SPECS[path]
    if spec.get("reference_name"):
        output["reference_name"] = spec["reference_name"]
    if spec.get("gt_rule_version"):
        output["gt_rule_version"] = spec["gt_rule_version"]
    for state in RESULT_STATES:
        output[state] = counts[state]
    output["known_rate"] = q6(safe_divide(counts["result_known"], len(items)))
    return output


def raw_sample_result(items, path):
    spec = FIELD_SPECS[path]
    output = base_result(items, path)
    scores = [unit_score(spec["metric"], item["state"], item["predicted"], item["reference"], spec, item["gt"].get("language")) for item in items]
    score_sum = sum(scores)
    output.update({
        "score_aggregation": "sample_micro",
        "score_sum_raw": score_sum,
        "max_score_raw": 2.0 * len(items),
        "score_sum": q6(score_sum),
        "max_score": q6(2.0 * len(items)),
        "score_rate": q6(safe_divide(score_sum, 2.0 * len(items))),
    })
    known = [item for item in items if item["state"] == "result_known"]
    metric = spec["metric"]
    auxiliary = {}
    if metric in ("integer", "normalized_absolute", "language"):
        correct = sum(item["predicted"] == item["reference"] for item in known)
        auxiliary.update({
            "exact_accuracy_known": q6(safe_divide(correct, len(known))),
            "exact_accuracy_e2e": q6(safe_divide(correct, len(items))),
        })
        if metric in ("integer", "normalized_absolute"):
            errors = [abs(item["predicted"] - item["reference"]) for item in known]
            auxiliary["mae_known"] = q6(mean_defined(errors))
        if metric == "normalized_absolute":
            normalized_errors = [
                abs(item["predicted"] - item["reference"])
                / max(item["predicted"], item["reference"], 1)
                for item in known
            ]
            auxiliary.update({
                "median_absolute_error_known": q6(
                    statistics.median(errors) if errors else None
                ),
                "mean_normalized_absolute_error_known": q6(
                    mean_defined(normalized_errors)
                ),
                "spearman_known": q6(pearson(
                    average_ranks([float(item["predicted"]) for item in known]),
                    average_ranks([float(item["reference"]) for item in known]),
                )),
            })
    elif metric in ("absolute_ratio", "continuous", "tolerance"):
        errors = [abs(float(item["predicted"]) - float(item["reference"])) for item in known]
        auxiliary.update({
            "mae_known": q6(mean_defined(errors)),
            "median_absolute_error_known": q6(statistics.median(errors) if errors else None),
            "spearman_known": q6(pearson(
                average_ranks([float(item["predicted"]) for item in known]),
                average_ranks([float(item["reference"]) for item in known]),
            )),
        })
        if metric == "tolerance":
            matched = sum(error <= spec["tolerance"] for error in errors)
            auxiliary.update({
                "within_tolerance_known": q6(safe_divide(matched, len(known))),
                "within_tolerance_e2e": q6(safe_divide(matched, len(items))),
            })
    output["auxiliary"] = auxiliary
    if "tolerance" in spec:
        output["tolerance"] = spec["tolerance"]
    if metric == "normalized_absolute":
        output["normalized_absolute_error_version"] = NORMALIZED_ABSOLUTE_ERROR_VERSION
        output["score_formula"] = (
            "2*(1-abs(predicted-reference)/max(predicted,reference,1))"
        )
    elif metric == "absolute_ratio":
        output["score_formula"] = "2*(1-abs(predicted-reference))"
    return output


def confusion_aux(items):
    known = [item for item in items if item["state"] == "result_known"]
    tp = sum(item["predicted"] is True and item["reference"] is True for item in known)
    fp = sum(item["predicted"] is True and item["reference"] is False for item in known)
    tn = sum(item["predicted"] is False and item["reference"] is False for item in known)
    fn_known = sum(item["predicted"] is False and item["reference"] is True for item in known)
    positives = sum(item["reference"] is True for item in items)
    negatives = len(items) - positives
    fn_e2e = positives - tp
    precision_known = safe_divide(tp, tp + fp)
    recall_known = safe_divide(tp, tp + fn_known)
    specificity_known = safe_divide(tn, tn + fp)
    precision_e2e = precision_known
    recall_e2e = safe_divide(tp, positives)
    specificity_e2e = safe_divide(tn, negatives)
    return {
        "true_positive_known": tp,
        "false_positive_known": fp,
        "true_negative_known": tn,
        "false_negative_known": fn_known,
        "false_negative_e2e": fn_e2e,
        "abstain_negative": negatives - fp - tn,
        "precision_known": q6(precision_known),
        "recall_known": q6(recall_known),
        "f1_known": q6(safe_divide(2 * tp, 2 * tp + fp + fn_known)),
        "specificity_known": q6(specificity_known),
        "balanced_accuracy_known": q6(mean_defined([recall_known, specificity_known]) if positives and negatives else None),
        "precision_e2e": q6(precision_e2e),
        "recall_e2e": q6(recall_e2e),
        "f1_e2e": q6(safe_divide(2 * tp, 2 * tp + fp + fn_e2e)),
        "specificity_e2e": q6(specificity_e2e),
        "balanced_accuracy_e2e": q6(mean_defined([recall_e2e, specificity_e2e]) if positives and negatives else None),
        "fpr_known": q6(safe_divide(fp, tn + fp)),
        "fpr_e2e": q6(safe_divide(fp, negatives)),
    }


def boolean_result(items, path):
    output = base_result(items, path)
    positives = [item for item in items if item["reference"] is True]
    negatives = [item for item in items if item["reference"] is False]
    spec = FIELD_SPECS[path]
    positive_sum = sum(unit_score("boolean", item["state"], item["predicted"], True, spec) for item in positives)
    negative_sum = sum(unit_score("boolean", item["state"], item["predicted"], False, spec) for item in negatives)
    positive_rate = safe_divide(positive_sum, 2.0 * len(positives))
    negative_rate = safe_divide(negative_sum, 2.0 * len(negatives))
    if positives and negatives:
        support_status = "both"
        rate = (positive_rate + negative_rate) / 2.0
    elif positives:
        support_status = "positive_only"
        rate = positive_rate
    elif negatives:
        support_status = "negative_only"
        rate = None
    else:
        support_status = "no_support"
        rate = None
    score_sum_raw = 2.0 * rate if rate is not None else None
    output.update({
        "score_aggregation": "class_balanced",
        "support_status": support_status,
        "positive_ground_truth": len(positives),
        "negative_ground_truth": len(negatives),
        "positive_score_sum_raw": positive_sum,
        "positive_max_score_raw": 2.0 * len(positives),
        "positive_rate": q6(positive_rate),
        "negative_score_sum_raw": negative_sum,
        "negative_max_score_raw": 2.0 * len(negatives),
        "negative_rate": q6(negative_rate),
        "score_sum_bal_raw": score_sum_raw,
        "max_score_bal_raw": 2.0 if rate is not None else None,
        "score_sum_bal": q6(score_sum_raw),
        "max_score_bal": 2.0 if rate is not None else None,
        "score_rate": q6(rate),
        "auxiliary": confusion_aux(items),
    })
    return output


def multilabel_result(gt_rows, predictions, path):
    spec = FIELD_SPECS[path]
    sample_items = state_items(gt_rows, predictions, path)
    output = base_result(sample_items, path)
    classes = {}
    macro_rates = []
    score_sum_bal = 0.0
    eligible_class_units = 0
    for label in MULTILABEL_CLASSES[path]:
        units = []
        for item in sample_items:
            mask = item["gt"]["label_mask_by_field"].get(path, {})
            if not mask.get(label, False):
                continue
            reference = label in set(item["reference"])
            predicted = label in item["predicted"] if item["state"] == "result_known" else None
            units.append({**item, "reference": reference, "predicted": predicted})
        eligible_class_units += len(units)
        class_result = boolean_result(units, path)
        class_result.pop("field", None)
        class_result["state_counts_scope"] = "eligible_class_units"
        classes[label] = class_result
        if class_result["score_rate"] is not None:
            rate = class_result["score_sum_bal_raw"] / class_result["max_score_bal_raw"]
            macro_rates.append(rate)
            score_sum_bal += 2.0 * rate
    output.update({
        "score_aggregation": "class_macro_balanced",
        "eligible_class_units": eligible_class_units,
        "gt_excluded_class_units": len(gt_rows) * len(MULTILABEL_CLASSES[path]) - eligible_class_units,
        "leakage_excluded_class_units": 0,
        "score_sum_bal_field_raw": score_sum_bal if macro_rates else None,
        "max_score_bal_field_raw": 2.0 * len(macro_rates) if macro_rates else None,
        "score_sum_bal_field": q6(score_sum_bal if macro_rates else None),
        "max_score_bal_field": q6(2.0 * len(macro_rates) if macro_rates else None),
        "score_rate": q6(mean_defined(macro_rates)),
        "classes": classes,
    })
    known_full = [item for item in sample_items if item["state"] == "result_known" and all(
        item["gt"]["label_mask_by_field"].get(path, {}).get(label, False)
        for label in MULTILABEL_CLASSES[path]
    )]
    eligible_full = [item for item in sample_items if all(
        item["gt"]["label_mask_by_field"].get(path, {}).get(label, False)
        for label in MULTILABEL_CLASSES[path]
    )]
    exact = sum(item["predicted"] == frozenset(item["reference"]) for item in known_full)
    output["auxiliary"] = {
        "exact_set_known": q6(safe_divide(exact, len(known_full))),
        "exact_set_e2e": q6(safe_divide(exact, len(eligible_full))),
    }
    return output


def grouped_result(items, path, group_key, group_kind):
    groups = defaultdict(list)
    for item in items:
        groups[group_key(item)].append(item)
    details = {}
    rates = []
    for name in sorted(groups):
        spec = FIELD_SPECS[path]
        if spec["metric"] == "boolean":
            result = boolean_result(groups[name], path)
        else:
            result = raw_sample_result(groups[name], path)
        details[name] = result
        rate = unquantized_rate(result)
        if rate is not None:
            rates.append(rate)
    output = base_result(items, path)
    rate = mean_defined(rates)
    output.update({
        "score_aggregation": "%s_macro" % group_kind,
        "score_sum_macro_raw": sum(rates) if rates else None,
        "max_score_macro_raw": len(rates) if rates else None,
        "score_sum_macro": q6(sum(rates) if rates else None),
        "max_score_macro": len(rates) if rates else None,
        "score_rate": q6(rate),
        "groups": details,
    })
    return output


def text_result(items, path):
    output = base_result(items, path)
    groups = defaultdict(list)
    for item in items:
        language = item["gt"]["language"]
        kind = "cer" if language in CJK_LANGUAGES else "wer"
        name = "%s|%s" % (language, kind)
        groups[name].append(item)
    details = {}
    rates = []
    for name in sorted(groups):
        group = groups[name]
        scores = [unit_score("text", item["state"], item["predicted"], item["reference"], FIELD_SPECS[path], item["gt"]["language"]) for item in group]
        rate = safe_divide(sum(scores), 2.0 * len(group))
        rates.append(rate)
        known = [item for item in group if item["state"] == "result_known"]
        known_distance = 0
        known_reference = 0
        e2e_distance = 0
        e2e_reference = 0
        exact_known = 0
        exact_e2e = 0
        for item in group:
            language = item["gt"]["language"]
            reference = text_units(item["reference"], language)
            hypothesis = text_units(item["predicted"], language) if item["state"] == "result_known" else []
            distance = edit_distance(reference, hypothesis)
            e2e_distance += distance
            e2e_reference += len(reference)
            if item["state"] == "result_known":
                known_distance += distance
                known_reference += len(reference)
                exact_known += reference == hypothesis
                exact_e2e += reference == hypothesis
        metric_kind = name.rsplit("|", 1)[1]
        details[name] = {
            "language": name.split("|", 1)[0],
            "metric_kind": metric_kind,
            "tokenizer_version": "char-tokenizer-v1" if metric_kind == "cer" else "word-tokenizer-v1",
            "normalization_version": NORMALIZATION_VERSION,
            "eligible_samples": len(group),
            "result_known": len(known),
            "score_sum_micro_raw": sum(scores),
            "max_score_micro_raw": 2.0 * len(group),
            "score_rate_micro": q6(rate),
            "%s_known" % metric_kind: q6(safe_divide(known_distance, known_reference)),
            "%s_e2e" % metric_kind: q6(safe_divide(e2e_distance, e2e_reference)),
            "reference_unit_count_known": known_reference,
            "reference_unit_count_e2e": e2e_reference,
            "valid_transcript_rate_e2e": q6(safe_divide(len(known), len(group))),
            "normalized_exact_accuracy_known": q6(safe_divide(exact_known, len(known))),
            "normalized_exact_accuracy_e2e": q6(safe_divide(exact_e2e, len(group))),
        }
    output.update({
        "score_aggregation": "asr_group_macro",
        "normalization_version": NORMALIZATION_VERSION,
        "text_edit_version": TEXT_EDIT_VERSION,
        "score_sum_macro_asr_raw": sum(rates) if rates else None,
        "max_score_macro_asr_raw": len(rates) if rates else None,
        "score_sum_macro_asr": q6(sum(rates) if rates else None),
        "max_score_macro_asr": len(rates) if rates else None,
        "score_rate": q6(mean_defined(rates)),
        "groups": details,
    })
    return output


def language_result(items, path):
    result = grouped_result(items, path, lambda item: item["reference"], "language")
    raw = raw_sample_result(items, path)
    result["score_aggregation"] = "language_macro"
    result["score_sum_micro_raw"] = raw["score_sum_raw"]
    result["max_score_micro_raw"] = raw["max_score_raw"]
    result["score_rate_micro"] = raw["score_rate"]
    known = [item for item in items if item["state"] == "result_known"]
    labels = sorted(set(item["reference"] for item in items))
    per_language = {}
    accuracy_known = []
    accuracy_e2e = []
    f1_known = []
    f1_e2e = []
    for label in labels:
        support_known = sum(item["reference"] == label for item in known)
        support_e2e = sum(item["reference"] == label for item in items)
        tp = sum(item["reference"] == label and item["predicted"] == label for item in known)
        fp = sum(item["reference"] != label and item["predicted"] == label for item in known)
        fn_known = sum(item["reference"] == label and item["predicted"] != label for item in known)
        fn_e2e = support_e2e - tp
        acc_k = safe_divide(tp, support_known)
        acc_e = safe_divide(tp, support_e2e)
        f1_k = safe_divide(2 * tp, 2 * tp + fp + fn_known)
        f1_e = safe_divide(2 * tp, 2 * tp + fp + fn_e2e)
        accuracy_known.append(acc_k)
        accuracy_e2e.append(acc_e)
        f1_known.append(f1_k)
        f1_e2e.append(f1_e)
        per_language[label] = {
            "support_known": support_known,
            "support_e2e": support_e2e,
            "accuracy_known": q6(acc_k),
            "accuracy_e2e": q6(acc_e),
            "f1_known": q6(f1_k),
            "f1_e2e": q6(f1_e),
        }
    correct = sum(item["predicted"] == item["reference"] for item in known)
    result["auxiliary"] = {
        "accuracy_known": q6(safe_divide(correct, len(known))),
        "accuracy_e2e": q6(safe_divide(correct, len(items))),
        "macro_accuracy_known": q6(mean_defined(accuracy_known)),
        "macro_accuracy_e2e": q6(mean_defined(accuracy_e2e)),
        "macro_f1_known": q6(mean_defined(f1_known)),
        "macro_f1_e2e": q6(mean_defined(f1_e2e)),
        "languages": per_language,
    }
    return result


def score_field(gt_rows, predictions, path):
    items = state_items(gt_rows, predictions, path)
    spec = FIELD_SPECS[path]
    if spec["metric"] == "gender_sequence":
        return gender_sequence_result(items, path)
    if spec["metric"] == "multilabel":
        return multilabel_result(gt_rows, predictions, path)
    if spec["metric"] == "text":
        return text_result(items, path)
    if spec["metric"] == "language":
        return language_result(items, path)
    if spec.get("lexical_grouped"):
        result = grouped_result(items, path, lambda item: item["gt"]["language"], "lexical_group")
        result["normalization_version"] = NORMALIZATION_VERSION
        result["lexical_rule_version"] = LEXICAL_RULE_VERSION
        if spec["metric"] == "normalized_absolute":
            micro = raw_sample_result(items, path)
            result["normalized_absolute_error_version"] = NORMALIZED_ABSOLUTE_ERROR_VERSION
            result["score_formula"] = (
                "2*(1-abs(predicted-reference)/max(predicted,reference,1))"
            )
            result["score_sum_micro_raw"] = micro["score_sum_raw"]
            result["max_score_micro_raw"] = micro["max_score_raw"]
            result["score_rate_micro"] = micro["score_rate"]
            result["auxiliary_micro"] = micro["auxiliary"]
        return result
    if spec["metric"] == "boolean":
        return boolean_result(items, path)
    return raw_sample_result(items, path)


def score_layers(scored_groups):
    output = {}
    for layer_id, display_name, fields in LAYER_DEFINITIONS:
        labels = {}
        label_rates = []
        for path in fields:
            group_rates = {}
            raw_rates = []
            for group_name in sorted(scored_groups):
                field = scored_groups[group_name]["fields"].get(path)
                rate = unquantized_rate(field) if field is not None else None
                if rate is not None:
                    group_rates[group_name] = q6(rate)
                    raw_rates.append(rate)
            if not raw_rates:
                continue
            label_rate = mean_defined(raw_rates)
            label_rates.append(label_rate)
            labels[path] = {
                "score_aggregation": "evaluation_group_macro",
                "evaluation_group_count": len(raw_rates),
                "evaluation_group_rates": group_rates,
                "score_sum_macro_groups_raw": sum(raw_rates),
                "max_score_macro_groups_raw": len(raw_rates),
                "score_rate": q6(label_rate),
            }
        layer_rate = mean_defined(label_rates)
        output[layer_id] = {
            "display_name": display_name,
            "score_aggregation": "label_first_macro",
            "defined_label_count": len(label_rates),
            "score_sum_macro_labels_raw": sum(label_rates) if label_rates else None,
            "max_score_macro_labels_raw": len(label_rates) if label_rates else None,
            "score_rate": q6(layer_rate),
            "labels": labels,
        }
    return output


def contract_check(prediction, target, dependencies, formula):
    dep_values = []
    for dependency in dependencies:
        state, value = prediction_state(prediction, dependency, FIELD_SPECS[dependency])
        if state != "result_known":
            return "not_checkable"
        dep_values.append(value)
    expected = formula(*dep_values)
    if expected is None:
        return "not_checkable"
    state, actual = prediction_state(prediction, target, FIELD_SPECS[target])
    if state != "result_known":
        return "mismatch"
    if isinstance(expected, float):
        return "pass" if abs(float(actual) - expected) <= 1e-6 else "mismatch"
    return "pass" if actual == expected else "mismatch"


def score_contracts(predictions):
    rules = (
        ("speaker_present_from_count", "speaker.speaker_present", ("speaker.speaker_count",), lambda count: count > 0),
        ("multi_speaker_from_count", "speaker.multi_speaker", ("speaker.speaker_count",), lambda count: count > 1),
        ("speaker_change_from_count", "speaker.speaker_change", ("speaker.speaker_change_count",), lambda count: count > 0),
        ("music_present_from_events", "sound_field_scene.music_present", ("sound_field_scene.speech_music_events",), lambda events: "music" in events),
        ("speaker_change_rate_per_min", "deterministic_derived.speaker_change_rate_per_min", ("speaker.speaker_change_count", "basic_acoustic.duration_sec"), lambda count, duration: q6(60.0 * count / duration) if duration > 0 else None),
        ("filler_ratio", "deterministic_derived.filler_ratio", ("language_content.filler", "language_content.word_count"), lambda filler, words: q6(filler / words) if words > 0 else None),
    )
    output = {}
    for name, target, dependencies, formula in rules:
        counts = Counter(contract_check(prediction, target, dependencies, formula) for prediction in predictions.values())
        checkable = counts["pass"] + counts["mismatch"]
        output[name] = {
            "contract_input_count": len(predictions),
            "contract_checkable_units": checkable,
            "pass_count_raw": counts["pass"],
            "mismatch_count_raw": counts["mismatch"],
            "not_checkable": counts["not_checkable"],
            "contract_pass_rate": q6(safe_divide(counts["pass"], checkable)),
        }
    return output


def score(
    benchmark_root,
    predictions_path,
    system="sure-tagger+deepseek-v4-pro",
    structural_prediction_owner="sure-tagger",
):
    benchmark_root = Path(benchmark_root).resolve()
    predictions_path = Path(predictions_path).resolve()
    ground_truth = build_ground_truth(benchmark_root)
    prediction_rows = load_jsonl(predictions_path)
    if prediction_rows and set(prediction_rows[0]) == {"sample_id", "tags"}:
        prediction_rows = [{"sample_id": row["sample_id"], **row["tags"]} for row in prediction_rows]
    predictions = keyed(prediction_rows, lambda row: row.get("sample_id"), "predictions")
    expected = set(ground_truth)
    actual = set(predictions)
    if actual - expected:
        raise ValueError("input_contract_failed: %d unknown prediction IDs" % len(actual - expected))
    if expected - actual:
        # Missing expected rows are valid per-unit runtime failures.
        pass

    groups = defaultdict(list)
    for row in ground_truth.values():
        groups[row["evaluation_group"]].append(row)
    scored_groups = {}
    for name in sorted(groups):
        rows = groups[name]
        fields = {}
        for path in CONTRACT_FIELDS + CAPABILITY_FIELDS:
            eligible = [row for row in rows if path in row["gt_mask"]]
            if eligible:
                fields[path] = score_field(eligible, predictions, path)
                fields[path]["input_sample_count"] = len(rows)
                fields[path]["gt_excluded_samples"] = len(rows) - len(eligible)
        capability_rates = [
            unquantized_rate(value) for path, value in fields.items()
            if FIELD_SPECS[path]["role"] == "capability" and unquantized_rate(value) is not None
        ]
        scored_groups[name] = {
            "track": rows[0]["track"],
            "source_group": rows[0]["source_group"],
            "sample_count": len(rows),
            "defined_capability_axis_count": len(capability_rates),
            "capability_group_rate": q6(mean_defined(capability_rates)),
            "fields": fields,
        }

    state_totals = {}
    for path in CONTRACT_FIELDS + CAPABILITY_FIELDS:
        counts = Counter()
        for prediction in predictions.values():
            state, _ = prediction_state(prediction, path, FIELD_SPECS[path])
            counts[state] += 1
        state_totals[path] = {state: counts[state] for state in RESULT_STATES}

    layer_summaries = score_layers(scored_groups)

    manifest_path = benchmark_root / "benchmark.manifest.jsonl"
    metadata_path = benchmark_root / "benchmark.metadata.jsonl"
    return {
        "schema_version": "benchmark-structural-score-v1.3",
        "system": system,
        "structural_prediction_owner": structural_prediction_owner,
        "descriptive_fields_scored": False,
        "speaker_gender_scored": True,
        "scoring_version": SCORING_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "field_universe_version": FIELD_UNIVERSE_VERSION,
        "input_contract_failed": False,
        "input_adapter": "finalized-fixed-schema-nested-json-v1+silence-ratio-alias+speaker-gender-sequence+profile-gender-compat",
        "expected_sample_count": len(ground_truth),
        "prediction_row_count": len(predictions),
        "missing_result_rows": len(expected - actual),
        "leakage_status": "clean",
        "versions": {
            "normalization_version": NORMALIZATION_VERSION,
            "lexical_rule_version": LEXICAL_RULE_VERSION,
            "event_enum_version": EVENT_ENUM_VERSION,
            "text_edit_version": TEXT_EDIT_VERSION,
            "normalized_absolute_error_version": NORMALIZED_ABSOLUTE_ERROR_VERSION,
            "layer_aggregation_version": LAYER_AGGREGATION_VERSION,
            "T1_silence_ratio_gt_version": SILENCE_RATIO_GT_VERSION,
            "T1_speaker_gender_gt_version": SPEAKER_GENDER_GT_VERSION,
            "speaker_gender_edit_version": GENDER_EDIT_VERSION,
            "unicode_data_version": unicodedata.unidata_version,
            "T1_speaker_rule_version": "speaker-gt-benchmark-v1-seed",
        },
        "inputs": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "benchmark_manifest": str(manifest_path),
            "benchmark_manifest_sha256": sha256_file(manifest_path),
            "benchmark_metadata": str(metadata_path),
            "benchmark_metadata_sha256": sha256_file(metadata_path),
            "T1_reference_manifest_sha256": sha256_file(benchmark_root / "t1-ami" / "manifest.jsonl"),
            "T2_metadata_sha256": sha256_file(benchmark_root / "t2-language-asr" / "metadata.jsonl"),
            "T3_closed_world_gt_sha256": sha256_file(benchmark_root / "t3-event-noise" / "closed_world_gt.jsonl"),
            "speaker_gender_gt": str(benchmark_root / "t1-ami" / "speaker_gender_gt" / "speaker_gender_gt.jsonl"),
            "speaker_gender_gt_sha256": sha256_file(benchmark_root / "t1-ami" / "speaker_gender_gt" / "speaker_gender_gt.jsonl"),
            "scorer": str(Path(__file__).resolve()),
            "scorer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "notes": [
            "descriptive semantic fields remain excluded; speaker.gender is scored structurally from descriptive.speaker_descriptions[].gender, with explicit speaker.profiles[].gender accepted as a compatibility source",
            "layer scores use label-first macro aggregation across evaluation groups",
            "basic_acoustic.nonspeech_ratio and speaker.overlap_ratio use direct absolute ratio error without a tolerance divisor",
            "T1 official silence_ratio GT is the complement of merged AMI utterance turns",
            "T1 speaker.gender GT is loaded from speaker_gender_gt.jsonl and ordered by first speaker appearance",
            "speaker.gender uses WER-style Levenshtein distance over gender sequences, normalized by reference speaker count to [0,2]",
            "basic_acoustic.silence_ratio predictions are accepted as an alias of basic_acoustic.nonspeech_ratio",
            "T2 filler/filler_ratio are enabled only for English because no other frozen filler lexicon is present",
            "language_content.filler, language_content.punctuation.punctuation_count, and language_content.punctuation.has_terminal_punctuation are retained in predictions but excluded from scoring",
            "T3 event/noise scores are split by DESED and FSD50K source group",
        ],
        "all_input_field_states": state_totals,
        "contracts": score_contracts(predictions),
        "evaluation_groups": scored_groups,
        "layer_summaries": layer_summaries,
    }


def render_report(result):
    lines = [
        "# Benchmark-v1 Label Scores: %s" % result["system"],
        "",
        "Scoring: `%s`; predictions: `%d/%d`; descriptive semantic fields are not scored (speaker.gender is structural)."
        % (result["scoring_version"], result["prediction_row_count"], result["expected_sample_count"]),
        "",
        "The score column is `score_rate` on a 0--100% scale. `n` is the field's eligible sample count; unavailable GT is excluded, not scored as zero.",
        "",
    ]
    lines.extend([
        "## Label-first Layer Summaries",
        "",
        "Each label is first averaged equally across evaluation groups; each layer is then averaged equally across its defined labels.",
        "",
        "| Layer | Defined labels | Score |",
        "| --- | ---: | ---: |",
    ])
    for layer_id in sorted(result["layer_summaries"]):
        layer = result["layer_summaries"][layer_id]
        rate = layer["score_rate"]
        lines.append("| %s | %d | %s |" % (
            layer["display_name"],
            layer["defined_label_count"],
            "n/a" if rate is None else "%.2f%%" % (100 * rate),
        ))
    lines.append("")
    for group_name, group in result["evaluation_groups"].items():
        lines.extend([
            "## %s" % group_name,
            "",
            "Samples: %d; defined capability axes: %d; equal-axis diagnostic mean: %s."
            % (
                group["sample_count"],
                group["defined_capability_axis_count"],
                "n/a" if group["capability_group_rate"] is None else "%.2f%%" % (100 * group["capability_group_rate"]),
            ),
            "",
            "| Role | Field | n | Known | Score |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for path, field in group["fields"].items():
            score_rate = field.get("score_rate")
            lines.append("| %s | `%s` | %d | %d | %s |" % (
                field["metric_role"],
                path,
                field["eligible_samples"],
                field["result_known"],
                "n/a" if score_rate is None else "%.2f%%" % (100 * score_rate),
            ))
        lines.append("")
    lines.extend(["## Contract Checks", "", "| Rule | Checkable | Pass rate |", "| --- | ---: | ---: |"])
    for name, value in result["contracts"].items():
        rate = value["contract_pass_rate"]
        lines.append("| `%s` | %d | %s |" % (
            name,
            value["contract_checkable_units"],
            "n/a" if rate is None else "%.2f%%" % (100 * rate),
        ))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--system", default="sure-tagger+deepseek-v4-pro")
    parser.add_argument("--structural-prediction-owner", default="sure-tagger")
    args = parser.parse_args(argv)
    result = score(
        args.benchmark_root,
        args.predictions,
        system=args.system,
        structural_prediction_owner=args.structural_prediction_owner,
    )
    atomic_write(args.output, json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write(args.report, render_report(result))
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "report": str(Path(args.report).resolve()),
        "prediction_rows": result["prediction_row_count"],
        "groups": len(result["evaluation_groups"]),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
