"""Deterministic VAD from native metadata speech segments."""

from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    speech_segments_to_silence_segments,
    validate_silence_segments,
)


TOOL_NAME = "native_metadata_vad"
METHOD = "native_metadata_speech_segments"


class NativeMetadataVadError(ValueError):
    pass


def run(sample, duration_sec, context=None, **_kwargs):
    # type: (dict, float, object) -> ToolResult
    del context
    if duration_sec is None or duration_sec <= 0:
        raise NativeMetadataVadError("duration_sec must be positive")
    speech_segments, source_key = speech_segments_from_native_metadata(
        sample,
        duration_sec,
    )
    if not speech_segments:
        raise NativeMetadataVadError("no native metadata speech segments found")
    silence_segments = speech_segments_to_silence_segments(
        speech_segments,
        duration_sec=duration_sec,
    )
    validate_silence_segments(silence_segments, duration_sec)
    return ToolResult(
        tag_path="basic_acoustic.silence_segments",
        value=silence_segments,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        status="observed",
        confidence=1.0,
        tool_type="deterministic",
        evidence={
            "source": "sample.native_metadata.%s" % source_key,
            "speech_segments": speech_segments,
            "duration_sec": duration_sec,
        },
    )


def speech_segments_from_native_metadata(sample, duration_sec):
    native_metadata = sample.get("native_metadata", {})
    if not isinstance(native_metadata, dict):
        raise NativeMetadataVadError("sample.native_metadata must be an object")

    direct_silence = native_metadata.get("silence_segments")
    if isinstance(direct_silence, list):
        silence_segments = _segments_from_list(
            direct_silence,
            duration_sec,
            native_metadata,
            "silence_segments",
        )
        if silence_segments:
            speech_segments = _speech_from_silence_segments(
                silence_segments,
                duration_sec,
            )
            return speech_segments, "silence_segments"

    for key in ("speech_segments", "vad_segments", "segments", "utterances", "words"):
        value = native_metadata.get(key)
        if not isinstance(value, list):
            continue
        segments = _segments_from_list(value, duration_sec, native_metadata, key)
        if segments:
            return segments, key

    raise NativeMetadataVadError("no native metadata speech segments found")


def _segments_from_list(items, duration_sec, native_metadata, key):
    parsed = []
    for item in items:
        segment = _parse_segment(item)
        if segment is None:
            continue
        if key == "words" and _word_is_punctuation(item):
            continue
        parsed.append(segment)
    return _normalize_segments(parsed, duration_sec, native_metadata)


def _speech_from_silence_segments(silence_segments, duration_sec):
    speech = []
    cursor = 0.0
    for segment in silence_segments:
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        if start_sec > cursor:
            speech.append({"start_sec": cursor, "end_sec": start_sec})
        cursor = max(cursor, end_sec)
    if cursor < duration_sec:
        speech.append({"start_sec": cursor, "end_sec": duration_sec})
    return _round_segments(speech)


def _parse_segment(item):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        start = _as_float(item[0])
        end = _as_float(item[1])
    elif isinstance(item, dict):
        start = _as_float(item.get("start_sec", item.get("start")))
        end = _as_float(item.get("end_sec", item.get("end")))
    else:
        return None
    if start is None or end is None or end <= start:
        return None
    return {"start_sec": start, "end_sec": end}


def _normalize_segments(segments, duration_sec, native_metadata):
    parsed = [
        dict(item)
        for item in segments
        if item.get("end_sec") > item.get("start_sec")
    ]
    if not parsed:
        return []
    duration_sec = float(duration_sec)
    parent_start = _as_float(native_metadata.get("start_sec", native_metadata.get("start")))
    if not _segments_fit_duration(parsed, duration_sec) and parent_start is not None:
        shifted = []
        for item in parsed:
            shifted.append(
                {
                    "start_sec": item["start_sec"] - parent_start,
                    "end_sec": item["end_sec"] - parent_start,
                }
            )
        if _segments_fit_duration(shifted, duration_sec):
            parsed = shifted

    clipped = []
    for item in parsed:
        start = max(0.0, float(item["start_sec"]))
        end = min(duration_sec, float(item["end_sec"]))
        if end > start:
            clipped.append({"start_sec": start, "end_sec": end})
    return _merge_segments(clipped, duration_sec)


def _segments_fit_duration(segments, duration_sec):
    for item in segments:
        if item["start_sec"] < -1e-6 or item["end_sec"] > duration_sec + 1e-6:
            return False
    return True


def _merge_segments(segments, duration_sec):
    items = sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"]))
    merged = []
    for item in items:
        current = {
            "start_sec": max(0.0, min(float(duration_sec), float(item["start_sec"]))),
            "end_sec": max(0.0, min(float(duration_sec), float(item["end_sec"]))),
        }
        if current["end_sec"] <= current["start_sec"]:
            continue
        if not merged or current["start_sec"] > merged[-1]["end_sec"]:
            merged.append(current)
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], current["end_sec"])
    return _round_segments(merged)


def _round_segments(segments):
    return [
        {
            "start_sec": round(float(item["start_sec"]), 6),
            "end_sec": round(float(item["end_sec"]), 6),
        }
        for item in segments
    ]


def _word_is_punctuation(item):
    if not isinstance(item, dict):
        return False
    text = str(item.get("w", item.get("word", item.get("text", "")))).strip()
    return bool(text) and not any(ch.isalnum() for ch in text)


def _as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result
