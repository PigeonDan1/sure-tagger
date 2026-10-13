"""Speaker diarization metadata and public tag derivation."""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tagger.tools.base import ToolResult


TOOL_VERSION = "speaker_metrics_v0.2.0"
METADATA_VERSION = "speaker_diarization_v0.1"


class SpeakerMetricsError(ValueError):
    """Raised when speaker timeline metadata cannot be computed."""


class SpeakerMetricsConfig:
    def __init__(
        self,
        min_segment_duration_sec=0.10,
        merge_same_speaker_gap_sec=0.30,
        speaker_change_max_gap_sec=1.00,
        min_speech_duration_sec=1.00,
        overlap_min_duration_sec=0.10,
        utterance_overlap_threshold=0.05,
    ):
        self.min_segment_duration_sec = float(min_segment_duration_sec)
        self.merge_same_speaker_gap_sec = float(merge_same_speaker_gap_sec)
        self.speaker_change_max_gap_sec = float(speaker_change_max_gap_sec)
        self.min_speech_duration_sec = float(min_speech_duration_sec)
        self.overlap_min_duration_sec = float(overlap_min_duration_sec)
        self.utterance_overlap_threshold = float(utterance_overlap_threshold)


def build_metadata_from_timeline(
    segments,
    duration_sec,
    sample_id,
    recording_id=None,
    input_kind="unknown_audio_layout",
    primary_route="moss_diarize",
    target_units=None,
    config=None,
):
    # type: (List[Dict[str, Any]], float, str, Optional[str], str, str, Optional[List[Dict[str, Any]]], Optional[SpeakerMetricsConfig]) -> Dict[str, Any]
    config = config or SpeakerMetricsConfig()
    duration_sec = _validate_duration(duration_sec)
    normalized = normalize_segments(segments, duration_sec, config)
    if not normalized:
        raise SpeakerMetricsError("no valid speaker segments")

    per_speaker_segments = _segments_by_speaker(normalized)
    merged_by_speaker = {
        speaker_id: merge_segments(items, config.merge_same_speaker_gap_sec, duration_sec)
        for speaker_id, items in per_speaker_segments.items()
    }
    merged_segments = []
    for speaker_id, items in merged_by_speaker.items():
        for item in items:
            item = dict(item)
            item["speaker_id"] = speaker_id
            merged_segments.append(item)
    merged_segments = sorted(merged_segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]))

    union_segments = union_intervals([
        {"start_sec": item["start_sec"], "end_sec": item["end_sec"]}
        for item in merged_segments
    ], duration_sec)
    overlap_segments = overlap_intervals(merged_by_speaker, duration_sec, config.overlap_min_duration_sec)
    union_speech_duration_sec = round(sum(_duration(item) for item in union_segments), 6)
    overlap_duration_sec = round(sum(_duration(item) for item in overlap_segments), 6)
    per_speaker_duration = {
        speaker_id: round(sum(_duration(item) for item in items), 6)
        for speaker_id, items in merged_by_speaker.items()
    }
    active_speakers = sorted([
        speaker_id for speaker_id, value in per_speaker_duration.items()
        if value >= config.min_segment_duration_sec
    ])
    if union_speech_duration_sec < config.min_speech_duration_sec:
        raise SpeakerMetricsError("speech duration below threshold")

    speaker_change_points = speaker_change_points_from_segments(merged_segments, config.speaker_change_max_gap_sec)
    speaker_count = len(active_speakers)
    dominant_ratio = (
        max(per_speaker_duration.values()) / union_speech_duration_sec
        if union_speech_duration_sec > 0 and per_speaker_duration
        else None
    )
    overlap_ratio_speech = (
        overlap_duration_sec / union_speech_duration_sec
        if union_speech_duration_sec > 0
        else None
    )
    overlap_ratio_audio = (
        overlap_duration_sec / duration_sec
        if duration_sec > 0
        else None
    )
    summary = {
        "speaker_count": speaker_count,
        "multi_speaker": speaker_count >= 2,
        "turn_count": len(merged_segments),
        "speaker_change_count": len(speaker_change_points),
        "speaker_change_points": [round(float(value), 3) for value in speaker_change_points],
        "speaker_change_rate_per_min": _rounded_rate(len(speaker_change_points), duration_sec),
        "overlap_ratio_speech": _round_ratio(overlap_ratio_speech),
        "overlap_ratio_audio": _round_ratio(overlap_ratio_audio),
        "dominant_speaker_ratio": _round_ratio(dominant_ratio),
        "speaker_balance": _round_ratio(speaker_balance(per_speaker_duration)),
        "crosstalk_level": crosstalk_level(overlap_ratio_speech),
        "union_speech_duration_sec": union_speech_duration_sec,
        "overlap_duration_sec": overlap_duration_sec,
    }
    speakers = []
    for speaker_id in active_speakers:
        speakers.append({
            "speaker_id": speaker_id,
            "source_channel_id": _first_source_channel(merged_by_speaker.get(speaker_id, [])),
            "speech_duration_sec": per_speaker_duration[speaker_id],
            "turn_count": len(merged_by_speaker.get(speaker_id, [])),
        })

    metadata = {
        "metadata_version": METADATA_VERSION,
        "sample_id": sample_id,
        "recording_id": recording_id or sample_id,
        "input_kind": input_kind,
        "primary_route": primary_route,
        "duration_sec": duration_sec,
        "speakers": speakers,
        "segments": _with_segment_ids(merged_segments),
        "overlap_segments": overlap_segments,
        "utterances": utterance_metadata(
            target_units or [],
            merged_by_speaker,
            overlap_segments,
            config,
            speaker_change_points=speaker_change_points,
        ),
        "recording_summary": summary,
        "quality": {
            "status": "ok",
            "warnings": [],
        },
    }
    return metadata


def build_metadata_from_channel_activity(
    channel_activity,
    duration_sec,
    sample_id,
    recording_id=None,
    input_kind="separated_headset_channels",
    target_units=None,
    config=None,
):
    # type: (Dict[str, Any], float, str, Optional[str], str, Optional[List[Dict[str, Any]]], Optional[SpeakerMetricsConfig]) -> Dict[str, Any]
    segments = []
    channels = channel_activity.get("channels", [])
    if not isinstance(channels, list):
        raise SpeakerMetricsError("channel_activity.channels must be a list")
    for channel_index, channel in enumerate(channels):
        channel_id = str(channel.get("channel_id", "ch%s" % channel_index))
        speaker_id = str(channel.get("speaker_id", "spk_%03d" % channel_index))
        for item in channel.get("speech_segments", []):
            segment = dict(item)
            segment["speaker_id"] = speaker_id
            segment["source_channel_id"] = channel_id
            segments.append(segment)
    return build_metadata_from_timeline(
        segments,
        duration_sec,
        sample_id,
        recording_id=recording_id,
        input_kind=input_kind,
        primary_route="channel_activity",
        target_units=target_units,
        config=config,
    )


def public_results_from_metadata(metadata):
    # type: (Dict[str, Any]) -> List[ToolResult]
    summary = metadata.get("recording_summary", {})
    utterance = _select_public_utterance(metadata)
    tool_name = "speaker_metrics"
    method = "utterance_diarization_metrics"
    evidence = {
        "metadata_version": metadata.get("metadata_version"),
        "primary_route": metadata.get("primary_route"),
        "input_kind": metadata.get("input_kind"),
    }
    if utterance is not None:
        fields = [
            ("speaker.speaker_count", _non_negative_int_or_none(utterance.get("active_speaker_count"))),
            ("speaker.multi_speaker", _bool_from_min_count(utterance.get("active_speaker_count"), 2)),
            ("speaker.speaker_change_count", _non_negative_int_or_none(utterance.get("speaker_change_count"))),
            ("speaker.speaker_change", _bool_from_positive_count(utterance.get("speaker_change_count"))),
            ("speaker.overlap_ratio", _ratio_or_none(utterance.get("overlap_ratio"))),
            ("speaker.speaker_overlap", _bool_or_none(utterance.get("is_overlapped"))),
        ]
    elif metadata.get("utterances"):
        fields = [
            ("speaker.speaker_count", None),
            ("speaker.multi_speaker", None),
            ("speaker.speaker_change_count", None),
            ("speaker.speaker_change", None),
            ("speaker.overlap_ratio", None),
            ("speaker.speaker_overlap", None),
        ]
    else:
        fields = [
            ("speaker.speaker_count", _non_negative_int_or_none(summary.get("speaker_count"))),
            ("speaker.multi_speaker", summary.get("multi_speaker")),
            ("speaker.speaker_change_count", _non_negative_int_or_none(summary.get("speaker_change_count"))),
            ("speaker.speaker_change", _bool_from_positive_count(summary.get("speaker_change_count"))),
            ("speaker.overlap_ratio", _ratio_or_none(summary.get("overlap_ratio_speech"))),
            ("speaker.speaker_overlap", _bool_from_positive_ratio(summary.get("overlap_ratio_speech"))),
        ]
    return [
        ToolResult(
            tag_path=tag_path,
            value=value,
            tool_name=tool_name,
            method=method,
            status="estimated" if value is not None else "failed",
            confidence=1.0 if value is not None else 0.0,
            tool_type="derived",
            tool_version=TOOL_VERSION,
            evidence=evidence,
        )
        for tag_path, value in fields
    ]


def _select_public_utterance(metadata):
    # type: (Dict[str, Any]) -> Optional[Dict[str, Any]]
    utterances = metadata.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        return None
    sample_id = str(metadata.get("sample_id", ""))
    for item in utterances:
        if isinstance(item, dict) and str(item.get("unit_id", "")) == sample_id:
            return item
    if len(utterances) == 1 and isinstance(utterances[0], dict):
        return utterances[0]
    return None


def _bool_from_min_count(value, minimum):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value) >= int(minimum)
    except (TypeError, ValueError):
        return None


def _non_negative_int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _ratio_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return None
    return _round_ratio(value)


def _bool_from_positive_count(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return None


def _bool_from_positive_ratio(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return None


def _bool_or_none(value):
    if isinstance(value, bool):
        return value
    return None


def normalize_segments(segments, duration_sec, config=None):
    # type: (Iterable[Dict[str, Any]], float, Optional[SpeakerMetricsConfig]) -> List[Dict[str, Any]]
    config = config or SpeakerMetricsConfig()
    duration_sec = _validate_duration(duration_sec)
    normalized = []
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker") or item.get("label")
        if speaker_id is None:
            continue
        start = _as_float(item.get("start_sec", item.get("start")))
        end = _as_float(item.get("end_sec", item.get("end")))
        if start is None or end is None:
            continue
        start = max(0.0, start)
        end = min(duration_sec, end)
        if end - start < config.min_segment_duration_sec:
            continue
        segment = {
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
            "speaker_id": normalize_speaker_id(speaker_id),
        }
        if item.get("source_channel_id") is not None:
            segment["source_channel_id"] = str(item.get("source_channel_id"))
        if item.get("text"):
            segment["text"] = str(item.get("text"))
        normalized.append(segment)
    return sorted(normalized, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]))


def normalize_speaker_id(value):
    raw = str(value).strip()
    if not raw:
        return "spk_unknown"
    lowered = raw.lower()
    if lowered.startswith("spk_"):
        return lowered
    if lowered.startswith("speaker_"):
        suffix = lowered.split("speaker_", 1)[1].strip("_")
        return "spk_%s" % suffix if suffix else "spk_unknown"
    if lowered.startswith("s") and lowered[1:].isdigit():
        return "spk_%03d" % int(lowered[1:])
    if lowered.isdigit():
        return "spk_%03d" % int(lowered)
    return lowered.replace(" ", "_")


def merge_segments(segments, merge_gap_sec, duration_sec):
    # type: (List[Dict[str, Any]], float, float) -> List[Dict[str, Any]]
    items = sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"]))
    merged = []
    for item in items:
        current = dict(item)
        current["start_sec"] = max(0.0, float(current["start_sec"]))
        current["end_sec"] = min(float(duration_sec), float(current["end_sec"]))
        if not merged or current["start_sec"] - merged[-1]["end_sec"] > merge_gap_sec:
            merged.append(current)
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], current["end_sec"])
            if current.get("text"):
                text = " ".join([merged[-1].get("text", ""), current.get("text", "")]).strip()
                if text:
                    merged[-1]["text"] = text
    return merged


def union_intervals(segments, duration_sec):
    # type: (List[Dict[str, Any]], float) -> List[Dict[str, float]]
    items = []
    for segment in segments:
        start = _as_float(segment.get("start_sec"))
        end = _as_float(segment.get("end_sec"))
        if start is None or end is None:
            continue
        start = max(0.0, start)
        end = min(float(duration_sec), end)
        if end <= start:
            continue
        items.append({"start_sec": start, "end_sec": end})
    items.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged = []
    for item in items:
        if not merged or item["start_sec"] > merged[-1]["end_sec"]:
            merged.append(dict(item))
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], item["end_sec"])
    return [{"start_sec": round(item["start_sec"], 6), "end_sec": round(item["end_sec"], 6)} for item in merged]


def overlap_intervals(segments_by_speaker, duration_sec, min_duration_sec=0.10):
    # type: (Dict[str, List[Dict[str, Any]]], float, float) -> List[Dict[str, Any]]
    events = []
    for speaker_id, segments in segments_by_speaker.items():
        for segment in segments:
            start = max(0.0, float(segment["start_sec"]))
            end = min(float(duration_sec), float(segment["end_sec"]))
            if end <= start:
                continue
            events.append((start, 1, speaker_id))
            events.append((end, -1, speaker_id))
    events.sort(key=lambda item: (item[0], -item[1]))
    active = set()
    previous_time = None
    overlaps = []
    for time_sec, delta, speaker_id in events:
        if previous_time is not None and time_sec > previous_time and len(active) >= 2:
            if time_sec - previous_time >= min_duration_sec:
                overlaps.append({
                    "start_sec": round(previous_time, 6),
                    "end_sec": round(time_sec, 6),
                    "speaker_ids": sorted(active),
                })
        if delta > 0:
            active.add(speaker_id)
        else:
            active.discard(speaker_id)
        previous_time = time_sec
    return overlaps


def speaker_change_points_from_segments(segments, max_gap_sec):
    # type: (List[Dict[str, Any]], float) -> List[float]
    points = []
    previous = None
    for segment in sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"])):
        if previous is not None and segment["speaker_id"] != previous["speaker_id"]:
            gap = float(segment["start_sec"]) - float(previous["end_sec"])
            if gap <= max_gap_sec:
                point = float(segment["start_sec"])
                if not points or abs(point - points[-1]) > 1e-6:
                    points.append(point)
        previous = segment
    return points


def utterance_metadata(target_units, segments_by_speaker, overlap_segments, config=None, speaker_change_points=None):
    # type: (List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Optional[SpeakerMetricsConfig], Optional[List[float]]) -> List[Dict[str, Any]]
    config = config or SpeakerMetricsConfig()
    speaker_change_points = speaker_change_points or []
    utterances = []
    previous_speaker = None
    for unit in target_units:
        unit_id = str(unit.get("unit_id", unit.get("utt_id", "")))
        start = _as_float(unit.get("start_sec", unit.get("start")))
        end = _as_float(unit.get("end_sec", unit.get("end")))
        if start is None or end is None or end <= start:
            continue
        duration = end - start
        per_speaker = {}
        for speaker_id, segments in segments_by_speaker.items():
            total = 0.0
            for segment in segments:
                total += interval_intersection(start, end, segment["start_sec"], segment["end_sec"])
            if total > 0:
                per_speaker[speaker_id] = total
        primary_speaker_id = None
        primary_duration = 0.0
        if per_speaker:
            primary_speaker_id, primary_duration = max(per_speaker.items(), key=lambda item: item[1])
        overlap_duration = 0.0
        for segment in overlap_segments:
            overlap_duration += interval_intersection(start, end, segment["start_sec"], segment["end_sec"])
        speech_union_duration = speech_union_duration_for_window(start, end, segments_by_speaker)
        overlap_ratio = (
            overlap_duration / speech_union_duration
            if speech_union_duration > 0
            else 0.0
        )
        active_speaker_count = len(per_speaker)
        speaker_change_count = sum(1 for point in speaker_change_points if start < float(point) < end)
        is_overlapped = overlap_ratio >= config.utterance_overlap_threshold
        turn_position = "single"
        if is_overlapped:
            turn_position = "overlap"
        elif primary_speaker_id is not None and previous_speaker == primary_speaker_id:
            turn_position = "continue"
        elif primary_speaker_id is not None:
            turn_position = "start"
        utterances.append({
            "unit_id": unit_id,
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
            "primary_speaker_id": primary_speaker_id,
            "active_speaker_count": active_speaker_count,
            "speaker_change_count": speaker_change_count,
            "speaker_change": speaker_change_count > 0,
            "is_overlapped": is_overlapped,
            "overlap_ratio": _round_ratio(overlap_ratio),
            "overlap_duration_sec": round(overlap_duration, 6),
            "speech_union_duration_sec": round(speech_union_duration, 6),
            "primary_speaker_coverage": _round_ratio(primary_duration / duration if duration > 0 else None),
            "turn_position": turn_position,
        })
        if primary_speaker_id is not None:
            previous_speaker = primary_speaker_id
    return utterances


def speech_union_duration_for_window(start, end, segments_by_speaker):
    # type: (float, float, Dict[str, List[Dict[str, Any]]]) -> float
    intervals = []
    for segments in segments_by_speaker.values():
        for segment in segments:
            clipped_start = max(float(start), float(segment["start_sec"]))
            clipped_end = min(float(end), float(segment["end_sec"]))
            if clipped_end > clipped_start:
                intervals.append((clipped_start, clipped_end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = []
    for item_start, item_end in intervals:
        if not merged or item_start > merged[-1][1]:
            merged.append([item_start, item_end])
        else:
            merged[-1][1] = max(merged[-1][1], item_end)
    return round(sum(item_end - item_start for item_start, item_end in merged), 6)


def interval_intersection(a_start, a_end, b_start, b_end):
    # type: (float, float, float, float) -> float
    start = max(float(a_start), float(b_start))
    end = min(float(a_end), float(b_end))
    return max(0.0, end - start)


def speaker_balance(per_speaker_duration):
    # type: (Dict[str, float]) -> Optional[float]
    values = [float(v) for v in per_speaker_duration.values() if v > 0]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    total = sum(values)
    entropy = 0.0
    for value in values:
        p = value / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(values))


def crosstalk_level(overlap_ratio_speech):
    value = overlap_ratio_speech
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        return "none"
    if value < 0.10:
        return "low"
    if value < 0.25:
        return "medium"
    return "high"


def _segments_by_speaker(segments):
    grouped = {}
    for item in segments:
        grouped.setdefault(item["speaker_id"], []).append(item)
    return grouped


def _duration(segment):
    return max(0.0, float(segment["end_sec"]) - float(segment["start_sec"]))


def _first_source_channel(segments):
    for item in segments:
        if item.get("source_channel_id") is not None:
            return item.get("source_channel_id")
    return None


def _with_segment_ids(segments):
    out = []
    for index, item in enumerate(segments):
        rec = dict(item)
        rec.setdefault("segment_id", "seg_%06d" % index)
        out.append(rec)
    return out


def _rounded_rate(count, duration_sec):
    if duration_sec <= 0:
        return None
    return round(float(count) / (duration_sec / 60.0), 6)


def _round_ratio(value):
    if value is None:
        return None
    return round(max(0.0, min(1.0, float(value))), 6)


def _as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _validate_duration(value):
    duration_sec = _as_float(value)
    if duration_sec is None or duration_sec <= 0:
        raise SpeakerMetricsError("duration_sec must be positive")
    return duration_sec
