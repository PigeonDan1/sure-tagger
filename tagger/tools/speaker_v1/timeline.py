"""Deterministic timeline summaries and cross-source event comparison."""

from tagger.tools.speaker.metrics import (
    SpeakerMetricsConfig,
    merge_segments,
    normalize_segments,
    overlap_intervals,
    speaker_change_points_from_segments,
    union_intervals,
)


TIMELINE_DERIVATION_VERSION = "speaker_timeline_derivation_v2.0-shadow.2"


def summarize_timeline(segments, duration_sec, min_activity_sec=0.10):
    config = SpeakerMetricsConfig(
        min_segment_duration_sec=min_activity_sec,
        merge_same_speaker_gap_sec=0.30,
        speaker_change_max_gap_sec=1.00,
        overlap_min_duration_sec=0.10,
    )
    normalized = normalize_segments(segments, duration_sec, config)
    grouped = {}
    for segment in normalized:
        grouped.setdefault(segment["speaker_id"], []).append(segment)
    # Activity intervals preserve observed gaps.  A separate continuity view
    # may bridge short same-speaker gaps for turn/change reasoning, but must
    # never be used to infer simultaneous activity.
    activity_by_speaker = {
        speaker_id: merge_segments(items, 0.0, duration_sec)
        for speaker_id, items in grouped.items()
    }
    continuity_by_speaker = {
        speaker_id: merge_segments(items, 0.30, duration_sec)
        for speaker_id, items in grouped.items()
    }
    activity_segments = _flatten_speaker_segments(activity_by_speaker)
    continuity_segments = _flatten_speaker_segments(continuity_by_speaker)
    speech_union = union_intervals(activity_segments, duration_sec)
    overlap_activity = overlap_intervals(
        activity_by_speaker, duration_sec, 0.0
    )
    overlaps = [
        item
        for item in overlap_activity
        if float(item["end_sec"]) - float(item["start_sec"]) >= 0.10
    ]
    change_candidates = speaker_change_points_from_segments(
        continuity_segments, 1.00
    )
    per_speaker_duration = {
        speaker_id: round(
            sum(item["end_sec"] - item["start_sec"] for item in items), 6
        )
        for speaker_id, items in activity_by_speaker.items()
    }
    material_speaker_set = {
        speaker_id
        for speaker_id, activity in per_speaker_duration.items()
        if activity >= min_activity_sec
    }
    # Public speaker indices follow first speech onset, independently of the
    # model's native cluster labels (for example SPEAKER_02 before SPEAKER_00).
    material_speakers = []
    for segment in activity_segments:
        speaker_id = segment["speaker_id"]
        if (
            speaker_id in material_speaker_set
            and speaker_id not in material_speakers
        ):
            material_speakers.append(speaker_id)
    return {
        "derivation_version": TIMELINE_DERIVATION_VERSION,
        # Retained for compatibility with speaker-text and change consumers.
        "segments": continuity_segments,
        # Canonical observed activity, with no inferred activity inside gaps.
        "activity_segments": activity_segments,
        "speaker_ids": material_speakers,
        "observed_speaker_count": len(material_speakers),
        "per_speaker_activity_sec": per_speaker_duration,
        "speech_union_segments": speech_union,
        "speech_union_duration_sec": round(_sum_duration(speech_union), 6),
        # overlap_activity_segments is unthresholded so downstream clean-region
        # selection can remove even overlap events below the reporting floor.
        "overlap_activity_segments": overlap_activity,
        "overlap_segments": overlaps,
        "overlap_duration_sec": round(_sum_duration(overlaps), 6),
        "overlap_observed": bool(overlaps),
        # These are onset candidates from a deterministic timeline projection.
        # They are not yet certified floor-transfer events.
        "change_candidate_points_sec": [
            round(float(item), 6) for item in change_candidates
        ],
        "change_observed": bool(change_candidates),
    }


def _flatten_speaker_segments(segments_by_speaker):
    flattened = []
    for speaker_id, items in segments_by_speaker.items():
        for item in items:
            record = dict(item)
            record["speaker_id"] = speaker_id
            flattened.append(record)
    flattened.sort(
        key=lambda item: (
            item["start_sec"],
            item["end_sec"],
            item["speaker_id"],
        )
    )
    return flattened


def timeline_comparison(timeline_a, timeline_b, collar_sec=0.25):
    overlap_a = timeline_a.get("overlap_segments", [])
    overlap_b = timeline_b.get("overlap_segments", [])
    return {
        "speaker_count_equal": (
            timeline_a.get("observed_speaker_count")
            == timeline_b.get("observed_speaker_count")
        ),
        "speaker_count_delta": (
            int(timeline_a.get("observed_speaker_count", 0))
            - int(timeline_b.get("observed_speaker_count", 0))
        ),
        "overlap_bool_equal": bool(overlap_a) == bool(overlap_b),
        "overlap_event_iou": interval_set_iou(overlap_a, overlap_b),
        "change_candidate_match_ratio": point_match_ratio(
            timeline_a.get("change_candidate_points_sec", []),
            timeline_b.get("change_candidate_points_sec", []),
            collar_sec,
        ),
        "collar_sec": float(collar_sec),
    }


def interval_set_iou(items_a, items_b):
    union_a = _interval_union(items_a)
    union_b = _interval_union(items_b)
    intersection = 0.0
    for start_a, end_a in union_a:
        for start_b, end_b in union_b:
            intersection += max(0.0, min(end_a, end_b) - max(start_a, start_b))
    duration_a = sum(end - start for start, end in union_a)
    duration_b = sum(end - start for start, end in union_b)
    denominator = duration_a + duration_b - intersection
    if denominator <= 0:
        return 1.0 if not union_a and not union_b else 0.0
    return round(intersection / denominator, 6)


def point_match_ratio(points_a, points_b, collar_sec):
    points_a = sorted(float(item) for item in points_a)
    points_b = sorted(float(item) for item in points_b)
    if not points_a and not points_b:
        return 1.0
    if not points_a or not points_b:
        return 0.0
    available = list(points_b)
    matches = 0
    for point in points_a:
        candidates = [
            (abs(point - other), index)
            for index, other in enumerate(available)
            if abs(point - other) <= collar_sec
        ]
        if not candidates:
            continue
        _distance, index = min(candidates)
        available.pop(index)
        matches += 1
    return round((2.0 * matches) / (len(points_a) + len(points_b)), 6)


def _interval_union(items):
    parsed = sorted(
        (
            float(item["start_sec"]),
            float(item["end_sec"]),
        )
        for item in items
        if float(item["end_sec"]) > float(item["start_sec"])
    )
    merged = []
    for start, end in parsed:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _sum_duration(items):
    return sum(
        max(0.0, float(item["end_sec"]) - float(item["start_sec"]))
        for item in items
    )
