"""Speaker-aware FireRed LID on longest clean segments for multi-speaker recordings."""

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

from tagger.tools.base import ToolResult
from tagger.tools.language_content.firered_lid_detector import (
    FireRedLidConfig,
    run as run_firered_lid,
)


TOOL_NAME = "firered_lid_detector"
TOOL_VERSION = "speaker_segment_firered_lid.2"
METHOD = "FireRed LID (longest clean speaker speech segment)"
TARGET_SAMPLE_RATE_HZ = 16000
FFMPEG_TIMEOUT_SEC = 300


class SpeakerSegmentLidConfig:
    """Optional speaker-aware LID. Disabled by default; full-audio FireRed LID is used."""

    def __init__(
        self,
        enabled=False,
        mode="multi_speaker_only",
        language_only_start_speaker_stage=True,
        min_speaker_duration_sec=2.0,
        min_total_duration_sec=3.0,
        fallback_to_full_audio=True,
        shadow_only=False,
        keep_artifacts=False,
    ):
        if mode != "multi_speaker_only":
            raise ValueError("speaker-segment LID mode must be multi_speaker_only")
        min_speaker_duration_sec = float(min_speaker_duration_sec)
        min_total_duration_sec = float(min_total_duration_sec)
        if not math.isfinite(min_speaker_duration_sec) or min_speaker_duration_sec <= 0:
            raise ValueError("min_speaker_duration_sec must be positive")
        if not math.isfinite(min_total_duration_sec) or min_total_duration_sec <= 0:
            raise ValueError("min_total_duration_sec must be positive")
        self.enabled = bool(enabled)
        self.mode = mode
        self.language_only_start_speaker_stage = bool(
            language_only_start_speaker_stage
        )
        self.min_speaker_duration_sec = min_speaker_duration_sec
        self.min_total_duration_sec = min_total_duration_sec
        self.fallback_to_full_audio = bool(fallback_to_full_audio)
        self.shadow_only = bool(shadow_only)
        self.keep_artifacts = bool(keep_artifacts)

    def to_record(self):
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "language_only_start_speaker_stage": (
                self.language_only_start_speaker_stage
            ),
            "min_speaker_duration_sec": self.min_speaker_duration_sec,
            "min_total_duration_sec": self.min_total_duration_sec,
            "timeline_source": "selected_speaker_v2_timeline",
            "speaker_audio_selection": (
                "remove_overlap_then_select_longest_speech_segment"
            ),
            "fallback_to_full_audio": self.fallback_to_full_audio,
            "shadow_only": self.shadow_only,
            "keep_artifacts": self.keep_artifacts,
        }


def run(
    audio_path,
    selected_timeline,
    full_audio_result,
    context=None,
    config=None,
    lid_config=None,
    client=None,
    artifact_dir=None,
    sample_id=None,
):
    """Run FireRed LID once per eligible speaker and publish one language."""

    config = config or SpeakerSegmentLidConfig()
    lid_config = lid_config or FireRedLidConfig()
    baseline = _baseline_result(full_audio_result)
    evidence = {
        "config": config.to_record(),
        "full_audio_result": baseline,
        "timeline": _timeline_provenance(selected_timeline),
        "speaker_results": [],
        "overlap_exclusion": {
            "applied": False,
            "intervals": [],
        },
        "language_duration_sec": {},
        "proposed_language": None,
        "decision": "full_audio_fallback",
        "fallback_reason": None,
    }

    if not config.enabled:
        return _fallback_result(baseline, evidence, "disabled", config)

    intervals_by_speaker, overlap_intervals = normalize_timeline(
        selected_timeline,
        return_overlap=True,
    )
    evidence["overlap_exclusion"] = {
        "applied": bool(overlap_intervals),
        "intervals": [
            {"start_sec": round(start, 6), "end_sec": round(end, 6)}
            for start, end in overlap_intervals
        ],
        "policy": "remove_overlap_then_select_longest_speech_segment",
    }
    durations = {
        speaker_id: _interval_duration(intervals)
        for speaker_id, intervals in intervals_by_speaker.items()
    }
    eligible = {
        speaker_id: intervals
        for speaker_id, intervals in intervals_by_speaker.items()
        if durations[speaker_id] + 1e-9 >= config.min_speaker_duration_sec
    }
    evidence["speaker_activity_sec"] = {
        key: round(value, 6) for key, value in sorted(durations.items())
    }
    evidence["eligible_speaker_ids"] = sorted(eligible)
    if len(eligible) < 2:
        return _fallback_result(
            baseline, evidence, "fewer_than_two_eligible_speakers", config
        )

    total_duration = sum(durations[speaker_id] for speaker_id in eligible)
    evidence["eligible_total_activity_sec"] = round(total_duration, 6)
    if total_duration + 1e-9 < config.min_total_duration_sec:
        return _fallback_result(
            baseline, evidence, "insufficient_total_speaker_audio", config
        )

    workspace_owner = None
    if config.keep_artifacts:
        workspace = _kept_artifact_dir(artifact_dir, sample_id)
    else:
        workspace_owner = tempfile.TemporaryDirectory(prefix="speaker-segment-lid-")
        workspace = Path(workspace_owner.name)

    try:
        pcm_frames, sample_rate_hz = _decode_pcm16_mono(audio_path)
        speaker_results = []
        for index, speaker_id in enumerate(sorted(eligible), start=1):
            segment_path = workspace / ("speaker_%03d.wav" % index)
            extracted_duration = _write_selected_segment_wav(
                segment_path,
                pcm_frames,
                sample_rate_hz,
                eligible[speaker_id],
            )
            try:
                result = run_firered_lid(
                    segment_path,
                    context=context,
                    config=lid_config,
                    client=client,
                )
            except Exception as exc:  # noqa: BLE001 - any missing vote fails closed.
                evidence["speaker_results"].append(
                    {
                        "speaker_id": speaker_id,
                        "activity_sec": round(durations[speaker_id], 6),
                        "extracted_duration_sec": round(extracted_duration, 6),
                        "status": "failed",
                        "error": "%s: %s" % (exc.__class__.__name__, exc),
                    }
                )
                return _fallback_result(
                    baseline, evidence, "speaker_lid_failed", config
                )
            item = {
                "speaker_id": speaker_id,
                "activity_sec": round(durations[speaker_id], 6),
                "extracted_duration_sec": round(extracted_duration, 6),
                "language": result.value,
                "confidence": result.confidence,
                "status": result.status,
            }
            if config.keep_artifacts:
                item["audio_path"] = str(segment_path)
            evidence["speaker_results"].append(item)
            speaker_results.append(item)

        vote = aggregate_language_votes(speaker_results)
        evidence["language_duration_sec"] = vote["language_duration_sec"]
        evidence["proposed_language"] = vote["language"]
        if vote["tie"]:
            return _fallback_result(
                baseline, evidence, "duration_weighted_vote_tie", config
            )

        evidence["decision"] = (
            "shadow_full_audio" if config.shadow_only else "longest_clean_segment"
        )
        evidence["fallback_reason"] = None
        if config.shadow_only:
            return _tool_result(
                baseline.get("language"),
                baseline.get("confidence"),
                evidence,
                "FireRed LID (longest clean segment shadow; full audio published)",
            )
        return _tool_result(
            vote["language"],
            vote["confidence"],
            evidence,
            METHOD,
        )
    except Exception as exc:  # noqa: BLE001 - segment preparation falls back safely.
        evidence["segment_error"] = "%s: %s" % (exc.__class__.__name__, exc)
        return _fallback_result(
            baseline, evidence, "speaker_audio_extraction_failed", config
        )
    finally:
        if workspace_owner is not None:
            workspace_owner.cleanup()


def normalize_timeline(selected_timeline, return_overlap=False):
    """Return one clean, longest speech interval per speaker.

    Overlap intervals are removed before selecting the longest interval.  The
    default return shape remains a speaker-to-intervals mapping for callers
    that only need the normalized timeline; ``return_overlap=True`` also
    returns the intervals removed from every speaker.
    """

    if not isinstance(selected_timeline, dict):
        return ({}, []) if return_overlap else {}
    raw_segments = selected_timeline.get("activity_segments")
    if not isinstance(raw_segments, list):
        raw_segments = selected_timeline.get("segments")
    if not isinstance(raw_segments, list):
        return ({}, []) if return_overlap else {}
    duration_sec = _finite_number(selected_timeline.get("duration_sec"))
    grouped = {}
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        start = _finite_number(item.get("start_sec"))
        end = _finite_number(item.get("end_sec"))
        if not speaker_id or start is None or end is None:
            continue
        start = max(0.0, start)
        if duration_sec is not None:
            end = min(duration_sec, end)
        if end <= start:
            continue
        grouped.setdefault(str(speaker_id), []).append((start, end))
    merged = {
        speaker_id: _merge_intervals(intervals)
        for speaker_id, intervals in sorted(grouped.items())
    }
    overlap_intervals = _timeline_overlap_intervals(selected_timeline, merged)
    clean = {}
    for speaker_id, intervals in merged.items():
        non_overlap = _subtract_intervals(intervals, overlap_intervals)
        longest = _longest_interval(non_overlap)
        clean[speaker_id] = [longest] if longest is not None else []
    if return_overlap:
        return clean, overlap_intervals
    return clean


def _timeline_overlap_intervals(selected_timeline, merged_by_speaker):
    """Return explicit or derived overlap intervals, clipped to the timeline."""

    duration_sec = _finite_number(
        selected_timeline.get("duration_sec")
        if isinstance(selected_timeline, dict)
        else None
    )
    supplied = []
    if isinstance(selected_timeline, dict):
        raw = selected_timeline.get("overlap_activity_segments")
        if not isinstance(raw, list):
            raw = selected_timeline.get("overlap_segments")
        if isinstance(raw, list):
            supplied = _parse_intervals(raw, duration_sec)
    if supplied:
        return _merge_intervals(supplied)

    derived = []
    speaker_items = sorted(merged_by_speaker.items())
    for index, (_left_speaker, left_intervals) in enumerate(speaker_items):
        for _right_speaker, right_intervals in speaker_items[index + 1 :]:
            for left_start, left_end in left_intervals:
                for right_start, right_end in right_intervals:
                    start = max(left_start, right_start)
                    end = min(left_end, right_end)
                    if end > start:
                        derived.append((start, end))
    return _merge_intervals(derived)


def _parse_intervals(raw_intervals, duration_sec=None):
    parsed = []
    for item in raw_intervals:
        if isinstance(item, dict):
            start = _finite_number(item.get("start_sec"))
            end = _finite_number(item.get("end_sec"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = _finite_number(item[0])
            end = _finite_number(item[1])
        else:
            continue
        if start is None or end is None:
            continue
        start = max(0.0, start)
        if duration_sec is not None:
            end = min(duration_sec, end)
        if end > start:
            parsed.append((start, end))
    return parsed


def _subtract_intervals(intervals, exclusions):
    result = []
    for start, end in intervals:
        pieces = [(start, end)]
        for cut_start, cut_end in exclusions:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    next_pieces.append((piece_start, min(cut_start, piece_end)))
                if cut_end < piece_end:
                    next_pieces.append((max(cut_end, piece_start), piece_end))
            pieces = next_pieces
        result.extend(
            (piece_start, piece_end)
            for piece_start, piece_end in pieces
            if piece_end > piece_start
        )
    return result


def _longest_interval(intervals):
    if not intervals:
        return None
    return max(
        intervals,
        key=lambda item: (item[1] - item[0], -item[0]),
    )


def aggregate_language_votes(speaker_results):
    """Select a language by the durations of the selected clean segments."""

    duration_by_language = {}
    confidence_by_language = {}
    for item in speaker_results:
        language = item.get("language")
        duration = _finite_number(item.get("activity_sec"))
        if not isinstance(language, str) or not language.strip() or duration is None:
            continue
        language = language.strip()
        duration_by_language[language] = (
            duration_by_language.get(language, 0.0) + duration
        )
        confidence = _finite_number(item.get("confidence"))
        if confidence is not None:
            weighted_sum, weight = confidence_by_language.get(language, (0.0, 0.0))
            confidence_by_language[language] = (
                weighted_sum + confidence * duration,
                weight + duration,
            )
    rounded = {
        key: round(value, 6) for key, value in sorted(duration_by_language.items())
    }
    if not duration_by_language:
        return {
            "language": None,
            "confidence": None,
            "tie": True,
            "language_duration_sec": rounded,
        }
    maximum = max(duration_by_language.values())
    winners = sorted(
        language
        for language, duration in duration_by_language.items()
        if abs(duration - maximum) <= 1e-9
    )
    if len(winners) != 1:
        return {
            "language": None,
            "confidence": None,
            "tie": True,
            "language_duration_sec": rounded,
        }
    language = winners[0]
    confidence = None
    if language in confidence_by_language:
        weighted_sum, weight = confidence_by_language[language]
        if weight > 0:
            confidence = round(weighted_sum / weight, 6)
    return {
        "language": language,
        "confidence": confidence,
        "tie": False,
        "language_duration_sec": rounded,
    }


def _baseline_result(value):
    value = value if isinstance(value, dict) else {}
    language = value.get("language")
    if not isinstance(language, str) or not language.strip():
        language = None
    else:
        language = language.strip()
    confidence = _finite_number(value.get("confidence"))
    return {
        "language": language,
        "confidence": confidence,
        "evidence_id": value.get("evidence_id"),
        "language_route": value.get("language_route"),
        "error": value.get("error"),
    }


def _fallback_result(baseline, evidence, reason, config):
    evidence["decision"] = "full_audio_fallback"
    evidence["fallback_reason"] = reason
    language = baseline.get("language") if config.fallback_to_full_audio else None
    confidence = baseline.get("confidence") if config.fallback_to_full_audio else None
    return _tool_result(
        language,
        confidence,
        evidence,
        "FireRed LID (full audio fallback)",
    )


def _tool_result(language, confidence, evidence, method):
    return ToolResult(
        tag_path="language_content.language",
        value=language,
        tool_name=TOOL_NAME,
        method=method,
        status="estimated" if language is not None else "failed",
        confidence=confidence,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence=evidence,
    )


def _timeline_provenance(selected_timeline):
    if not isinstance(selected_timeline, dict):
        return None
    return {
        "source": selected_timeline.get("source"),
        "evidence_id": selected_timeline.get("evidence_id"),
        "duration_sec": selected_timeline.get("duration_sec"),
    }


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_duration(intervals):
    return sum(end - start for start, end in intervals)


def _decode_pcm16_mono(audio_path):
    path = Path(audio_path)
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getcomptype() == "NONE"
                and source.getnchannels() == 1
                and source.getsampwidth() == 2
                and source.getframerate() == TARGET_SAMPLE_RATE_HZ
            ):
                return source.readframes(source.getnframes()), source.getframerate()
    except (OSError, EOFError, wave.Error):
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode this audio format")
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE_HZ),
        "-f",
        "s16le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FFMPEG_TIMEOUT_SEC,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace")[-1000:]
        raise RuntimeError("ffmpeg failed to decode source audio: %s" % detail)
    return completed.stdout, TARGET_SAMPLE_RATE_HZ


def _write_selected_segment_wav(path, pcm_frames, sample_rate_hz, intervals):
    if len(intervals) != 1:
        raise RuntimeError(
            "speaker-aware LID requires exactly one selected speech interval"
        )
    bytes_per_frame = 2
    frame_count = len(pcm_frames) // bytes_per_frame
    start, end = intervals[0]
    start_frame = min(frame_count, max(0, int(round(start * sample_rate_hz))))
    end_frame = min(frame_count, max(start_frame, int(round(end * sample_rate_hz))))
    selected_frames = end_frame - start_frame
    if selected_frames <= 0:
        raise RuntimeError("selected speaker speech segment is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(bytes_per_frame)
        sink.setframerate(sample_rate_hz)
        sink.writeframes(
            pcm_frames[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
        )
    return selected_frames / float(sample_rate_hz)


def _kept_artifact_dir(artifact_dir, sample_id):
    if artifact_dir is None:
        raise ValueError("artifact_dir is required when keep_artifacts is enabled")
    sample_key = str(sample_id or "sample")
    safe_key = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in sample_key
    )
    path = Path(artifact_dir) / "language_segment_lid" / safe_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None
