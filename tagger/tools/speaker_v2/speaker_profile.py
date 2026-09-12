"""Speaker-local acoustic profile calculations for speaker-v2.

The phase 0/1 adapter deliberately does not load a neural model.  It consumes
the selected diarization timeline, existing speech-coverage evidence and text
evidence, then emits a small public profile plus auditable internal metrics.
"""

from __future__ import division

import math
import re
import shutil
import subprocess
import sys
import wave
from array import array


PROFILE_SCHEMA_VERSION = "speaker_v2.speaker_profile.v0.2"
MIN_RATE_DURATION_SEC = 3.0
MIN_RATE_UNITS = 8
MIN_ACOUSTIC_DURATION_SEC = 0.8
DEFAULT_COMPRESSED_SAMPLE_RATE_HZ = 16000
FFMPEG_TIMEOUT_SEC = 120

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*")
_FILLER_RE = re.compile(r"^(?:嗯+|呃+|啊+|um+|uh+|er+)$", re.I)


def compute_speaker_profiles(
    decision_timeline,
    coverage_evidence=None,
    text_evidence=None,
    audio_path=None,
    duration_sec=None,
    sample_rate_hz=None,
):
    """Return ``{"profiles": ..., "details": ...}`` for one timeline.

    ``decision_timeline`` is an evidence record, not a raw model response.
    ``profiles`` is ``None`` when no usable decision timeline exists and an
    empty list when a valid timeline contains no speech.  Public values are
    intentionally conservative; raw acoustic statistics stay in ``details``.
    """

    if not _usable_timeline(decision_timeline):
        return {"profiles": None, "details": {"status": "no_timeline"}}
    summary = (decision_timeline.get("payload") or {}).get(
        "timeline_summary", {}
    )
    segments = _timeline_segments(summary)
    if not segments:
        return {"profiles": [], "details": {"status": "no_speech"}}

    overlap = _intervals(
        summary.get("overlap_activity_segments")
        or summary.get("overlap_segments")
        or []
    )
    speech = _coverage_intervals(coverage_evidence)
    if speech is None:
        speech = _merge_intervals(
            [_pair(segment["start_sec"], segment["end_sec"]) for segment in segments]
        )
    else:
        speech = _merge_intervals(speech)
    clean_by_speaker = {}
    ordered_ids = []
    for segment in segments:
        speaker = segment["speaker_id"]
        if speaker not in clean_by_speaker:
            clean_by_speaker[speaker] = []
            ordered_ids.append(speaker)
        base = [(segment["start_sec"], segment["end_sec"])]
        base = _subtract_intervals(base, overlap)
        base = _intersect_intervals(base, speech)
        clean_by_speaker[speaker].extend(base)
    clean_by_speaker = {
        speaker: _merge_intervals(intervals)
        for speaker, intervals in clean_by_speaker.items()
    }
    if not any(clean_by_speaker.values()):
        return {"profiles": [], "details": {"status": "no_speech"}}

    text_segments = _collect_text_segments(decision_timeline, text_evidence)
    text_by_speaker = _assign_text_segments(text_segments, segments)
    profiles = []
    acoustic_values = []
    details = {
        "status": "observed",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "timeline_evidence_id": decision_timeline.get("evidence_id"),
        "speakers": [],
        "overlap_excluded": bool(overlap),
        "rate_basis": "text_units_over_clean_speech_active_seconds",
        "rate_band_thresholds": {
            "cjk_char_per_sec": {"slow_below": 3.0, "fast_above": 6.0},
            "word_per_min": {"slow_below": 100.0, "fast_above": 170.0},
        },
    }
    for index, speaker in enumerate(ordered_ids, 1):
        intervals = clean_by_speaker.get(speaker, [])
        active_duration = _duration(intervals)
        rate = _speech_rate(text_by_speaker.get(speaker, []), intervals)
        acoustic = _acoustic_profile(
            intervals,
            audio_path,
            sample_rate_hz=sample_rate_hz,
        )
        profile = {
            "speaker_id": "speaker_%d" % index,
            "gender": None,
            "speech_rate": rate,
            "pitch": acoustic.get("pitch"),
            "speaker_volume": acoustic.get("speaker_volume"),
        }
        profiles.append(profile)
        acoustic_values.append(acoustic)
        details["speakers"].append(
            {
                "speaker_id": profile["speaker_id"],
                "source_speaker_id": str(speaker),
                "clean_speech_duration_sec": round(active_duration, 6),
                "text_unit_count": rate.pop("_unit_count", 0),
                "text_duration_sec": rate.pop("_duration_sec", 0.0),
                "rate_language": rate.pop("_language", None),
                "rate_segments": rate.pop("_segments", 0),
                "pitch_hz_median": acoustic.get("pitch_hz_median"),
                "pitch_voiced_ratio": acoustic.get("pitch_voiced_ratio"),
                "rms_db_relative": acoustic.get("rms_db_relative"),
                "acoustic_duration_sec": acoustic.get("duration_sec", 0.0),
                "acoustic_reason": acoustic.get("reason"),
            }
        )
    _assign_relative_volume(profiles, acoustic_values, details)
    return {"profiles": profiles, "details": details}


def _usable_timeline(evidence):
    return bool(
        isinstance(evidence, dict)
        and "speaker_timeline" in evidence.get("capabilities", [])
        and evidence.get("status") in ("observed", "estimated")
        and evidence.get("quality", {}).get("usable", True)
    )


def _timeline_segments(summary):
    if not isinstance(summary, dict):
        return []
    # activity_segments preserves observed speech boundaries. ``segments`` is
    # a continuity view that may bridge short gaps for change detection.
    raw = summary.get("activity_segments") or summary.get("segments") or []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker_id")
        try:
            start = max(0.0, float(item.get("start_sec")))
            end = max(0.0, float(item.get("end_sec")))
        except (TypeError, ValueError):
            continue
        if not speaker or end <= start:
            continue
        result.append(
            {
                "speaker_id": str(speaker),
                "start_sec": start,
                "end_sec": end,
                "text": str(item.get("text", "") or "").strip(),
            }
        )
    return sorted(result, key=lambda item: (item["start_sec"], item["end_sec"]))


def _coverage_intervals(evidence):
    if not evidence:
        return None
    candidates = evidence if isinstance(evidence, (list, tuple)) else [evidence]
    usable = []
    saw_usable = False
    for item in candidates:
        if not isinstance(item, dict) or item.get("status") not in ("observed", "estimated"):
            continue
        if not item.get("quality", {}).get("usable", True):
            continue
        saw_usable = True
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        raw = payload.get("speech_segments") or payload.get("segments") or []
        usable.extend(_intervals(raw))
    return usable if saw_usable else None


def _intervals(raw):
    result = []
    for item in raw or []:
        if isinstance(item, dict):
            start, end = item.get("start_sec"), item.get("end_sec")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end > start:
            result.append((max(0.0, start), end))
    return result


def _pair(start, end):
    return (float(start), float(end))


def _merge_intervals(intervals):
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if end > start
    )
    result = []
    for start, end in ordered:
        if result and start <= result[-1][1] + 1e-6:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _intersect_intervals(left, right):
    result = []
    for start, end in left:
        for other_start, other_end in right:
            overlap_start = max(start, other_start)
            overlap_end = min(end, other_end)
            if overlap_end > overlap_start:
                result.append((overlap_start, overlap_end))
    return _merge_intervals(result)


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


def _duration(intervals):
    return sum(max(0.0, end - start) for start, end in intervals)


def _collect_text_segments(decision_timeline, text_evidence):
    result = []
    summary = (decision_timeline.get("payload") or {}).get("timeline_summary", {})
    for item in _timeline_segments(summary):
        if item.get("text"):
            result.append(item)
    if result:
        return result
    joint_text = []
    for evidence in text_evidence or []:
        payload = evidence.get("payload", {})
        source_summary = payload.get("timeline_summary", {})
        for item in _timeline_segments(source_summary):
            if item.get("text"):
                joint_text.append(item)
    if joint_text:
        return joint_text
    lexical_text = []
    for evidence in text_evidence or []:
        payload = evidence.get("payload", {})
        for item in payload.get("lexical_units", []) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", item.get("token", "")) or "").strip()
            try:
                start, end = float(item.get("start_sec")), float(item.get("end_sec"))
            except (TypeError, ValueError):
                continue
            if text and end > start:
                lexical_text.append(
                    {"start_sec": start, "end_sec": end, "text": text}
                )
    return lexical_text


def _assign_text_segments(text_segments, timeline_segments):
    assigned = {}
    for text in text_segments:
        text_start, text_end = text["start_sec"], text["end_sec"]
        candidates = []
        for segment in timeline_segments:
            overlap = min(text_end, segment["end_sec"]) - max(text_start, segment["start_sec"])
            if overlap > 0:
                candidates.append((overlap, segment["speaker_id"]))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) <= 1e-6:
            continue
        speaker = candidates[0][1]
        assigned.setdefault(speaker, []).append(text)
    return assigned


def _speech_rate(text_segments, intervals):
    unit_count = 0
    active_duration = 0.0
    rates = []
    languages = []
    for text in text_segments:
        units, language = _count_text_units(text.get("text", ""))
        if not units:
            continue
        text_intervals = _intersect_intervals(
            [(text["start_sec"], text["end_sec"])], intervals
        )
        seconds = _duration(text_intervals)
        if seconds <= 0:
            continue
        unit_count += units
        active_duration += seconds
        rates.append((units / seconds, units, seconds))
        languages.append(language)
    language = _dominant_language(languages)
    result = {
        "band": None,
        "value": None,
        "unit": None,
        "_unit_count": unit_count,
        "_duration_sec": round(active_duration, 6),
        "_language": language,
        "_segments": len(rates),
    }
    if active_duration < MIN_RATE_DURATION_SEC or unit_count < MIN_RATE_UNITS:
        return result
    value = unit_count / active_duration
    if language == "cjk":
        unit = "zh_char_per_sec"
        band = "slow" if value < 3.0 else "fast" if value > 6.0 else "normal"
        public_value = round(value, 2)
    else:
        unit = "word_per_min"
        value *= 60.0
        band = "slow" if value < 100.0 else "fast" if value > 170.0 else "normal"
        public_value = round(value, 1)
    if _is_variable_rate(rates, language):
        band = "variable"
    result.update({"band": band, "value": public_value, "unit": unit})
    return result


def _count_text_units(text):
    text = str(text or "")
    cjk = _CJK_RE.findall(text)
    latin = [item for item in _LATIN_WORD_RE.findall(text) if not _FILLER_RE.match(item)]
    if len(cjk) >= max(1, len(latin)):
        return len(cjk), "cjk"
    return len(latin), "latin"


def _dominant_language(languages):
    if not languages:
        return None
    return "cjk" if languages.count("cjk") >= languages.count("latin") else "latin"


def _is_variable_rate(rates, language):
    if len(rates) < 2:
        return False
    values = [item[0] * (60.0 if language != "cjk" else 1.0) for item in rates]
    low, high = min(values), max(values)
    return low > 0 and high / low >= 1.8 and all(item[1] >= 4 for item in rates)


def _acoustic_profile(intervals, audio_path, sample_rate_hz=None):
    empty = {
        "pitch": None,
        "speaker_volume": None,
        "pitch_hz_median": None,
        "pitch_voiced_ratio": None,
        "rms_db_relative": None,
        "duration_sec": 0.0,
        "reason": "audio_unavailable",
    }
    if not audio_path or _duration(intervals) < MIN_ACOUSTIC_DURATION_SEC:
        empty["reason"] = "insufficient_clean_audio"
        return empty
    samples, rate = _read_interval_samples(
        audio_path, _limit_intervals(intervals, 30.0), sample_rate_hz
    )
    if not samples or not rate:
        return empty
    if rate < 12000:
        empty["reason"] = "low_sample_rate"
        return empty
    rms = _rms(samples)
    if rms <= 1e-7:
        empty["reason"] = "silent_audio"
        return empty
    f0_values, voiced_ratio = _estimate_f0(samples, rate)
    result = dict(empty)
    result["duration_sec"] = round(len(samples) / float(rate), 6)
    result["rms_db_relative"] = round(20.0 * math.log10(max(rms, 1e-7)), 3)
    result["pitch_voiced_ratio"] = round(voiced_ratio, 6)
    window_size = max(1, int(rate * 0.5))
    window_rms = [
        _rms(samples[offset : offset + window_size])
        for offset in range(0, len(samples), window_size)
        if samples[offset : offset + window_size]
    ]
    window_db = [
        20.0 * math.log10(max(window_value, 1e-7))
        for window_value in window_rms
        if window_value > 1e-7
    ]
    if len(window_db) >= 2 and max(window_db) - min(window_db) >= 6.0:
        result["speaker_volume"] = "variable"
    else:
        result["speaker_volume"] = "normal"
    if f0_values and voiced_ratio >= 0.2:
        median = _median(f0_values)
        result["pitch_hz_median"] = round(median, 3)
        result["pitch"] = _pitch_band(median)
        result["reason"] = None
    else:
        result["reason"] = "insufficient_voiced_frames"
    return result


def _pitch_band(median):
    return "low" if median < 140 else "high" if median > 220 else "mid"


def _assign_relative_volume(profiles, acoustic_values, details):
    values = [
        item.get("rms_db_relative")
        for item in acoustic_values
        if item.get("rms_db_relative") is not None
    ]
    if not values:
        return
    reference = _median(values)
    for profile, acoustic in zip(profiles, acoustic_values):
        value = acoustic.get("rms_db_relative")
        if value is None:
            profile["speaker_volume"] = None
            continue
        if acoustic.get("speaker_volume") == "variable":
            profile["speaker_volume"] = "variable"
            continue
        delta = value - reference
        profile["speaker_volume"] = (
            "low" if delta <= -3.0 else "loud" if delta >= 3.0 else "normal"
        )
    details["volume_basis"] = "within_clip_median_rms_db"


def _limit_intervals(intervals, maximum_duration_sec):
    remaining = float(maximum_duration_sec)
    result = []
    for start, end in intervals:
        if remaining <= 0:
            break
        length = min(end - start, remaining)
        if length > 0:
            result.append((start, start + length))
            remaining -= length
    return result


def _read_interval_samples(path, intervals, sample_rate_hz=None):
    try:
        source = wave.open(str(path), "rb")
    except Exception:
        return _read_compressed_interval_samples(path, intervals, sample_rate_hz)
    try:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frame_count = source.getnframes()
        if channels < 1 or width not in (1, 2, 3, 4) or rate <= 0:
            return [], None
        samples = []
        for start, end in intervals:
            start_frame = min(frame_count, max(0, int(start * rate)))
            end_frame = min(frame_count, max(start_frame, int(end * rate)))
            source.setpos(start_frame)
            raw = source.readframes(max(0, end_frame - start_frame))
            samples.extend(_decode_pcm(raw, width, channels))
        return samples, int(sample_rate_hz or rate)
    finally:
        source.close()


def _read_compressed_interval_samples(path, intervals, sample_rate_hz=None):
    """Decode compressed audio through ffmpeg and return mono float samples."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [], None
    rate = int(sample_rate_hz or DEFAULT_COMPRESSED_SAMPLE_RATE_HZ)
    if rate <= 0:
        return [], None
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [], None
    if completed.returncode != 0 or not completed.stdout:
        return [], None
    decoded = array("f")
    try:
        decoded.frombytes(completed.stdout)
    except (TypeError, ValueError):
        return [], None
    if sys.byteorder != "little":
        decoded.byteswap()
    frame_count = len(decoded)
    samples = []
    for start, end in intervals:
        start_frame = min(frame_count, max(0, int(start * rate)))
        end_frame = min(frame_count, max(start_frame, int(end * rate)))
        samples.extend(decoded[start_frame:end_frame])
    return samples, rate


def _decode_pcm(raw, width, channels):
    step = width * channels
    result = []
    for offset in range(0, len(raw) - step + 1, step):
        values = []
        for channel in range(channels):
            chunk = raw[offset + channel * width : offset + (channel + 1) * width]
            if width == 1:
                value = (chunk[0] - 128) / 128.0
            else:
                value = int.from_bytes(chunk, byteorder="little", signed=True)
                value /= float(1 << (8 * width - 1))
            values.append(value)
        result.append(sum(values) / len(values))
    return result


def _rms(samples):
    return math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))


def _estimate_f0(samples, rate):
    frame = max(160, int(rate * 0.03))
    step = max(80, int(rate * 0.01))
    values = []
    total = 0
    frame_starts = list(range(0, max(0, len(samples) - frame + 1), step))
    if len(frame_starts) > 200:
        stride = len(frame_starts) / 200.0
        frame_starts = [frame_starts[int(index * stride)] for index in range(200)]
    voiced = 0
    for start in frame_starts:
        chunk = samples[start : start + frame]
        total += 1
        energy = _rms(chunk)
        if energy < 0.01:
            continue
        lag_min = max(2, int(rate / 400.0))
        lag_max = min(frame - 2, int(rate / 50.0))
        best_lag, best_score = None, 0.0
        for lag in range(lag_min, lag_max + 1):
            score = sum(chunk[index] * chunk[index - lag] for index in range(lag, len(chunk)))
            score /= max(1, len(chunk) - lag)
            if score > best_score:
                best_lag, best_score = lag, score
        if best_lag is not None and best_score > energy * energy * 0.12:
            values.append(float(rate) / best_lag)
            voiced += 1
    return values, voiced / float(total or 1)


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


__all__ = ["PROFILE_SCHEMA_VERSION", "compute_speaker_profiles"]
