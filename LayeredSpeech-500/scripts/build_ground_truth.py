#!/usr/bin/env python3
"""Build strict, no-inference ground truth for the 500 rendered sessions.

Only values copied from explicit generation metadata, read from the rendered
WAV header, measured from the selected RIR, or calculated deterministically
from the render timeline are populated. Model outputs and descriptive
judgements are never used.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "data" / "scene_scripts_v4.jsonl"
OUTPUT_ROOT = Path(os.environ.get("RELEASE_OUTPUT_ROOT", ROOT / "runs/v4_multilingual"))
SESSION_PATH = OUTPUT_ROOT / "manifests" / "sessions.json"
RIR_METRICS_PATH = OUTPUT_ROOT / "qc" / "rir_metrics.jsonl"
GT_ROOT = OUTPUT_ROOT / "ground_truth"

FRAME_SNR_VERSION = "known_stems_brouhaha_detailed_snr.v2"
FRAME_SNR_DURATION_SEC = 2.0
FRAME_SNR_HOP_SEC = 0.01
FRAME_SNR_ACTIVE_THRESHOLD = 1e-6
FRAME_SNR_NORMALIZATION_EPSILON = 1e-8
FRAME_SNR_NOISE_POWER_EPSILON = 1e-3
FRAME_SNR_RATIO_FLOOR = 1e-10

STRUCTURAL_KEYS = (
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
    "language_content",
    "deterministic_derived",
)

DESCRIPTIVE_KEYS = (
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def round6(value: float | int | None) -> float | None:
    return None if value is None else round(float(value), 6)


def read_audio_header(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        duration = handle.getnframes() / float(sample_rate)
    return sample_rate, channels, duration


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError(f"frame SNR requires mono stem: {path}")
        if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
            raise ValueError(f"frame SNR requires PCM-16 stem: {path}")
        sample_rate = handle.getframerate()
        values = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float64)
    return values / 32768.0, sample_rate


def compute_frame_snr_ground_truth(sample_id: str) -> dict[str, Any]:
    """Apply Brouhaha-maker's detailed-SNR recipe to known render stems.

    Brouhaha is trained with 2 s / 10 ms sliding-window SNR targets. Speech is
    masked by VAD before the window energies are computed. The render already
    contains the final speech/noise scaling, so both components receive only a
    common normalization before applying the reference formula.
    """
    session_root = OUTPUT_ROOT / "sessions" / sample_id
    wet_paths = sorted((session_root / "stems").glob("speaker_*.wav"))
    dry_paths = sorted((session_root / "dry_stems").glob("speaker_*.wav"))
    noise_path = session_root / "stems" / "noise.wav"
    if not wet_paths or len(wet_paths) != len(dry_paths) or not noise_path.is_file():
        raise ValueError(f"missing or mismatched stems for {sample_id}")

    noise, sample_rate = read_pcm16_mono(noise_path)
    sample_count = noise.size
    clean = np.zeros(sample_count, dtype=np.float64)
    dry_active = np.zeros(sample_count, dtype=np.bool_)
    for wet_path, dry_path in zip(wet_paths, dry_paths):
        wet, wet_rate = read_pcm16_mono(wet_path)
        dry, dry_rate = read_pcm16_mono(dry_path)
        if wet_rate != sample_rate or dry_rate != sample_rate:
            raise ValueError(f"stem sample-rate mismatch for {sample_id}")
        if wet.size != sample_count or dry.size != sample_count:
            raise ValueError(f"stem length mismatch for {sample_id}")
        clean += wet
        dry_active |= np.abs(dry) > FRAME_SNR_ACTIVE_THRESHOLD

    if not np.any(dry_active):
        raise ValueError(f"no speech-active samples for {sample_id}")

    normalization = (
        float(np.sqrt(np.mean(np.square(clean[dry_active]))))
        + FRAME_SNR_NORMALIZATION_EPSILON
    )
    clean = (clean / normalization) * dry_active
    noise = noise / normalization

    window_samples = int(FRAME_SNR_DURATION_SEC * sample_rate)
    hop_samples = int(FRAME_SNR_HOP_SEC * sample_rate)
    starts = np.arange(0, sample_count - window_samples + 1, hop_samples)
    ends = np.arange(window_samples - 1, sample_count, hop_samples)
    if starts.size == 0 or starts.size != ends.size:
        raise ValueError(f"audio is shorter than one SNR window for {sample_id}")

    # Keep the same cumsum indexing and constants as Brouhaha-maker's
    # compute_detailed_snr implementation.
    clean_energy = np.cumsum(np.square(clean), dtype=np.float64)
    noise_energy = np.cumsum(np.square(noise), dtype=np.float64)
    speech_frame_energy = clean_energy[ends] - clean_energy[starts]
    noise_frame_energy = noise_energy[ends] - noise_energy[starts]
    signal_to_noise = np.maximum(
        speech_frame_energy
        / (noise_frame_energy + FRAME_SNR_NOISE_POWER_EPSILON),
        FRAME_SNR_RATIO_FLOOR,
    )
    frame_snr_db = 10.0 * np.log10(signal_to_noise)
    if not np.all(np.isfinite(frame_snr_db)):
        raise ValueError(f"non-finite frame SNR for {sample_id}")

    frame_count = int(frame_snr_db.size)
    return {
        "mean_db": round(float(np.mean(frame_snr_db)), 6),
        "aggregation": "arithmetic_mean_of_all_detailed_snr_windows_db",
        "frame_duration_sec": FRAME_SNR_DURATION_SEC,
        "frame_hop_sec": FRAME_SNR_HOP_SEC,
        "frame_grid_reference": "Brouhaha-maker compute_detailed_snr defaults",
        "valid_frame_policy": "all complete 2 s windows; no frame-level VAD filter",
        "valid_frame_count": frame_count,
        "total_frame_count": frame_count,
        "valid_frame_ratio": 1.0,
        "vad_policy": "mask post-RIR speech where every dry stem abs(sample) <= 1e-6",
        "normalization": "common speech/noise gain from post-RIR speech RMS on VAD-active samples",
        "normalization_epsilon": FRAME_SNR_NORMALIZATION_EPSILON,
        "noise_power_epsilon": FRAME_SNR_NOISE_POWER_EPSILON,
        "signal_to_noise_floor": FRAME_SNR_RATIO_FLOOR,
        "speech_source": "VAD-masked sum of post-RIR speaker stems",
        "background_source": "noise stem including music when configured",
        "version": FRAME_SNR_VERSION,
    }


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError(f"invalid render interval: {start}, {end}")
        if not merged or start > merged[-1][1] + 1e-7:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(round(float(start), 6), round(float(end), 6)) for start, end in merged]


def complement_intervals(
    duration: float, speech_intervals: list[tuple[float, float]]
) -> list[dict[str, float]]:
    """Return non-human-speech gaps from the final rendered turn timeline."""
    result: list[dict[str, float]] = []
    cursor = 0.0
    for raw_start, raw_end in speech_intervals:
        start = max(0.0, min(duration, raw_start))
        end = max(0.0, min(duration, raw_end))
        if start > cursor + 1e-7:
            result.append({"start_sec": round(cursor, 6), "end_sec": round(start, 6)})
        cursor = max(cursor, end)
    if cursor < duration - 1e-7:
        result.append({"start_sec": round(cursor, 6), "end_sec": round(duration, 6)})
    return result


def empty_descriptive() -> dict[str, None]:
    # Arrays are also null: the request is to provide no descriptive GT at all.
    return {key: None for key in DESCRIPTIVE_KEYS}


def sample_language(scene: dict[str, Any]) -> str | None:
    if scene.get("language_mode") != "monolingual":
        return None
    return scene["layers"]["L1_lexical_pragmatic"].get("language")


def languages_present(scene: dict[str, Any]) -> list[str]:
    explicit = scene.get("languages_present")
    if isinstance(explicit, list) and explicit:
        return sorted({str(value) for value in explicit if value})

    languages = {
        span.get("language")
        for turn in scene.get("script", {}).get("turns", [])
        for span in (turn.get("language_spans") or [])
        if span.get("language")
    }
    primary = scene["layers"]["L1_lexical_pragmatic"].get("language")
    if primary:
        languages.add(primary)
    if scene.get("language_mode") == "code_switch":
        languages.add("en")
    return sorted(languages)


def known_status(status: str, source: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "source": source, **extra}


def null_status(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def build_record(
    scene: dict[str, Any],
    session: dict[str, Any],
    rir_metric: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = session["session_id"]
    if scene.get("scene_id") != sample_id or rir_metric.get("scene_id") != sample_id:
        raise ValueError(f"source alignment mismatch for {sample_id}")

    audio_path = OUTPUT_ROOT / session["audio_path"]
    sample_rate, channels, wav_duration = read_audio_header(audio_path)
    manifest_duration = float(session["duration_seconds"])
    if sample_rate != int(session["sample_rate"]):
        raise ValueError(f"sample-rate mismatch for {sample_id}")
    if abs(wav_duration - manifest_duration) > 1e-5:
        raise ValueError(f"duration mismatch for {sample_id}")

    l1 = scene["layers"]["L1_lexical_pragmatic"]
    l5 = scene["layers"]["L5_acoustic"]
    script_turns = scene.get("script", {}).get("turns", [])
    rendered_turns = session.get("turns", [])
    if len(script_turns) != len(rendered_turns):
        raise ValueError(f"script/render turn-count mismatch for {sample_id}")

    temporal_turns: list[dict[str, Any]] = []
    intervals: list[tuple[float, float]] = []
    speaker_sequence: list[int] = []
    previous_end: float | None = None
    for index, (script_turn, render_turn) in enumerate(zip(script_turns, rendered_turns)):
        speaker_index = int(render_turn["speaker_index"])
        start = round(float(render_turn["start"]), 6)
        end = round(float(render_turn["end"]), 6)
        intervals.append((start, end))
        speaker_sequence.append(speaker_index)
        temporal_turns.append(
            {
                "turn_id": script_turn.get("turn_id", f"t{index + 1:02d}"),
                "utterance_id": render_turn.get("utterance_id"),
                "speaker_id": f"speaker_{speaker_index + 1}",
                "text": render_turn.get("text"),
                "language": script_turn.get("language"),
                "language_spans": copy.deepcopy(script_turn.get("language_spans")),
                "start_sec": start,
                "end_sec": end,
                "overlap_with_previous": bool(
                    previous_end is not None and start < previous_end - 1e-7
                ),
            }
        )
        previous_end = end

    configured_speaker_count = int(session["num_speakers"])
    used_speaker_count = len(set(speaker_sequence))
    if configured_speaker_count != used_speaker_count:
        raise ValueError(
            f"configured/used speaker-count mismatch for {sample_id}: "
            f"{configured_speaker_count} != {used_speaker_count}"
        )

    duration = round(wav_duration, 6)
    merged_speech = merge_intervals(intervals)
    silence_segments = complement_intervals(duration, merged_speech)
    nonspeech_ratio = round(
        sum(item["end_sec"] - item["start_sec"] for item in silence_segments) / duration,
        6,
    )
    change_count = sum(a != b for a, b in zip(speaker_sequence, speaker_sequence[1:]))
    overlap_ratio = round6(session.get("overlap_ratio"))
    achieved_snr = round6(session.get("achieved_snr_db"))
    frame_snr_ground_truth = compute_frame_snr_ground_truth(sample_id)
    expected_content = copy.deepcopy(l1.get("expected_language_content") or {})

    tags: dict[str, Any] = {
        "basic_acoustic": {
            "duration_sec": duration,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            # The schema defines these as non-human-speech intervals. The
            # final manifest's utterance intervals are explicit native metadata.
            "nonspeech_ratio": nonspeech_ratio,
            "silence_segments": silence_segments,
        },
        "audio_quality": {
            "snr_db": achieved_snr,
            "dnsmos_sig": None,
            "dnsmos_bak": None,
            "dnsmos_ovrl": None,
            "dnsmos_p808": None,
        },
        "room_acoustic": {
            "far_field": None,
            "rt60_sec": round6(rir_metric.get("measured_rt60")),
            "c50_db": round6(rir_metric.get("measured_c50")),
        },
        "sound_field_scene": {
            "speech_music_events": copy.deepcopy(l5.get("speech_music_events")),
            "music_present": l5.get("music_present"),
            "external_noise_type": copy.deepcopy(l5.get("external_noise_type")),
            "noise_composition": copy.deepcopy(l5.get("noise_composition")),
        },
        "speaker": {
            "speaker_count": configured_speaker_count,
            "speaker_present": configured_speaker_count > 0,
            "asr_transcript": l1.get("reference_transcript"),
            "multi_speaker": configured_speaker_count >= 2,
            "speaker_change_count": change_count,
            "speaker_change": change_count > 0,
            "overlap_ratio": overlap_ratio,
            "speaker_overlap": overlap_ratio > 0 if overlap_ratio is not None else None,
            "profiles": None,
        },
        "language_content": {
            "language": sample_language(scene),
            "word_count": expected_content.get("word_count"),
            "punctuation": copy.deepcopy(expected_content.get("punctuation")),
            "repetition": copy.deepcopy(expected_content.get("repetition")),
            "filler": expected_content.get("filler"),
        },
        "deterministic_derived": {
            "speaker_change_rate_per_min": (
                round(change_count * 60.0 / duration, 6) if duration > 0 else None
            ),
            "filler_ratio": None,
        },
    }
    word_count = tags["language_content"]["word_count"]
    filler = tags["language_content"]["filler"]
    if word_count and filler is not None:
        tags["deterministic_derived"]["filler_ratio"] = round(filler / word_count, 6)

    descriptive = empty_descriptive()
    generation = {
        "dataset_version": session.get("dataset_version"),
        "grid_version": session.get("grid_version"),
        "scene_script": str(SCRIPT_PATH),
        "session_manifest": str(SESSION_PATH),
        "rir_metrics": str(RIR_METRICS_PATH),
        "acoustic_scene_id": scene.get("acoustic_scene_id"),
        "acoustic_environment": scene.get("acoustic_environment")
        or session.get("environment"),
        "language_mode": scene.get("language_mode"),
        "languages_present": languages_present(scene),
        "primary_language": l1.get("language"),
        "text_variant": scene.get("text_variant"),
        "environment": session.get("environment"),
        "background_asset": session.get("background_asset"),
        "noise_asset": session.get("provenance", {}).get("noise_asset"),
        "rir_backend": session.get("provenance", {}).get("rir_backend"),
        "rir_asset": session.get("provenance", {}).get("rir_asset"),
    }
    field_status = {
        "basic_acoustic.duration_sec": known_status("exact_measured", "rendered WAV header"),
        "basic_acoustic.sample_rate_hz": known_status("exact_measured", "rendered WAV header"),
        "basic_acoustic.channels": known_status("exact_measured", "rendered WAV header"),
        "basic_acoustic.nonspeech_ratio": known_status(
            "exact_derived", "complement of final session utterance intervals"
        ),
        "basic_acoustic.silence_segments": known_status(
            "exact_derived", "complement of final session utterance intervals"
        ),
        "audio_quality.snr_db": known_status("exact_render_manifest", "session achieved_snr_db"),
        "audio_quality.dnsmos_sig": null_status("DNSMOS was not run"),
        "audio_quality.dnsmos_bak": null_status("DNSMOS was not run"),
        "audio_quality.dnsmos_ovrl": null_status("DNSMOS was not run"),
        "audio_quality.dnsmos_p808": null_status("DNSMOS was not run"),
        "room_acoustic.far_field": null_status("no explicit far-field label"),
        "room_acoustic.rt60_sec": known_status(
            "exact_post_render_measurement", "qc/rir_metrics.jsonl measured_rt60"
        ),
        "room_acoustic.c50_db": known_status(
            "exact_post_render_measurement", "qc/rir_metrics.jsonl measured_c50"
        ),
        "sound_field_scene.speech_music_events": known_status(
            "exact_render_control", "L5 explicit render configuration"
        ),
        "sound_field_scene.music_present": known_status(
            "exact_render_control", "L5 explicit render configuration"
        ),
        "sound_field_scene.external_noise_type": known_status(
            "exact_render_control", "L5 explicit render configuration"
        ),
        "sound_field_scene.noise_composition": known_status(
            "exact_render_control", "L5 explicit render configuration"
        ),
        "speaker.speaker_count": known_status("exact_render_manifest", "session num_speakers"),
        "speaker.speaker_present": known_status("exact_derived", "speaker_count > 0"),
        "speaker.asr_transcript": known_status(
            "exact_reference_text", "scene script reference_transcript"
        ),
        "speaker.multi_speaker": known_status("exact_derived", "speaker_count >= 2"),
        "speaker.speaker_change_count": known_status(
            "exact_derived", "rendered turn speaker sequence"
        ),
        "speaker.speaker_change": known_status("exact_derived", "speaker_change_count > 0"),
        "speaker.overlap_ratio": known_status("exact_render_manifest", "session overlap_ratio"),
        "speaker.speaker_overlap": known_status("exact_derived", "overlap_ratio > 0"),
        "speaker.profiles": null_status("speaker profile requires annotation or acoustic analysis"),
        "language_content.language": (
            known_status("exact_script_label", "L1 explicit language")
            if tags["language_content"]["language"] is not None
            else null_status("schema has no single label for mixed/code-switch audio")
        ),
        "language_content.word_count": known_status(
            "exact_reference_text_rule", "L1 frozen expected_language_content"
        ),
        "language_content.punctuation": known_status(
            "exact_reference_text_rule", "L1 frozen expected_language_content"
        ),
        "language_content.repetition": known_status(
            "exact_reference_text_rule", "L1 frozen expected_language_content"
        ),
        "language_content.filler": known_status(
            "exact_reference_text_rule", "L1 frozen expected_language_content"
        ),
        "deterministic_derived.speaker_change_rate_per_min": known_status(
            "exact_derived", "speaker_change_count * 60 / duration_sec"
        ),
        "deterministic_derived.filler_ratio": known_status(
            "exact_derived", "filler / word_count"
        ),
        "frame_snr_ground_truth.mean_db": known_status(
            "exact_derived",
            (
                "post-RIR speech stems and rendered noise stem using the "
                "Brouhaha-maker detailed-SNR recipe"
            ),
        ),
        "descriptive": null_status("descriptive ground truth intentionally disabled"),
    }

    enriched = {
        "sample_id": sample_id,
        "audio_path": str(audio_path),
        "tags": tags,
        "descriptive": descriptive,
        "frame_snr_ground_truth": frame_snr_ground_truth,
        "temporal_ground_truth": {
            "timebase": "seconds",
            "timestamp_source": "rendered_session_manifest",
            "turns": temporal_turns,
            "speech_placement_intervals": [
                {"start_sec": start, "end_sec": end}
                for start, end in merged_speech
            ],
            "nonspeech_intervals": silence_segments,
        },
        "generation": generation,
        "field_status": field_status,
    }
    return enriched, tags


def validate_tags(tags: dict[str, Any]) -> None:
    if tuple(tags.keys()) != STRUCTURAL_KEYS:
        raise ValueError(f"unexpected structural groups: {tuple(tags.keys())}")

    basic = tags["basic_acoustic"]
    if basic["duration_sec"] <= 0 or basic["sample_rate_hz"] <= 0 or basic["channels"] <= 0:
        raise ValueError("invalid WAV header values")
    segments = basic["silence_segments"]
    if not isinstance(segments, list):
        raise ValueError("silence_segments must be a list from the render timeline")
    previous_end = 0.0
    for segment in segments:
        start = segment.get("start_sec")
        end = segment.get("end_sec")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError("invalid silence segment shape")
        if not (0 <= start < end <= basic["duration_sec"]):
            raise ValueError("silence segment is out of bounds")
        if start < previous_end:
            raise ValueError("silence segments overlap or are unsorted")
        previous_end = end
    expected_nonspeech = round(
        sum(segment["end_sec"] - segment["start_sec"] for segment in segments)
        / basic["duration_sec"],
        6,
    )
    if basic["nonspeech_ratio"] != expected_nonspeech:
        raise ValueError("nonspeech ratio mismatch")

    quality = tags["audio_quality"]
    for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        if quality[key] is not None:
            raise ValueError(f"{key} must be null")

    room = tags["room_acoustic"]
    if room["far_field"] is not None:
        raise ValueError("far_field must be null")
    if room["rt60_sec"] is None or room["c50_db"] is None:
        raise ValueError("RIR measurements are missing")

    speaker = tags["speaker"]
    if speaker["profiles"] is not None:
        raise ValueError("speaker.profiles must be null")
    if speaker["speaker_present"] != (speaker["speaker_count"] > 0):
        raise ValueError("speaker_present mismatch")
    if speaker["multi_speaker"] != (speaker["speaker_count"] >= 2):
        raise ValueError("multi_speaker mismatch")
    if speaker["speaker_change"] != (speaker["speaker_change_count"] > 0):
        raise ValueError("speaker_change mismatch")
    if speaker["speaker_overlap"] != (speaker["overlap_ratio"] > 0):
        raise ValueError("speaker_overlap mismatch")

    derived = tags["deterministic_derived"]
    expected_rate = round(
        speaker["speaker_change_count"] * 60.0 / basic["duration_sec"], 6
    )
    if derived["speaker_change_rate_per_min"] != expected_rate:
        raise ValueError("speaker-change rate mismatch")
    word_count = tags["language_content"]["word_count"]
    filler = tags["language_content"]["filler"]
    expected_filler_ratio = round(filler / word_count, 6) if word_count and filler is not None else None
    if derived["filler_ratio"] != expected_filler_ratio:
        raise ValueError("filler ratio mismatch")


def write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sample_id",
        "audio_path",
        "acoustic_scene_id",
        "acoustic_environment",
        "language",
        "languages_present",
        "language_mode",
        "speaker_count",
        "speaker_change_count",
        "overlap_ratio",
        "duration_sec",
        "sample_rate_hz",
        "channels",
        "snr_db",
        "frame_snr_mean_db",
        "rt60_sec",
        "c50_db",
        "music_present",
        "external_noise_type",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            tags = row["tags"]
            generation = row["generation"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "audio_path": row["audio_path"],
                    "acoustic_scene_id": generation.get("acoustic_scene_id"),
                    "acoustic_environment": generation.get("acoustic_environment"),
                    "language": tags["language_content"]["language"],
                    "languages_present": "|".join(generation.get("languages_present") or []),
                    "language_mode": generation.get("language_mode"),
                    "speaker_count": tags["speaker"]["speaker_count"],
                    "speaker_change_count": tags["speaker"]["speaker_change_count"],
                    "overlap_ratio": tags["speaker"]["overlap_ratio"],
                    "duration_sec": tags["basic_acoustic"]["duration_sec"],
                    "sample_rate_hz": tags["basic_acoustic"]["sample_rate_hz"],
                    "channels": tags["basic_acoustic"]["channels"],
                    "snr_db": tags["audio_quality"]["snr_db"],
                    "frame_snr_mean_db": row["frame_snr_ground_truth"]["mean_db"],
                    "rt60_sec": tags["room_acoustic"]["rt60_sec"],
                    "c50_db": tags["room_acoustic"]["c50_db"],
                    "music_present": tags["sound_field_scene"]["music_present"],
                    "external_noise_type": "|".join(
                        tags["sound_field_scene"]["external_noise_type"] or []
                    ),
                }
            )


def main() -> None:
    scenes = {row["scene_id"]: row for row in load_jsonl(SCRIPT_PATH)}
    sessions = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    rir_metrics = {row["scene_id"]: row for row in load_jsonl(RIR_METRICS_PATH)}
    if not isinstance(sessions, list):
        raise ValueError("sessions manifest must be a JSON array")
    session_ids = [row["session_id"] for row in sessions]
    if len(session_ids) != 500 or len(set(session_ids)) != 500:
        raise ValueError("expected 500 unique rendered sessions")
    if set(session_ids) != set(scenes) or set(session_ids) != set(rir_metrics):
        raise ValueError("scene/session/RIR ID sets differ")

    enriched_rows: list[dict[str, Any]] = []
    tags_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    counters = {
        key: Counter()
        for key in ("language", "language_mode", "speaker_count", "music_present")
    }
    for session in sessions:
        sample_id = session["session_id"]
        enriched, tags = build_record(scenes[sample_id], session, rir_metrics[sample_id])
        validate_tags(tags)
        if any(value is not None for value in enriched["descriptive"].values()):
            raise ValueError(f"descriptive values must all be null for {sample_id}")
        enriched_rows.append(enriched)
        tags_rows.append(tags)
        schema_rows.append({**copy.deepcopy(tags), "descriptive": copy.deepcopy(enriched["descriptive"])})
        counters["language"][str(tags["language_content"]["language"])] += 1
        counters["language_mode"][str(enriched["generation"]["language_mode"])] += 1
        counters["speaker_count"][str(tags["speaker"]["speaker_count"])] += 1
        counters["music_present"][str(tags["sound_field_scene"]["music_present"]).lower()] += 1

    GT_ROOT.mkdir(parents=True, exist_ok=False)
    write_jsonl(GT_ROOT / "ground_truth.jsonl", enriched_rows)
    write_jsonl(GT_ROOT / "schema_ground_truth.jsonl", schema_rows)
    write_jsonl(GT_ROOT / "sure_tagger_gt_tags_only.jsonl", tags_rows)
    write_index_csv(GT_ROOT / "ground_truth_index.csv", enriched_rows)

    null_fields = {
        "basic_acoustic.nonspeech_ratio": 0,
        "basic_acoustic.silence_segments": 0,
        "audio_quality.dnsmos_sig": 500,
        "audio_quality.dnsmos_bak": 500,
        "audio_quality.dnsmos_ovrl": 500,
        "audio_quality.dnsmos_p808": 500,
        "room_acoustic.far_field": 500,
        "speaker.profiles": 500,
        "descriptive.*": 500,
        "language_content.language": counters["language"].get("None", 0),
    }
    summary = {
        "dataset": OUTPUT_ROOT.name,
        "schema": "Ground Truth field policy in ground_truth/GROUND_TRUTH_POLICY.md",
        "policy": "strict_no_inference_v5_brouhaha_detailed_snr",
        "sample_count": len(enriched_rows),
        "audio_hours": round(
            sum(row["tags"]["basic_acoustic"]["duration_sec"] for row in enriched_rows)
            / 3600.0,
            4,
        ),
        "counts": {key: dict(value) for key, value in counters.items()},
        "populated_sources": {
            "rendered_wav_header": [
                "basic_acoustic.duration_sec",
                "basic_acoustic.sample_rate_hz",
                "basic_acoustic.channels",
            ],
            "session_manifest": [
                "audio_quality.snr_db",
                "speaker.speaker_count",
                "speaker.overlap_ratio",
            ],
            "rendered_stems": [
                "frame_snr_ground_truth.mean_db",
            ],
            "scene_script_explicit_labels": [
                "sound_field_scene.*",
                "speaker.asr_transcript",
                "language_content.*",
            ],
            "rir_measurement_file": [
                "room_acoustic.rt60_sec",
                "room_acoustic.c50_db",
            ],
            "deterministic_formulas": [
                "speaker boolean/count projections",
                "speaker change fields",
                "deterministic_derived.*",
            ],
        },
        "null_field_counts": null_fields,
        "models_run": [],
        "frame_snr_ground_truth": {
            "version": FRAME_SNR_VERSION,
            "sample_count": len(enriched_rows),
            "frame_duration_sec": FRAME_SNR_DURATION_SEC,
            "frame_hop_sec": FRAME_SNR_HOP_SEC,
            "active_sample_threshold": FRAME_SNR_ACTIVE_THRESHOLD,
            "normalization_epsilon": FRAME_SNR_NORMALIZATION_EPSILON,
            "noise_power_epsilon": FRAME_SNR_NOISE_POWER_EPSILON,
            "signal_to_noise_floor": FRAME_SNR_RATIO_FLOOR,
            "aggregation": "arithmetic_mean_of_all_detailed_snr_windows_db",
            "valid_frame_policy": (
                "all complete 2 s windows; no frame-level VAD filter"
            ),
            "dataset_mean_db": round(
                sum(
                    row["frame_snr_ground_truth"]["mean_db"]
                    for row in enriched_rows
                )
                / len(enriched_rows),
                6,
            ),
            "minimum_mean_db": min(
                row["frame_snr_ground_truth"]["mean_db"] for row in enriched_rows
            ),
            "maximum_mean_db": max(
                row["frame_snr_ground_truth"]["mean_db"] for row in enriched_rows
            ),
        },
    }
    (GT_ROOT / "ground_truth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# 500 条音频 Ground Truth（无推理版）

本目录含 {len(enriched_rows)} 条音频的 ground truth，按 `tag-schema(2).md` 组织。构建过程不运行任何模型，也不根据场景名称推断标签。

## 文件

- `ground_truth.jsonl`：带样本 ID、来源、时间戳和字段状态的完整记录。
- `schema_ground_truth.jsonl`：每行只含 schema 的 8 个顶层分组。
- `sure_tagger_gt_tags_only.jsonl`：每行只含 sure-tagger 结构化输出使用的 7 个非 descriptive 分组。
- `ground_truth_index.csv`：便于筛选的扁平索引。
- `ground_truth_summary.json`：数量、来源和 null 字段统计。

完整记录额外包含独立的 `frame_snr_ground_truth`，它不属于
`tag-schema(2).md`，因此不会写入 `schema_ground_truth.jsonl` 或
`sure_tagger_gt_tags_only.jsonl`。该字段使用 Brouhaha-maker 训练标签的
2 秒窗 / 10 ms hop 规则，不使用 Brouhaha 模型输出。

## 填值原则

| 数据来源 | 填入字段 |
| --- | --- |
| 最终 WAV header | duration、sample rate、channels |
| 最终 session manifest | achieved SNR、说话人数、重叠率 |
| 最终 speech/noise stems | Brouhaha detailed-SNR 训练标签口径的 2 秒滑窗 SNR 算术平均 |
| 渲染轮次序列 | 说话人切换，以及由其确定性计算的字段 |
| scene script 的显式配置/文本统计 | 音乐与噪声配置、reference transcript、语言内容统计 |
| `qc/rir_metrics.jsonl` | measured RT60、measured C50 |

`speaker.profiles`、四项 DNSMOS、`room_acoustic.far_field` 和全部 `descriptive` 子字段均为 `null`。`nonspeech_ratio` 与 `silence_segments` 使用最终 session manifest 的 utterance 时间戳补集；这表示未安排人类语音的区间，不等同于物理静音。多语种和 code-switch 样本无法用 schema 的单一语言值表达，`language_content.language` 为 `null`；已知语言列表保留在完整记录的 `generation.languages_present` 中。

本发布版只生成当前无推理 Ground Truth。
"""
    (GT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    policy_doc = """# Ground Truth 字段策略

| Schema 字段 | 新版取值 | 直接来源 |
| --- | --- | --- |
| `basic_acoustic.duration_sec/sample_rate_hz/channels` | 已知值 | 最终 WAV header |
| `basic_acoustic.nonspeech_ratio/silence_segments` | 已知值 | 最终 session utterance 时间戳的补集；表示非人类语音区间 |
| `audio_quality.snr_db` | 已知值 | session `achieved_snr_db` |
| `frame_snr_ground_truth.mean_db` | 已知值（独立于 schema） | post-RIR speech stems 与 noise stem；仅平均与 dry speech 相交的完整帧 |
| `audio_quality.dnsmos_*` | `null` | 未运行 DNSMOS |
| `room_acoustic.rt60_sec/c50_db` | 已知值 | RIR 实测文件 |
| `room_acoustic.far_field` | `null` | 无显式标签 |
| `sound_field_scene.*` | 已知值 | L5 渲染配置 |
| `speaker.speaker_count/overlap_ratio` | 已知值 | session manifest |
| `speaker.speaker_change*` | 已知值 | 渲染轮次序列的确定性计算 |
| `speaker.asr_transcript` | 已知值 | 合成脚本 reference transcript |
| `speaker.profiles` | `null` | 不使用目标 profile 充当音频真值 |
| `language_content.*` | 已知值或 `null` | 冻结脚本文本统计；混合语言的单值 language 为 `null` |
| `deterministic_derived.*` | 已知值 | schema 公式 |
| `descriptive.*` | `null` | 禁止描述性推断 |
"""
    (GT_ROOT / "GROUND_TRUTH_POLICY.md").write_text(policy_doc, encoding="utf-8")

    print(
        json.dumps(
            {
                "sample_count": len(enriched_rows),
                "output": str(GT_ROOT),
                "models_run": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
