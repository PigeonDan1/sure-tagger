#!/usr/bin/env python3
"""Build the L1-L5 speech-scene dataset with dots.tts and saasr-mini.

The pipeline is resumable and has six stages:

  prepare -> base -> edit -> derive -> render -> qc

Base speech is deduplicated by (text, reference voice). Slow and fast variants
are produced by dots.tts.edit. Derived sources receive a deterministic RMS
target, then the scene renderer adds overlap, procedural RIR, environmental
backgrounds, and the requested SNR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter, resample_poly

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SIM_REPO = WORKSPACE
sys.path.insert(0, str(SIM_REPO / "src"))

from saasr_mini.audio import rms, stable_seed, write_audio  # noqa: E402
from saasr_mini.noise import generate_noise  # noqa: E402
from saasr_mini.pipeline import simulate  # noqa: E402

SCENE_SOURCE = Path(os.environ.get("L1_L5_SCENE_SOURCE", WORKSPACE / "data/scene_scripts_v4.jsonl"))
OUTPUT_DEFAULT = WORKSPACE / "runs/v4_multilingual"
V3_OUTPUT = Path(os.environ["L1_L5_V3_OUTPUT"]) if os.environ.get("L1_L5_V3_OUTPUT") else None
ASSETS_ROOT = Path(os.environ.get("RELEASE_ASSETS_ROOT", WORKSPACE / "external_assets"))
BASE_MODEL = Path(os.environ.get("DOTS_BASE_MODEL", WORKSPACE / "models/dots_tts_base"))
EDIT_MODEL = Path(os.environ.get("DOTS_EDIT_MODEL", WORKSPACE / "models/dots_tts_edit"))
VOICE_CONFIG = Path(os.environ.get("L1_L5_VOICE_CONFIG", WORKSPACE / "data/voice_bank_v4.json"))
FLEURS_ENHANCED_ROOT = ASSETS_ROOT / "fleurs" / "audio_enhanced_deepfilternet"
FLEURS_TRANSCRIPT_ROOT = ASSETS_ROOT / "fleurs" / "transcripts"
SOURCE_SR = 24_000
ASSET_SECONDS = 90
SEED = 20260911
_NORMAL_FACTOR = os.environ.get("L1_L5_NORMAL_FACTOR")
RATE_FACTORS = {
    "slow": float(os.environ.get("L1_L5_SLOW_FACTOR", "0.67")),
    "normal": float(_NORMAL_FACTOR) if _NORMAL_FACTOR else None,
    "fast": float(os.environ.get("L1_L5_FAST_FACTOR", "1.50")),
}
VOLUME_RMS_DBFS = {"low": -34.0, "normal": -24.0, "loud": -14.0, None: -24.0}
PITCH_VOICES = {"low": [5, 7], "mid": [3, 6, 8], "high": [4], None: [5, 4, 3]}
DERIVATION_VERSION = "l1_l5_source_derive.v5"
FORCE_NORMAL_SPEED = os.environ.get("L1_L5_FORCE_NORMAL_SPEED", "").lower() in {"1", "true", "yes"}
CLEAN_RENDER = os.environ.get("L1_L5_CLEAN_RENDER", "").lower() in {"1", "true", "yes"}
REAL_MUSIC_FILENAMES = (
    "ikoliks_aj-background-music-320427.mp3",
    "mickeyscat-moment-of-peace-mickeyscat-554494.mp3",
)
# Music is supplied under the external asset root after license review.
REAL_MUSIC_SOURCE_ROOT = Path(
    os.environ.get(
        "L1_L5_MUSIC_SOURCE_ROOT",
        str(ASSETS_ROOT / "music"),
    )
)
MUSIC_REFERENCE_RMS = 0.3270884

NOISE_PROFILE_BY_ENVIRONMENT = {
    "dry": "office",
    "meeting_room": "office",
    "open_space": "street",
    "restaurant": "cafe",
    "public_transit": "station",
    "vehicle": "train",
    "residential": "office",
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reuse_v3_sources(output: Path) -> dict[str, int]:
    """Reuse byte-identical v3 sources for unchanged text/voice combinations."""
    if V3_OUTPUT is None or not (V3_OUTPUT / "spec/base_specs.jsonl").is_file():
        return {"base": 0, "edit": 0, "derived": 0}
    reused = Counter()
    old_base = _load_jsonl(V3_OUTPUT / "spec/base_specs.jsonl")
    old_edit = _load_jsonl(V3_OUTPUT / "spec/edit_specs.jsonl")
    old_derived = _load_jsonl(V3_OUTPUT / "spec/derived_specs.jsonl")
    base_index = {(int(item["voice_index"]), item["text"]): item for item in old_base}
    edit_index = {(item["text"], round(float(item["rate_factor"]), 2)): item for item in old_edit}
    derived_index = {
        (item["text"], item.get("language", "zh"), item.get("target_volume"), round(float(item["target_speed"]), 4)): item
        for item in old_derived
    }
    for spec in _load_jsonl(output / "spec/base_specs.jsonl"):
        old = base_index.get((int(spec["voice_index"]), spec["text"]))
        if old is None or not Path(old["output"]).is_file() or not Path(old["native_output"]).is_file():
            continue
        _link_or_copy(Path(old["output"]), Path(spec["output"]))
        _link_or_copy(Path(old["native_output"]), Path(spec["native_output"]))
        _json_dump(Path(spec["output"] + ".json"), {**spec, "stage": "base", "sha256": _sha256(Path(spec["output"])), "native_sha256": _sha256(Path(spec["native_output"]))})
        reused["base"] += 1
    for spec in _load_jsonl(output / "spec/edit_specs.jsonl"):
        old = edit_index.get((spec["text"], round(float(spec["rate_factor"]), 2)))
        if old is None or not Path(old["output"]).is_file():
            continue
        _link_or_copy(Path(old["output"]), Path(spec["output"]))
        old_meta = Path(old["output"] + ".json")
        metadata = json.loads(old_meta.read_text(encoding="utf-8")) if old_meta.is_file() else {}
        _json_dump(Path(spec["output"] + ".json"), {**metadata, **spec, "stage": "edit", "sha256": _sha256(Path(spec["output"]))})
        reused["edit"] += 1
    for spec in _load_jsonl(output / "spec/derived_specs.jsonl"):
        key = (spec["text"], spec.get("language", "zh"), spec.get("target_volume"), round(float(spec["target_speed"]), 4))
        old = derived_index.get(key)
        if old is None or not Path(old["output"]).is_file():
            continue
        _link_or_copy(Path(old["output"]), Path(spec["output"]))
        old_meta = Path(old["output"] + ".json")
        metadata = json.loads(old_meta.read_text(encoding="utf-8")) if old_meta.is_file() else {}
        _json_dump(Path(spec["output"] + ".json"), {**metadata, **spec, "derivation_version": DERIVATION_VERSION, "sha256": _sha256(Path(spec["output"]))})
        reused["derived"] += 1
    return dict(reused)


def _short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _task_seed(key: str) -> int:
    # NumPy's legacy MT19937 path used by dots.tts requires an unsigned
    # 32-bit seed. Reserve the all-ones value and keep every task reproducible.
    return (SEED + int(_short_hash(key, 8), 16)) % (2**32 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chinese_units(text: str) -> int:
    import re

    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def _english_units(text: str) -> int:
    import re

    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text))


def _text_units(text: str, language: str) -> int:
    import re

    language = (language or "").lower()
    # Japanese and Korean do not normally separate words with spaces. Treat
    # their script characters as speech units; otherwise a complete sentence
    # is counted as one token and speed calibration collapses it to silence.
    if language == "zh":
        return _chinese_units(text) + _english_units(text)
    if language == "ja":
        script = len(re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
        return script + _english_units(text)
    if language == "ko":
        script = len(re.findall(r"[\uac00-\ud7af]", text))
        return script + _english_units(text)
    return _english_units(text) or len(re.findall(r"[^\W\d_]+|\d+", text, re.UNICODE))


def _active_span_seconds(audio: np.ndarray, sample_rate: int, top_db: float = 30.0) -> float:
    values = np.abs(np.asarray(audio, dtype=np.float32))
    if not values.size:
        return 0.0
    peak = float(np.max(values))
    if peak <= 1e-12:
        return 0.0
    active = np.flatnonzero(values > peak * (10.0 ** (-top_db / 20.0)))
    if not active.size:
        return 0.0
    return (int(active[-1]) - int(active[0]) + 1) / sample_rate


def _active_rms(audio: np.ndarray, top_db: float = 40.0) -> float:
    values = np.asarray(audio, dtype=np.float32)
    if not values.size:
        return 0.0
    peak = float(np.max(np.abs(values)))
    active = values[np.abs(values) >= peak * (10.0 ** (-top_db / 20.0))]
    return rms(active) if active.size else 0.0


def _trim_active_edges(audio: np.ndarray, sample_rate: int, pad_seconds: float = 0.02) -> np.ndarray:
    """Remove model-generated leading/trailing silence around one TTS chunk."""
    values = np.asarray(audio, dtype=np.float32)
    if not values.size:
        return values
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-8:
        return values
    active = np.flatnonzero(np.abs(values) >= peak * 0.01)
    if not active.size:
        return values
    padding = max(0, round(float(sample_rate) * pad_seconds))
    start = max(0, int(active[0]) - padding)
    end = min(values.size, int(active[-1]) + padding + 1)
    return values[start:end].astype(np.float32, copy=False)


def _compress_long_silences(audio: np.ndarray, sample_rate: int, max_silence_seconds: float = 0.28) -> np.ndarray:
    """Bound abnormal internal TTS gaps while preserving short phonetic pauses."""
    values = np.asarray(audio, dtype=np.float32)
    if not values.size:
        return values
    frame = max(1, round(sample_rate * 0.01))
    energy = np.array(
        [float(np.sqrt(np.mean(values[i : i + frame] ** 2))) for i in range(0, values.size, frame)],
        dtype=np.float32,
    )
    threshold = max(1e-4, float(np.max(energy)) * 0.01)
    silent = energy < threshold
    max_frames = max(1, round(max_silence_seconds / 0.01))
    pieces: list[np.ndarray] = []
    cursor = 0
    index = 0
    while index < silent.size:
        if not silent[index]:
            index += 1
            continue
        end = index
        while end < silent.size and silent[end]:
            end += 1
        run_frames = end - index
        if run_frames > max_frames:
            keep_frames = max_frames
            start_sample = index * frame
            end_sample = min(values.size, (index + keep_frames) * frame)
            pieces.append(values[cursor:start_sample])
            pieces.append(values[start_sample:end_sample])
            cursor = min(values.size, end * frame)
        index = end
    pieces.append(values[cursor:])
    return np.concatenate([piece for piece in pieces if piece.size], dtype=np.float32) if pieces else values


def _normalize_active_rms(audio: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float]:
    measured = _active_rms(audio)
    target = 10.0 ** (target_dbfs / 20.0)
    gain = target / max(measured, 1e-8)
    result = np.asarray(audio, dtype=np.float32) * gain
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 0.98:
        result *= 0.98 / peak
        gain *= 0.98 / peak
    return result.astype(np.float32), 20.0 * math.log10(max(gain, 1e-12))


def _atempo_factors(factor: float) -> list[float]:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"invalid atempo factor: {factor}")
    values: list[float] = []
    while factor < 0.5:
        values.append(0.5)
        factor /= 0.5
    while factor > 2.0:
        values.append(2.0)
        factor /= 2.0
    if abs(factor - 1.0) > 0.005:
        values.append(factor)
    return values


def _fallback_time_stretch(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Apply the requested rate factor when dots.tts.edit is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for edit fallback")
    factors = _atempo_factors(float(factor))
    if not factors:
        return np.asarray(audio, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="l1_l5_edit_fallback_") as temp_dir:
        source = Path(temp_dir) / "source.wav"
        target = Path(temp_dir) / "target.wav"
        sf.write(source, np.asarray(audio, dtype=np.float32), int(sample_rate), subtype="FLOAT")
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-filter:a",
                ",".join(f"atempo={value:.8f}" for value in factors),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(target),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg edit fallback failed: {completed.stderr.decode(errors='replace')}")
        result, returned_rate = sf.read(target, dtype="float32", always_2d=False)
        if int(returned_rate) != int(sample_rate):
            raise RuntimeError(f"edit fallback returned unexpected sample rate {returned_rate}")
        return np.asarray(result, dtype=np.float32)


def _calibrate_speed(
    audio: np.ndarray, text: str, target_rate: float, language: str = "zh"
) -> tuple[np.ndarray, list[float], float, float]:
    initial_span = _active_span_seconds(audio, SOURCE_SR)
    unit_count = _text_units(text, language)
    if initial_span < 0.20 and unit_count >= 5:
        raise ValueError(
            f"source speech is too short for {unit_count} text units: {initial_span:.3f}s"
        )
    initial_rate = unit_count / max(initial_span, 1e-8)
    if initial_rate > max(20.0, target_rate * 8.0):
        raise ValueError(
            f"source speech rate is implausible for {unit_count} text units: {initial_rate:.2f}/s"
        )
    factors = _atempo_factors(target_rate / max(initial_rate, 1e-8))
    if not factors:
        return audio.astype(np.float32), [], initial_rate, initial_rate
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for pitch-preserving speed calibration")
    result = np.asarray(audio, dtype=np.float32)
    applied: list[float] = []
    # A second pass removes small filter/edge-silence deviations from target.
    for iteration in range(2):
        current_span = _active_span_seconds(result, SOURCE_SR)
        current_rate = _text_units(text, language) / max(current_span, 1e-8)
        correction = target_rate / max(current_rate, 1e-8)
        step_factors = _atempo_factors(correction)
        if not step_factors or abs(current_rate - target_rate) / target_rate <= 0.025:
            break
        with tempfile.TemporaryDirectory(prefix="l1_l5_atempo_") as temp_dir:
            source = Path(temp_dir) / "source.wav"
            target = Path(temp_dir) / "target.wav"
            sf.write(source, result, SOURCE_SR, subtype="FLOAT")
            chain = ",".join(f"atempo={value:.8f}" for value in step_factors)
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-filter:a",
                    chain,
                    "-ar",
                    str(SOURCE_SR),
                    "-ac",
                    "1",
                    str(target),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"ffmpeg atempo failed: {completed.stderr.decode(errors='replace')}")
            result, rate = sf.read(target, dtype="float32", always_2d=False)
            if int(rate) != SOURCE_SR:
                raise RuntimeError(f"ffmpeg returned unexpected sample rate {rate}")
            result = np.asarray(result, dtype=np.float32)
        applied.extend(step_factors)
    final_span = _active_span_seconds(result, SOURCE_SR)
    final_rate = _text_units(text, language) / max(final_span, 1e-8)
    return result, applied, initial_rate, final_rate


def _load_voice_bank(output: Path) -> dict[int, dict[str, Any]]:
    source = json.loads(VOICE_CONFIG.read_text(encoding="utf-8"))
    voices = {int(index): dict(item) for index, item in source.items()}
    for item in voices.values():
        prompt = Path(item["prompt_audio"])
        if not prompt.is_absolute():
            if prompt.parts[0] == "external_assets":
                prompt = ASSETS_ROOT.joinpath(*prompt.parts[1:])
            else:
                prompt = WORKSPACE / prompt
        item["prompt_audio"] = str(prompt.resolve())
    # The original FLEURS prompts were truncated to three seconds. Use the
    # enhanced, full-length prompts supplied in the shared reference pool and
    # keep a copy inside the dataset for reproducible downstream synthesis.
    extra_by_language: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in sorted(voices.items()):
        if str(item.get("source", "")).startswith("FLEURS"):
            extra_by_language[str(item.get("language"))].append((index, item))
    for language, entries in extra_by_language.items():
        for slot, (index, item) in enumerate(entries):
            source_audio = FLEURS_ENHANCED_ROOT / f"fleurs_{language}_{slot:02d}.wav"
            if not source_audio.is_file():
                raise FileNotFoundError(f"missing enhanced FLEURS reference: {source_audio}")
            destination = output / "references/fleurs_enhanced" / source_audio.name
            if not destination.is_file() or destination.stat().st_size != source_audio.stat().st_size:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_audio, destination)
            transcript = FLEURS_TRANSCRIPT_ROOT / f"fleurs_{language}_{slot:02d}.txt"
            prompt_text = transcript.read_text(encoding="utf-8").strip() if transcript.is_file() else item.get("prompt_text", "")
            voices[index] = {
                **item,
                "prompt_audio": str(destination.resolve()),
                "prompt_audio_original": str(source_audio.resolve()),
                "prompt_text": prompt_text,
                "source": "FLEURS-DeepFilterNet",
                "reference_slot": slot,
            }
    missing = [item["prompt_audio"] for item in voices.values() if not Path(item["prompt_audio"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing reference audio: {missing[:3]}")
    payload = {str(index): item for index, item in voices.items()}
    _json_dump(output / "references/voice_bank.json", payload)
    return voices


def _choose_voice(pitch: str | None, record_index: int, speaker_index: int) -> int:
    choices = PITCH_VOICES[pitch]
    return choices[(record_index + speaker_index) % len(choices)]


def _room(reverb_bucket: str, speakers: int) -> dict[str, Any]:
    if reverb_bucket == "较低":
        dimensions = [6.0, 5.0, 3.0]
        microphone = [3.0, 2.5, 1.2]
        # About two metres keeps the direct-to-reverberant ratio near the
        # low-reverb C50 target instead of producing an unrealistically dry RIR.
        candidates = [[1.0, 2.5, 1.2], [5.0, 2.5, 1.2], [3.0, 4.5, 1.2]]
    else:
        dimensions = [11.0, 8.0, 3.8]
        microphone = [1.5, 1.5, 1.2]
        candidates = [[5.5, 3.2, 1.2], [7.5, 5.5, 1.2], [4.0, 6.2, 1.2]]
    return {
        "dimensions_m": dimensions,
        "microphone_position_m": microphone,
        "speaker_positions_m": candidates[:speakers],
    }


def _music(samples: int, sample_rate: int, seed: int) -> np.ndarray:
    """Create a deterministic, royalty-free instrumental background."""
    rng = np.random.default_rng(seed)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    chord_roots = np.array([110.0, 146.83, 98.0, 130.81], dtype=np.float64)
    chord_index = (time // 4.0).astype(np.int64) % len(chord_roots)
    root = chord_roots[chord_index]
    signal = np.zeros(samples, dtype=np.float64)
    for ratio, weight in ((1.0, 0.32), (1.25, 0.24), (1.5, 0.20), (2.0, 0.12)):
        phase = 2.0 * np.pi * np.cumsum(root * ratio) / sample_rate
        signal += weight * np.sin(phase)
    melody_steps = np.array([1.0, 1.125, 1.25, 1.5, 1.25, 1.125, 1.0, 0.75])
    melody = 440.0 * melody_steps[((time * 2.0).astype(np.int64)) % len(melody_steps)]
    melody_phase = 2.0 * np.pi * np.cumsum(melody) / sample_rate
    note_phase = np.mod(time, 0.5) / 0.5
    note_envelope = np.exp(-3.2 * note_phase) * np.sin(np.pi * np.minimum(1.0, note_phase * 8.0))
    signal += 0.16 * np.sin(melody_phase) * note_envelope
    for beat in np.arange(0.0, samples / sample_rate, 0.5):
        start = int(beat * sample_rate)
        length = min(samples - start, int(0.055 * sample_rate))
        if length <= 0:
            continue
        burst = rng.standard_normal(length) * np.exp(-np.linspace(0.0, 8.0, length))
        signal[start : start + length] += 0.08 * burst
    fade = min(samples // 4, sample_rate)
    if fade:
        signal[:fade] *= np.linspace(0.0, 1.0, fade)
    return signal.astype(np.float32)


@lru_cache(maxsize=4)
def _decode_real_music(path: str) -> np.ndarray:
    """Decode a user-provided music file once, using ffmpeg for MP3 support."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(SOURCE_SR),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError(f"decoded music asset is empty: {path}")
    audio -= np.mean(audio, dtype=np.float32)
    audio *= MUSIC_REFERENCE_RMS / max(rms(audio), 1e-8)
    return audio.astype(np.float32)


def _real_music_path(output: Path, variant: int) -> Path:
    music_root = output / "assets/backgrounds"
    candidates = [music_root / filename for filename in REAL_MUSIC_FILENAMES]
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "music-enabled scenes require the two real music assets; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return candidates[variant % len(candidates)]


def _real_music(samples: int, output: Path, variant: int) -> np.ndarray:
    audio = _decode_real_music(str(_real_music_path(output, variant).resolve()))
    if audio.size < samples:
        repeats = math.ceil(samples / audio.size)
        audio = np.tile(audio, repeats)
    return audio[:samples].copy()


def _resolve_measured_rir(record: dict[str, Any]) -> str | None:
    selection = record.get("rir_selection") or {}
    if selection.get("mode") != "measured":
        return None
    asset_path = selection.get("asset_path")
    if not asset_path:
        raise FileNotFoundError(f"{record.get('scene_id')}: measured RIR has no asset_path")
    relative = Path(str(asset_path))
    if relative.parts and relative.parts[0] == "rir":
        relative = Path(*relative.parts[1:])
    path = (ASSETS_ROOT / "rir" / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{record.get('scene_id')}: missing measured RIR {path}; "
            "provide RELEASE_ASSETS_ROOT/rir or use a licensed synthetic-only run"
        )
    return str(path)


def _event_layer(labels: set[str], samples: int, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    result = np.zeros(samples, dtype=np.float64)
    white = rng.standard_normal(samples)
    if labels & {"Static", "Hiss", "Radio", "Telephone", "Public address system", "Echo"}:
        b, a = butter(2, [900 / (sample_rate / 2), min(0.96, 6500 / (sample_rate / 2))], btype="band")
        result += 0.06 * lfilter(b, a, white)
    if "Telephone" in labels:
        gate = ((time % 6.0) < 0.8).astype(np.float64)
        result += gate * (0.025 * np.sin(2 * np.pi * 440 * time) + 0.02 * np.sin(2 * np.pi * 480 * time))
    if labels & {"Wind", "Wind noise (microphone)"}:
        b, a = butter(2, min(0.95, 420 / (sample_rate / 2)), btype="low")
        wind = lfilter(b, a, white)
        result += 0.45 * wind * (0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * time) ** 2)
    if "Whistle" in labels:
        for start_sec in np.arange(2.0, samples / sample_rate, 8.5):
            start = int(start_sec * sample_rate)
            length = min(samples - start, int(0.32 * sample_rate))
            local = np.arange(length) / sample_rate
            result[start : start + length] += 0.13 * np.sin(2 * np.pi * (1800 + 500 * local) * local) * np.sin(np.pi * local / max(local[-1], 1e-6))
    mechanical = labels & {
        "Chair", "Clicking", "Computer keyboard", "Cutlery", "Dishes, pots, and pans",
        "Glass", "Paper rustling", "Tools", "Writing", "Cash register",
    }
    if mechanical:
        event_rate = 1.8 if labels & {"Computer keyboard", "Writing"} else 0.65
        for start in rng.integers(0, samples, size=max(1, int(samples / sample_rate * event_rate))):
            length = min(samples - int(start), rng.integers(max(8, sample_rate // 500), max(9, sample_rate // 30)))
            if length <= 0:
                continue
            burst = rng.standard_normal(length) * np.exp(-np.linspace(0.0, 9.0, length))
            result[int(start) : int(start) + length] += 0.12 * burst
    if labels & {"Vehicle", "Train", "Air conditioning"}:
        b, a = butter(2, min(0.95, 260 / (sample_rate / 2)), btype="low")
        result += 0.35 * lfilter(b, a, white)
    if "Television" in labels:
        b, a = butter(2, [250 / (sample_rate / 2), min(0.96, 3500 / (sample_rate / 2))], btype="band")
        result += 0.12 * lfilter(b, a, white) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * time))
    return result.astype(np.float32)


def _background_asset(
    output: Path,
    environment: str,
    composition: dict[str, list[str]],
    music_present: bool,
    variant: int,
) -> Path:
    labels = {item for values in composition.values() for item in values}
    music_path = _real_music_path(output, variant) if music_present else None
    key_payload = json.dumps(
        {
            "environment": environment,
            "labels": sorted(labels),
            "music": music_present,
            "music_source": music_path.name if music_path else None,
            "music_version": "real_asset.v1" if music_path else None,
            "variant": variant,
        },
        sort_keys=True,
    )
    key = _short_hash(key_payload, 16)
    path = output / "assets/backgrounds" / environment / f"bg_{key}.wav"
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = ASSET_SECONDS * SOURCE_SR
    seed = stable_seed(f"background:{key}", SEED)
    base = generate_noise(NOISE_PROFILE_BY_ENVIRONMENT[environment], samples, SOURCE_SR, seed)
    events = _event_layer(labels, samples, SOURCE_SR, seed + 1)
    value = 0.55 * base + 0.75 * events
    if music_present:
        value += 0.85 * _real_music(samples, output, variant)
    value = np.asarray(value, dtype=np.float32)
    value -= np.mean(value, dtype=np.float32)
    value *= 0.10 / max(rms(value), 1e-8)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 0.9:
        value *= 0.9 / peak
    sf.write(path, value, SOURCE_SR, subtype="PCM_16")
    _json_dump(
        metadata_path,
        {
            "asset_id": key,
            "environment": environment,
            "labels": sorted(labels),
            "music_present": music_present,
            "music_source": music_path.name if music_path else None,
            "duration_seconds": ASSET_SECONDS,
            "sample_rate": SOURCE_SR,
            "generator": "environment events plus real music asset",
            "seed": seed,
            "sha256": _sha256(path),
        },
    )
    return path


def _select_records(
    records: list[dict[str, Any]], limit: int | None, smoke: bool
) -> list[dict[str, Any]]:
    if smoke:
        dimensions = [
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 1
            and r["layers"]["L4_paralinguistic"]["speech_rate_bucket"] == "慢",
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 1
            and r["layers"]["L5_acoustic"]["music_present"]
            and r["layers"]["L4_paralinguistic"]["speech_rate_bucket"] == "常速",
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 2
            and not r["layers"]["L3_temporal"]["speaker_overlap"]
            and r["layers"]["L4_paralinguistic"]["speech_rate_bucket"] == "常速",
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 2
            and r["layers"]["L3_temporal"]["speaker_overlap"],
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 3
            and not r["layers"]["L3_temporal"]["speaker_overlap"],
            lambda r: r["layers"]["L2_speaker"]["speaker_count_target"] == 3
            and r["layers"]["L3_temporal"]["speaker_overlap"],
            lambda r: r["layers"]["L5_acoustic"]["sample_rate_hz_target"] == 8_000,
            lambda r: r["layers"]["L5_acoustic"]["sample_rate_hz_target"] == 16_000,
            lambda r: r["layers"]["L4_paralinguistic"]["speech_rate_bucket"] == "快"
            and r["layers"]["L5_acoustic"]["snr_bucket"] == "高",
        ]
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for predicate in dimensions:
            match = next(item for item in records if item["scene_id"] not in used and predicate(item))
            selected.append(match)
            used.add(match["scene_id"])
        return selected
    return records[:limit] if limit is not None else records


def build_jobs(
    output: Path, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_specs: dict[str, dict[str, Any]] = {}
    edit_specs: dict[str, dict[str, Any]] = {}
    derived_specs: dict[str, dict[str, Any]] = {}
    sessions: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        l1 = record["layers"]["L1_lexical_pragmatic"]
        l2 = record["layers"]["L2_speaker"]
        l3 = record["layers"]["L3_temporal"]
        l4 = record["layers"]["L4_paralinguistic"]
        l5 = record["layers"]["L5_acoustic"]
        profiles = {item["speaker_id"]: item for item in l2["profiles"]}
        language = l1.get("language")
        assigned_voices = {item["speaker_id"]: item for item in record["reference_voices"]}
        dialogue_turns: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(record["script"]["turns"]):
            speaker_index = int(turn["speaker_id"].split("_")[-1]) - 1
            profile = profiles[turn["speaker_id"]]
            voice_index = int(assigned_voices[turn["speaker_id"]]["voice_index"])
            turn_language = turn.get("language") or language or "en"
            base_key = "b_" + _short_hash(f"{voice_index}\n{turn_language}\n{turn['text']}")
            base_path = output / "sources/base" / f"{base_key}.wav"
            native_path = output / "sources/native" / f"{base_key}.wav"
            base_specs.setdefault(
                base_key,
                {
                    "key": base_key,
                    "voice_index": voice_index,
                    "text": turn["text"],
                    "language": turn_language,
                    "target_pitch": profile["pitch"],
                    "output": str(base_path),
                    "native_output": str(native_path),
                    "seed": _task_seed(base_key),
                },
            )
            speed = "常速" if FORCE_NORMAL_SPEED else l4["speech_rate_bucket"]
            speed_id = {"慢": "slow", "常速": "normal", "快": "fast"}[speed]
            rate_factor = RATE_FACTORS[speed_id] if turn_language in {"zh", "en"} else None
            source_key = base_key
            source_path = base_path
            source_stage = "base"
            if rate_factor is not None:
                edit_key = "e_" + _short_hash(f"{base_key}:{rate_factor:.2f}")
                source_key = edit_key
                source_path = output / "sources/edited" / f"{edit_key}.wav"
                source_stage = "edit"
                edit_specs.setdefault(
                    edit_key,
                    {
                        "key": edit_key,
                        "base_key": base_key,
                        "text": turn["text"],
                        "language": turn_language,
                        "rate_factor": rate_factor,
                        "instruction": f'<rate, factor={rate_factor:.2f}>{turn["text"]}</rate>',
                        "edit_kind": "rate",
                        "base_output": str(base_path),
                        "native_input": str(native_path),
                        "output": str(source_path),
                        "seed": _task_seed(edit_key),
                    },
                )
            volume = "normal" if CLEAN_RENDER else profile["speaker_volume"]
            volume_key = "d_" + _short_hash(f"{source_key}:{volume}")
            volume_path = output / "sources/derived" / f"{volume_key}.wav"
            derived_specs.setdefault(
                volume_key,
                {
                    "key": volume_key,
                    "source_key": source_key,
                    "source_stage": source_stage,
                    "source": str(source_path),
                    "output": str(volume_path),
                    "text": turn["text"],
                    "target_speed": (
                        (2.8 if turn_language != "zh" else 4.2)
                        if FORCE_NORMAL_SPEED
                        else (
                            l4.get("speech_rate_target_word_per_sec")
                            if turn_language != "zh"
                            else l4["speech_rate_target_zh_char_per_sec"]
                        )
                    ),
                    "language": turn_language,
                    "speed_bucket": speed_id,
                    "target_pitch": profile["pitch"],
                    "target_volume": volume,
                    "target_rms_dbfs": VOLUME_RMS_DBFS[volume],
                },
            )
            dialogue_turns.append(
                {
                    "utterance_id": f"u{turn_index:03d}",
                    "speaker_index": speaker_index,
                    "text": turn["text"],
                    "language": turn_language,
                    "language_spans": turn.get("language_spans", []),
                    "input_audio_path": str(volume_path),
                    "overlap_hint": "interrupt" if turn["overlap_with_previous"] else "none",
                }
            )
            source_rows.append(
                {
                    "utterance_id": f"u{turn_index:03d}",
                    "base_key": base_key,
                    "source_key": source_key,
                    "derived_key": volume_key,
                    "voice_index": voice_index,
                    "target_pitch": profile["pitch"],
                    "target_volume": volume,
                    "speed_bucket": speed_id,
                    "rate_factor": rate_factor,
                    "language": turn_language,
                }
            )
        dialogue = {
            "scene_id": record["scene_id"],
            "topic": l1["descriptive_topic_target"],
            "language": language or "multilingual",
            "speakers": [
                {
                    "speaker_index": index,
                    "speaker_id": f"speaker_{index + 1}",
                    "role": next(
                        turn["role"]
                        for turn in record["script"]["turns"]
                        if turn["speaker_id"] == f"speaker_{index + 1}"
                    ),
                    "language": next((item["language"] for item in dialogue_turns if item["speaker_index"] == index), language),
                    "reference_voice_index": assigned_voices[f"speaker_{index + 1}"]["voice_index"],
                    "reference_pool": assigned_voices[f"speaker_{index + 1}"]["reference_pool"],
                }
                for index in range(l2["speaker_count_target"])
            ],
            "utterances": dialogue_turns,
        }
        dialogue_path = output / "spec/dialogues" / f"{record['scene_id']}.json"
        _json_dump(dialogue_path, dialogue)
        environment = record.get("acoustic_environment") or record["rir_selection"][
            "target_environment"
        ]
        noise_path = _background_asset(
            output,
            environment,
            l5["noise_composition"],
            l5["music_present"],
            int(record["scene_id"].split("-")[-1]) % 5,
        )
        if CLEAN_RENDER:
            session_environment = None
            session_noise_path = None
            session_noise_environment = None
            session_snr_db = None
            session_rt60 = 0.0
            session_rir_path = None
        else:
            session_environment = environment
            session_noise_path = str(noise_path)
            session_noise_environment = environment
            session_snr_db = l5["snr_db_target"]
            session_rt60 = l5["rt60_sec_target"]
            session_rir_path = _resolve_measured_rir(record)
        sessions.append(
            {
                "id": record["scene_id"],
                "dialogue": f"dialogues/{record['scene_id']}.json",
                "speakers": l2["speaker_count_target"],
                "sample_rate": l5["sample_rate_hz_target"],
                "overlap_ratio": l3["overlap_ratio_target"],
                "snr_db": session_snr_db,
                "rt60_seconds": session_rt60,
                "environment": session_environment,
                "noise_path": session_noise_path,
                "noise_environment": session_noise_environment,
            "language": language,
                "rir_path": session_rir_path,
                "rir_selection": record["rir_selection"],
                "room": _room(l5["reverb_bucket"], l2["speaker_count_target"]),
                "source_rows": source_rows,
                "target_record": record,
            }
        )
    return list(base_specs.values()), list(edit_specs.values()), list(derived_specs.values()), sessions


def _write_specs(
    output: Path,
    base_specs: list[dict[str, Any]],
    edit_specs: list[dict[str, Any]],
    derived_specs: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> None:
    _jsonl_dump(output / "spec/base_specs.jsonl", base_specs)
    _jsonl_dump(output / "spec/edit_specs.jsonl", edit_specs)
    _jsonl_dump(output / "spec/derived_specs.jsonl", derived_specs)
    _json_dump(output / "spec/matrix.json", sessions)
    counts = {
        "scenes": len(sessions),
        "base_specs": len(base_specs),
        "edit_specs": len(edit_specs),
        "derived_specs": len(derived_specs),
        "sample_rates": dict(Counter(str(item["sample_rate"]) for item in sessions)),
        "speaker_counts": dict(Counter(str(item["speakers"]) for item in sessions)),
    }
    _json_dump(output / "spec/counts.json", counts)


def prepare(output: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    for filename in REAL_MUSIC_FILENAMES:
        destination = output / "assets/backgrounds" / filename
        if not destination.is_file():
            source = REAL_MUSIC_SOURCE_ROOT / filename
            if not source.is_file():
                raise FileNotFoundError(f"real music asset not found: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    _load_voice_bank(output)
    base_specs, edit_specs, derived_specs, sessions = build_jobs(output, records)
    _write_specs(output, base_specs, edit_specs, derived_specs, sessions)
    reused = (
        {"base": 0, "edit": 0, "derived": 0}
        if os.environ.get("L1_L5_DISABLE_SOURCE_REUSE", "").lower() in {"1", "true", "yes"}
        else _reuse_v3_sources(output)
    )
    print(json.dumps({"reused_from_v3": reused}, ensure_ascii=False), flush=True)
    return {
        "scenes": len(sessions),
        "base_specs": len(base_specs),
        "edit_specs": len(edit_specs),
        "derived_specs": len(derived_specs),
    }


def _split_base_text(text: str, language: str | None) -> list[str]:
    """Bound vocoder memory while preserving one logical source utterance."""
    import re

    # Chinese/Japanese/Korean prompts can still expand to many acoustic patches
    # because each character is modeled separately; keep those requests small.
    limit = 20
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    chunks: list[str] = []
    for sentence in (item.strip() for item in sentences if item.strip()):
        remaining = sentence
        while len(remaining) > limit:
            window = remaining[: limit + 1]
            candidates = [window.rfind(mark) + 1 for mark in ("，", ",", "、", ":", "：", " ")]
            cut = max((value for value in candidates if value >= limit // 2), default=limit)
            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            chunks.append(remaining)
    return chunks or [text]


def _tts_worker(
    stage: str, gpu: int, worker_slot: int, specs_path: str, voice_bank_path: str
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    specs = _load_jsonl(Path(specs_path))
    voices = {int(key): value for key, value in json.loads(Path(voice_bank_path).read_text()).items()}
    from saasr_mini.editing import DotsTtsEditBackend
    from saasr_mini.speech import DotsTtsBackend
    force_edit_fallback = (
        stage == "edit"
        and os.environ.get("L1_L5_FORCE_EDIT_FALLBACK", "").lower()
        in {"1", "true", "yes"}
    )
    strict_edit = (
        stage == "edit"
        and os.environ.get("L1_L5_STRICT_DOTS_EDIT", "").lower()
        in {"1", "true", "yes"}
    )
    if strict_edit and force_edit_fallback:
        raise RuntimeError("strict Dots edit mode cannot be combined with forced fallback")

    if stage == "base":
        backend: Any = DotsTtsBackend(
            BASE_MODEL,
            voices,
            SOURCE_SR,
            python=Path(os.environ.get("DOTS_BASE_PYTHON", WORKSPACE / "models/runtime/.venv/bin/python")),
            server=WORKSPACE / "scripts/dots_tts_base_rpc_server.py",
            device=f"cuda:{gpu}",
            precision="float16",
            num_steps=10,
            guidance_scale=1.2,
            speaker_scale=1.5,
            seed=SEED,
            # Enhanced FLEURS prompts are about seven seconds (~40 patches).
            # Leave enough decode room for the requested utterance instead of
            # letting the prompt consume the 48-patch generation budget.
            max_generate_length=96,
            work_dir=WORKSPACE / ".runtime/l1_l5_base" / f"gpu{gpu}_worker{worker_slot}",
        )
    else:
        backend = None
        if not force_edit_fallback:
            backend = DotsTtsEditBackend(
                EDIT_MODEL,
                SOURCE_SR,
                python=Path(os.environ.get("DOTS_EDIT_PYTHON", WORKSPACE / "models/runtime/.venv/bin/python")),
                server=WORKSPACE / "scripts/dots_tts_edit_rpc_server.py",
                device=f"cuda:{gpu}",
                precision="bfloat16",
                num_steps=10,
                guidance_scale=1.2,
                speaker_scale=1.5,
                seed=SEED,
                max_generate_length=1200,
                work_dir=WORKSPACE / ".runtime/l1_l5_edit" / f"gpu{gpu}_worker{worker_slot}",
            )
    failures: list[dict[str, str]] = []
    edit_backend_error: str | None = (
        "forced fallback: dots.tts.edit disabled for this run"
        if force_edit_fallback
        else None
    )
    try:
        for item_index, spec in enumerate(specs, 1):
            out = Path(spec["output"])
            done = out.with_suffix(out.suffix + ".json")
            if out.is_file() and done.is_file():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                if stage == "base":
                    chunks = _split_base_text(spec["text"], spec.get("language"))
                    requested_voice = int(spec["voice_index"])
                    voice_candidates = [requested_voice]
                    # A few multilingual prompts can deterministically emit
                    # EOS immediately. Retry the complete logical utterance
                    # with the other reference from the same language before
                    # marking the source as failed.
                    if str(voices.get(requested_voice, {}).get("source", "")).startswith("FLEURS"):
                        voice_candidates.extend(
                            index
                            for index, voice in sorted(voices.items())
                            if index != requested_voice
                            and voice.get("language") == spec.get("language")
                            and str(voice.get("source", "")).startswith("FLEURS")
                        )
                    audio_parts: list[np.ndarray] = []
                    native_parts: list[np.ndarray] = []
                    selected_voice = requested_voice
                    attempt_errors: list[str] = []
                    for candidate_voice in voice_candidates:
                        try:
                            candidate_audio: list[np.ndarray] = []
                            candidate_native: list[np.ndarray] = []
                            for chunk_index, chunk in enumerate(chunks):
                                backend.seed = (int(spec["seed"]) + chunk_index) % (2**32 - 1)
                                candidate_audio.append(
                                    _trim_active_edges(
                                        backend.synthesize(
                                            chunk,
                                            int(candidate_voice),
                                            language=spec.get("language"),
                                        ),
                                        SOURCE_SR,
                                    )
                                )
                                if backend.last_native_output_path is None:
                                    raise RuntimeError("dots.tts did not expose its native output")
                                native_audio, native_rate = sf.read(
                                    backend.last_native_output_path, dtype="float32"
                                )
                                if int(native_rate) != 48_000:
                                    raise RuntimeError(f"unexpected dots.tts native rate: {native_rate}")
                                candidate_native.append(
                                    _trim_active_edges(np.asarray(native_audio, dtype=np.float32), 48_000)
                                )
                            audio_parts = candidate_audio
                            native_parts = candidate_native
                            selected_voice = candidate_voice
                            break
                        except Exception as exc:
                            attempt_errors.append(f"voice {candidate_voice}: {exc!r}")
                    if not audio_parts:
                        raise RuntimeError("; ".join(attempt_errors))
                    source_pause = np.zeros(round(SOURCE_SR * 0.08), dtype=np.float32)
                    native_pause = np.zeros(round(48_000 * 0.08), dtype=np.float32)
                    audio = np.concatenate(
                        [part for index, value in enumerate(audio_parts) for part in ((source_pause,) if index else ()) + (value,)]
                    )
                    expected_units = _text_units(spec["text"], spec.get("language", "en"))
                    speech_span = _active_span_seconds(audio, SOURCE_SR)
                    if not np.isfinite(audio).all() or (
                        expected_units >= 5 and speech_span < 0.20
                    ) or _active_rms(audio) < 1e-5:
                        raise RuntimeError(
                            f"dots.tts produced invalid base speech: span={speech_span:.3f}s, "
                            f"units={expected_units}"
                        )
                    write_audio(out, audio, SOURCE_SR)
                    native = Path(spec["native_output"])
                    native.parent.mkdir(parents=True, exist_ok=True)
                    native_audio = np.concatenate(
                        [part for index, value in enumerate(native_parts) for part in ((native_pause,) if index else ()) + (value,)]
                    )
                    write_audio(native, native_audio, 48_000)
                    metadata = {
                        "stage": stage,
                        "key": spec["key"],
                        "seed": spec["seed"],
                        "voice_index": spec["voice_index"],
                        "voice_index_used": selected_voice,
                        "voice_fallback": selected_voice != requested_voice,
                        "text": spec["text"],
                        "language": spec.get("language"),
                        "synthesis_chunks": chunks,
                        "synthesis_chunk_count": len(chunks),
                        "sample_rate": SOURCE_SR,
                        "sha256": _sha256(out),
                        "native_sha256": _sha256(native),
                    }
                else:
                    if backend is not None:
                        backend.seed = int(spec["seed"])
                        backend._edit_index = 0
                    base_audio, base_rate = sf.read(spec["base_output"], dtype="float32")
                    fallback_error = edit_backend_error
                    if fallback_error is None:
                        try:
                            result = backend.edit(
                                base_audio,
                                int(base_rate),
                                source_text=spec["text"],
                                instruction=spec["instruction"],
                                source_audio_path=Path(spec["native_input"]),
                            )
                            edited_audio = result.audio
                            edit_runtime = result.metadata
                            source_span = _active_span_seconds(np.asarray(base_audio), int(base_rate))
                            edited_span = _active_span_seconds(edited_audio, SOURCE_SR)
                            expected_units = _text_units(spec["text"], spec.get("language", "en"))
                            if (
                                not np.isfinite(edited_audio).all()
                                or _active_rms(edited_audio) < 1e-5
                                or (expected_units >= 5 and edited_span < 0.20)
                                or edited_span < max(0.20, source_span * 0.25)
                            ):
                                raise RuntimeError(
                                    f"dots.tts.edit produced invalid speech: "
                                    f"source_span={source_span:.3f}s edited_span={edited_span:.3f}s"
                                )
                        except Exception as exc:
                            edit_backend_error = repr(exc)
                            fallback_error = edit_backend_error
                            if strict_edit:
                                raise RuntimeError(
                                    f"dots.tts.edit failed in strict mode: {edit_backend_error}"
                                ) from exc
                    if fallback_error is not None:
                        if strict_edit:
                            raise RuntimeError(
                                f"dots.tts.edit unavailable in strict mode: {fallback_error}"
                            )
                        edited_audio = _fallback_time_stretch(
                            base_audio, int(base_rate), float(spec["rate_factor"])
                        )
                        edit_runtime = {
                            "backend": "fallback_time_stretch",
                            "fallback_error": fallback_error,
                            "rate_factor": float(spec["rate_factor"]),
                        }
                    write_audio(out, edited_audio, SOURCE_SR)
                    source_span = _active_span_seconds(np.asarray(base_audio), int(base_rate))
                    edited_span = _active_span_seconds(edited_audio, SOURCE_SR)
                    metadata = {
                        "stage": stage,
                        "key": spec["key"],
                        "seed": spec["seed"],
                        "instruction": spec["instruction"],
                        "rate_factor": spec["rate_factor"],
                        "text": spec["text"],
                        "sample_rate": SOURCE_SR,
                        "source_active_span_seconds": round(source_span, 6),
                        "edited_active_span_seconds": round(edited_span, 6),
                        "active_span_ratio": round(edited_span / max(source_span, 1e-8), 6),
                        "runtime": edit_runtime,
                        "edit_backend_fallback": fallback_error is not None,
                        "sha256": _sha256(out),
                    }
                _json_dump(done, metadata)
                print(f"[{stage} gpu={gpu}] {item_index}/{len(specs)} {spec['key']}", flush=True)
            except Exception as exc:  # keep the remaining shard recoverable
                failures.append({"key": spec["key"], "error": repr(exc)})
                print(f"[{stage} gpu={gpu}] FAILED {spec['key']}: {exc}", file=sys.stderr, flush=True)
    finally:
        if backend is not None:
            backend.close()
    if failures:
        _json_dump(Path(specs_path).with_suffix(".failures.json"), failures)
        raise RuntimeError(f"{stage} gpu={gpu} failed for {len(failures)} specs")


def run_tts_stage(stage: str, output: Path, gpus: list[int]) -> None:
    specs = _load_jsonl(output / f"spec/{stage}_specs.jsonl")
    only_keys = {
        value.strip()
        for value in os.environ.get("L1_L5_ONLY_KEYS", "").split(",")
        if value.strip()
    }
    if only_keys:
        specs = [spec for spec in specs if spec["key"] in only_keys]
        if not specs:
            print(f"[{stage}] no specs matched L1_L5_ONLY_KEYS", flush=True)
            return
    assignments = [[] for _ in gpus]
    for index, spec in enumerate(specs):
        assignments[index % len(gpus)].append(spec)
    worker_root = output / "spec/workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_slot, (gpu, assigned) in enumerate(zip(gpus, assignments, strict=True)):
        if not assigned:
            continue
        shard = worker_root / f"{stage}_gpu{gpu}_worker{worker_slot}.jsonl"
        _jsonl_dump(shard, assigned)
        process = context.Process(
            target=_tts_worker,
            args=(stage, gpu, worker_slot, str(shard), str(output / "references/voice_bank.json")),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode]
    if failures:
        raise RuntimeError(f"{stage} workers failed with exit codes {failures}")


def derive_sources(output: Path) -> None:
    specs = _load_jsonl(output / "spec/derived_specs.jsonl")
    failures: list[dict[str, str]] = []
    for index, spec in enumerate(specs, 1):
        out = Path(spec["output"])
        done = out.with_suffix(out.suffix + ".json")
        if out.is_file() and done.is_file():
            existing = json.loads(done.read_text(encoding="utf-8"))
            if existing.get("derivation_version") == DERIVATION_VERSION:
                continue
        try:
            audio, sample_rate = sf.read(spec["source"], dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            if int(sample_rate) != SOURCE_SR:
                common = math.gcd(int(sample_rate), SOURCE_SR)
                audio = resample_poly(audio, SOURCE_SR // common, int(sample_rate) // common)
            speed_calibrated, atempo, initial_rate, calibrated_rate = _calibrate_speed(
                audio, spec["text"], float(spec["target_speed"]), spec.get("language", "zh")
            )
            if CLEAN_RENDER:
                speed_calibrated = _compress_long_silences(
                    speed_calibrated, SOURCE_SR, max_silence_seconds=0.15
                )
                speed_calibrated, extra_atempo, _, calibrated_rate = _calibrate_speed(
                    speed_calibrated, spec["text"], float(spec["target_speed"]), spec.get("language", "zh")
                )
                atempo.extend(extra_atempo)
            derived, gain_db = _normalize_active_rms(
                speed_calibrated, float(spec["target_rms_dbfs"])
            )
            write_audio(out, derived, SOURCE_SR)
            active_span = _active_span_seconds(derived, SOURCE_SR)
            unit_count = _text_units(spec["text"], spec.get("language", "zh"))
            actual_rate = unit_count / max(active_span, 1e-8)
            target_speed = float(spec["target_speed"])
            if (
                not np.isfinite(derived).all()
                or (unit_count >= 5 and active_span < 0.20)
                or actual_rate > max(20.0, target_speed * 8.0)
            ):
                raise ValueError(
                    f"derived speech is invalid: span={active_span:.3f}s, "
                    f"rate={actual_rate:.2f}/s, units={unit_count}"
                )
            _json_dump(
                done,
                {
                    **spec,
                    "derivation_version": DERIVATION_VERSION,
                    "sample_rate": SOURCE_SR,
                    "gain_db": round(gain_db, 4),
                    "atempo_factors": [round(value, 8) for value in atempo],
                    "pre_calibration_text_units_per_sec": round(initial_rate, 4),
                    "calibrated_text_units_per_sec": round(calibrated_rate, 4),
                    "active_rms_dbfs": round(20 * math.log10(max(_active_rms(derived), 1e-12)), 4),
                    "active_span_seconds": round(active_span, 6),
                    "estimated_text_units_per_sec": round(actual_rate, 4),
                    "sha256": _sha256(out),
                },
            )
            if index % 100 == 0:
                print(f"[derive] {index}/{len(specs)}", flush=True)
        except Exception as exc:
            failures.append({"key": spec["key"], "error": repr(exc)})
    if failures:
        _json_dump(output / "spec/derive_failures.json", failures)
        raise RuntimeError(f"derive failed for {len(failures)} specs")


def _scene_config(session: dict[str, Any]) -> dict[str, Any]:
    config = {
        "id": session["id"],
        "dialogue": session["dialogue"],
        "speakers": session["speakers"],
        "overlap_ratio": session["overlap_ratio"],
        "max_simultaneous_speakers": 2,
        "snr_db": session["snr_db"],
        "environment": session["environment"],
        "noise_path": session["noise_path"],
        "noise_environment": session["noise_environment"],
        "rir_seed_key": session["id"],
        "noise_seed_key": session["id"],
        "room": {**session["room"], "rt60_seconds": session["rt60_seconds"]},
    }
    if session.get("rir_path"):
        config["rir_path"] = session["rir_path"]
        # Some measured RIR filenames (for example, shower/bathroom captures)
        # do not contain a canonical environment token. The scene's selected
        # environment is the explicit source of truth in that case.
        config["rir_environment"] = session["environment"]
    return config


def render_scenes(output: Path) -> list[dict[str, Any]]:
    sessions = json.loads((output / "spec/matrix.json").read_text(encoding="utf-8"))
    all_manifests: list[dict[str, Any]] = []
    for sample_rate in sorted({int(item["sample_rate"]) for item in sessions}):
        group = [item for item in sessions if int(item["sample_rate"]) == sample_rate]
        config = {
            "seed": SEED,
            "audio": {"sample_rate": sample_rate, "peak_dbfs": -1.0},
            "speech_backend": {"type": "fixture"},
            "speech_edit_backend": {"type": "none"},
            "scenes": [_scene_config(item) for item in group],
        }
        config_path = output / f"spec/batch_{sample_rate}.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifests = simulate(config_path, output)
        by_id = {item["id"]: item for item in group}
        for manifest in manifests:
            session = by_id[manifest["session_id"]]
            record = session["target_record"]
            manifest.update(
                {
                    "dataset_version": output.name,
                    "grid_version": record["grid_version"],
                    "acoustic_scene_id": record["acoustic_scene_id"],
                    "acoustic_environment": session["environment"],
                    "target_layers": record["layers"],
                    "script": record["script"],
                    "source_rows": session["source_rows"],
                    "background_asset": session["noise_path"],
                }
            )
            _json_dump(output / "sessions" / manifest["session_id"] / "manifest.json", manifest)
        all_manifests.extend(manifests)
    all_manifests.sort(key=lambda item: item["session_id"])
    _json_dump(output / "manifests/sessions.json", all_manifests)
    _jsonl_dump(
        output / "exports/sa_asr.jsonl",
        [
            {
                "id": item["session_id"],
                "audio": item["audio_path"],
                "sample_rate": item["sample_rate"],
                "turns": [
                    {key: turn[key] for key in ("speaker", "start", "end", "text")}
                    for turn in item["turns"]
                ],
            }
            for item in all_manifests
        ],
    )
    return all_manifests


def _bucket_speed(value: float, language: str = "zh") -> str:
    if language != "zh":
        return "slow" if value < 2.4 else "fast" if value > 3.3 else "normal"
    return "slow" if value < 3.0 else "fast" if value > 6.0 else "normal"


def _rir_metrics(path: Path) -> tuple[float | None, float | None]:
    values, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if not values.size or not np.any(np.abs(values) > 1e-12):
        return None, None
    direct = int(np.argmax(np.abs(values)))
    tail = values[direct:]
    split = min(tail.size, round(0.05 * sample_rate))
    early = float(np.sum(np.square(tail[:split])))
    late = float(np.sum(np.square(tail[split:])))
    c50 = 10.0 * math.log10(max(early, 1e-12) / max(late, 1e-12))
    energy = np.cumsum(np.square(tail)[::-1])[::-1]
    energy_db = 10.0 * np.log10(np.maximum(energy / max(energy[0], 1e-20), 1e-20))
    mask = (energy_db <= -5.0) & (energy_db >= -25.0)
    if np.count_nonzero(mask) < 2:
        return None, c50
    times = np.arange(tail.size, dtype=np.float64) / sample_rate
    slope, _ = np.polyfit(times[mask], energy_db[mask], 1)
    rt60 = -60.0 / slope if slope < 0 else None
    return rt60, c50


def run_qc(output: Path) -> dict[str, Any]:
    sessions = json.loads((output / "spec/matrix.json").read_text(encoding="utf-8"))
    manifests = json.loads((output / "manifests/sessions.json").read_text(encoding="utf-8"))
    by_session = {item["id"]: item for item in sessions}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rir_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        scene_id = manifest["session_id"]
        session = by_session[scene_id]
        mixture_path = output / manifest["audio_path"]
        try:
            info = sf.info(mixture_path)
        except Exception as exc:
            failures.append({"scene_id": scene_id, "check": "wav_readable", "observed": repr(exc)})
            continue
        expected_rate = int(session["sample_rate"])
        if info.samplerate != expected_rate or info.channels != 1 or info.frames <= 0:
            failures.append(
                {
                    "scene_id": scene_id,
                    "check": "audio_format",
                    "expected": {"sample_rate": expected_rate, "channels": 1, "nonempty": True},
                    "observed": {"sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames},
                }
            )
        if session["snr_db"] is not None:
            if manifest["achieved_snr_db"] is None or abs(float(manifest["achieved_snr_db"]) - float(session["snr_db"])) > 0.1:
                failures.append({"scene_id": scene_id, "check": "snr", "expected": session["snr_db"], "observed": manifest["achieved_snr_db"]})
        if abs(float(manifest["overlap_ratio"]) - float(session["overlap_ratio"])) > 0.04:
            failures.append({"scene_id": scene_id, "check": "overlap", "expected": session["overlap_ratio"], "observed": manifest["overlap_ratio"]})
        if int(manifest["num_speakers"]) != int(session["speakers"]):
            failures.append({"scene_id": scene_id, "check": "speaker_count", "expected": session["speakers"], "observed": manifest["num_speakers"]})
        rir_path = output / "sessions" / scene_id / "rirs/speaker_00.wav"
        rt60, c50 = _rir_metrics(rir_path)
        target_c50 = session["target_record"]["layers"]["L5_acoustic"]["c50_db_target"]
        rir_rows.append({"scene_id": scene_id, "target_rt60": session["rt60_seconds"], "measured_rt60": round(rt60, 4) if rt60 else None, "target_c50": target_c50, "measured_c50": round(c50, 4) if c50 else None})
        if session["rt60_seconds"] != 0:
            if rt60 is None or abs(rt60 - float(session["rt60_seconds"])) > 0.25:
                warnings.append({"scene_id": scene_id, "check": "rir_rt60", "target": session["rt60_seconds"], "observed": rt60})
            if c50 is None or abs(c50 - float(target_c50)) > 6.0:
                warnings.append({"scene_id": scene_id, "check": "rir_c50", "target": target_c50, "observed": c50})
    derived = _load_jsonl(output / "spec/derived_specs.jsonl")
    speed_counts: Counter[str] = Counter()
    speed_mismatches: list[dict[str, Any]] = []
    for spec in derived:
        metadata_path = Path(spec["output"]).with_suffix(".wav.json")
        if not metadata_path.is_file():
            failures.append({"source_id": spec["key"], "check": "derived_metadata_missing"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        language = spec.get("language", "zh")
        observed_rate = float(metadata["estimated_text_units_per_sec"])
        observed = _bucket_speed(observed_rate, language)
        speed_counts[observed] += 1
        if observed != spec["speed_bucket"]:
            speed_mismatches.append(
                {
                    "source_id": spec["key"],
                    "target": spec["speed_bucket"],
                    "observed": observed,
                    "estimated_text_units_per_sec": observed_rate,
                    "language": language,
                }
            )
    if speed_mismatches:
        warnings.extend({"check": "source_speed_proxy", **item} for item in speed_mismatches)
    index_rows = []
    for manifest in manifests:
        index_rows.append(
            {
                "scene_id": manifest["session_id"],
                "audio": manifest["audio_path"],
                "duration_seconds": manifest["duration_seconds"],
                "sample_rate": manifest["sample_rate"],
                "speaker_count": manifest["num_speakers"],
                "acoustic_scene_id": manifest["acoustic_scene_id"],
                "acoustic_environment": manifest["environment"],
                "snr_db": manifest["achieved_snr_db"],
                "rt60_seconds": manifest["rt60_seconds"],
                "overlap_ratio": manifest["overlap_ratio"],
                "music_present_target": manifest["target_layers"]["L5_acoustic"]["music_present"],
            }
        )
    _jsonl_dump(output / "index.jsonl", index_rows)
    _jsonl_dump(
        output / "tagger/tagger_manifest.jsonl",
        [
            {
                "id": row["scene_id"],
                "audio_path": str((output / row["audio"]).resolve()),
                "transcript": by_session[row["scene_id"]]["target_record"]["layers"]["L1_lexical_pragmatic"]["reference_transcript"],
            }
            for row in index_rows
        ],
    )
    _jsonl_dump(output / "qc/rir_metrics.jsonl", rir_rows)
    summary = {
        "dataset": output.name,
        "scene_count": len(manifests),
        "duration_hours": round(sum(float(item["duration_seconds"]) for item in manifests) / 3600.0, 4),
        "sample_rate_counts": dict(Counter(str(item["sample_rate"]) for item in manifests)),
        "speaker_count_counts": dict(Counter(str(item["num_speakers"]) for item in manifests)),
        "snr_counts": dict(Counter(str(item["target_snr_db"]) for item in manifests)),
        "music_counts": dict(Counter(str(item["target_layers"]["L5_acoustic"]["music_present"]).lower() for item in manifests)),
        "speed_proxy_counts": dict(speed_counts),
        "speed_proxy_mismatch_count": len(speed_mismatches),
        "hard_failure_count": len(failures),
        "warning_count": len(warnings),
        "qc_passed": not failures,
    }
    _json_dump(output / "qc_report.json", {"summary": summary, "failures": failures, "warnings": warnings})
    _json_dump(output / "summary.json", summary)
    readme = f"""# {output.name}

Generated from `data/scene_scripts_v4.jsonl` with dots.tts-base,
dots.tts.edit, and the deterministic saasr-mini acoustic renderer. English
multi-speaker scenes use AMI references; Chinese scenes use the AliMeeting and
AISHELL-3 reference pools; extra-language scenes use FLEURS references. Per-turn
language and code-switch spans are retained in the dialogue metadata. Measured
RIR selections are passed through when available, with synthetic RIR used for
the remaining scenes.

- Scenes: {summary['scene_count']}
- Duration: {summary['duration_hours']} hours
- Sample rates: {summary['sample_rate_counts']}
- Speaker counts: {summary['speaker_count_counts']}
- Hard QC failures: {summary['hard_failure_count']}
- QC warnings: {summary['warning_count']}

Key files:

- `index.jsonl`: compact dataset index
- `manifests/sessions.json`: full rendering and L1-L5 target metadata
- `sessions/<scene_id>/`: mixture, stems, utterances, RIR, RTTM, and manifest
- `sources/`: deduplicated base, native, edited, and level-derived speech
- `spec/`: source jobs, dialogues, render matrix, and batch configs
- `tagger/tagger_manifest.jsonl`: input manifest for downstream sure-tagger validation
- `qc_report.json`: deterministic generation checks and proxy warnings
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _load_prepared(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _load_jsonl(output / "spec/base_specs.jsonl"),
        _load_jsonl(output / "spec/edit_specs.jsonl"),
        _load_jsonl(output / "spec/derived_specs.jsonl"),
        json.loads((output / "spec/matrix.json").read_text(encoding="utf-8")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--stage", choices=("prepare", "base", "edit", "derive", "render", "qc", "all"), default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument(
        "--scene-ids",
        help="comma-separated scene IDs to prepare instead of the full scene script",
    )
    parser.add_argument("--smoke", action="store_true", help="build a nine-scene coverage sample")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    stages = ["prepare", "base", "edit", "derive", "render", "qc"] if args.stage == "all" else [args.stage]
    if "prepare" in stages:
        source_records = _load_jsonl(SCENE_SOURCE)
        if args.scene_ids:
            scene_ids = {value.strip() for value in args.scene_ids.split(",") if value.strip()}
            source_records = [record for record in source_records if record.get("scene_id") in scene_ids]
            missing = scene_ids - {record.get("scene_id") for record in source_records}
            if missing:
                raise ValueError(f"unknown scene IDs: {sorted(missing)}")
            records = source_records
        else:
            records = _select_records(source_records, args.limit_scenes, args.smoke)
        counts = prepare(output, records)
        print(json.dumps({"stage": "prepare", "output": str(output), **counts}, ensure_ascii=False), flush=True)
    else:
        _load_prepared(output)
    for stage in stages:
        if stage == "prepare":
            continue
        if stage in {"base", "edit"}:
            run_tts_stage(stage, output, gpus)
        elif stage == "derive":
            derive_sources(output)
        elif stage == "render":
            manifests = render_scenes(output)
            print(json.dumps({"stage": stage, "scenes": len(manifests)}), flush=True)
        elif stage == "qc":
            print(json.dumps({"stage": stage, **run_qc(output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
