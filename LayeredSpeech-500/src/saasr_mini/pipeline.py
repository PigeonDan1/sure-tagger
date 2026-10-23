from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audio import convolve_same_start, load_audio, peak_dbfs, rms, stable_seed, write_audio
from .editing import SpeechEditBackend, build_speech_edit_backend
from .environment import canonical_environment, validate_scene_environment
from .noise import PROFILE_METADATA, generate_legacy_noise, generate_noise, load_noise_asset
from .scheduling import max_concurrency, schedule
from .schemas import Config, Dialogue, Scene, Utterance
from .speech import SpeechBackend, build_speech_backend


def _triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _validate_rir_paths(scene: Scene) -> None:
    """Reject mixed measured RIR assets without same-room metadata.

    The current RIR library does not carry a machine-readable room/family
    identifier.  Using unrelated measured assets for different speakers would
    therefore create a physically inconsistent scene.  A single asset can be
    reused for all speakers; a future family registry can relax this check.
    """
    if scene.rir_paths is None:
        return
    unique_paths = {path.resolve() for path in scene.rir_paths}
    if len(unique_paths) > 1:
        listed = ", ".join(path.name for path in scene.rir_paths)
        raise ValueError(
            f"{scene.scene_id}: rir_paths must reference one shared RIR asset "
            "for every speaker; mixed assets are rejected until a verified "
            f"same-room RIR family is configured ({listed})"
        )


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Config must be a mapping")
    audio = raw.get("audio", {})
    scenes: list[Scene] = []
    for item in raw.get("scenes", []):
        room = item["room"]
        positions = [_triplet(point, "speaker position") for point in room["speaker_positions_m"]]
        scene = Scene(
            scene_id=str(item["id"]),
            dialogue=(path.parent / item["dialogue"]).resolve(),
            speakers=int(item["speakers"]),
            overlap_ratio=float(item["overlap_ratio"]),
            max_simultaneous_speakers=int(item.get("max_simultaneous_speakers", 2)),
            room_dimensions_m=_triplet(room["dimensions_m"], "room dimensions"),
            microphone_position_m=_triplet(room["microphone_position_m"], "microphone position"),
            speaker_positions_m=positions,
            rt60_seconds=float(room["rt60_seconds"]),
            snr_db=float(item["snr_db"]) if item.get("snr_db") is not None else None,
            environment=canonical_environment(item.get("environment")),
            noise_profile=(
                str(item["noise_profile"]) if item.get("noise_profile") is not None else None
            ),
            noise_path=(
                (path.parent / str(item["noise_path"])).resolve()
                if item.get("noise_path") is not None
                else None
            ),
            noise_environment=canonical_environment(item.get("noise_environment")),
            rir_path=(
                (path.parent / str(item["rir_path"])).resolve()
                if item.get("rir_path") is not None
                else None
            ),
            rir_environment=canonical_environment(item.get("rir_environment")),
            rir_paths=(
                [
                    (path.parent / str(rir_path)).resolve()
                    for rir_path in item["rir_paths"]
                ]
                if item.get("rir_paths") is not None
                else None
            ),
            rir_seed_key=(
                str(item["rir_seed_key"]) if item.get("rir_seed_key") is not None else None
            ),
            noise_seed_key=(
                str(item["noise_seed_key"])
                if item.get("noise_seed_key") is not None
                else None
            ),
        )
        if not 1 <= scene.speakers <= 12:
            raise ValueError(f"{scene.scene_id}: speakers must be between 1 and 12")
        if len(positions) != scene.speakers:
            raise ValueError(f"{scene.scene_id}: one room position is required per speaker")
        if scene.noise_profile is not None and scene.noise_path is not None:
            raise ValueError(f"{scene.scene_id}: choose noise_profile or noise_path, not both")
        if scene.noise_profile is not None and scene.noise_profile not in PROFILE_METADATA:
            choices = ", ".join(PROFILE_METADATA)
            raise ValueError(
                f"{scene.scene_id}: unknown noise_profile {scene.noise_profile!r}; "
                f"choose one of: {choices}"
            )
        if scene.noise_path is not None and not scene.noise_path.is_file():
            raise FileNotFoundError(
                f"{scene.scene_id}: noise asset does not exist: {scene.noise_path}"
            )
        if scene.rir_path is not None and not scene.rir_path.is_file():
            raise FileNotFoundError(
                f"{scene.scene_id}: RIR asset does not exist: {scene.rir_path}"
            )
        if scene.rir_paths is not None:
            if len(scene.rir_paths) != scene.speakers:
                raise ValueError(
                    f"{scene.scene_id}: one rir_paths entry is required per speaker"
                )
            missing_rirs = [rir for rir in scene.rir_paths if not rir.is_file()]
            if missing_rirs:
                raise FileNotFoundError(
                    f"{scene.scene_id}: RIR assets do not exist: {missing_rirs}"
                )
        if scene.rir_path is not None and scene.rir_paths is not None:
            raise ValueError(f"{scene.scene_id}: choose rir_path or rir_paths, not both")
        _validate_rir_paths(scene)
        validate_scene_environment(
            scene_id=scene.scene_id,
            environment=scene.environment,
            noise_profile=scene.noise_profile,
            noise_path=scene.noise_path,
            noise_environment=scene.noise_environment,
            rir_paths=(
                scene.rir_paths
                if scene.rir_paths is not None
                else ([scene.rir_path] if scene.rir_path is not None else [])
            ),
            rir_environment=scene.rir_environment,
        )
        if scene.rt60_seconds < 0:
            raise ValueError(f"{scene.scene_id}: rt60_seconds must be non-negative")
        if not 0 <= scene.overlap_ratio < 0.8:
            raise ValueError(f"{scene.scene_id}: invalid overlap ratio")
        scenes.append(scene)
    if not scenes:
        raise ValueError("Config must define at least one scene")
    speech_backend = raw.get("speech_backend") or {}
    speech_edit_backend = raw.get("speech_edit_backend") or {}
    if not isinstance(speech_backend, dict):
        raise TypeError("speech_backend must be a mapping")
    if not isinstance(speech_edit_backend, dict):
        raise TypeError("speech_edit_backend must be a mapping")
    return Config(
        seed=int(raw.get("seed", 0)),
        sample_rate=int(audio.get("sample_rate", 16_000)),
        peak_dbfs=float(audio.get("peak_dbfs", -1.0)),
        scenes=scenes,
        speaker_rms_dbfs=(
            float(audio["speaker_rms_dbfs"])
            if audio.get("speaker_rms_dbfs") is not None
            else None
        ),
        speech_backend=speech_backend,
        speech_edit_backend=speech_edit_backend,
    )


def load_dialogue(path: Path, expected_speakers: int) -> Dialogue:
    raw = json.loads(path.read_text(encoding="utf-8"))
    speakers = raw["speakers"]
    utterances = [
        Utterance(
            utterance_id=str(item["utterance_id"]),
            speaker_index=int(item["speaker_index"]),
            text=str(item["text"]),
            overlap_hint=str(item.get("overlap_hint", "none")),
            instructions=(
                str(item["instructions"]).strip()
                if item.get("instructions") is not None
                else None
            ),
            edit_instruction=(
                str(item["edit_instruction"]).strip()
                if item.get("edit_instruction") is not None
                else None
            ),
            input_audio_path=(
                (path.parent / str(item["input_audio_path"])).resolve()
                if item.get("input_audio_path") is not None
                else None
            ),
        )
        for item in raw["utterances"]
    ]
    indices = {int(item["speaker_index"]) for item in speakers}
    if indices != set(range(expected_speakers)):
        raise ValueError(f"{path.name}: speaker indices must be 0..{expected_speakers - 1}")
    if {item.speaker_index for item in utterances} != indices:
        raise ValueError(f"{path.name}: every speaker must have at least one utterance")
    if any(item.overlap_hint not in {"none", "backchannel", "interrupt"} for item in utterances):
        raise ValueError(f"{path.name}: unknown overlap hint")
    return Dialogue(
        scene_id=str(raw["scene_id"]),
        topic=str(raw["topic"]),
        language=str(raw["language"]),
        speakers=speakers,
        utterances=utterances,
    )


def _rir(scene: Scene, speaker_index: int, sample_rate: int, seed: int) -> np.ndarray:
    """Load a selected measured RIR or generate the legacy approximation."""
    rir_path = (
        scene.rir_paths[speaker_index]
        if scene.rir_paths is not None
        else scene.rir_path
    )
    if rir_path is not None:
        response = load_audio(rir_path, sample_rate)
        response = np.asarray(response, dtype=np.float32)
        if not response.size or not np.any(np.isfinite(response)):
            raise ValueError(f"RIR asset is empty or invalid: {scene.rir_path}")
        response = np.nan_to_num(response, copy=False)
        source = np.asarray(scene.speaker_positions_m[speaker_index], dtype=np.float64)
        microphone = np.asarray(scene.microphone_position_m, dtype=np.float64)
        distance = float(np.linalg.norm(source - microphone))
        response = response / max(float(np.sqrt(np.sum(np.square(response)))), 1e-8)
        response *= 1.0 / max(1.0, distance)
        return response.astype(np.float32)
    if scene.rt60_seconds == 0:
        # A unit impulse is the explicit clean/no-reverb condition.
        return np.ones(1, dtype=np.float32)
    source = np.asarray(scene.speaker_positions_m[speaker_index], dtype=np.float64)
    microphone = np.asarray(scene.microphone_position_m, dtype=np.float64)
    distance = float(np.linalg.norm(source - microphone))
    length = max(1, round(min(1.2, scene.rt60_seconds * 1.15) * sample_rate))
    result = np.zeros(length, dtype=np.float64)
    result[0] = 1.0 / max(1.0, distance)

    dimensions = np.asarray(scene.room_dimensions_m, dtype=np.float64)
    reflection_distances: list[float] = []
    for axis in range(3):
        low_image = source.copy()
        low_image[axis] *= -1
        high_image = source.copy()
        high_image[axis] = 2 * dimensions[axis] - high_image[axis]
        reflection_distances.extend(
            [float(np.linalg.norm(low_image - microphone)), float(np.linalg.norm(high_image - microphone))]
        )
    for order, reflection_distance in enumerate(sorted(reflection_distances)):
        delay = round(max(0.0, reflection_distance - distance) / 343.0 * sample_rate)
        if 0 < delay < length:
            result[delay] += (0.58 ** (1 + order // 3)) / max(1.0, reflection_distance)

    rir_seed_key = scene.rir_seed_key or scene.scene_id
    rng = np.random.default_rng(stable_seed(f"rir:{rir_seed_key}:{speaker_index}", seed))
    start = min(length, round(0.025 * sample_rate))
    tail_time = np.arange(length - start, dtype=np.float64) / sample_rate
    decay = 10 ** (-3.0 * tail_time / scene.rt60_seconds)
    result[start:] += rng.normal(0.0, 0.014, length - start) * decay
    result /= max(float(np.sqrt(np.sum(result**2))), 1e-8)
    return result.astype(np.float32)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return _relative(path, root)
    except ValueError:
        return str(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _active_samples(audio: np.ndarray, *, top_db: float = 40.0) -> np.ndarray:
    """Extract samples used for speaker-level RMS calibration."""
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0:
        return values
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-12:
        return values[:0]
    threshold = peak * (10.0 ** (-top_db / 20.0))
    return values[np.abs(values) >= threshold]


def _active_span_seconds(
    audio: np.ndarray, sample_rate: int, *, top_db: float = 30.0
) -> float:
    """Duration from the first to last active sample, including internal pauses."""
    values = np.abs(np.asarray(audio, dtype=np.float32))
    if values.size == 0:
        return 0.0
    peak = float(np.max(values))
    if peak <= 1e-12:
        return 0.0
    active = np.flatnonzero(values > peak * (10.0 ** (-top_db / 20.0)))
    if active.size == 0:
        return 0.0
    return (int(active[-1]) - int(active[0]) + 1) / sample_rate


def _calibrate_speaker_levels(
    utterance_audio: list[np.ndarray],
    speaker_indices: list[int],
    speaker_count: int,
    target_dbfs: float | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Apply one fixed active-RMS gain to every utterance from each speaker."""
    if target_dbfs is None:
        return utterance_audio, []

    target_rms = 10.0 ** (target_dbfs / 20.0)
    active_by_speaker: list[list[np.ndarray]] = [[] for _ in range(speaker_count)]
    for audio, speaker_index in zip(utterance_audio, speaker_indices, strict=True):
        active = _active_samples(audio)
        if active.size:
            active_by_speaker[speaker_index].append(active)

    gains = np.ones(speaker_count, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for speaker_index, samples in enumerate(active_by_speaker):
        if not samples:
            rows.append(
                {
                    "speaker_index": speaker_index,
                    "target_rms_dbfs": round(target_dbfs, 4),
                    "measured_rms_dbfs_before": None,
                    "measured_rms_dbfs_after": None,
                    "gain_db": 0.0,
                    "status": "no_active_audio",
                }
            )
            continue
        combined = np.concatenate(samples)
        measured_rms = rms(combined)
        gains[speaker_index] = target_rms / max(measured_rms, 1e-8)
        gain_db = 20.0 * math.log10(max(gains[speaker_index], 1e-12))
        rows.append(
            {
                "speaker_index": speaker_index,
                "target_rms_dbfs": round(target_dbfs, 4),
                "measured_rms_dbfs_before": round(
                    20.0 * math.log10(max(measured_rms, 1e-12)), 4
                ),
                "measured_rms_dbfs_after": round(target_dbfs, 4),
                "gain_db": round(gain_db, 4),
                "status": "calibrated",
            }
        )

    calibrated = [
        (np.asarray(audio, dtype=np.float32) * gains[speaker_index]).astype(np.float32)
        for audio, speaker_index in zip(utterance_audio, speaker_indices, strict=True)
    ]
    return calibrated, rows


def _render_scene(
    config: Config,
    scene: Scene,
    output_dir: Path,
    speech_backend: SpeechBackend,
    speech_edit_backend: SpeechEditBackend | None,
) -> dict[str, Any]:
    dialogue = load_dialogue(scene.dialogue, scene.speakers)
    if dialogue.scene_id != scene.scene_id:
        raise ValueError(f"Scene ID mismatch: {scene.scene_id} != {dialogue.scene_id}")
    if speech_edit_backend is None and any(
        item.edit_instruction is not None for item in dialogue.utterances
    ):
        raise ValueError(
            f"{scene.scene_id}: dialogue contains edit_instruction but "
            "speech_edit_backend is not configured"
        )

    source_utterance_audio: list[np.ndarray] = []
    native_source_paths: list[Path | None] = []
    utterance_speakers: list[int] = []
    precomputed_count = 0
    for item in dialogue.utterances:
        if item.input_audio_path is not None:
            if not item.input_audio_path.is_file():
                raise FileNotFoundError(
                    f"{scene.scene_id}/{item.utterance_id}: input audio does not exist: "
                    f"{item.input_audio_path}"
                )
            source_utterance_audio.append(load_audio(item.input_audio_path, config.sample_rate))
            native_source_paths.append(item.input_audio_path)
            precomputed_count += 1
        elif item.instructions is None:
            # Preserve compatibility with third-party backends that implement
            # the original two-argument SpeechBackend contract.
            source_utterance_audio.append(
                speech_backend.synthesize(item.text, item.speaker_index)
            )
            native_path = getattr(speech_backend, "last_native_output_path", None)
            native_source_paths.append(Path(native_path) if native_path is not None else None)
        else:
            source_utterance_audio.append(
                speech_backend.synthesize(
                    item.text,
                    item.speaker_index,
                    instructions=item.instructions,
                )
            )
            native_path = getattr(speech_backend, "last_native_output_path", None)
            native_source_paths.append(Path(native_path) if native_path is not None else None)
        utterance_speakers.append(item.speaker_index)

    resolved_edit_instructions = [
        (
            speech_edit_backend.resolve_instruction(item.text, item.edit_instruction)
            if speech_edit_backend is not None
            else None
        )
        for item in dialogue.utterances
    ]
    raw_utterance_audio = list(source_utterance_audio)
    speech_edit_rows: list[dict[str, Any] | None] = [None] * len(dialogue.utterances)
    edited_count = sum(instruction is not None for instruction in resolved_edit_instructions)
    base_backend_released_before_edit = False
    if edited_count:
        assert speech_edit_backend is not None
        # Base and edit checkpoints are both large. Finish the source pass and
        # release base TTS before loading the edit runtime, even on multi-GPU hosts.
        release_source_backend = getattr(speech_backend, "release_for_edit", None)
        if callable(release_source_backend):
            release_source_backend()
            base_backend_released_before_edit = True
        for index, (item, source_audio, native_path, instruction) in enumerate(
            zip(
                dialogue.utterances,
                source_utterance_audio,
                native_source_paths,
                resolved_edit_instructions,
                strict=True,
            )
        ):
            if instruction is None:
                continue
            result = speech_edit_backend.edit(
                source_audio,
                config.sample_rate,
                source_text=item.text,
                instruction=instruction,
                source_audio_path=native_path,
            )
            raw_utterance_audio[index] = result.audio
            source_duration = source_audio.size / config.sample_rate
            edited_duration = result.audio.size / config.sample_rate
            source_active_span = _active_span_seconds(source_audio, config.sample_rate)
            edited_active_span = _active_span_seconds(result.audio, config.sample_rate)
            edit_input_value = result.metadata.get("edit_source_audio_path") or native_path
            speech_edit_rows[index] = {
                    "utterance_id": item.utterance_id,
                    "backend": speech_edit_backend.name,
                    "instruction": instruction,
                    "source_file_duration_seconds": round(source_duration, 6),
                    "edited_file_duration_seconds": round(edited_duration, 6),
                    "file_duration_ratio": round(
                        edited_duration / max(source_duration, 1e-12), 6
                    ),
                    "source_active_span_seconds": round(source_active_span, 6),
                    "edited_active_span_seconds": round(edited_active_span, 6),
                    "active_span_ratio": (
                        round(edited_active_span / source_active_span, 6)
                        if source_active_span > 0
                        else None
                    ),
                    "edit_input_audio_path": (
                        _manifest_path(Path(edit_input_value), output_dir)
                        if edit_input_value is not None
                        else None
                    ),
                    "used_existing_source_file": bool(
                        result.metadata.get("used_existing_source_file", native_path is not None)
                    ),
                    "runtime_request_id": result.metadata.get("request_id"),
                    "runtime_seconds": result.metadata.get("time_used"),
                    "runtime_rtf": result.metadata.get("rtf"),
            }

    utterance_audio, speaker_calibration = _calibrate_speaker_levels(
        raw_utterance_audio,
        utterance_speakers,
        scene.speakers,
        config.speaker_rms_dbfs,
    )
    durations = [audio.size / config.sample_rate for audio in utterance_audio]
    turns, overlap_ratio = schedule(
        dialogue.utterances,
        durations,
        config.sample_rate,
        scene.overlap_ratio,
        scene.max_simultaneous_speakers,
    )

    rirs = [_rir(scene, index, config.sample_rate, config.seed) for index in range(scene.speakers)]
    last_sample = max(turn.end_sample for turn in turns) + max(item.size for item in rirs)
    dry_stems = np.zeros((scene.speakers, last_sample), dtype=np.float32)
    for turn, audio in zip(turns, utterance_audio, strict=True):
        # Scheduling rounds seconds to samples independently from the source
        # waveform length.  Align the write to the common span so a one-sample
        # rounding difference cannot abort an otherwise valid scene.
        start = max(0, turn.start_sample)
        end = min(turn.end_sample, last_sample)
        span = max(0, min(end - start, audio.size))
        if span:
            dry_stems[turn.speaker_index, start : start + span] += audio[:span]

    wet_stems = np.zeros_like(dry_stems)
    for index, rir in enumerate(rirs):
        wet = convolve_same_start(dry_stems[index], rir)[:last_sample]
        wet_stems[index, : wet.size] = wet
    clean = wet_stems.sum(axis=0)
    active = np.any(np.abs(dry_stems) > 1e-6, axis=0)
    noise = np.zeros(last_sample, dtype=np.float32)
    achieved_snr_db: float | None = None
    noise_backend: str | None = None
    if scene.snr_db is not None:
        if scene.noise_path is not None:
            noise = load_noise_asset(scene.noise_path, last_sample, config.sample_rate)
            noise_backend = "audio_asset"
        elif scene.noise_profile is not None:
            noise = generate_noise(
                scene.noise_profile,
                last_sample,
                config.sample_rate,
                stable_seed(
                    f"noise:{scene.noise_seed_key or scene.scene_id}:{scene.noise_profile}",
                    config.seed,
                ),
            )
            noise_backend = "procedural_environment_profile"
        else:
            # Preserve the original default noise for existing configurations.
            noise = generate_legacy_noise(
                last_sample,
                config.sample_rate,
                stable_seed(f"noise:{scene.noise_seed_key or scene.scene_id}", config.seed),
            )
            noise_backend = "synthetic_fixture"
        desired = rms(clean[active]) / (10 ** (scene.snr_db / 20))
        noise *= desired / max(rms(noise[active]), 1e-8)
        achieved_snr_db = 20 * math.log10(rms(clean[active]) / rms(noise[active]))

    mixture = clean + noise
    target_peak = 10 ** (config.peak_dbfs / 20)
    gain = min(1.0, target_peak / max(float(np.max(np.abs(mixture))), 1e-8))
    mixture *= gain
    wet_stems *= gain
    dry_stems *= gain
    noise *= gain

    scene_dir = output_dir / "sessions" / scene.scene_id
    mixture_path = scene_dir / "mixture.wav"
    write_audio(mixture_path, mixture, config.sample_rate)
    utterance_paths: list[Path] = []
    raw_utterance_paths: list[Path] = []
    source_utterance_paths: list[Path | None] = []
    for item, source_audio, raw_audio, audio in zip(
        dialogue.utterances,
        source_utterance_audio,
        raw_utterance_audio,
        utterance_audio,
        strict=True,
    ):
        utterance_path = scene_dir / "utterances" / f"{item.utterance_id}.wav"
        write_audio(utterance_path, audio, config.sample_rate)
        utterance_paths.append(utterance_path)
        raw_utterance_path = scene_dir / "raw_utterances" / f"{item.utterance_id}.wav"
        write_audio(raw_utterance_path, raw_audio, config.sample_rate)
        raw_utterance_paths.append(raw_utterance_path)
        if edited_count:
            source_utterance_path = (
                scene_dir / "source_utterances" / f"{item.utterance_id}.wav"
            )
            write_audio(source_utterance_path, source_audio, config.sample_rate)
            source_utterance_paths.append(source_utterance_path)
        else:
            source_utterance_paths.append(None)
    for index in range(scene.speakers):
        write_audio(scene_dir / "stems" / f"speaker_{index:02d}.wav", wet_stems[index], config.sample_rate)
        write_audio(
            scene_dir / "dry_stems" / f"speaker_{index:02d}.wav",
            dry_stems[index],
            config.sample_rate,
        )
        write_audio(scene_dir / "rirs" / f"speaker_{index:02d}.wav", rirs[index], config.sample_rate, pcm16=False)
    if scene.snr_db is not None:
        write_audio(scene_dir / "stems" / "noise.wav", noise, config.sample_rate)

    turn_rows = [
        {
            **asdict(turn),
            "speaker": f"spk{turn.speaker_index:02d}",
            "start": round(turn.start, 6),
            "end": round(turn.end, 6),
            "source_audio_path": _relative(source_path, output_dir),
            "raw_source_audio_path": _relative(raw_source_path, output_dir),
            "pre_edit_source_audio_path": (
                _relative(pre_edit_source_path, output_dir)
                if pre_edit_source_path is not None
                else None
            ),
            "edit_instruction": resolved_edit_instruction,
            "speech_edit": speech_edit_row,
            "input_audio_path": (
                _manifest_path(item.input_audio_path, output_dir)
                if item.input_audio_path is not None
                else None
            ),
        }
        for (
            turn,
            item,
            source_path,
            raw_source_path,
            pre_edit_source_path,
            resolved_edit_instruction,
            speech_edit_row,
        ) in zip(
            turns,
            dialogue.utterances,
            utterance_paths,
            raw_utterance_paths,
            source_utterance_paths,
            resolved_edit_instructions,
            speech_edit_rows,
            strict=True,
        )
    ]
    manifest = {
        "session_id": scene.scene_id,
        "audio_path": _relative(mixture_path, output_dir),
        "sample_rate": config.sample_rate,
        "duration_seconds": round(mixture.size / config.sample_rate, 6),
        "num_speakers": scene.speakers,
        "rt60_seconds": scene.rt60_seconds,
        "environment": scene.environment,
        "room": {
            "dimensions_m": list(scene.room_dimensions_m),
            "microphone_position_m": list(scene.microphone_position_m),
            "speaker_positions_m": [list(position) for position in scene.speaker_positions_m],
        },
        "overlap_ratio": round(overlap_ratio, 6),
        "max_concurrency": max_concurrency(turns),
        "target_snr_db": scene.snr_db,
        "achieved_snr_db": round(achieved_snr_db, 4) if achieved_snr_db is not None else None,
        "peak_dbfs": round(peak_dbfs(mixture), 4),
        "global_gain_db": round(20 * math.log10(max(gain, 1e-12)), 4),
        "speaker_calibration": speaker_calibration,
        "speech_edits": [row for row in speech_edit_rows if row is not None],
        "topic": dialogue.topic,
        "language": dialogue.language,
        "turns": turn_rows,
        "provenance": {
            "dialogue": scene.dialogue.name,
            "speech_backend": (
                "precomputed_audio"
                if precomputed_count == len(dialogue.utterances)
                else (
                    f"{speech_backend.name}+precomputed_audio"
                    if precomputed_count
                    else speech_backend.name
                )
            ),
            "speech_edit_backend": (
                speech_edit_backend.name if edited_count and speech_edit_backend else None
            ),
            "edited_utterance_count": edited_count,
            "base_backend_released_before_edit": base_backend_released_before_edit,
            "noise_backend": noise_backend,
            "noise_profile": scene.noise_profile,
            "noise_asset": scene.noise_path.name if scene.noise_path is not None else None,
            "noise_environment": scene.noise_environment,
            "rir_environment": scene.rir_environment,
            "rir_backend": (
                "audio_asset"
                if (scene.rir_path is not None or scene.rir_paths is not None)
                else "deterministic_approximation"
            ),
            "rir_asset": (
                scene.rir_path.name
                if scene.rir_path is not None
                else (scene.rir_paths[0].name if scene.rir_paths else None)
            ),
            "rir_assets": (
                [rir.name for rir in scene.rir_paths]
                if scene.rir_paths is not None
                else ([scene.rir_path.name] * scene.speakers if scene.rir_path else [])
            ),
            "rir_seed_key": scene.rir_seed_key or scene.scene_id,
            "noise_seed_key": (
                scene.noise_seed_key or scene.scene_id if scene.snr_db is not None else None
            ),
            "speaker_leveling": (
                "fixed_active_rms" if config.speaker_rms_dbfs is not None else None
            ),
            "precomputed_utterance_count": precomputed_count,
            "seed": config.seed,
        },
    }
    _write_json(scene_dir / "manifest.json", manifest)
    rttm = "".join(
        f"SPEAKER {scene.scene_id} 1 {turn.start:.6f} {turn.end - turn.start:.6f} "
        f"<NA> <NA> spk{turn.speaker_index:02d} <NA> <NA>\n"
        for turn in turns
    )
    (scene_dir / "segments.rttm").write_text(rttm, encoding="utf-8")
    return manifest


def simulate(
    config_path: Path,
    output_dir: Path,
    *,
    speech_backend: SpeechBackend | None = None,
    speech_edit_backend: SpeechEditBackend | None = None,
) -> list[dict[str, Any]]:
    """Run every configured scene and return its manifest."""
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    backend = speech_backend or build_speech_backend(
        config.speech_backend, config_path, config.sample_rate, config.seed
    )
    edit_backend = speech_edit_backend or build_speech_edit_backend(
        config.speech_edit_backend, config_path, config.sample_rate
    )
    try:
        manifests = [
            _render_scene(config, scene, output_dir, backend, edit_backend)
            for scene in config.scenes
        ]
    finally:
        backend.close()
        if edit_backend is not None:
            edit_backend.close()
    _write_json(output_dir / "manifests" / "sessions.json", manifests)
    export_path = output_dir / "exports" / "sa_asr.jsonl"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as handle:
        for manifest in manifests:
            row = {
                "id": manifest["session_id"],
                "audio": manifest["audio_path"],
                "sample_rate": manifest["sample_rate"],
                "turns": [
                    {
                        "speaker": turn["speaker"],
                        "start": turn["start"],
                        "end": turn["end"],
                        "text": turn["text"],
                    }
                    for turn in manifest["turns"]
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifests
