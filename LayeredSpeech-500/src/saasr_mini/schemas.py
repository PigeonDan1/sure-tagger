from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    speaker_index: int
    text: str
    overlap_hint: str = "none"
    instructions: str | None = None
    edit_instruction: str | None = None
    input_audio_path: Path | None = None


@dataclass(frozen=True)
class Dialogue:
    scene_id: str
    topic: str
    language: str
    speakers: list[dict[str, Any]]
    utterances: list[Utterance]


@dataclass(frozen=True)
class Turn:
    utterance_id: str
    speaker_index: int
    text: str
    start: float
    end: float
    start_sample: int
    end_sample: int
    instructions: str | None = None
    edit_instruction: str | None = None


@dataclass(frozen=True)
class Scene:
    scene_id: str
    dialogue: Path
    speakers: int
    overlap_ratio: float
    max_simultaneous_speakers: int
    room_dimensions_m: tuple[float, float, float]
    microphone_position_m: tuple[float, float, float]
    speaker_positions_m: list[tuple[float, float, float]]
    rt60_seconds: float
    snr_db: float | None
    environment: str | None = None
    noise_profile: str | None = None
    noise_path: Path | None = None
    noise_environment: str | None = None
    rir_path: Path | None = None
    rir_paths: list[Path] | None = None
    rir_environment: str | None = None
    rir_seed_key: str | None = None
    noise_seed_key: str | None = None


@dataclass(frozen=True)
class Config:
    seed: int
    sample_rate: int
    peak_dbfs: float
    scenes: list[Scene]
    speaker_rms_dbfs: float | None = None
    speech_backend: dict[str, Any] = field(default_factory=dict)
    speech_edit_backend: dict[str, Any] = field(default_factory=dict)
