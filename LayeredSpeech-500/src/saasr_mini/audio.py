from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

EPSILON = 1e-12


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + EPSILON))


def active_rms(audio: np.ndarray, *, top_db: float = 40.0) -> float:
    """Return RMS after excluding samples near the waveform peak's floor."""
    values = np.asarray(audio)
    if values.size == 0:
        return 0.0
    peak = float(np.max(np.abs(values)))
    if peak <= EPSILON:
        return 0.0
    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.abs(values) >= threshold
    if not np.any(active):
        return 0.0
    return rms(values[active])


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * math.log10(max(peak, EPSILON))


def write_audio(path: Path, audio: np.ndarray, sample_rate: int, *, pcm16: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".partial{path.suffix}")
    sf.write(
        temporary,
        np.asarray(audio, dtype=np.float32),
        sample_rate,
        subtype="PCM_16" if pcm16 else "FLOAT",
    )
    temporary.replace(path)


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load an audio asset as mono float32 and resample it when needed."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim > 1:
        values = values.mean(axis=1)
    if int(source_rate) != int(sample_rate):
        common = math.gcd(int(source_rate), int(sample_rate))
        values = resample_poly(
            values.astype(np.float64), int(sample_rate) // common, int(source_rate) // common
        ).astype(np.float32)
    return values.astype(np.float32)


def convolve_same_start(audio: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    return fftconvolve(audio, impulse_response, mode="full").astype(np.float32)


def interval_overlap_ratio(intervals: list[tuple[float, float]]) -> float:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous: float | None = None
    union = overlap = 0.0
    for time, delta in events:
        if previous is not None and time > previous:
            width = time - previous
            union += width if active >= 1 else 0.0
            overlap += width if active >= 2 else 0.0
        active += delta
        previous = time
    return overlap / union if union else 0.0
