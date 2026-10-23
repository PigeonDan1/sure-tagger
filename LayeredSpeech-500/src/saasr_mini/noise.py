"""Deterministic environmental-noise profiles and audio-asset loading.

The profiles are intentionally synthetic and license-free.  They model broad
spectral and temporal characteristics of common recording environments; they
are not replacements for licensed field recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly

from .audio import rms


PROFILE_METADATA: dict[str, dict[str, str]] = {
    "office": {
        "label": "office",
        "description": "steady HVAC and electrical hum with sparse keyboard-like transients",
    },
    "airport": {
        "label": "airport",
        "description": "air-handling rumble, diffuse terminal crowd band, and gate activity bursts",
    },
    "station": {
        "label": "station",
        "description": (
            "platform/terminal crowd bed, low mechanical rumble, and intermittent activity"
        ),
    },
    "train": {
        "label": "train",
        "description": "rolling low-frequency carriage rumble with rail modulation and rattling",
    },
    "cafe": {
        "label": "cafe",
        "description": "dense indoor crowd bed with dish and table activity transients",
    },
    "street": {
        "label": "street",
        "description": "road-traffic rumble, tire noise, and occasional passing activity",
    },
}

SUPPORTED_PROFILES = tuple(PROFILE_METADATA)


def _bandpass(signal: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    low = max(1.0, min(float(low), nyquist * 0.90))
    high = max(low + 1.0, min(float(high), nyquist * 0.96))
    if high >= nyquist:
        high = nyquist * 0.96
    b, a = butter(2, [low / nyquist, high / nyquist], btype="bandpass")
    return lfilter(b, a, signal).astype(np.float32)


def _lowpass(signal: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    cutoff = min(float(cutoff), sample_rate * 0.45)
    b, a = butter(2, cutoff / (sample_rate / 2.0), btype="low")
    return lfilter(b, a, signal).astype(np.float32)


def _modulation(
    rng: np.random.Generator, samples: int, sample_rate: int, cutoff: float
) -> np.ndarray:
    source = rng.standard_normal(samples).astype(np.float32)
    envelope = np.abs(_lowpass(source, sample_rate, cutoff))
    envelope /= max(float(np.percentile(envelope, 95)), 1e-6)
    return np.clip(0.55 + 0.70 * envelope, 0.35, 1.35).astype(np.float32)


def _transient_bed(
    rng: np.random.Generator,
    samples: int,
    sample_rate: int,
    events_per_second: float,
    low: float,
    high: float,
    duration_ms: tuple[float, float],
) -> np.ndarray:
    result = np.zeros(samples, dtype=np.float32)
    count = int(rng.poisson(max(0.0, events_per_second * samples / sample_rate)))
    if count == 0:
        return result
    positions = rng.integers(0, max(1, samples), size=count)
    for position in positions:
        length = max(8, round(float(rng.uniform(*duration_ms)) * sample_rate / 1000.0))
        end = min(samples, int(position) + length)
        if end <= position:
            continue
        burst = _bandpass(
            rng.standard_normal(end - position).astype(np.float32), sample_rate, low, high
        )
        envelope = np.exp(-np.linspace(0.0, 5.0, end - position, dtype=np.float32))
        result[position:end] += burst * envelope * float(rng.uniform(0.20, 0.65))
    return result


def _tone_bursts(
    rng: np.random.Generator,
    samples: int,
    sample_rate: int,
    events_per_second: float,
    frequencies: tuple[float, ...],
) -> np.ndarray:
    result = np.zeros(samples, dtype=np.float32)
    count = int(rng.poisson(max(0.0, events_per_second * samples / sample_rate)))
    for position in rng.integers(0, max(1, samples), size=count):
        length = round(float(rng.uniform(0.20, 0.75)) * sample_rate)
        end = min(samples, int(position) + length)
        if end <= position:
            continue
        time = np.arange(end - position, dtype=np.float32) / sample_rate
        frequency = float(rng.choice(frequencies))
        envelope = np.sin(np.linspace(0.0, np.pi, end - position, dtype=np.float32)) ** 2
        result[position:end] += np.sin(2.0 * np.pi * frequency * time) * envelope * 0.045
    return result


def _finish(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - np.mean(signal, dtype=np.float32)
    target_rms = 0.10
    signal *= target_rms / max(rms(signal), 1e-8)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0.85:
        signal *= 0.85 / peak
    return signal.astype(np.float32)


def generate_noise(profile: str, samples: int, sample_rate: int, seed: int) -> np.ndarray:
    """Generate one deterministic mono noise profile at ``sample_rate``."""
    if profile not in PROFILE_METADATA:
        choices = ", ".join(SUPPORTED_PROFILES)
        raise ValueError(f"Unknown noise profile {profile!r}; choose one of: {choices}")
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(samples).astype(np.float32)
    time = np.arange(samples, dtype=np.float32) / sample_rate

    if profile == "office":
        hvac = _lowpass(white, sample_rate, 420.0)
        hum = np.sin(2.0 * np.pi * 50.0 * time) + 0.25 * np.sin(2.0 * np.pi * 100.0 * time)
        activity = _transient_bed(rng, samples, sample_rate, 1.8, 900.0, 4200.0, (8.0, 55.0))
        signal = 0.95 * hvac + 0.07 * hum + 0.28 * activity
    elif profile == "airport":
        air = _lowpass(white, sample_rate, 480.0)
        crowd = _bandpass(white, sample_rate, 250.0, 2400.0) * _modulation(
            rng, samples, sample_rate, 0.25
        )
        activity = _transient_bed(
            rng, samples, sample_rate, 0.55, 700.0, 5200.0, (35.0, 180.0)
        )
        chimes = _tone_bursts(rng, samples, sample_rate, 0.035, (440.0, 660.0, 880.0))
        signal = 0.72 * air + 0.46 * crowd + 0.22 * activity + chimes
    elif profile == "station":
        rumble = _lowpass(white, sample_rate, 230.0)
        crowd = _bandpass(white, sample_rate, 180.0, 3000.0) * _modulation(
            rng, samples, sample_rate, 0.45
        )
        activity = _transient_bed(
            rng, samples, sample_rate, 0.90, 300.0, 4200.0, (45.0, 260.0)
        )
        signal = 0.82 * rumble + 0.52 * crowd + 0.30 * activity
    elif profile == "train":
        rolling = _lowpass(white, sample_rate, 180.0)
        wheel_mod = 0.75 + 0.25 * np.sin(
            2.0 * np.pi * 6.2 * time + 0.7 * np.sin(2.0 * np.pi * 0.18 * time)
        )
        carriage = _bandpass(white, sample_rate, 300.0, 2600.0)
        rattle = _transient_bed(rng, samples, sample_rate, 2.5, 1200.0, 6000.0, (4.0, 22.0))
        signal = 1.00 * rolling * wheel_mod + 0.30 * carriage + 0.18 * rattle
    elif profile == "cafe":
        crowd = _bandpass(white, sample_rate, 220.0, 3100.0) * _modulation(
            rng, samples, sample_rate, 0.7
        )
        room = _lowpass(white, sample_rate, 420.0)
        activity = _transient_bed(rng, samples, sample_rate, 2.2, 1200.0, 6500.0, (12.0, 90.0))
        signal = 0.78 * crowd + 0.36 * room + 0.25 * activity
    elif profile == "street":
        engine = _lowpass(white, sample_rate, 190.0) * _modulation(rng, samples, sample_rate, 0.12)
        tires = _bandpass(white, sample_rate, 500.0, 4200.0)
        passing = _transient_bed(rng, samples, sample_rate, 0.25, 180.0, 1500.0, (220.0, 900.0))
        signal = 0.92 * engine + 0.42 * tires + 0.26 * passing
    else:  # pragma: no cover - guarded by the profile check above
        raise AssertionError(profile)
    return _finish(signal)


def generate_legacy_noise(samples: int, sample_rate: int, seed: int) -> np.ndarray:
    """Generate the original low-pass ventilation fixture for old configs."""
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, samples)
    b, a = butter(2, 900 / (sample_rate / 2), btype="low")
    ventilation = lfilter(b, a, white)
    time = np.arange(samples) / sample_rate
    hum = np.sin(2 * np.pi * 60 * time) + 0.35 * np.sin(2 * np.pi * 120 * time)
    return (ventilation + 0.12 * hum).astype(np.float32)


def load_noise_asset(path: Path, samples: int, sample_rate: int) -> np.ndarray:
    """Read, mono-convert, resample, and tile/crop a noise asset."""
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Noise asset must contain a non-empty mono or stereo waveform: {path}")
    if int(source_rate) != sample_rate:
        audio = resample_poly(audio, sample_rate, int(source_rate)).astype(np.float32)
    if audio.size < samples:
        repeats = int(np.ceil(samples / audio.size))
        audio = np.tile(audio, repeats)
    audio = audio[:samples]
    audio = audio - np.mean(audio, dtype=np.float32)
    return np.nan_to_num(audio, copy=False).astype(np.float32)
