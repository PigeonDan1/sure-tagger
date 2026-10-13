"""Shared deterministic audio IO helpers for acoustic tag tools."""

import json
from pathlib import Path
import subprocess
import wave


class WavInfo:
    def __init__(
        self,
        path,
        duration_sec,
        sample_rate_hz,
        channels,
        sample_width_bytes,
        frame_count,
        compression_type,
        compression_name,
        method,
        extra_evidence=None,
    ):
        self.path = path
        self.duration_sec = duration_sec
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.frame_count = frame_count
        self.compression_type = compression_type
        self.compression_name = compression_name
        self.method = method
        self.extra_evidence = extra_evidence or {}

    def base_evidence(self):
        evidence = {
            "audio_path": self.path,
            "frame_count": self.frame_count,
            "sample_width_bytes": self.sample_width_bytes,
            "compression_type": self.compression_type,
            "compression_name": self.compression_name,
        }
        evidence.update(self.extra_evidence)
        return evidence


def get_audio_info(audio_path, context=None):
    # type: (Union[str, Path], Optional[dict]) -> WavInfo
    path = Path(audio_path)
    if context is None:
        return probe_audio_info(path)

    cache = context.setdefault("audio_info_by_path", {})
    key = str(path)
    if key not in cache:
        cache[key] = probe_audio_info(path)
    return cache[key]


def probe_audio_info(audio_path):
    # type: (Union[str, Path]) -> WavInfo
    """Probe deterministic WAV header fields.

    PCM WAV files are read with Python's stdlib wave module. WAV encodings not
    supported by that module fall back to ffprobe with fixed arguments.
    """

    path = Path(audio_path)
    try:
        return _probe_python_wave(path)
    except wave.Error:
        return _probe_ffprobe(path)


def _probe_python_wave(path):
    with wave.open(str(path), "rb") as wav:
        frame_count = wav.getnframes()
        sample_rate_hz = wav.getframerate()
        channels = wav.getnchannels()
        sample_width_bytes = wav.getsampwidth()
        compression_type = wav.getcomptype()
        compression_name = wav.getcompname()

    duration_sec = frame_count / sample_rate_hz if sample_rate_hz else 0.0
    return WavInfo(
        path=str(path),
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
        compression_type=compression_type,
        compression_name=compression_name,
        method="python_wave_header",
    )


def _probe_ffprobe(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration,codec_name,codec_type,bits_per_sample,bits_per_raw_sample:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(command)
    except OSError as exc:
        raise RuntimeError("ffprobe is required for non-PCM WAV probing") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffprobe failed for %s" % path) from exc

    payload = json.loads(output.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no audio stream: %s" % path)

    stream = streams[0]
    sample_rate_hz = int(stream["sample_rate"])
    channels = int(stream["channels"])
    duration_raw = stream.get("duration") or payload.get("format", {}).get("duration")
    duration_sec = float(duration_raw) if duration_raw is not None else 0.0
    bits_per_sample = (
        stream.get("bits_per_sample") or stream.get("bits_per_raw_sample") or 0
    )
    try:
        sample_width_bytes = int(bits_per_sample) // 8
    except (TypeError, ValueError):
        sample_width_bytes = 0
    frame_count = int(round(duration_sec * sample_rate_hz)) if sample_rate_hz else 0
    codec_name = stream.get("codec_name", "")

    return WavInfo(
        path=str(path),
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
        compression_type=codec_name,
        compression_name=codec_name,
        method="ffprobe_stream",
        extra_evidence={"ffprobe_stream": stream},
    )

