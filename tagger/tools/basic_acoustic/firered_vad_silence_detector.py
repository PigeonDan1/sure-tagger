"""FireRed VAD silence segment tool.

This module intentionally contains no alternative VAD implementation. If the
FireRed dependency, model files, or inference call is unavailable, callers must
leave silence tags as null instead of substituting another detector.
"""

from pathlib import Path

from tagger.local_config import FIRERED_VAD_MODEL_DIR, FIRERED_VAD_PYTHON
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.firered_audio import (
    SUPPORTED_SAMPLE_RATE_HZ,
    prepare_firered_audio,
)
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "firered_vad_silence_detector"
METHOD = "FireRed VAD"
ROUND_DIGITS = 6
BOUNDARY_EPSILON_SEC = 1e-3


class FireRedVadError(RuntimeError):
    """Raised when FireRed VAD cannot produce a valid silence result."""


class FireRedVadConfig:
    """Fixed non-streaming FireRed VAD configuration."""

    def __init__(
        self,
        model_dir=None,
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
        normalize_to_16k_mono_pcm=True,
        merge_adjacent_silence_gap_sec=0.0,
        subprocess_python=None,
    ):
        configured_model_dir = FIRERED_VAD_MODEL_DIR.strip()
        configured_python = getattr(FIRERED_VAD_PYTHON, "strip", lambda: "")()
        selected_model_dir = model_dir or configured_model_dir
        self.model_dir = _resolve_model_dir(selected_model_dir)
        self.use_gpu = bool(use_gpu)
        self.smooth_window_size = smooth_window_size
        self.speech_threshold = speech_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.chunk_max_frame = chunk_max_frame
        self.normalize_to_16k_mono_pcm = normalize_to_16k_mono_pcm
        self.merge_adjacent_silence_gap_sec = merge_adjacent_silence_gap_sec
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    def cache_key(self):
        return (
            self.model_dir,
            self.use_gpu,
            self.smooth_window_size,
            self.speech_threshold,
            self.min_speech_frame,
            self.max_speech_frame,
            self.min_silence_frame,
            self.merge_silence_frame,
            self.extend_speech_frame,
            self.chunk_max_frame,
            self.normalize_to_16k_mono_pcm,
            self.merge_adjacent_silence_gap_sec,
            self.subprocess_python,
        )

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "use_gpu": self.use_gpu,
            "smooth_window_size": self.smooth_window_size,
            "speech_threshold": self.speech_threshold,
            "min_speech_frame": self.min_speech_frame,
            "max_speech_frame": self.max_speech_frame,
            "min_silence_frame": self.min_silence_frame,
            "merge_silence_frame": self.merge_silence_frame,
            "extend_speech_frame": self.extend_speech_frame,
            "chunk_max_frame": self.chunk_max_frame,
            "normalize_to_16k_mono_pcm": self.normalize_to_16k_mono_pcm,
            "merge_adjacent_silence_gap_sec": self.merge_adjacent_silence_gap_sec,
            "supported_sample_rate_hz": SUPPORTED_SAMPLE_RATE_HZ,
            "subprocess_python": self.subprocess_python,
        }


class FireRedVadClient:
    """Thin adapter around the official fireredvad Python API."""

    def __init__(self, config=None):
        self.config = config or FireRedVadConfig()
        self._vad = None

    def detect_speech_segments(self, audio_path, context=None):
        wav_path, cleanup = self._prepare_audio(audio_path, context)
        try:
            result = self._detect(wav_path, context)
        finally:
            if cleanup is not None:
                cleanup.cleanup()
        return result.get("timestamps", [])

    def _prepare_audio(self, audio_path, context=None):
        return prepare_firered_audio(
            audio_path,
            context=context,
            normalize_to_16k_mono_pcm=self.config.normalize_to_16k_mono_pcm,
            error_class=FireRedVadError,
            tool_label="FireRed VAD",
            temp_prefix="firered_vad_",
        )

    def _detect(self, wav_path, context=None):
        vad = self._get_vad(context)
        try:
            result, _probs = vad.detect(str(wav_path))
        except Exception as exc:  # noqa: BLE001 - converted to internal warning upstream.
            raise FireRedVadError("FireRed VAD inference failed") from exc
        if not isinstance(result, dict):
            raise FireRedVadError("FireRed VAD returned a non-object result")
        return result

    def _get_vad(self, context=None):
        if context is None:
            if self._vad is None:
                self._vad = self._load_vad()
            return self._vad

        cache = context.setdefault("firered_vad_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_vad()
        return cache[key]

    def _load_vad(self):
        if not self.config.model_dir:
            raise FireRedVadError(
                "FireRed VAD model dir is not configured; set "
                "tagger/local_config.py:FIRERED_VAD_MODEL_DIR"
            )

        model_dir = Path(self.config.model_dir)
        missing_files = [
            str(model_dir / name)
            for name in ("cmvn.ark", "model.pth.tar")
            if not (model_dir / name).exists()
        ]
        if missing_files:
            raise FireRedVadError(
                "FireRed VAD model files are missing: %s" % ", ".join(missing_files)
            )

        try:
            from fireredvad import FireRedVad, FireRedVadConfig as UpstreamConfig
        except ImportError as exc:
            raise FireRedVadError(
                "fireredvad package is not importable; install FireRedVAD first"
            ) from exc

        upstream_config = UpstreamConfig(
            use_gpu=self.config.use_gpu,
            smooth_window_size=self.config.smooth_window_size,
            speech_threshold=self.config.speech_threshold,
            min_speech_frame=self.config.min_speech_frame,
            max_speech_frame=self.config.max_speech_frame,
            min_silence_frame=self.config.min_silence_frame,
            merge_silence_frame=self.config.merge_silence_frame,
            extend_speech_frame=self.config.extend_speech_frame,
            chunk_max_frame=self.config.chunk_max_frame,
        )
        return FireRedVad.from_pretrained(str(model_dir), upstream_config)


class FireRedVadSubprocessClient:
    """Adapter that runs FireRed VAD in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or FireRedVadConfig()

    def detect_speech_segments(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "firered_vad_detect",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["speech_segments"]


def _resolve_model_dir(configured_path):
    if not configured_path:
        return ""

    base_path = Path(str(configured_path)).expanduser()
    candidates = [
        base_path,
        base_path / "VAD",
        base_path / "pretrained_models" / "FireRedVAD" / "VAD",
        base_path / "checkpoints" / "FireRedVAD" / "VAD",
        base_path / "repro_bundle" / "weights" / "FireRedVAD" / "VAD",
    ]
    for candidate in candidates:
        if (candidate / "cmvn.ark").exists() and (
            candidate / "model.pth.tar"
        ).exists():
            return str(candidate)
    return str(base_path)


def run(audio_path, duration_sec, context=None, config=None, client=None, **_kwargs):
    if duration_sec is None or duration_sec <= 0:
        raise FireRedVadError("duration_sec must be positive before FireRed VAD")

    config = config or FireRedVadConfig()
    client = client or _default_client(config)
    speech_segments = client.detect_speech_segments(audio_path, context=context)
    silence_segments = speech_segments_to_silence_segments(
        speech_segments,
        duration_sec=duration_sec,
        merge_adjacent_gap_sec=config.merge_adjacent_silence_gap_sec,
    )
    return ToolResult(
        tag_path="basic_acoustic.silence_segments",
        value=silence_segments,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        evidence={
            "fire_red_vad_config": config.to_record(),
            "speech_segments": _round_segments(speech_segments),
        },
    )


def _default_client(config):
    if config.subprocess_python:
        return FireRedVadSubprocessClient(config)
    return FireRedVadClient(config)


def _subprocess_config(config):
    return {
        "model_dir": config.model_dir,
        "use_gpu": config.use_gpu,
        "smooth_window_size": config.smooth_window_size,
        "speech_threshold": config.speech_threshold,
        "min_speech_frame": config.min_speech_frame,
        "max_speech_frame": config.max_speech_frame,
        "min_silence_frame": config.min_silence_frame,
        "merge_silence_frame": config.merge_silence_frame,
        "extend_speech_frame": config.extend_speech_frame,
        "chunk_max_frame": config.chunk_max_frame,
        "normalize_to_16k_mono_pcm": config.normalize_to_16k_mono_pcm,
        "merge_adjacent_silence_gap_sec": config.merge_adjacent_silence_gap_sec,
        "subprocess_python": "",
    }


def speech_segments_to_silence_segments(
    speech_segments,
    duration_sec,
    merge_adjacent_gap_sec=0.0,
):
    duration_sec = _require_duration(duration_sec)
    speech = _normalize_segments(speech_segments, duration_sec, "speech_segments")
    silence = []
    cursor = 0.0
    for segment in speech:
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        if start_sec > cursor + BOUNDARY_EPSILON_SEC:
            silence.append({"start_sec": cursor, "end_sec": start_sec})
        cursor = max(cursor, end_sec)
    if cursor < duration_sec - BOUNDARY_EPSILON_SEC:
        silence.append({"start_sec": cursor, "end_sec": duration_sec})

    silence = merge_adjacent_segments(silence, max_gap_sec=merge_adjacent_gap_sec)
    return _round_segments(silence)


def merge_adjacent_segments(segments, max_gap_sec=0.0):
    normalized = []
    max_gap_sec = float(max_gap_sec)
    for segment in segments:
        if not normalized:
            normalized.append(dict(segment))
            continue
        previous = normalized[-1]
        if (
            segment["start_sec"] - previous["end_sec"]
            <= max_gap_sec + BOUNDARY_EPSILON_SEC
        ):
            previous["end_sec"] = max(previous["end_sec"], segment["end_sec"])
        else:
            normalized.append(dict(segment))
    return normalized


def validate_silence_segments(segments, duration_sec):
    duration_sec = _require_duration(duration_sec)
    if segments is None:
        raise FireRedVadError("silence_segments must not be null")
    if not isinstance(segments, list):
        raise FireRedVadError("silence_segments must be a list")

    previous_end = None
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, dict):
            raise FireRedVadError(
                "silence_segments segment %s must be an object" % index
            )
        if set(raw_segment.keys()) != set(["start_sec", "end_sec"]):
            raise FireRedVadError(
                "silence_segments segment %s must contain start_sec and end_sec"
                % index
            )
        start_sec, end_sec = _parse_segment(raw_segment, "silence_segments", index)
        if start_sec < 0 or end_sec > duration_sec:
            raise FireRedVadError("silence_segments segment is outside audio duration")
        if start_sec >= end_sec:
            raise FireRedVadError("silence_segments segment must satisfy start < end")
        if (
            previous_end is not None
            and start_sec < previous_end - BOUNDARY_EPSILON_SEC
        ):
            raise FireRedVadError("silence_segments must be sorted and non-overlapping")
        previous_end = end_sec


def _normalize_segments(segments, duration_sec, field_name):
    if segments is None:
        raise FireRedVadError("%s must not be null" % field_name)
    if not isinstance(segments, list):
        raise FireRedVadError("%s must be a list" % field_name)

    parsed = []
    for index, raw_segment in enumerate(segments):
        start_sec, end_sec = _parse_segment(raw_segment, field_name, index)
        start_sec = _snap_boundary(start_sec, duration_sec)
        end_sec = _snap_boundary(end_sec, duration_sec)
        if start_sec < 0 or end_sec > duration_sec:
            raise FireRedVadError("%s segment is outside audio duration" % field_name)
        if start_sec >= end_sec:
            raise FireRedVadError("%s segment must satisfy start < end" % field_name)
        parsed.append({"start_sec": start_sec, "end_sec": end_sec})

    parsed.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged = []
    for segment in parsed:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if segment["start_sec"] <= previous["end_sec"] + BOUNDARY_EPSILON_SEC:
            previous["end_sec"] = max(previous["end_sec"], segment["end_sec"])
        else:
            merged.append(segment)
    return merged


def _parse_segment(raw_segment, field_name, index):
    if isinstance(raw_segment, dict):
        start_sec = raw_segment.get("start_sec")
        end_sec = raw_segment.get("end_sec")
    elif isinstance(raw_segment, (list, tuple)) and len(raw_segment) == 2:
        start_sec, end_sec = raw_segment
    else:
        raise FireRedVadError(
            "%s segment %s must be an object or pair" % (field_name, index)
        )
    start_sec = _require_number(start_sec, "%s[%s].start_sec" % (field_name, index))
    end_sec = _require_number(end_sec, "%s[%s].end_sec" % (field_name, index))
    return start_sec, end_sec


def _require_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FireRedVadError("%s must be a number" % path)
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise FireRedVadError("%s must be finite" % path)
    return value


def _require_duration(duration_sec):
    duration_sec = _require_number(duration_sec, "duration_sec")
    if duration_sec <= 0:
        raise FireRedVadError("duration_sec must be positive")
    return duration_sec


def _snap_boundary(value, duration_sec):
    if -BOUNDARY_EPSILON_SEC <= value < 0:
        return 0.0
    if duration_sec < value <= duration_sec + BOUNDARY_EPSILON_SEC:
        return duration_sec
    return value


def _round_segments(segments):
    rounded = []
    for raw_segment in segments:
        if isinstance(raw_segment, dict):
            start_sec = raw_segment.get("start_sec")
            end_sec = raw_segment.get("end_sec")
        else:
            start_sec, end_sec = raw_segment
        rounded.append(
            {
                "start_sec": round(float(start_sec), ROUND_DIGITS),
                "end_sec": round(float(end_sec), ROUND_DIGITS),
            }
        )
    return rounded
