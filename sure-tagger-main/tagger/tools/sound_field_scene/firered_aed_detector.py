"""FireRed non-streaming audio-event detector.

FireRed AED recognizes speech, singing, and music. Detected event names and the
backward-compatible music decision are public; event segments and frame ratios
remain internal evidence.
"""

from pathlib import Path

from tagger.local_config import FIRERED_AED_MODEL_DIR, FIRERED_AED_PYTHON
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.firered_audio import (
    SUPPORTED_SAMPLE_RATE_HZ,
    prepare_firered_audio,
)
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "firered_aed_detector"
METHOD = "FireRed AED"
EVENT_NAMES = ("speech", "singing", "music")
ROUND_DIGITS = 6
BOUNDARY_EPSILON_SEC = 1e-3


class FireRedAedError(RuntimeError):
    """Raised when FireRed AED cannot produce valid event results."""


class FireRedAedConfig:
    """Fixed non-streaming FireRed AED configuration."""

    def __init__(
        self,
        model_dir=None,
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        singing_threshold=0.5,
        music_threshold=0.5,
        min_singing_ratio=0.10,
        min_music_ratio=0.10,
        min_event_frame=20,
        max_event_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
        normalize_to_16k_mono_pcm=True,
        subprocess_python=None,
    ):
        configured_model_dir = getattr(FIRERED_AED_MODEL_DIR, "strip", lambda: "")()
        configured_python = getattr(FIRERED_AED_PYTHON, "strip", lambda: "")()
        self.model_dir = _resolve_model_dir(model_dir or configured_model_dir)
        self.use_gpu = bool(use_gpu)
        self.smooth_window_size = smooth_window_size
        self.speech_threshold = speech_threshold
        self.singing_threshold = singing_threshold
        self.music_threshold = music_threshold
        self.min_singing_ratio = _require_probability(
            min_singing_ratio, "min_singing_ratio"
        )
        self.min_music_ratio = _require_probability(
            min_music_ratio, "min_music_ratio"
        )
        self.min_event_frame = min_event_frame
        self.max_event_frame = max_event_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.chunk_max_frame = chunk_max_frame
        self.normalize_to_16k_mono_pcm = bool(normalize_to_16k_mono_pcm)
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    def cache_key(self):
        return (
            self.model_dir,
            self.use_gpu,
            self.smooth_window_size,
            self.speech_threshold,
            self.singing_threshold,
            self.music_threshold,
            self.min_singing_ratio,
            self.min_music_ratio,
            self.min_event_frame,
            self.max_event_frame,
            self.min_silence_frame,
            self.merge_silence_frame,
            self.extend_speech_frame,
            self.chunk_max_frame,
            self.normalize_to_16k_mono_pcm,
            self.subprocess_python,
        )

    def to_record(self):
        record = _subprocess_config(self)
        record.update(
            {
                "supported_sample_rate_hz": SUPPORTED_SAMPLE_RATE_HZ,
                "public_mapping": {
                    "speech_music_events": (
                        "detected event names in model class order; singing "
                        "and music only when their event ratios meet the "
                        "configured min_*_ratio floors"
                    ),
                    "music_present": "music event present",
                },
                "subprocess_python": self.subprocess_python,
            }
        )
        return record


class FireRedAedClient:
    """Thin adapter around the official fireredvad AED Python API."""

    def __init__(self, config=None):
        self.config = config or FireRedAedConfig()
        self._aed = None

    def detect_audio_events(self, audio_path, context=None):
        wav_path, cleanup = prepare_firered_audio(
            audio_path,
            context=context,
            normalize_to_16k_mono_pcm=self.config.normalize_to_16k_mono_pcm,
            error_class=FireRedAedError,
            tool_label="FireRed AED",
            temp_prefix="firered_aed_",
        )
        try:
            aed = self._get_aed(context)
            try:
                result, _probs = aed.detect(str(wav_path))
            except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
                raise FireRedAedError("FireRed AED inference failed") from exc
        finally:
            if cleanup is not None:
                cleanup.cleanup()

        if not isinstance(result, dict):
            raise FireRedAedError("FireRed AED returned a non-object result")
        return result

    def _get_aed(self, context=None):
        if context is None:
            if self._aed is None:
                self._aed = self._load_aed()
            return self._aed

        cache = context.setdefault("firered_aed_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_aed()
        return cache[key]

    def _load_aed(self):
        if not self.config.model_dir:
            raise FireRedAedError(
                "FireRed AED model dir is not configured; set "
                "tagger/local_config.py:FIRERED_AED_MODEL_DIR"
            )

        model_dir = Path(self.config.model_dir)
        missing_files = [
            str(model_dir / name)
            for name in ("cmvn.ark", "model.pth.tar")
            if not (model_dir / name).exists()
        ]
        if missing_files:
            raise FireRedAedError(
                "FireRed AED model files are missing: %s" % ", ".join(missing_files)
            )

        try:
            from fireredvad import FireRedAed, FireRedAedConfig as UpstreamConfig
        except ImportError as exc:
            raise FireRedAedError(
                "fireredvad package is not importable; install FireRedVAD first"
            ) from exc

        upstream_config = UpstreamConfig(
            use_gpu=self.config.use_gpu,
            smooth_window_size=self.config.smooth_window_size,
            speech_threshold=self.config.speech_threshold,
            singing_threshold=self.config.singing_threshold,
            music_threshold=self.config.music_threshold,
            min_event_frame=self.config.min_event_frame,
            max_event_frame=self.config.max_event_frame,
            min_silence_frame=self.config.min_silence_frame,
            merge_silence_frame=self.config.merge_silence_frame,
            extend_speech_frame=self.config.extend_speech_frame,
            chunk_max_frame=self.config.chunk_max_frame,
        )
        return FireRedAed.from_pretrained(str(model_dir), upstream_config)


class FireRedAedSubprocessClient:
    """Adapter that runs FireRed AED in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or FireRedAedConfig()

    def detect_audio_events(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "firered_aed_detect",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, duration_sec, context=None, config=None, client=None, **_kwargs):
    duration_sec = _require_duration(duration_sec)
    config = config or FireRedAedConfig()
    client = client or _default_client(config)
    raw_output = client.detect_audio_events(audio_path, context)
    event_segments, event_ratios = validate_aed_output(raw_output, duration_sec)

    audio_events = [name for name in EVENT_NAMES if event_segments[name]]
    # singing and music require substantive content: each event's ratio must
    # meet its config floor. Frame-level false positives on speech (voiced
    # frames with singing/music confidence above the frame thresholds)
    # produce short segments whose ratio is far below the floor, so the
    # gates keep them out of the public events and out of music_present.
    # Raw segments stay in the internal evidence regardless.
    gate_floors = {
        "singing": config.min_singing_ratio,
        "music": config.min_music_ratio,
    }
    gated = {
        name: (
            name in audio_events
            and event_ratios[name] < gate_floors[name]
        )
        for name in gate_floors
    }
    audio_events = [name for name in audio_events if not gated.get(name, False)]
    values = {
        "sound_field_scene.speech_music_events": audio_events,
        "sound_field_scene.music_present": "music" in audio_events,
    }
    evidence = {
        "config": config.to_record(),
        "event_segments": event_segments,
        "event_ratios": event_ratios,
        "event_gates": {
            name: {
                "min_ratio": gate_floors[name],
                "gated": gated[name],
            }
            for name in gate_floors
        },
    }
    return [
        ToolResult(
            tag_path=tag_path,
            value=value,
            tool_name=TOOL_NAME,
            method=METHOD,
            status="estimated",
            confidence=1.0,
            tool_type="model",
            tool_version=TOOL_VERSION,
            evidence=evidence,
        )
        for tag_path, value in values.items()
    ]


def validate_aed_output(raw_output, duration_sec):
    duration_sec = _require_duration(duration_sec)
    if not isinstance(raw_output, dict):
        raise FireRedAedError("FireRed AED output must be an object")

    raw_segments = raw_output.get("event2timestamps")
    raw_ratios = raw_output.get("event2ratio")
    if not isinstance(raw_segments, dict):
        raise FireRedAedError("FireRed AED event2timestamps must be an object")
    if not isinstance(raw_ratios, dict):
        raise FireRedAedError("FireRed AED event2ratio must be an object")

    segments = {}
    ratios = {}
    for event_name in EVENT_NAMES:
        if event_name not in raw_segments or event_name not in raw_ratios:
            raise FireRedAedError("FireRed AED output is missing %s" % event_name)
        segments[event_name] = _normalize_segments(
            raw_segments[event_name],
            duration_sec,
            "event2timestamps.%s" % event_name,
        )
        ratio = _require_number(
            raw_ratios[event_name],
            "event2ratio.%s" % event_name,
        )
        if ratio < 0 or ratio > 1:
            raise FireRedAedError("FireRed AED event ratio must be within [0, 1]")
        ratios[event_name] = round(ratio, ROUND_DIGITS)
    return segments, ratios


def _default_client(config):
    if config.subprocess_python:
        return FireRedAedSubprocessClient(config)
    return FireRedAedClient(config)


def _resolve_model_dir(configured_path):
    if not configured_path:
        return ""

    base_path = Path(str(configured_path)).expanduser()
    candidates = [
        base_path,
        base_path / "AED",
        base_path / "pretrained_models" / "FireRedVAD" / "AED",
        base_path / "checkpoints" / "FireRedVAD" / "AED",
        base_path / "repro_bundle" / "weights" / "FireRedVAD" / "AED",
    ]
    for candidate in candidates:
        if (candidate / "cmvn.ark").exists() and (
            candidate / "model.pth.tar"
        ).exists():
            return str(candidate)
    return str(base_path)


def _subprocess_config(config):
    return {
        "model_dir": config.model_dir,
        "use_gpu": config.use_gpu,
        "smooth_window_size": config.smooth_window_size,
        "speech_threshold": config.speech_threshold,
        "singing_threshold": config.singing_threshold,
        "music_threshold": config.music_threshold,
        "min_singing_ratio": config.min_singing_ratio,
        "min_music_ratio": config.min_music_ratio,
        "min_event_frame": config.min_event_frame,
        "max_event_frame": config.max_event_frame,
        "min_silence_frame": config.min_silence_frame,
        "merge_silence_frame": config.merge_silence_frame,
        "extend_speech_frame": config.extend_speech_frame,
        "chunk_max_frame": config.chunk_max_frame,
        "normalize_to_16k_mono_pcm": config.normalize_to_16k_mono_pcm,
        "subprocess_python": "",
    }


def _normalize_segments(raw_segments, duration_sec, field_name):
    if not isinstance(raw_segments, list):
        raise FireRedAedError("%s must be a list" % field_name)

    normalized = []
    previous_end = None
    for index, raw_segment in enumerate(raw_segments):
        if isinstance(raw_segment, dict):
            start_sec = raw_segment.get("start_sec")
            end_sec = raw_segment.get("end_sec")
        elif isinstance(raw_segment, (list, tuple)) and len(raw_segment) == 2:
            start_sec, end_sec = raw_segment
        else:
            raise FireRedAedError(
                "%s segment %s must be an object or pair" % (field_name, index)
            )
        start_sec = _snap_boundary(
            _require_number(start_sec, "%s[%s].start_sec" % (field_name, index)),
            duration_sec,
        )
        end_sec = _snap_boundary(
            _require_number(end_sec, "%s[%s].end_sec" % (field_name, index)),
            duration_sec,
        )
        if start_sec < 0 or end_sec > duration_sec:
            raise FireRedAedError("%s segment is outside audio duration" % field_name)
        if start_sec >= end_sec:
            raise FireRedAedError("%s segment must satisfy start < end" % field_name)
        if previous_end is not None and start_sec < previous_end - BOUNDARY_EPSILON_SEC:
            raise FireRedAedError("%s must be sorted and non-overlapping" % field_name)
        normalized.append(
            {
                "start_sec": round(start_sec, ROUND_DIGITS),
                "end_sec": round(end_sec, ROUND_DIGITS),
            }
        )
        previous_end = end_sec
    return normalized


def _require_duration(value):
    duration_sec = _require_number(value, "duration_sec")
    if duration_sec <= 0:
        raise FireRedAedError("duration_sec must be positive before FireRed AED")
    return duration_sec


def _require_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FireRedAedError("%s must be a number" % path)
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise FireRedAedError("%s must be finite" % path)
    return value


def _require_probability(value, path):
    value = _require_number(value, path)
    if value < 0 or value > 1:
        raise FireRedAedError("%s must be within [0, 1]" % path)
    return value


def _snap_boundary(value, duration_sec):
    if -BOUNDARY_EPSILON_SEC <= value < 0:
        return 0.0
    if duration_sec < value <= duration_sec + BOUNDARY_EPSILON_SEC:
        return duration_sec
    return value
