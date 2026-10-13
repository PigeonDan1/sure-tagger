"""Local GPU adapter for NVIDIA Nemotron 3 Diarization."""

import copy
import hashlib
import math
from pathlib import Path
import re
import subprocess
import time

from tagger.tools.subprocess_runner import run_subprocess_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "speaker" / "nvidia" / "Nemotron-3-Diarization"
)
DEFAULT_SUBPROCESS_PYTHON = (
    PROJECT_ROOT
    / ".runtime"
    / "nemotron3_diarization_py311_transformers519_torch280_v1"
    / "bin"
    / "python"
)

MODEL_VERSION = (
    "nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e"
)
MODEL_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
CONFIG_SHA256 = "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873"
PROCESSOR_CONFIG_SHA256 = (
    "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e"
)
SOURCE_NAME = "nemotron_3_diarization"
SOURCE_KIND = "nvidia_nemotron_3_diarization"
DEPENDENCY_GROUPS = ("G_nemotron_3_diarization",)
MODEL_SLOT_KEY = "nemotron3_diarization_estimate"
TOOL_VERSION = "nemotron3_diarization_adapter.v1"
SCHEMA_VERSION = "nemotron3.diarization.timeline.v1"
SAMPLE_RATE_HZ = 16000
MAX_SPEAKERS = 8
ROUND_DIGITS = 6


class Nemotron3DiarizationError(RuntimeError):
    """Raised when the local Nemotron model cannot return a speaker timeline."""


class Nemotron3DiarizationConfig:
    def __init__(
        self,
        model_dir=None,
        subprocess_python=None,
        timeout_sec=1800,
        device="cuda:0",
        verify_model_assets=True,
    ):
        self.model_dir = str(model_dir or DEFAULT_MODEL_DIR)
        self.subprocess_python = str(
            DEFAULT_SUBPROCESS_PYTHON
            if subprocess_python is None
            else subprocess_python or ""
        )
        self.timeout_sec = int(timeout_sec)
        self.device = str(device or "cuda:0")
        self.verify_model_assets = bool(verify_model_assets)
        if self.timeout_sec < 1:
            raise ValueError("Nemotron 3 timeout_sec must be positive")
        if re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("Nemotron 3 requires an explicit CUDA device")

    def cache_key(self):
        return (self.model_dir, self.device, self.verify_model_assets)

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "subprocess_python": self.subprocess_python,
            "timeout_sec": self.timeout_sec,
            "device": self.device,
            "verify_model_assets": self.verify_model_assets,
        }


class Nemotron3DiarizationClient:
    """Process-local model client, reused by the JSONL subprocess worker."""

    def __init__(self, config=None):
        self.config = config or Nemotron3DiarizationConfig(subprocess_python="")
        self.model_dir = _resolve_path(self.config.model_dir)
        if not self.model_dir.is_dir():
            raise Nemotron3DiarizationError(
                "Nemotron 3 model directory is missing: %s" % self.model_dir
            )
        _check_model_files(self.model_dir, self.config.verify_model_assets)
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self._model_load_sec = None

    def diarize(self, audio_path, context=None):
        del context
        audio_path = _resolve_path(audio_path)
        if not audio_path.is_file():
            raise Nemotron3DiarizationError("audio file is missing: %s" % audio_path)
        self._load_model()
        audio, preparation = _read_audio(audio_path)
        started = time.perf_counter()
        try:
            inputs = self._processor(audio, sampling_rate=SAMPLE_RATE_HZ).to(
                self._device, dtype=self._model.dtype
            )
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits
            self._torch.cuda.synchronize(self._device)
            native = self._processor.extract_speaker_dict(
                logits, inputs.attention_mask
            )[0]
        except Exception as exc:
            raise Nemotron3DiarizationError(
                "Nemotron 3 inference failed: %s" % exc
            ) from exc
        duration = len(audio) / SAMPLE_RATE_HZ
        segments = native_segments_to_timeline(native, duration)
        return {
            "schema_version": SCHEMA_VERSION,
            "segments": segments,
            "runtime": {
                "device": str(self._device),
                "gpu": self._torch.cuda.get_device_name(self._device),
                "torch_version": self._torch.__version__,
                "model_load_sec": round(self._model_load_sec, ROUND_DIGITS),
                "inference_sec": round(time.perf_counter() - started, ROUND_DIGITS),
                "model_version": MODEL_VERSION,
                "model_dir": str(self.model_dir),
                "model_sha256": MODEL_SHA256 if self.config.verify_model_assets else None,
                "maximum_speakers": MAX_SPEAKERS,
                "input_audio_path": str(audio_path),
                "preprocessing": preparation,
            },
        }

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        except ImportError as exc:
            raise Nemotron3DiarizationError(
                "Nemotron 3 Transformers runtime is unavailable"
            ) from exc
        if not torch.cuda.is_available():
            raise Nemotron3DiarizationError("Nemotron 3 requires CUDA")
        device = torch.device(self.config.device)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise Nemotron3DiarizationError(
                "Nemotron 3 CUDA device is unavailable: %s" % device
            )
        started = time.perf_counter()
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            model = AutoModelForAudioFrameClassification.from_pretrained(
                str(self.model_dir), local_files_only=True
            ).to(device).eval()
        except Exception as exc:
            raise Nemotron3DiarizationError(
                "Nemotron 3 checkpoint loading failed: %s" % exc
            ) from exc
        speakers = int(model.config.head_config.num_speakers)
        if speakers != MAX_SPEAKERS:
            raise Nemotron3DiarizationError(
                "Nemotron 3 checkpoint has %d speaker channels, expected %d"
                % (speakers, MAX_SPEAKERS)
            )
        torch.cuda.synchronize(device)
        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        self._model_load_sec = time.perf_counter() - started


class Nemotron3DiarizationSubprocessClient:
    def __init__(self, config=None):
        self.config = config or Nemotron3DiarizationConfig()

    def diarize(self, audio_path, context=None):
        audio_path = _resolve_path(audio_path)
        if not audio_path.is_file():
            raise Nemotron3DiarizationError("audio file is missing: %s" % audio_path)
        config_record = self.config.to_record()
        config_record["subprocess_python"] = ""
        try:
            result = run_subprocess_tool(
                self.config.subprocess_python,
                MODEL_SLOT_KEY,
                {"audio_path": str(audio_path), "config": config_record},
                context=context,
                timeout_sec=self.config.timeout_sec,
            )
        except Exception as exc:
            raise Nemotron3DiarizationError(
                "Nemotron 3 subprocess failed: %s" % exc
            ) from exc
        if isinstance(result, dict) and "output" in result:
            result = result["output"]
        return normalize_result(result)


def diarize(audio_path, config=None, context=None, client=None):
    config = config or Nemotron3DiarizationConfig()
    if client is None:
        client = (
            Nemotron3DiarizationSubprocessClient(config)
            if config.subprocess_python
            else Nemotron3DiarizationClient(config)
        )
    return client.diarize(audio_path, context=context)


def native_segments_to_timeline(native_segments, duration_sec):
    if not isinstance(native_segments, list):
        raise Nemotron3DiarizationError("Nemotron 3 returned no segment list")
    duration_sec = float(duration_sec)
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise Nemotron3DiarizationError("audio duration must be positive")
    segments = []
    for native in native_segments:
        if not isinstance(native, dict):
            raise Nemotron3DiarizationError("native segment must be an object")
        start = _finite_number(native.get("Start"), "Start")
        end = _finite_number(native.get("End"), "End")
        speaker = native.get("Speaker")
        if isinstance(speaker, bool) or not isinstance(speaker, int):
            raise Nemotron3DiarizationError("native Speaker must be an integer")
        if not 0 <= speaker < MAX_SPEAKERS:
            raise Nemotron3DiarizationError("native Speaker is outside 0..7")
        if end < start:
            raise Nemotron3DiarizationError("native segment has End before Start")
        start = round(max(0.0, min(duration_sec, start)), ROUND_DIGITS)
        end = round(max(0.0, min(duration_sec, end)), ROUND_DIGITS)
        if end <= start:
            continue
        segments.append(
            {
                "start_sec": start,
                "end_sec": end,
                "speaker_id": "speaker_%d" % (speaker + 1),
                "overlap": False,
            }
        )
    segments.sort(key=lambda row: (row["start_sec"], row["end_sec"], row["speaker_id"]))
    for index, row in enumerate(segments):
        for earlier in segments[:index]:
            if earlier["end_sec"] <= row["start_sec"]:
                continue
            if earlier["speaker_id"] != row["speaker_id"]:
                earlier["overlap"] = True
                row["overlap"] = True
    return segments


def normalize_result(result):
    if not isinstance(result, dict):
        raise Nemotron3DiarizationError("Nemotron 3 returned a non-object result")
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise Nemotron3DiarizationError("Nemotron 3 result has no segment list")
    runtime = result.get("runtime", {})
    if not isinstance(runtime, dict):
        raise Nemotron3DiarizationError("Nemotron 3 runtime must be an object")
    normalized = copy.deepcopy(result)
    previous = None
    for segment in segments:
        if not isinstance(segment, dict):
            raise Nemotron3DiarizationError("timeline segment must be an object")
        start = _finite_number(segment.get("start_sec"), "start_sec")
        end = _finite_number(segment.get("end_sec"), "end_sec")
        speaker = segment.get("speaker_id")
        if start < 0 or end <= start:
            raise Nemotron3DiarizationError("timeline segment bounds are invalid")
        if not isinstance(speaker, str) or not speaker:
            raise Nemotron3DiarizationError("timeline speaker_id is invalid")
        key = (start, end, speaker)
        if previous is not None and key < previous:
            raise Nemotron3DiarizationError("timeline segments are not sorted")
        previous = key
    normalized["segments"] = segments
    normalized["runtime"] = runtime
    return normalized


def _read_audio(path):
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise Nemotron3DiarizationError(
            "soundfile, numpy and scipy are required for Nemotron 3 audio decoding"
        ) from exc
    try:
        waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        channels = waveform.shape[1]
        audio = waveform[:, 0]
        if sample_rate != SAMPLE_RATE_HZ:
            common = math.gcd(int(sample_rate), SAMPLE_RATE_HZ)
            audio = resample_poly(
                audio, SAMPLE_RATE_HZ // common, int(sample_rate) // common
            )
        decoder = "soundfile"
    except (OSError, RuntimeError) as exc:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:a:0", "-af", "pan=mono|c0=c0", "-ar", str(SAMPLE_RATE_HZ),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ]
        try:
            audio = np.frombuffer(subprocess.check_output(command), dtype="<f4")
        except (OSError, subprocess.CalledProcessError) as fallback_exc:
            raise Nemotron3DiarizationError(
                "audio decoding failed for %s: %s" % (path, fallback_exc)
            ) from exc
        sample_rate = None
        channels = None
        decoder = "ffmpeg"
    if audio.size == 0 or not np.isfinite(audio).all():
        raise Nemotron3DiarizationError("decoded audio is empty or non-finite")
    return np.ascontiguousarray(audio, dtype=np.float32), {
        "decoder": decoder,
        "source_sample_rate_hz": sample_rate,
        "source_channels": channels,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channel_selection": 0,
    }


def _check_model_files(model_dir, verify_hashes):
    files = (
        ("model.safetensors", MODEL_SHA256),
        ("config.json", CONFIG_SHA256),
        ("processor_config.json", PROCESSOR_CONFIG_SHA256),
    )
    for filename, expected_hash in files:
        path = model_dir / filename
        if not path.is_file():
            raise Nemotron3DiarizationError("model asset is missing: %s" % path)
        if verify_hashes and _file_sha256(path) != expected_hash:
            raise Nemotron3DiarizationError("model asset SHA-256 mismatch: %s" % path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Nemotron3DiarizationError("%s must be numeric" % label)
    value = float(value)
    if not math.isfinite(value):
        raise Nemotron3DiarizationError("%s must be finite" % label)
    return value
