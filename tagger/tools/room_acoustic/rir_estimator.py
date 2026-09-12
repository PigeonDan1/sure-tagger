"""Rec-RIR room impulse response estimator.

This module intentionally contains no non-Rec-RIR fallback. If the Rec-RIR
repository, checkpoint, dependencies, or inference output is unavailable,
callers must leave `room_acoustic.rir` as null.
"""

from numbers import Real
from pathlib import Path
import math
import sys
import tempfile

from tagger.local_config import (
    RECRIR_CHECKPOINT_PATH,
    RECRIR_CONFIG_PATH,
    RECRIR_MODEL_VERSION,
    RECRIR_PYTHON,
    RECRIR_REPO_DIR,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "rir_estimator"
METHOD = "Rec-RIR"
MODEL_NAME = "Rec-RIR"
SUPPORTED_SAMPLE_RATE_HZ = 16000
SUPPORTED_CHANNELS = 1
ROUND_DIGITS = 8
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIN_EXPECTED_CHECKPOINT_BYTES = 1024 * 1024
MAX_RIR_SECONDS = 2.0


class RecRirError(RuntimeError):
    """Raised when Rec-RIR cannot produce a valid RIR payload."""


class RecRirConfig:
    """Fixed Rec-RIR configuration used by the sound-field pipeline."""

    def __init__(
        self,
        repo_dir=None,
        config_path=None,
        checkpoint_path=None,
        use_gpu=False,
        model_version=None,
        max_rir_seconds=MAX_RIR_SECONDS,
        subprocess_python=None,
    ):
        configured_repo_dir = getattr(RECRIR_REPO_DIR, "strip", lambda: "")()
        configured_config_path = getattr(RECRIR_CONFIG_PATH, "strip", lambda: "")()
        configured_checkpoint_path = getattr(
            RECRIR_CHECKPOINT_PATH,
            "strip",
            lambda: "",
        )()
        configured_python = getattr(RECRIR_PYTHON, "strip", lambda: "")()
        self.repo_dir = _resolve_project_path(repo_dir or configured_repo_dir)
        self.config_path = _resolve_project_path(
            config_path or configured_config_path
        )
        self.checkpoint_path = _resolve_project_path(
            checkpoint_path or configured_checkpoint_path
        )
        self.use_gpu = bool(use_gpu)
        self.model_version = model_version or RECRIR_MODEL_VERSION
        self.max_rir_seconds = float(max_rir_seconds)
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )
        if self.max_rir_seconds <= 0:
            raise RecRirError("max_rir_seconds must be positive")

    def cache_key(self):
        return (
            self.repo_dir,
            self.config_path,
            self.checkpoint_path,
            self.use_gpu,
            self.model_version,
            self.max_rir_seconds,
            self.subprocess_python,
        )

    def to_record(self):
        return {
            "model": MODEL_NAME,
            "repo_dir": self.repo_dir,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "model_version": self.model_version,
            "use_gpu": self.use_gpu,
            "supported_sample_rate_hz": SUPPORTED_SAMPLE_RATE_HZ,
            "supported_channels": SUPPORTED_CHANNELS,
            "max_rir_seconds": self.max_rir_seconds,
            "subprocess_python": self.subprocess_python,
        }


class RecRirClient:
    """Thin adapter around the official Rec-RIR Python modules."""

    def __init__(self, config=None):
        self.config = config or RecRirConfig()
        self._runtime = None

    def estimate_rir(self, audio_path, context=None):
        runtime = self._get_runtime(context)
        model_audio_path, cleanup_path = self._prepare_model_audio(audio_path)
        try:
            input_wav = runtime["transform"].load_wav(
                str(model_audio_path),
                SUPPORTED_SAMPLE_RATE_HZ,
            ).to(runtime["device"])
            rir = runtime["pim"].process(
                input_wav,
                runtime["model"],
                runtime["transform"],
                runtime["device"],
            )
        except Exception as exc:  # noqa: BLE001 - converted to internal warning.
            raise RecRirError("Rec-RIR inference failed") from exc
        finally:
            if cleanup_path is not None:
                try:
                    Path(cleanup_path).unlink()
                except OSError:
                    pass

        values = _to_python_value(rir)
        if isinstance(values, list) and values and isinstance(values[0], list):
            if len(values) != 1:
                raise RecRirError("Rec-RIR returned multi-channel RIR output")
            values = values[0]
        return {
            "sample_rate_hz": SUPPORTED_SAMPLE_RATE_HZ,
            "samples": values,
        }

    def _get_runtime(self, context=None):
        if context is None:
            if self._runtime is None:
                self._runtime = self._load_runtime()
            return self._runtime

        cache = context.setdefault("recrir_runtime_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_runtime()
        return cache[key]

    def _load_runtime(self):
        self._validate_paths()
        self._add_repo_dir_to_path()
        try:
            import torch
            import toml
            from trainer_inferencer.utils import initialize_module
        except ImportError as exc:
            raise RecRirError(
                "Rec-RIR dependencies are not importable; install the official "
                "Rec-RIR requirements first"
            ) from exc

        try:
            config = toml.load(self.config.config_path)
            transform = initialize_module(
                config["acoustic"]["path"],
                config["acoustic"]["args"],
            )
            model = initialize_module(config["model"]["path"], config["model"]["args"])
            device_name = (
                "cuda:0"
                if self.config.use_gpu and torch.cuda.is_available()
                else "cpu"
            )
            device = torch.device(device_name)
            model.to(device)

            checkpoint = torch.load(self.config.checkpoint_path, map_location=device)
            state_dict = {}
            for key, value in checkpoint["model"].items():
                if any(name in key for name in ("ops", "params")):
                    continue
                clean_key = key[7:] if key.startswith("module.") else key
                state_dict[clean_key] = value
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            pim = initialize_module(config["EM_algo"]["path"], config["EM_algo"]["args"])
        except Exception as exc:  # noqa: BLE001 - model loading failures are internal.
            raise RecRirError("Rec-RIR model loading failed") from exc

        return {
            "transform": transform,
            "model": model,
            "pim": pim,
            "device": device,
        }

    def _validate_paths(self):
        repo_dir = Path(self.config.repo_dir)
        config_path = Path(self.config.config_path)
        checkpoint_path = Path(self.config.checkpoint_path)
        if not repo_dir.exists():
            raise RecRirError("Rec-RIR repo dir is missing: %s" % repo_dir)
        if not config_path.exists() or not config_path.is_file():
            raise RecRirError("Rec-RIR config is missing: %s" % config_path)
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise RecRirError("Rec-RIR checkpoint is missing: %s" % checkpoint_path)
        if checkpoint_path.stat().st_size < MIN_EXPECTED_CHECKPOINT_BYTES:
            raise RecRirError(
                "Rec-RIR checkpoint is too small or incomplete: %s"
                % checkpoint_path
            )
        with checkpoint_path.open("rb") as handle:
            header = handle.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RecRirError(
                "Rec-RIR checkpoint is a Git LFS pointer, not model weights: %s"
                % checkpoint_path
            )

    def _add_repo_dir_to_path(self):
        repo_dir = str(Path(self.config.repo_dir))
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    def _prepare_model_audio(self, audio_path):
        try:
            import torchaudio
            import torchaudio.functional as audio_functional
        except ImportError as exc:
            raise RecRirError(
                "Rec-RIR audio preprocessing dependencies are not importable"
            ) from exc

        try:
            info = torchaudio.info(str(audio_path), backend="soundfile")
            if (
                info.sample_rate == SUPPORTED_SAMPLE_RATE_HZ
                and info.num_channels == SUPPORTED_CHANNELS
            ):
                return Path(audio_path), None

            waveform, sample_rate_hz = torchaudio.load(
                str(audio_path),
                channels_first=True,
                backend="soundfile",
            )
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate_hz != SUPPORTED_SAMPLE_RATE_HZ:
                waveform = audio_functional.resample(
                    waveform,
                    sample_rate_hz,
                    SUPPORTED_SAMPLE_RATE_HZ,
                )
            if waveform.numel() <= 0:
                raise RecRirError("Rec-RIR input audio is empty after preprocessing")

            handle = tempfile.NamedTemporaryFile(
                prefix="tagger_recrir_",
                suffix=".wav",
                delete=False,
            )
            temp_path = Path(handle.name)
            handle.close()
            torchaudio.save(
                str(temp_path),
                waveform.cpu(),
                SUPPORTED_SAMPLE_RATE_HZ,
                backend="soundfile",
            )
            return temp_path, temp_path
        except RecRirError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to internal warning.
            raise RecRirError("Rec-RIR audio preprocessing failed") from exc


class RecRirSubprocessClient:
    """Adapter that runs Rec-RIR in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or RecRirConfig()

    def estimate_rir(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "recrir_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    config = config or RecRirConfig()
    client = client or _default_client(config)
    output = client.estimate_rir(audio_path, context=context)
    payload = normalize_rir_payload(
        output,
        max_samples=int(SUPPORTED_SAMPLE_RATE_HZ * config.max_rir_seconds),
    )
    return ToolResult(
        tag_path="room_acoustic.rir",
        value=payload,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        status="estimated",
        confidence=1.0,
        tool_type="model_inference",
        evidence={
            "recrir_config": config.to_record(),
            "sample_count": len(payload["samples"]),
        },
    )


def _default_client(config):
    if config.subprocess_python:
        return RecRirSubprocessClient(config)
    return RecRirClient(config)


def _subprocess_config(config):
    return {
        "repo_dir": config.repo_dir,
        "config_path": config.config_path,
        "checkpoint_path": config.checkpoint_path,
        "use_gpu": config.use_gpu,
        "model_version": config.model_version,
        "max_rir_seconds": config.max_rir_seconds,
        "subprocess_python": "",
    }


def normalize_rir_payload(payload, max_samples=None):
    return _parse_rir_payload(payload, max_samples=max_samples, normalize_peak=True)


def validate_rir_payload(payload):
    return _parse_rir_payload(payload, max_samples=None, normalize_peak=False)


def _parse_rir_payload(payload, max_samples=None, normalize_peak=False):
    if not isinstance(payload, dict):
        raise RecRirError("Rec-RIR output must be an object")
    if set(payload.keys()) != set(["sample_rate_hz", "samples"]):
        raise RecRirError("Rec-RIR output must contain only sample_rate_hz and samples")

    sample_rate_hz = payload["sample_rate_hz"]
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise RecRirError("rir.sample_rate_hz must be a positive integer")

    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise RecRirError("rir.samples must be a non-empty array")
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    normalized = []
    peak = 0.0
    for index, value in enumerate(samples):
        number = _require_finite_number(value, "rir.samples[%s]" % index)
        if not normalize_peak and abs(number) > 1.0:
            raise RecRirError("rir.samples[%s] is outside [-1, 1]" % index)
        peak = max(peak, abs(number))
        normalized.append(number)
    if peak <= 0.0:
        raise RecRirError("rir.samples must contain non-zero energy")

    if normalize_peak and peak > 1.0:
        normalized = [value / peak for value in normalized]

    return {
        "sample_rate_hz": sample_rate_hz,
        "samples": [round(float(value), ROUND_DIGITS) for value in normalized],
    }


def _require_finite_number(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RecRirError("%s must be a number" % path)
    value = float(value)
    if not math.isfinite(value):
        raise RecRirError("%s must be finite" % path)
    return value


def _to_python_value(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _resolve_project_path(configured_path):
    if not configured_path:
        return ""
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)
