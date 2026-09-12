"""Qwen3-ASR-0.6B adapter for multilingual speaker transcription."""

import copy
from pathlib import Path
import time

from tagger.tools.base import ToolResult
from tagger.tools.firered_audio import prepare_firered_audio
from tagger.tools.subprocess_runner import run_subprocess_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-ASR-0.6B"
DEFAULT_SUBPROCESS_PYTHON = (
    PROJECT_ROOT
    / ".runtime"
    / "qwen3_asr_0_6b_py311_torch280_cu128_v1"
    / "bin"
    / "python"
)

TOOL_NAME = "qwen3_asr_0_6b"
METHOD = "Qwen3-ASR-0.6B"
TOOL_VERSION = "qwen3_asr_0_6b_v1.0.0"
SOURCE_VERSION = (
    "ModelScope:Qwen/Qwen3-ASR-0.6B@3b885f72b1733a6a50dc17a597fb4135c3d656a0"
)
SCHEMA_VERSION = "qwen3.asr.transcription.v1"
CHECKPOINT_SHA256 = (
    "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
ROUND_DIGITS = 6


class Qwen3AsrError(RuntimeError):
    """Raised when Qwen3-ASR cannot produce a valid transcription."""


class Qwen3AsrConfig:
    def __init__(
        self,
        model_dir=None,
        subprocess_python=None,
        device="cuda:0",
        torch_dtype="bfloat16",
        max_new_tokens=4096,
        timeout_sec=1800,
        normalize_to_16k_mono_pcm=True,
    ):
        self.model_dir = str(model_dir or DEFAULT_MODEL_DIR)
        self.subprocess_python = str(
            DEFAULT_SUBPROCESS_PYTHON
            if subprocess_python is None
            else subprocess_python or ""
        )
        self.device = str(device or "cuda:0")
        self.torch_dtype = str(torch_dtype or "bfloat16")
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_sec = int(timeout_sec)
        self.normalize_to_16k_mono_pcm = bool(normalize_to_16k_mono_pcm)
        if self.torch_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("unsupported Qwen3 ASR torch_dtype: %s" % self.torch_dtype)
        if self.max_new_tokens < 1:
            raise ValueError("Qwen3 ASR max_new_tokens must be positive")
        if self.timeout_sec < 1:
            raise ValueError("Qwen3 ASR timeout_sec must be positive")

    def cache_key(self):
        return (
            self.model_dir,
            self.subprocess_python,
            self.device,
            self.torch_dtype,
            self.max_new_tokens,
            self.normalize_to_16k_mono_pcm,
        )

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "timeout_sec": self.timeout_sec,
            "normalize_to_16k_mono_pcm": self.normalize_to_16k_mono_pcm,
        }


class _DirectQwen3AsrRuntime:
    def __init__(self, config):
        self.config = config
        self.model_dir = _resolve_project_path(config.model_dir).resolve()
        _validate_model_files(self.model_dir)
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise Qwen3AsrError(
                "qwen-asr and torch must be importable in the Qwen3 ASR runtime"
            ) from exc

        self._torch = torch
        self.device = _resolve_device(torch, config.device)
        dtype = getattr(torch, config.torch_dtype)
        started = time.time()
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                str(self.model_dir),
                dtype=dtype,
                device_map=str(self.device),
                max_inference_batch_size=1,
                max_new_tokens=config.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - normalize model failures.
            raise Qwen3AsrError(
                "Qwen3 ASR checkpoint loading failed: %s" % exc
            ) from exc
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.model_load_sec = time.time() - started

    def transcribe(self, audio_path):
        started = time.time()
        try:
            results = self.model.transcribe(
                audio=str(audio_path),
                language=None,
                return_time_stamps=False,
            )
        except Exception as exc:  # noqa: BLE001 - normalize inference failures.
            raise Qwen3AsrError("Qwen3 ASR inference failed: %s" % exc) from exc
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        if not isinstance(results, list) or len(results) != 1:
            raise Qwen3AsrError("Qwen3 ASR returned a non-single result list")
        native = results[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "text": getattr(native, "text", ""),
            "language": getattr(native, "language", "") or None,
            "timestamps": [],
            "model": {
                "name": "Qwen3-ASR-0.6B",
                "model_dir": str(self.model_dir),
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            "runtime": {
                "device": str(self.device),
                "dtype": self.config.torch_dtype,
                "model_load_sec": round(self.model_load_sec, ROUND_DIGITS),
                "inference_sec": round(time.time() - started, ROUND_DIGITS),
                "max_new_tokens": self.config.max_new_tokens,
                "torch_version": self._torch.__version__,
                "torch_cuda_version": self._torch.version.cuda,
            },
        }


class Qwen3AsrClient:
    def __init__(self, config=None):
        self.config = config or Qwen3AsrConfig()
        self._runtime = None

    def transcribe(self, audio_path, context=None):
        original_path = Path(audio_path).expanduser().resolve()
        if not original_path.is_file():
            raise Qwen3AsrError("audio file does not exist: %s" % original_path)
        prepared_path, cleanup = _prepare_audio(original_path, self.config, context)
        try:
            runtime = self._get_runtime(context)
            result = normalize_result(
                runtime.transcribe(prepared_path), audio_path=original_path
            )
            result.setdefault("runtime", {})
            result["runtime"].update(
                {
                    "input_audio_path": str(original_path),
                    "prepared_audio_path": str(prepared_path),
                    "audio_normalized": str(prepared_path) != str(original_path),
                }
            )
            return result
        finally:
            if cleanup is not None:
                cleanup.cleanup()

    def _get_runtime(self, context=None):
        if context is None:
            if self._runtime is None:
                self._runtime = _DirectQwen3AsrRuntime(self.config)
            return self._runtime
        cache = context.setdefault("qwen3_asr_runtime_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = _DirectQwen3AsrRuntime(self.config)
        return cache[key]


class Qwen3AsrSubprocessClient:
    def __init__(self, config=None):
        self.config = config or Qwen3AsrConfig()

    def transcribe(self, audio_path, context=None):
        original_path = Path(audio_path).expanduser().resolve()
        if not original_path.is_file():
            raise Qwen3AsrError("audio file does not exist: %s" % original_path)
        prepared_path, cleanup = _prepare_audio(original_path, self.config, context)
        try:
            result = run_subprocess_tool(
                self.config.subprocess_python,
                "qwen3_asr_estimate",
                {
                    "audio_path": str(prepared_path),
                    "config": _subprocess_config(self.config),
                },
                context=context,
                timeout_sec=self.config.timeout_sec,
            )
            if isinstance(result, dict) and "output" in result:
                result = result["output"]
            normalized = normalize_result(result, audio_path=original_path)
            normalized.setdefault("runtime", {})
            normalized["runtime"].update(
                {
                    "input_audio_path": str(original_path),
                    "prepared_audio_path": str(prepared_path),
                    "audio_normalized": str(prepared_path) != str(original_path),
                }
            )
            return normalized
        except Qwen3AsrError:
            raise
        except Exception as exc:  # noqa: BLE001 - expose one stable tool error.
            raise Qwen3AsrError(
                "Qwen3 ASR subprocess inference failed: %s" % exc
            ) from exc
        finally:
            if cleanup is not None:
                cleanup.cleanup()


def transcribe(audio_path, config=None, context=None, client=None):
    config = config or Qwen3AsrConfig()
    if client is None:
        client = (
            Qwen3AsrSubprocessClient(config)
            if config.subprocess_python
            else Qwen3AsrClient(config)
        )
    try:
        return client.transcribe(audio_path, context=context)
    except Qwen3AsrError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize custom client failures.
        raise Qwen3AsrError("Qwen3 ASR transcription failed") from exc


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    result = transcribe(audio_path, config=config, context=context, client=client)
    return ToolResult(
        tag_path="speaker.asr_transcript",
        value=result,
        tool_name=TOOL_NAME,
        method=METHOD,
        status="estimated",
        confidence=0.0,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence={
            "source_version": SOURCE_VERSION,
            "schema_version": result.get("schema_version", SCHEMA_VERSION),
            "text": result.get("text", ""),
            "language": result.get("language"),
            "timestamps": copy.deepcopy(result.get("timestamps", [])),
            "runtime": copy.deepcopy(result.get("runtime", {})),
            "model": copy.deepcopy(result.get("model", {})),
        },
    )


def normalize_result(value, audio_path=None):
    if not isinstance(value, dict):
        raise Qwen3AsrError("Qwen3 ASR result must be an object")
    text = value.get("text", value.get("transcript", ""))
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    language = value.get("language") or value.get("lang")
    if language is not None:
        if not isinstance(language, str):
            raise Qwen3AsrError("Qwen3 ASR language must be a string")
        language = language.strip() or None
    result = {
        "schema_version": str(value.get("schema_version") or SCHEMA_VERSION),
        "text": " ".join(text.replace("\t", " ").split()),
        "language": language,
        "timestamps": [],
        "runtime": copy.deepcopy(value.get("runtime", {})),
        "model": copy.deepcopy(value.get("model", {})),
    }
    if audio_path is not None:
        result["audio_path"] = str(Path(audio_path).expanduser().resolve())
    return result


def _resolve_device(torch, device_name):
    try:
        device = torch.device(str(device_name))
    except Exception as exc:  # noqa: BLE001
        raise Qwen3AsrError("unsupported Qwen3 ASR device: %s" % device_name) from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Qwen3AsrError("CUDA was requested for Qwen3 ASR but is unavailable")
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise Qwen3AsrError("Qwen3 ASR CUDA device %s is unavailable" % index)
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise Qwen3AsrError("unsupported Qwen3 ASR device: %s" % device_name)
    return device


def _prepare_audio(audio_path, config, context):
    try:
        return prepare_firered_audio(
            audio_path,
            context=context,
            normalize_to_16k_mono_pcm=config.normalize_to_16k_mono_pcm,
            error_class=Qwen3AsrError,
            tool_label="Qwen3 ASR",
            temp_prefix="qwen3_asr_",
        )
    except Qwen3AsrError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Qwen3AsrError("failed to prepare Qwen3 ASR audio") from exc


def _validate_model_files(model_dir):
    missing = [
        name for name in REQUIRED_MODEL_FILES if not (Path(model_dir) / name).is_file()
    ]
    if missing:
        raise Qwen3AsrError(
            "incomplete Qwen3-ASR-0.6B model at %s; missing %s"
            % (model_dir, ", ".join(missing))
        )


def _resolve_project_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _subprocess_config(config):
    record = config.to_record()
    record["subprocess_python"] = ""
    record["normalize_to_16k_mono_pcm"] = False
    record.pop("timeout_sec", None)
    return record


__all__ = [
    "CHECKPOINT_SHA256",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SUBPROCESS_PYTHON",
    "Qwen3AsrClient",
    "Qwen3AsrConfig",
    "Qwen3AsrError",
    "Qwen3AsrSubprocessClient",
    "normalize_result",
    "run",
    "transcribe",
]
