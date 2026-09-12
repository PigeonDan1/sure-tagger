"""FireRedASR2-AED multilingual ASR adapter for speaker-v2.

The FireRed checkpoint is intentionally kept out of the main interpreter.  In
normal pipeline use :class:`FireRedAsrSubprocessClient` sends one request to a
long-lived ``firered_asr_estimate`` worker, where the 4.7 GB AED checkpoint is
loaded once and reused.  The in-process client is retained for small smoke
tests and callers that already run in the dedicated FireRed environment.

The upstream deployment adapter lives in ``.deploy/fireredasr2_aed`` in the
shared ``tagger`` checkout.  This module loads that adapter by path rather than
importing its heavyweight dependencies at module import time.
"""

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

from tagger.tools.base import ToolResult
from tagger.tools.firered_audio import prepare_firered_audio
from tagger.tools.subprocess_runner import run_subprocess_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHARED_TAGGER_ROOT = PROJECT_ROOT.parent / "tagger"

_DEPLOYMENT_CANDIDATES = (
    PROJECT_ROOT / ".deploy" / "fireredasr2_aed",
    _SHARED_TAGGER_ROOT / ".deploy" / "fireredasr2_aed",
)


def _first_existing_directory(candidates):
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


DEFAULT_DEPLOYMENT_DIR = _first_existing_directory(_DEPLOYMENT_CANDIDATES)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "FireRedASR2-AED"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "models" / "FireRedASR2S"
DEFAULT_LID_MODEL_DIR = (
    DEFAULT_SOURCE_DIR / "pretrained_models" / "FireRedLID"
)
DEFAULT_SUBPROCESS_PYTHON = (
    PROJECT_ROOT
    / ".runtime"
    / "fireredasr2_aed_py311_torch280_cu128_v1"
    / "bin"
    / "python"
)

TOOL_NAME = "firered_asr2_aed"
METHOD = "FireRedASR2-AED"
TOOL_VERSION = "firered_asr2_aed_v1.0.0"
# The checkpoint is the ModelScope mirror of the FireRedTeam release.  Keep a
# single provenance spelling across adapter, evidence and evaluation docs.
SOURCE_VERSION = "ModelScope:xukaituo/FireRedASR2-AED@shared-20260831"
SCHEMA_VERSION = "fireredasr2.aed.transcription.v1"
CHECKPOINT_SHA256 = (
    "4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054"
)
LID_CHECKPOINT_SHA256 = (
    "7dee2a280e9b11d5241a0e3d4fa60ee1520a036a2e8385f17960371cfea10093"
)
EXPECTED_SAMPLE_RATE_HZ = 16000
REQUIRED_MODEL_FILES = (
    "cmvn.ark",
    "dict.txt",
    "train_bpe1000.model",
    "model.pth.tar",
)
REQUIRED_LID_MODEL_FILES = (
    "cmvn.ark",
    "dict.txt",
    "model.pth.tar",
)
ROUND_DIGITS = 6


class FireRedAsrError(RuntimeError):
    """Raised when FireRedASR2-AED cannot produce a valid result."""


class FireRedAsrConfig:
    """Configuration for the FireRedASR2-AED adapter.

    ``model_dir`` and ``source_dir`` may be absolute paths or paths relative
    to the sure-tagger project root.  ``deployment_dir`` points at the small
    wrapper containing ``FireRedAedRuntime``; it does not contain the model
    weights themselves.
    """

    def __init__(
        self,
        model_dir=None,
        source_dir=None,
        subprocess_python=None,
        device="auto",
        use_half=False,
        beam_size=3,
        timeout_sec=900,
        normalize_to_16k_mono_pcm=True,
        deployment_dir=None,
        adapter_path=None,
        enable_lid=True,
        lid_model_dir=None,
        lid_use_half=False,
        verify_lid_model_asset=True,
        # Friendly aliases used by a few deployment scripts.
        model_path=None,
        source_path=None,
        use_gpu=None,
        half=None,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError("unexpected FireRed ASR config fields: %s" % unknown)
        if model_dir is None:
            model_dir = model_path
        if source_dir is None:
            source_dir = source_path
        if half is not None:
            use_half = half
        if use_gpu is not None:
            # ``use_gpu`` is an upstream-style boolean.  An explicit device
            # still wins when it is supplied instead of the default ``auto``.
            if str(device) == "auto":
                device = "cuda:0" if use_gpu else "cpu"

        self.model_dir = str(model_dir or DEFAULT_MODEL_DIR)
        self.source_dir = str(source_dir or DEFAULT_SOURCE_DIR)
        self.subprocess_python = str(
            DEFAULT_SUBPROCESS_PYTHON
            if subprocess_python is None
            else subprocess_python or ""
        )
        self.device = str(device or "auto")
        self.use_half = bool(use_half)
        self.beam_size = int(beam_size)
        self.timeout_sec = int(timeout_sec)
        self.normalize_to_16k_mono_pcm = bool(normalize_to_16k_mono_pcm)
        selected_deployment_dir = deployment_dir
        if selected_deployment_dir is None and adapter_path is not None:
            selected_deployment_dir = Path(str(adapter_path)).expanduser().parent
        self.deployment_dir = str(
            selected_deployment_dir or DEFAULT_DEPLOYMENT_DIR
        )
        self.adapter_path = str(adapter_path or "")
        self.enable_lid = bool(enable_lid)
        # Keep the LID checkpoint beside the selected FireRedASR2S source by
        # default.  This also makes a custom source tree self-contained.
        self.lid_model_dir = str(
            lid_model_dir
            or (Path(self.source_dir) / "pretrained_models" / "FireRedLID")
        )
        self.lid_use_half = bool(lid_use_half)
        self.verify_lid_model_asset = bool(verify_lid_model_asset)

        if self.beam_size < 1:
            raise ValueError("FireRed ASR beam_size must be positive")
        if self.timeout_sec <= 0:
            raise ValueError("FireRed ASR timeout_sec must be positive")

    def cache_key(self):
        return (
            self.model_dir,
            self.source_dir,
            self.subprocess_python,
            self.device,
            self.use_half,
            self.beam_size,
            self.normalize_to_16k_mono_pcm,
            self.deployment_dir,
            self.adapter_path,
            self.enable_lid,
            self.lid_model_dir,
            self.lid_use_half,
            self.verify_lid_model_asset,
        )

    def to_record(self):
        """Return a JSON-safe record suitable for run manifests and workers."""

        return {
            "model_dir": self.model_dir,
            "source_dir": self.source_dir,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "use_half": self.use_half,
            "beam_size": self.beam_size,
            "timeout_sec": self.timeout_sec,
            "normalize_to_16k_mono_pcm": self.normalize_to_16k_mono_pcm,
            "deployment_dir": self.deployment_dir,
            "adapter_path": self.adapter_path,
            "enable_lid": self.enable_lid,
            "lid_model_dir": self.lid_model_dir,
            "lid_use_half": self.lid_use_half,
            "verify_lid_model_asset": self.verify_lid_model_asset,
        }


class _DirectFireRedRuntime:
    """Minimal reusable runtime built directly from FireRedASR2S sources."""

    def __init__(self, config):
        self.config = config
        self.model_dir = _resolve_project_path(config.model_dir).resolve()
        self.source_dir = _resolve_project_path(config.source_dir).resolve()
        _validate_model_files(self.model_dir)

        package_root = self.source_dir / "fireredasr2s"
        if not package_root.is_dir():
            raise FireRedAsrError(
                "FireRedASR2S source package is missing: %s" % package_root
            )
        package_path = str(package_root)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

        try:
            import torch
            from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
        except ImportError as exc:
            raise FireRedAsrError(
                "FireRedASR2S and torch must be importable in the ASR runtime"
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(torch, config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        upstream_config = FireRedAsr2Config(
            use_gpu=self.device.type == "cuda",
            use_half=config.use_half,
            beam_size=config.beam_size,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        started = time.time()
        try:
            self.model = FireRedAsr2.from_pretrained(
                "aed", str(self.model_dir), upstream_config
            )
        except Exception as exc:  # noqa: BLE001 - normalized for pipeline use.
            raise FireRedAsrError(
                "FireRedASR2-AED checkpoint loading failed: %s" % exc
            ) from exc
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.model_load_s = time.time() - started

        # LID is a routing aid, not a reason to lose an otherwise valid ASR
        # observation.  Keep its load failure on the runtime so the caller can
        # record a deterministic language_error while returning ASR text.
        self.lid_runtime = None
        self.lid_load_error = None
        if not config.enable_lid:
            self.lid_status = "disabled"
        else:
            try:
                self.lid_runtime = _DirectFireRedLidRuntime(
                    config,
                    torch_module=torch,
                    device=self.device,
                )
                self.lid_status = "loaded"
            except Exception as exc:  # noqa: BLE001 - ASR remains usable.
                self.lid_load_error = _format_runtime_error(exc)
                self.lid_status = "unavailable"

    @staticmethod
    def _resolve_device(torch, device_name):
        if str(device_name) == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            device = torch.device(str(device_name))
        except Exception as exc:  # noqa: BLE001
            raise FireRedAsrError(
                "unsupported FireRed ASR device: %s" % device_name
            ) from exc
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise FireRedAsrError(
                    "CUDA was requested for FireRed ASR but is unavailable"
                )
            index = 0 if device.index is None else int(device.index)
            if index >= torch.cuda.device_count():
                raise FireRedAsrError(
                    "FireRed ASR CUDA device %s is unavailable" % index
                )
            device = torch.device("cuda", index)
        elif device.type != "cpu":
            raise FireRedAsrError(
                "unsupported FireRed ASR device: %s" % device_name
            )
        return device

    def transcribe(self, audio_path):
        started = time.time()
        try:
            results = self.model.transcribe(
                ["input-0000"],
                [str(audio_path)],
            )
        except Exception as exc:  # noqa: BLE001
            raise FireRedAsrError("FireRed ASR inference failed") from exc
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        elapsed = time.time() - started
        if not isinstance(results, list) or len(results) != 1:
            raise FireRedAsrError(
                "FireRed ASR returned a non-single result list"
            )
        native = results[0]
        if not isinstance(native, dict):
            raise FireRedAsrError("FireRed ASR returned a non-object result")
        output = copy.deepcopy(native)
        output.setdefault("schema_version", SCHEMA_VERSION)
        output.setdefault("audio_path", str(audio_path))
        output.setdefault(
            "model",
            {
                "name": "FireRedASR2-AED",
                "model_dir": str(self.model_dir),
                "source_dir": str(self.source_dir),
            },
        )
        output.setdefault(
            "runtime",
            {
                "device": str(self.device),
                "device_name": (
                    self._torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else "CPU"
                ),
                "dtype": self._dtype_name(),
                "beam_size": self.config.beam_size,
                "model_load_s": round(self.model_load_s, ROUND_DIGITS),
                "batch_inference_s": round(elapsed, ROUND_DIGITS),
                "batch_size": 1,
                "torch_version": self._torch.__version__,
                "torch_cuda_version": self._torch.version.cuda,
            },
        )
        return output

    def detect_language(self, audio_path):
        if not self.config.enable_lid:
            raise FireRedAsrError("FireRed LID is disabled")
        if self.lid_runtime is None:
            detail = self.lid_load_error or "runtime was not initialized"
            raise FireRedAsrError("FireRed LID is unavailable: %s" % detail)
        return self.lid_runtime.detect_language(audio_path)

    # Keep a verb matching the ASR API available to deployment callers that
    # treat language identification as a second transcription operation.
    def transcribe_language(self, audio_path):
        return self.detect_language(audio_path)

    def _dtype_name(self):
        if not self.config.use_half:
            return "float32"
        if self.device.type == "cuda" and self._torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"


class _DirectFireRedLidRuntime:
    """Reusable FireRed LID runtime loaded beside the AED model."""

    def __init__(self, config, torch_module=None, device=None):
        self.config = config
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise FireRedAsrError(
                    "torch is required for FireRed LID"
                ) from exc
        self._torch = torch_module
        if device is None:
            device = _DirectFireRedRuntime._resolve_device(
                torch_module, config.device
            )
        self.device = device
        self.source_dir = _resolve_project_path(config.source_dir).resolve()
        self.model_dir = _resolve_project_path(config.lid_model_dir).resolve()
        self.checkpoint_sha256 = _validate_lid_model_files(
            self.model_dir,
            verify_checkpoint=config.verify_lid_model_asset,
        )

        package_root = self.source_dir / "fireredasr2s"
        if not package_root.is_dir():
            raise FireRedAsrError(
                "FireRedASR2S source package is missing: %s" % package_root
            )
        package_path = str(package_root)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

        try:
            from fireredlid.lid import (
                FireRedLid,
                FireRedLidConfig,
            )
        except ImportError as exc:
            raise FireRedAsrError(
                "FireRed LID sources must be importable in the ASR runtime"
            ) from exc

        upstream_config = FireRedLidConfig(
            use_gpu=self.device.type == "cuda",
            use_half=config.lid_use_half,
        )
        started = time.time()
        try:
            self.model = FireRedLid.from_pretrained(
                str(self.model_dir), upstream_config
            )
        except Exception as exc:  # noqa: BLE001 - normalized at caller.
            raise FireRedAsrError(
                "FireRed LID checkpoint loading failed: %s" % exc
            ) from exc
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        self.model_load_s = time.time() - started

    def detect_language(self, audio_path):
        started = time.time()
        try:
            results = self.model.process(
                ["input-0000"],
                [str(audio_path)],
            )
        except Exception as exc:  # noqa: BLE001
            raise FireRedAsrError("FireRed LID inference failed") from exc
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        elapsed = time.time() - started
        if not isinstance(results, list) or len(results) != 1:
            raise FireRedAsrError(
                "FireRed LID returned a non-single result list"
            )
        native = results[0]
        if not isinstance(native, dict):
            raise FireRedAsrError("FireRed LID returned a non-object result")
        observation = _normalize_lid_observation(native)
        observation["runtime"] = {
            "device": str(self.device),
            "device_name": (
                self._torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "CPU"
            ),
            "dtype": (
                "float16" if self.config.lid_use_half else "float32"
            ),
            "model_load_s": round(self.model_load_s, ROUND_DIGITS),
            "inference_s": round(elapsed, ROUND_DIGITS),
            "torch_version": self._torch.__version__,
            "torch_cuda_version": self._torch.version.cuda,
            "lid_model_sha256": self.checkpoint_sha256,
            "lid_model_expected_sha256": LID_CHECKPOINT_SHA256,
            "lid_model_asset_verified": self.checkpoint_sha256 is not None,
        }
        observation["model"] = {
            "name": "FireRedLID",
            "model_dir": str(self.model_dir),
            "source_dir": str(self.source_dir),
            "checkpoint_sha256": self.checkpoint_sha256,
            "expected_checkpoint_sha256": LID_CHECKPOINT_SHA256,
            "model_asset_verified": self.checkpoint_sha256 is not None,
        }
        return observation


class _DeploymentFireRedRuntime:
    """Delegate an external AED runtime while adding same-process FireRed LID."""

    def __init__(self, asr_runtime, config):
        self._asr_runtime = asr_runtime
        self.config = config
        self.lid_runtime = None
        self.lid_load_error = None
        if not config.enable_lid:
            self.lid_status = "disabled"
            return

        try:
            torch_module = getattr(asr_runtime, "_torch", None)
            if torch_module is None:
                import torch as torch_module
            device = getattr(asr_runtime, "device", None)
            if device is None or not hasattr(device, "type"):
                device = _DirectFireRedRuntime._resolve_device(
                    torch_module, config.device
                )
            self.lid_runtime = _DirectFireRedLidRuntime(
                config,
                torch_module=torch_module,
                device=device,
            )
            self.lid_status = "loaded"
        except Exception as exc:  # noqa: BLE001 - preserve external ASR.
            self.lid_load_error = _format_runtime_error(exc)
            self.lid_status = "unavailable"

    def transcribe(self, audio_path):
        return self._asr_runtime.transcribe(audio_path)

    def detect_language(self, audio_path):
        if not self.config.enable_lid:
            raise FireRedAsrError("FireRed LID is disabled")
        if self.lid_runtime is None:
            detail = self.lid_load_error or "runtime was not initialized"
            raise FireRedAsrError("FireRed LID is unavailable: %s" % detail)
        return self.lid_runtime.detect_language(audio_path)

    def transcribe_language(self, audio_path):
        return self.detect_language(audio_path)

    def __getattr__(self, name):
        # Preserve useful attributes (model/device/load timing) exposed by the
        # deployment wrapper for callers that inspect runtime provenance.
        return getattr(self._asr_runtime, name)


class FireRedAsrClient:
    """In-process wrapper around the deployed ``FireRedAedRuntime``."""

    def __init__(self, config=None):
        self.config = config or FireRedAsrConfig()
        self._runtime = None

    def transcribe(self, audio_path, context=None):
        original_path = Path(audio_path).expanduser().resolve()
        if not original_path.is_file():
            raise FireRedAsrError("audio file does not exist: %s" % original_path)

        prepared_path, cleanup = self._prepare_audio(original_path, context)
        try:
            runtime = self._get_runtime(context)
            started = time.time()
            try:
                raw_result = runtime.transcribe(prepared_path)
            except Exception as exc:  # noqa: BLE001 - normalize model errors.
                raise FireRedAsrError("FireRed ASR inference failed") from exc
            result = normalize_result(
                raw_result,
                audio_path=original_path,
            )
            result.setdefault("runtime", {})
            result["runtime"].setdefault(
                "adapter_elapsed_sec", round(time.time() - started, ROUND_DIGITS)
            )
            result["runtime"]["input_audio_path"] = str(original_path)
            result["runtime"]["prepared_audio_path"] = str(prepared_path)
            result["runtime"]["audio_normalized"] = str(prepared_path) != str(
                original_path
            )
            self._attach_language_observation(result, runtime, prepared_path)
            return result
        finally:
            if cleanup is not None:
                cleanup.cleanup()

    def _attach_language_observation(self, result, runtime, audio_path):
        """Run the same-process LID model without invalidating ASR output."""

        runtime_info = result.setdefault("runtime", {})
        if not self.config.enable_lid:
            runtime_info["lid_status"] = "disabled"
            return

        detect = getattr(runtime, "detect_language", None)
        if not callable(detect):
            error = "runtime does not expose FireRed LID"
            result["language_error"] = error
            runtime_info["lid_status"] = "unavailable"
            runtime_info["lid_error"] = error
            return

        started = time.time()
        try:
            raw_lid = detect(audio_path)
            observation = _normalize_lid_observation(raw_lid)
            _merge_lid_observation(result, observation)
            runtime_info["lid_status"] = "ok"
            runtime_info["lid_inference_s"] = round(
                time.time() - started, ROUND_DIGITS
            )
            if isinstance(raw_lid, dict) and isinstance(
                raw_lid.get("runtime"), dict
            ):
                runtime_info["lid_runtime"] = copy.deepcopy(raw_lid["runtime"])
            if isinstance(raw_lid, dict) and isinstance(raw_lid.get("model"), dict):
                runtime_info["lid_model"] = copy.deepcopy(raw_lid["model"])
        except Exception as exc:  # noqa: BLE001 - preserve ASR text.
            error = _format_runtime_error(exc)
            result["language_error"] = error
            runtime_info["lid_status"] = (
                "unavailable"
                if getattr(runtime, "lid_load_error", None)
                else "error"
            )
            runtime_info["lid_error"] = error
            runtime_info["lid_inference_s"] = round(
                time.time() - started, ROUND_DIGITS
            )

    def _prepare_audio(self, audio_path, context=None):
        try:
            return prepare_firered_audio(
                audio_path,
                context=context,
                normalize_to_16k_mono_pcm=self.config.normalize_to_16k_mono_pcm,
                error_class=FireRedAsrError,
                tool_label="FireRed ASR2-AED",
                temp_prefix="firered_asr2_aed_",
            )
        except FireRedAsrError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize decode failures.
            raise FireRedAsrError("failed to prepare FireRed ASR audio") from exc

    def _get_runtime(self, context=None):
        if context is None:
            if self._runtime is None:
                self._runtime = self._load_runtime()
            return self._runtime

        cache = context.setdefault("firered_asr_runtime_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_runtime()
        return cache[key]

    def _load_runtime(self):
        # Direct loading from the checked-out FireRedASR2S source is the
        # canonical path and does not require a deployment wrapper.  An
        # explicit adapter_path is useful for a separately packaged runtime.
        if not self.config.adapter_path:
            return _DirectFireRedRuntime(self.config)

        adapter_path = _resolve_adapter_path(self.config)
        module = _load_deployment_module(adapter_path)
        runtime_class = getattr(module, "FireRedAedRuntime", None)
        if runtime_class is None:
            runtime_class = getattr(module, "FireRedAsrRuntime", None)
        if runtime_class is None:
            raise FireRedAsrError(
                "FireRed deployment adapter does not expose FireRedAedRuntime: %s"
                % adapter_path
            )

        try:
            asr_runtime = runtime_class(
                model_dir=_resolve_project_path(self.config.model_dir),
                source_dir=_resolve_project_path(self.config.source_dir),
                device=self.config.device,
                use_half=self.config.use_half,
                beam_size=self.config.beam_size,
            )
            return _DeploymentFireRedRuntime(asr_runtime, self.config)
        except Exception as exc:  # noqa: BLE001 - normalize load failures.
            raise FireRedAsrError(
                "FireRed ASR runtime loading failed: %s" % exc
            ) from exc


class FireRedAsrSubprocessClient:
    """FireRed client using the shared JSONL subprocess worker."""

    def __init__(self, config=None):
        self.config = config or FireRedAsrConfig()

    def transcribe(self, audio_path, context=None):
        original_path = Path(audio_path).expanduser().resolve()
        if not original_path.is_file():
            raise FireRedAsrError("audio file does not exist: %s" % original_path)

        # Conversion is done in the caller environment.  The model worker then
        # sees a deterministic PCM16 WAV and does not need ffmpeg/libsndfile
        # compatibility with every source format.
        prepared_path, cleanup = _prepare_audio_for_subprocess(
            original_path,
            self.config,
            context,
        )
        try:
            result = run_subprocess_tool(
                self.config.subprocess_python,
                "firered_asr_estimate",
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
            normalized["runtime"]["input_audio_path"] = str(original_path)
            normalized["runtime"]["prepared_audio_path"] = str(prepared_path)
            normalized["runtime"]["audio_normalized"] = str(prepared_path) != str(
                original_path
            )
            return normalized
        except FireRedAsrError:
            raise
        except Exception as exc:  # noqa: BLE001 - include a stable tool error.
            raise FireRedAsrError(
                "FireRed ASR subprocess inference failed: %s"
                % _format_runtime_error(exc)
            ) from exc
        finally:
            if cleanup is not None:
                cleanup.cleanup()


def transcribe(audio_path, config=None, context=None, client=None):
    """Transcribe one audio file and return the normalized result dictionary."""

    config = config or FireRedAsrConfig()
    if client is None:
        client = (
            FireRedAsrSubprocessClient(config)
            if config.subprocess_python
            else FireRedAsrClient(config)
        )
    try:
        return client.transcribe(audio_path, context=context)
    except FireRedAsrError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize custom client failures.
        raise FireRedAsrError("FireRed ASR transcription failed") from exc


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    """Return a :class:`ToolResult` for one FireRed ASR invocation.

    The value is the complete normalized transcription object rather than only
    the text, because timestamp and confidence data are useful to the speaker
    evidence layer.  A model/decoding failure raises ``FireRedAsrError``; the
    caller can then record missing evidence without fabricating text.
    """

    result = transcribe(audio_path, config=config, context=context, client=client)
    confidence = result.get("confidence")
    return ToolResult(
        tag_path="speaker.asr_transcript",
        value=result,
        tool_name=TOOL_NAME,
        method=METHOD,
        status="estimated",
        confidence=confidence if confidence is not None else 0.0,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence={
            "source_version": SOURCE_VERSION,
            "schema_version": result.get("schema_version", SCHEMA_VERSION),
            "text": result.get("text", ""),
            "confidence": confidence,
            "lang": result.get("lang"),
            "language": result.get("language"),
            "language_confidence": result.get("language_confidence"),
            "language_error": result.get("language_error"),
            "timestamps": copy.deepcopy(result.get("timestamps", [])),
            "duration_s": result.get("duration_s"),
            "rtf": result.get("rtf"),
            "runtime": copy.deepcopy(result.get("runtime", {})),
            "model": copy.deepcopy(result.get("model", {})),
        },
    )


def validate_result(value, duration_sec=None):
    """Validate and normalize a FireRed deployment result.

    This accepts the deployment adapter's dictionary and a few equivalent
    timestamp spellings, making the boundary between the upstream model and
    speaker-v2 deterministic.  It intentionally allows an empty ``text``:
    an empty model prediction is a valid observation and must be recorded as
    such instead of silently replaced by another transcript.
    """

    return normalize_result(value, duration_sec=duration_sec)


def normalize_result(value, audio_path=None, duration_sec=None):
    if not isinstance(value, dict):
        raise FireRedAsrError("FireRed ASR result must be an object")

    text = value.get("text", value.get("transcript", ""))
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.replace("\t", " ").split())

    confidence = _optional_probability(value.get("confidence"))
    result_duration = value.get("duration_s", value.get("dur_s"))
    if result_duration is None:
        result_duration = duration_sec
    if result_duration is None:
        result_duration = 0.0
    result_duration = _nonnegative_number(result_duration, "duration_s")
    if duration_sec is not None:
        supplied_duration = _nonnegative_number(duration_sec, "duration_sec")
        if result_duration <= 0:
            result_duration = supplied_duration
        elif supplied_duration > 0:
            # The caller's probed duration is the authoritative bound.  Small
            # decoder rounding differences are expected, so use the minimum.
            result_duration = min(result_duration, supplied_duration)

    raw_timestamps = value.get("timestamps", value.get("timestamp", []))
    timestamps = _normalize_timestamps(raw_timestamps, result_duration)
    rtf = value.get("rtf")
    if rtf is None:
        rtf = value.get("real_time_factor")
    rtf = _optional_nonnegative_number(rtf, "rtf")

    normalized = {
        "schema_version": str(value.get("schema_version", SCHEMA_VERSION)),
        "text": text,
        "confidence": confidence,
        "duration_s": round(result_duration, ROUND_DIGITS),
        "rtf": rtf,
        "timestamps": timestamps,
        "timestamp_unit": str(
            value.get(
                "timestamp_unit",
                "chinese_character_or_english_word",
            )
        ),
    }
    # FireRed AED itself has no language-classification head; the colocated
    # FireRed LID observation supplies these fields.  Keep both aliases so the
    # speaker router has one stable key while downstream callers can inspect
    # the upstream spelling too.
    language_fields = _normalize_language_fields(value)
    normalized.update(language_fields)
    language_confidence = value.get("language_confidence")
    if language_confidence is None and "lang_confidence" in value:
        language_confidence = value.get("lang_confidence")
    if language_confidence is not None:
        normalized["language_confidence"] = _optional_probability(
            language_confidence
        )
    if "language_error" in value:
        language_error = value.get("language_error")
        if not isinstance(language_error, str) or not language_error.strip():
            raise FireRedAsrError(
                "FireRed ASR language_error must be a non-empty string"
            )
        normalized["language_error"] = language_error.strip()
    for key in ("language_duration_s", "language_dur_s"):
        if key in value:
            normalized["language_duration_s"] = _optional_nonnegative_number(
                value.get(key), key
            )
            break
    for key in ("language_rtf", "language_real_time_factor"):
        if key in value:
            normalized["language_rtf"] = _optional_nonnegative_number(
                value.get(key), key
            )
            break
    if "sentences" in value:
        normalized["sentences"] = copy.deepcopy(value["sentences"])
    if audio_path is not None:
        normalized["audio_path"] = str(Path(audio_path).expanduser().resolve())
    for key in ("model", "runtime"):
        if isinstance(value.get(key), dict):
            normalized[key] = copy.deepcopy(value[key])
    return normalized


def _normalize_timestamps(raw_timestamps, duration_sec):
    if raw_timestamps is None:
        raw_timestamps = []
    if not isinstance(raw_timestamps, (list, tuple)):
        raise FireRedAsrError("FireRed ASR timestamps must be a list")

    normalized = []
    previous_start = 0.0
    previous_end = 0.0
    for index, item in enumerate(raw_timestamps):
        if isinstance(item, dict):
            token = item.get("token", item.get("text", ""))
            start = item.get("start_s", item.get("start"))
            end = item.get("end_s", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            token, start, end = item[0], item[1], item[2]
        else:
            raise FireRedAsrError(
                "FireRed ASR timestamp %s must be an object or triple" % index
            )
        if token is None:
            token = ""
        if not isinstance(token, str):
            token = str(token)
        token = token.strip()
        start = _nonnegative_number(start, "timestamps[%s].start_s" % index)
        end = _nonnegative_number(end, "timestamps[%s].end_s" % index)
        if end <= start:
            raise FireRedAsrError(
                "FireRed ASR timestamp %s has non-positive duration" % index
            )
        if duration_sec > 0 and end > duration_sec + 1e-3:
            raise FireRedAsrError(
                "FireRed ASR timestamp %s exceeds audio duration" % index
            )
        if normalized and (start + 1e-6 < previous_start or end + 1e-6 < previous_end):
            raise FireRedAsrError("FireRed ASR timestamps are not monotonic")
        normalized.append(
            {
                "token": token,
                "start_s": round(start, ROUND_DIGITS),
                "end_s": round(end, ROUND_DIGITS),
            }
        )
        previous_start = start
        previous_end = end
    return normalized


def _normalize_language_fields(value):
    """Canonicalize ``lang``/``language`` while rejecting disagreements."""

    has_lang = "lang" in value
    has_language = "language" in value
    if not has_lang and not has_language:
        return {}

    lang = _language_string(value.get("lang"), "lang") if has_lang else None
    language = (
        _language_string(value.get("language"), "language")
        if has_language
        else None
    )
    if lang is not None and language is not None and lang != language:
        raise FireRedAsrError(
            "FireRed ASR lang and language fields disagree: %s != %s"
            % (lang, language)
        )
    canonical = lang or language
    return {"lang": canonical, "language": canonical}


def _language_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise FireRedAsrError(
            "FireRed ASR %s must be a non-empty string" % field_name
        )
    return value.strip()


def _normalize_lid_observation(value):
    """Validate one upstream FireRed LID result and expose stable aliases."""

    if not isinstance(value, dict):
        raise FireRedAsrError("FireRed LID result must be an object")
    fields = _normalize_language_fields(value)
    if not fields:
        raise FireRedAsrError("FireRed LID result has no language")

    confidence = value.get("language_confidence")
    if confidence is None and "lang_confidence" in value:
        confidence = value.get("lang_confidence")
    if confidence is None and "confidence" in value:
        confidence = value.get("confidence")

    observation = dict(fields)
    observation["language_confidence"] = _optional_probability(confidence)
    # Retain the upstream field names in the nested observation as well.  The
    # ASR result itself uses language_* names so its own confidence/duration
    # fields cannot be accidentally overwritten.
    observation["confidence"] = observation["language_confidence"]
    for source_key, target_key in (
        ("duration_s", "language_duration_s"),
        ("dur_s", "language_duration_s"),
    ):
        if source_key in value and value.get(source_key) is not None:
            observation[target_key] = _optional_nonnegative_number(
                value.get(source_key), source_key
            )
            observation["dur_s"] = observation[target_key]
            break
    for source_key, target_key in (
        ("rtf", "language_rtf"),
        ("real_time_factor", "language_rtf"),
    ):
        if source_key in value and value.get(source_key) is not None:
            observation[target_key] = _optional_nonnegative_number(
                value.get(source_key), source_key
            )
            observation["rtf"] = observation[target_key]
            break
    return observation


def _merge_lid_observation(result, observation):
    result["lang"] = observation["lang"]
    result["language"] = observation["language"]
    if observation.get("language_confidence") is not None:
        result["language_confidence"] = observation["language_confidence"]
    for key in ("language_duration_s", "language_rtf"):
        if key in observation:
            result[key] = observation[key]


def _format_runtime_error(exc):
    if isinstance(exc, FireRedAsrError):
        message = str(exc)
        return message or exc.__class__.__name__
    message = str(exc)
    if message:
        return "%s: %s" % (exc.__class__.__name__, message)
    return exc.__class__.__name__


def _optional_probability(value):
    if value is None:
        return None
    number = _finite_number(value, "confidence")
    if number < 0 or number > 1:
        raise FireRedAsrError("FireRed ASR confidence must be within [0, 1]")
    return round(number, ROUND_DIGITS)


def _optional_nonnegative_number(value, field_name):
    if value is None:
        return None
    return round(_nonnegative_number(value, field_name), ROUND_DIGITS)


def _nonnegative_number(value, field_name):
    number = _finite_number(value, field_name)
    if number < 0:
        raise FireRedAsrError("%s must be non-negative" % field_name)
    return number


def _finite_number(value, field_name):
    if isinstance(value, bool):
        raise FireRedAsrError("%s must be numeric" % field_name)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FireRedAsrError("%s must be numeric" % field_name) from exc
    if not math.isfinite(number):
        raise FireRedAsrError("%s must be finite" % field_name)
    return number


def _resolve_project_path(value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_model_files(model_dir):
    missing = [
        name
        for name in REQUIRED_MODEL_FILES
        if not (Path(model_dir) / name).is_file()
    ]
    if missing:
        raise FireRedAsrError(
            "incomplete FireRedASR2-AED model at %s; missing %s"
            % (model_dir, ", ".join(missing))
        )


def _validate_lid_model_files(model_dir, verify_checkpoint=True):
    missing = [
        name
        for name in REQUIRED_LID_MODEL_FILES
        if not (Path(model_dir) / name).is_file()
    ]
    if missing:
        raise FireRedAsrError(
            "incomplete FireRedLID model at %s; missing %s"
            % (model_dir, ", ".join(missing))
        )
    if not verify_checkpoint:
        return None
    checkpoint_path = Path(model_dir) / "model.pth.tar"
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256 != LID_CHECKPOINT_SHA256:
        raise FireRedAsrError(
            "FireRedLID checkpoint SHA256 mismatch at %s: expected %s, got %s"
            % (checkpoint_path, LID_CHECKPOINT_SHA256, actual_sha256)
        )
    return actual_sha256


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_adapter_path(config):
    if config.adapter_path:
        path = Path(config.adapter_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = _resolve_project_path(config.deployment_dir) / "fireredasr2_aed.py"
    if not path.is_file():
        raise FireRedAsrError(
            "FireRed deployment adapter is missing: %s" % path
        )
    return path.resolve()


def _load_deployment_module(adapter_path):
    digest = hashlib.sha1(str(adapter_path).encode("utf-8")).hexdigest()[:12]
    module_name = "_sure_tagger_firered_asr_deploy_%s" % digest
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, str(adapter_path))
    if spec is None or spec.loader is None:
        raise FireRedAsrError(
            "cannot import FireRed deployment adapter: %s" % adapter_path
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - normalize dependency errors.
        sys.modules.pop(module_name, None)
        raise FireRedAsrError(
            "FireRed deployment adapter import failed: %s" % adapter_path
        ) from exc
    return module


def _prepare_audio_for_subprocess(audio_path, config, context):
    try:
        return prepare_firered_audio(
            audio_path,
            context=context,
            normalize_to_16k_mono_pcm=config.normalize_to_16k_mono_pcm,
            error_class=FireRedAsrError,
            tool_label="FireRed ASR2-AED",
            temp_prefix="firered_asr2_aed_",
        )
    except FireRedAsrError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FireRedAsrError("failed to prepare FireRed ASR audio") from exc


def _subprocess_config(config):
    record = config.to_record()
    # The parent has already normalized source audio before sending a worker
    # request; this also prevents a child from invoking ffmpeg unexpectedly.
    record["subprocess_python"] = ""
    record["normalize_to_16k_mono_pcm"] = False
    record.pop("timeout_sec", None)
    return record


# Alias spellings make the adapter convenient for callers that use the model's
# all-caps acronym while preserving the naming convention of the existing
# speaker_v2 modules.
FireRedASRConfig = FireRedAsrConfig
FireRedASRClient = FireRedAsrClient
FireRedASRSubprocessClient = FireRedAsrSubprocessClient
FireRedASRError = FireRedAsrError


__all__ = [
    "CHECKPOINT_SHA256",
    "DEFAULT_DEPLOYMENT_DIR",
    "DEFAULT_LID_MODEL_DIR",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_SUBPROCESS_PYTHON",
    "LID_CHECKPOINT_SHA256",
    "REQUIRED_LID_MODEL_FILES",
    "FireRedASRClient",
    "FireRedASRError",
    "FireRedASRConfig",
    "FireRedASRSubprocessClient",
    "FireRedAsrClient",
    "FireRedAsrConfig",
    "FireRedAsrError",
    "FireRedAsrSubprocessClient",
    "normalize_result",
    "run",
    "transcribe",
    "validate_result",
]
