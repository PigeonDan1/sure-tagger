"""Brouhaha SNR and C50 estimator.

This module intentionally contains no alternative SNR or C50 estimator. If the
Brouhaha dependency, checkpoint, or inference output is unavailable, callers
must leave the corresponding tags as null instead of substituting rules,
metadata, or another model.

The public tag is ``audio_quality.snr_db``. Brouhaha C50 is only published as
internal evidence (``internal.brouhaha_c50_db``) for cross-validation with the
Rec-RIR C50 (``room_acoustic.c50_db``); it never enters the tags-only output.
"""

from numbers import Real
from pathlib import Path
import math
import sys

from tagger.local_config import (
    BROUHAHA_MODEL_PATH,
    BROUHAHA_MODEL_VERSION,
    BROUHAHA_PYTHON,
    BROUHAHA_REPO_DIR,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "brouhaha_signal_estimator"
METHOD = "Brouhaha"
MODEL_NAME = "Brouhaha"
ROUND_DIGITS = 6
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGGREGATION = "mean"
MIN_EXPECTED_CHECKPOINT_BYTES = 1024 * 1024


class BrouhahaError(RuntimeError):
    """Raised when Brouhaha cannot produce valid SNR/C50 results."""


class BrouhahaConfig:
    """Fixed Brouhaha configuration used by the tagging pipeline."""

    def __init__(
        self,
        model_path=None,
        repo_dir=None,
        use_gpu=False,
        aggregation=DEFAULT_AGGREGATION,
        model_version=None,
        subprocess_python=None,
    ):
        configured_model_path = getattr(BROUHAHA_MODEL_PATH, "strip", lambda: "")()
        configured_repo_dir = getattr(BROUHAHA_REPO_DIR, "strip", lambda: "")()
        configured_python = getattr(BROUHAHA_PYTHON, "strip", lambda: "")()
        self.model_path = _resolve_model_path(model_path or configured_model_path)
        self.repo_dir = _resolve_repo_dir(repo_dir or configured_repo_dir)
        self.use_gpu = bool(use_gpu)
        self.aggregation = aggregation
        self.model_version = model_version or BROUHAHA_MODEL_VERSION
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )
        if self.aggregation != DEFAULT_AGGREGATION:
            raise BrouhahaError("Brouhaha aggregation must be fixed to mean")

    def cache_key(self):
        return (
            self.model_path,
            self.repo_dir,
            self.use_gpu,
            self.aggregation,
            self.model_version,
            self.subprocess_python,
        )

    def to_record(self):
        return {
            "model": MODEL_NAME,
            "model_path": self.model_path,
            "repo_dir": self.repo_dir,
            "model_version": self.model_version,
            "use_gpu": self.use_gpu,
            "aggregation": self.aggregation,
            "subprocess_python": self.subprocess_python,
            "output_field_mapping": {
                "snr": "audio_quality.snr_db",
                "c50": "internal.brouhaha_c50_db",
            },
        }


class BrouhahaClient:
    """Thin adapter around the official Brouhaha Python API."""

    def __init__(self, config=None):
        self.config = config or BrouhahaConfig()
        self._pipeline = None

    def estimate(self, audio_path, context=None):
        pipeline = self._get_pipeline(context)
        file_record = {
            "uri": Path(audio_path).stem,
            "audio": str(audio_path),
        }
        try:
            output = pipeline(file_record)
        except Exception as exc:  # noqa: BLE001 - converted to internal warning upstream.
            raise BrouhahaError("Brouhaha inference failed") from exc
        if not isinstance(output, dict):
            raise BrouhahaError("Brouhaha returned a non-object output")
        return output

    def _get_pipeline(self, context=None):
        if context is None:
            if self._pipeline is None:
                self._pipeline = self._load_pipeline()
            return self._pipeline

        cache = context.setdefault("brouhaha_pipeline_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_pipeline()
        return cache[key]

    def _load_pipeline(self):
        self._validate_model_path()
        self._add_repo_dir_to_path()
        try:
            import torch
            from pyannote.audio import Model
            from brouhaha.pipeline import RegressiveActivityDetectionPipeline
        except ImportError as exc:
            raise BrouhahaError(
                "Brouhaha dependencies are not importable; install the official "
                "brouhaha package and pyannote.audio first"
            ) from exc

        try:
            model = Model.from_pretrained(Path(self.config.model_path), strict=False)
            device = torch.device(
                "cuda"
                if self.config.use_gpu and torch.cuda.is_available()
                else "cpu"
            )
            model.to(device)
            pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
            if hasattr(pipeline, "to"):
                pipeline.to(device)
            if hasattr(pipeline, "instantiate"):
                pipeline.instantiate(pipeline.default_parameters())
        except Exception as exc:  # noqa: BLE001 - model loading failures are internal.
            raise BrouhahaError("Brouhaha model loading failed") from exc

        return pipeline

    def _validate_model_path(self):
        if not self.config.model_path:
            raise BrouhahaError(
                "Brouhaha model path is not configured; set "
                "tagger/local_config.py:BROUHAHA_MODEL_PATH"
            )

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise BrouhahaError("Brouhaha checkpoint is missing: %s" % model_path)
        if not model_path.is_file():
            raise BrouhahaError("Brouhaha checkpoint is not a file: %s" % model_path)
        if model_path.stat().st_size < MIN_EXPECTED_CHECKPOINT_BYTES:
            raise BrouhahaError(
                "Brouhaha checkpoint is too small or incomplete: %s" % model_path
            )

        with model_path.open("rb") as handle:
            header = handle.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise BrouhahaError(
                "Brouhaha checkpoint is a Git LFS pointer, not model weights: %s"
                % model_path
            )

    def _add_repo_dir_to_path(self):
        if not self.config.repo_dir:
            return
        repo_dir = Path(self.config.repo_dir)
        if repo_dir.exists() and str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))


class BrouhahaSubprocessClient:
    """Adapter that runs Brouhaha in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or BrouhahaConfig()

    def estimate(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "brouhaha_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    config = config or BrouhahaConfig()
    client = client or _default_client(config)
    output = client.estimate(audio_path, context=context)

    return [
        _build_result(
            tag_path="audio_quality.snr_db",
            source_field="snr",
            output=output,
            config=config,
        ),
        _build_result(
            tag_path="internal.brouhaha_c50_db",
            source_field="c50",
            output=output,
            config=config,
        ),
    ]


def _default_client(config):
    if config.subprocess_python:
        return BrouhahaSubprocessClient(config)
    return BrouhahaClient(config)


def _subprocess_config(config):
    return {
        "model_path": config.model_path,
        "repo_dir": config.repo_dir,
        "use_gpu": config.use_gpu,
        "aggregation": config.aggregation,
        "model_version": config.model_version,
        "subprocess_python": "",
    }


def _build_result(tag_path, source_field, output, config):
    value, error = _extract_scalar(output, source_field)
    evidence = {
        "brouhaha_config": config.to_record(),
        "source_field": source_field,
        "aggregation": config.aggregation,
    }
    status = "estimated"
    if error is not None:
        status = "failed"
        evidence["error"] = error

    return ToolResult(
        tag_path=tag_path,
        value=value,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        status=status,
        confidence=1.0 if error is None else 0.0,
        tool_type="model_inference",
        evidence=evidence,
    )


def _extract_scalar(output, source_field):
    if source_field not in output:
        return None, "Brouhaha output is missing field: %s" % source_field
    try:
        value = _mean_finite_numbers(output[source_field], source_field)
    except BrouhahaError as exc:
        return None, str(exc)
    return round(value, ROUND_DIGITS), None


def _mean_finite_numbers(raw_value, source_field):
    values = list(_iter_numeric_values(_to_python_value(raw_value), source_field))
    if not values:
        raise BrouhahaError("Brouhaha field %s has no numeric values" % source_field)
    for value in values:
        if not math.isfinite(value):
            raise BrouhahaError(
                "Brouhaha field %s contains NaN or Inf" % source_field
            )
    return float(sum(values)) / float(len(values))


def _iter_numeric_values(value, source_field):
    if isinstance(value, (list, tuple)):
        for item in value:
            for number in _iter_numeric_values(
                _to_python_value(item),
                source_field,
            ):
                yield number
        return

    if isinstance(value, bool) or not isinstance(value, Real):
        raise BrouhahaError(
            "Brouhaha field %s contains a non-numeric value" % source_field
        )
    yield float(value)


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


def _resolve_model_path(configured_path):
    if not configured_path:
        return ""

    base_path = _project_relative_path(configured_path)
    candidates = [base_path]
    if base_path.is_dir():
        candidates.extend(
            [
                base_path / "best.ckpt",
                base_path / "models" / "best" / "checkpoints" / "best.ckpt",
                base_path
                / "brouhaha-vad"
                / "models"
                / "best"
                / "checkpoints"
                / "best.ckpt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _resolve_repo_dir(configured_path):
    if not configured_path:
        return ""
    return str(_project_relative_path(configured_path))


def _project_relative_path(configured_path):
    path = Path(str(configured_path)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
