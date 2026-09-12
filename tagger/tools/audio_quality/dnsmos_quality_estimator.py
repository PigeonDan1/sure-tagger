"""DNSMOS P.835 and P.808 no-reference speech quality estimator.

The inference windowing, mel features, and score calibration follow Microsoft's
``DNSMOS/dnsmos_local.py`` implementation. Tool failures are surfaced to the
pipeline so public DNSMOS tags remain null; no proxy metric is substituted.
"""

import math
from numbers import Real
from pathlib import Path

from tagger.local_config import (
    DNSMOS_MODEL_VERSION,
    DNSMOS_P808_MODEL_PATH,
    DNSMOS_PERSONALIZED_MODEL_PATH,
    DNSMOS_PRIMARY_MODEL_PATH,
    DNSMOS_PYTHON,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "dnsmos_quality_estimator"
METHOD = "DNSMOS P.835/P.808"
SAMPLING_RATE_HZ = 16000
INPUT_LENGTH_SEC = 9.01
INPUT_LENGTH_SAMPLES = int(INPUT_LENGTH_SEC * SAMPLING_RATE_HZ)
HOP_LENGTH_SAMPLES = SAMPLING_RATE_HZ
ROUND_DIGITS = 6
MIN_EXPECTED_MODEL_BYTES = 100 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_FIELDS = (
    ("sig", "audio_quality.dnsmos_sig"),
    ("bak", "audio_quality.dnsmos_bak"),
    ("ovrl", "audio_quality.dnsmos_ovrl"),
    ("p808", "audio_quality.dnsmos_p808"),
)


class DnsmosError(RuntimeError):
    """Raised when DNSMOS cannot produce valid quality scores."""


class DnsmosConfig:
    """Fixed local DNSMOS configuration used by the tagging pipeline."""

    def __init__(
        self,
        primary_model_path=None,
        p808_model_path=None,
        personalized_model_path=None,
        personalized=False,
        model_version=None,
        subprocess_python=None,
        use_gpu=False,
    ):
        configured_python = getattr(DNSMOS_PYTHON, "strip", lambda: "")()
        self.primary_model_path = _resolve_path(
            primary_model_path or DNSMOS_PRIMARY_MODEL_PATH
        )
        self.p808_model_path = _resolve_path(
            p808_model_path or DNSMOS_P808_MODEL_PATH
        )
        self.personalized_model_path = _resolve_path(
            personalized_model_path or DNSMOS_PERSONALIZED_MODEL_PATH
        )
        self.personalized = bool(personalized)
        self.model_version = model_version or DNSMOS_MODEL_VERSION
        self.use_gpu = bool(use_gpu)
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    @property
    def selected_primary_model_path(self):
        if self.personalized:
            return self.personalized_model_path
        return self.primary_model_path

    def cache_key(self):
        return (
            self.selected_primary_model_path,
            self.p808_model_path,
            self.personalized,
            self.model_version,
            self.subprocess_python,
            self.use_gpu,
        )

    def to_record(self):
        return {
            "primary_model_path": self.selected_primary_model_path,
            "p808_model_path": self.p808_model_path,
            "personalized": self.personalized,
            "model_version": self.model_version,
            "use_gpu": self.use_gpu,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "input_length_sec": INPUT_LENGTH_SEC,
            "subprocess_python": self.subprocess_python,
            "output_field_mapping": dict(OUTPUT_FIELDS),
        }


class DnsmosClient:
    """Adapter around the ONNX models shipped in Microsoft's repository."""

    def __init__(self, config=None):
        self.config = config or DnsmosConfig()
        self._primary_session = None
        self._p808_session = None

    def estimate(self, audio_path, context=None):
        del context
        np, librosa, sf = _load_audio_dependencies()
        primary_session, p808_session = self._get_sessions()

        try:
            audio, input_sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
            raise DnsmosError("DNSMOS could not read the audio file") from exc

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise DnsmosError("DNSMOS audio must contain at least one mono frame")
        if not np.all(np.isfinite(audio)):
            raise DnsmosError("DNSMOS audio contains NaN or Inf")

        if int(input_sample_rate) != SAMPLING_RATE_HZ:
            try:
                audio = librosa.resample(
                    audio,
                    orig_sr=int(input_sample_rate),
                    target_sr=SAMPLING_RATE_HZ,
                )
            except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
                raise DnsmosError("DNSMOS audio resampling failed") from exc
            audio = np.asarray(audio, dtype=np.float32)

        original_length = int(audio.size)
        if audio.size < INPUT_LENGTH_SAMPLES:
            while audio.size < INPUT_LENGTH_SAMPLES:
                audio = np.concatenate((audio, audio))

        num_hops = int(math.floor(audio.size / SAMPLING_RATE_HZ - INPUT_LENGTH_SEC)) + 1
        if num_hops <= 0:
            raise DnsmosError("DNSMOS could not construct an inference window")

        sig_scores = []
        bak_scores = []
        ovrl_scores = []
        p808_scores = []
        for index in range(num_hops):
            start = index * HOP_LENGTH_SAMPLES
            segment = audio[start : start + INPUT_LENGTH_SAMPLES]
            if segment.size != INPUT_LENGTH_SAMPLES:
                continue

            primary_input = segment.astype(np.float32, copy=False)[None, :]
            mel_input = _audio_melspec(segment[:-160], np, librosa)[None, :, :]
            try:
                primary_output = primary_session.run(
                    None, {"input_1": primary_input}
                )[0][0]
                p808_output = p808_session.run(None, {"input_1": mel_input})[0][0][0]
            except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
                raise DnsmosError("DNSMOS ONNX inference failed") from exc

            if len(primary_output) != 3:
                raise DnsmosError("DNSMOS primary model returned an invalid shape")
            raw_sig, raw_bak, raw_ovrl = [float(value) for value in primary_output]
            sig, bak, ovrl = _calibrate_scores(
                raw_sig,
                raw_bak,
                raw_ovrl,
                personalized=self.config.personalized,
            )
            sig_scores.append(sig)
            bak_scores.append(bak)
            ovrl_scores.append(ovrl)
            p808_scores.append(float(p808_output))

        if not sig_scores:
            raise DnsmosError("DNSMOS produced no complete inference windows")

        output = {
            "sig": _finite_mean(sig_scores, "SIG"),
            "bak": _finite_mean(bak_scores, "BAK"),
            "ovrl": _finite_mean(ovrl_scores, "OVRL"),
            "p808": _finite_mean(p808_scores, "P808"),
            "num_hops": len(sig_scores),
            "audio_length_sec": original_length / float(SAMPLING_RATE_HZ),
        }
        return output

    def _get_sessions(self):
        if self._primary_session is not None and self._p808_session is not None:
            return self._primary_session, self._p808_session

        primary_path = Path(self.config.selected_primary_model_path)
        p808_path = Path(self.config.p808_model_path)
        _validate_model_file(primary_path, "primary")
        _validate_model_file(p808_path, "P.808")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DnsmosError(
                "onnxruntime is not importable in the DNSMOS Python environment"
            ) from exc

        if self.config.use_gpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise DnsmosError(
                    "DNSMOS GPU mode requested but CUDAExecutionProvider is unavailable; "
                    "refusing CPU fallback (available: %s)" % available
                )
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            self._primary_session = ort.InferenceSession(
                str(primary_path), providers=providers
            )
            self._p808_session = ort.InferenceSession(
                str(p808_path), providers=providers
            )
            if self.config.use_gpu:
                for name, session in (
                    ("primary", self._primary_session),
                    ("P.808", self._p808_session),
                ):
                    active = tuple(session.get_providers())
                    if "CUDAExecutionProvider" not in active:
                        raise DnsmosError(
                            "DNSMOS %s session did not activate CUDAExecutionProvider: %s"
                            % (name, active)
                        )
        except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
            raise DnsmosError("DNSMOS model loading failed") from exc
        return self._primary_session, self._p808_session


class DnsmosSubprocessClient:
    """Adapter that runs DNSMOS in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or DnsmosConfig()

    def estimate(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "dnsmos_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    config = config or DnsmosConfig()
    client = client or _default_client(config)
    output = client.estimate(audio_path, context=context)
    results = []
    for source_field, tag_path in OUTPUT_FIELDS:
        results.append(_build_result(tag_path, source_field, output, config))
    return results


def _default_client(config):
    if config.subprocess_python:
        return DnsmosSubprocessClient(config)
    return DnsmosClient(config)


def _subprocess_config(config):
    return {
        "primary_model_path": config.primary_model_path,
        "p808_model_path": config.p808_model_path,
        "personalized_model_path": config.personalized_model_path,
        "personalized": config.personalized,
        "model_version": config.model_version,
        "use_gpu": config.use_gpu,
        "subprocess_python": "",
    }


def _build_result(tag_path, source_field, output, config):
    value, error = _extract_score(output, source_field)
    evidence = {
        "dnsmos_config": config.to_record(),
        "source_field": source_field,
        "num_hops": output.get("num_hops"),
        "audio_length_sec": output.get("audio_length_sec"),
    }
    if error is not None:
        evidence["error"] = error
    return ToolResult(
        tag_path=tag_path,
        value=value,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        status="estimated" if error is None else "failed",
        confidence=1.0 if error is None else 0.0,
        tool_type="model_inference",
        evidence=evidence,
    )


def _extract_score(output, source_field):
    if source_field not in output:
        return None, "DNSMOS output is missing field: %s" % source_field
    value = output[source_field]
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, "DNSMOS field %s is not numeric" % source_field
    value = float(value)
    if not math.isfinite(value):
        return None, "DNSMOS field %s contains NaN or Inf" % source_field
    if value < 1.0 or value > 5.0:
        return None, "DNSMOS field %s is outside the MOS range [1, 5]" % source_field
    return round(value, ROUND_DIGITS), None


def _load_audio_dependencies():
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise DnsmosError(
            "DNSMOS dependencies are not importable; install librosa, numpy, "
            "soundfile, and onnxruntime"
        ) from exc
    return np, librosa, sf


def _audio_melspec(audio, np, librosa):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLING_RATE_HZ,
        n_fft=321,
        hop_length=160,
        n_mels=120,
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40.0) / 40.0
    return np.asarray(mel_spec.T, dtype=np.float32)


def _calibrate_scores(sig, bak, ovrl, personalized):
    if personalized:
        sig_coefficients = (-0.01019296, 0.02751166, 1.19576786, -0.24348726)
        bak_coefficients = (-0.04976499, 0.44276479, -0.1644611, 0.96883132)
        ovrl_coefficients = (-0.00533021, 0.005101, 1.18058466, -0.11236046)
    else:
        sig_coefficients = (-0.08397278, 1.22083953, 0.0052439)
        bak_coefficients = (-0.13166888, 1.60915514, -0.39604546)
        ovrl_coefficients = (-0.06766283, 1.11546468, 0.04602535)
    return (
        _polyval(sig_coefficients, sig),
        _polyval(bak_coefficients, bak),
        _polyval(ovrl_coefficients, ovrl),
    )


def _polyval(coefficients, value):
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return float(result)


def _finite_mean(values, field):
    if not values or any(not math.isfinite(value) for value in values):
        raise DnsmosError("DNSMOS %s output contains NaN or Inf" % field)
    return float(sum(values)) / float(len(values))


def _validate_model_file(path, label):
    if not path.exists() or not path.is_file():
        raise DnsmosError("DNSMOS %s model is missing: %s" % (label, path))
    if path.stat().st_size < MIN_EXPECTED_MODEL_BYTES:
        raise DnsmosError(
            "DNSMOS %s model is too small or incomplete: %s" % (label, path)
        )
    with path.open("rb") as source:
        header = source.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise DnsmosError("DNSMOS %s model is a Git LFS pointer: %s" % (label, path))


def _resolve_path(configured_path):
    if not configured_path:
        return ""
    path = Path(str(configured_path)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)
