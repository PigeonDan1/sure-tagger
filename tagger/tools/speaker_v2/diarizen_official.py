"""Official DiariZen speaker diarization adapter.

The upstream pipeline is kept in a dedicated Python environment and loaded by
one long-lived JSONL subprocess per top-level GPU worker.  FireRed ASR is not
part of this adapter: an ASR failure and a diarization failure are independent
evidence outcomes.
"""

from __future__ import print_function

import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import time

from tagger.tools.subprocess_runner import run_subprocess_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOYMENT_ROOT = PROJECT_ROOT / "models" / "diarizen-official"
DEFAULT_MODEL_DIR = DEFAULT_DEPLOYMENT_ROOT / "model"
DEFAULT_EMBEDDING_MODEL = (
    DEFAULT_DEPLOYMENT_ROOT
    / "pyannote-wespeaker-voxceleb-resnet34-LM"
    / "pytorch_model.bin"
)
DEFAULT_SOURCE_DIR = DEFAULT_DEPLOYMENT_ROOT / "source"
DEFAULT_SUBPROCESS_PYTHON = (
    PROJECT_ROOT / ".runtime" / "diarizen_official_py310" / "bin" / "python"
)

MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
MODEL_REVISION = "f27b9ffbedcf422856d104ecee9b94be37ea578e"
MODEL_VERSION = "%s@%s" % (MODEL_ID, MODEL_REVISION)
MODEL_SHA256 = "689a10f0cbde6345b4bf9b1670cf3196488634b34e5b07bea0fe40d508e1196b"
MODEL_SIZE_BYTES = 278071526
CONFIG_SHA256 = "6eb3dab9c3892efc4a271ef5f88b5453e6e96789961084a15e88ca352de98503"
PLDA_SHA256 = "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255"
XVEC_TRANSFORM_SHA256 = (
    "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f"
)

WESPEAKER_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
WESPEAKER_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
WESPEAKER_VERSION = "%s@%s" % (WESPEAKER_MODEL_ID, WESPEAKER_REVISION)
WESPEAKER_SHA256 = (
    "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
)
WESPEAKER_SIZE_BYTES = 26645418

CODE_REPOSITORY = "github:BUTSpeechFIT/DiariZen"
CODE_REVISION = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
CODE_VERSION = "%s@%s" % (CODE_REPOSITORY, CODE_REVISION)
CODE_ARCHIVE_SHA256 = (
    "067793e5dba7aed2930031137bf537b833999e9f9c5abb5ce0ec083e8dfaddde"
)
SOURCE_NAME = "diarizen_large_s80_v2_official"
SOURCE_KIND = "official_diarizen_pipeline"
DEPENDENCY_GROUPS = (
    "G_diarizen_large_s80_v2_official",
    "G_pyannote_wespeaker_voxceleb_resnet34_lm",
    "G_diarizen_vbx_plda",
)
TOOL_VERSION = "diarizen_official_adapter.v1"
SCHEMA_VERSION = "diarizen.official.timeline.v1"
MODEL_SLOT_KEY = "diarizen_official_estimate"
MINIMUM_PIPELINE_AUDIO_SEC = 16.0
ROUND_DIGITS = 6


class OfficialDiariZenError(RuntimeError):
    """Raised when the official DiariZen pipeline cannot return a timeline."""


class OfficialDiariZenConfig:
    def __init__(
        self,
        model_dir=None,
        embedding_model=None,
        source_dir=None,
        subprocess_python=None,
        timeout_sec=1800,
        device="cuda:0",
        pad_short_audio=True,
    ):
        self.model_dir = str(model_dir or DEFAULT_MODEL_DIR)
        self.embedding_model = str(embedding_model or DEFAULT_EMBEDDING_MODEL)
        self.source_dir = str(source_dir or DEFAULT_SOURCE_DIR)
        self.subprocess_python = str(
            DEFAULT_SUBPROCESS_PYTHON
            if subprocess_python is None
            else subprocess_python or ""
        )
        self.timeout_sec = int(timeout_sec)
        self.device = str(device or "cuda:0")
        self.pad_short_audio = bool(pad_short_audio)
        if self.timeout_sec <= 0:
            raise ValueError("official DiariZen timeout_sec must be positive")
        if self.device not in ("auto", "cuda", "cuda:0", "cpu"):
            raise ValueError("unsupported official DiariZen device: %s" % self.device)

    def cache_key(self):
        return (
            self.model_dir,
            self.embedding_model,
            self.source_dir,
            self.device,
            self.pad_short_audio,
        )

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "embedding_model": self.embedding_model,
            "source_dir": self.source_dir,
            "subprocess_python": self.subprocess_python,
            "timeout_sec": self.timeout_sec,
            "device": self.device,
            "pad_short_audio": self.pad_short_audio,
        }


class OfficialDiariZenClient:
    """In-process client used inside the dedicated model worker."""

    def __init__(self, config=None):
        self.config = config or OfficialDiariZenConfig(subprocess_python="")
        self.model_dir = _required_directory(self.config.model_dir, "model directory")
        self.embedding_model = _required_file(
            self.config.embedding_model, "WeSpeaker checkpoint"
        )
        self.source_dir = _required_directory(self.config.source_dir, "source directory")
        _required_file(self.model_dir / "config.toml", "model config")
        _required_file(self.model_dir / "pytorch_model.bin", "model checkpoint")
        _required_file(self.model_dir / "plda" / "plda.npz", "PLDA model")
        _required_file(
            self.model_dir / "plda" / "xvec_transform.npz", "x-vector transform"
        )
        self._torch = None
        self._torchaudio = None
        self._pipeline = None
        self._model_load_sec = None

    def diarize(self, audio_path, context=None):
        del context
        audio_path = _required_file(audio_path, "audio")
        self._load_pipeline()
        prepared_path, cleanup, original_duration, padded = self._prepare_audio(
            audio_path
        )
        started = time.time()
        empty_timeline_reason = None
        try:
            annotation = self._pipeline(
                str(prepared_path), sess_name=audio_path.stem
            )
            if self._torch.cuda.is_available():
                self._torch.cuda.synchronize()
            segments = annotation_to_segments(annotation, original_duration)
        except Exception as exc:
            if _is_no_active_speaker_error(exc):
                # Upstream pyannote reconstruct computes num_clusters=-1 when
                # every hard cluster is the inactive sentinel (-2). This is a
                # valid empty diarization outcome, not an adapter failure.
                segments = []
                empty_timeline_reason = "no_active_speaker_clusters"
            else:
                raise OfficialDiariZenError(
                    "official DiariZen inference failed: %s" % _format_error(exc)
                ) from exc
        finally:
            if cleanup is not None:
                cleanup.cleanup()
        elapsed = time.time() - started
        return {
            "schema_version": SCHEMA_VERSION,
            "segments": segments,
            "runtime": {
                "device": str(self._pipeline.device),
                "torch_version": self._torch.__version__,
                "torchaudio_version": self._torchaudio.__version__,
                "model_load_sec": round(self._model_load_sec, ROUND_DIGITS),
                "inference_sec": round(elapsed, ROUND_DIGITS),
                "input_audio_path": str(audio_path),
                "prepared_audio_path": str(prepared_path),
                "short_audio_padded": padded,
                "original_duration_sec": round(original_duration, ROUND_DIGITS),
                "model_version": MODEL_VERSION,
                "embedding_version": WESPEAKER_VERSION,
                "code_version": CODE_VERSION,
                "config_sha256": CONFIG_SHA256,
                "empty_timeline_reason": empty_timeline_reason,
            },
        }

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        source_path = str(self.source_dir)
        pyannote_path = str(self.source_dir / "pyannote-audio")
        for path in (source_path, pyannote_path):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import torch
            import torchaudio
            from diarizen.pipelines.inference import DiariZenPipeline
        except Exception as exc:
            raise OfficialDiariZenError(
                "official DiariZen dependencies are not importable: %s"
                % _format_error(exc)
            ) from exc
        wants_cuda = self.config.device != "cpu"
        if wants_cuda and not torch.cuda.is_available():
            raise OfficialDiariZenError("CUDA was requested but is unavailable")
        if self.config.device == "cpu" and torch.cuda.is_available():
            raise OfficialDiariZenError(
                "upstream DiariZen auto-selects CUDA; CPU requires CUDA to be hidden"
            )
        started = time.time()
        try:
            self._pipeline = DiariZenPipeline(
                diarizen_hub=self.model_dir,
                embedding_model=str(self.embedding_model),
            )
        except Exception as exc:
            raise OfficialDiariZenError(
                "official DiariZen model loading failed: %s" % _format_error(exc)
            ) from exc
        self._torch = torch
        self._torchaudio = torchaudio
        self._model_load_sec = time.time() - started

    def _prepare_audio(self, audio_path):
        try:
            waveform, sample_rate = self._torchaudio.load(str(audio_path))
        except Exception as exc:
            raise OfficialDiariZenError(
                "failed to decode audio: %s" % _format_error(exc)
            ) from exc
        if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
            raise OfficialDiariZenError("decoded audio is empty")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise OfficialDiariZenError("decoded audio has invalid sample rate")
        original_duration = float(waveform.shape[1]) / float(sample_rate)
        if original_duration >= MINIMUM_PIPELINE_AUDIO_SEC:
            return audio_path, None, original_duration, False
        if not self.config.pad_short_audio:
            raise OfficialDiariZenError(
                "audio is shorter than the upstream 16 second window"
            )
        target_frames = int(math.ceil(MINIMUM_PIPELINE_AUDIO_SEC * sample_rate))
        pad_frames = max(0, target_frames - int(waveform.shape[1]))
        padded = self._torch.nn.functional.pad(waveform, (0, pad_frames))
        cleanup = tempfile.TemporaryDirectory(prefix="diarizen_official_")
        prepared_path = Path(cleanup.name) / "padded.wav"
        try:
            self._torchaudio.save(str(prepared_path), padded, sample_rate)
        except Exception:
            cleanup.cleanup()
            raise
        return prepared_path, cleanup, original_duration, True


class OfficialDiariZenSubprocessClient:
    """Parent-side client for the long-lived official DiariZen worker."""

    def __init__(self, config=None):
        self.config = config or OfficialDiariZenConfig()

    def diarize(self, audio_path, context=None):
        audio_path = _required_file(audio_path, "audio")
        try:
            result = run_subprocess_tool(
                self.config.subprocess_python,
                MODEL_SLOT_KEY,
                {
                    "audio_path": str(audio_path),
                    "config": _subprocess_config(self.config),
                },
                context=context,
                timeout_sec=self.config.timeout_sec,
            )
        except Exception as exc:
            raise OfficialDiariZenError(
                "official DiariZen subprocess failed: %s" % _format_error(exc)
            ) from exc
        if isinstance(result, dict) and "output" in result:
            result = result["output"]
        return normalize_result(result)


def diarize(audio_path, config=None, context=None, client=None):
    config = config or OfficialDiariZenConfig()
    if client is None:
        client = (
            OfficialDiariZenSubprocessClient(config)
            if config.subprocess_python
            else OfficialDiariZenClient(config)
        )
    return client.diarize(audio_path, context=context)


def normalize_result(result):
    if not isinstance(result, dict):
        raise OfficialDiariZenError("official DiariZen returned a non-object result")
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise OfficialDiariZenError("official DiariZen result has no segment list")
    normalized = copy.deepcopy(result)
    normalized["segments"] = _validate_segments(segments)
    runtime = normalized.get("runtime", {})
    if not isinstance(runtime, dict):
        raise OfficialDiariZenError("official DiariZen runtime must be an object")
    normalized["runtime"] = runtime
    return normalized


def annotation_to_segments(annotation, duration_sec):
    if not hasattr(annotation, "itertracks"):
        raise OfficialDiariZenError("upstream result is not a pyannote Annotation")
    duration_sec = float(duration_sec)
    raw = []
    for turn, track, label in annotation.itertracks(yield_label=True):
        start = max(0.0, min(duration_sec, float(turn.start)))
        end = max(0.0, min(duration_sec, float(turn.end)))
        if end <= start:
            continue
        raw.append((start, end, str(label), str(track)))
    raw.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    speaker_ids = {}
    for _start, _end, label, _track in raw:
        if label not in speaker_ids:
            speaker_ids[label] = "speaker_%d" % (len(speaker_ids) + 1)
    segments = []
    for index, (start, end, label, _track) in enumerate(raw):
        overlap = any(
            other_label != label
            and min(end, other_end) > max(start, other_start)
            for other_index, (other_start, other_end, other_label, _other_track)
            in enumerate(raw)
            if other_index != index
        )
        segments.append(
            {
                "start_sec": round(start, ROUND_DIGITS),
                "end_sec": round(end, ROUND_DIGITS),
                "speaker_id": speaker_ids[label],
                "overlap": bool(overlap),
            }
        )
    return _validate_segments(segments)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_segments(segments):
    normalized = []
    previous_key = None
    for item in segments:
        if not isinstance(item, dict):
            raise OfficialDiariZenError("timeline segment must be an object")
        start = item.get("start_sec")
        end = item.get("end_sec")
        speaker_id = item.get("speaker_id")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise OfficialDiariZenError("segment start must be numeric")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            raise OfficialDiariZenError("segment end must be numeric")
        if not math.isfinite(float(start)) or not math.isfinite(float(end)):
            raise OfficialDiariZenError("segment bounds must be finite")
        if float(end) <= float(start):
            raise OfficialDiariZenError("segment must satisfy start < end")
        if not isinstance(speaker_id, str) or not speaker_id:
            raise OfficialDiariZenError("segment speaker_id must be non-empty")
        record = {
            "start_sec": round(float(start), ROUND_DIGITS),
            "end_sec": round(float(end), ROUND_DIGITS),
            "speaker_id": speaker_id,
            "overlap": bool(item.get("overlap", False)),
        }
        key = (record["start_sec"], record["end_sec"], record["speaker_id"])
        if previous_key is not None and key < previous_key:
            raise OfficialDiariZenError("timeline segments must be sorted")
        normalized.append(record)
        previous_key = key
    return normalized


def _subprocess_config(config):
    record = config.to_record()
    record["subprocess_python"] = ""
    return record


def _required_file(path, label):
    path = _resolve_project_path(path)
    if not path.is_file():
        raise OfficialDiariZenError("%s is missing: %s" % (label, path))
    return path.resolve()


def _required_directory(path, label):
    path = _resolve_project_path(path)
    if not path.is_dir():
        raise OfficialDiariZenError("%s is missing: %s" % (label, path))
    return path.resolve()


def _resolve_project_path(path):
    path = Path(str(path)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _format_error(exc):
    message = str(exc).strip()
    return "%s%s" % (
        exc.__class__.__name__,
        ": %s" % message if message else "",
    )


def _is_no_active_speaker_error(exc):
    return isinstance(exc, ValueError) and str(exc).strip() == (
        "negative dimensions are not allowed"
    )
