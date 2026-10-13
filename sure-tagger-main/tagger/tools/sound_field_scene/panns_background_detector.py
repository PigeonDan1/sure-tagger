"""PANNs Cnn14 background-sound detector.

The registered tool maps AudioSet clip probabilities to the existing public
``sound_field_scene.sound`` classification list. Primary speech, silence, and
acoustic-scene labels are excluded; music and other non-speech events count as
background sound. Scores remain internal evidence.
"""

import csv
from pathlib import Path
import sys

from tagger.local_config import (
    PANNS_CHECKPOINT_PATH,
    PANNS_MODEL_VERSION,
    PANNS_PYTHON,
    PANNS_REPO_DIR,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "panns_background_detector"
METHOD = "PANNs Cnn14 AudioSet audio tagging"
DEFAULT_THRESHOLD = 0.30
SAMPLE_RATE_HZ = 32000
CHUNK_DURATION_SEC = 10.0
CLASSES_NUM = 527
TOP_EVENTS_LIMIT = 10
ROUND_DIGITS = 6

# Primary speech is not background sound. Chatter, speech babble, singing, and
# other human non-speech events remain eligible because they can be background
# interference in an ASR sample.
EXCLUDED_MIDS = frozenset(
    [
        "/m/09x0r",  # Speech
        "/m/05zppz",  # Male speech
        "/m/02zsn",  # Female speech
        "/m/0ytgt",  # Child speech
        "/m/01h8n0",  # Conversation
        "/m/02qldy",  # Narration, monologue
        "/m/0261r1",  # Babbling
        "/m/0brhx",  # Speech synthesizer
        "/m/028v0c",  # Silence
        "/t/dd00125",  # Inside, small room
        "/t/dd00126",  # Inside, large room or hall
        "/t/dd00127",  # Inside, public space
        "/t/dd00128",  # Outside, urban or manmade
        "/t/dd00129",  # Outside, rural or natural
        "/m/01b9nn",  # Reverberation
        "/m/01jnbd",  # Echo
    ]
)


class PannsBackgroundError(RuntimeError):
    """Raised when PANNs cannot produce a valid background-sound result."""


class PannsBackgroundConfig:
    """Fixed PANNs Cnn14 inference and public mapping configuration."""

    def __init__(
        self,
        repo_dir=None,
        checkpoint_path=None,
        model_version=PANNS_MODEL_VERSION,
        use_gpu=False,
        threshold=DEFAULT_THRESHOLD,
        sample_rate_hz=SAMPLE_RATE_HZ,
        chunk_duration_sec=CHUNK_DURATION_SEC,
        subprocess_python=None,
    ):
        configured_repo = getattr(PANNS_REPO_DIR, "strip", lambda: "")()
        configured_checkpoint = getattr(
            PANNS_CHECKPOINT_PATH, "strip", lambda: ""
        )()
        configured_python = getattr(PANNS_PYTHON, "strip", lambda: "")()
        self.repo_dir = str(repo_dir or configured_repo)
        self.checkpoint_path = str(checkpoint_path or configured_checkpoint)
        self.model_version = str(model_version)
        self.use_gpu = bool(use_gpu)
        self.threshold = _require_probability(threshold, "threshold")
        self.sample_rate_hz = _require_positive_int(
            sample_rate_hz, "sample_rate_hz"
        )
        self.chunk_duration_sec = _require_positive_number(
            chunk_duration_sec, "chunk_duration_sec"
        )
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    def cache_key(self):
        return (
            self.repo_dir,
            self.checkpoint_path,
            self.model_version,
            self.use_gpu,
            self.threshold,
            self.sample_rate_hz,
            self.chunk_duration_sec,
            self.subprocess_python,
        )

    def to_record(self):
        record = _subprocess_config(self)
        record.update(
            {
                "threshold": self.threshold,
                "excluded_mids": sorted(EXCLUDED_MIDS),
                "public_mapping": (
                    "score-ranked eligible AudioSet display names at or above threshold"
                ),
                "subprocess_python": self.subprocess_python,
            }
        )
        return record


class PannsBackgroundClient:
    """Adapter around the pinned upstream Cnn14 implementation."""

    def __init__(self, config=None):
        self.config = config or PannsBackgroundConfig()
        self._model = None
        self._device = None
        self._labels = None

    def estimate(self, audio_path, context=None):
        model, device, labels = self._get_runtime(context)
        try:
            import librosa
            import numpy as np
            import torch
        except ImportError as exc:
            raise PannsBackgroundError(
                "PANNs runtime requires librosa, numpy, torch, and torchlibrosa"
            ) from exc

        try:
            audio, _sample_rate = librosa.load(
                str(audio_path),
                sr=self.config.sample_rate_hz,
                mono=True,
                dtype="float32",
            )
        except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
            raise PannsBackgroundError("PANNs audio loading failed") from exc

        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise PannsBackgroundError("PANNs audio must be a non-empty mono waveform")
        if not np.isfinite(audio).all():
            raise PannsBackgroundError("PANNs audio contains non-finite samples")

        chunk_samples = int(
            round(self.config.sample_rate_hz * self.config.chunk_duration_sec)
        )
        if chunk_samples <= 0:
            raise PannsBackgroundError("PANNs chunk size must be positive")

        max_scores = None
        chunk_count = 0
        for offset in range(0, int(audio.size), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            if chunk.size < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - chunk.size), mode="constant")
            waveform = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).unsqueeze(0)
            waveform = waveform.to(device)
            try:
                with torch.no_grad():
                    output = model(waveform, None)["clipwise_output"]
            except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
                raise PannsBackgroundError("PANNs Cnn14 inference failed") from exc
            scores = output.detach().cpu().numpy()
            if scores.shape != (1, CLASSES_NUM):
                raise PannsBackgroundError(
                    "PANNs output must have shape (1, %s)" % CLASSES_NUM
                )
            scores = scores[0]
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise PannsBackgroundError(
                    "PANNs output scores must be finite probabilities"
                )
            max_scores = scores if max_scores is None else np.maximum(max_scores, scores)
            chunk_count += 1

        summary = select_background_events(labels, max_scores.tolist())
        summary["chunk_count"] = chunk_count
        return summary

    def _get_runtime(self, context=None):
        if context is None:
            if self._model is None:
                self._model, self._device, self._labels = self._load_runtime()
            return self._model, self._device, self._labels

        cache = context.setdefault("panns_background_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_runtime()
        return cache[key]

    def _load_runtime(self):
        repo_dir = Path(self.config.repo_dir)
        checkpoint_path = Path(self.config.checkpoint_path)
        models_path = repo_dir / "pytorch" / "models.py"
        labels_path = repo_dir / "metadata" / "class_labels_indices.csv"
        missing = [
            str(path)
            for path in (models_path, labels_path, checkpoint_path)
            if not path.is_file()
        ]
        if missing:
            raise PannsBackgroundError(
                "PANNs files are missing: %s" % ", ".join(missing)
            )

        try:
            import torch
        except ImportError as exc:
            raise PannsBackgroundError("torch is not importable in PANNs runtime") from exc
        if self.config.use_gpu:
            if not torch.cuda.is_available():
                raise PannsBackgroundError("PANNs GPU was requested but CUDA is unavailable")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        pytorch_dir = str((repo_dir / "pytorch").resolve())
        if pytorch_dir not in sys.path:
            sys.path.insert(0, pytorch_dir)
        try:
            from models import Cnn14
        except ImportError as exc:
            raise PannsBackgroundError(
                "pinned audioset_tagging_cnn source is not importable"
            ) from exc

        labels = load_audioset_labels(labels_path)
        model = Cnn14(
            sample_rate=self.config.sample_rate_hz,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=CLASSES_NUM,
        )
        try:
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=True
            )
            state_dict = checkpoint["model"]
            model.load_state_dict(state_dict)
        except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
            raise PannsBackgroundError("PANNs checkpoint is invalid") from exc
        model.to(device)
        model.eval()
        return model, device, labels


class PannsBackgroundSubprocessClient:
    """Adapter that runs PANNs in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or PannsBackgroundConfig()

    def estimate(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "panns_background_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    config = config or PannsBackgroundConfig()
    client = client or _default_client(config)
    raw_output = client.estimate(audio_path, context)
    summary = validate_panns_output(raw_output)
    value = detected_background_labels(summary, config.threshold)
    evidence = {
        "config": config.to_record(),
        "chunk_count": summary["chunk_count"],
        "max_background_score": summary["max_background_score"],
        "winning_event": summary["winning_event"],
        "top_background_events": summary["top_background_events"],
    }
    return ToolResult(
        tag_path="sound_field_scene.sound",
        value=value,
        tool_name=TOOL_NAME,
        method=METHOD,
        status="estimated",
        confidence=1.0,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence=evidence,
    )


def detected_background_labels(summary, threshold):
    """Return unique public labels at or above the configured threshold."""
    threshold = _require_probability(threshold, "threshold")
    labels = []
    for event in summary["top_background_events"]:
        if event["score"] < threshold:
            break
        display_name = event["display_name"]
        if display_name not in labels:
            labels.append(display_name)
    return labels


def load_audioset_labels(labels_path):
    labels = []
    with Path(labels_path).open("r", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["index", "mid", "display_name"]:
            raise PannsBackgroundError("AudioSet labels CSV has an invalid header")
        for row in reader:
            try:
                index = int(row["index"])
            except (TypeError, ValueError) as exc:
                raise PannsBackgroundError("AudioSet label index is invalid") from exc
            labels.append(
                {
                    "index": index,
                    "mid": row["mid"],
                    "display_name": row["display_name"],
                }
            )
    if len(labels) != CLASSES_NUM or [item["index"] for item in labels] != list(
        range(CLASSES_NUM)
    ):
        raise PannsBackgroundError(
            "AudioSet labels must contain indexes 0 through %s" % (CLASSES_NUM - 1)
        )
    if len(set(item["mid"] for item in labels)) != CLASSES_NUM:
        raise PannsBackgroundError("AudioSet label mids must be unique")
    return labels


def select_background_events(labels, scores, top_k=TOP_EVENTS_LIMIT):
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise PannsBackgroundError("PANNs labels and scores must be lists")
    if len(labels) != len(scores) or not labels:
        raise PannsBackgroundError("PANNs labels and scores must have equal length")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise PannsBackgroundError("top_k must be a positive integer")

    eligible = []
    seen_mids = set()
    for position, (label, raw_score) in enumerate(zip(labels, scores)):
        if not isinstance(label, dict):
            raise PannsBackgroundError("PANNs label must be an object")
        index = label.get("index")
        mid = label.get("mid")
        display_name = label.get("display_name")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise PannsBackgroundError("PANNs label indexes must match score order")
        if not isinstance(mid, str) or not mid or mid in seen_mids:
            raise PannsBackgroundError("PANNs label mids must be non-empty and unique")
        if not isinstance(display_name, str) or not display_name:
            raise PannsBackgroundError("PANNs display names must be non-empty strings")
        seen_mids.add(mid)
        score = _require_probability(raw_score, "scores[%s]" % position)
        if mid not in EXCLUDED_MIDS:
            eligible.append(
                {
                    "index": index,
                    "mid": mid,
                    "display_name": display_name,
                    "score": round(score, ROUND_DIGITS),
                }
            )
    if not eligible:
        raise PannsBackgroundError("PANNs output has no eligible background classes")

    eligible.sort(key=lambda item: (-item["score"], item["index"]))
    top_events = eligible[:top_k]
    return {
        "max_background_score": top_events[0]["score"],
        "winning_event": dict(top_events[0]),
        "top_background_events": top_events,
    }


def validate_panns_output(raw_output):
    if not isinstance(raw_output, dict):
        raise PannsBackgroundError("PANNs output must be an object")
    chunk_count = raw_output.get("chunk_count")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
    ):
        raise PannsBackgroundError("PANNs chunk_count must be a positive integer")

    max_score = _require_probability(
        raw_output.get("max_background_score"), "max_background_score"
    )
    winning_event = _validate_event(raw_output.get("winning_event"), "winning_event")
    top_events_raw = raw_output.get("top_background_events")
    if not isinstance(top_events_raw, list) or not top_events_raw:
        raise PannsBackgroundError("PANNs top_background_events must be non-empty")
    if len(top_events_raw) > TOP_EVENTS_LIMIT:
        raise PannsBackgroundError("PANNs returned too many top background events")
    top_events = [
        _validate_event(item, "top_background_events[%s]" % index)
        for index, item in enumerate(top_events_raw)
    ]
    if any(
        top_events[index]["score"] < top_events[index + 1]["score"]
        for index in range(len(top_events) - 1)
    ):
        raise PannsBackgroundError("PANNs top background events must be score-sorted")
    if len(set(item["mid"] for item in top_events)) != len(top_events):
        raise PannsBackgroundError("PANNs top background event mids must be unique")
    if winning_event != top_events[0] or abs(max_score - winning_event["score"]) > 1e-6:
        raise PannsBackgroundError("PANNs winning event does not match maximum score")
    return {
        "chunk_count": chunk_count,
        "max_background_score": round(max_score, ROUND_DIGITS),
        "winning_event": winning_event,
        "top_background_events": top_events,
    }


def _validate_event(raw_event, path):
    if not isinstance(raw_event, dict) or set(raw_event.keys()) != set(
        ["index", "mid", "display_name", "score"]
    ):
        raise PannsBackgroundError("%s must be a complete event object" % path)
    index = raw_event["index"]
    mid = raw_event["mid"]
    display_name = raw_event["display_name"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PannsBackgroundError("%s.index must be a non-negative integer" % path)
    if not isinstance(mid, str) or not mid or mid in EXCLUDED_MIDS:
        raise PannsBackgroundError("%s.mid must be an eligible AudioSet id" % path)
    if not isinstance(display_name, str) or not display_name:
        raise PannsBackgroundError("%s.display_name must be non-empty" % path)
    return {
        "index": index,
        "mid": mid,
        "display_name": display_name,
        "score": round(
            _require_probability(raw_event["score"], path + ".score"),
            ROUND_DIGITS,
        ),
    }


def _default_client(config):
    if config.subprocess_python:
        return PannsBackgroundSubprocessClient(config)
    return PannsBackgroundClient(config)


def _subprocess_config(config):
    return {
        "repo_dir": config.repo_dir,
        "checkpoint_path": config.checkpoint_path,
        "model_version": config.model_version,
        "use_gpu": config.use_gpu,
        "threshold": config.threshold,
        "sample_rate_hz": config.sample_rate_hz,
        "chunk_duration_sec": config.chunk_duration_sec,
        "subprocess_python": "",
    }


def _require_probability(value, path):
    value = _require_number(value, path)
    if value < 0 or value > 1:
        raise PannsBackgroundError("%s must be within [0, 1]" % path)
    return value


def _require_positive_number(value, path):
    value = _require_number(value, path)
    if value <= 0:
        raise PannsBackgroundError("%s must be positive" % path)
    return value


def _require_positive_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PannsBackgroundError("%s must be a positive integer" % path)
    return value


def _require_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PannsBackgroundError("%s must be a number" % path)
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise PannsBackgroundError("%s must be finite" % path)
    return value
