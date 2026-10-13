"""DASS (Distilled Audio State-space Models) background-noise type detector.

The registered tool maps DASS AudioSet clip probabilities to the public
``sound_field_scene.external_noise_type`` classification list: one key per
present docs/DASS.md category (④ 音乐, ② 动物, ③ 机械, ⑤ 自然, ⑥ 形态,
⑦ 声道/环境 — i.e. ``music``/``animal``/``mechanical``/``nature``/
``formless``/``channel_environment``), derived from the full 527-class
sigmoid vector at or above the configured threshold and ordered by each
category's best score. ① 人类声音 and unclassified labels stay in internal
evidence only. By default primary speech, silence, acoustic-scene,
reverberation, and echo labels are excluded and never drive a category
(``Silence`` would otherwise flag clean-speech samples as ``formless``);
the exclusion is all-or-nothing: ``exclude_classes=False``
(``--no-exclusion``) keeps every AudioSet class eligible so the raw class
distribution stays visible. Scores remain internal evidence.

The DASS id2label order matches the pinned PANNs AudioSet CSV, so the same
exclusion policy applies. The upstream checkpoint was deployed by
sure-harness (saurabhati/DASS_medium_AudioSet_48.9, 49M params, AudioSet-2M
mAP 48.9) and is reused from the project-local ``models/DASS`` copy.

The concrete labels per category are published separately in
``sound_field_scene.noise_composition``: the full 527-class sigmoid vector
is bucketed per docs/DASS.md into music/animal/mechanical/nature/formless/
channel-environment labels (top-k at or above a separate composition
threshold), with the music bucket gated by FireRed AED ``music_present``.

Inference is GPU-first: the default configuration requests CUDA and fails
explicitly when CUDA is unavailable.  Pass ``use_gpu=False`` (or the CLI
``--dass-use-cpu`` option) for an intentional CPU run.  The CUDA device is
the process-local logical device 0; ``CUDA_VISIBLE_DEVICES`` can be used by
the caller to select which physical GPU that logical device represents.
"""

from pathlib import Path

from tagger.local_config import (
    DASS_MODEL_DIR,
    DASS_MODEL_VERSION,
    DASS_PYTHON,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.sound_field_scene.dass_categories import (
    PUBLIC_COMPOSITION_CATEGORIES,
    build_category_composition,
    classify_dass_label,
)
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "dass_noise_type_detector"
METHOD = "DASS medium AudioSet-2M audio tagging (mAP 48.9)"
# Calibrated on the phase2 sample set (2026-08-24): DASS-medium sigmoid
# scores for real noise classes are soft (0.1-0.45), while clean speech stays
# below 0.15. 0.25 recovers real noise labels without false positives on
# clean speech; pass --dass-threshold to override.
DEFAULT_THRESHOLD = 0.25
# Aligned with DEFAULT_THRESHOLD (2026-08-25): both knobs were calibrated on
# the phase2 sample set (0.25 for categories, 0.30 for composition), but the
# gap produced "category present, composition empty" rows (e.g. CHiME4/TUT
# mechanical labels in the 0.25-0.30 band). A single default guarantees every
# present category has a non-empty composition bucket.
COMPOSITION_DEFAULT_THRESHOLD = 0.25
COMPOSITION_DEFAULT_TOP_K = 3
SAMPLE_RATE_HZ = 16000
CLASSES_NUM = 527
TOP_EVENTS_LIMIT = 10
ROUND_DIGITS = 6
# The upstream DASS feature extractor hardcodes frame hop 160 at 16 kHz.
FRAME_HOP_SAMPLES = 160

# Default exclusion policy: these classes are not background-noise types.
# Chatter, speech babble, singing, and other human non-speech events remain
# eligible because they can be background interference in an ASR sample.
DEFAULT_EXCLUDED_DISPLAY_NAMES = frozenset(
    [
        # Primary speech.
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
        "Speech synthesizer",
        # Not sound events.
        "Silence",
        # Acoustic scenes and room effects.
        "Inside, small room",
        "Inside, large room or hall",
        "Inside, public space",
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Reverberation",
        "Echo",
    ]
)


def effective_excluded_display_names(exclude_classes=True):
    """Return the excluded AudioSet display names for a policy flag.

    The exclusion is all-or-nothing: ``exclude_classes=False`` disables every
    exclusion so the raw DASS class distribution stays visible.
    """
    if exclude_classes:
        return DEFAULT_EXCLUDED_DISPLAY_NAMES
    return frozenset()


class DassNoiseTypeError(RuntimeError):
    """Raised when DASS cannot produce a valid noise-type result."""


class DassNoiseTypeConfig:
    """Fixed DASS inference and public mapping configuration.

    DASS is GPU-first by default.  ``use_gpu=False`` is an explicit CPU
    opt-out; a requested GPU is never silently downgraded when CUDA is not
    available.
    """

    def __init__(
        self,
        model_dir=None,
        model_version=DASS_MODEL_VERSION,
        use_gpu=True,
        threshold=DEFAULT_THRESHOLD,
        exclude_classes=True,
        composition_threshold=COMPOSITION_DEFAULT_THRESHOLD,
        composition_top_k=COMPOSITION_DEFAULT_TOP_K,
        subprocess_python=None,
    ):
        configured_model = getattr(DASS_MODEL_DIR, "strip", lambda: "")()
        configured_python = getattr(DASS_PYTHON, "strip", lambda: "")()
        self.model_dir = str(model_dir or configured_model)
        self.model_version = str(model_version)
        self.use_gpu = bool(use_gpu)
        self.threshold = _require_probability(threshold, "threshold")
        self.exclude_classes = bool(exclude_classes)
        self.composition_threshold = _require_probability(
            composition_threshold, "composition_threshold"
        )
        self.composition_top_k = _require_positive_int(
            composition_top_k, "composition_top_k"
        )
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    def cache_key(self):
        return (
            self.model_dir,
            self.model_version,
            self.use_gpu,
            self.threshold,
            self.exclude_classes,
            self.composition_threshold,
            self.composition_top_k,
            self.subprocess_python,
        )

    def to_record(self):
        record = _subprocess_config(self)
        record.update(
            {
                "threshold": self.threshold,
                "exclude_classes": self.exclude_classes,
                "excluded_display_names": sorted(
                    effective_excluded_display_names(self.exclude_classes)
                ),
                "public_mapping": (
                    "docs/DASS.md category keys driven by eligible labels "
                    "at or above threshold, ordered by per-category best "
                    "score"
                ),
                "composition_threshold": self.composition_threshold,
                "composition_top_k": self.composition_top_k,
                "composition_mapping": (
                    "category-bucketed top-k labels per docs/DASS.md, music "
                    "gated by FireRed AED music_present"
                ),
                "subprocess_python": self.subprocess_python,
            }
        )
        return record


class DassNoiseTypeClient:
    """Adapter around the local DASS AudioSet classification checkpoint."""

    def __init__(self, config=None):
        self.config = config or DassNoiseTypeConfig()
        self._model = None
        self._feature_extractor = None
        self._device = None
        self._labels = None
        self._window_samples = None

    def estimate(self, audio_path, context=None):
        model, feature_extractor, device, labels, window_samples = self._get_runtime(
            context
        )
        try:
            import librosa
            import numpy as np
            import torch
        except ImportError as exc:
            raise DassNoiseTypeError(
                "DASS runtime requires librosa, numpy, and torch"
            ) from exc

        try:
            audio, _sample_rate = librosa.load(
                str(audio_path),
                sr=SAMPLE_RATE_HZ,
                mono=True,
                dtype="float32",
            )
        except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
            raise DassNoiseTypeError("DASS audio loading failed") from exc

        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise DassNoiseTypeError("DASS audio must be a non-empty mono waveform")
        if not np.isfinite(audio).all():
            raise DassNoiseTypeError("DASS audio contains non-finite samples")

        max_scores = None
        chunk_count = 0
        for offset in range(0, int(audio.size), window_samples):
            chunk = audio[offset : offset + window_samples]
            if chunk.size < window_samples:
                chunk = np.pad(chunk, (0, window_samples - chunk.size), mode="constant")
            inputs = feature_extractor(
                chunk,
                sampling_rate=SAMPLE_RATE_HZ,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            try:
                with torch.no_grad():
                    logits = torch.sigmoid(model(**inputs).logits)[0]
            except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
                raise DassNoiseTypeError("DASS inference failed") from exc
            scores = logits.detach().cpu().numpy()
            if scores.shape != (CLASSES_NUM,):
                raise DassNoiseTypeError(
                    "DASS output must have shape (%s,)" % CLASSES_NUM
                )
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise DassNoiseTypeError(
                    "DASS output scores must be finite probabilities"
                )
            max_scores = scores if max_scores is None else np.maximum(max_scores, scores)
            chunk_count += 1

        summary = select_noise_type_events(
            labels, max_scores.tolist(), exclude_classes=self.config.exclude_classes
        )
        summary["chunk_count"] = chunk_count
        # Full vector for the categorized evidence layer (composition is not
        # affected by the exclusion policy).
        summary["labels"] = [dict(label) for label in labels]
        summary["scores"] = [round(score, ROUND_DIGITS) for score in max_scores.tolist()]
        return summary

    def _get_runtime(self, context=None):
        if context is None:
            if self._model is None:
                (
                    self._model,
                    self._feature_extractor,
                    self._device,
                    self._labels,
                    self._window_samples,
                ) = self._load_runtime()
            return (
                self._model,
                self._feature_extractor,
                self._device,
                self._labels,
                self._window_samples,
            )

        cache = context.setdefault("dass_noise_type_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_runtime()
        return cache[key]

    def _load_runtime(self):
        model_dir = Path(self.config.model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        preprocessor_path = model_dir / "preprocessor_config.json"
        missing = [
            str(path)
            for path in (config_path, weights_path, preprocessor_path)
            if not path.is_file()
        ]
        if missing:
            raise DassNoiseTypeError(
                "DASS checkpoint files are missing: %s" % ", ".join(missing)
            )

        try:
            import torch
        except ImportError as exc:
            raise DassNoiseTypeError("torch is not importable in DASS runtime") from exc
        if self.config.use_gpu:
            if not torch.cuda.is_available():
                raise DassNoiseTypeError(
                    "DASS GPU was requested but CUDA is unavailable; "
                    "set CUDA_VISIBLE_DEVICES to a usable GPU or opt out "
                    "with use_gpu=False"
                )
            # ``cuda`` means logical device 0 and intentionally respects the
            # process-level CUDA_VISIBLE_DEVICES mapping.
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as exc:
            raise DassNoiseTypeError(
                "transformers is not importable in DASS runtime"
            ) from exc

        try:
            model = (
                AutoModelForAudioClassification.from_pretrained(
                    str(model_dir), trust_remote_code=True
                )
                .to(device)
                .eval()
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
        except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
            raise DassNoiseTypeError("DASS checkpoint loading failed") from exc

        labels = _load_id2label(model.config)
        max_length = _require_positive_int(
            getattr(feature_extractor, "max_length", None), "max_length"
        )
        window_samples = max_length * FRAME_HOP_SAMPLES
        return model, feature_extractor, device, labels, window_samples


class DassNoiseTypeSubprocessClient:
    """Adapter that runs DASS in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or DassNoiseTypeConfig()

    def estimate(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "dass_noise_type_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(
    audio_path,
    context=None,
    config=None,
    client=None,
    music_present=None,
    **_kwargs,
):
    config = config or DassNoiseTypeConfig()
    client = client or _default_client(config)
    raw_output = client.estimate(audio_path, context)
    summary = validate_dass_output(
        raw_output, exclude_classes=config.exclude_classes
    )
    noise_labels = detected_noise_categories(
        summary, config.threshold, exclude_classes=config.exclude_classes
    )
    composition, category_events = build_category_composition(
        summary["labels"],
        summary["scores"],
        threshold=config.composition_threshold,
        top_k=config.composition_top_k,
        music_present=music_present,
    )
    evidence = {
        "config": config.to_record(),
        "chunk_count": summary["chunk_count"],
        "max_noise_score": summary["max_noise_score"],
        "winning_event": summary["winning_event"],
        "top_noise_events": summary["top_noise_events"],
        "category_events": category_events,
        "composition": dict(composition),
        "music_gate": {
            "aed_music_present": music_present,
            "gated": music_present is False,
        },
    }
    return [
        ToolResult(
            tag_path="sound_field_scene.external_noise_type",
            value=noise_labels,
            tool_name=TOOL_NAME,
            method=METHOD,
            status="estimated",
            confidence=1.0,
            tool_type="model",
            tool_version=TOOL_VERSION,
            evidence=evidence,
        ),
        ToolResult(
            tag_path="sound_field_scene.noise_composition",
            value=composition,
            tool_name=TOOL_NAME,
            method=METHOD,
            status="estimated",
            confidence=1.0,
            tool_type="model",
            tool_version=TOOL_VERSION,
            evidence=evidence,
        ),
    ]


def detected_noise_categories(summary, threshold, exclude_classes=True):
    """Return the public docs/DASS.md categories present in the full vector.

    The category set mirrors ``noise_composition``'s public keys: a category
    is present when any of its labels in the full 527-class vector reaches
    ``threshold``. The exclusion policy applies here exactly as for the
    ranked top events — primary speech, silence, acoustic-scene,
    reverberation, and echo labels never drive a category (``Silence`` would
    otherwise flag clean-speech samples as ``formless``); pass
    ``exclude_classes=False`` to let them in. Human and unclassified labels
    never surface. Categories are ordered by their best label score,
    highest first (ties resolved by the public category order).
    """
    threshold = _require_probability(threshold, "threshold")
    excluded = effective_excluded_display_names(exclude_classes)
    full_labels = summary["labels"]
    full_scores = summary["scores"]
    if not isinstance(full_labels, list) or not isinstance(full_scores, list):
        raise DassNoiseTypeError("DASS summary must include full labels and scores")
    if len(full_labels) != len(full_scores):
        raise DassNoiseTypeError("DASS full labels and scores must have equal length")
    best = {category: -1.0 for category in PUBLIC_COMPOSITION_CATEGORIES}
    seen_names = set()
    for label, raw_score in zip(full_labels, full_scores):
        if not isinstance(label, dict):
            raise DassNoiseTypeError("DASS full label must be an object")
        display_name = label.get("display_name")
        if not isinstance(display_name, str) or not display_name:
            raise DassNoiseTypeError("DASS full display names must be non-empty strings")
        if display_name in seen_names:
            raise DassNoiseTypeError("DASS full display names must be unique")
        seen_names.add(display_name)
        if display_name in excluded:
            continue
        category = classify_dass_label(display_name)
        if category not in best:
            continue
        score = _require_probability(raw_score, "scores[%s]" % label.get("index"))
        if score > best[category]:
            best[category] = score
    return [
        category
        for category in sorted(
            best,
            key=lambda category: (
                -best[category],
                PUBLIC_COMPOSITION_CATEGORIES.index(category),
            ),
        )
        if best[category] >= threshold
    ]


def detected_noise_type_labels(summary, threshold):
    """Return the unique eligible AudioSet display names in the ranked top
    events at or above the configured threshold.

    Retained as a validation/convenience helper for the summary payload; the
    public ``external_noise_type`` value now carries category keys produced
    by :func:`detected_noise_categories` instead.
    """
    threshold = _require_probability(threshold, "threshold")
    labels = []
    for event in summary["top_noise_events"]:
        if event["score"] < threshold:
            break
        display_name = event["display_name"]
        if display_name not in labels:
            labels.append(display_name)
    return labels


def _load_id2label(config):
    id2label = getattr(config, "id2label", None)
    if not isinstance(id2label, dict) or not id2label:
        raise DassNoiseTypeError("DASS config id2label must be a non-empty object")
    labels = []
    for index in range(CLASSES_NUM):
        # transformers converts string keys to int keys when the config is
        # instantiated; accept both.
        display_name = id2label.get(str(index))
        if display_name is None:
            display_name = id2label.get(index)
        if not isinstance(display_name, str) or not display_name:
            raise DassNoiseTypeError(
                "DASS id2label must cover indexes 0 through %s" % (CLASSES_NUM - 1)
            )
        labels.append(
            {
                "index": index,
                "display_name": display_name,
            }
        )
    return labels


def select_noise_type_events(
    labels, scores, top_k=TOP_EVENTS_LIMIT, exclude_classes=True
):
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise DassNoiseTypeError("DASS labels and scores must be lists")
    if len(labels) != len(scores) or not labels:
        raise DassNoiseTypeError("DASS labels and scores must have equal length")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise DassNoiseTypeError("top_k must be a positive integer")
    excluded = effective_excluded_display_names(exclude_classes)

    eligible = []
    seen_names = set()
    for position, (label, raw_score) in enumerate(zip(labels, scores)):
        if not isinstance(label, dict):
            raise DassNoiseTypeError("DASS label must be an object")
        index = label.get("index")
        display_name = label.get("display_name")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise DassNoiseTypeError("DASS label indexes must match score order")
        if not isinstance(display_name, str) or not display_name:
            raise DassNoiseTypeError("DASS display names must be non-empty strings")
        if display_name in seen_names:
            raise DassNoiseTypeError("DASS display names must be unique")
        seen_names.add(display_name)
        score = _require_probability(raw_score, "scores[%s]" % position)
        if display_name not in excluded:
            eligible.append(
                {
                    "index": index,
                    "display_name": display_name,
                    "score": round(score, ROUND_DIGITS),
                }
            )
    if not eligible:
        raise DassNoiseTypeError("DASS output has no eligible noise classes")

    eligible.sort(key=lambda item: (-item["score"], item["index"]))
    top_events = eligible[:top_k]
    return {
        "max_noise_score": top_events[0]["score"],
        "winning_event": dict(top_events[0]),
        "top_noise_events": top_events,
    }


def validate_dass_output(raw_output, exclude_classes=True):
    if not isinstance(raw_output, dict):
        raise DassNoiseTypeError("DASS output must be an object")
    chunk_count = raw_output.get("chunk_count")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
    ):
        raise DassNoiseTypeError("DASS chunk_count must be a positive integer")

    max_score = _require_probability(
        raw_output.get("max_noise_score"), "max_noise_score"
    )
    winning_event = _validate_event(
        raw_output.get("winning_event"), "winning_event", exclude_classes
    )
    top_events_raw = raw_output.get("top_noise_events")
    if not isinstance(top_events_raw, list) or not top_events_raw:
        raise DassNoiseTypeError("DASS top_noise_events must be non-empty")
    if len(top_events_raw) > TOP_EVENTS_LIMIT:
        raise DassNoiseTypeError("DASS returned too many top noise events")
    top_events = [
        _validate_event(item, "top_noise_events[%s]" % index, exclude_classes)
        for index, item in enumerate(top_events_raw)
    ]
    if any(
        top_events[index]["score"] < top_events[index + 1]["score"]
        for index in range(len(top_events) - 1)
    ):
        raise DassNoiseTypeError("DASS top noise events must be score-sorted")
    if len(set(item["index"] for item in top_events)) != len(top_events):
        raise DassNoiseTypeError("DASS top noise event indexes must be unique")
    if winning_event != top_events[0] or abs(max_score - winning_event["score"]) > 1e-6:
        raise DassNoiseTypeError("DASS winning event does not match maximum score")

    full_labels = raw_output.get("labels")
    full_scores = raw_output.get("scores")
    if not isinstance(full_labels, list) or not isinstance(full_scores, list):
        raise DassNoiseTypeError("DASS output must include full labels and scores")
    if len(full_labels) != CLASSES_NUM or len(full_scores) != CLASSES_NUM:
        raise DassNoiseTypeError(
            "DASS full labels and scores must have length %s" % CLASSES_NUM
        )
    seen_full_names = set()
    validated_labels = []
    validated_scores = []
    for position, (label, raw_score) in enumerate(
        zip(full_labels, full_scores)
    ):
        if not isinstance(label, dict):
            raise DassNoiseTypeError("DASS full label must be an object")
        index = label.get("index")
        display_name = label.get("display_name")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise DassNoiseTypeError("DASS full label indexes must match score order")
        if not isinstance(display_name, str) or not display_name:
            raise DassNoiseTypeError("DASS full display names must be non-empty strings")
        if display_name in seen_full_names:
            raise DassNoiseTypeError("DASS full display names must be unique")
        seen_full_names.add(display_name)
        score = _require_probability(
            raw_score, "scores[%s]" % position
        )
        validated_labels.append({"index": index, "display_name": display_name})
        validated_scores.append(round(score, ROUND_DIGITS))
    return {
        "chunk_count": chunk_count,
        "max_noise_score": round(max_score, ROUND_DIGITS),
        "winning_event": winning_event,
        "top_noise_events": top_events,
        "labels": validated_labels,
        "scores": validated_scores,
    }


def _validate_event(raw_event, path, exclude_classes=True):
    excluded = effective_excluded_display_names(exclude_classes)
    if not isinstance(raw_event, dict) or set(raw_event.keys()) != set(
        ["index", "display_name", "score"]
    ):
        raise DassNoiseTypeError("%s must be a complete event object" % path)
    index = raw_event["index"]
    display_name = raw_event["display_name"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise DassNoiseTypeError("%s.index must be a non-negative integer" % path)
    if not isinstance(display_name, str) or not display_name:
        raise DassNoiseTypeError("%s.display_name must be non-empty" % path)
    if display_name in excluded:
        raise DassNoiseTypeError("%s.display_name must be an eligible AudioSet label" % path)
    return {
        "index": index,
        "display_name": display_name,
        "score": round(
            _require_probability(raw_event["score"], path + ".score"),
            ROUND_DIGITS,
        ),
    }


def _default_client(config):
    if config.subprocess_python:
        return DassNoiseTypeSubprocessClient(config)
    return DassNoiseTypeClient(config)


def _subprocess_config(config):
    return {
        "model_dir": config.model_dir,
        "model_version": config.model_version,
        "use_gpu": config.use_gpu,
        "threshold": config.threshold,
        "exclude_classes": config.exclude_classes,
        "composition_threshold": config.composition_threshold,
        "composition_top_k": config.composition_top_k,
        "subprocess_python": "",
    }


def _require_probability(value, path):
    value = _require_number(value, path)
    if value < 0 or value > 1:
        raise DassNoiseTypeError("%s must be within [0, 1]" % path)
    return value


def _require_positive_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DassNoiseTypeError("%s must be a positive integer" % path)
    return value


def _require_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DassNoiseTypeError("%s must be a number" % path)
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise DassNoiseTypeError("%s must be finite" % path)
    return value
