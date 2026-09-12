"""SpeechBrain ECAPA speaker identity adapter for predicted timeline regions.

The adapter deliberately accepts only sample-local audio and regions selected
from an upstream *predicted* timeline.  Native metadata, reference annotations,
and trial labels are rejected before model inference.
"""

import math
from pathlib import Path
import threading

from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_VERSION = "ecapa_identity_v2.0-shadow.1"
MODEL_ID = "speechbrain_ecapa_voxceleb"
MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
MODEL_VERSION = "speechbrain/spkrec-ecapa-voxceleb@%s" % MODEL_REVISION
CHECKPOINT_SHA256 = (
    "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
)
HYPERPARAMS_SHA256 = (
    "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41"
)

# Frozen on the dev split of the 1k AMI utterance atomic identity benchmark.
# This profile must not be described as a predicted-region production
# calibration until the end-to-end shadow gate has passed.
DEFAULT_THRESHOLD = 0.34445778
DEFAULT_CALIBRATION_PROFILE_ID = "ami_utterance_1k_v1_ecapa_atomic_dev_platt_v1"
DEFAULT_PLATT_A = 2.542341658976709
DEFAULT_PLATT_B = -3.9259496365892312
DEFAULT_PLATT_SCORE_MEAN = 0.11357820589536304
DEFAULT_PLATT_SCORE_SCALE = 0.15337247889292588

_MODEL_INIT_LOCK = threading.Lock()
_FORBIDDEN_EXACT_FIELDS = {
    "annotation",
    "annotations",
    "global_speaker_id",
    "gold",
    "gold_same",
    "is_target",
    "label",
    "local_speaker_id",
    "native_metadata",
    "reference_transcript",
    "speaker",
    "target",
}
_FORBIDDEN_FIELD_PREFIXES = ("gold_", "ground_truth", "reference_")
_REGION_FIELDS = (
    "region_id",
    "speaker_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "selection",
    "source_segment_index",
    "overlap_exclusion_applied",
)


class EcapaIdentityError(RuntimeError):
    """Raised when ECAPA identity inference cannot produce valid output."""


class EcapaIdentityConfig:
    def __init__(
        self,
        model_dir,
        subprocess_python="",
        device="cpu",
        threshold=DEFAULT_THRESHOLD,
        min_region_duration_sec=0.80,
        max_regions_per_speaker=2,
        torch_num_threads=4,
        timeout_sec=300,
        runtime_cache_dir="",
        calibration_profile_id=DEFAULT_CALIBRATION_PROFILE_ID,
        platt_a=DEFAULT_PLATT_A,
        platt_b=DEFAULT_PLATT_B,
        platt_score_mean=DEFAULT_PLATT_SCORE_MEAN,
        platt_score_scale=DEFAULT_PLATT_SCORE_SCALE,
    ):
        self.model_dir = str(model_dir)
        self.subprocess_python = str(subprocess_python or "")
        self.device = str(device or "cpu")
        self.threshold = float(threshold)
        self.min_region_duration_sec = float(min_region_duration_sec)
        self.max_regions_per_speaker = int(max_regions_per_speaker)
        self.torch_num_threads = int(torch_num_threads)
        self.timeout_sec = int(timeout_sec)
        self.runtime_cache_dir = str(runtime_cache_dir or "")
        self.calibration_profile_id = str(calibration_profile_id or "")
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        self.platt_score_mean = float(platt_score_mean)
        self.platt_score_scale = float(platt_score_scale)
        self._validate()

    def _validate(self):
        if not self.model_dir:
            raise ValueError("ECAPA model_dir is required")
        if not math.isfinite(self.threshold) or not -1.0 <= self.threshold <= 1.0:
            raise ValueError("ECAPA threshold must be a finite cosine score")
        if self.min_region_duration_sec <= 0:
            raise ValueError("ECAPA min_region_duration_sec must be positive")
        if self.max_regions_per_speaker < 1:
            raise ValueError("ECAPA max_regions_per_speaker must be positive")
        if self.torch_num_threads < 1:
            raise ValueError("ECAPA torch_num_threads must be positive")
        if self.timeout_sec < 1:
            raise ValueError("ECAPA timeout_sec must be positive")
        calibration_values = (
            self.platt_a,
            self.platt_b,
            self.platt_score_mean,
            self.platt_score_scale,
        )
        if not all(math.isfinite(value) for value in calibration_values):
            raise ValueError("ECAPA Platt calibration values must be finite")
        if self.platt_score_scale <= 0:
            raise ValueError("ECAPA platt_score_scale must be positive")

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "threshold": self.threshold,
            "min_region_duration_sec": self.min_region_duration_sec,
            "max_regions_per_speaker": self.max_regions_per_speaker,
            "torch_num_threads": self.torch_num_threads,
            "timeout_sec": self.timeout_sec,
            "runtime_cache_dir": self.runtime_cache_dir,
            "calibration_profile_id": self.calibration_profile_id,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "platt_score_mean": self.platt_score_mean,
            "platt_score_scale": self.platt_score_scale,
        }


class EcapaIdentityClient:
    def __init__(self, config):
        self.config = config
        self._model = None

    def compare_regions(self, audio_path, regions, context=None):
        del context
        safe_regions = _validate_regions(regions)
        if len(safe_regions) < 2:
            raise EcapaIdentityError("fewer than two ECAPA candidate regions")
        if any(
            item["duration_sec"] + 1e-6 < self.config.min_region_duration_sec
            for item in safe_regions
        ):
            raise EcapaIdentityError(
                "ECAPA candidate region is shorter than the configured minimum"
            )
        embeddings = self._embeddings_for_regions(audio_path, safe_regions)
        comparisons = _build_comparisons(safe_regions, embeddings, self.config)
        return {
            "regions": safe_regions,
            "comparisons": comparisons,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "hyperparams_sha256": HYPERPARAMS_SHA256,
            "score_kind": "cosine_similarity",
            "calibration_profile_id": self.config.calibration_profile_id or None,
        }

    def _embeddings_for_regions(self, audio_path, regions):
        try:
            import numpy as np
            import soundfile as sf
            import torch
        except ImportError as exc:
            raise EcapaIdentityError(
                "ECAPA inference requires numpy, soundfile, and torch"
            ) from exc

        try:
            audio, sample_rate = sf.read(
                str(audio_path), dtype="float32", always_2d=False
            )
        except Exception as exc:
            raise EcapaIdentityError("failed to decode ECAPA source audio") from exc
        if int(sample_rate) != 16000:
            raise EcapaIdentityError(
                "ECAPA requires 16 kHz audio, got %s" % sample_rate
            )
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio):
            raise EcapaIdentityError("ECAPA source audio is empty")
        if not bool(np.all(np.isfinite(audio))):
            raise EcapaIdentityError("ECAPA source audio contains non-finite samples")

        model = self._get_model()
        embeddings = {}
        for region in regions:
            start_frame = max(0, int(round(region["start_sec"] * sample_rate)))
            end_frame = min(len(audio), int(round(region["end_sec"] * sample_rate)))
            if end_frame <= start_frame:
                raise EcapaIdentityError(
                    "ECAPA candidate region is empty: %s" % region["region_id"]
                )
            minimum_frames = int(
                round(self.config.min_region_duration_sec * sample_rate)
            )
            if end_frame - start_frame < minimum_frames:
                raise EcapaIdentityError(
                    "ECAPA decoded candidate region is shorter than the configured "
                    "minimum: %s" % region["region_id"]
                )
            signal = torch.from_numpy(
                np.asarray(audio[start_frame:end_frame], dtype=np.float32)
            ).unsqueeze(0)
            try:
                with torch.no_grad():
                    embedding = model.encode_batch(signal, normalize=False)
                values = embedding.detach().cpu().reshape(-1).tolist()
            except Exception as exc:
                raise EcapaIdentityError(
                    "ECAPA embedding inference failed for %s" % region["region_id"]
                ) from exc
            embeddings[region["region_id"]] = _normalize_embedding(values)
        return embeddings

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
        except ImportError as exc:
            raise EcapaIdentityError(
                "ECAPA requires torch and speechbrain in its model environment"
            ) from exc

        model_dir = Path(self.config.model_dir).expanduser().resolve()
        required = (
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        )
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise EcapaIdentityError(
                "ECAPA model directory is missing required files: %s"
                % ", ".join(missing)
            )
        cache_dir = (
            Path(self.config.runtime_cache_dir).expanduser().resolve()
            if self.config.runtime_cache_dir
            else model_dir / "runtime_cache"
        )
        with _MODEL_INIT_LOCK:
            if self._model is None:
                torch.set_num_threads(self.config.torch_num_threads)
                try:
                    self._model = EncoderClassifier.from_hparams(
                        source=str(model_dir),
                        savedir=str(cache_dir),
                        overrides={"pretrained_path": str(model_dir)},
                        run_opts={"device": self.config.device},
                    )
                    self._model.eval()
                except Exception as exc:
                    raise EcapaIdentityError("ECAPA model loading failed") from exc
        return self._model


class EcapaIdentitySubprocessClient:
    def __init__(self, config):
        self.config = config

    def compare_regions(self, audio_path, regions, context=None):
        safe_regions = _validate_regions(regions)
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "ecapa_identity_estimate",
            {
                "audio_path": str(audio_path),
                "regions": safe_regions,
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


def compare(audio_path, timeline_summary, config, context=None):
    """Compare clean regions derived only from a predicted timeline."""

    regions = select_candidate_regions(timeline_summary, config)
    if len(regions) < 2:
        raise EcapaIdentityError(
            "fewer than two sufficiently long candidate regions"
        )
    client = (
        EcapaIdentitySubprocessClient(config)
        if config.subprocess_python
        else EcapaIdentityClient(config)
    )
    return client.compare_regions(audio_path, regions, context=context)


def select_candidate_regions(timeline_summary, config):
    """Apply the CAM++ predicted-timeline non-overlap region contract."""

    _reject_private_input(timeline_summary)
    safe_timeline = _predicted_timeline_view(timeline_summary)
    return _campplus_select_candidate_regions(safe_timeline, config)


def validate_subprocess_request(request):
    """Reject worker payloads that contain anything outside inference scope."""

    if not isinstance(request, dict):
        raise EcapaIdentityError("ECAPA subprocess request must be an object")
    allowed = {"audio_path", "regions", "config"}
    unexpected = sorted(set(request) - allowed)
    if unexpected:
        raise EcapaIdentityError(
            "ECAPA subprocess request contains forbidden fields: %s"
            % ", ".join(unexpected)
        )
    missing = sorted(allowed - set(request))
    if missing:
        raise EcapaIdentityError(
            "ECAPA subprocess request is missing fields: %s" % ", ".join(missing)
        )
    _reject_private_input(request["regions"])
    return request


def _campplus_select_candidate_regions(timeline_summary, config):
    # Keep one source of truth for candidate selection.  The import is lazy so
    # the lightweight orchestration process does not load model dependencies.
    from tagger.tools.speaker_v2.campplus_identity import (
        select_candidate_regions as campplus_select_candidate_regions,
    )

    return campplus_select_candidate_regions(timeline_summary, config)


def _predicted_timeline_view(timeline_summary):
    if not isinstance(timeline_summary, dict):
        raise EcapaIdentityError("predicted timeline summary must be an object")
    source_segments = timeline_summary.get(
        "activity_segments", timeline_summary.get("segments", [])
    )
    overlap_segments = timeline_summary.get(
        "overlap_activity_segments", timeline_summary.get("overlap_segments", [])
    )
    if not isinstance(source_segments, list) or not isinstance(overlap_segments, list):
        raise EcapaIdentityError("predicted timeline segments must be lists")
    return {
        "activity_segments": [
            _timeline_segment(item, require_speaker=True) for item in source_segments
        ],
        "overlap_activity_segments": [
            _timeline_segment(item, require_speaker=False) for item in overlap_segments
        ],
    }


def _timeline_segment(item, require_speaker):
    if not isinstance(item, dict):
        raise EcapaIdentityError("predicted timeline segment must be an object")
    required = {"start_sec", "end_sec"}
    if require_speaker:
        required.add("speaker_id")
    missing = sorted(required - set(item))
    if missing:
        raise EcapaIdentityError(
            "predicted timeline segment is missing fields: %s" % ", ".join(missing)
        )
    try:
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])
    except (TypeError, ValueError) as exc:
        raise EcapaIdentityError("predicted timeline interval must be numeric") from exc
    if not math.isfinite(start_sec) or not math.isfinite(end_sec):
        raise EcapaIdentityError("predicted timeline interval must be finite")
    if start_sec < 0 or end_sec <= start_sec:
        raise EcapaIdentityError("predicted timeline interval is invalid")
    result = {"start_sec": start_sec, "end_sec": end_sec}
    if require_speaker:
        result["speaker_id"] = str(item["speaker_id"])
    return result


def _validate_regions(regions):
    _reject_private_input(regions)
    if not isinstance(regions, (list, tuple)):
        raise EcapaIdentityError("ECAPA regions must be a list")
    safe = []
    seen = set()
    for item in regions:
        if not isinstance(item, dict):
            raise EcapaIdentityError("ECAPA candidate region must be an object")
        required = {"region_id", "speaker_id", "start_sec", "end_sec"}
        missing = sorted(required - set(item))
        if missing:
            raise EcapaIdentityError(
                "ECAPA candidate region is missing fields: %s" % ", ".join(missing)
            )
        region_id = str(item["region_id"])
        if not region_id or region_id in seen:
            raise EcapaIdentityError("ECAPA candidate region IDs must be unique")
        try:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (TypeError, ValueError) as exc:
            raise EcapaIdentityError("ECAPA candidate interval must be numeric") from exc
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise EcapaIdentityError("ECAPA candidate interval must be finite")
        if start_sec < 0 or end_sec <= start_sec:
            raise EcapaIdentityError("ECAPA candidate interval is invalid")
        record = {key: item[key] for key in _REGION_FIELDS if key in item}
        record.update(
            {
                "region_id": region_id,
                "speaker_id": str(item["speaker_id"]),
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "duration_sec": round(end_sec - start_sec, 6),
            }
        )
        safe.append(record)
        seen.add(region_id)
    return safe


def _build_comparisons(regions, embeddings, config):
    normalized = {}
    embedding_size = None
    for region in regions:
        region_id = region["region_id"]
        if region_id not in embeddings:
            raise EcapaIdentityError(
                "ECAPA embedding is missing for region %s" % region_id
            )
        normalized[region_id] = _normalize_embedding(embeddings[region_id])
        if embedding_size is None:
            embedding_size = len(normalized[region_id])
        elif len(normalized[region_id]) != embedding_size:
            raise EcapaIdentityError("ECAPA embedding dimensions do not match")

    comparisons = []
    for left in range(len(regions)):
        for right in range(left + 1, len(regions)):
            left_region = regions[left]
            right_region = regions[right]
            score = sum(
                a * b
                for a, b in zip(
                    normalized[left_region["region_id"]],
                    normalized[right_region["region_id"]],
                )
            )
            score = max(-1.0, min(1.0, float(score)))
            same_cluster = (
                left_region["speaker_id"] == right_region["speaker_id"]
            )
            probability_same = (
                round(_platt_probability(score, config), 8)
                if config.calibration_profile_id
                else None
            )
            comparisons.append(
                {
                    "comparison_kind": (
                        "within_source_cluster"
                        if same_cluster
                        else "cross_source_cluster"
                    ),
                    "region_ids": [
                        left_region["region_id"],
                        right_region["region_id"],
                    ],
                    "speaker_pair": [
                        left_region["speaker_id"],
                        right_region["speaker_id"],
                    ],
                    "score": round(score, 8),
                    "decision": "same" if score >= config.threshold else "different",
                    "threshold": config.threshold,
                    "model_output_text": None,
                    "probability_same": probability_same,
                    "calibration_profile_id": config.calibration_profile_id or None,
                }
            )
    return comparisons


def _normalize_embedding(value):
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        flattened = _flatten_numbers(value)
        norm = math.sqrt(sum(item * item for item in flattened))
    except Exception as exc:
        raise EcapaIdentityError("ECAPA returned an invalid embedding") from exc
    if not flattened or not math.isfinite(norm) or norm <= 0:
        raise EcapaIdentityError("ECAPA returned a zero or non-finite embedding")
    return [item / norm for item in flattened]


def _flatten_numbers(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite embedding value")
    return [number]


def _platt_probability(score, config):
    logit = config.platt_a * (
        (float(score) - config.platt_score_mean) / config.platt_score_scale
    ) + config.platt_b
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _reject_private_input(value, path="input"):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            forbidden = normalized in _FORBIDDEN_EXACT_FIELDS or any(
                normalized.startswith(prefix) for prefix in _FORBIDDEN_FIELD_PREFIXES
            )
            if forbidden:
                raise EcapaIdentityError(
                    "private gold field is forbidden in ECAPA inference input: %s.%s"
                    % (path, key)
                )
            _reject_private_input(item, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_input(item, "%s[%d]" % (path, index))


def _subprocess_config(config):
    result = config.to_record()
    result["subprocess_python"] = ""
    result.pop("timeout_sec", None)
    return result
