"""Brouhaha speech-coverage adapter for the speaker-v2 shadow pipeline.

The adapter deliberately exposes only utterance-level speech coverage.  A
Brouhaha activity annotation is not speaker-attributed and therefore must not
be interpreted as speaker count, multi-speaker, overlap, or change evidence.
"""

from __future__ import print_function

import hashlib
import math
from numbers import Real
from pathlib import Path
import time

from tagger.tools.basic_acoustic.brouhaha_signal_estimator import (
    BrouhahaClient,
    BrouhahaConfig,
    BrouhahaError,
    BrouhahaSubprocessClient,
)
from tagger.tools.basic_acoustic.brouhaha_vad_silence_detector import (
    extract_speech_segments,
)
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    speech_segments_to_silence_segments,
)


TOOL_VERSION = "brouhaha_coverage_v2.1-shadow.1"
MODEL_ID = "brouhaha_vad_v0.9.0"
MODEL_SHA256 = (
    "9c237e4a7b1de8b456dbee25db853342bf374b19d8732b72b61356519e390ae1"
)
EVIDENCE_TYPE = "speech_coverage"
CAPABILITIES = (EVIDENCE_TYPE,)
DEPENDENCY_GROUPS = ("G_brouhaha_activity_v0_9_0",)
BOUNDARY_EPSILON_SEC = 1e-3
ROUND_DIGITS = 6
BINARIZATION_PARAMETERS = {
    "onset": 0.780,
    "offset": 0.780,
    "min_duration_on_sec": 0.0,
    "min_duration_off_sec": 0.0,
    "source": "pinned_upstream_default_parameters",
}


class BrouhahaCoverageError(RuntimeError):
    """Raised when Brouhaha cannot produce valid coverage output."""


def estimate_coverage(
    audio_path,
    duration_sec,
    config=None,
    context=None,
    client=None,
):
    """Return deterministic utterance-level speech-coverage observations.

    ``client`` is injectable for tests.  Production calls select the existing
    basic-acoustic subprocess client whenever ``config.subprocess_python`` is
    configured, so the shared ``brouhaha_estimate`` worker remains the only
    out-of-process inference implementation.
    """

    duration_sec = _positive_duration(duration_sec)
    config = config or BrouhahaConfig()
    client = client or _default_client(config)
    started = time.time()
    try:
        output = client.estimate(audio_path, context=context)
        raw_speech_segments = extract_speech_segments(output)
        speech_segments = _normalize_speech_segments(
            raw_speech_segments,
            duration_sec,
        )
        silence_segments = speech_segments_to_silence_segments(
            speech_segments,
            duration_sec=duration_sec,
        )
    except BrouhahaCoverageError:
        raise
    except BrouhahaError as exc:
        raise BrouhahaCoverageError(str(exc)) from exc
    except Exception as exc:
        raise BrouhahaCoverageError("Brouhaha coverage inference failed") from exc

    speech_duration_sec = round(
        sum(
            segment["end_sec"] - segment["start_sec"]
            for segment in speech_segments
        ),
        ROUND_DIGITS,
    )
    return {
        "adapter_version": TOOL_VERSION,
        "model_id": MODEL_ID,
        "model_version": str(config.model_version),
        "evidence_type": EVIDENCE_TYPE,
        "capabilities": list(CAPABILITIES),
        "dependency_groups": list(DEPENDENCY_GROUPS),
        "raw_speech_segments": raw_speech_segments,
        "speech_segments": speech_segments,
        "silence_segments": silence_segments,
        "speech_duration_sec": speech_duration_sec,
        "speech_coverage_ratio": round(
            speech_duration_sec / duration_sec,
            ROUND_DIGITS,
        ),
        "binarization": dict(BINARIZATION_PARAMETERS),
        "boundary_postprocess": "clip_sort_merge_to_audio_duration",
        "runtime": {
            "elapsed_sec": round(time.time() - started, ROUND_DIGITS),
            "execution": (
                "shared_subprocess_worker"
                if config.subprocess_python
                else "in_process"
            ),
            "python": str(config.subprocess_python or ""),
            "device": "cuda" if config.use_gpu else "cpu",
        },
    }


def verify_model_asset(model_path):
    """Verify the pinned Brouhaha checkpoint and return its SHA256."""

    path = Path(str(model_path)).expanduser().resolve()
    if not path.is_file():
        raise BrouhahaCoverageError(
            "pinned Brouhaha checkpoint does not exist: %s" % path
        )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != MODEL_SHA256:
        raise BrouhahaCoverageError(
            "Brouhaha checkpoint SHA256 mismatch: expected %s, got %s"
            % (MODEL_SHA256, actual_sha256)
        )
    return actual_sha256


def _default_client(config):
    if config.subprocess_python:
        return BrouhahaSubprocessClient(config)
    return BrouhahaClient(config)


def _normalize_speech_segments(raw_segments, duration_sec):
    clipped = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise BrouhahaCoverageError(
                "raw speech segment %s must be an object" % index
            )
        start_sec = _finite_number(
            raw_segment.get("start_sec"),
            "raw_speech_segments[%s].start_sec" % index,
        )
        end_sec = _finite_number(
            raw_segment.get("end_sec"),
            "raw_speech_segments[%s].end_sec" % index,
        )
        start_sec = max(0.0, start_sec)
        end_sec = min(duration_sec, end_sec)
        if start_sec >= end_sec:
            continue
        clipped.append(
            {
                "start_sec": round(start_sec, ROUND_DIGITS),
                "end_sec": round(end_sec, ROUND_DIGITS),
            }
        )

    clipped.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged = []
    for segment in clipped:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        if segment["start_sec"] <= previous["end_sec"] + BOUNDARY_EPSILON_SEC:
            previous["end_sec"] = round(
                max(previous["end_sec"], segment["end_sec"]),
                ROUND_DIGITS,
            )
        else:
            merged.append(dict(segment))
    return merged


def _positive_duration(value):
    value = _finite_number(value, "duration_sec")
    if value <= 0:
        raise BrouhahaCoverageError("duration_sec must be positive")
    return round(value, ROUND_DIGITS)


def _finite_number(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BrouhahaCoverageError("%s must be numeric" % field_name)
    value = float(value)
    if not math.isfinite(value):
        raise BrouhahaCoverageError("%s must be finite" % field_name)
    return value
