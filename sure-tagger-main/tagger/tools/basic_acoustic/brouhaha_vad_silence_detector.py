"""Diagnostic Brouhaha VAD silence segment extractor.

This tool is only for comparing Brouhaha VAD boundaries against FireRed VAD.
It must not be used as the public source of basic_acoustic.silence_segments.
"""

import math
from numbers import Real

from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.audio_quality.brouhaha_signal_estimator import (
    BrouhahaClient,
    BrouhahaConfig,
    BrouhahaError,
)
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    speech_segments_to_silence_segments,
)


TOOL_NAME = "brouhaha_vad_silence_detector"
METHOD = "Brouhaha VAD"


def run(audio_path, duration_sec, context=None, config=None, client=None, **_kwargs):
    if duration_sec is None or duration_sec <= 0:
        raise BrouhahaError("duration_sec must be positive before Brouhaha VAD")

    config = config or BrouhahaConfig()
    client = client or BrouhahaClient(config)
    output = client.estimate(audio_path, context=context)
    raw_speech_segments = extract_speech_segments(output)
    speech_segments = clip_segments_to_duration(raw_speech_segments, duration_sec)
    silence_segments = speech_segments_to_silence_segments(
        speech_segments,
        duration_sec=duration_sec,
    )
    return ToolResult(
        tag_path="diagnostic.brouhaha_silence_segments",
        value=silence_segments,
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method=METHOD,
        tool_type="model_inference",
        evidence={
            "brouhaha_config": config.to_record(),
            "raw_speech_segments": raw_speech_segments,
            "speech_segments": speech_segments,
            "boundary_postprocess": "clip_to_audio_duration",
            "note": "diagnostic_only_not_public_silence_source",
        },
    )


def extract_speech_segments(output):
    if not isinstance(output, dict):
        raise BrouhahaError("Brouhaha output must be an object")
    if "annotation" not in output:
        raise BrouhahaError("Brouhaha output is missing field: annotation")

    annotation = output["annotation"]
    if isinstance(annotation, list):
        return _normalize_raw_segments(annotation)

    if hasattr(annotation, "itersegments"):
        return _normalize_raw_segments(
            [
                {"start_sec": segment.start, "end_sec": segment.end}
                for segment in annotation.itersegments()
            ]
        )

    if hasattr(annotation, "itertracks"):
        return _normalize_raw_segments(
            [
                {"start_sec": segment.start, "end_sec": segment.end}
                for segment, _track in annotation.itertracks()
            ]
        )

    raise BrouhahaError("Brouhaha annotation has no iterable segments")


def _normalize_raw_segments(raw_segments):
    segments = []
    for index, segment in enumerate(raw_segments):
        if isinstance(segment, dict):
            start_sec = segment.get("start_sec")
            end_sec = segment.get("end_sec")
        elif isinstance(segment, (list, tuple)) and len(segment) == 2:
            start_sec, end_sec = segment
        else:
            raise BrouhahaError(
                "Brouhaha annotation segment %s must be an object or pair" % index
            )
        segments.append(
            {
                "start_sec": round(_require_number(start_sec, "start_sec"), 6),
                "end_sec": round(_require_number(end_sec, "end_sec"), 6),
            }
        )
    return segments


def clip_segments_to_duration(raw_segments, duration_sec):
    duration_sec = _require_number(duration_sec, "duration_sec")
    if duration_sec <= 0:
        raise BrouhahaError("duration_sec must be positive before Brouhaha VAD")

    clipped = []
    for segment in raw_segments:
        start_sec = max(0.0, segment["start_sec"])
        end_sec = min(duration_sec, segment["end_sec"])
        if start_sec < end_sec:
            clipped.append(
                {
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                }
            )
    return clipped


def _require_number(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BrouhahaError("Brouhaha annotation %s must be numeric" % field_name)
    value = float(value)
    if not math.isfinite(value):
        raise BrouhahaError("Brouhaha annotation %s must be finite" % field_name)
    return value
