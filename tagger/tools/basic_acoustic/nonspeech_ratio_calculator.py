"""Tool for ``basic_acoustic.nonspeech_ratio``.

The value is derived from FireRed VAD non-speech segments.  It is the fraction
of the clip not detected as human speech, rather than a physical silence or
low-energy measurement.
"""

from tagger.tools.base import ToolResult
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    METHOD,
    validate_silence_segments,
)


TOOL_NAME = "nonspeech_ratio_calculator"


def run(silence_segments, duration_sec, **_kwargs):
    if duration_sec is None or duration_sec <= 0:
        raise ValueError("duration_sec must be positive before non-speech ratio")
    validate_silence_segments(silence_segments, duration_sec)
    silence_total = sum(
        segment["end_sec"] - segment["start_sec"] for segment in silence_segments
    )
    ratio = silence_total / duration_sec
    if ratio < 0 or ratio > 1:
        raise ValueError("nonspeech_ratio is outside [0, 1]")
    return ToolResult(
        tag_path="basic_acoustic.nonspeech_ratio",
        value=round(ratio, 6),
        tool_name=TOOL_NAME,
        method=METHOD,
        evidence={"silence_segments": silence_segments, "duration_sec": duration_sec},
    )
