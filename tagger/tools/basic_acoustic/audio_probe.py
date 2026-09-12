"""Tool for basic acoustic tags read from the audio file."""

from tagger.tools.acoustic_io import get_audio_info
from tagger.tools.base import ToolResult


TOOL_NAME = "audio_probe"


def run(audio_path, context=None, **_kwargs):
    info = get_audio_info(audio_path, context)
    evidence = info.base_evidence()
    return [
        ToolResult(
            tag_path="basic_acoustic.duration_sec",
            value=round(info.duration_sec, 6),
            tool_name=TOOL_NAME,
            method=info.method,
            evidence=evidence,
        ),
        ToolResult(
            tag_path="basic_acoustic.sample_rate_hz",
            value=info.sample_rate_hz,
            tool_name=TOOL_NAME,
            method=info.method,
            evidence=evidence,
        ),
        ToolResult(
            tag_path="basic_acoustic.channels",
            value=info.channels,
            tool_name=TOOL_NAME,
            method=info.method,
            evidence=evidence,
        ),
    ]
