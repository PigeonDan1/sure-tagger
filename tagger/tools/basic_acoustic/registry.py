"""Registry of basic acoustic tag tools."""

from tagger.tools.basic_acoustic import audio_probe
from tagger.tools.basic_acoustic import firered_vad_silence_detector
from tagger.tools.basic_acoustic import native_metadata_vad
from tagger.tools.basic_acoustic import nonspeech_ratio_calculator


AUDIO_PROBE_TOOL = {
    "tag_path": "basic_acoustic",
    "tool_name": audio_probe.TOOL_NAME,
    "run": audio_probe.run,
}

FIRERED_VAD_SILENCE_TOOL = {
    "tag_path": "basic_acoustic.silence_segments",
    "tool_name": firered_vad_silence_detector.TOOL_NAME,
    "run": firered_vad_silence_detector.run,
}

NATIVE_METADATA_VAD_TOOL = {
    "tag_path": "basic_acoustic.silence_segments",
    "tool_name": native_metadata_vad.TOOL_NAME,
    "run": native_metadata_vad.run,
}

NONSPEECH_RATIO_TOOL = {
    "tag_path": "basic_acoustic.nonspeech_ratio",
    "tool_name": nonspeech_ratio_calculator.TOOL_NAME,
    "run": nonspeech_ratio_calculator.run,
}

BASIC_ACOUSTIC_TOOLS = [
    AUDIO_PROBE_TOOL,
    NATIVE_METADATA_VAD_TOOL,
    FIRERED_VAD_SILENCE_TOOL,
    NONSPEECH_RATIO_TOOL,
]
