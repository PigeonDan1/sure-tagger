"""Registry of audio quality tag tools."""

from tagger.tools.audio_quality import brouhaha_signal_estimator
from tagger.tools.audio_quality import dnsmos_quality_estimator


BROUHAHA_ACOUSTIC_TOOL = {
    "tag_path": "audio_quality",
    "tool_name": brouhaha_signal_estimator.TOOL_NAME,
    "run": brouhaha_signal_estimator.run,
}

DNSMOS_QUALITY_TOOL = {
    "tag_path": "audio_quality",
    "tool_name": dnsmos_quality_estimator.TOOL_NAME,
    "run": dnsmos_quality_estimator.run,
}

AUDIO_QUALITY_TOOLS = [
    BROUHAHA_ACOUSTIC_TOOL,
    DNSMOS_QUALITY_TOOL,
]
