"""Registry of room acoustic tag tools."""

from tagger.tools.room_acoustic import c50_estimator
from tagger.tools.room_acoustic import rir_estimator
from tagger.tools.room_acoustic import rt60_estimator


RECRIR_RIR_TOOL = {
    "tag_path": "room_acoustic.rir",
    "tool_name": rir_estimator.TOOL_NAME,
    "run": rir_estimator.run,
}

RT60_TOOL = {
    "tag_path": "room_acoustic.rt60_sec",
    "tool_name": rt60_estimator.TOOL_NAME,
    "run": rt60_estimator.run,
}

C50_TOOL = {
    "tag_path": "room_acoustic.c50_db",
    "tool_name": c50_estimator.TOOL_NAME,
    "run": c50_estimator.run,
}

RIR_RELATED_TOOLS = [
    RECRIR_RIR_TOOL,
    RT60_TOOL,
    C50_TOOL,
]

ROOM_ACOUSTIC_TOOLS = list(RIR_RELATED_TOOLS)
