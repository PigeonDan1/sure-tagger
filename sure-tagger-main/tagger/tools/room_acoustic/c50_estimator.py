"""C50 clarity estimator derived from a registered Rec-RIR output."""

import math

from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.room_acoustic.rir_estimator import (
    METHOD,
    RecRirError,
    validate_rir_payload,
)


TOOL_NAME = "c50_estimator"
ROUND_DIGITS = 6
EARLY_WINDOW_SEC = 0.05


def run(rir, context=None, **_kwargs):
    value = estimate_c50_db(rir)
    return ToolResult(
        tag_path="room_acoustic.c50_db",
        value=round(value, ROUND_DIGITS),
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method="%s_clarity_c50" % METHOD,
        status="estimated",
        confidence=1.0,
        evidence={"early_window_sec": EARLY_WINDOW_SEC},
    )


def estimate_c50_db(rir):
    payload = validate_rir_payload(rir)
    samples = payload["samples"]
    sample_rate_hz = payload["sample_rate_hz"]
    direct_index = max(range(len(samples)), key=lambda index: abs(samples[index]))
    early_end = min(
        len(samples),
        direct_index + int(round(EARLY_WINDOW_SEC * sample_rate_hz)),
    )
    early_energy = sum(sample * sample for sample in samples[direct_index:early_end])
    late_energy = sum(sample * sample for sample in samples[early_end:])

    if early_energy <= 0.0 or late_energy <= 0.0:
        raise RecRirError("RIR early and late energy must be positive for C50")

    c50 = 10.0 * math.log10(early_energy / late_energy)
    if not math.isfinite(c50):
        raise RecRirError("C50 estimate is invalid")
    return c50

