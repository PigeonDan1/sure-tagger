"""RT60 estimator derived from a registered Rec-RIR output."""

import math

from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.room_acoustic.rir_estimator import (
    METHOD,
    RecRirError,
    validate_rir_payload,
)


TOOL_NAME = "rt60_estimator"
ROUND_DIGITS = 6
FIT_UPPER_DB = -5.0
FIT_LOWER_DB = -25.0


def run(rir, context=None, **_kwargs):
    value = estimate_rt60_sec(rir)
    return ToolResult(
        tag_path="room_acoustic.rt60_sec",
        value=round(value, ROUND_DIGITS),
        tool_name=TOOL_NAME,
        tool_version=TOOL_VERSION,
        method="%s_schroeder_t20" % METHOD,
        status="estimated",
        confidence=1.0,
        evidence={
            "fit_upper_db": FIT_UPPER_DB,
            "fit_lower_db": FIT_LOWER_DB,
        },
    )


def estimate_rt60_sec(rir):
    payload = validate_rir_payload(rir)
    samples = payload["samples"]
    sample_rate_hz = payload["sample_rate_hz"]
    energy = [sample * sample for sample in samples]
    total_energy = sum(energy)
    if total_energy <= 0.0:
        raise RecRirError("RIR energy must be positive before RT60 estimation")

    edc = []
    running = 0.0
    for value in reversed(energy):
        running += value
        edc.append(running)
    edc.reverse()

    points = []
    for index, value in enumerate(edc):
        if value <= 0.0:
            continue
        db = 10.0 * math.log10(value / total_energy)
        if FIT_LOWER_DB <= db <= FIT_UPPER_DB:
            points.append((float(index) / sample_rate_hz, db))

    if len(points) < 2:
        raise RecRirError("RIR decay has too few points for RT60 estimation")

    slope = _linear_regression_slope(points)
    if slope >= 0.0 or not math.isfinite(slope):
        raise RecRirError("RIR decay slope is invalid for RT60 estimation")

    rt60 = -60.0 / slope
    if not math.isfinite(rt60) or rt60 < 0.0:
        raise RecRirError("RT60 estimate is invalid")
    return rt60


def _linear_regression_slope(points):
    count = float(len(points))
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    mean_x = sum_x / count
    mean_y = sum_y / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) * (x - mean_x) for x, _y in points)
    if denominator <= 0.0:
        raise RecRirError("RT60 regression denominator is zero")
    return numerator / denominator

