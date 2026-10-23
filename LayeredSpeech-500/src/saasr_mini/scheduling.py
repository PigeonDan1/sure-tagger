from __future__ import annotations

from dataclasses import replace

from .audio import interval_overlap_ratio
from .schemas import Turn, Utterance


def max_concurrency(turns: list[Turn]) -> int:
    events: list[tuple[float, int]] = []
    for turn in turns:
        events.extend(((turn.start, 1), (turn.end, -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def _candidate(
    utterances: list[Utterance], durations: list[float], sample_rate: int, scale: float
) -> list[Turn]:
    turns: list[Turn] = []
    cursor = 0.35
    previous_speaker: int | None = None
    for utterance, duration in zip(utterances, durations, strict=True):
        weight = {"none": 0.18, "backchannel": 0.72, "interrupt": 0.5}[utterance.overlap_hint]
        if previous_speaker == utterance.speaker_index:
            weight = 0.0
        overlap = scale * weight * min(duration, turns[-1].end - turns[-1].start) if turns else 0
        start = max(0.05, cursor + 0.16 - overlap)
        end = start + duration
        turns.append(
            Turn(
                utterance_id=utterance.utterance_id,
                speaker_index=utterance.speaker_index,
                text=utterance.text,
                instructions=utterance.instructions,
                edit_instruction=utterance.edit_instruction,
                start=start,
                end=end,
                start_sample=round(start * sample_rate),
                end_sample=round(end * sample_rate),
            )
        )
        cursor = end
        previous_speaker = utterance.speaker_index
    return turns


def schedule(
    utterances: list[Utterance],
    durations: list[float],
    sample_rate: int,
    target_overlap: float,
    concurrency_limit: int,
) -> tuple[list[Turn], float]:
    best: list[Turn] | None = None
    best_error = float("inf")
    for step in range(601):
        candidate = _candidate(utterances, durations, sample_rate, step / 100.0)
        if max_concurrency(candidate) > concurrency_limit:
            continue
        ratio = interval_overlap_ratio([(turn.start, turn.end) for turn in candidate])
        error = abs(ratio - target_overlap)
        if error < best_error:
            best, best_error = candidate, error
    if best is None:
        raise RuntimeError(
            f"Could not schedule target overlap {target_overlap:.3f} within the concurrency limit"
        )
    # Some edited turn-duration combinations cannot reach the requested
    # overlap while respecting the concurrency limit.  A legal candidate with
    # minimum error is still a valid scene; preserve it and let the manifest
    # expose the achieved ratio instead of aborting the whole batch.
    # Recalculate timestamps from integer sample positions for sample-exact annotations.
    best = [
        replace(
            turn,
            start=turn.start_sample / sample_rate,
            end=turn.end_sample / sample_rate,
        )
        for turn in best
    ]
    ratio = interval_overlap_ratio([(turn.start, turn.end) for turn in best])
    return best, ratio
