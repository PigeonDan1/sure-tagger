"""Speaker-text tracks that retain their acoustic dependency lineage."""

import re

from tagger.tools.speaker_v2.contracts import stable_id


LEXICAL_TRACK_VERSION = "speaker_text_track_v2.0-shadow.1"


def speaker_text_track(timeline_evidence):
    summary = timeline_evidence.get("payload", {}).get("timeline_summary", {})
    units = []
    for index, segment in enumerate(summary.get("segments", [])):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        units.append(
            {
                "unit_id": "lex_%06d" % index,
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "speaker_id": segment["speaker_id"],
                "text": text,
                "assignment": "source_native_joint",
                "timestamp_method": "joint_segment_interval",
            }
        )
    identity = {
        "track_version": LEXICAL_TRACK_VERSION,
        "sample_id": timeline_evidence["scope"]["sample_id"],
        "source_evidence_id": timeline_evidence["evidence_id"],
        "dependency_groups": list(timeline_evidence["dependency_groups"]),
        "units": units,
    }
    identity["track_id"] = stable_id("texttrack", identity)
    return identity


def project_asr_track(asr_evidence, timeline_evidence, min_occupancy_sec=0.05):
    """Project independent ASR intervals onto a speaker timeline.

    The result inherits both evidence groups and is therefore a comparison or
    ambiguity guard, never an additional speaker-event vote.
    """

    _validate_projection_scope(asr_evidence, timeline_evidence)
    asr_units = asr_evidence.get("payload", {}).get("lexical_units", [])
    speaker_segments = timeline_evidence.get("payload", {}).get(
        "timeline_summary", {}
    ).get("segments", [])
    units = []
    for index, raw in enumerate(asr_units):
        start = float(raw["start_sec"])
        end = float(raw["end_sec"])
        occupancy = {}
        for segment in speaker_segments:
            overlap = max(
                0.0,
                min(end, float(segment["end_sec"]))
                - max(start, float(segment["start_sec"])),
            )
            if overlap >= min_occupancy_sec:
                speaker_id = segment["speaker_id"]
                occupancy[speaker_id] = occupancy.get(speaker_id, 0.0) + overlap
        candidates = sorted(
            occupancy,
            key=lambda speaker_id: (-occupancy[speaker_id], speaker_id),
        )
        if len(candidates) == 1:
            assignment = "assigned"
            speaker_id = candidates[0]
        elif len(candidates) > 1:
            assignment = "ambiguous"
            speaker_id = None
        else:
            assignment = "unassigned"
            speaker_id = None
        units.append(
            {
                "unit_id": str(raw.get("unit_id", "lex_%06d" % index)),
                "start_sec": start,
                "end_sec": end,
                "text": str(raw.get("text", "")),
                "speaker_id": speaker_id,
                "speaker_candidates": candidates,
                "speaker_occupancy_sec": {
                    key: round(value, 6) for key, value in occupancy.items()
                },
                "assignment": assignment,
                "timestamp_method": raw.get(
                    "timestamp_method", "asr_segment_interval"
                ),
            }
        )
    dependencies = sorted(
        set(asr_evidence.get("dependency_groups", [])).union(
            timeline_evidence.get("dependency_groups", [])
        )
    )
    identity = {
        "track_version": LEXICAL_TRACK_VERSION,
        "sample_id": timeline_evidence["scope"]["sample_id"],
        "source_evidence_ids": [
            asr_evidence["evidence_id"],
            timeline_evidence["evidence_id"],
        ],
        "dependency_groups": dependencies,
        "units": units,
        "roles_allowed": [
            "diagnostic",
            "boundary_ambiguity_guard",
            "speaker_assignment_disagreement",
        ],
        "speaker_event_vote": False,
    }
    identity["track_id"] = stable_id("texttrack", identity)
    return identity


def text_track_comparison(track_a, track_b):
    tokens_a = _tokens(" ".join(item["text"] for item in track_a.get("units", [])))
    tokens_b = _tokens(" ".join(item["text"] for item in track_b.get("units", [])))
    distance = _edit_distance(tokens_a, tokens_b)
    denominator = max(1, len(tokens_a), len(tokens_b))
    dependencies = sorted(
        set(track_a.get("dependency_groups", [])).union(
            track_b.get("dependency_groups", [])
        )
    )
    result = {
        "comparison_version": LEXICAL_TRACK_VERSION,
        "track_ids": [track_a["track_id"], track_b["track_id"]],
        "dependency_groups": dependencies,
        "word_error_rate_symmetric_diagnostic": round(
            min(1.0, distance / float(denominator)), 6
        ),
        "roles_allowed": ["diagnostic", "boundary_ambiguity_guard"],
        "speaker_event_vote": False,
    }
    result["comparison_id"] = stable_id("textcmp", result)
    return result


def speaker_assignment_comparison(track_a, track_b):
    """Compare how identical ASR units land on two speaker timelines."""

    units_a = {str(item["unit_id"]): item for item in track_a.get("units", [])}
    units_b = {str(item["unit_id"]): item for item in track_b.get("units", [])}
    common_ids = sorted(set(units_a).intersection(units_b))
    state_matches = []
    coverage_disagreements = []
    ambiguity_disagreements = []
    for unit_id in common_ids:
        state_a = units_a[unit_id].get("assignment")
        state_b = units_b[unit_id].get("assignment")
        state_matches.append(state_a == state_b)
        covered_a = state_a in ("assigned", "ambiguous")
        covered_b = state_b in ("assigned", "ambiguous")
        if covered_a != covered_b:
            coverage_disagreements.append(unit_id)
        if (state_a == "ambiguous") != (state_b == "ambiguous"):
            ambiguity_disagreements.append(unit_id)
    dependencies = sorted(
        set(track_a.get("dependency_groups", [])).union(
            track_b.get("dependency_groups", [])
        )
    )
    result = {
        "comparison_version": LEXICAL_TRACK_VERSION,
        "track_ids": [track_a["track_id"], track_b["track_id"]],
        "dependency_groups": dependencies,
        "common_unit_count": len(common_ids),
        "assignment_state_agreement_ratio": (
            round(sum(state_matches) / float(len(state_matches)), 6)
            if state_matches
            else None
        ),
        "coverage_disagreement_unit_ids": coverage_disagreements,
        "ambiguity_disagreement_unit_ids": ambiguity_disagreements,
        "speaker_ids_compared": False,
        "roles_allowed": [
            "diagnostic",
            "boundary_ambiguity_guard",
            "speaker_assignment_disagreement",
        ],
        "speaker_event_vote": False,
    }
    result["comparison_id"] = stable_id("assigncmp", result)
    return result


def _tokens(text):
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, 1):
        current = [index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _validate_projection_scope(asr_evidence, timeline_evidence):
    asr_scope = asr_evidence.get("scope", {})
    timeline_scope = timeline_evidence.get("scope", {})
    for key in ("sample_id", "level", "start_sec", "end_sec"):
        if asr_scope.get(key) != timeline_scope.get(key):
            raise ValueError("ASR and speaker timeline scopes do not match")
    asr_hash = asr_evidence.get("applicability", {}).get("audio_sha256")
    timeline_hash = timeline_evidence.get("applicability", {}).get(
        "audio_sha256"
    )
    if asr_hash and timeline_hash and str(asr_hash) != str(timeline_hash):
        raise ValueError("ASR and speaker timeline audio hashes do not match")
