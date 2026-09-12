"""Conservative per-claim resolver for the v2 shadow deployment."""

from tagger.tools.speaker_v2.contracts import (
    closure_independent,
    stable_id,
    validate_evidence,
)
from tagger.tools.speaker_v2.timeline import timeline_comparison


FUSION_SCHEMA_VERSION = "speaker_fusion_artifact_v2.0-shadow.1"
ALLOWED_CLAIM_STATUS = set(
    ["certified", "supported", "conflicted", "insufficient"]
)


def resolve(sample_id, duration_sec, evidence, hypotheses=None):
    _validate_evidence_scope(sample_id, duration_sec, evidence)
    timelines = _usable_by_capability(evidence, "speaker_timeline")
    identities = _usable_by_capability(evidence, "speaker_identity_comparison")
    coverage = _usable_by_capability(evidence, "speech_coverage")
    observations = [_timeline_observation(item) for item in timelines]
    comparisons = []
    for left in range(len(timelines)):
        for right in range(left + 1, len(timelines)):
            if closure_independent(
                evidence, timelines[left], timelines[right]
            ):
                comparisons.append(
                    {
                        "evidence_ids": [
                            timelines[left]["evidence_id"],
                            timelines[right]["evidence_id"],
                        ],
                        "comparison": timeline_comparison(
                            timelines[left]["payload"]["timeline_summary"],
                            timelines[right]["payload"]["timeline_summary"],
                        ),
                    }
                )

    count_claim = _resolve_count(evidence, timelines, identities)
    multi_claim = _resolve_boolean_claim(
        "multi_speaker",
        timelines,
        lambda summary: summary["observed_speaker_count"] >= 2,
        identities=identities,
        coverage=coverage,
        all_evidence=evidence,
    )
    overlap_claim = _resolve_boolean_claim(
        "speaker_overlap",
        timelines,
        lambda summary: bool(summary["overlap_segments"]),
        identities=identities,
        coverage=coverage,
        all_evidence=evidence,
        comparisons=comparisons,
        event_key="overlap_event_iou",
        event_min=0.50,
    )
    change_claim = _resolve_boolean_claim(
        "speaker_change",
        timelines,
        lambda summary: bool(summary["change_candidate_points_sec"]),
        identities=identities,
        coverage=coverage,
        all_evidence=evidence,
        comparisons=comparisons,
        event_key="change_candidate_match_ratio",
        event_min=0.80,
    )
    claims = {
        "speaker_count": count_claim,
        "multi_speaker": multi_claim,
        "speaker_overlap": overlap_claim,
        "speaker_change": change_claim,
    }
    artifact_identity = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "scope": {
            "level": "utterance",
            "start_sec": 0.0,
            "end_sec": round(float(duration_sec), 6),
        },
        "profile": "v2-shadow",
        "evidence_ids": sorted(item["evidence_id"] for item in evidence),
        "claims": claims,
        "hypothesis_case_ids": sorted(
            item["case_id"] for item in (hypotheses or []) if item is not None
        ),
    }
    fusion = dict(artifact_identity)
    fusion.update(
        {
            "fusion_id": stable_id("fusion", artifact_identity),
            "timeline_observations": observations,
            "timeline_comparisons": comparisons,
            "hypotheses": list(hypotheses or []),
            "public_adapter": {
                "enabled": False,
                "reason": "shadow profile never writes public metadata",
                "speaker": {
                    "multi_speaker": None,
                    "speaker_overlap": None,
                    "speaker_change": None,
                },
            },
        }
    )
    validate_fusion(fusion)
    return fusion


def validate_fusion(fusion):
    if fusion.get("schema_version") != FUSION_SCHEMA_VERSION:
        raise ValueError("unsupported fusion schema")
    if fusion.get("scope", {}).get("level") != "utterance":
        raise ValueError("fusion scope must be utterance-level")
    for claim in fusion.get("claims", {}).values():
        if claim.get("status") not in ALLOWED_CLAIM_STATUS:
            raise ValueError("invalid claim status")
    if any(
        value is not None
        for value in fusion.get("public_adapter", {})
        .get("speaker", {})
        .values()
    ):
        raise ValueError("v2-shadow must not emit public speaker values")


def _resolve_count(all_evidence, timelines, identities):
    if not timelines:
        return {
            "status": "insufficient",
            "observed_values": [],
            "supported_lower_bound": None,
            "certified_lower_bound": None,
            "certified_upper_bound": None,
            "exact": None,
            "reason": "no usable full-sample speaker timeline",
        }
    counts = [
        int(item["payload"]["timeline_summary"]["observed_speaker_count"])
        for item in timelines
    ]
    observed_values = [
        {
            "evidence_id": item["evidence_id"],
            "source": item["source"]["name"],
            "value": count,
        }
        for item, count in zip(timelines, counts)
    ]
    supported_lower = 1 if max(counts) >= 1 else 0
    different_pair_observations = []
    cluster_consistency_diagnostics = []
    different_pairs = []
    for identity in identities:
        quality = identity.get("quality", {})
        identity_eligible = bool(quality.get("calibration_profile_id"))
        identity_eligible = identity_eligible and bool(
            quality.get("candidate_selection_independent", False)
        )
        identity_eligible = identity_eligible and bool(
            quality.get("validator_dependency_closure_independent", False)
        )
        identity_eligible = identity_eligible and all(
            closure_independent(all_evidence, identity, timeline)
            for timeline in timelines
        )
        for comparison in identity.get("payload", {}).get("comparisons", []):
            if comparison.get("decision") != "different":
                continue
            observation = {
                "evidence_id": identity["evidence_id"],
                "speaker_pair": comparison.get("speaker_pair"),
                "falsifier_eligible": identity_eligible,
            }
            if comparison.get("comparison_kind") in (
                "within_source_cluster",
                "within_cluster",
            ):
                cluster_consistency_diagnostics.append(observation)
                continue
            different_pair_observations.append(observation)
            if identity_eligible:
                different_pairs.append(observation)
    if different_pairs:
        supported_lower = max(supported_lower, 2)
    elif len(timelines) >= 2:
        pair_lower_bounds = [
            min(counts[left], counts[right])
            for left in range(len(timelines))
            for right in range(left + 1, len(timelines))
            if closure_independent(
                all_evidence, timelines[left], timelines[right]
            )
        ]
        if pair_lower_bounds:
            supported_lower = max(supported_lower, max(pair_lower_bounds))

    status = "conflicted" if len(set(counts)) > 1 else "supported"
    return {
        "status": status,
        "observed_values": observed_values,
        "observed_min": min(counts),
        "observed_max": max(counts),
        "observed_lower_bound_candidate": max(counts),
        "supported_lower_bound": supported_lower,
        "certified_lower_bound": None,
        "certified_upper_bound": None,
        "exact": None,
        "identity_different_pair_support": different_pairs,
        "identity_different_pair_observations": different_pair_observations,
        "identity_cluster_consistency_diagnostics": (
            cluster_consistency_diagnostics
        ),
        "reason": (
            "timeline counts disagree"
            if status == "conflicted"
            else "no calibrated upper-bound certificate in shadow deployment"
        ),
        "required_for_certification": [
            "independent full-scope timeline agreement",
            "identity partition coverage",
            "domain calibration for both lower and upper bound",
        ],
    }


def _resolve_boolean_claim(
    claim_name,
    timelines,
    predicate,
    identities,
    coverage,
    all_evidence,
    comparisons=None,
    event_key=None,
    event_min=None,
):
    if not timelines:
        return {
            "status": "insufficient",
            "candidate_value": None,
            "public_value": None,
            "observations": [],
            "reason": "no usable full-sample timeline",
        }
    observations = [
        {
            "evidence_id": item["evidence_id"],
            "source": item["source"]["name"],
            "value": bool(predicate(item["payload"]["timeline_summary"])),
        }
        for item in timelines
    ]
    values = [item["value"] for item in observations]
    candidate = values[0] if len(set(values)) == 1 else None
    status = "conflicted" if candidate is None else "supported"
    reasons = []
    if status == "conflicted":
        reasons.append("independent timeline bool values disagree")

    independent_timeline_pairs = [
        (timelines[left], timelines[right])
        for left in range(len(timelines))
        for right in range(left + 1, len(timelines))
        if closure_independent(
            all_evidence, timelines[left], timelines[right]
        )
    ]
    event_agreement = False
    if event_key is None:
        event_agreement = bool(independent_timeline_pairs)
    else:
        event_agreement = any(
            item["comparison"].get(event_key, 0.0) >= event_min
            for item in (comparisons or [])
        )
    if (
        candidate is True
        and event_key is not None
        and independent_timeline_pairs
        and not event_agreement
    ):
        status = "conflicted"
        reasons.append("positive event locations disagree across timelines")
    relevant_identity_pairs = set()
    for timeline in timelines:
        relevant_identity_pairs.update(
            _claim_speaker_pairs(
                timeline["payload"]["timeline_summary"], claim_name
            )
        )
    identity_guard_observed = any(
        comparison.get("decision") == "different"
        and comparison.get("comparison_kind")
        not in ("within_source_cluster", "within_cluster")
        and tuple(
            sorted(str(item) for item in comparison.get("speaker_pair", []))
        )
        in relevant_identity_pairs
        for item in identities
        for comparison in item.get("payload", {}).get("comparisons", [])
    )
    pair_checks = []
    for pair in independent_timeline_pairs:
        pair_ids = [item["evidence_id"] for item in pair]
        pair_event_agreement = _pair_event_agreement(
            pair_ids, comparisons, event_key, event_min
        )
        identity_witnesses = _identity_witnesses(
            all_evidence, identities, pair, claim_name
        )
        coverage_witnesses = _coverage_witnesses(
            all_evidence, coverage, pair
        )
        pair_checks.append(
            {
                "timeline_evidence_ids": pair_ids,
                "event_alignment": pair_event_agreement,
                "identity_evidence_ids": [
                    item["evidence_id"] for item in identity_witnesses
                ],
                "coverage_evidence_ids": [
                    item["evidence_id"] for item in coverage_witnesses
                ],
                "calibration_ready": _calibration_ready(pair),
                "joint_negative_ready": _joint_negative_ready(
                    pair, claim_name
                ),
            }
        )

    positive_witness = next(
        (
            item
            for item in pair_checks
            if item["event_alignment"]
            and item["identity_evidence_ids"]
            and item["coverage_evidence_ids"]
            and item["calibration_ready"]
        ),
        None,
    )
    negative_witness = next(
        (
            item
            for item in pair_checks
            if item["coverage_evidence_ids"]
            and item["calibration_ready"]
            and item["joint_negative_ready"]
        ),
        None,
    )
    identity_guard_calibrated = any(
        item["identity_evidence_ids"] for item in pair_checks
    )
    coverage_guard = any(item["coverage_evidence_ids"] for item in pair_checks)
    calibration_ready = any(item["calibration_ready"] for item in pair_checks)
    joint_negative_ready = any(
        item["joint_negative_ready"] for item in pair_checks
    )

    can_certify_true = candidate is True and positive_witness is not None
    can_certify_false = candidate is False and negative_witness is not None
    if can_certify_true or can_certify_false:
        status = "certified"
    elif status != "conflicted":
        reasons.append("one or more certification roles/calibration gates are missing")

    return {
        "status": status,
        "candidate_value": candidate,
        "public_value": candidate if status == "certified" else None,
        "observations": observations,
        "roles": {
            "independent_event_sources": len(independent_timeline_pairs) >= 1,
            "event_alignment": event_agreement,
            "identity_guard_observed": identity_guard_observed,
            "identity_guard_calibrated": identity_guard_calibrated,
            "coverage_guard": coverage_guard,
            "calibration_ready": calibration_ready,
            "joint_negative_ready": joint_negative_ready,
        },
        "certification_witness": (
            positive_witness if can_certify_true else negative_witness
            if can_certify_false
            else None
        ),
        "independent_pair_checks": pair_checks,
        "reason": "; ".join(reasons) or "all certification gates passed",
        "required_for_certification": [
            "two dependency-independent full-scope event sources",
            "event-local agreement for positive claims",
            "independent identity and speech-coverage guards",
            "frozen domain calibration",
            "joint-negative calibration for false claims",
        ],
        "claim": claim_name,
    }


def _usable_by_capability(evidence, capability):
    return [
        item
        for item in evidence
        if capability in item.get("capabilities", [])
        and item.get("status") in ("observed", "estimated")
        and item.get("quality", {}).get("usable", True)
    ]


def _validate_evidence_scope(sample_id, duration_sec, evidence):
    expected_duration = round(float(duration_sec), 6)
    audio_hashes = set()
    for item in evidence:
        validate_evidence(item)
        scope = item.get("scope", {})
        if str(scope.get("sample_id")) != str(sample_id):
            raise ValueError("cross-sample evidence is not allowed")
        if scope.get("level") != "utterance":
            raise ValueError("evidence scope must be utterance-level")
        if round(float(scope.get("start_sec", -1.0)), 6) != 0.0:
            raise ValueError("evidence must start at the current utterance")
        if round(float(scope.get("end_sec", -1.0)), 6) != expected_duration:
            raise ValueError("evidence duration does not match the utterance")
        audio_hash = item.get("applicability", {}).get("audio_sha256")
        if audio_hash:
            audio_hashes.add(str(audio_hash))
    if len(audio_hashes) > 1:
        raise ValueError("cross-audio evidence is not allowed")


def _timeline_observation(item):
    summary = item["payload"]["timeline_summary"]
    return {
        "evidence_id": item["evidence_id"],
        "source": item["source"],
        "dependency_groups": item["dependency_groups"],
        "observed_speaker_count": summary["observed_speaker_count"],
        "overlap_observed": summary["overlap_observed"],
        "change_observed": summary["change_observed"],
    }


def _calibration_ready(records):
    unique = {item["evidence_id"]: item for item in records}.values()
    return bool(unique) and all(
        item.get("quality", {}).get("calibration_profile_id")
        and item.get("quality", {}).get("counts_for_certification", True)
        for item in unique
    )


def _joint_negative_ready(timelines, claim_name):
    if any(
        not item.get("quality", {}).get("counts_for_certification", True)
        for item in timelines
    ):
        return False
    profile_ids = [
        item.get("quality", {}).get("joint_negative_profile_id")
        for item in timelines
    ]
    if not profile_ids or not all(profile_ids) or len(set(profile_ids)) != 1:
        return False
    return all(
        claim_name
        in item.get("quality", {}).get("joint_negative_claims", [])
        for item in timelines
    )


def _pair_event_agreement(pair_ids, comparisons, event_key, event_min):
    if event_key is None:
        return True
    pair_set = set(pair_ids)
    return any(
        set(item.get("evidence_ids", [])) == pair_set
        and item.get("comparison", {}).get(event_key, 0.0) >= event_min
        for item in (comparisons or [])
    )


def _identity_witnesses(
    all_evidence, identities, timeline_pair, claim_name
):
    relevant_pairs = set()
    for timeline in timeline_pair:
        relevant_pairs.update(
            _claim_speaker_pairs(
                timeline["payload"]["timeline_summary"], claim_name
            )
        )
    witnesses = []
    for item in identities:
        quality = item.get("quality", {})
        if not quality.get("calibration_profile_id"):
            continue
        if not quality.get("candidate_selection_independent", False):
            continue
        if not quality.get(
            "validator_dependency_closure_independent", False
        ):
            continue
        if not all(
            closure_independent(all_evidence, item, timeline)
            for timeline in timeline_pair
        ):
            continue
        if not any(
            comparison.get("decision") == "different"
            and comparison.get("comparison_kind")
            not in ("within_source_cluster", "within_cluster")
            and tuple(
                sorted(str(item) for item in comparison.get("speaker_pair", []))
            )
            in relevant_pairs
            for comparison in item.get("payload", {}).get("comparisons", [])
        ):
            continue
        witnesses.append(item)
    return witnesses


def _coverage_witnesses(all_evidence, coverage, timeline_pair):
    witnesses = []
    for item in coverage:
        quality = item.get("quality", {})
        if not quality.get("full_scope_invocation", False):
            continue
        if not item.get("payload", {}).get("speech_segments"):
            continue
        if not all(
            closure_independent(all_evidence, item, timeline)
            for timeline in timeline_pair
        ):
            continue
        witnesses.append(item)
    return witnesses


def _claim_speaker_pairs(summary, claim_name):
    if claim_name == "multi_speaker":
        speakers = summary.get("speaker_ids", [])
        return {
            tuple(sorted((str(speakers[left]), str(speakers[right]))))
            for left in range(len(speakers))
            for right in range(left + 1, len(speakers))
        }
    segments = summary.get("activity_segments") or summary.get("segments", [])
    pairs = set()
    if claim_name == "speaker_overlap":
        for left_index, left in enumerate(segments):
            for right in segments[left_index + 1 :]:
                if left["speaker_id"] == right["speaker_id"]:
                    continue
                overlap = min(left["end_sec"], right["end_sec"]) - max(
                    left["start_sec"], right["start_sec"]
                )
                if overlap > 0:
                    pairs.add(
                        tuple(
                            sorted(
                                (
                                    str(left["speaker_id"]),
                                    str(right["speaker_id"]),
                                )
                            )
                        )
                    )
    elif claim_name == "speaker_change":
        ordered = sorted(
            segments,
            key=lambda item: (
                item["start_sec"],
                item["end_sec"],
                item["speaker_id"],
            ),
        )
        for left, right in zip(ordered, ordered[1:]):
            if left["speaker_id"] == right["speaker_id"]:
                continue
            if float(right["start_sec"]) - float(left["end_sec"]) <= 1.0:
                pairs.add(
                    tuple(
                        sorted(
                            (
                                str(left["speaker_id"]),
                                str(right["speaker_id"]),
                            )
                        )
                    )
                )
    return pairs
