"""Deterministic, versioned counterfactual hypothesis templates."""

from tagger.tools.speaker_v2.contracts import closure_independent, stable_id


HYPOTHESIS_TEMPLATE_VERSION = "speaker_hypothesis_templates_v2.0-shadow.1"


def build_count_hypothesis_case(sample_id, timeline_records, all_evidence=None):
    """Freeze H1/H2/H_other before targeted identity acquisition.

    A case is emitted only for two usable, independent timeline sources whose
    observed speaker counts disagree. H1 and H2 follow source order; neither is
    assumed to be the truth.
    """

    usable = [
        item
        for item in timeline_records
        if item.get("status") in ("observed", "estimated")
        and item.get("quality", {}).get("usable", True)
        and item.get("payload", {}).get("timeline_summary")
    ]
    if len(usable) < 2:
        return None
    pair = next(
        (
            (left, right)
            for left_index, left in enumerate(usable)
            for right in usable[left_index + 1 :]
            if closure_independent(
                all_evidence or timeline_records, left, right
            )
            and left["payload"]["timeline_summary"][
                "observed_speaker_count"
            ]
            != right["payload"]["timeline_summary"][
                "observed_speaker_count"
            ]
        ),
        None,
    )
    if pair is None:
        return None
    source_a, source_b = pair
    groups_a = set(source_a.get("dependency_groups", []))
    groups_b = set(source_b.get("dependency_groups", []))
    count_a = source_a["payload"]["timeline_summary"][
        "observed_speaker_count"
    ]
    count_b = source_b["payload"]["timeline_summary"][
        "observed_speaker_count"
    ]
    if count_a == count_b:
        return None

    higher = source_a if count_a > count_b else source_b
    higher_speakers = higher["payload"]["timeline_summary"]["speaker_ids"]
    candidate_pairs = [
        [higher_speakers[left], higher_speakers[right]]
        for left in range(len(higher_speakers))
        for right in range(left + 1, len(higher_speakers))
    ]
    conflict = {
        "source_a_evidence_id": source_a["evidence_id"],
        "source_b_evidence_id": source_b["evidence_id"],
        "source_a_count": count_a,
        "source_b_count": count_b,
        "higher_count_source_evidence_id": higher["evidence_id"],
        "candidate_speaker_pairs": candidate_pairs,
    }
    case_identity = {
        "sample_id": sample_id,
        "claim": "speaker_count",
        "template_id": "count_mismatch_two_timelines",
        "template_version": HYPOTHESIS_TEMPLATE_VERSION,
        "conflict": conflict,
    }
    prohibited_groups = sorted(groups_a.union(groups_b))
    common_prediction = {
        "region": {"kind": "sample_local_clean_segments"},
        "required_capability": "speaker_identity_comparison",
        "allowed_validator_groups": ["G_identity_independent"],
        "prohibited_validator_groups": prohibited_groups,
        "falsifier_type": "calibrated_independent_falsifier",
    }
    branches = [
        {
            "hypothesis_id": "H1",
            "assumption": {
                "source_evidence_id": source_a["evidence_id"],
                "speaker_count": count_a,
            },
            "predictions": [
                dict(
                    common_prediction,
                    predicate=(
                        "higher-count source clusters follow its claimed "
                        "same/different identity partition"
                    ),
                )
            ],
            "status": "untested",
            "tests": [],
        },
        {
            "hypothesis_id": "H2",
            "assumption": {
                "source_evidence_id": source_b["evidence_id"],
                "speaker_count": count_b,
            },
            "predictions": [
                dict(
                    common_prediction,
                    predicate=(
                        "at least one extra cluster from the higher-count "
                        "source is the same speaker"
                    ),
                )
            ],
            "status": "untested",
            "tests": [],
        },
        {
            "hypothesis_id": "H_other",
            "assumption": {
                "description": (
                    "both timelines are incomplete or another failure mode "
                    "explains the mismatch"
                )
            },
            "predictions": [
                {
                    "predicate": (
                        "identity checks do not produce a complete, calibrated "
                        "partition consistent with either timeline"
                    ),
                    "region": {"kind": "sample_local_clean_segments"},
                    "required_capability": "speaker_identity_comparison",
                    "allowed_validator_groups": ["G_identity_independent"],
                    "prohibited_validator_groups": prohibited_groups,
                    "falsifier_type": "diagnostic_mismatch",
                }
            ],
            "status": "viable",
            "tests": [],
        },
    ]
    case = dict(case_identity)
    case.update(
        {
            "case_id": stable_id("hyp", case_identity),
            "scope": {"sample_id": sample_id, "level": "utterance"},
            "branches": branches,
            "acquisition_plan": [
                {
                    "action": "campplus_clean_segment_pair_matrix",
                    "source_evidence_id": higher["evidence_id"],
                    "candidate_speaker_pairs": candidate_pairs,
                    "frozen_before_acquisition": True,
                }
            ],
            "certification_effect": "none",
            "termination_reason": "awaiting_targeted_identity_evidence",
        }
    )
    return case


def evaluate_count_hypothesis_case(case, identity_records):
    """Attach independent tests without turning hypothesis survival into truth."""

    if case is None:
        return None
    evaluated = _deep_copy(case)
    prohibited = set()
    for branch in evaluated["branches"]:
        for prediction in branch.get("predictions", []):
            prohibited.update(prediction.get("prohibited_validator_groups", []))
    usable = []
    for record in identity_records:
        if record.get("status") not in ("observed", "estimated"):
            continue
        if "speaker_identity_comparison" not in record.get("capabilities", []):
            continue
        usable.append(record)

    decisions = []
    source_evidence_ids = {
        evaluated["conflict"]["source_a_evidence_id"],
        evaluated["conflict"]["source_b_evidence_id"],
    }
    for record in usable:
        quality = record.get("quality", {})
        falsifier_eligible = bool(quality.get("calibration_profile_id"))
        falsifier_eligible = falsifier_eligible and bool(
            quality.get("candidate_selection_independent", False)
        )
        falsifier_eligible = falsifier_eligible and bool(
            quality.get("validator_dependency_closure_independent", False)
        )
        falsifier_eligible = falsifier_eligible and not bool(
            prohibited.intersection(record.get("dependency_groups", []))
        )
        falsifier_eligible = falsifier_eligible and not bool(
            source_evidence_ids.intersection(
                record.get("lineage", {}).get("parent_evidence_ids", [])
            )
        )
        for comparison in record.get("payload", {}).get("comparisons", []):
            pair = comparison.get("speaker_pair") or []
            normalized_pair = tuple(sorted(str(item) for item in pair))
            candidate_pairs = {
                tuple(sorted(str(item) for item in item_pair))
                for item_pair in evaluated["conflict"]["candidate_speaker_pairs"]
            }
            if normalized_pair not in candidate_pairs:
                continue
            if comparison.get("comparison_kind") in (
                "within_source_cluster",
                "within_cluster",
            ):
                continue
            decision = comparison.get("decision")
            if decision in ("same", "different"):
                decisions.append(
                    {
                        "evidence_id": record["evidence_id"],
                        "speaker_pair": comparison.get("speaker_pair"),
                        "decision": decision,
                        "score": comparison.get("score"),
                        "falsifier_eligible": falsifier_eligible,
                    }
                )

    for branch in evaluated["branches"]:
        branch["tests"] = decisions
    if not decisions:
        evaluated["termination_reason"] = "identity_evidence_missing_or_dependent"
        return evaluated

    eligible_decisions = [
        item for item in decisions if item["falsifier_eligible"]
    ]
    has_same = any(item["decision"] == "same" for item in eligible_decisions)
    has_different = any(
        item["decision"] == "different" for item in eligible_decisions
    )
    by_id = {item["hypothesis_id"]: item for item in evaluated["branches"]}
    if has_same:
        # The higher-count exact partition cannot survive a calibrated same-ID
        # result for one of its supposedly distinct clusters.
        source_a = evaluated["conflict"]["source_a_evidence_id"]
        source_b = evaluated["conflict"]["source_b_evidence_id"]
        count_a = evaluated["conflict"]["source_a_count"]
        count_b = evaluated["conflict"]["source_b_count"]
        higher_hypothesis = "H1" if count_a > count_b else "H2"
        by_id[higher_hypothesis]["status"] = "falsified"
    if has_different:
        # A lower-count hypothesis is falsified only when its count is one and
        # the tested pair is demonstrably different. More complex partitions
        # require complete pair coverage and stay viable here.
        count_a = evaluated["conflict"]["source_a_count"]
        count_b = evaluated["conflict"]["source_b_count"]
        if min(count_a, count_b) == 1:
            lower_hypothesis = "H1" if count_a < count_b else "H2"
            by_id[lower_hypothesis]["status"] = "falsified"
    for hypothesis_id in ("H1", "H2"):
        if by_id[hypothesis_id]["status"] == "untested":
            by_id[hypothesis_id]["status"] = "viable"

    survivors = [
        item["hypothesis_id"]
        for item in evaluated["branches"]
        if item["status"] == "viable"
    ]
    evaluated["termination_reason"] = (
        "calibrated_test_complete"
        if eligible_decisions
        else "uncalibrated_identity_diagnostic"
    )
    # H_other deliberately remains viable until an independent complete
    # explanation exists. Hypothesis tests therefore do not certify a claim.
    if len(survivors) < len(evaluated["branches"]):
        evaluated["certification_effect"] = "reenter_resolver"
    return evaluated


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value
