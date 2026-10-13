"""Claim-aware resolver layered on the frozen legacy claim logic."""

import copy
import math
import re

from tagger.tools.speaker_v2._legacy import load_legacy_module
from tagger.tools.speaker_v2.contracts import stable_id
from tagger.tools.speaker_v2.profiles import (
    ClaimPolicyError,
    NEMOTRON_SOURCE,
    MOSS_SOURCE,
    claim_policy_hash,
    validate_claim_policy,
)


_LEGACY = load_legacy_module("resolver")
FUSION_SCHEMA_VERSION = _LEGACY.FUSION_SCHEMA_VERSION
ALLOWED_CLAIM_STATUS = _LEGACY.ALLOWED_CLAIM_STATUS
EVALUATION_OUTPUT_SCHEMA_VERSION = "speaker_v2.evaluation_output.v2"
_UNSET = object()


def resolve(
    sample_id,
    duration_sec,
    evidence,
    hypotheses=None,
    claim_policy=_UNSET,
    profile_id="legacy-shadow",
    speaker_profiles=_UNSET,
    language_route=None,
):
    """Resolve speaker claims using a versioned per-claim source policy.

    Omitting ``claim_policy`` is a compatibility shim for older direct callers
    and returns the frozen resolver output byte-for-byte.  The v2 pipeline must
    always pass an expanded policy explicitly; explicit ``None`` fails closed.
    Guards are recorded as diagnostics in this first shadow release and never
    silently become equal decision votes.
    """

    if claim_policy is _UNSET:
        return _LEGACY.resolve(
            sample_id,
            duration_sec,
            evidence,
            hypotheses=hypotheses,
        )
    if claim_policy is None:
        raise ClaimPolicyError("claim_policy cannot be null")
    validate_claim_policy(claim_policy)
    language_selected_source = _language_selected_source(
        claim_policy, language_route
    )
    if speaker_profiles is _UNSET:
        speaker_profiles = _profiles_from_evidence(evidence)
    if speaker_profiles is not _UNSET:
        speaker_profiles = _normalize_speaker_profiles(speaker_profiles)
    profile_id = str(profile_id)

    # This preserves all raw observations/comparisons and performs the frozen
    # scope, lineage, finite-number and cross-audio validation once.
    fusion = _LEGACY.resolve(
        sample_id,
        duration_sec,
        evidence,
        hypotheses=hypotheses,
    )
    routed_claims = {}
    for claim_name, rule in claim_policy["claims"].items():
        routed_claims[claim_name] = _resolve_one_claim(
            claim_name,
            rule,
            sample_id,
            duration_sec,
            evidence,
            hypotheses,
            claim_policy,
            language_selected_source,
        )
    fusion["claims"] = routed_claims
    fusion["run_profile"] = profile_id
    fusion["claim_policy"] = copy.deepcopy(claim_policy)
    fusion["policy_version"] = str(claim_policy["policy_version"])
    fusion["policy_hash"] = claim_policy_hash(claim_policy)
    if language_route is not None:
        fusion["language_route"] = copy.deepcopy(language_route)
    fusion["derived_metrics"] = _derive_numeric_metrics(
        fusion["claims"], evidence
    )
    fusion["evaluation_output"] = build_evaluation_output(
        fusion["claims"],
        fusion["derived_metrics"],
        speaker_profiles=speaker_profiles,
    )
    if speaker_profiles is not _UNSET:
        fusion["speaker_profiles"] = copy.deepcopy(speaker_profiles)
    _publish_public_adapter(fusion, fusion["evaluation_output"]["speaker"])
    identity = {
        "schema_version": fusion["schema_version"],
        "sample_id": fusion["sample_id"],
        "scope": fusion["scope"],
        "profile": fusion["profile"],
        "run_profile": fusion["run_profile"],
        "policy_version": fusion["policy_version"],
        "policy_hash": fusion["policy_hash"],
        "derived_metrics": fusion["derived_metrics"],
        "evaluation_output": fusion["evaluation_output"],
        "evidence_ids": fusion["evidence_ids"],
        "claims": fusion["claims"],
        "hypothesis_case_ids": fusion["hypothesis_case_ids"],
    }
    fusion["fusion_id"] = stable_id("fusion", identity)
    validate_fusion(fusion)
    return fusion


def validate_fusion(fusion):
    legacy_view = copy.deepcopy(fusion)
    legacy_adapter = legacy_view.setdefault("public_adapter", {})
    legacy_adapter["enabled"] = False
    legacy_adapter["speaker"] = {
        field: None
        for field in (
            "speaker_count",
            "multi_speaker",
            "speaker_change_count",
            "speaker_change",
            "overlap_ratio",
            "speaker_overlap",
        )
    }
    _LEGACY.validate_fusion(legacy_view)
    if "run_profile" in fusion:
        policy = fusion.get("claim_policy")
        validate_claim_policy(policy)
        if fusion.get("policy_version") != policy.get("policy_version"):
            raise ValueError("fusion policy version mismatch")
        if fusion.get("policy_hash") != claim_policy_hash(policy):
            raise ValueError("fusion policy hash mismatch")
        expected_output = build_evaluation_output(
            fusion.get("claims", {}),
            fusion.get("derived_metrics", {}),
            speaker_profiles=fusion.get("speaker_profiles", _UNSET),
        )
        if fusion.get("evaluation_output") != expected_output:
            raise ValueError("evaluation output does not match resolved claims")
        public_adapter = fusion.get("public_adapter", {})
        if public_adapter.get("enabled") is not True:
            raise ValueError("public adapter must be enabled")
        if public_adapter.get("speaker") != expected_output["speaker"]:
            raise ValueError("public adapter does not match resolved speaker output")


def build_evaluation_output(
    claims, derived_metrics=None, speaker_profiles=_UNSET
):
    derived_metrics = derived_metrics or {}
    claim_outputs = {}
    for claim_name in (
        "speaker_count",
        "multi_speaker",
        "speaker_overlap",
        "speaker_change",
    ):
        claim = claims.get(claim_name, {})
        if claim_name == "speaker_count":
            claim_outputs[claim_name] = _count_output(claim)
        else:
            claim_outputs[claim_name] = _boolean_output(claim)

    metric_outputs = {
        metric_name: _derived_metric_output(derived_metrics.get(metric_name, {}))
        for metric_name in ("speaker_change_count", "overlap_ratio")
    }

    speaker = {
        "speaker_count": claim_outputs["speaker_count"]["value"],
        "multi_speaker": claim_outputs["multi_speaker"]["value"],
        "speaker_change_count": metric_outputs["speaker_change_count"][
            "value"
        ],
        "speaker_change": claim_outputs["speaker_change"]["value"],
        "overlap_ratio": metric_outputs["overlap_ratio"]["value"],
        "speaker_overlap": claim_outputs["speaker_overlap"]["value"],
    }
    if speaker_profiles is not _UNSET:
        speaker["profiles"] = copy.deepcopy(speaker_profiles)
    return {
        "schema_version": EVALUATION_OUTPUT_SCHEMA_VERSION,
        "artifact_purpose": "speaker_metadata",
        "mode": "direct",
        "production_eligible": True,
        "public_metadata_published": True,
        "speaker_count": claim_outputs["speaker_count"]["value"],
        "speaker_change_count": metric_outputs["speaker_change_count"]["value"],
        "overlap_ratio": metric_outputs["overlap_ratio"]["value"],
        "speaker": speaker,
        "claims": claim_outputs,
        "metrics": metric_outputs,
    }


def _count_output(claim):
    status = claim.get("status")
    value = _count_candidate(claim)
    return _output_record(claim, status, value, "observed_values")


def _boolean_output(claim):
    status = claim.get("status")
    value = claim.get("candidate_value")
    if status in ("conflicted", "insufficient") or not isinstance(value, bool):
        value = None
    return _output_record(claim, status, value, "candidate_value")


def _count_candidate(claim):
    if claim.get("status") in ("conflicted", "insufficient"):
        return None
    values = [
        item.get("value")
        for item in claim.get("observed_values", [])
        if isinstance(item, dict)
    ]
    if not values or any(isinstance(value, bool) for value in values):
        return None
    if not all(isinstance(value, int) and value >= 0 for value in values):
        return None
    return values[0] if len(set(values)) == 1 else None


def _output_record(claim, status, value, source_field):
    if value is not None:
        availability = "available"
    elif status == "conflicted":
        availability = "conflicted"
    elif status == "insufficient":
        availability = "missing"
    else:
        availability = "missing_candidate"
    route = claim.get("route", {})
    return {
        "value": value,
        "availability": availability,
        "source_field": source_field,
        "source_claim_status": status,
        "decision_sources": list(route.get("decision_sources", [])),
        "decision_evidence_ids": list(
            route.get("decision_evidence_ids", [])
        ),
    }


def _derived_metric_output(metric):
    status = metric.get("status")
    candidate = metric.get("candidate_value")
    value = candidate if status == "supported" else None
    if value is not None:
        availability = "available"
    elif status == "conflicted":
        availability = "conflicted"
    elif status == "insufficient":
        availability = "missing"
    else:
        availability = "missing_candidate"
    return {
        "value": value,
        "availability": availability,
        "source_field": "candidate_value",
        "source_claim": metric.get("source_claim"),
        "source_claim_status": metric.get("source_claim_status"),
        "decision_sources": list(metric.get("decision_sources", [])),
        "decision_evidence_ids": list(
            metric.get("decision_evidence_ids", [])
        ),
        "observations": copy.deepcopy(metric.get("observations", [])),
    }


def _publish_public_adapter(fusion, speaker_output):
    public_adapter = fusion.setdefault("public_adapter", {})
    public_adapter["enabled"] = True
    public_adapter["reason"] = "direct claim-policy resolver output"
    public_adapter["speaker"] = copy.deepcopy(speaker_output)


def _profiles_from_evidence(evidence):
    candidates = [
        item
        for item in evidence
        if "speaker_profile" in item.get("capabilities", [])
        and item.get("status") in ("observed", "estimated", "missing")
    ]
    if not candidates:
        return _UNSET
    return copy.deepcopy(candidates[0].get("payload", {}).get("profiles"))


def _normalize_speaker_profiles(value):
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    normalized = []
    seen_ids = set()
    for item in value:
        if not isinstance(item, dict) or not re.match(
            r"^speaker_[1-9][0-9]*$", str(item.get("speaker_id", ""))
        ):
            return None
        speaker_id = str(item["speaker_id"])
        if speaker_id in seen_ids:
            return None
        seen_ids.add(speaker_id)
        rate = item.get("speech_rate")
        if not isinstance(rate, dict):
            return None
        band = rate.get("band")
        unit = rate.get("unit")
        number = rate.get("value")
        if band not in (None, "slow", "normal", "fast", "variable"):
            return None
        if unit not in (None, "zh_char_per_sec", "word_per_min"):
            return None
        if number is not None and (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            return None
        if unit is None and number is not None:
            return None
        if item.get("pitch") not in (None, "low", "mid", "high", "variable"):
            return None
        if item.get("speaker_volume") not in (
            None,
            "low",
            "normal",
            "loud",
            "variable",
        ):
            return None
        if item.get("gender") not in (None, "male", "female"):
            return None
        normalized.append(
            {
                "speaker_id": str(item["speaker_id"]),
                "gender": item.get("gender"),
                "speech_rate": {
                    "band": band,
                    "value": number,
                    "unit": unit,
                },
                "pitch": item.get("pitch"),
                "speaker_volume": item.get("speaker_volume"),
            }
        )
    return normalized


def _derive_numeric_metrics(claims, evidence):
    evidence_by_id = {
        item.get("evidence_id"): item
        for item in evidence
        if item.get("evidence_id")
    }
    definitions = (
        (
            "speaker_change_count",
            "speaker_change",
            _change_count_from_summary,
        ),
        (
            "overlap_ratio",
            "speaker_overlap",
            _overlap_ratio_from_summary,
        ),
    )
    return {
        metric_name: _derive_numeric_metric(
            metric_name,
            claim_name,
            claims.get(claim_name, {}),
            evidence_by_id,
            extractor,
        )
        for metric_name, claim_name, extractor in definitions
    }


def _derive_numeric_metric(
    metric_name, claim_name, claim, evidence_by_id, extractor
):
    route = claim.get("route", {})
    decision_ids = list(route.get("decision_evidence_ids", []))
    observations = []
    for evidence_id in decision_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        summary = item.get("payload", {}).get("timeline_summary", {})
        value = extractor(summary)
        observations.append(
            {
                "evidence_id": evidence_id,
                "source": item.get("source", {}).get("name"),
                "value": value,
            }
        )
    values = [item["value"] for item in observations]
    if not values or any(value is None for value in values):
        status = "insufficient"
        candidate = None
    elif len(set(values)) > 1:
        status = "conflicted"
        candidate = None
    else:
        status = "supported"
        candidate = values[0]
    return {
        "metric": metric_name,
        "source_claim": claim_name,
        "source_claim_status": claim.get("status"),
        "status": status,
        "candidate_value": candidate,
        "observations": observations,
        "decision_sources": list(route.get("decision_sources", [])),
        "decision_evidence_ids": decision_ids,
    }


def _change_count_from_summary(summary):
    points = summary.get("change_candidate_points_sec")
    if not isinstance(points, list):
        return None
    return len(points)


def _overlap_ratio_from_summary(summary):
    overlap_duration = summary.get("overlap_duration_sec")
    speech_duration = summary.get("speech_union_duration_sec")
    if (
        isinstance(overlap_duration, bool)
        or isinstance(speech_duration, bool)
        or not isinstance(overlap_duration, (int, float))
        or not isinstance(speech_duration, (int, float))
        or speech_duration <= 0
    ):
        return None
    ratio = float(overlap_duration) / float(speech_duration)
    if ratio < 0.0 or ratio > 1.0:
        return None
    return round(ratio, 6)


def _resolve_one_claim(
    claim_name,
    rule,
    sample_id,
    duration_sec,
    evidence,
    hypotheses,
    policy,
    language_selected_source,
):
    usable = _usable_timeline_by_source(evidence)
    if rule["selection"] == "language_routed":
        primary = _ordered_available([language_selected_source], usable)
        selected = primary[:1]
        selection = "language_route" if selected else "none"
        fallback_reason = (
            None
            if selected
            else "selected language-route source is unavailable"
        )
        routed_sources = {language_selected_source}
    else:
        primary = _ordered_available(rule["primary_sources"], usable)
        if primary:
            selected = (
                primary if rule["selection"] == "all_usable" else primary[:1]
            )
            selection = "primary"
            fallback_reason = None
        else:
            selected = _ordered_available(rule["fallback_sources"], usable)[:1]
            selection = "fallback" if selected else "none"
            fallback_reason = "no usable primary source"
        routed_sources = set(rule["primary_sources"]) | set(
            rule["fallback_sources"]
        )

    selected_ids = {item["evidence_id"] for item in selected}
    routed_evidence = [
        item
        for item in evidence
        if "speaker_timeline" not in item.get("capabilities", [])
        or item.get("evidence_id") in selected_ids
    ]
    resolved = _LEGACY.resolve(
        sample_id,
        duration_sec,
        routed_evidence,
        hypotheses=hypotheses,
    )["claims"][claim_name]
    resolved = copy.deepcopy(resolved)
    guards = _ordered_available(rule["guard_sources"], usable)
    excluded = _ordered_available(rule["excluded_sources"], usable)
    all_decision_sources = {item["source"]["name"] for item in selected}
    diagnostics = [
        item
        for source, items in usable.items()
        for item in items
        if source not in all_decision_sources
    ]
    resolved["route"] = {
        "policy_version": str(policy["policy_version"]),
        "selection": selection,
        "decision_evidence_ids": [item["evidence_id"] for item in selected],
        "decision_sources": [item["source"]["name"] for item in selected],
        "guard_observations": [
            _route_observation(claim_name, item, rule.get("guard_rules", {}))
            for item in guards
        ],
        "excluded_observations": [
            _route_observation(claim_name, item, {}) for item in excluded
        ],
        "diagnostic_evidence_ids": sorted(
            item["evidence_id"] for item in diagnostics
        ),
        "configured_primary_sources": list(rule["primary_sources"]),
        "configured_fallback_sources": list(rule["fallback_sources"]),
        "configured_guard_sources": list(rule["guard_sources"]),
        "configured_excluded_sources": list(rule["excluded_sources"]),
        "language_selected_source": (
            language_selected_source
            if rule["selection"] == "language_routed"
            else None
        ),
        "unavailable_configured_sources": sorted(
            source for source in routed_sources if source not in usable
        ),
        "fallback_reason": fallback_reason,
        "guards_affect_candidate": False,
    }
    return resolved


def _language_selected_source(policy, language_route):
    requires_route = any(
        rule.get("selection") == "language_routed"
        for rule in policy["claims"].values()
    )
    if not requires_route:
        return None
    if not isinstance(language_route, dict):
        raise ClaimPolicyError(
            "language_routed claim policy requires a language_route object"
        )
    route_name = language_route.get("language_route")
    if route_name in ("chinese", "english"):
        return MOSS_SOURCE
    if route_name == "non_chinese_or_unknown":
        return NEMOTRON_SOURCE
    raise ClaimPolicyError("unsupported language route: %s" % route_name)


def _usable_timeline_by_source(evidence):
    result = {}
    for item in evidence:
        if "speaker_timeline" not in item.get("capabilities", []):
            continue
        if item.get("status") not in ("observed", "estimated"):
            continue
        if not item.get("quality", {}).get("usable", True):
            continue
        result.setdefault(item["source"]["name"], []).append(item)
    return result


def _ordered_available(source_names, usable):
    return [item for source in source_names for item in usable.get(source, [])]


def _route_observation(claim_name, item, guard_rules):
    summary = item["payload"]["timeline_summary"]
    if claim_name == "speaker_count":
        value = int(summary["observed_speaker_count"])
    elif claim_name == "multi_speaker":
        value = int(summary["observed_speaker_count"]) >= 2
    elif claim_name == "speaker_overlap":
        value = bool(summary["overlap_segments"])
    elif claim_name == "speaker_change":
        value = bool(summary["change_candidate_points_sec"])
    else:
        raise ClaimPolicyError("unsupported routed claim: %s" % claim_name)
    source = item["source"]["name"]
    return {
        "evidence_id": item["evidence_id"],
        "source": source,
        "value": value,
        "rule": guard_rules.get(source, "diagnostic_only"),
    }
