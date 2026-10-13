"""Typed and deterministic contracts for speaker-v2 evidence."""

import hashlib
import json
import math


EVIDENCE_SCHEMA_VERSION = "speaker_evidence_v2.0-shadow.1"
ALLOWED_EVIDENCE_STATUS = set(["observed", "estimated", "missing", "failed"])


class EvidenceContractError(ValueError):
    """Raised when an evidence record violates the v2 contract."""


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_id(prefix, value):
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return "%s_%s" % (prefix, digest[:24])


def build_evidence(
    sample_id,
    duration_sec,
    evidence_type,
    source_name,
    source_version,
    source_kind,
    capabilities,
    dependency_groups,
    payload,
    status="estimated",
    quality=None,
    lineage=None,
    runtime=None,
    applicability=None,
):
    """Build one immutable sample-local evidence record.

    The evidence ID excludes volatile run time and wall-clock fields, but
    includes status and quality because calibration/applicability gates can
    change fusion. Repeating the same model/config on the same payload and
    certification metadata therefore yields the same ID and never creates an
    additional independent vote.
    """

    _validate_finite(payload, "payload")
    scope = {
        "sample_id": str(sample_id),
        "start_sec": 0.0,
        "end_sec": _positive_duration(duration_sec),
        "level": "utterance",
    }
    identity = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_type": str(evidence_type),
        "scope": scope,
        "source": {
            "name": str(source_name),
            "version": str(source_version),
            "kind": str(source_kind),
        },
        "capabilities": sorted(set(str(item) for item in capabilities)),
        "dependency_groups": sorted(
            set(str(item) for item in dependency_groups)
        ),
        "lineage": _normalize_lineage(lineage),
        "applicability": dict(applicability or {}),
        "payload": payload,
        "status": str(status),
        "quality": dict(quality or {}),
    }
    record = dict(identity)
    record["evidence_id"] = stable_id("ev", identity)
    record["runtime"] = dict(runtime or {})
    validate_evidence(record)
    return record


def build_missing_evidence(
    sample_id,
    duration_sec,
    evidence_type,
    source_name,
    source_version,
    source_kind,
    capabilities,
    dependency_groups,
    reason,
    runtime=None,
    applicability=None,
):
    return build_evidence(
        sample_id=sample_id,
        duration_sec=duration_sec,
        evidence_type=evidence_type,
        source_name=source_name,
        source_version=source_version,
        source_kind=source_kind,
        capabilities=capabilities,
        dependency_groups=dependency_groups,
        payload={},
        status="missing",
        quality={"usable": False, "reason": str(reason)},
        runtime=runtime,
        applicability=applicability,
    )


def validate_evidence(record):
    if not isinstance(record, dict):
        raise EvidenceContractError("evidence must be an object")
    required = set(
        [
            "schema_version",
            "evidence_id",
            "evidence_type",
            "scope",
            "source",
            "capabilities",
            "dependency_groups",
            "lineage",
            "applicability",
            "payload",
            "status",
            "quality",
            "runtime",
        ]
    )
    missing = sorted(required - set(record))
    if missing:
        raise EvidenceContractError(
            "evidence missing fields: %s" % ", ".join(missing)
        )
    if record["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceContractError("unsupported evidence schema version")
    if record["status"] not in ALLOWED_EVIDENCE_STATUS:
        raise EvidenceContractError("unsupported evidence status")
    scope = record["scope"]
    if not isinstance(scope, dict) or scope.get("level") != "utterance":
        raise EvidenceContractError("scope must be utterance-level")
    duration = _positive_duration(scope.get("end_sec"))
    if float(scope.get("start_sec", -1)) != 0.0 or duration <= 0:
        raise EvidenceContractError("scope must cover the current sample")
    groups = record["dependency_groups"]
    if not isinstance(groups, list) or not groups:
        raise EvidenceContractError("dependency_groups must be a non-empty list")
    if len(groups) != len(set(groups)):
        raise EvidenceContractError("dependency_groups must be unique")
    capabilities = record["capabilities"]
    if not isinstance(capabilities, list):
        raise EvidenceContractError("capabilities must be a list")
    _validate_finite(record["payload"], "payload")
    expected = dict(record)
    evidence_id = expected.pop("evidence_id")
    expected.pop("runtime")
    if evidence_id != stable_id("ev", expected):
        raise EvidenceContractError("evidence_id does not match content")


def dependency_closure(records, evidence_ids):
    """Return transitive evidence IDs and groups for the requested records."""

    by_id = {item["evidence_id"]: item for item in records}
    pending = list(evidence_ids)
    seen = set()
    groups = set()
    while pending:
        evidence_id = pending.pop()
        if evidence_id in seen:
            continue
        item = by_id.get(evidence_id)
        if item is None:
            raise EvidenceContractError(
                "unknown lineage evidence_id: %s" % evidence_id
            )
        seen.add(evidence_id)
        groups.update(item.get("dependency_groups", []))
        pending.extend(item.get("lineage", {}).get("parent_evidence_ids", []))
    return {
        "evidence_ids": sorted(seen),
        "dependency_groups": sorted(groups),
    }


def independent(record_a, record_b):
    """Return direct dependency-group independence.

    This helper is retained for callers that only have two standalone records.
    Fusion and certification code must use :func:`closure_independent` so that
    parent evidence cannot be hidden behind a derived model output.
    """

    return not set(record_a.get("dependency_groups", [])).intersection(
        record_b.get("dependency_groups", [])
    )


def closure_independent(records, record_a, record_b):
    """Conservatively compare complete evidence dependency closures.

    Missing lineage records make independence unprovable and therefore return
    ``False``. A descendant also cannot be independent of its ancestor even if
    the two records declare different direct model groups.
    """

    try:
        closure_a = dependency_closure(records, [record_a["evidence_id"]])
        closure_b = dependency_closure(records, [record_b["evidence_id"]])
    except (EvidenceContractError, KeyError, TypeError):
        return False
    if set(closure_a["evidence_ids"]).intersection(closure_b["evidence_ids"]):
        return False
    return not set(closure_a["dependency_groups"]).intersection(
        closure_b["dependency_groups"]
    )


def _normalize_lineage(lineage):
    lineage = dict(lineage or {})
    return {
        "parent_evidence_ids": sorted(
            set(str(item) for item in lineage.get("parent_evidence_ids", []))
        ),
        "derived_audio_ids": sorted(
            set(str(item) for item in lineage.get("derived_audio_ids", []))
        ),
    }


def _positive_duration(value):
    if isinstance(value, bool):
        raise EvidenceContractError("duration must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise EvidenceContractError("duration must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise EvidenceContractError("duration must be positive and finite")
    return round(value, 6)


def _validate_finite(value, path):
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceContractError("%s contains a non-finite number" % path)
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, "%s[%s]" % (path, index))
