"""Policy-aware artifact compatibility layer for speaker-v2.

All filesystem layout and atomic-write behavior comes from the frozen
``speaker_v1/artifacts.py`` snapshot.  This layer only adds the versioned run
profile, expanded claim policy, and complete model execution inventory.
Old callers that do not provide profile fields are recorded as
``legacy-shadow`` and keep the original call signatures and artifact paths.
FireRed ASR, Qwen3 ASR, and official DiariZen have independent inventory
entries because each uses its own runtime and model assets.
"""

import copy
import gzip
import json
from pathlib import Path

from tagger.tools.speaker_v2._legacy import load_legacy_module
from tagger.tools.speaker_v2.profiles import (
    MODEL_KEYS,
    claim_policy_hash,
    expand_profile,
    validate_claim_policy,
)
from tagger.tools.speaker_v2.speaker_profile import PROFILE_SCHEMA_VERSION


_LEGACY = load_legacy_module("artifacts")

# Public helpers retain the frozen implementations and behavior.
write_json_gz_atomic = _LEGACY.write_json_gz_atomic
safe_stem = _LEGACY.safe_stem
_artifact_id = _LEGACY._artifact_id
_write_json_atomic = _LEGACY._write_json_atomic
_fsync_directory = _LEGACY._fsync_directory


def write_sample_artifacts(
    output_dir,
    sample_id,
    evidence,
    fusion,
    artifact_root=None,
):
    """Write legacy-compatible sample artifacts with policy provenance."""

    enriched_fusion = _enrich_fusion_policy(fusion)
    paths = _LEGACY.write_sample_artifacts(
        output_dir,
        sample_id,
        evidence,
        enriched_fusion,
        artifact_root=artifact_root,
    )

    # Keep the historical artifact path while storing the direct publication
    # output and its policy provenance.
    certification = {
        "schema_version": enriched_fusion["schema_version"],
        "sample_id": enriched_fusion["sample_id"],
        "fusion_id": enriched_fusion["fusion_id"],
        "profile": enriched_fusion["profile"],
        "run_profile": enriched_fusion["run_profile"],
        "policy_version": enriched_fusion["policy_version"],
        "policy_hash": enriched_fusion["policy_hash"],
        "speaker_profile_schema_version": PROFILE_SCHEMA_VERSION,
        "claim_policy": copy.deepcopy(enriched_fusion["claim_policy"]),
        "claims": enriched_fusion["claims"],
        "public_adapter": enriched_fusion["public_adapter"],
    }
    write_json_gz_atomic(paths["certification_artifact"], certification)
    public_speaker = copy.deepcopy(
        enriched_fusion.get("public_adapter", {}).get("speaker", {})
    )
    compat_metadata = {
        "schema_version": "speaker_v2.compat_metadata.2",
        "speaker_profile_schema_version": PROFILE_SCHEMA_VERSION,
        "sample_id": enriched_fusion["sample_id"],
        "profile": enriched_fusion["profile"],
        "speaker": public_speaker,
        "speaker_count": public_speaker.get("speaker_count"),
        "fusion_id": enriched_fusion["fusion_id"],
        "published": bool(
            enriched_fusion.get("public_adapter", {}).get("enabled")
        ),
    }
    write_json_gz_atomic(paths["compat_metadata"], compat_metadata)
    evaluation_output = enriched_fusion.get("evaluation_output")
    if isinstance(evaluation_output, dict):
        evaluation_artifact = {
            "schema_version": evaluation_output.get("schema_version"),
            "sample_id": enriched_fusion["sample_id"],
            "fusion_id": enriched_fusion["fusion_id"],
            "run_profile": enriched_fusion["run_profile"],
            "policy_version": enriched_fusion["policy_version"],
            "policy_hash": enriched_fusion["policy_hash"],
            "speaker_profile_schema_version": PROFILE_SCHEMA_VERSION,
            "output": copy.deepcopy(evaluation_output),
        }
        evaluation_path = write_json_gz_atomic(
            Path(paths["sample_dir"]) / "evaluation_output.json.gz",
            evaluation_artifact,
        )
        paths["evaluation_output_artifact"] = str(evaluation_path)
    return paths


def write_run_manifest(
    output_dir,
    input_manifest,
    input_manifest_sha256,
    config,
    summary,
    sample_ids=None,
    max_samples=None,
    fail_fast=False,
    resume=False,
):
    """Write the frozen run manifest plus policy and model inventory."""

    path = _LEGACY.write_run_manifest(
        output_dir,
        input_manifest,
        input_manifest_sha256,
        config,
        summary,
        sample_ids=sample_ids,
        max_samples=max_samples,
        fail_fast=fail_fast,
        resume=resume,
    )
    with Path(path).open("r", encoding="utf-8") as source:
        value = json.load(source)

    expanded = _expanded_run_profile(config, summary)
    policy = copy.deepcopy(expanded["claim_policy"])
    value["run_profile"] = expanded["profile_id"]
    value["policy_version"] = policy["policy_version"]
    value["policy_hash"] = policy["policy_hash"]
    value["claim_policy"] = policy
    value["profile_model_defaults"] = copy.deepcopy(
        expanded["profile_model_defaults"]
    )
    value["public_adapter_enabled"] = True
    value["evaluation_output"] = {
        "enabled": True,
        "mode": "direct",
        "artifact_purpose": "speaker_metadata",
        "production_eligible": True,
        "public_adapter_enabled": True,
        "speaker_profile": {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "model_required": bool(getattr(config, "enable_gender", True)),
            "enabled": bool(getattr(config, "enable_speaker_profile", True)),
            "evidence_sources": [
                "speaker_profile_deterministic",
                "speaker_gender_language_routed",
            ],
        },
        "speaker_fields": [
            "speaker_count",
            "multi_speaker",
            "speaker_change_count",
            "speaker_change",
            "overlap_ratio",
            "speaker_overlap",
            "profiles",
            "asr_transcript",
        ],
    }
    value["models"] = _model_inventory(config, expanded)
    _write_json_atomic(path, value)
    return path


def _enrich_fusion_policy(fusion):
    result = copy.deepcopy(fusion)
    policy = result.get("claim_policy")
    run_profile = result.get("run_profile")
    if policy is None:
        run_profile = _normalize_profile_id(run_profile or "legacy-shadow")
        policy = expand_profile(run_profile)["claim_policy"]
    else:
        policy = copy.deepcopy(policy)
        validate_claim_policy(policy)
        policy_profile = _normalize_profile_id(
            policy.get("profile_id") or run_profile or "legacy-shadow"
        )
        if run_profile is not None and _normalize_profile_id(run_profile) != policy_profile:
            raise ValueError("fusion run_profile does not match claim_policy")
        run_profile = policy_profile

    policy_hash = claim_policy_hash(policy)
    if result.get("policy_version") not in (None, policy["policy_version"]):
        raise ValueError("fusion policy_version does not match claim_policy")
    if result.get("policy_hash") not in (None, policy_hash):
        raise ValueError("fusion policy_hash does not match claim_policy")
    policy["policy_hash"] = policy_hash
    result["run_profile"] = str(run_profile)
    result["policy_version"] = str(policy["policy_version"])
    result["policy_hash"] = policy_hash
    result["claim_policy"] = policy
    return result


def _expanded_run_profile(config, summary):
    summary = summary or {}
    explicit = _first_not_none(
        summary.get("expanded_run_profile"),
        summary.get("expanded_profile"),
        summary.get("run_profile"),
        getattr(config, "expanded_run_profile", None),
        getattr(config, "expanded_profile", None),
        getattr(config, "run_profile", None),
        getattr(config, "profile_id", None),
        getattr(config, "profile", None),
        summary.get("profile"),
    )
    if isinstance(explicit, dict):
        profile_id = _normalize_profile_id(explicit.get("profile_id"))
        expanded = copy.deepcopy(explicit)
        if "models" not in expanded:
            expanded["models"] = expand_profile(profile_id)["models"]
    else:
        profile_id = _normalize_profile_id(explicit or "legacy-shadow")
        expanded = expand_profile(profile_id)

    policy = _first_not_none(
        summary.get("claim_policy"),
        getattr(config, "claim_policy", None),
        expanded.get("claim_policy"),
    )
    policy = copy.deepcopy(policy)
    validate_claim_policy(policy)
    if _normalize_profile_id(policy.get("profile_id")) != profile_id:
        raise ValueError("run profile does not match claim_policy profile_id")
    policy_hash = claim_policy_hash(policy)
    if policy.get("policy_hash") not in (None, policy_hash):
        raise ValueError("claim_policy hash does not match content")
    policy["policy_hash"] = policy_hash
    expanded["profile_id"] = profile_id
    expanded["claim_policy"] = policy
    canonical_defaults = {
        name: bool(enabled)
        for name, enabled in expand_profile(profile_id)["models"].items()
    }
    supplied_defaults = _first_not_none(
        summary.get("profile_model_defaults"),
        getattr(config, "profile_model_defaults", None),
        expanded.get("profile_model_defaults"),
    )
    if supplied_defaults is None:
        profile_model_defaults = canonical_defaults
    elif not isinstance(supplied_defaults, dict):
        raise ValueError("profile_model_defaults must be an object")
    else:
        profile_model_defaults = {
            str(name): bool(enabled)
            for name, enabled in supplied_defaults.items()
        }
        if profile_model_defaults != canonical_defaults:
            raise ValueError(
                "profile_model_defaults do not match the selected run profile"
            )
    expanded["profile_model_defaults"] = profile_model_defaults
    return expanded


def _model_inventory(config, expanded):
    result = {}
    profile_id = expanded["profile_id"]
    profile_models = dict(
        expanded.get("profile_model_defaults")
        or expand_profile(profile_id)["models"]
    )
    explicit_reasons = _explicit_disabled_reasons(config)
    for name in MODEL_KEYS:
        profile_enabled = bool(profile_models.get(name, False))
        enable_attribute = "enable_%s" % name
        if hasattr(config, enable_attribute):
            enabled = bool(getattr(config, enable_attribute))
        else:
            enabled = profile_enabled
        model_config = getattr(config, "%s_config" % name, None)
        reason = None
        if not enabled:
            reason = explicit_reasons.get(name)
            if not reason and not profile_enabled:
                reason = "disabled_by_run_profile:%s" % profile_id
            if not reason and hasattr(config, enable_attribute):
                reason = "disabled_by_run_configuration"
            if not reason:
                reason = "model_configuration_unavailable"
        result[name] = {
            "enabled": enabled,
            "disabled_reason": reason,
            "profile_default_enabled": profile_enabled,
            "config": _config_record(model_config),
        }
    return result


def _explicit_disabled_reasons(config):
    result = {}
    for attribute in ("model_disabled_reasons", "disabled_model_reasons"):
        value = getattr(config, attribute, None)
        if isinstance(value, dict):
            result.update(
                (str(key), str(reason))
                for key, reason in value.items()
                if reason is not None
            )
    for name in MODEL_KEYS:
        value = getattr(config, "%s_disabled_reason" % name, None)
        if value is not None:
            result[name] = str(value)
    return result


def _config_record(config):
    if config is None:
        return None
    if hasattr(config, "to_record"):
        value = config.to_record()
    else:
        value = dict(vars(config))
    return _sanitize_config(value)


def _sanitize_config(value, key=None):
    if key is not None and _secret_key(key):
        return None
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_config(item_value, key=str(item_key))
            for item_key, item_value in value.items()
            if not _secret_key(item_key)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _secret_key(value):
    normalized = str(value).strip().lower()
    return normalized in {
        "access_token",
        "api_key",
        "password",
        "secret",
        "token",
    } or normalized.endswith(
        ("_access_token", "_api_key", "_password", "_secret", "_token")
    )


def _normalize_profile_id(value):
    value = str(value or "legacy-shadow")
    if value == "v2-shadow":
        return "legacy-shadow"
    return value


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def read_json_gz(path):
    """Small test/runbook helper; not used by the pipeline."""

    with gzip.open(str(path), "rt", encoding="utf-8") as source:
        return json.load(source)
