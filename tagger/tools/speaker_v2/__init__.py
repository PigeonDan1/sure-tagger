"""Claim-aware speaker evidence components for the v2 shadow profiles."""

from tagger.tools.speaker_v2.contracts import EVIDENCE_SCHEMA_VERSION
from tagger.tools.speaker_v2.profiles import POLICY_VERSION, available_profiles
from tagger.tools.speaker_v2.resolver import FUSION_SCHEMA_VERSION
from tagger.tools.speaker_v2.speaker_profile import PROFILE_SCHEMA_VERSION


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "FUSION_SCHEMA_VERSION",
    "POLICY_VERSION",
    "available_profiles",
    "PROFILE_SCHEMA_VERSION",
]
