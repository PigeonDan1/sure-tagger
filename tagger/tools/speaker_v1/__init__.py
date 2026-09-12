"""Evidence-first speaker tagging components.

This package is intentionally side-by-side with ``tagger.tools.speaker``.
Nothing in this package changes the v0 public speaker tags.
"""

from tagger.tools.speaker_v2.contracts import EVIDENCE_SCHEMA_VERSION
from tagger.tools.speaker_v2.resolver import FUSION_SCHEMA_VERSION


__all__ = ["EVIDENCE_SCHEMA_VERSION", "FUSION_SCHEMA_VERSION"]
