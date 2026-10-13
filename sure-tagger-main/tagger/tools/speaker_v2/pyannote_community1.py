"""pyannote Community-1 adapter inherited from the frozen v1 snapshot."""

from tagger.tools.speaker_v2._legacy import install_legacy_alias


install_legacy_alias(__name__, "pyannote_community1")

