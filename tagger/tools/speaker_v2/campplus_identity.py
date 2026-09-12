"""Legacy CAM++ adapter retained for fallback and rollback profiles."""

from tagger.tools.speaker_v2._legacy import install_legacy_alias


install_legacy_alias(__name__, "campplus_identity")

