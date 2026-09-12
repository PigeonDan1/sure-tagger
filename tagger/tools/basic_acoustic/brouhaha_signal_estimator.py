"""Backward-compatibility shim for the Brouhaha signal estimator.

The estimator moved to ``tagger/tools/audio_quality/brouhaha_signal_estimator.py``
as part of the per-tag-group tools layout. This module re-exports its public
API so callers outside this tree (e.g. ``tagger/tools/speaker_v2/`` helpers
maintained by other owners) keep importing the historical path. New code should
import from ``tagger.tools.audio_quality.brouhaha_signal_estimator`` directly.
"""

from tagger.tools.audio_quality.brouhaha_signal_estimator import (  # noqa: F401
    BrouhahaClient,
    BrouhahaConfig,
    BrouhahaError,
    BrouhahaSubprocessClient,
)
