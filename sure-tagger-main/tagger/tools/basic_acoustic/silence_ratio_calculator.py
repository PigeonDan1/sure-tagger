"""Compatibility import for the former ratio-calculator module name.

New code should import :mod:`nonspeech_ratio_calculator`; the public tag has
always been ``basic_acoustic.nonspeech_ratio`` since the rename.
"""

from tagger.tools.basic_acoustic.nonspeech_ratio_calculator import (  # noqa: F401
    TOOL_NAME,
    run,
)
