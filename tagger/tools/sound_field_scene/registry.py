"""Registry of sound-field and acoustic-scene tag tools."""

from tagger.tools.sound_field_scene import dass_noise_type_detector
from tagger.tools.sound_field_scene import firered_aed_detector


FIRERED_AED_TOOL = {
    "tag_path": "sound_field_scene",
    "tool_name": firered_aed_detector.TOOL_NAME,
    "run": firered_aed_detector.run,
}

# Deprecated on 2026-08-25: the PANNs stage and its public
# ``sound_field_scene.sound`` field were removed (noise_composition
# supersedes them). ``panns_background_detector`` stays importable for
# future cross-validation evidence work, but it is not registered and must
# not enter public output.

DASS_NOISE_TYPE_TOOL = {
    "tag_path": "sound_field_scene.external_noise_type",
    "tool_name": dass_noise_type_detector.TOOL_NAME,
    "run": dass_noise_type_detector.run,
}

SOUND_FIELD_SCENE_TOOLS = [
    FIRERED_AED_TOOL,
    DASS_NOISE_TYPE_TOOL,
]
