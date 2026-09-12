"""Registry for language-content tag tools."""

from tagger.tools.language_content import deterministic
from tagger.tools.language_content import firered_lid_detector


DETERMINISTIC_LANGUAGE_CONTENT_TOOL = {
    "tool_name": "deterministic_language_content",
    "tag_paths": [
        "language_content.word_count",
        "language_content.punctuation",
        "language_content.repetition",
        "language_content.filler",
    ],
    "run": deterministic.run_all,
}

FIRERED_LID_LANGUAGE_CONTENT_TOOL = {
    "tool_name": firered_lid_detector.TOOL_NAME,
    "tag_path": "language_content.language",
    "run": firered_lid_detector.run,
}
