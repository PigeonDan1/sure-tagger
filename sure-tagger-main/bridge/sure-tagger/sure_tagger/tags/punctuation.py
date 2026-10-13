from sure_tagger.schemas import make_tag
from sure_tagger.text.tokenizer import punctuation_counts


TOOL_VERSION = "punctuation_v0.1.0"


def tag(record, config=None):
    text = record["sample"].get("text", {}).get("transcript", "")
    counts = punctuation_counts(text)
    value = {
        "punctuation_count": counts["punctuation_count"],
        "has_terminal_punctuation": counts["has_terminal_punctuation"],
    }
    return make_tag(value, 1.0, "unicode_punctuation_counter", TOOL_VERSION, "L0", counts)
