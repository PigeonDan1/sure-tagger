from sure_tagger.schemas import make_tag
from sure_tagger.text.tokenizer import tokenize, word_tokens


TOOL_VERSION = "word_count_v0.1.0"


def tag(record, config=None):
    text = record["sample"].get("text", {}).get("transcript", "")
    words = word_tokens(text)
    details = {
        "word_count": len(words),
        "character_count": len(text),
        "token_count": len(tokenize(text)),
        "tokenizer": (config or {}).get("tokenizer", "simple_multilingual_v0"),
    }
    return make_tag(len(words), 1.0, "deterministic_tokenizer", TOOL_VERSION, "L0", details)
