from sure_tagger.schemas import make_tag
from sure_tagger.text.tokenizer import word_tokens


TOOL_VERSION = "filler_v0.1.0"
DEFAULT_FILLERS = set(["uh", "um", "erm", "er", "ah", "oh", "hmm", "mm", "yeah"])


def tag(record, config=None):
    config = config or {}
    fillers = set([w.lower() for w in config.get("words", list(DEFAULT_FILLERS))])
    text = record["sample"].get("text", {}).get("transcript", "")
    tokens = word_tokens(text)
    items = []
    for idx, tok in enumerate(tokens):
        norm = tok.lower()
        if norm in fillers:
            items.append({"token": tok, "normalized": norm, "token_index": idx})
    ratio = float(len(items)) / float(len(tokens)) if tokens else 0.0
    details = {
        "filler_count": len(items),
        "filler_ratio": ratio,
        "items": items,
        "lexicon_version": config.get("lexicon_version", "filler_en_v0"),
    }
    return make_tag(len(items), 1.0, "lexicon_rule", TOOL_VERSION, "L0", details)
