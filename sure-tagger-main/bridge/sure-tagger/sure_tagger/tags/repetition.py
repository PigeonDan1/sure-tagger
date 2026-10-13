from sure_tagger.schemas import make_tag
from sure_tagger.text.tokenizer import normalized_word_tokens


TOOL_VERSION = "repetition_v0.1.0"


def _find_repetitions(tokens, max_ngram):
    spans = []
    n_tokens = len(tokens)
    for n in range(int(max_ngram), 0, -1):
        i = 0
        while i + 2 * n <= n_tokens:
            left = tokens[i:i + n]
            right = tokens[i + n:i + 2 * n]
            if left == right:
                spans.append({
                    "type": "ngram_%d" % n,
                    "text": " ".join(left + right),
                    "start_token": i,
                    "end_token": i + 2 * n,
                })
                i += 2 * n
            else:
                i += 1
    return spans


def tag(record, config=None):
    config = config or {}
    max_ngram = int(config.get("max_ngram", 3))
    text = record["sample"].get("text", {}).get("transcript", "")
    tokens = normalized_word_tokens(text)
    spans = _find_repetitions(tokens, max_ngram)
    value = {
        "has_repetition": bool(spans),
        "repetition_count": len(spans),
    }
    details = {
        "repeated_spans": spans,
        "max_ngram": max_ngram,
        "consecutive_only": True,
    }
    return make_tag(value, 1.0, "token_ngram_rule", TOOL_VERSION, "L0", details)
