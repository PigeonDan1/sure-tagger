"""Deterministic language-content tags from sample transcripts."""

import re
import unicodedata

from tagger.tools.base import ToolResult


TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\w\s]",
    re.UNICODE,
)

LANGUAGE_TOOL_VERSION = "language_v0.1.0"
WORD_COUNT_TOOL_VERSION = "word_count_v0.1.0"
PUNCTUATION_TOOL_VERSION = "punctuation_v0.1.0"
REPETITION_TOOL_VERSION = "repetition_v0.1.0"
FILLER_TOOL_VERSION = "filler_v0.1.0"

DEFAULT_FILLERS = set(["uh", "um", "erm", "er", "ah", "oh", "hmm", "mm", "yeah"])


def run_all(transcript, config=None, include_language=False):
    # type: (str, object, bool) -> list
    config = config or {}
    text = transcript or ""
    results = [
        count_words(text, config.get("word_count", {})),
        count_punctuation(text),
        detect_repetition(text, config.get("repetition", {})),
        count_fillers(text, config.get("filler", {})),
    ]
    if include_language:
        results.insert(0, detect_language(text, config.get("language", {})))
    return results


def detect_language(text, config=None):
    config = config or {}
    counts = {}
    content_chars = 0
    for ch in text:
        script = _script(ch)
        counts[script] = counts.get(script, 0) + 1
        if script not in ("punct_or_space", "digit"):
            content_chars += 1

    if content_chars == 0:
        value = "unknown"
        confidence = 0.0
    else:
        ratios = {}
        for key, count in counts.items():
            if key not in ("punct_or_space", "digit"):
                ratios[key] = float(count) / float(content_chars)
        dominant = max(ratios.items(), key=lambda item: item[1])[0] if ratios else "unknown"
        value = {
            "latin": "en",
            "cjk": "zh",
            "cyrillic": "ru",
            "arabic": "ar",
        }.get(dominant, "unknown")
        confidence = ratios.get(dominant, 0.0)
        min_chars = int(config.get("min_chars_for_confident_prediction", 10))
        if content_chars < min_chars:
            confidence = min(confidence, 0.6)

    return ToolResult(
        tag_path="language_content.language",
        value=value,
        tool_name="language_detector",
        method="unicode_script_heuristic",
        status="estimated",
        confidence=confidence,
        tool_type="deterministic",
        tool_version=LANGUAGE_TOOL_VERSION,
        evidence={
            "script_distribution": counts,
            "content_char_count": content_chars,
        },
    )


def count_words(text, config=None):
    config = config or {}
    words = word_tokens(text)
    return ToolResult(
        tag_path="language_content.word_count",
        value=len(words),
        tool_name="word_count_calculator",
        method="simple_multilingual_tokenizer",
        status="estimated",
        confidence=1.0,
        tool_type="deterministic",
        tool_version=WORD_COUNT_TOOL_VERSION,
        evidence={
            "character_count": len(text),
            "token_count": len(tokenize(text)),
            "tokenizer": config.get("tokenizer", "simple_multilingual_v0"),
        },
    )


def count_punctuation(text):
    counts = punctuation_counts(text)
    value = {
        "punctuation_count": counts["punctuation_count"],
        "has_terminal_punctuation": counts["has_terminal_punctuation"],
    }
    return ToolResult(
        tag_path="language_content.punctuation",
        value=value,
        tool_name="punctuation_counter",
        method="unicode_punctuation_counter",
        status="estimated",
        confidence=1.0,
        tool_type="deterministic",
        tool_version=PUNCTUATION_TOOL_VERSION,
        evidence=counts,
    )


def detect_repetition(text, config=None):
    config = config or {}
    max_ngram = int(config.get("max_ngram", 3))
    spans = _find_repetitions(normalized_word_tokens(text), max_ngram)
    value = {
        "has_repetition": bool(spans),
        "repetition_count": len(spans),
    }
    return ToolResult(
        tag_path="language_content.repetition",
        value=value,
        tool_name="repetition_detector",
        method="consecutive_token_ngram_rule",
        status="estimated",
        confidence=1.0,
        tool_type="deterministic",
        tool_version=REPETITION_TOOL_VERSION,
        evidence={
            "repeated_spans": spans,
            "max_ngram": max_ngram,
            "consecutive_only": True,
        },
    )


def count_fillers(text, config=None):
    config = config or {}
    fillers = set([word.lower() for word in config.get("words", list(DEFAULT_FILLERS))])
    tokens = word_tokens(text)
    items = []
    for index, token in enumerate(tokens):
        normalized = token.lower()
        if normalized in fillers:
            items.append(
                {
                    "token": token,
                    "normalized": normalized,
                    "token_index": index,
                }
            )
    return ToolResult(
        tag_path="language_content.filler",
        value=len(items),
        tool_name="filler_counter",
        method="lexicon_rule",
        status="estimated",
        confidence=1.0,
        tool_type="deterministic",
        tool_version=FILLER_TOOL_VERSION,
        evidence={
            "filler_count": len(items),
            "filler_ratio": float(len(items)) / float(len(tokens)) if tokens else 0.0,
            "items": items,
            "lexicon_version": config.get("lexicon_version", "filler_en_v0"),
        },
    )


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def is_word_token(token):
    if not token:
        return False
    if re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", token):
        return True
    if re.match(r"^\d+(?:\.\d+)?$", token):
        return True
    if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
        return True
    return False


def word_tokens(text):
    return [token for token in tokenize(text) if is_word_token(token)]


def normalized_word_tokens(text):
    return [token.lower() for token in word_tokens(text)]


def punctuation_counts(text):
    counts = {
        "comma": 0,
        "period": 0,
        "question_mark": 0,
        "exclamation_mark": 0,
        "semicolon": 0,
        "colon": 0,
        "other": 0,
    }
    total = 0
    for ch in text or "":
        if not unicodedata.category(ch).startswith("P"):
            continue
        total += 1
        if ch in [",", "\uff0c"]:
            counts["comma"] += 1
        elif ch in [".", "\u3002"]:
            counts["period"] += 1
        elif ch in ["?", "\uff1f"]:
            counts["question_mark"] += 1
        elif ch in ["!", "\uff01"]:
            counts["exclamation_mark"] += 1
        elif ch in [";", "\uff1b"]:
            counts["semicolon"] += 1
        elif ch in [":", "\uff1a"]:
            counts["colon"] += 1
        else:
            counts["other"] += 1
    counts["punctuation_count"] = total
    stripped = (text or "").strip()
    counts["has_terminal_punctuation"] = bool(
        stripped and stripped[-1] in ".\u3002?!\uff1f\uff01"
    )
    return counts


def _find_repetitions(tokens, max_ngram):
    spans = []
    n_tokens = len(tokens)
    for ngram_size in range(int(max_ngram), 0, -1):
        index = 0
        while index + 2 * ngram_size <= n_tokens:
            left = tokens[index : index + ngram_size]
            right = tokens[index + ngram_size : index + 2 * ngram_size]
            if left == right:
                spans.append(
                    {
                        "type": "ngram_%d" % ngram_size,
                        "text": " ".join(left + right),
                        "start_token": index,
                        "end_token": index + 2 * ngram_size,
                    }
                )
                index += 2 * ngram_size
            else:
                index += 1
    return spans


def _script(ch):
    code = ord(ch)
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return "latin"
    if 0x4E00 <= code <= 0x9FFF:
        return "cjk"
    if 0x0400 <= code <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= code <= 0x06FF:
        return "arabic"
    if ch.isdigit():
        return "digit"
    category = unicodedata.category(ch)
    if category.startswith("P") or category.startswith("Z"):
        return "punct_or_space"
    return "other"
