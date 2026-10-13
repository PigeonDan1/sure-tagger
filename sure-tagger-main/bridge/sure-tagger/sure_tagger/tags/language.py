import unicodedata

from sure_tagger.schemas import make_tag


TOOL_VERSION = "language_v0.1.0"


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
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("Z"):
        return "punct_or_space"
    return "other"


def tag(record, config=None):
    config = config or {}
    text = record["sample"].get("text", {}).get("transcript", "")
    counts = {}
    content_chars = 0
    for ch in text:
        s = _script(ch)
        counts[s] = counts.get(s, 0) + 1
        if s not in ("punct_or_space", "digit"):
            content_chars += 1

    if content_chars == 0:
        value = "unknown"
        confidence = 0.0
    else:
        ratios = {}
        for k, v in counts.items():
            if k not in ("punct_or_space", "digit"):
                ratios[k] = float(v) / float(content_chars)
        dominant = max(ratios.items(), key=lambda kv: kv[1])[0] if ratios else "unknown"
        if dominant == "latin":
            value = "en"
        elif dominant == "cjk":
            value = "zh"
        elif dominant == "cyrillic":
            value = "ru"
        elif dominant == "arabic":
            value = "ar"
        else:
            value = "unknown"
        confidence = ratios.get(dominant, 0.0)
        min_chars = int(config.get("min_chars_for_confident_prediction", 10))
        if content_chars < min_chars:
            confidence = min(confidence, 0.6)

    details = {
        "script_distribution": counts,
        "content_char_count": content_chars,
        "note": "heuristic_lid; use fastText/CLD3 when available for production",
    }
    return make_tag(value, confidence, "unicode_script_heuristic", TOOL_VERSION, "L1", details)
