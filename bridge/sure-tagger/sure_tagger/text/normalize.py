import html
import re
import unicodedata


NO_SPACE_BEFORE = set([",", ".", "?", "!", ":", ";", "%", ")", "]", "}"])
NO_SPACE_AFTER = set(["(", "[", "{", "$", "#"])


def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def is_punctuation(token):
    if not token:
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in token)


def join_tokens(tokens):
    out = []
    for token in tokens:
        if token is None:
            continue
        token = html.unescape(str(token))
        if token == "":
            continue
        if not out:
            out.append(token)
            continue
        if token in NO_SPACE_BEFORE or is_punctuation(token):
            out[-1] = out[-1] + token
        elif out[-1] and out[-1][-1] in NO_SPACE_AFTER:
            out[-1] = out[-1] + token
        else:
            out.append(token)
    return normalize_space(" ".join(out))


def normalize_transcript(text):
    text = html.unescape(text or "")
    text = normalize_space(text)
    text = re.sub(r"\s+([,.;:?!%)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{\$#])\s+", r"\1", text)
    return text.strip()
