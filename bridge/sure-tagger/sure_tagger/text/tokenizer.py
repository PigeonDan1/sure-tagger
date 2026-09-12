import re
import unicodedata


TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\w\s]",
    re.UNICODE,
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
    return [t for t in tokenize(text) if is_word_token(t)]


def normalized_word_tokens(text):
    return [t.lower() for t in word_tokens(text)]


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
        if ch in [",", "，"]:
            counts["comma"] += 1
        elif ch in [".", "。"]:
            counts["period"] += 1
        elif ch in ["?", "？"]:
            counts["question_mark"] += 1
        elif ch in ["!", "！"]:
            counts["exclamation_mark"] += 1
        elif ch in [";","；"]:
            counts["semicolon"] += 1
        elif ch in [":","："]:
            counts["colon"] += 1
        else:
            counts["other"] += 1
    counts["punctuation_count"] = total
    stripped = (text or "").strip()
    counts["has_terminal_punctuation"] = bool(stripped and stripped[-1] in ".。?!？！")
    return counts
