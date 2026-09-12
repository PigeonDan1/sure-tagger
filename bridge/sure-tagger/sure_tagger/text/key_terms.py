import re
from collections import Counter

from sure_tagger.text.tokenizer import normalized_word_tokens


STOPWORDS = set("""
a an and are as at be but by can for from had has have he her his i in is it
its me my of on or our she so that the their them they this to was we were
what when where which who will with you your um uh er ah oh okay yeah
hi did cool there yes big allergic tail designing
""".split())


def extract_proper_nouns(text, max_items=12):
    text = text or ""
    candidates = []
    # Consecutive title-case words and uppercase acronyms.
    pattern = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-zA-Z0-9]+|[A-Z]{2,}))*\b")
    for m in pattern.finditer(text):
        value = m.group(0).strip()
        if len(value) <= 1:
            continue
        if value.lower() in STOPWORDS:
            continue
        words = value.split()
        if len(words) == 1 and not value.isupper():
            prefix = text[:m.start()].rstrip()
            # Most single title-case words at sentence start are capitalization artifacts.
            if not prefix or prefix[-1] in ".?!:\n":
                continue
        candidates.append(value)
    seen = set()
    out = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def extract_keywords(text, max_items=12):
    tokens = [t for t in normalized_word_tokens(text) if len(t) > 2 and t not in STOPWORDS]
    counts = Counter(tokens)
    out = [w for w, _ in counts.most_common(max_items)]
    return out
