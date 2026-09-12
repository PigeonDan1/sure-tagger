"""Closed raw-only input schema validation."""


class InputSchemaError(ValueError):
    """Raised when a record does not match the final raw-only input schema."""


TOP_LEVEL_KEYS = set(["corpus", "sample"])
CORPUS_KEYS = set(["dataset_name", "source_urls", "native_metadata"])
SOURCE_URL_KEYS = set(["article", "github", "huggingface", "dataset_card"])
SAMPLE_KEYS = set(["sample_id", "audio", "text", "native_metadata"])
AUDIO_KEYS = set(["path"])
TEXT_KEYS = set(["transcript"])


def validate_input_record(record):
    """Reject records outside the closed raw-only runtime schema."""

    _require_dict(record, "record")
    _require_exact_keys(record, TOP_LEVEL_KEYS, "record")

    corpus = record["corpus"]
    sample = record["sample"]
    _require_dict(corpus, "corpus")
    _require_dict(sample, "sample")
    _require_exact_keys(corpus, CORPUS_KEYS, "corpus")
    _require_exact_keys(sample, SAMPLE_KEYS, "sample")

    if not isinstance(corpus["dataset_name"], str):
        raise InputSchemaError("corpus.dataset_name must be a string")
    _require_dict(corpus["source_urls"], "corpus.source_urls")
    _require_exact_keys(corpus["source_urls"], SOURCE_URL_KEYS, "corpus.source_urls")
    for key, value in corpus["source_urls"].items():
        if not isinstance(value, list):
            raise InputSchemaError("corpus.source_urls.%s must be a list" % key)
    _require_dict(corpus["native_metadata"], "corpus.native_metadata")

    if not isinstance(sample["sample_id"], str):
        raise InputSchemaError("sample.sample_id must be a string")
    _require_dict(sample["audio"], "sample.audio")
    _require_exact_keys(sample["audio"], AUDIO_KEYS, "sample.audio")
    if not isinstance(sample["audio"]["path"], str):
        raise InputSchemaError("sample.audio.path must be a string")
    _require_dict(sample["text"], "sample.text")
    _require_exact_keys(sample["text"], TEXT_KEYS, "sample.text")
    if not isinstance(sample["text"]["transcript"], str):
        raise InputSchemaError("sample.text.transcript must be a string")
    _require_dict(sample["native_metadata"], "sample.native_metadata")


def _require_dict(value, path):
    if not isinstance(value, dict):
        raise InputSchemaError("%s must be an object" % path)


def _require_exact_keys(value, allowed_keys, path):
    keys = set(value.keys())
    extra = sorted(keys - allowed_keys)
    missing = sorted(allowed_keys - keys)
    if extra or missing:
        parts = []
        if extra:
            parts.append("extra keys: %s" % ", ".join(extra))
        if missing:
            parts.append("missing keys: %s" % ", ".join(missing))
        raise InputSchemaError("%s schema mismatch (%s)" % (path, "; ".join(parts)))
