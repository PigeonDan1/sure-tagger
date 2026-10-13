#!/usr/bin/env python3
"""Run the two DeepSeek extraction paths with checkpointed concurrency.

The API user message is deliberately limited to one upstream text value per
request.  For ``captioner`` that value is the raw Qwen caption.  For
``sure-tagger`` it is a compact JSON representation of the seven structural
tag groups; all caller metadata is removed before the request is made.

Input JSONL rows normally look like::

    {"sample_id": "sample-001", "input_text": "..."}

Captioner rows may instead use ``caption`` or ``text_path``.  Sure-tagger
rows may use ``tags`` (or contain the structural groups at the top level).
Successful complete prediction envelopes are written to ``results.jsonl``;
per-sample checkpoint records live under ``items/`` and can be reused after an
interruption; only network/transient API failures are retried.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import socket
import sys
import tempfile
import threading
import time
import urllib.error
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import extract_captioner_deepseek_once as captioner_once  # noqa: E402
import extract_sure_tagger_deepseek_once as sure_once  # noqa: E402


SYSTEM_ALIASES = {
    "captioner": "captioner",
    "caption": "captioner",
    "qwen": "captioner",
    "sure-tagger": "sure-tagger",
    "sure_tagger": "sure-tagger",
    "sure": "sure-tagger",
}
STRUCTURAL_GROUPS = tuple(sure_once.STRUCTURAL_GROUPS)
STRUCTURAL_FIELDS = sure_once.STRUCTURAL_FIELDS
SCHEMA_VERSION = sure_once.SCHEMA_VERSION
FIELD_UNIVERSE_VERSION = sure_once.FIELD_UNIVERSE_VERSION

DEFAULT_API_FILE = SCRIPT_DIR.parent / "api" / "deepseek-api.txt"
DEFAULT_PROMPTS = {
    "captioner": SCRIPT_DIR / "captioner_deepseek_extraction_prompt.txt",
    # The merged prompt replaces the descriptive-only prompt: one call now
    # returns the 12 descriptive fields *and* the language verdict.  The old
    # file is kept on disk as the reproduction record for the published run
    # whose ``prompt_sha256`` is 866c1f7a...; it is no longer a default.
    "sure-tagger": SCRIPT_DIR / "sure_tagger_deepseek_language_descriptive_prompt.txt",
}
SURE_TAGGER_PROMPT_VERSIONS = {
    "v1": SCRIPT_DIR / "sure_tagger_deepseek_language_descriptive_prompt.txt",
}
DEFAULT_SURE_TAGGER_PROMPT_VERSION = "v1"
DEFAULT_OUTPUT_DIRS = {
    "captioner": SCRIPT_DIR.parent / "evaluation" / "benchmark-v1" / "captioner-deepseek",
    "sure-tagger": SCRIPT_DIR.parent / "evaluation" / "benchmark-v1" / "sure-tagger-deepseek",
}

RETRYABLE_TEXT = re.compile(r"(?i)(sk-[A-Za-z0-9._-]{12,})")
RETRYABLE_STATUS = frozenset((408, 409, 425, 429, 500, 502, 503, 504))
HTTP_STATUS_TEXT = re.compile(r"(?i)\bHTTP\s*([0-9]{3})\b")
# ``call_api`` currently wraps transport exceptions in RuntimeError, so keep
# this classifier usable both for wrapped errors and for direct exceptions in
# tests or future callers.  Response/schema errors deliberately do not match.
NETWORK_ERROR_TEXT = re.compile(
    r"(?i)(?:"
    r"\bhttp\s*(?:408|409|425|429|5\d\d)\b"
    r"|timeout|timed\s*out|"
    r"remote\s*(?:disconnected|end\s+closed)"
    r"|connection\s*(?:aborted|reset|refused|closed|error|failure|timed\s*out)"
    r"|connection(?:aborted|reset|refused|closed|error)"
    r"|connect(?:timeout|error)"
    r"|broken\s+pipe|socket\s+error|"
    r"ssl(?:error)?|ssleof|unexpected[_\s-]*eof|\beof\b|"
    r"temporary\s+failure|transient|name\s+(?:or\s+service|resolution)|"
    r"max\s+retries\s+exceeded|network\s+unreachable|no\s+route|"
    r"gaierror"
    r")"
)
NETWORK_ERRNOS = frozenset((101, 104, 110, 111, 113))


def is_network_failure(error):
    """Return whether *error* is a transport/transient failure safe to retry.

    DeepSeek's single-request helper predates the batch retry policy and emits
    generic ``RuntimeError`` instances.  Inspect both exception attributes and
    its rendered message so HTTP status and wrapped urllib/socket failures are
    still classified correctly.  A known non-retryable HTTP status takes
    precedence over a misleading word such as ``timeout`` in its body.
    """

    status = getattr(error, "status", None)
    if status is None:
        # urllib.error.HTTPError exposes ``code`` rather than ``status``.
        status = getattr(error, "code", None)
    if status is None:
        match = HTTP_STATUS_TEXT.search(str(error))
        if match:
            try:
                status = int(match.group(1))
            except (TypeError, ValueError):
                status = None
    if isinstance(status, int):
        return status in RETRYABLE_STATUS or status >= 500

    if isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
        ),
    ):
        return True
    if isinstance(error, OSError) and getattr(error, "errno", None) in NETWORK_ERRNOS:
        return True
    return NETWORK_ERROR_TEXT.search(str(error)) is not None


def _all_network_errors(value):
    """Return true only when every recorded error is network/transient.

    Older checkpoints store retry attempts as one ``; ``-joined string.  The
    split is intentionally conservative: an embedded semicolon can suppress
    a retry, but it can never turn a mixed network/deterministic history into
    an automatic retry.
    """

    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    parts = []
    for value in values:
        parts.extend(
            part.strip()
            for part in str(value or "").split("; ")
            if part.strip()
        )
    return bool(parts) and all(is_network_failure(part) for part in parts)


def checkpoint_retryable(record):
    """Return whether a failed checkpoint should be retried by default.

    New records carry an explicit ``network_error`` type.  Older checkpoints
    used ``api_error`` for every exception, so infer retryability from every
    recorded error segment only when all segments identify network/transient
    faults.
    """

    if not isinstance(record, dict) or record.get("status") != "failed":
        return False
    error_type = record.get("error_type")
    if error_type in (
        "upstream_failed",
        "invalid_input",
        "validation_error",
        "model_output_error",
        "auth_error",
        "deterministic_error",
    ):
        return False
    # Unknown typed failures are safer to treat as terminal.  ``api_error``
    # and ``network_error`` are the only types emitted by this runner; an
    # absent type denotes a legacy checkpoint that needs text inference.
    if error_type not in (None, "api_error", "network_error"):
        return False
    # Prefer a future structured error list when present; current and legacy
    # records use the joined ``error`` string.
    value = record.get("errors")
    if value is None:
        value = record.get("error", "")
    return _all_network_errors(value)


class BatchInputError(ValueError):
    """A malformed input row that must never be sent to the API."""


class BatchItem(object):
    __slots__ = (
        "sample_id",
        "input_text",
        "source_index",
        "source_hash",
        "pre_error",
        "pre_error_type",
    )

    def __init__(
        self,
        sample_id,
        input_text=None,
        source_index=0,
        source_hash=None,
        pre_error=None,
        pre_error_type=None,
    ):
        self.sample_id = sample_id
        self.input_text = input_text
        self.source_index = int(source_index)
        self.source_hash = source_hash
        self.pre_error = pre_error
        self.pre_error_type = pre_error_type


def strict_json(text):
    """Parse JSON while rejecting duplicate keys and non-finite constants."""

    def reject_constant(value):
        raise ValueError("non-finite JSON constant: %s" % value)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: %s" % key)
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def redact(value, secret=None):
    text = str(value)
    if secret:
        text = text.replace(secret, "<redacted>")
    return RETRYABLE_TEXT.sub("<redacted>", text)[:2000]


def normalize_system(value):
    key = str(value or "").strip().lower()
    if key not in SYSTEM_ALIASES:
        raise ValueError("--system must be captioner or sure-tagger")
    return SYSTEM_ALIASES[key]


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise BatchInputError("input manifest does not exist: %s" % path)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = strict_json(line)
            except (TypeError, ValueError) as exc:
                raise BatchInputError(
                    "invalid JSON at %s:%d: %s" % (path, line_number, exc)
                )
            rows.append((line_number, value, line.encode("utf-8")))
    if not rows:
        raise BatchInputError("input manifest is empty: %s" % path)
    return rows


def _nested_get(value, path):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _sample_id_from_row(row, index, id_manifest=None):
    candidates = (
        row.get("sample_id") if isinstance(row, dict) else None,
        _nested_get(row, ("sample", "sample_id")) if isinstance(row, dict) else None,
        _nested_get(row, ("item", "sample_id")) if isinstance(row, dict) else None,
    )
    embedded_id = next(
        (value for value in candidates if isinstance(value, str) and value.strip()),
        None,
    )
    sample_id = embedded_id
    if id_manifest is not None:
        if index > len(id_manifest):
            raise BatchInputError("ID manifest has no row %d" % index)
        mapped_id = id_manifest[index - 1]
        if embedded_id is not None and embedded_id.strip() != mapped_id:
            raise BatchInputError(
                "row %d sample_id %r disagrees with ID manifest %r"
                % (index, embedded_id, mapped_id)
            )
        sample_id = mapped_id
    if sample_id is None:
        raise BatchInputError(
            "row %d has no sample_id (provide --id-manifest for tags-only rows)"
            % index
        )
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise BatchInputError("row %d sample_id is empty" % index)
    return sample_id.strip()


def _load_id_manifest(path):
    if path is None:
        return None
    rows = read_jsonl(path)
    result = []
    for index, row, _ in rows:
        if isinstance(row, str):
            value = row
        elif isinstance(row, dict):
            value = row.get("sample_id") or _nested_get(row, ("sample", "sample_id"))
        else:
            value = None
        if not isinstance(value, str) or not value.strip():
            raise BatchInputError("ID manifest row %d has no sample_id" % index)
        result.append(value.strip())
    return result


def _read_text_file(value, base_dir):
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BatchInputError("cannot read text file %s: %s" % (path, exc))
    try:
        return data.decode("utf-8"), data
    except UnicodeDecodeError as exc:
        raise BatchInputError("text file is not UTF-8 (%s): %s" % (path, exc))


def _caption_text_from_row(row, base_dir, text_field=None):
    if isinstance(row, str):
        return row, row.encode("utf-8")
    if not isinstance(row, dict):
        raise BatchInputError("captioner row must be an object")

    if text_field:
        value = _nested_get(row, tuple(text_field.split(".")))
        if isinstance(value, str):
            return value, value.encode("utf-8")
        raise BatchInputError("text field %s is not a string" % text_field)

    for key in ("input_text", "caption", "caption_text", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value, value.encode("utf-8")

    for key in ("text_path", "caption_path", "input_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _read_text_file(value, base_dir)

    # A common Qwen result stores the caption in a nested result object.
    for path in (("result", "caption"), ("output", "caption")):
        value = _nested_get(row, path)
        if isinstance(value, str):
            return value, value.encode("utf-8")
    raise BatchInputError(
        "captioner row has no input_text/caption/text_path; refusing to serialize metadata"
    )


def _structural_candidate(value):
    """Find structural groups without accepting caller metadata as evidence."""

    if not isinstance(value, dict):
        return None
    if all(group in value for group in STRUCTURAL_GROUPS):
        return value
    for key in ("tags", "structural", "prediction", "payload", "result"):
        nested = value.get(key)
        if isinstance(nested, dict) and all(group in nested for group in STRUCTURAL_GROUPS):
            return nested
    return None


def _validate_and_canonicalize_structural(value, sample_id):
    structural = _structural_candidate(value)
    if structural is None:
        raise BatchInputError(
            "sure-tagger input for %s lacks all structural groups" % sample_id
        )
    # Select only the frozen groups.  This intentionally drops sample_id,
    # audio paths, GT, descriptive text, and any future metadata fields.
    selected = {group: structural[group] for group in STRUCTURAL_GROUPS}
    try:
        canonical = sure_once.canonical_structural(selected)
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchInputError("invalid structural tags for %s: %s" % (sample_id, exc))

    # Reuse the authoritative complete-payload validator for structural type,
    # range, enum, and nested-key checks.  Empty descriptive values are valid.
    empty_descriptive = {key: None for key in captioner_once.DESCRIPTIVE_KEYS}
    candidate = {
        "sample_id": sample_id,
        "schema_version": SCHEMA_VERSION,
        "field_universe_version": FIELD_UNIVERSE_VERSION,
        "basic_acoustic": copy.deepcopy(canonical["basic_acoustic"]),
        "audio_quality": copy.deepcopy(canonical["audio_quality"]),
        "room_acoustic": copy.deepcopy(canonical["room_acoustic"]),
        "sound_field_scene": copy.deepcopy(canonical["sound_field_scene"]),
        "speaker": copy.deepcopy(canonical["speaker"]),
        "language_content": copy.deepcopy(canonical["language_content"]),
        "deterministic_derived": copy.deepcopy(canonical["deterministic_derived"]),
        "descriptive": empty_descriptive,
    }
    try:
        captioner_once.validate_payload(candidate, sample_id)
    except (captioner_once.ValidationError, ValueError, TypeError) as exc:
        raise BatchInputError("invalid structural tags for %s: %s" % (sample_id, exc))
    return canonical


def _sure_text_from_row(row, sample_id, base_dir, text_field=None):
    # The explicit input_text form is accepted, but it is parsed and rebuilt
    # below so the API never receives an ID or wrapper accidentally.
    source_bytes = None
    if isinstance(row, str):
        raw_text = row
        source_bytes = raw_text.encode("utf-8")
        value = strict_json(raw_text)
    elif isinstance(row, dict):
        if text_field:
            raw_value = _nested_get(row, tuple(text_field.split(".")))
            if isinstance(raw_value, str):
                raw_text = raw_value
                source_bytes = raw_value.encode("utf-8")
                try:
                    value = strict_json(raw_value)
                except ValueError as exc:
                    raise BatchInputError("sure input_text is not JSON for %s: %s" % (sample_id, exc))
            elif isinstance(raw_value, dict):
                value = raw_value
                raw_text = json.dumps(raw_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                source_bytes = raw_text.encode("utf-8")
            else:
                raise BatchInputError("text field %s is not JSON text/object" % text_field)
        elif isinstance(row.get("input_text"), str):
            raw_text = row["input_text"]
            source_bytes = raw_text.encode("utf-8")
            try:
                value = strict_json(raw_text)
            except ValueError as exc:
                raise BatchInputError("sure input_text is not JSON for %s: %s" % (sample_id, exc))
        elif isinstance(row.get("input_text"), dict):
            # Permit a structured value in local manifests, but still rebuild
            # the exact wire text from the frozen structural groups below.
            value = row["input_text"]
            raw_text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            source_bytes = raw_text.encode("utf-8")
        elif isinstance(row.get("tags"), dict) or _structural_candidate(row) is not None:
            value = row
            raw_text = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            source_bytes = raw_text.encode("utf-8")
        else:
            for key in ("tags_path", "input_path", "text_path"):
                path_value = row.get(key)
                if isinstance(path_value, str) and path_value.strip():
                    raw_text, source_bytes = _read_text_file(path_value, base_dir)
                    try:
                        value = strict_json(raw_text)
                    except ValueError as exc:
                        raise BatchInputError("sure input file is not JSON for %s: %s" % (sample_id, exc))
                    break
            else:
                raise BatchInputError("sure-tagger row has no input_text or structural tags")
    else:
        raise BatchInputError("sure-tagger row must be an object or JSON text")

    canonical = _validate_and_canonicalize_structural(value, sample_id)
    # This is the exact string sent to DeepSeek.  It contains only structural
    # groups and is independent of all source metadata.
    sent_text = sure_once.descriptive_input(canonical)
    return sent_text, source_bytes


def _load_caption_directory(path):
    files = sorted(Path(path).rglob("*.txt"), key=lambda item: str(item))
    if not files:
        raise BatchInputError("no .txt files found under %s" % path)
    rows = []
    for index, file_path in enumerate(files, 1):
        data = file_path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BatchInputError("caption file is not UTF-8 (%s): %s" % (file_path, exc))
        rows.append((index, {"sample_id": file_path.stem, "input_text": text}, data))
    return rows


def load_items(system, input_path, id_manifest_path=None, text_field=None, limit=None):
    input_path = Path(input_path).resolve()
    if input_path.is_dir():
        if system != "captioner":
            raise BatchInputError("sure-tagger input must be JSONL, not a directory")
        rows = _load_caption_directory(input_path)
    else:
        rows = read_jsonl(input_path)
    id_manifest = _load_id_manifest(id_manifest_path)
    items = []
    seen = set()
    for row_number, row, raw_line in rows:
        if limit is not None and len(items) >= limit:
            break
        if not isinstance(row, (dict, str)):
            raise BatchInputError("row %d must be an object or string" % row_number)
        sample_id = _sample_id_from_row(row, row_number, id_manifest)
        if sample_id in seen:
            raise BatchInputError("duplicate sample_id: %s" % sample_id)
        seen.add(sample_id)
        source_hash = sha256_bytes(raw_line)
        pre_error = None
        pre_error_type = None
        status = row.get("status") if isinstance(row, dict) else None
        # Upstream failures are retained in the audit output but are never sent
        # to DeepSeek with a fabricated placeholder text.
        if isinstance(status, str) and status.lower() not in (
            "success",
            "ok",
            "succeeded",
            "completed",
        ):
            pre_error = "upstream status=%s" % status
            pre_error_type = "upstream_failed"
        try:
            if pre_error is None:
                if system == "captioner":
                    text, text_bytes = _caption_text_from_row(row, input_path.parent, text_field)
                    if not isinstance(text, str) or not text.strip():
                        raise BatchInputError("caption text is empty")
                    sent_text = text
                    # Preserve the actual input bytes for an audit hash.  The
                    # request itself uses the decoded UTF-8 text only.
                    source_hash = sha256_bytes(text_bytes)
                else:
                    sent_text, original_bytes = _sure_text_from_row(
                        row, sample_id, input_path.parent, text_field
                    )
                    if not sent_text.strip():
                        raise BatchInputError("sure structural input is empty")
                    source_hash = sha256_bytes(original_bytes)
            else:
                sent_text = None
        except (BatchInputError, ValueError, OSError) as exc:
            pre_error = redact(exc)
            pre_error_type = "invalid_input"
            sent_text = None
        items.append(
            BatchItem(
                sample_id,
                input_text=sent_text,
                source_index=row_number,
                source_hash=source_hash,
                pre_error=pre_error,
                pre_error_type=pre_error_type,
            )
        )
    if not items:
        raise BatchInputError("no input rows selected")
    if id_manifest is not None and len(id_manifest) != len(rows):
        # A mismatch usually indicates that a tags-only output was truncated;
        # fail before any request is made.
        raise BatchInputError(
            "ID manifest row count %d does not match input row count %d"
            % (len(id_manifest), len(rows))
        )
    return items, input_path


def _item_filename(sample_id):
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".json"


def _load_checkpoints(output_dir):
    result = {}
    item_dir = Path(output_dir) / "items"
    if not item_dir.is_dir():
        return result
    for path in sorted(item_dir.glob("*.json")):
        try:
            value = strict_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sample_id = value.get("sample_id") if isinstance(value, dict) else None
        if isinstance(sample_id, str) and sample_id:
            result[sample_id] = value
    return result


def _checkpoint_is_reusable(record, item, prompt_hash, config_fingerprint, retry_failed):
    if not isinstance(record, dict) or record.get("sample_id") != item.sample_id:
        return False
    if record.get("prompt_sha256") != prompt_hash:
        return False
    if record.get("config_fingerprint") != config_fingerprint:
        return False
    expected_input_hash = (
        sha256_text(item.input_text) if item.input_text is not None else None
    )
    if record.get("input_sha256") != expected_input_hash:
        return False
    status = record.get("status")
    if status == "success":
        payload = record.get("payload")
        try:
            captioner_once.validate_payload(payload, item.sample_id)
        except (captioner_once.ValidationError, ValueError, TypeError):
            return False
        return True
    # Failed checkpoints are terminal by default unless their recorded error
    # is explicitly a network/transient failure.  ``--no-retry-failed`` keeps
    # the historical opt-out and reuses every failed checkpoint.
    return status == "failed" and (not retry_failed or not checkpoint_retryable(record))


def _failure_record(item, prompt_hash, config_fingerprint, error, attempts=0, elapsed=0.0, error_type=None):
    return {
        "sample_id": item.sample_id,
        "status": "failed",
        "error_type": error_type or "api_error",
        "error": redact(error),
        "attempts": int(attempts),
        "elapsed_sec": round(float(elapsed), 3),
        "input_sha256": sha256_text(item.input_text or "") if item.input_text is not None else None,
        "source_sha256": item.source_hash,
        "prompt_sha256": prompt_hash,
        "config_fingerprint": config_fingerprint,
    }


def _success_record(
    item,
    payload,
    prompt_hash,
    config_fingerprint,
    usage,
    attempts,
    elapsed,
    verification=None,
):
    record = {
        "sample_id": item.sample_id,
        "status": "success",
        "payload": payload,
        "input_sha256": sha256_text(item.input_text),
        "source_sha256": item.source_hash,
        "prompt_sha256": prompt_hash,
        "config_fingerprint": config_fingerprint,
        "usage": usage if isinstance(usage, dict) else {},
        "attempts": int(attempts),
        "elapsed_sec": round(float(elapsed), 3),
    }
    # Stored with the checkpoint, not only in the sidecar, so a resumed run can
    # rebuild the audit file instead of losing every verdict it did not redo.
    if verification is not None:
        record["language_verification"] = verification
    return record


def _captioner_payload(raw, sample_id):
    if not isinstance(raw, dict):
        raise captioner_once.ValidationError("DeepSeek output must be an object")
    payload = copy.deepcopy(raw)
    payload["sample_id"] = sample_id
    captioner_once.canonicalize_descriptive_evidence(payload)
    captioner_once.validate_payload(payload, sample_id)
    return payload


def _sure_payload(raw, structural, sample_id, require_language_dependency_review=False):
    """Build the published envelope for one sure-tagger sample.

    Returns ``(candidate, language_verification_row)``.  The row is audit
    evidence only: it never reaches the published representation, but the
    merge gate requires it before it will accept a changed
    ``language_content.language``.
    """
    descriptive, verification = sure_once.split_sure_tagger_output(raw)
    candidate = {
        "sample_id": sample_id,
        "schema_version": SCHEMA_VERSION,
        "field_universe_version": FIELD_UNIVERSE_VERSION,
        "basic_acoustic": copy.deepcopy(structural["basic_acoustic"]),
        "audio_quality": copy.deepcopy(structural["audio_quality"]),
        "room_acoustic": copy.deepcopy(structural["room_acoustic"]),
        "sound_field_scene": copy.deepcopy(structural["sound_field_scene"]),
        "speaker": copy.deepcopy(structural["speaker"]),
        "language_content": copy.deepcopy(structural["language_content"]),
        "deterministic_derived": copy.deepcopy(structural["deterministic_derived"]),
        "descriptive": descriptive,
    }
    row = sure_once.language_verification_row(
        structural, verification, sample_id,
        require_dependency_review=require_language_dependency_review,
    )
    if row["verdict_state"] == "pending_dependency_review":
        candidate["descriptive"] = {key: None for key in sure_once.DESCRIPTIVE_ORDER}
    # The sanctioned language override must land before evidence canonicalization,
    # because ``evidence_tags`` may cite the published language.
    candidate["language_content"]["language"] = row["published_language"]
    captioner_once.canonicalize_descriptive_evidence(candidate)
    captioner_once.validate_payload(candidate, sample_id)
    return candidate, row


def _retry_delay(attempt, base, maximum, jitter):
    delay = min(float(maximum), float(base) * (2.0 ** max(0, attempt - 1)))
    if jitter:
        delay += random.random() * min(1.0, delay * 0.1)
    return delay


def process_item(
    item,
    system,
    config,
    prompt,
    prompt_hash,
    config_fingerprint,
    timeout,
    retries,
    max_tokens,
    thinking,
    backoff_base,
    backoff_max,
    jitter,
    require_language_dependency_review=False,
):
    started = time.time()
    if item.pre_error is not None:
        return _failure_record(
            item,
            prompt_hash,
            config_fingerprint,
            item.pre_error,
            error_type=item.pre_error_type or "invalid_input",
        )

    errors = []
    error_type = "api_error"
    attempts = 0
    while attempts <= retries:
        attempts += 1
        try:
            raw, usage = captioner_once.call_api(
                config["api_key"],
                captioner_once.endpoint(config["base_url"]),
                config["model"],
                prompt,
                item.input_text,
                timeout,
                max_tokens,
                thinking,
            )
        except Exception as exc:  # Transport and provider errors vary widely.
            retryable = is_network_failure(exc)
            error_type = "network_error" if retryable else "api_error"
            errors.append(redact(exc, config.get("api_key")))
            # Only the API call itself is eligible for network retries.  Local
            # parsing/validation below is deterministic for a given response.
            if not retryable or attempts > retries:
                break
            time.sleep(_retry_delay(attempts, backoff_base, backoff_max, jitter))
            continue

        try:
            verification = None
            if system == "captioner":
                payload = _captioner_payload(raw, item.sample_id)
            else:
                # Structural input was validated and canonicalized during load.
                structural_value = strict_json(item.input_text)
                structural = _validate_and_canonicalize_structural(
                    structural_value, item.sample_id
                )
                payload, verification = _sure_payload(
                    raw, structural, item.sample_id,
                    require_language_dependency_review=require_language_dependency_review,
                )
        except Exception as exc:
            # Once a response has arrived, parsing/schema failures are local
            # deterministic errors.  Keep them distinct from transport errors
            # so a message containing words such as ``timeout`` cannot trigger
            # a later network retry.
            errors.append(redact(exc, config.get("api_key")))
            error_type = "validation_error"
            break
        return _success_record(
            item,
            payload,
            prompt_hash,
            config_fingerprint,
            usage,
            attempts,
            time.time() - started,
            verification=verification,
        )
    return _failure_record(
        item,
        prompt_hash,
        config_fingerprint,
        "; ".join(errors[-4:]) or "request failed",
        attempts=attempts,
        elapsed=time.time() - started,
        error_type=error_type,
    )


def _write_checkpoint(output_dir, record):
    item_dir = Path(output_dir) / "items"
    atomic_write_json(item_dir / _item_filename(record["sample_id"]), record)


def _aggregate(items, records, output_dir, system, config, prompt_hash, config_fingerprint, run_meta):
    by_id = {record.get("sample_id"): record for record in records if isinstance(record, dict)}
    # Keep the aggregate one-row-per-input even if an unexpected worker-level
    # exception prevented a result from being returned.  Such rows are marked
    # failed rather than silently disappearing from coverage accounting.
    ordered = []
    for item in items:
        record = by_id.get(item.sample_id)
        if record is None:
            record = _failure_record(
                item,
                prompt_hash,
                config_fingerprint,
                "worker returned no result",
                error_type="internal_missing_result",
            )
        ordered.append(record)
    successes = [record for record in ordered if record.get("status") == "success"]
    failures = [record for record in ordered if record.get("status") != "success"]
    output_dir = Path(output_dir)
    atomic_write_text(
        output_dir / "results.jsonl",
        "".join(
            json.dumps(record["payload"], ensure_ascii=False, sort_keys=True) + "\n"
            for record in successes
        ),
    )
    atomic_write_text(
        output_dir / "failures.jsonl",
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in failures),
    )
    atomic_write_text(
        output_dir / "status.jsonl",
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered),
    )
    summary = dict(run_meta)
    summary.update(
        {
            "system": system,
            "sample_count": len(items),
            "success_count": len(successes),
            "failed_count": len(failures),
            "failed_sample_ids": [record["sample_id"] for record in failures],
            "results_path": str((output_dir / "results.jsonl").resolve()),
            "failures_path": str((output_dir / "failures.jsonl").resolve()),
            "status_path": str((output_dir / "status.jsonl").resolve()),
            "prompt_sha256": prompt_hash,
            "config_fingerprint": config_fingerprint,
            "model": config["model"],
            "endpoint": captioner_once.endpoint(config["base_url"]),
        }
    )
    language_block = _language_verification_summary(successes, output_dir)
    if language_block is not None:
        summary["language_verification"] = language_block
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _language_verification_summary(successes, output_dir):
    """Write the per-sample language verdict audit file and summarize it.

    Returns ``None`` for systems that carry no verdict, so the captioner run
    keeps its summary shape unchanged.
    """
    rows = [
        record["language_verification"]
        for record in successes
        if isinstance(record.get("language_verification"), dict)
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: str(row.get("sample_id")))
    path = Path(output_dir) / "language_verification.jsonl"
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    states = {}
    transitions = {}
    for row in rows:
        state = row.get("verdict_state")
        states[state] = states.get(state, 0) + 1
        if row.get("changed"):
            key = "%s -> %s" % (row.get("lid_language"), row.get("published_language"))
            transitions[key] = transitions.get(key, 0) + 1
    ordered = sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "verdict_count": len(rows),
        "changed_count": sum(1 for row in rows if row.get("changed")),
        "state_counts": states,
        "transition_counts": dict(ordered[:20]),
        "transition_kind_count": len(ordered),
        "path": str(path.resolve()),
    }


def resolve_prompt_path(system, prompt=None, prompt_version=None):
    """Return ``(prompt_path, version_label)`` for one batch.

    ``version_label`` is ``v1`` for the sure-tagger pack, ``custom``
    when ``--prompt`` points at another file, and ``None`` for captioner.
    An explicit ``--prompt`` always wins over ``--prompt-version``.
    """
    if prompt is not None:
        path = Path(prompt).resolve()
        if system != "sure-tagger":
            if prompt_version not in (None, ""):
                raise ValueError("--prompt-version is only valid for --system sure-tagger")
            return path, None
        for name, candidate in SURE_TAGGER_PROMPT_VERSIONS.items():
            if candidate.resolve() == path:
                return path, name
        return path, "custom"
    if system != "sure-tagger":
        if prompt_version not in (None, ""):
            raise ValueError("--prompt-version is only valid for --system sure-tagger")
        return DEFAULT_PROMPTS[system].resolve(), None
    version = prompt_version or DEFAULT_SURE_TAGGER_PROMPT_VERSION
    if version not in SURE_TAGGER_PROMPT_VERSIONS:
        raise ValueError("unknown sure-tagger prompt version: %s" % version)
    return SURE_TAGGER_PROMPT_VERSIONS[version].resolve(), version


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEM_ALIASES))
    parser.add_argument("--input-manifest", "--input", dest="input_manifest", required=True, type=Path)
    parser.add_argument("--id-manifest", type=Path, default=None)
    parser.add_argument(
        "--text-field",
        default=None,
        help="dotted field containing the sole upstream text (default: auto-detect input_text/caption)",
    )
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument(
        "--prompt-version",
        default=None,
        choices=sorted(SURE_TAGGER_PROMPT_VERSIONS),
        help=(
            "sure-tagger prompt pack. Default is v1, the published "
            "language+descriptive prompt. Overridden by --prompt."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--require-language-dependency-review", action="store_true",
        help="For plus outputs, feed language verdicts back to the evidence loop and withhold descriptions until reviewed.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="ignore existing checkpoints")
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="reuse all failed checkpoints (default retries only network/transient failures)",
    )
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--backoff-max", type=float, default=30.0)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs/config without API requests")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    system = normalize_system(args.system)
    if args.workers < 1 or args.retries < 0 or args.timeout < 1 or args.max_tokens < 1:
        raise SystemExit("workers/retries/timeout/max-tokens values are invalid")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")
    if args.backoff_base < 0 or args.backoff_max < 0:
        raise SystemExit("backoff values must be non-negative")

    try:
        prompt_path, prompt_version = resolve_prompt_path(
            system, args.prompt, args.prompt_version
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIRS[system].resolve()
    api_file = Path(args.api_file).resolve()
    if not prompt_path.is_file():
        raise SystemExit("prompt file not found: %s" % prompt_path)
    if not api_file.is_file():
        raise SystemExit("API file not found: %s" % api_file)

    try:
        items, input_path = load_items(
            system,
            args.input_manifest,
            id_manifest_path=args.id_manifest,
            text_field=args.text_field,
            limit=args.limit,
        )
    except (BatchInputError, OSError, ValueError) as exc:
        raise SystemExit("input validation failed: %s" % redact(exc))

    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_hash = sha256_text(prompt)
    try:
        config = captioner_once.read_api_file(api_file)
    except (OSError, ValueError) as exc:
        raise SystemExit("API configuration failed: %s" % redact(exc))

    # Fingerprint only request-affecting values; the secret itself is never
    # persisted.  This prevents accidentally mixing runs with another model,
    # prompt, or thinking mode when resuming.
    fingerprint_value = {
        "system": system,
        "model": config["model"],
        "endpoint": captioner_once.endpoint(config["base_url"]),
        "prompt_sha256": prompt_hash,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    if args.require_language_dependency_review:
        fingerprint_value["require_language_dependency_review"] = True
    config_fingerprint = sha256_text(
        json.dumps(fingerprint_value, sort_keys=True, separators=(",", ":"))
    )
    input_hash = sha256_bytes(input_path.read_bytes()) if input_path.is_file() else None
    run_meta = {
        "input_manifest": str(input_path),
        "input_manifest_sha256": input_hash,
        "input_contract": "one_upstream_text_value_per_request",
        "prompt_path": str(prompt_path),
        "prompt_version": prompt_version,
        "workers": args.workers,
        "retries": args.retries,
        "timeout_sec": args.timeout,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "require_language_dependency_review": args.require_language_dependency_review,
        "source_count": len(items),
        "api_request_count": 0,
        "resume_skipped_count": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Read any prior configuration before replacing it.  A changed prompt,
    # model, endpoint, or request mode must not silently reuse old payloads.
    if not args.overwrite:
        previous_config_path = output_dir / "run_config.json"
        if previous_config_path.is_file():
            try:
                previous_config = strict_json(previous_config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_config = None
            previous_fingerprint = (
                previous_config.get("config_fingerprint")
                if isinstance(previous_config, dict)
                else None
            )
            if previous_fingerprint and previous_fingerprint != config_fingerprint:
                raise SystemExit(
                    "existing run_config fingerprint differs; use --overwrite or a new --output-dir"
                )
    atomic_write_json(
        output_dir / "run_config.json",
        dict(run_meta, **fingerprint_value, config_fingerprint=config_fingerprint),
    )

    if args.dry_run:
        invalid = [item.sample_id for item in items if item.pre_error is not None]
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "system": system,
                    "sample_count": len(items),
                    "invalid_input_count": len(invalid),
                    "invalid_sample_ids": invalid,
                    "workers": args.workers,
                    "input_contract": run_meta["input_contract"],
                    "prompt_sha256": prompt_hash,
                    "prompt_version": prompt_version,
                    "prompt_path": str(prompt_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if not invalid else 2

    checkpoints = {} if args.overwrite else _load_checkpoints(output_dir)
    records = []
    pending = []
    retry_failed = not args.no_retry_failed
    for item in items:
        old = checkpoints.get(item.sample_id)
        if old is not None and _checkpoint_is_reusable(
            old, item, prompt_hash, config_fingerprint, retry_failed
        ):
            records.append(old)
            run_meta["resume_skipped_count"] += 1
        else:
            pending.append(item)

    print(
        "DeepSeek batch: system=%s samples=%d pending=%d workers=%d" 
        % (system, len(items), len(pending), args.workers),
        flush=True,
    )
    lock = threading.Lock()
    completed = 0
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    process_item,
                    item,
                    system,
                    config,
                    prompt,
                    prompt_hash,
                    config_fingerprint,
                    args.timeout,
                    args.retries,
                    args.max_tokens,
                    args.thinking,
                    args.backoff_base,
                    args.backoff_max,
                    not args.no_jitter,
                    args.require_language_dependency_review,
                )
                for item in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                _write_checkpoint(output_dir, record)
                with lock:
                    records.append(record)
                    completed += 1
                    run_meta["api_request_count"] += record.get("attempts", 0)
                    print(
                        "%d/%d %s %s" 
                        % (completed, len(pending), record["sample_id"], record["status"]),
                        flush=True,
                    )

    summary = _aggregate(
        items,
        records,
        output_dir,
        system,
        config,
        prompt_hash,
        config_fingerprint,
        run_meta,
    )
    # Keep the resumable configuration in sync with the final audit counts.
    atomic_write_json(
        output_dir / "run_config.json",
        dict(
            run_meta,
            **fingerprint_value,
            config_fingerprint=config_fingerprint,
            success_count=summary["success_count"],
            failed_count=summary["failed_count"],
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: %s" % redact(exc), file=sys.stderr)
        raise SystemExit(1)
