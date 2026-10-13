#!/usr/bin/env python3
"""Run Qwen Omni captions for the structural benchmark.

The benchmark manifest contains ground-truth and native metadata for audit
purposes.  This runner deliberately reads only ``sample.sample_id`` and
``sample.audio.path`` from each row.  By default the request sent to Qwen
contains exactly one audio part (no text, filename, transcript, metadata, or
reference label).  A fixed task instruction can be enabled explicitly with
``--with-prompt`` when an endpoint requires one.

Each sample is written to an individual JSON/TXT pair with atomic replacement.
Successful pairs are resumed only when their audio and prompt fingerprints
still match the current run.  ``results.jsonl`` and ``summary.json`` are
re-built from the manifest order after every invocation, so interruption is
safe and a later invocation retries only failed samples with explicit
network/transient errors; deterministic failures are recorded once.
"""

from __future__ import print_function

import argparse
import base64
import concurrent.futures
import hashlib
import ipaddress
import json
import mimetypes
import os
import random
import re
import socket
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_MANIFEST = DEFAULT_BENCHMARK_ROOT / "benchmark.manifest.jsonl"
DEFAULT_OUTPUT = COMPARE_ROOT / "evaluation" / "benchmark-v1" / "captioner_raw"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"

DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
# The native DashScope service accepts local media after the SDK-style OSS
# upload flow.  Keep this separate from ``--endpoint`` so existing compatible
# checkpoints remain resumable when the transport is switched.
DEFAULT_NATIVE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)
DEFAULT_NATIVE_UPLOAD_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/uploads"
DEFAULT_MODEL = "qwen-omni-turbo"

# This is a fixed task instruction, not sample-specific input.  It is disabled
# by default to honor the strict audio-only input contract.  In particular, do
# not interpolate paths, ids, transcripts, or metadata into this string.
DEFAULT_PROMPT = (
    "Listen to the entire audio clip and write a detailed, objective English "
    "audio caption. Describe audible events in temporal order, speakers and "
    "speaker count only when clearly supported, language only when clearly "
    "supported, music, background and environment, room acoustics, recording "
    "quality, and relevant uncertainty. Do not invent identities, location, "
    "culture, or facts. Do not produce a verbatim transcript; summarize speech "
    "content only when it is useful to characterize the audio. Output caption "
    "text only, with no Markdown, JSON, headings, or prefatory commentary."
)

RETRYABLE_STATUS = set((408, 409, 425, 429, 500, 502, 503, 504))
TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._-]{20,}")
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
NETWORK_ERROR_RE = re.compile(
    r"(?i)(?:"
    r"\bhttp\s*(?:408|409|425|429|5\d\d)\b"
    r"|timeout|timed\s*out|"
    r"remote\s*(?:disconnected|end\s+closed)"
    r"|connection\s*(?:aborted|reset|refused|closed|error|failure|timed\s*out)"
    r"|connection(?:aborted|reset|refused|closed|error)"
    r"|connect(?:timeout|error)"
    r"|broken\s+pipe|socket\s+error|"
    r"ssl(?:error)?|ssleof|unexpected[_\s-]*eof|\beof\b|"
    r"temporary\s+failure|name\s+(?:or\s+service|resolution)|"
    r"max\s+retries\s+exceeded|network\s+unreachable|no\s+route|"
    r"gaierror"
    r")"
)
HTTP_STATUS_RE = re.compile(r"(?i)\b(?:http|status|code)\s*[:=]?\s*(\d{3})\b")


class RequestFailure(Exception):
    """An HTTP or response-level failure, with retry policy attached."""

    def __init__(self, message, retryable=True, status=None):
        Exception.__init__(self, message)
        self.retryable = bool(retryable)
        self.status = status


def is_network_failure(error):
    """Return whether an error is safe to retry as a transient transport fault."""

    status = getattr(error, "status", None)
    if not isinstance(status, int):
        match = HTTP_STATUS_RE.search(str(error))
        if match:
            try:
                status = int(match.group(1))
            except (TypeError, ValueError):
                status = None
    if isinstance(status, int):
        # A known HTTP status takes precedence over words in its response
        # body (for example, HTTP 400 with a message mentioning "timeout").
        return status in RETRYABLE_STATUS or status >= 500
    return NETWORK_ERROR_RE.search(str(error)) is not None


def _response_message(payload):
    """Extract a short provider error message without assuming a schema."""

    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if message:
            return str(message)
    for key in ("message", "code"):
        if payload.get(key):
            return str(payload[key])
    return ""


class NativeOssUploader(object):
    """Upload one local audio file through DashScope's native OSS flow.

    The official SDK obtains a short-lived upload certificate, posts a
    multipart form to the returned OSS host, then sends the resulting
    ``oss://`` URI to the native multimodal-generation endpoint.  This small
    implementation keeps the runner's stdlib-only import surface intact and
    lazily uses ``requests`` only when ``--native-oss`` is selected.
    """

    REQUIRED_CERTIFICATE_FIELDS = (
        "upload_host",
        "upload_dir",
        "oss_access_key_id",
        "signature",
        "policy",
        "x_oss_object_acl",
        "x_oss_forbid_overwrite",
    )

    def __init__(self, model, upload_endpoint, api_key, timeout):
        self.model = model
        self.upload_endpoint = upload_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._certificate = None
        self._certificate_obtained_at = None
        self._certificate_lock = threading.Lock()
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._session_local = threading.local()

    def _session(self):
        """Return one requests session per worker thread for pooling."""

        session = getattr(self._session_local, "session", None)
        if session is None:
            try:
                import requests
            except ImportError as error:
                raise RequestFailure(
                    "native-oss requires the requests package: %s" % error,
                    retryable=False,
                )
            session = requests.Session()
            self._session_local.session = session
        return session

    def _get_certificate(self, force=False):
        with self._certificate_lock:
            if self._certificate is not None and not force:
                # DashScope certificates are short-lived (normally 300 s).
                # Do not begin a large upload with a certificate that is near
                # expiry; a slow/retransmitted upload can otherwise fail after
                # the policy expires and every retry would reuse the same one.
                obtained_at = self._certificate_obtained_at
                try:
                    lifetime = float(self._certificate.get("expire_in_seconds", 300))
                except (TypeError, ValueError):
                    lifetime = 300.0
                safety_window = min(60.0, max(15.0, lifetime * 0.2))
                if (
                    obtained_at is not None
                    and time.monotonic() - obtained_at < max(1.0, lifetime - safety_window)
                ):
                    return self._certificate
                self._certificate = None
                self._certificate_obtained_at = None
            try:
                response = self._session().get(
                    self.upload_endpoint,
                    params={"action": "getPolicy", "model": self.model},
                    headers={
                        "Authorization": "Bearer " + self.api_key,
                        "Accept": "application/json",
                        "User-Agent": "benchmark-qwen-omni-captioner/1.0",
                    },
                    timeout=self.timeout,
                )
            except Exception as error:
                raise RequestFailure(
                    "native OSS policy request failed: %s"
                    % redact(error, self.api_key),
                    retryable=True,
                )
            try:
                payload = response.json()
            except Exception:
                payload = {}
            if response.status_code != 200:
                raise RequestFailure(
                    "native OSS policy HTTP %s: %s"
                    % (
                        response.status_code,
                        redact(_response_message(payload) or response.text[:300], self.api_key),
                    ),
                    retryable=response.status_code in RETRYABLE_STATUS
                    or response.status_code >= 500,
                    status=response.status_code,
                )
            certificate = payload.get("output") if isinstance(payload, dict) else None
            if not isinstance(certificate, dict) and isinstance(payload, dict):
                certificate = payload.get("data")
            if not isinstance(certificate, dict):
                raise RequestFailure(
                    "native OSS policy response has no certificate",
                    retryable=True,
                    status=response.status_code,
                )
            missing = [
                key
                for key in self.REQUIRED_CERTIFICATE_FIELDS
                if not certificate.get(key)
            ]
            if missing:
                raise RequestFailure(
                    "native OSS policy certificate missing fields: %s"
                    % ",".join(missing),
                    retryable=True,
                    status=response.status_code,
                )
            self._certificate = certificate
            self._certificate_obtained_at = time.monotonic()
            return certificate

    def _invalidate_certificate(self):
        """Force the next upload attempt to obtain a fresh short-lived policy."""

        with self._certificate_lock:
            self._certificate = None
            self._certificate_obtained_at = None

    def _cache_key(self, audio_path):
        try:
            stat = audio_path.stat()
            return (str(audio_path.resolve()), stat.st_size, stat.st_mtime)
        except OSError:
            return (str(audio_path), None, None)

    def _cached(self, cache_key):
        with self._cache_lock:
            return self._cache.get(cache_key)

    def _remember(self, cache_key, uri):
        with self._cache_lock:
            self._cache[cache_key] = uri

    def upload(self, audio_path):
        """Return an OSS URI for ``audio_path``, reusing uploads on retries."""

        cache_key = self._cache_key(audio_path)
        cached = self._cached(cache_key)
        if cached:
            return cached

        # A fresh certificate is useful after a transient/expired policy.  Do
        # not retry an arbitrary number of multipart uploads: the caller's
        # normal sample retry loop handles network failures.
        # Keep the object name opaque: the model receives the OSS URI, so a
        # basename containing a sample ID would otherwise become metadata in
        # the nominally audio-only request.
        object_name = uuid.uuid4().hex + ".wav"
        for certificate_attempt in range(2):
            certificate = self._get_certificate(force=certificate_attempt > 0)
            key = str(certificate["upload_dir"]).rstrip("/") + "/" + object_name
            form_data = {
                "OSSAccessKeyId": certificate["oss_access_key_id"],
                "Signature": certificate["signature"],
                "policy": certificate["policy"],
                "key": key,
                "x-oss-object-acl": certificate["x_oss_object_acl"],
                "x-oss-forbid-overwrite": certificate["x_oss_forbid_overwrite"],
                "success_action_status": "200",
                "x-oss-content-type": mimetypes.guess_type(str(audio_path))[0]
                or "application/octet-stream",
            }
            try:
                with audio_path.open("rb") as stream:
                    response = self._session().post(
                        certificate["upload_host"],
                        files={
                            "file": (
                                object_name,
                                stream,
                                form_data["x-oss-content-type"],
                            )
                        },
                        data=form_data,
                        headers={
                            "Accept": "application/json",
                            "Date": format_http_date(),
                            "User-Agent": "benchmark-qwen-omni-captioner/1.0",
                        },
                        timeout=self.timeout,
                    )
            except RequestFailure:
                raise
            except Exception as error:
                # Network disconnects and socket write timeouts do not carry an
                # HTTP status, but they commonly happen after a short-lived OSS
                # policy has expired.  Never reuse that policy on the retry.
                self._invalidate_certificate()
                raise RequestFailure(
                    "native OSS upload failed: %s" % redact(error, self.api_key),
                    retryable=True,
                )
            if response.status_code == 200:
                uri = "oss://" + key
                self._remember(cache_key, uri)
                return uri
            try:
                payload = response.json()
            except Exception:
                payload = {}
            # A stale policy commonly appears as 403/409.  Refresh it once;
            # other statuses follow the normal retry policy.
            if response.status_code in (401, 403, 409) and certificate_attempt == 0:
                self._invalidate_certificate()
                continue
            if response.status_code != 200:
                self._invalidate_certificate()
            raise RequestFailure(
                "native OSS upload HTTP %s: %s"
                % (
                    response.status_code,
                    redact(_response_message(payload) or response.text[:300], self.api_key),
                ),
                retryable=response.status_code in RETRYABLE_STATUS
                or response.status_code >= 500,
                status=response.status_code,
            )
        raise RequestFailure("native OSS upload failed after refreshing policy", retryable=True)


def format_http_date():
    """Return an RFC-1123 date used by the signed OSS multipart request."""

    from email.utils import formatdate

    return formatdate(usegmt=True)


def build_native_request_body(oss_uri, model, max_tokens=1400):
    """Build the native payload with exactly one user audio content item."""

    if not isinstance(oss_uri, str) or not oss_uri.startswith("oss://"):
        raise ValueError("native audio reference must be an oss:// URI")
    body = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"audio": oss_uri}],
                }
            ]
        },
        "parameters": {
            "result_format": "message",
            "temperature": 0.2,
            "max_tokens": int(max_tokens),
        },
    }
    assert_native_request_body(body)
    return body


def assert_native_request_body(body):
    """Fail fast if native mode ever leaks text or sample metadata."""

    if set(body) != {"model", "input", "parameters"}:
        raise AssertionError("unexpected native top-level request fields")
    input_data = body.get("input")
    if not isinstance(input_data, dict) or set(input_data) != {"messages"}:
        raise AssertionError("unexpected native input fields")
    messages = input_data["messages"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise AssertionError("native request must contain one message")
    message = messages[0]
    if set(message) != {"role", "content"} or message.get("role") != "user":
        raise AssertionError("unexpected native message fields")
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise AssertionError("native request must contain one content item")
    item = content[0]
    if not isinstance(item, dict) or set(item) != {"audio"}:
        raise AssertionError("native request content must be audio-only")
    if not isinstance(item["audio"], str) or not item["audio"].startswith("oss://"):
        raise AssertionError("native audio content must be an OSS URI")
    parameters = body.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {
        "result_format",
        "temperature",
        "max_tokens",
    }:
        raise AssertionError("unexpected native parameters")


def native_content_text(response_json):
    """Extract assistant text from a native multimodal response."""

    output = response_json.get("output") if isinstance(response_json, dict) else None
    if not isinstance(output, dict):
        return "", None
    choices = output.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message") or {}
        if isinstance(message, dict):
            return content_text(message.get("content")), choice.get("finish_reason")
    return content_text(output.get("text")), output.get("finish_reason")


def request_caption_native_oss(
    audio_path,
    key,
    endpoint,
    model,
    prompt,
    timeout,
    max_tokens,
    native_uploader=None,
):
    """Upload one audio clip and invoke native DashScope generation."""

    if prompt is not None:
        raise RequestFailure(
            "native-oss only supports the strict audio-only input contract",
            retryable=False,
        )
    if native_uploader is None:
        native_uploader = NativeOssUploader(
            model=model,
            upload_endpoint=DEFAULT_NATIVE_UPLOAD_ENDPOINT,
            api_key=key,
            timeout=timeout,
        )
    oss_uri = native_uploader.upload(audio_path)
    body = build_native_request_body(oss_uri, model, max_tokens=max_tokens)
    try:
        response = native_uploader._session().post(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json; charset=utf-8",
                "X-DashScope-OssResourceResolve": "enable",
                "User-Agent": "benchmark-qwen-omni-captioner/1.0",
            },
            timeout=timeout,
        )
    except Exception as error:
        raise RequestFailure(
            "native generation request failed: %s" % redact(error, key),
            retryable=True,
        )
    try:
        response_json = response.json()
    except Exception:
        response_json = {}
    if response.status_code != 200:
        raise RequestFailure(
            "native generation HTTP %s: %s"
            % (
                response.status_code,
                redact(_response_message(response_json) or response.text[:300], key),
            ),
            retryable=response.status_code in RETRYABLE_STATUS
            or response.status_code >= 500,
            status=response.status_code,
        )
    caption, finish_reason = native_content_text(response_json)
    if not caption:
        raise RequestFailure(
            "native response content is empty",
            retryable=True,
            status=response.status_code,
        )
    if finish_reason not in (None, "stop"):
        raise RequestFailure(
            "finish_reason=%s" % finish_reason,
            retryable=True,
            status=response.status_code,
        )
    return caption, response_json


def parse_endpoint_ips(value):
    """Parse a comma-separated list of IPv4 literals, preserving order."""

    if value is None:
        return []
    addresses = []
    seen = set()
    for raw in str(value).split(","):
        candidate = raw.strip()
        if not candidate:
            raise ValueError("endpoint IP list contains an empty entry")
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            raise ValueError("endpoint IP is invalid: %s" % candidate)
        if address.version != 4:
            raise ValueError("endpoint IP must be IPv4: %s" % candidate)
        normalized = str(address)
        if normalized not in seen:
            addresses.append(normalized)
            seen.add(normalized)
    if not addresses:
        raise ValueError("endpoint IP list is empty")
    return addresses


class EndpointIPv4Resolver(object):
    """Pin or rotate IPv4 addresses for one endpoint without changing TLS SNI.

    urllib still receives the original hostname in the URL, so HTTPS certificate
    validation and SNI remain tied to that hostname.  Only the address returned
    by ``getaddrinfo`` for the selected endpoint is replaced.
    """

    def __init__(self, endpoint, addresses):
        parsed = urllib.parse.urlsplit(endpoint)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("endpoint has no hostname for IPv4 resolution")
        self.hostname = hostname.rstrip(".").lower()
        if isinstance(addresses, str):
            addresses = [addresses]
        self.addresses = tuple(parse_endpoint_ips(",".join(addresses)))
        self._lock = threading.Lock()
        self._next = 0
        self._previous = None
        self._original = socket.getaddrinfo

    def _select(self):
        with self._lock:
            address = self.addresses[self._next % len(self.addresses)]
            self._next += 1
        return address

    def __call__(self, host, port, family=0, socktype=0, proto=0, flags=0):
        normalized = host.decode("idna") if isinstance(host, bytes) else host
        if (
            isinstance(normalized, str)
            and normalized.rstrip(".").lower() == self.hostname
            and family in (0, socket.AF_INET)
            and socktype in (0, socket.SOCK_STREAM)
        ):
            selected = self._select()
            return [
                (
                    socket.AF_INET,
                    socktype or socket.SOCK_STREAM,
                    proto or socket.IPPROTO_TCP,
                    "",
                    (selected, port),
                )
            ]
        resolver = self._previous or self._original
        return resolver(host, port, family, socktype, proto, flags)

    def install(self):
        """Install this resolver for the current process and return self."""

        if self._previous is not None:
            raise RuntimeError("endpoint resolver is already installed")
        self._previous = socket.getaddrinfo
        socket.getaddrinfo = self
        return self

    def restore(self):
        """Restore the process-wide resolver after all worker threads finish."""

        if self._previous is not None:
            previous = self._previous
            self._previous = None
            if socket.getaddrinfo is self:
                socket.getaddrinfo = previous


def redact(value, secret):
    text = str(value)
    if secret:
        text = text.replace(secret, "<redacted>")
    return TOKEN_PATTERN.sub("<redacted>", text)


def read_api_key(path):
    """Extract exactly the key token, ignoring labels and trailing notes."""

    raw = path.read_text(encoding="utf-8")
    match = TOKEN_PATTERN.search(raw)
    if match is None:
        raise ValueError("api file does not contain a valid Bailian key")
    return match.group(0)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def load_manifest(path, benchmark_root, audio_root=None):
    """Load and validate the audio index without retaining metadata fields."""

    rows = []
    seen_ids = set()
    seen_paths = set()
    root_resolved = benchmark_root.resolve()
    staging_resolved = audio_root.resolve() if audio_root is not None else None

    def under(path_value, root_value):
        if root_value is None:
            return False
        try:
            path_value.relative_to(root_value)
            return True
        except ValueError:
            return False
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as error:
                raise ValueError("invalid JSON at manifest line %d: %s" % (line_number, error))
            try:
                sample = raw["sample"]
                sample_id = sample["sample_id"]
                relative_audio = sample["audio"]["path"]
            except (KeyError, TypeError):
                raise ValueError(
                    "manifest line %d lacks sample.sample_id/audio.path" % line_number
                )
            if not isinstance(sample_id, str) or not SAMPLE_ID_PATTERN.match(sample_id):
                raise ValueError("invalid sample_id at manifest line %d" % line_number)
            if sample_id in seen_ids:
                raise ValueError("duplicate sample_id: %s" % sample_id)
            if not isinstance(relative_audio, str) or not relative_audio:
                raise ValueError("invalid audio path for %s" % sample_id)
            relative_path = Path(relative_audio)
            if not relative_path.is_absolute() and ".." in relative_path.parts:
                raise ValueError("unsafe audio path for %s: %s" % (sample_id, relative_audio))
            candidate = (
                relative_path
                if relative_path.is_absolute()
                else benchmark_root / relative_path
            )
            if (
                not relative_path.is_absolute()
                and audio_root is not None
                and not candidate.exists()
            ):
                candidate = audio_root / relative_path
            # Runtime manifests produced by prepare_benchmark_runtime.py use
            # absolute paths to flat symlinks.  Resolve those links, while
            # accepting an explicitly supplied staging root as a trusted input
            # location too.
            lexical_candidate = Path(os.path.abspath(str(candidate)))
            audio_path = candidate.resolve()
            if not (
                under(audio_path, root_resolved)
                or under(lexical_candidate, staging_resolved)
            ):
                raise ValueError("audio path escapes allowed roots for %s" % sample_id)
            if not audio_path.is_file():
                raise ValueError("audio file missing for %s: %s" % (sample_id, audio_path))
            if audio_path in seen_paths:
                raise ValueError("duplicate audio path: %s" % audio_path)
            seen_ids.add(sample_id)
            seen_paths.add(audio_path)
            # Keep only fields needed by the runner.  Never pass ``raw`` onward
            # to request construction, which prevents accidental metadata leaks.
            rows.append({"sample_id": sample_id, "audio_path": audio_path})
    if not rows:
        raise ValueError("manifest has no samples")
    return rows


def build_request_body(audio_bytes, prompt, model, max_tokens=1400):
    """Build a request whose only sample data is one inline audio clip.

    ``prompt=None`` is the strict mode: the content array has exactly one
    ``input_audio`` part.  A non-None prompt is an explicit compatibility mode
    for providers that reject an empty text instruction.
    """

    encoded = base64.b64encode(audio_bytes).decode("ascii")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64," + encoded,
                            "format": "wav",
                        },
                    }
                ],
            }
        ],
        "modalities": ["text"],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": int(max_tokens),
    }
    if prompt is not None:
        body["messages"][0]["content"].append({"type": "text", "text": prompt})
    assert_request_body(body, prompt)
    return body


def assert_request_body(body, prompt):
    """Fail fast if a future edit adds sample-specific fields to the payload."""

    if set(body) != {
        "model",
        "messages",
        "modalities",
        "stream",
        "temperature",
        "max_tokens",
    }:
        raise AssertionError("unexpected top-level request fields")
    if not isinstance(body["messages"], list) or len(body["messages"]) != 1:
        raise AssertionError("request must contain exactly one user message")
    message = body["messages"][0]
    if set(message) != {"role", "content"} or message.get("role") != "user":
        raise AssertionError("unexpected message fields")
    content = message.get("content")
    expected_parts = 1 if prompt is None else 2
    if not isinstance(content, list) or len(content) != expected_parts:
        raise AssertionError("request content does not match input mode")
    audio_parts = [item for item in content if isinstance(item, dict) and item.get("type") == "input_audio"]
    text_parts = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(audio_parts) != 1 or len(text_parts) != (0 if prompt is None else 1):
        raise AssertionError("request must contain exactly one audio part")
    if prompt is not None and text_parts[0].get("text") != prompt:
        raise AssertionError("request text differs from fixed prompt")
    if set(audio_parts[0]) != {"type", "input_audio"}:
        raise AssertionError("unexpected audio fields")
    if set(audio_parts[0]["input_audio"]) != {"data", "format"}:
        raise AssertionError("unexpected input_audio fields")
    data = audio_parts[0]["input_audio"].get("data", "")
    if not isinstance(data, str) or not data.startswith("data:audio/wav;base64,"):
        raise AssertionError("audio must be an inline WAV data URL")


def content_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return ""


def request_caption(audio_path, key, endpoint, model, prompt, timeout, max_tokens):
    audio_bytes = audio_path.read_bytes()
    body = build_request_body(audio_bytes, prompt, model, max_tokens=max_tokens)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "benchmark-qwen-omni-captioner/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.getcode()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise RequestFailure(
            "HTTP %s: %s" % (error.code, redact(detail, key)),
            retryable=error.code in RETRYABLE_STATUS or error.code >= 500,
            status=error.code,
        )
    except (urllib.error.URLError, OSError) as error:
        raise RequestFailure(
            "request failed: %s" % redact(error, key), retryable=True
        )
    try:
        response_json = json.loads(raw)
    except ValueError:
        raise RequestFailure(
            "HTTP %s returned non-JSON response: %s"
            % (status, redact(raw[:300], key)),
            retryable=True,
            status=status,
        )
    if status != 200:
        error = response_json.get("error", {}) if isinstance(response_json, dict) else {}
        message = error.get("message") if isinstance(error, dict) else ""
        if not message and isinstance(response_json, dict):
            message = response_json.get("message", "")
        raise RequestFailure(
            "HTTP %s: %s" % (status, redact(message, key)),
            retryable=status in RETRYABLE_STATUS or status >= 500,
            status=status,
        )
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RequestFailure("response has no choices", retryable=True, status=status)
    message = choices[0].get("message") or {}
    caption = content_text(message.get("content")) if isinstance(message, dict) else ""
    if not caption:
        raise RequestFailure("response content is empty", retryable=True, status=status)
    finish_reason = choices[0].get("finish_reason")
    if finish_reason not in (None, "stop"):
        raise RequestFailure(
            "finish_reason=%s" % finish_reason, retryable=True, status=status
        )
    return caption, response_json


def request_caption_with_gate(
    sample, key, args, request_semaphore=None, native_uploader=None
):
    """Run one network attempt, optionally limiting simultaneous requests."""

    request_fn = (
        request_caption_native_oss
        if getattr(args, "transport", "compatible") == "native-oss"
        else request_caption
    )

    def invoke():
        if request_fn is request_caption_native_oss:
            return request_fn(
                sample["audio_path"],
                key,
                endpoint=args.native_endpoint,
                model=args.model,
                prompt=args.prompt if not args.audio_only else None,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                native_uploader=native_uploader,
            )
        return request_fn(
            sample["audio_path"],
            key,
            endpoint=args.endpoint,
            model=args.model,
            prompt=args.prompt if not args.audio_only else None,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )

    if request_semaphore is None:
        return invoke()
    request_semaphore.acquire()
    try:
        return invoke()
    finally:
        request_semaphore.release()


def item_paths(output_dir, sample_id):
    return (
        output_dir / "items" / (sample_id + ".json"),
        output_dir / "captions" / (sample_id + ".txt"),
    )


def read_resumable(
    item_path,
    caption_path,
    sample,
    prompt_sha256,
    model,
    endpoint,
    input_contract,
):
    if not item_path.is_file() or not caption_path.is_file():
        return None
    try:
        record = json.loads(item_path.read_text(encoding="utf-8"))
        caption = caption_path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("status") != "success":
        return None
    if record.get("sample_id") != sample["sample_id"]:
        return None
    if record.get("audio_sha256") != sample["audio_sha256"]:
        return None
    if record.get("prompt_sha256") != prompt_sha256:
        return None
    if record.get("model") != model or record.get("endpoint") != endpoint:
        return None
    if record.get("input_contract") != input_contract:
        return None
    if not caption or record.get("caption", "").strip() != caption:
        return None
    return record


def read_terminal_failure(
    item_path,
    sample,
    prompt_sha256,
    model,
    endpoint,
    input_contract,
):
    """Return a matching deterministic failure, or ``None`` when retryable.

    Failed checkpoints are normally revisited on a later invocation.  A
    provider/input/model error is already terminal, however, and resubmitting
    it only creates duplicate traffic.  Require the same request fingerprint
    used for successful resume checks so an old checkpoint cannot suppress a
    request after the input contract changes.
    """

    if not item_path.is_file():
        return None
    try:
        record = json.loads(item_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("status") != "failed":
        return None
    if record.get("sample_id") != sample["sample_id"]:
        return None
    if record.get("audio_sha256") != sample["audio_sha256"]:
        return None
    if record.get("prompt_sha256") != prompt_sha256:
        return None
    if record.get("model") != model or record.get("endpoint") != endpoint:
        return None
    if record.get("input_contract") != input_contract:
        return None

    errors = record.get("errors")
    if isinstance(errors, str):
        errors = [errors]
    elif not isinstance(errors, (list, tuple)):
        errors = []
    # Mixed or missing diagnostics are conservatively terminal: there is no
    # evidence that the failure is a transient network condition.
    if errors and all(is_network_failure(str(error)) for error in errors):
        return None
    return record


def process_one(
    sample,
    key,
    args,
    output_dir,
    prompt_sha256,
    request_semaphore=None,
    native_uploader=None,
):
    sample_id = sample["sample_id"]
    item_path, caption_path = item_paths(output_dir, sample_id)
    cached = read_resumable(
        item_path,
        caption_path,
        sample,
        prompt_sha256,
        args.model,
        args.endpoint,
        args.input_contract,
    )
    if cached is not None and not args.overwrite:
        cached = dict(cached)
        cached["resumed"] = True
        return cached
    if not args.overwrite:
        terminal = read_terminal_failure(
            item_path,
            sample,
            prompt_sha256,
            args.model,
            args.endpoint,
            args.input_contract,
        )
        if terminal is not None:
            terminal = dict(terminal)
            terminal["resumed"] = True
            return terminal

    started = time.time()
    errors = []
    attempts = 0
    caption = ""
    response_json = None
    while attempts <= args.retries:
        attempts += 1
        try:
            caption, response_json = request_caption_with_gate(
                sample,
                key,
                args,
                request_semaphore=request_semaphore,
                native_uploader=native_uploader,
            )
            break
        except RequestFailure as error:
            errors.append(redact(error, key))
            if (
                not error.retryable
                or attempts > args.retries
                or not is_network_failure(error)
            ):
                break
            # A small bounded jitter avoids synchronized retries across workers.
            delay = min(30.0, 2.0 ** (attempts - 1)) + random.random() * 0.25
            time.sleep(delay)
        except Exception as error:
            # File/read errors are recorded per sample and do not abort the pool.
            errors.append(redact(error, key))
            break
    elapsed = round(time.time() - started, 3)
    common = {
        "sample_id": sample_id,
        "audio_path": str(sample["audio_path"]),
        "audio_sha256": sample["audio_sha256"],
        "status": "success" if caption else "failed",
        "model": args.model,
        "endpoint": args.endpoint,
        "prompt_sha256": prompt_sha256,
        "attempts": attempts,
        "elapsed_sec": elapsed,
        "input_contract": args.input_contract,
    }
    if caption:
        common["caption"] = caption
        # Keep the provider response for diagnostics, but never include the key
        # or request body in persisted output.
        common["response"] = response_json
        atomic_write_text(caption_path, caption + "\n")
    else:
        common["errors"] = errors[-4:]
    atomic_write_json(item_path, common)
    return common


def collect_records(samples, output_dir, prompt_sha256, args):
    records = []
    for sample in samples:
        item_path, caption_path = item_paths(output_dir, sample["sample_id"])
        record = read_resumable(
            item_path,
            caption_path,
            sample,
            prompt_sha256,
            args.model,
            args.endpoint,
            args.input_contract,
        )
        if record is None:
            try:
                candidate = json.loads(item_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                candidate = None
            # Preserve a failed attempt for diagnostics, but never count an
            # old/stale success as a valid result after an input change.
            if isinstance(candidate, dict) and candidate.get("status") == "failed":
                record = candidate
            else:
                record = {
                    "sample_id": sample["sample_id"],
                    "audio_path": str(sample["audio_path"]),
                    "audio_sha256": sample["audio_sha256"],
                    "status": "missing",
                    "model": args.model,
                    "endpoint": args.endpoint,
                    "prompt_sha256": prompt_sha256,
                    "input_contract": args.input_contract,
                    "errors": ["no complete item record"],
                }
        records.append(record)
    return records


def write_aggregate(samples, records, output_dir, args, manifest, prompt_sha256):
    by_id = {record.get("sample_id"): record for record in records if isinstance(record, dict)}
    ordered = [by_id[sample["sample_id"]] for sample in samples]
    atomic_write_text(
        output_dir / "results.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in ordered
        ),
    )
    success = sum(record.get("status") == "success" for record in ordered)
    failed = len(ordered) - success
    summary = {
        "dataset": args.dataset,
        "manifest": str(manifest),
        "benchmark_root": str(args.benchmark_root),
        "sample_count": len(ordered),
        "success_count": success,
        "failed_count": failed,
        "model": args.model,
        "endpoint": args.endpoint,
        "endpoint_ips": list(args.endpoint_ips),
        "transport": args.transport,
        "native_endpoint": args.native_endpoint
        if args.transport == "native-oss"
        else None,
        "native_upload_endpoint": args.native_upload_endpoint
        if args.transport == "native-oss"
        else None,
        "workers": args.workers,
        "max_inflight": args.max_inflight,
        "effective_max_inflight": args.effective_max_inflight,
        "retries": args.retries,
        "timeout_sec": args.timeout,
        "max_tokens": args.max_tokens,
        "prompt_sha256": prompt_sha256,
        "input_contract": args.input_contract,
        "failed_sample_ids": [
            record.get("sample_id")
            for record in ordered
            if record.get("status") != "success"
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument(
        "--audio-root",
        default=None,
        help="optional trusted flat staging root for absolute runtime-manifest paths",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--api-file", default=str(DEFAULT_API_FILE))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--transport",
        choices=("compatible", "native-oss"),
        default="compatible",
        help="API transport; native-oss uploads each audio clip to DashScope OSS",
    )
    parser.add_argument(
        "--native-oss",
        dest="transport",
        action="store_const",
        const="native-oss",
        help="shortcut for --transport native-oss",
    )
    parser.add_argument(
        "--native-endpoint",
        default=DEFAULT_NATIVE_ENDPOINT,
        help="native multimodal-generation endpoint used by --native-oss",
    )
    parser.add_argument(
        "--native-upload-endpoint",
        default=DEFAULT_NATIVE_UPLOAD_ENDPOINT,
        help="DashScope upload-policy endpoint used by --native-oss",
    )
    parser.add_argument(
        "--endpoint-ips",
        "--endpoint-ip",
        dest="endpoint_ips",
        default=None,
        help=(
            "optional comma-separated IPv4 literals for the endpoint; "
            "connections rotate in the listed order while TLS uses the URL hostname"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audio-only",
        dest="audio_only",
        action="store_true",
        default=True,
        help="send exactly one input_audio content part (default)",
    )
    mode.add_argument(
        "--with-prompt",
        dest="audio_only",
        action="store_false",
        help="also send the fixed --prompt text (compatibility mode)",
    )
    parser.add_argument("--dataset", default="structural-benchmark-v1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help=(
            "optional maximum simultaneous API requests; defaults to workers "
            "and does not change the worker pool size"
        ),
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the manifest and request contract without making API calls",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 0 or args.timeout < 1 or args.max_tokens < 1:
        parser.error("workers/retries/timeout/max-tokens values are invalid")
    if args.max_inflight is not None and (
        args.max_inflight < 1 or args.max_inflight > args.workers
    ):
        parser.error("max-inflight must be between 1 and workers")
    args.effective_max_inflight = args.max_inflight or args.workers
    if not args.endpoint.strip() or not args.model.strip():
        parser.error("endpoint/model must be non-empty")
    if not args.native_endpoint.strip() or not args.native_upload_endpoint.strip():
        parser.error("native endpoint URLs must be non-empty")
    if args.transport == "native-oss" and not args.audio_only:
        parser.error("--native-oss requires strict --audio-only input")
    if args.endpoint_ips is not None:
        try:
            args.endpoint_ips = parse_endpoint_ips(args.endpoint_ips)
        except ValueError as error:
            parser.error(str(error))
        resolver_url = (
            args.native_endpoint
            if args.transport == "native-oss"
            else args.endpoint
        )
        if not urllib.parse.urlsplit(resolver_url).hostname:
            parser.error(
                "endpoint must contain a hostname when endpoint IPs are set"
            )
    else:
        args.endpoint_ips = []
    if not args.audio_only and not args.prompt.strip():
        parser.error("prompt must be non-empty when --with-prompt is selected")
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.manifest = Path(args.manifest).resolve()
    args.benchmark_root = Path(args.benchmark_root).resolve()
    args.audio_root = Path(args.audio_root).resolve() if args.audio_root else None
    args.output_dir = Path(args.output_dir).resolve()
    args.api_file = Path(args.api_file).resolve()
    args.input_contract = (
        "audio_only" if args.audio_only else "audio_plus_fixed_instruction"
    )
    # Native mode always uses the audio-only contract and a separate endpoint;
    # ``args.endpoint`` remains the logical endpoint recorded in checkpoints so
    # compatible successes can still be resumed without rewriting their schema.
    if args.transport == "native-oss" and not args.audio_only:
        raise ValueError("native-oss requires audio-only input")
    effective_prompt = None if args.audio_only else args.prompt
    samples = load_manifest(args.manifest, args.benchmark_root, args.audio_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    # Hashes are computed before any network work and are used to reject stale
    # cache entries after an input audio file or prompt changes.
    for sample in samples:
        sample["audio_sha256"] = sha256_file(sample["audio_path"])
    prompt_sha256 = hashlib.sha256(
        (effective_prompt or "").encode("utf-8")
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "items").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "captions").mkdir(parents=True, exist_ok=True)

    config = {
        "manifest": str(args.manifest),
        "benchmark_root": str(args.benchmark_root),
        "audio_root": str(args.audio_root) if args.audio_root else None,
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "model": args.model,
        "endpoint": args.endpoint,
        "transport": args.transport,
        "native_endpoint": args.native_endpoint
        if args.transport == "native-oss"
        else None,
        "native_upload_endpoint": args.native_upload_endpoint
        if args.transport == "native-oss"
        else None,
        "endpoint_ips": list(args.endpoint_ips),
        "workers": args.workers,
        "max_inflight": args.max_inflight,
        "effective_max_inflight": args.effective_max_inflight,
        "retries": args.retries,
        "timeout_sec": args.timeout,
        "max_tokens": args.max_tokens,
        "prompt_sha256": prompt_sha256,
        "sample_count": len(samples),
        "input_contract": args.input_contract,
    }
    atomic_write_json(args.output_dir / "run_config.json", config)

    if args.dry_run:
        # Use a tiny sentinel byte string only to inspect the shape; no request
        # is sent and no real audio is encoded into output.
        if args.transport == "native-oss":
            body = build_native_request_body(
                "oss://dry-run/audio.wav", args.model, max_tokens=args.max_tokens
            )
            request_content_types = [
                "audio"
                for _ in body["input"]["messages"][0]["content"]
            ]
        else:
            body = build_request_body(
                b"dry-run", effective_prompt, args.model, max_tokens=args.max_tokens
            )
            request_content_types = [
                part["type"] for part in body["messages"][0]["content"]
            ]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "sample_count": len(samples),
                    "workers": args.workers,
                    "endpoint_ips": list(args.endpoint_ips),
                    "max_inflight": args.max_inflight,
                    "effective_max_inflight": args.effective_max_inflight,
                    "input_contract": args.input_contract,
                    "transport": args.transport,
                    "request_content_types": request_content_types,
                    "prompt_sha256": prompt_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    key = read_api_key(args.api_file)
    native_uploader = (
        NativeOssUploader(
            model=args.model,
            upload_endpoint=args.native_upload_endpoint,
            api_key=key,
            timeout=args.timeout,
        )
        if args.transport == "native-oss"
        else None
    )
    # Futures are intentionally submitted only after all input validation and
    # hashing have completed, so malformed manifests fail before API traffic.
    # The optional resolver changes only transport address selection; it never
    # changes the request body or the audio-only input contract.
    resolver_endpoint = (
        args.native_endpoint if args.transport == "native-oss" else args.endpoint
    )
    resolver = (
        EndpointIPv4Resolver(resolver_endpoint, args.endpoint_ips)
        if args.endpoint_ips
        else None
    )
    if resolver is not None:
        resolver.install()
    request_semaphore = (
        threading.BoundedSemaphore(args.max_inflight)
        if args.max_inflight is not None
        else None
    )
    try:
        completed = 0
        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    process_one,
                    sample,
                    key,
                    args,
                    args.output_dir,
                    prompt_sha256,
                    request_semaphore,
                    native_uploader,
                )
                for sample in samples
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    record = future.result()
                except Exception as error:
                    # A worker-level failure should not hide the rest of the batch.
                    record = {"status": "failed", "errors": [redact(error, key)]}
                with lock:
                    completed += 1
                    print(
                        "%d/%d %s %s"
                        % (
                            completed,
                            len(samples),
                            record.get("sample_id", "<unknown>"),
                            record.get("status", "failed"),
                        ),
                        flush=True,
                    )
    finally:
        if resolver is not None:
            resolver.restore()

    records = collect_records(samples, args.output_dir, prompt_sha256, args)
    summary = write_aggregate(
        samples,
        records,
        args.output_dir,
        args,
        args.manifest,
        prompt_sha256,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
