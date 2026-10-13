#!/usr/bin/env python3
"""Run a checkpointed Qwen3-Omni-Flash -> DeepSeek full-schema batch.

The benchmark manifest is used only to resolve an opaque sample ID and its
audio path. Qwen receives one audio clip plus fixed task instructions;
DeepSeek receives only the parsed Qwen candidate. The two stages have separate
thread pools so DeepSeek can repair completed Qwen candidates while Qwen keeps
processing later clips.
"""

from __future__ import print_function

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import os
import random
import re
import sys
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import extract_captioner_deepseek_once as deepseek_api  # noqa: E402
import run_qwen3_omni_flash_full_demo as full_demo  # noqa: E402
import run_qwen_benchmark as benchmark_api  # noqa: E402


COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_MANIFEST = DEFAULT_BENCHMARK_ROOT / "benchmark.manifest.jsonl"
DEFAULT_OUTPUT_DIR = (
    COMPARE_ROOT / "evaluation" / "qwen3_omni_flash_full_schema_benchmark"
)
DEFAULT_QWEN_PROMPT = SCRIPT_DIR / "qwen3_omni_flash_full_schema_prompt.txt"
DEFAULT_DEEPSEEK_PROMPT = SCRIPT_DIR / "deepseek_full_schema_repair_prompt.txt"
DEFAULT_QWEN_API = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"
DEFAULT_DEEPSEEK_API = COMPARE_ROOT / "api" / "deepseek-api.txt"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-omni-flash"
SCHEMA_VERSION = "structural-prediction-v0.1"
FIELD_UNIVERSE_VERSION = "structural-v0.1"
TOKEN_RE = re.compile(r"(?i)(sk-[A-Za-z0-9._-]{12,})")

QWEN_USER_PROMPT = (
    "Listen to the attached audio according to the system message and return only the required "
    "full-schema JSON.\nThis is an audio-only candidate annotation: every value must come from "
    "the audio or metadata explicitly exposed by the audio part; use null when uncertain."
)
DEEPSEEK_RETRY_INSTRUCTION = (
    "\n\nThis is a retry after local validation failed. The user message contains "
    "omni_candidate, previous_repair, and validation_error. Treat omni_candidate as the only "
    "factual evidence, repair the validation error and any resulting conflicts, and return only "
    "one complete schema JSON object."
)


class ProviderError(RuntimeError):
    def __init__(self, message, retryable=False, status=None):
        RuntimeError.__init__(self, message)
        self.retryable = bool(retryable)
        self.status = status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact(value, *secrets):
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return TOKEN_RE.sub("<redacted>", text)[:2000]


def atomic_write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
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


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def item_paths(output_dir, stage, sample_id):
    base = Path(output_dir) / stage
    return {
        "item": base / "items" / (sample_id + ".json"),
        "status": base / "status" / (sample_id + ".json"),
        "raw_dir": base / "raw",
    }


def envelope(sample_id, payload):
    return dict(
        {
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
            "field_universe_version": FIELD_UNIVERSE_VERSION,
        },
        **payload
    )


def read_qwen_resume(output_dir, sample, prompt_hash, fingerprint):
    paths = item_paths(output_dir, "omni", sample["sample_id"])
    record = load_json(paths["status"])
    payload = load_json(paths["item"])
    if not isinstance(record, dict) or record.get("status") != "success":
        return None
    if record.get("sample_id") != sample["sample_id"]:
        return None
    if record.get("audio_sha256") != sample["audio_sha256"]:
        return None
    if record.get("prompt_sha256") != prompt_hash:
        return None
    if record.get("config_fingerprint") != fingerprint:
        return None
    if not isinstance(payload, dict) or sha256_text(canonical_json(payload)) != record.get(
        "candidate_sha256"
    ):
        return None
    resumed = dict(record)
    resumed["candidate"] = payload
    resumed["resumed"] = True
    return resumed


def read_deepseek_resume(output_dir, sample_id, candidate, prompt_hash, fingerprint):
    paths = item_paths(output_dir, "deepseek", sample_id)
    record = load_json(paths["status"])
    payload = load_json(paths["item"])
    if not isinstance(record, dict) or record.get("status") != "success":
        return None
    if record.get("sample_id") != sample_id:
        return None
    if record.get("input_sha256") != sha256_text(canonical_json(candidate)):
        return None
    if record.get("prompt_sha256") != prompt_hash:
        return None
    if record.get("config_fingerprint") != fingerprint:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        full_demo.validate_full_schema(payload)
    except (TypeError, ValueError, full_demo.ValidationError):
        return None
    resumed = dict(record)
    resumed["payload"] = payload
    resumed["resumed"] = True
    return resumed


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strip_extra_schema_keys(payload):
    """Delete only keys forbidden by the exact schema and report their paths."""

    removed = []

    def retain(value, allowed, path):
        if not isinstance(value, dict):
            return
        for key in sorted(set(value) - set(allowed)):
            removed.append(path + "." + key if path else key)
            del value[key]

    retain(payload, full_demo.TOP_LEVEL_KEYS, "")
    if not isinstance(payload, dict):
        return removed
    for group, keys in full_demo.GROUP_KEYS.items():
        retain(payload.get(group), keys, group)

    basic = payload.get("basic_acoustic")
    if isinstance(basic, dict) and isinstance(basic.get("silence_segments"), list):
        for index, item in enumerate(basic["silence_segments"]):
            retain(
                item,
                {"start_sec", "end_sec"},
                "basic_acoustic.silence_segments[%d]" % index,
            )

    scene = payload.get("sound_field_scene")
    if isinstance(scene, dict):
        retain(
            scene.get("noise_composition"),
            set(full_demo.NOISE_CLASSES),
            "sound_field_scene.noise_composition",
        )

    speaker = payload.get("speaker")
    if isinstance(speaker, dict) and isinstance(speaker.get("profiles"), list):
        for index, item in enumerate(speaker["profiles"]):
            profile_path = "speaker.profiles[%d]" % index
            retain(
                item,
                {"speaker_id", "speech_rate", "pitch", "speaker_volume"},
                profile_path,
            )
            if isinstance(item, dict):
                retain(
                    item.get("speech_rate"),
                    {"band", "value", "unit"},
                    profile_path + ".speech_rate",
                )

    language = payload.get("language_content")
    if isinstance(language, dict):
        retain(
            language.get("punctuation"),
            {"punctuation_count", "has_terminal_punctuation"},
            "language_content.punctuation",
        )
        retain(
            language.get("repetition"),
            {"has_repetition", "repetition_count"},
            "language_content.repetition",
        )

    descriptive = payload.get("descriptive")
    if isinstance(descriptive, dict):
        nested = (
            ("facets", full_demo.FACET_KEYS),
            ("speaker_descriptions", full_demo.SPEAKER_DESCRIPTION_KEYS),
            ("scene_hypotheses", full_demo.HYPOTHESIS_KEYS),
        )
        for field, keys in nested:
            values = descriptive.get(field)
            if isinstance(values, list):
                for index, item in enumerate(values):
                    retain(item, keys, "descriptive.%s[%d]" % (field, index))
    return removed


def retry_delay(attempt, base=2.0, maximum=30.0):
    return min(maximum, base * (2.0 ** max(0, attempt - 1))) + random.random() * 0.25


def qwen_audio_bytes(audio_path, audio_format):
    if audio_format == "wav":
        return Path(audio_path).read_bytes()
    if audio_format != "mp3":
        raise ValueError("unsupported Qwen audio format: %s" % audio_format)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        "-f",
        "mp3",
        "pipe:1",
    ]
    try:
        return subprocess.check_output(command, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise RuntimeError("ffmpeg MP3 fallback failed: %s %s" % (error, detail[:1000]))


def qwen_messages(system_prompt, user_prompt, audio_bytes, audio_format):
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/%s;base64,%s" % (audio_format, encoded),
                        "format": audio_format,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def qwen_request(audio_path, key, args, system_prompt, audio_format="wav"):
    audio_bytes = qwen_audio_bytes(audio_path, audio_format)
    return full_demo.qwen_stream(
        args.qwen_base_url,
        key,
        args.qwen_model,
        qwen_messages(system_prompt, QWEN_USER_PROMPT, audio_bytes, audio_format),
        args.qwen_max_tokens,
        args.qwen_timeout,
    )


def process_qwen(
    sample, key, args, system_prompt, prompt_hash, fingerprint, api_label=None
):
    sample_id = sample["sample_id"]
    paths = item_paths(args.output_dir, "omni", sample_id)
    if not args.overwrite:
        resumed = read_qwen_resume(
            args.output_dir, sample, prompt_hash, fingerprint
        )
        if resumed is not None:
            return resumed

    started = time.time()
    errors = []
    network_attempts = 0
    output_attempts = 0
    total_attempts = 0
    previous_record = load_json(paths["status"])
    previous_errors = previous_record.get("errors", []) if isinstance(previous_record, dict) else []
    use_mp3_fallback = (
        Path(sample["audio_path"]).stat().st_size > 2 * 1024 * 1024
        and any("write operation timed out" in str(error) for error in previous_errors)
    )
    request_audio_format = "mp3" if use_mp3_fallback else "wav"
    request_audio_formats = []
    while network_attempts <= args.qwen_retries:
        total_attempts += 1
        request_audio_formats.append(request_audio_format)
        raw_text = None
        try:
            raw_text, usage = qwen_request(
                sample["audio_path"], key, args, system_prompt, request_audio_format
            )
        except Exception as error:
            retryable = benchmark_api.is_network_failure(error)
            errors.append(redact(error, key))
            if not retryable or network_attempts >= args.qwen_retries:
                break
            network_attempts += 1
            if (
                "write operation timed out" in str(error)
                and Path(sample["audio_path"]).stat().st_size > 2 * 1024 * 1024
            ):
                request_audio_format = "mp3"
            time.sleep(retry_delay(network_attempts))
            continue

        raw_path = paths["raw_dir"] / (
            "%s.attempt_%02d.txt" % (sample_id, total_attempts)
        )
        atomic_write_text(raw_path, raw_text.strip() + "\n")
        try:
            candidate = full_demo.qwen_api.parse_json_object(raw_text)
        except (TypeError, ValueError) as error:
            errors.append("model_output_error: %s" % redact(error, key))
            if output_attempts >= args.qwen_output_retries:
                break
            output_attempts += 1
            continue

        validation_error = None
        try:
            full_demo.validate_full_schema(candidate)
        except (TypeError, ValueError, full_demo.ValidationError) as error:
            # DeepSeek is responsible for repairing a parsed but invalid candidate.
            validation_error = str(error)
        candidate_hash = sha256_text(canonical_json(candidate))
        atomic_write_json(paths["item"], candidate)
        record = {
            "sample_id": sample_id,
            "status": "success",
            "audio_path": str(sample["audio_path"]),
            "audio_sha256": sample["audio_sha256"],
            "candidate_sha256": candidate_hash,
            "candidate_validation_error": validation_error,
            "model": args.qwen_model,
            "base_url": args.qwen_base_url,
            "prompt_sha256": prompt_hash,
            "config_fingerprint": fingerprint,
            "attempts": total_attempts,
            "request_audio_formats": request_audio_formats,
            "usage": usage if isinstance(usage, dict) else {},
            "raw_output": str(raw_path.resolve()),
            "elapsed_sec": round(time.time() - started, 3),
            "completed_at": utc_now(),
        }
        if api_label is not None:
            record["qwen_api_file"] = api_label
        atomic_write_json(paths["status"], record)
        result = dict(record)
        result["candidate"] = candidate
        return result

    record = {
        "sample_id": sample_id,
        "status": "failed",
        "audio_path": str(sample["audio_path"]),
        "audio_sha256": sample["audio_sha256"],
        "model": args.qwen_model,
        "base_url": args.qwen_base_url,
        "prompt_sha256": prompt_hash,
        "config_fingerprint": fingerprint,
        "attempts": total_attempts,
        "request_audio_formats": request_audio_formats,
        "errors": errors[-6:],
        "elapsed_sec": round(time.time() - started, 3),
        "completed_at": utc_now(),
    }
    if api_label is not None:
        record["qwen_api_file"] = api_label
    atomic_write_json(paths["status"], record)
    return record


def deepseek_raw_request(config, system_prompt, user_text, timeout, max_tokens, thinking):
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": int(max_tokens),
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": thinking},
    }
    endpoint = deepseek_api.endpoint(config["base_url"])
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + config["api_key"],
            "Content-Type": "application/json",
            "User-Agent": "qwen-deepseek-full-schema-batch/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            response_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise ProviderError(
            "HTTP %s: %s" % (error.code, redact(detail, config["api_key"])),
            retryable=error.code in benchmark_api.RETRYABLE_STATUS or error.code >= 500,
            status=error.code,
        )
    except (urllib.error.URLError, OSError) as error:
        raise ProviderError(
            "request failed: %s" % redact(error, config["api_key"]),
            retryable=True,
        )
    try:
        response_json = deepseek_api.strict_json(response_text)
    except ValueError as error:
        raise ProviderError("non-JSON API response: %s" % error, retryable=True, status=status)
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderError("response has no choices", retryable=True, status=status)
    choice = choices[0]
    if choice.get("finish_reason") not in (None, "stop"):
        raise ProviderError(
            "finish_reason=%s" % choice.get("finish_reason"),
            retryable=True,
            status=status,
        )
    output_text = deepseek_api.content_text(choice.get("message") or {})
    if not output_text:
        raise ProviderError("response content is empty", retryable=True, status=status)
    return output_text, response_json.get("usage") or {}


def process_deepseek(
    sample_id,
    candidate,
    config,
    args,
    system_prompt,
    prompt_hash,
    fingerprint,
):
    paths = item_paths(args.output_dir, "deepseek", sample_id)
    if not args.overwrite:
        resumed = read_deepseek_resume(
            args.output_dir, sample_id, candidate, prompt_hash, fingerprint
        )
        if resumed is not None:
            return resumed

    started = time.time()
    input_hash = sha256_text(canonical_json(candidate))
    user_text = canonical_json(candidate)
    errors = []
    attempt_records = []
    request_count = 0
    for repair_attempt in range(1, args.deepseek_repair_attempts + 1):
        attempt_prompt = system_prompt
        if repair_attempt > 1:
            attempt_prompt += DEEPSEEK_RETRY_INSTRUCTION
        network_attempt = 0
        output_text = None
        usage = {}
        request_thinking = args.deepseek_thinking
        request_thinking_modes = []
        while network_attempt <= args.deepseek_retries:
            request_count += 1
            request_thinking_modes.append(request_thinking)
            try:
                output_text, usage = deepseek_raw_request(
                    config,
                    attempt_prompt,
                    user_text,
                    args.deepseek_timeout,
                    args.deepseek_max_tokens,
                    request_thinking,
                )
                break
            except ProviderError as error:
                message = redact(error, config.get("api_key"))
                errors.append(message)
                if not error.retryable or network_attempt >= args.deepseek_retries:
                    break
                network_attempt += 1
                if "finish_reason=length" in message and request_thinking == "enabled":
                    request_thinking = "disabled"
                time.sleep(retry_delay(network_attempt))
        if output_text is None:
            if errors and not benchmark_api.is_network_failure(errors[-1]):
                break
            continue

        raw_path = paths["raw_dir"] / (
            "%s.attempt_%02d.txt" % (sample_id, repair_attempt)
        )
        atomic_write_text(raw_path, output_text.strip() + "\n")
        attempt_record = {
            "attempt": repair_attempt,
            "raw_output": str(raw_path.resolve()),
            "usage": usage if isinstance(usage, dict) else {},
            "request_thinking_modes": request_thinking_modes,
        }
        repaired = None
        try:
            repaired = deepseek_api.strict_json(output_text)
            if not isinstance(repaired, dict):
                raise ValueError("DeepSeek output root is not an object")
            removed_extra_paths = strip_extra_schema_keys(repaired)
            if removed_extra_paths:
                attempt_record["removed_extra_paths"] = removed_extra_paths
            parsed_path = (
                Path(args.output_dir)
                / "deepseek"
                / "attempts"
                / ("%s.attempt_%02d.json" % (sample_id, repair_attempt))
            )
            atomic_write_json(parsed_path, repaired)
            attempt_record["parsed_output"] = str(parsed_path.resolve())
            full_demo.validate_full_schema(repaired)
        except (TypeError, ValueError, full_demo.ValidationError) as error:
            validation_error = redact(error, config.get("api_key"))
            errors.append("validation_error: " + validation_error)
            attempt_record["validation_error"] = validation_error
            attempt_records.append(attempt_record)
            user_text = canonical_json(
                {
                    "omni_candidate": candidate,
                    "previous_repair": repaired,
                    "validation_error": validation_error,
                }
            )
            continue

        attempt_records.append(attempt_record)
        atomic_write_json(paths["item"], repaired)
        record = {
            "sample_id": sample_id,
            "status": "success",
            "input_sha256": input_hash,
            "output_sha256": sha256_text(canonical_json(repaired)),
            "model": config["model"],
            "base_url": config["base_url"],
            "prompt_sha256": prompt_hash,
            "config_fingerprint": fingerprint,
            "request_count": request_count,
            "repair_attempts": attempt_records,
            "elapsed_sec": round(time.time() - started, 3),
            "completed_at": utc_now(),
        }
        atomic_write_json(paths["status"], record)
        result = dict(record)
        result["payload"] = repaired
        return result

    record = {
        "sample_id": sample_id,
        "status": "failed",
        "input_sha256": input_hash,
        "model": config["model"],
        "base_url": config["base_url"],
        "prompt_sha256": prompt_hash,
        "config_fingerprint": fingerprint,
        "request_count": request_count,
        "repair_attempts": attempt_records,
        "errors": errors[-8:],
        "elapsed_sec": round(time.time() - started, 3),
        "completed_at": utc_now(),
    }
    atomic_write_json(paths["status"], record)
    return record


def prepare_samples(manifest, benchmark_root, limit=None):
    rows = benchmark_api.load_manifest(manifest, benchmark_root)
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        row["audio_sha256"] = sha256_file(row["audio_path"])
    return rows


def write_progress(output_dir, state, last_stage=None, last_sample_id=None):
    value = dict(state)
    value["updated_at"] = utc_now()
    if last_stage is not None:
        value["last_stage"] = last_stage
        value["last_sample_id"] = last_sample_id
    atomic_write_json(Path(output_dir) / "progress.json", value)


def write_aggregates(output_dir, samples, qwen_records, deepseek_records, state, config):
    output_dir = Path(output_dir)
    qwen_ordered = [qwen_records.get(item["sample_id"]) for item in samples]
    deepseek_ordered = [deepseek_records.get(item["sample_id"]) for item in samples]
    atomic_write_text(
        output_dir / "omni_results.jsonl",
        "".join(
            json.dumps(envelope(record["sample_id"], record["candidate"]), ensure_ascii=False, sort_keys=True)
            + "\n"
            for record in qwen_ordered
            if isinstance(record, dict) and record.get("status") == "success"
        ),
    )
    atomic_write_text(
        output_dir / "final_results.jsonl",
        "".join(
            json.dumps(envelope(record["sample_id"], record["payload"]), ensure_ascii=False, sort_keys=True)
            + "\n"
            for record in deepseek_ordered
            if isinstance(record, dict) and record.get("status") == "success"
        ),
    )
    atomic_write_text(
        output_dir / "omni_status.jsonl",
        "".join(
            json.dumps({key: value for key, value in record.items() if key != "candidate"}, ensure_ascii=False, sort_keys=True)
            + "\n"
            for record in qwen_ordered
            if isinstance(record, dict)
        ),
    )
    atomic_write_text(
        output_dir / "final_status.jsonl",
        "".join(
            json.dumps({key: value for key, value in record.items() if key != "payload"}, ensure_ascii=False, sort_keys=True)
            + "\n"
            for record in deepseek_ordered
            if isinstance(record, dict)
        ),
    )
    summary = dict(state)
    summary.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "qwen_model": config["qwen_model"],
            "deepseek_model": config["deepseek_model"],
            "omni_results": str((output_dir / "omni_results.jsonl").resolve()),
            "final_results": str((output_dir / "final_results.jsonl").resolve()),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qwen-prompt", type=Path, default=DEFAULT_QWEN_PROMPT)
    parser.add_argument("--deepseek-prompt", type=Path, default=DEFAULT_DEEPSEEK_PROMPT)
    parser.add_argument(
        "--qwen-api-file",
        type=Path,
        action="append",
        dest="qwen_api_files",
        default=None,
        help="Qwen API key file; repeat to assign samples across multiple keys",
    )
    parser.add_argument("--deepseek-api-file", type=Path, default=DEFAULT_DEEPSEEK_API)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-workers", type=int, default=5)
    parser.add_argument("--deepseek-workers", type=int, default=5)
    parser.add_argument("--qwen-retries", type=int, default=4)
    parser.add_argument("--qwen-output-retries", type=int, default=1)
    parser.add_argument("--deepseek-retries", type=int, default=4)
    parser.add_argument("--deepseek-repair-attempts", type=int, default=3)
    parser.add_argument("--qwen-timeout", type=int, default=300)
    parser.add_argument("--deepseek-timeout", type=int, default=300)
    parser.add_argument("--qwen-max-tokens", type=int, default=8000)
    parser.add_argument("--deepseek-max-tokens", type=int, default=16000)
    parser.add_argument(
        "--deepseek-thinking", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    if args.qwen_api_files is None:
        args.qwen_api_files = [DEFAULT_QWEN_API]
    integer_values = (
        args.qwen_workers,
        args.deepseek_workers,
        args.qwen_timeout,
        args.deepseek_timeout,
        args.qwen_max_tokens,
        args.deepseek_max_tokens,
        args.deepseek_repair_attempts,
    )
    if any(value < 1 for value in integer_values):
        raise ValueError("workers, timeouts, token limits, and repair attempts must be positive")
    if args.qwen_retries < 0 or args.qwen_output_retries < 0 or args.deepseek_retries < 0:
        raise ValueError("retry counts must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    required_paths = [
        args.manifest,
        args.qwen_prompt,
        args.deepseek_prompt,
        args.deepseek_api_file,
    ] + list(args.qwen_api_files)
    for path in required_paths:
        if not path.resolve().is_file():
            raise ValueError("required file does not exist: %s" % path.resolve())


def acquire_lock(output_dir):
    lock_path = Path(output_dir) / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        value = load_json(lock_path)
        pid = value.get("pid") if isinstance(value, dict) else None
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
            except OSError:
                lock_path.unlink()
                return acquire_lock(output_dir)
        raise RuntimeError("another batch appears to be running: %s" % lock_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}) + "\n")
    return lock_path


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    args.manifest = args.manifest.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qwen_prompt = args.qwen_prompt.resolve().read_text(encoding="utf-8").strip()
    deepseek_prompt = args.deepseek_prompt.resolve().read_text(encoding="utf-8").strip()
    qwen_prompt_hash = sha256_text(qwen_prompt)
    deepseek_prompt_hash = sha256_text(deepseek_prompt)
    qwen_api_files = [path.resolve() for path in args.qwen_api_files]
    qwen_keys = [benchmark_api.read_api_key(path) for path in qwen_api_files]
    deepseek_config = deepseek_api.read_api_file(args.deepseek_api_file.resolve())

    qwen_fingerprint_value = {
        "model": args.qwen_model,
        "base_url": args.qwen_base_url,
        "prompt_sha256": qwen_prompt_hash,
        "user_prompt_sha256": sha256_text(QWEN_USER_PROMPT),
        "max_tokens": args.qwen_max_tokens,
        "timeout": args.qwen_timeout,
    }
    deepseek_fingerprint_value = {
        "model": deepseek_config["model"],
        "base_url": deepseek_config["base_url"],
        "prompt_sha256": deepseek_prompt_hash,
        "max_tokens": args.deepseek_max_tokens,
        "timeout": args.deepseek_timeout,
        "thinking": args.deepseek_thinking,
    }
    qwen_fingerprint = sha256_text(canonical_json(qwen_fingerprint_value))
    deepseek_fingerprint = sha256_text(canonical_json(deepseek_fingerprint_value))
    samples = prepare_samples(args.manifest, args.benchmark_root, args.limit)
    manifest_hash = sha256_file(args.manifest)
    run_config = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "benchmark_root": str(args.benchmark_root),
        "sample_count": len(samples),
        "qwen_api_files": [str(path) for path in qwen_api_files],
        "qwen_api_count": len(qwen_keys),
        "qwen_workers_per_api": args.qwen_workers,
        "qwen_workers": args.qwen_workers * len(qwen_keys),
        "deepseek_workers": args.deepseek_workers,
        "qwen_retries": args.qwen_retries,
        "qwen_output_retries": args.qwen_output_retries,
        "qwen_large_wav_timeout_fallback": "mp3_16khz_mono_64kbps",
        "deepseek_retries": args.deepseek_retries,
        "deepseek_repair_attempts": args.deepseek_repair_attempts,
        "deepseek_length_fallback_thinking": "disabled",
        "deepseek_strip_extra_schema_keys": True,
        "input_contract": "audio_and_fixed_prompt_to_qwen_then_candidate_json_only_to_deepseek",
        "qwen": dict(qwen_fingerprint_value, config_fingerprint=qwen_fingerprint),
        "deepseek": dict(
            deepseek_fingerprint_value, config_fingerprint=deepseek_fingerprint
        ),
        "created_at": utc_now(),
    }
    old_config = load_json(args.output_dir / "run_config.json")
    if (
        not args.overwrite
        and isinstance(old_config, dict)
        and (
            old_config.get("manifest_sha256") != manifest_hash
            or old_config.get("sample_count") != len(samples)
            or old_config.get("qwen", {}).get("config_fingerprint") != qwen_fingerprint
            or old_config.get("deepseek", {}).get("config_fingerprint")
            != deepseek_fingerprint
        )
    ):
        raise RuntimeError(
            "existing run configuration differs; use a new output directory or --overwrite"
        )
    atomic_write_json(args.output_dir / "run_config.json", run_config)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "sample_count": len(samples),
                    "qwen_api_count": len(qwen_keys),
                    "qwen_workers_per_api": args.qwen_workers,
                    "qwen_workers": args.qwen_workers * len(qwen_keys),
                    "deepseek_workers": args.deepseek_workers,
                    "output_dir": str(args.output_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    lock_path = acquire_lock(args.output_dir)
    qwen_records = {}
    deepseek_records = {}
    state = {
        "status": "running",
        "sample_count": len(samples),
        "qwen_completed": 0,
        "qwen_success": 0,
        "qwen_failed": 0,
        "qwen_resumed": 0,
        "deepseek_completed": 0,
        "deepseek_success": 0,
        "deepseek_failed": 0,
        "deepseek_resumed": 0,
        "qwen_api_count": len(qwen_keys),
        "qwen_workers_per_api": args.qwen_workers,
        "qwen_workers": args.qwen_workers * len(qwen_keys),
        "deepseek_workers": args.deepseek_workers,
        "started_at": utc_now(),
        "pid": os.getpid(),
    }
    write_progress(args.output_dir, state)
    print(
        "Full-schema batch: samples=%d qwen_apis=%d qwen_workers=%d (%d/api) "
        "deepseek_workers=%d output=%s"
        % (
            len(samples),
            len(qwen_keys),
            args.qwen_workers * len(qwen_keys),
            args.qwen_workers,
            args.deepseek_workers,
            args.output_dir,
        ),
        flush=True,
    )

    try:
        with contextlib.ExitStack() as stack:
            qwen_pools = [
                stack.enter_context(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.qwen_workers,
                        thread_name_prefix="qwen%d" % (index + 1),
                    )
                )
                for index in range(len(qwen_keys))
            ]
            deepseek_pool = stack.enter_context(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.deepseek_workers, thread_name_prefix="deepseek"
                )
            )
            qwen_futures = {}
            deepseek_futures = {}
            for sample_index, sample in enumerate(samples):
                resumed = None
                if not args.overwrite:
                    resumed = read_qwen_resume(
                        args.output_dir, sample, qwen_prompt_hash, qwen_fingerprint
                    )
                if resumed is not None:
                    qwen_records[sample["sample_id"]] = resumed
                    state["qwen_completed"] += 1
                    state["qwen_success"] += 1
                    state["qwen_resumed"] += 1
                    deep_resumed = read_deepseek_resume(
                        args.output_dir,
                        sample["sample_id"],
                        resumed["candidate"],
                        deepseek_prompt_hash,
                        deepseek_fingerprint,
                    )
                    if deep_resumed is not None:
                        deepseek_records[sample["sample_id"]] = deep_resumed
                        state["deepseek_completed"] += 1
                        state["deepseek_success"] += 1
                        state["deepseek_resumed"] += 1
                    else:
                        future = deepseek_pool.submit(
                            process_deepseek,
                            sample["sample_id"],
                            resumed["candidate"],
                            deepseek_config,
                            args,
                            deepseek_prompt,
                            deepseek_prompt_hash,
                            deepseek_fingerprint,
                        )
                        deepseek_futures[future] = sample["sample_id"]
                    continue
                api_index = sample_index % len(qwen_keys)
                qwen_key = qwen_keys[api_index]
                api_label = str(qwen_api_files[api_index])
                future = qwen_pools[api_index].submit(
                    process_qwen,
                    sample,
                    qwen_key,
                    args,
                    qwen_prompt,
                    qwen_prompt_hash,
                    qwen_fingerprint,
                    api_label,
                )
                qwen_futures[future] = (sample, qwen_key)

            write_progress(args.output_dir, state)
            while qwen_futures or deepseek_futures:
                all_futures = list(qwen_futures) + list(deepseek_futures)
                done, _ = concurrent.futures.wait(
                    all_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    if future in qwen_futures:
                        sample, qwen_key = qwen_futures.pop(future)
                        sample_id = sample["sample_id"]
                        try:
                            record = future.result()
                        except Exception as error:
                            record = {
                                "sample_id": sample_id,
                                "status": "failed",
                                "errors": ["worker_error: " + redact(error, qwen_key)],
                            }
                            atomic_write_json(
                                item_paths(args.output_dir, "omni", sample_id)["status"],
                                record,
                            )
                        qwen_records[sample_id] = record
                        state["qwen_completed"] += 1
                        if record.get("status") == "success":
                            state["qwen_success"] += 1
                            deep_future = deepseek_pool.submit(
                                process_deepseek,
                                sample_id,
                                record["candidate"],
                                deepseek_config,
                                args,
                                deepseek_prompt,
                                deepseek_prompt_hash,
                                deepseek_fingerprint,
                            )
                            deepseek_futures[deep_future] = sample_id
                        else:
                            state["qwen_failed"] += 1
                        print(
                            "Qwen %d/%d %s %s"
                            % (
                                state["qwen_completed"],
                                len(samples),
                                sample_id,
                                record.get("status"),
                            ),
                            flush=True,
                        )
                        write_progress(args.output_dir, state, "qwen", sample_id)
                    else:
                        sample_id = deepseek_futures.pop(future)
                        try:
                            record = future.result()
                        except Exception as error:
                            record = {
                                "sample_id": sample_id,
                                "status": "failed",
                                "errors": [
                                    "worker_error: "
                                    + redact(error, deepseek_config.get("api_key"))
                                ],
                            }
                            atomic_write_json(
                                item_paths(args.output_dir, "deepseek", sample_id)["status"],
                                record,
                            )
                        deepseek_records[sample_id] = record
                        state["deepseek_completed"] += 1
                        if record.get("status") == "success":
                            state["deepseek_success"] += 1
                        else:
                            state["deepseek_failed"] += 1
                        print(
                            "DeepSeek %d/%d %s %s"
                            % (
                                state["deepseek_completed"],
                                len(samples),
                                sample_id,
                                record.get("status"),
                            ),
                            flush=True,
                        )
                        write_progress(args.output_dir, state, "deepseek", sample_id)

        summary = write_aggregates(
            args.output_dir,
            samples,
            qwen_records,
            deepseek_records,
            state,
            {
                "qwen_model": args.qwen_model,
                "deepseek_model": deepseek_config["model"],
            },
        )
        state["status"] = "complete"
        state["completed_at"] = summary["completed_at"]
        write_progress(args.output_dir, state)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if state["qwen_failed"] == 0 and state["deepseek_failed"] == 0 else 2
    finally:
        try:
            lock_value = load_json(lock_path)
            if isinstance(lock_value, dict) and lock_value.get("pid") == os.getpid():
                lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("ERROR: %s" % redact(error), file=sys.stderr)
        raise SystemExit(1)
