#!/usr/bin/env python3
"""Run the audio-only descriptive GT prompt on one local WAV file."""

from __future__ import print_function

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_AUDIO = (
    COMPARE_ROOT / "dataset" / "Ami" / "data" / "audio" / "IS1006c_utterance_00027.wav"
)
DEFAULT_PROMPT = SCRIPT_DIR / "qwen3_5_omni_plus_descriptive_gt_prompt.txt"
DEFAULT_API_FILE = COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt"
DEFAULT_OUTPUT = (
    COMPARE_ROOT
    / "evaluation"
    / "qwen3_5_omni_plus_descriptive_gt_demo"
    / "IS1006c_utterance_00027.json"
)
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
KEY_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._-]{20,}")
DESCRIPTIVE_KEYS = (
    "caption",
    "topic",
    "background_condition",
    "dominant_content_impression",
    "clarity_intelligibility_impression",
    "background_sound_summary",
    "descriptive_phrases",
    "facets",
    "speaker_descriptions",
    "scene_hypotheses",
    "usability_notes",
    "uncertainties",
)
SPEAKER_GENDERS = frozenset(("male", "female"))


def read_api_key(path):
    raw = Path(path).read_text(encoding="utf-8")
    match = KEY_RE.search(raw)
    if not match:
        raise RuntimeError("API file does not contain a Bailian key")
    return match.group(0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact(value, secret):
    text = str(value)
    return text.replace(secret, "<redacted>") if secret else text


def delta_text(delta):
    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)
    return ""


def json_delta_text(delta):
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def usage_dict(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result or {"repr": str(usage)}


def parse_json_object(text):
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except ValueError:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, end = decoder.raw_decode(candidate[index:])
            except ValueError:
                continue
            if candidate[index + end :].strip():
                continue
            break
        if value is None:
            raise ValueError("model response is not a JSON object")
    if not isinstance(value, dict):
        raise ValueError("model response root is not an object")
    return value


def validate_descriptive(value):
    if set(value) != {"descriptive"}:
        raise ValueError("root keys must be exactly ['descriptive']")
    descriptive = value["descriptive"]
    if not isinstance(descriptive, dict) or set(descriptive) != set(DESCRIPTIVE_KEYS):
        raise ValueError("descriptive keys do not match the 12-key contract")
    for field in ("facets", "speaker_descriptions", "scene_hypotheses"):
        items = descriptive[field]
        if items is None:
            continue
        if not isinstance(items, list):
            raise ValueError("descriptive.%s must be an array or null" % field)
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("descriptive.%s contains a non-object" % field)
            if item.get("evidence_tags") != []:
                raise ValueError(
                    "audio-only descriptive.%s evidence_tags must be []" % field
                )
    profiles = descriptive["speaker_descriptions"]
    if profiles:
        ids = [item.get("speaker_id") for item in profiles]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("speaker_descriptions speaker_id must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError("speaker_descriptions speaker_id values must be unique")
        for speaker in profiles:
            if "gender" not in speaker:
                raise ValueError("speaker_descriptions items must contain gender")
            if speaker["gender"] is not None and speaker["gender"] not in SPEAKER_GENDERS:
                raise ValueError(
                    "speaker_descriptions.gender must be male, female, or null"
                )


def make_messages(prompt_text, audio_bytes, user_text=None):
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    content = []
    if user_text is not None:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string when provided")
        content.append({"type": "text", "text": user_text})
    content.append(
        {
            "type": "input_audio",
            "input_audio": {
                "data": "data:audio/wav;base64," + encoded,
                "format": "wav",
            },
        }
    )
    return [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": content,
        },
    ]


def make_request_body(prompt_text, audio_bytes, model, max_tokens, user_text=None):
    return {
        "model": model,
        "messages": make_messages(prompt_text, audio_bytes, user_text=user_text),
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }


def stream_with_urllib(
    base_url, api_key, prompt_text, audio_bytes, model, max_tokens, user_text=None
):
    body = make_request_body(
        prompt_text, audio_bytes, model, max_tokens, user_text=user_text
    )
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "qwen3-5-omni-plus-descriptive-gt-demo/1.0",
        },
        method="POST",
    )
    text_parts = []
    usage = None
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError("HTTP %s: %s" % (error.code, redact(detail, api_key)))
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                text_parts.append(json_delta_text(choices[0].get("delta")))
    return "".join(text_parts), usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max-tokens", type=int, default=4000)
    args = parser.parse_args(argv)

    audio_path = args.audio.resolve()
    prompt_path = args.prompt.resolve()
    if not audio_path.is_file():
        parser.error("audio file does not exist: %s" % audio_path)
    if not prompt_path.is_file():
        parser.error("prompt file does not exist: %s" % prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        parser.error("prompt file is empty")

    api_key = os.getenv("DASHSCOPE_API_KEY") or read_api_key(args.api_file.resolve())
    audio_bytes = audio_path.read_bytes()
    try:
        try:
            from openai import OpenAI
        except ImportError:
            raw_text, usage = stream_with_urllib(
                args.base_url,
                api_key,
                prompt_text,
                audio_bytes,
                args.model,
                args.max_tokens,
            )
            transport = "urllib-fallback"
        else:
            client = OpenAI(api_key=api_key, base_url=args.base_url.rstrip("/"))
            completion = client.chat.completions.create(
                model=args.model,
                messages=make_messages(prompt_text, audio_bytes),
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
                temperature=0,
                max_tokens=args.max_tokens,
            )
            text_parts = []
            usage = None
            for chunk in completion:
                if getattr(chunk, "usage", None) is not None:
                    usage = usage_dict(chunk.usage)
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    text_parts.append(delta_text(choices[0].delta))
            raw_text = "".join(text_parts)
            transport = "openai-sdk"
    except Exception as error:
        print("Qwen request failed: %s" % redact(error, api_key), file=sys.stderr)
        return 1

    raw_text = raw_text.strip()
    if not raw_text:
        print("Qwen returned empty text", file=sys.stderr)
        return 1
    try:
        parsed = parse_json_object(raw_text)
        validate_descriptive(parsed)
    except ValueError as error:
        print("invalid descriptive JSON: %s" % error, file=sys.stderr)
        print(raw_text, file=sys.stderr)
        return 1

    result = {
        "sample_id": audio_path.stem,
        "audio_path": str(audio_path),
        "audio_sha256": sha256_file(audio_path),
        "model": args.model,
        "base_url": args.base_url,
        "transport": transport,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_text(prompt_text),
        "usage": usage,
        "raw_text": raw_text,
        "descriptive": parsed["descriptive"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("saved: %s" % args.output.resolve(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
