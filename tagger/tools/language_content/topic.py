"""Legacy OpenAI Responses topic classifier, unused by the active pipeline."""

import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from tagger import local_config
from tagger.tools.base import ToolResult


TOOL_NAME = "topic_classifier"
TOOL_VERSION = "topic_openai_responses_v0.1.0"
PROMPT_VERSION = "topic_hierarchical_v0.5.0"
DEFAULT_CACHE_PATH = "outputs/cache/topic_openai_responses_cache.jsonl"

BACKCHANNEL_WORDS = set(
    [
        "ah",
        "aha",
        "alright",
        "eh",
        "er",
        "erm",
        "fine",
        "great",
        "hm",
        "hmm",
        "huh",
        "mhm",
        "mm",
        "no",
        "oh",
        "okay",
        "ok",
        "right",
        "sure",
        "uh",
        "um",
        "yeah",
        "yep",
        "yes",
    ]
)

TAXONOMY = {
    "version": "general_topic_v0.1.0",
    "labels": {
        "academic_research": [
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "computer_science",
            "engineering",
            "medicine",
            "economics",
            "psychology",
            "philosophy",
            "linguistics",
            "history",
            "interdisciplinary",
        ],
        "technology_engineering": [
            "artificial_intelligence",
            "software_engineering",
            "data_science",
            "cybersecurity",
            "hardware",
            "robotics",
            "telecommunications",
            "product_design",
            "user_experience",
            "manufacturing",
        ],
        "business_management": [
            "strategy",
            "marketing_sales",
            "finance_accounting",
            "operations",
            "human_resources",
            "entrepreneurship",
            "project_management",
            "customer_success",
            "procurement",
        ],
        "law_policy_government": [
            "law",
            "regulation",
            "public_policy",
            "government_services",
            "compliance",
            "international_relations",
            "public_safety",
        ],
        "health_medicine": [
            "clinical_medicine",
            "public_health",
            "pharmacy",
            "mental_health",
            "fitness",
            "nutrition",
            "healthcare_operations",
        ],
        "education_training": [
            "lecture",
            "tutorial",
            "exam_preparation",
            "classroom_discussion",
            "language_learning",
            "professional_training",
            "mentoring",
        ],
        "culture_media_arts": [
            "literature",
            "music",
            "film_tv",
            "gaming",
            "visual_art",
            "religion",
            "media_production",
            "pop_culture",
        ],
        "news_current_events": [
            "politics",
            "economy",
            "local_news",
            "international_news",
            "climate_environment",
            "social_issues",
            "breaking_news",
        ],
        "daily_life_social": [
            "family",
            "food",
            "shopping",
            "housing",
            "travel_transportation",
            "interpersonal_chat",
            "personal_experience",
            "small_talk",
        ],
        "customer_service_support": [
            "product_inquiry",
            "troubleshooting",
            "complaint",
            "billing",
            "account_support",
            "appointment",
            "refund_exchange",
        ],
        "meeting_workflow": [
            "agenda",
            "scheduling",
            "status_update",
            "decision",
            "action_item",
            "brainstorming",
            "coordination",
            "opening_closing",
        ],
        "other": ["unknown", "insufficient_context", "mixed_topics", "non_speech"],
    },
}

TOPIC_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "major_topic": {"type": "string"},
        "minor_topic": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "topic_keywords": {"type": "array", "items": {"type": "string"}},
        "proper_nouns": {"type": "array", "items": {"type": "string"}},
        "reason_short": {"type": "string"},
        "secondary_topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "major_topic": {"type": "string"},
                    "minor_topic": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["major_topic", "minor_topic", "confidence"],
            },
        },
    },
    "required": [
        "major_topic",
        "minor_topic",
        "confidence",
        "topic_keywords",
        "proper_nouns",
        "reason_short",
        "secondary_topics",
    ],
}


class TopicError(RuntimeError):
    """Raised when topic classification cannot return a valid label."""


class TopicConfig:
    def __init__(
        self,
        enabled=None,
        provider=None,
        model=None,
        base_url=None,
        api_key=None,
        api_key_path=None,
        model_provider=None,
        codex_config_path=None,
        timeout_sec=None,
        temperature=None,
        use_json_schema=None,
        cache_enabled=None,
        cache_path=None,
        fallback=None,
        short_guard_enabled=None,
        short_guard_max_tokens=None,
    ):
        self.enabled = _coalesce_bool(
            enabled,
            getattr(local_config, "TOPIC_ENABLE", False),
        )
        self.provider = provider or getattr(
            local_config,
            "TOPIC_PROVIDER",
            "openai_responses",
        )
        self.model = (
            model if model is not None else getattr(local_config, "TOPIC_MODEL", "")
        )
        self.base_url = (
            base_url if base_url is not None else getattr(local_config, "TOPIC_BASE_URL", "")
        )
        self.api_key = (
            api_key if api_key is not None else getattr(local_config, "TOPIC_API_KEY", "")
        )
        self.api_key_path = (
            api_key_path
            if api_key_path is not None
            else getattr(local_config, "TOPIC_API_KEY_PATH", "api.txt")
        )
        self.model_provider = (
            model_provider
            if model_provider is not None
            else getattr(local_config, "TOPIC_MODEL_PROVIDER", "")
        )
        self.codex_config_path = (
            codex_config_path
            if codex_config_path is not None
            else getattr(local_config, "TOPIC_CODEX_CONFIG_PATH", "")
        )
        self.timeout_sec = int(
            timeout_sec
            if timeout_sec is not None
            else getattr(local_config, "TOPIC_TIMEOUT_SEC", 180)
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(local_config, "TOPIC_TEMPERATURE", None)
        )
        self.use_json_schema = _coalesce_bool(
            use_json_schema,
            getattr(local_config, "TOPIC_USE_JSON_SCHEMA", False),
        )
        self.cache_enabled = _coalesce_bool(
            cache_enabled,
            getattr(local_config, "TOPIC_CACHE_ENABLED", True),
        )
        self.cache_path = (
            cache_path
            if cache_path is not None
            else getattr(local_config, "TOPIC_CACHE_PATH", DEFAULT_CACHE_PATH)
        )
        self.fallback = (
            fallback
            if fallback is not None
            else getattr(local_config, "TOPIC_FALLBACK", "null")
        )
        self.short_guard_enabled = _coalesce_bool(
            short_guard_enabled,
            getattr(local_config, "TOPIC_SHORT_GUARD_ENABLED", True),
        )
        self.short_guard_max_tokens = int(
            short_guard_max_tokens
            if short_guard_max_tokens is not None
            else getattr(local_config, "TOPIC_SHORT_GUARD_MAX_TOKENS", 3)
        )

    def to_record(self):
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key or self.api_key_path),
            "api_key_path": self.api_key_path,
            "model_provider": self.model_provider,
            "codex_config_path": self.codex_config_path,
            "timeout_sec": self.timeout_sec,
            "temperature": self.temperature,
            "use_json_schema": self.use_json_schema,
            "cache_enabled": self.cache_enabled,
            "cache_path": self.cache_path,
            "fallback": self.fallback,
            "short_guard_enabled": self.short_guard_enabled,
            "short_guard_max_tokens": self.short_guard_max_tokens,
        }


class OpenAIResponsesTopicClient:
    def __init__(self, config=None):
        self.config = config or TopicConfig()

    def complete_json(self, prompt):
        if self.config.provider != "openai_responses":
            raise TopicError("unsupported topic provider: %s" % self.config.provider)
        settings = _resolve_openai_settings(self.config)
        api_key = settings.get("api_key")
        if not api_key:
            raise TopicError("topic OpenAI API key is not configured")
        model = settings.get("model")
        if not model or model == "default":
            raise TopicError("topic OpenAI model is not configured")
        text_format = {"type": "json_object"}
        if self.config.use_json_schema:
            text_format = {
                "type": "json_schema",
                "name": "topic_response",
                "schema": TOPIC_RESPONSE_SCHEMA,
                "strict": True,
            }
        body = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": "Return strict JSON only."}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {"format": text_format},
            "store": False,
        }
        if self.config.temperature is not None:
            body["temperature"] = float(self.config.temperature)
        base_url = settings.get("base_url", "https://api.openai.com/v1").rstrip("/")
        request = urllib.request.Request(
            base_url + "/responses",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_sec,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise TopicError(
                "topic OpenAI Responses API HTTP %s: %s" % (exc.code, detail[:2000])
            )
        except urllib.error.URLError as exc:
            raise TopicError("topic OpenAI Responses API request failed: %s" % exc)
        except ValueError as exc:
            raise TopicError("topic OpenAI Responses API returned invalid JSON") from exc
        content = _response_text(payload)
        if not content:
            raise TopicError("topic OpenAI Responses API returned no output text")
        try:
            return _extract_json_object(content)
        except ValueError as exc:
            raise TopicError("topic OpenAI Responses API output is not JSON") from exc


class JsonlCache:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self._items = None

    def get(self, key):
        if not self.path:
            return None
        self._load()
        return self._items.get(key)

    def set(self, key, value):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(
                json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
            )
        if self._items is not None:
            self._items[key] = value

    def _load(self):
        if self._items is not None:
            return
        self._items = {}
        if not self.path or not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = row.get("key")
                if key:
                    self._items[key] = row.get("value")


def run(record, context=None, config=None, client=None):
    # type: (dict, object, TopicConfig, object) -> ToolResult
    config = config or TopicConfig()
    context = context or {}
    text = _transcript(record)
    llm_text = text
    topic_context = dict(context)
    topic_context["target_granularity"] = "sample"
    topic_context.setdefault("evidence_scope", "sample")
    topic_context.setdefault("evidence_sample_count", 1)

    if _is_non_content_utterance(text, config):
        payload = {
            "major_topic": "other",
            "minor_topic": "insufficient_context",
            "confidence": 0.95,
            "topic_keywords": [],
            "proper_nouns": [],
            "reason_short": "Target utterance is a short non-content response.",
            "secondary_topics": [],
        }
        return _result(payload, config, "deterministic_non_content_utterance_guard")

    prompt = build_topic_prompt(
        TAXONOMY,
        llm_text,
        topic_context,
        _dataset_metadata(record),
    )
    cache_key = _stable_hash(
        {
            "tool": TOOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "taxonomy": TAXONOMY["version"],
            "provider": config.provider,
            "model": config.model,
            "use_json_schema": config.use_json_schema,
            "prompt": prompt,
        }
    )
    cache = JsonlCache(config.cache_path) if config.cache_enabled else None
    payload = cache.get(cache_key) if cache else None
    method = "llm_hierarchical_classification_cached" if payload else None
    if payload is None:
        try:
            payload = (client or OpenAIResponsesTopicClient(config)).complete_json(prompt)
            method = "llm_hierarchical_classification"
            if cache:
                cache.set(cache_key, payload)
        except TopicError as exc:
            if config.fallback == "heuristic":
                payload = heuristic_topic(text)
                method = "heuristic_fallback_after_topic_api_error"
            elif config.fallback == "null":
                return ToolResult(
                    tag_path="language_content.topic",
                    value=None,
                    tool_name=TOOL_NAME,
                    method="topic_openai_responses_failed",
                    status="failed",
                    confidence=0.0,
                    tool_type="external_api",
                    tool_version=TOOL_VERSION,
                    evidence={
                        "error": str(exc),
                        "config": config.to_record(),
                        "prompt_version": PROMPT_VERSION,
                        "taxonomy": TAXONOMY["version"],
                    },
                )
            else:
                raise
    return _result(payload, config, method or "llm_hierarchical_classification")


def build_topic_prompt(taxonomy, transcript_text, context, dataset_metadata=None):
    labels = {}
    for major, minors in taxonomy["labels"].items():
        labels[major] = {"minors": list(minors)}
    rules = [
        "Return strict JSON only.",
        "Use exactly one major_topic and one minor_topic from the taxonomy.",
        "minor_topic must belong to the chosen major_topic.",
        "If evidence is insufficient, use other/insufficient_context.",
        "Classify the target transcript; use context only as supporting evidence.",
        "Do not invent proper nouns.",
    ]
    payload = {
        "task": "Classify an ASR transcript unit into one major_topic and one minor_topic.",
        "rules": rules,
        "output_schema_summary": {
            "major_topic": "string",
            "minor_topic": "string",
            "confidence": "number in [0, 1]",
            "topic_keywords": "array of strings",
            "proper_nouns": "array of strings",
            "reason_short": "short explanation",
        },
        "taxonomy": {"version": taxonomy["version"], "labels": labels},
        "dataset_metadata": dataset_metadata or {},
        "transcript_text": transcript_text or "",
        "context": context or {},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise TopicError("topic payload must be an object")
    value = payload.get("value") if isinstance(payload.get("value"), dict) else payload
    major = value.get("major_topic")
    minor = value.get("minor_topic")
    labels = TAXONOMY["labels"]
    if major not in labels:
        repaired_major = _major_for_minor(minor, labels)
        if repaired_major is None:
            raise TopicError("invalid major_topic: %s" % major)
        major = repaired_major
    if minor not in labels[major]:
        repaired_major = _major_for_minor(minor, labels)
        if repaired_major is None:
            raise TopicError(
                "invalid minor_topic %s for major_topic %s" % (minor, major)
            )
        major = repaired_major
    confidence = _float_in_range(value.get("confidence", 0.0), 0.0, 1.0)
    return {
        "major_topic": major,
        "minor_topic": minor,
        "confidence": confidence,
        "topic_keywords": _string_list(value.get("topic_keywords", [])),
        "proper_nouns": _string_list(value.get("proper_nouns", [])),
        "reason_short": str(value.get("reason_short", ""))[:240],
        "secondary_topics": value.get("secondary_topics", []),
    }


def _major_for_minor(minor, labels):
    for major, minors in labels.items():
        if minor in minors:
            return major
    return None


def heuristic_topic(text):
    full = (text or "").lower()
    rules = [
        (
            ["speech recognition", "asr", "tagger", "dataset", "transcript"],
            "technology_engineering",
            "artificial_intelligence",
            0.74,
        ),
        (
            ["software", "code", "debug", "pipeline", "api", "database"],
            "technology_engineering",
            "software_engineering",
            0.72,
        ),
        (
            ["meeting", "agenda", "schedule", "action item", "decision"],
            "meeting_workflow",
            "coordination",
            0.65,
        ),
        (
            ["budget", "price", "finance", "accounting"],
            "business_management",
            "finance_accounting",
            0.72,
        ),
        (
            ["doctor", "patient", "medicine", "health"],
            "health_medicine",
            "clinical_medicine",
            0.70,
        ),
    ]
    for cues, major, minor, confidence in rules:
        if any(cue in full for cue in cues):
            return {
                "major_topic": major,
                "minor_topic": minor,
                "confidence": confidence,
                "topic_keywords": _keywords(full),
                "proper_nouns": [],
                "reason_short": "Heuristic fallback based on lexical cues.",
                "secondary_topics": [],
            }
    return {
        "major_topic": "other",
        "minor_topic": "insufficient_context",
        "confidence": 0.35,
        "topic_keywords": _keywords(full),
        "proper_nouns": [],
        "reason_short": "Heuristic fallback found no strong topical cue.",
        "secondary_topics": [],
    }


def _result(payload, config, method):
    clean = validate_payload(payload)
    value = "%s/%s" % (clean["major_topic"], clean["minor_topic"])
    return ToolResult(
        tag_path="language_content.topic",
        value=value,
        tool_name=TOOL_NAME,
        method=method,
        status="estimated",
        confidence=clean["confidence"],
        tool_type="external_api" if method.startswith("llm_") else "deterministic",
        tool_version=TOOL_VERSION,
        evidence={
            "major_topic": clean["major_topic"],
            "minor_topic": clean["minor_topic"],
            "topic_keywords": clean["topic_keywords"],
            "proper_nouns": clean["proper_nouns"],
            "reason_short": clean["reason_short"],
            "secondary_topics": clean["secondary_topics"],
            "prompt_version": PROMPT_VERSION,
            "taxonomy": TAXONOMY["version"],
            "config": config.to_record(),
        },
    )


def _resolve_openai_settings(config):
    codex = _load_codex_config(config.codex_config_path)
    top = codex.get("top", {})
    sections = codex.get("sections", {})
    provider_name = (
        config.model_provider
        or os.environ.get("OPENAI_MODEL_PROVIDER")
        or top.get("model_provider")
    )
    provider_section = (
        sections.get("model_providers.%s" % provider_name, {}) if provider_name else {}
    )
    provider_env = (
        sections.get("model_providers.%s.env" % provider_name, {}) if provider_name else {}
    )
    return {
        "api_key": (
            os.environ.get("OPENAI_API_KEY")
            or config.api_key
            or _read_first_line(config.api_key_path)
            or provider_env.get("OPENAI_API_KEY")
        ),
        "model": (
            os.environ.get("OPENAI_MODEL")
            or config.model
            or top.get("model")
        ),
        "base_url": (
            os.environ.get("OPENAI_BASE_URL")
            or config.base_url
            or provider_section.get("base_url")
            or "https://api.openai.com/v1"
        ),
    }


def _load_codex_config(path=None):
    path = path or os.environ.get("CODEX_CONFIG_PATH") or os.path.expanduser(
        "~/.codex/config.toml"
    )
    result = {"top": {}, "sections": {}}
    if not path or not os.path.exists(path):
        return result
    section = "top"
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1].strip()
                result["sections"].setdefault(section, {})
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            value = _parse_toml_value(value)
            if section == "top":
                result["top"][key.strip()] = value
            else:
                result["sections"].setdefault(section, {})[key.strip()] = value
    return result


def _parse_toml_value(raw):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    return raw


def _read_first_line(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            value = line.strip()
            if value:
                return value
    return None


def _response_text(payload):
    content = payload.get("output_text")
    if content:
        return content
    parts = []
    for item in payload.get("output", []):
        for piece in item.get("content", []):
            if piece.get("type") in ("output_text", "text"):
                parts.append(piece.get("text", ""))
    return "\n".join(parts)


def _extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")


def _stable_hash(value):
    import hashlib

    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _transcript(record):
    return record.get("sample", {}).get("text", {}).get("transcript", "") or ""


def _dataset_metadata(record):
    return record.get("corpus", {}).get("native_metadata", {})


def _is_non_content_utterance(text, config):
    if not config.short_guard_enabled:
        return False
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())
    if not tokens or len(tokens) > config.short_guard_max_tokens:
        return False
    return all(token in BACKCHANNEL_WORDS for token in tokens)


def _keywords(text, max_items=8):
    tokens = re.findall(r"[a-z][a-z0-9_'-]{2,}", (text or "").lower())
    stop = set(["the", "and", "for", "that", "this", "with", "you", "are", "was"])
    seen = set()
    result = []
    for token in tokens:
        if token in stop or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max_items:
            break
    return result


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _float_in_range(value, lower, upper):
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = lower
    if result != result:
        result = lower
    return max(lower, min(upper, result))


def _coalesce_bool(value, default):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)
