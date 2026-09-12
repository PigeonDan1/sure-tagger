import os
import re

import yaml

from sure_tagger.llm.cache import JsonlCache, stable_hash
from sure_tagger.llm.client import LLMClient, LLMError
from sure_tagger.llm.prompts import TOPIC_PROMPT_VERSION, build_topic_prompt
from sure_tagger.schemas import make_tag
from sure_tagger.text.key_terms import extract_keywords, extract_proper_nouns


TOOL_VERSION = "topic_v0.4.0"


BACKCHANNEL_WORDS = set([
    "ah",
    "aha",
    "alright",
    "aye",
    "eh",
    "er",
    "erm",
    "fine",
    "great",
    "hm",
    "hmm",
    "huh",
    "kay",
    "mhm",
    "mm",
    "no",
    "oh",
    "okay",
    "ok",
    "right",
    "sure",
    "uh",
    "uhh",
    "um",
    "umm",
    "yeah",
    "yep",
    "yes",
])


def load_taxonomy(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def taxonomy_labels(taxonomy):
    return taxonomy.get("labels", {})


def validate_topic_payload(payload, taxonomy):
    labels = taxonomy_labels(taxonomy)
    if "value" in payload and isinstance(payload["value"], dict):
        value = payload["value"]
        major = value.get("major_topic")
        minor = value.get("minor_topic")
    else:
        major = payload.get("major_topic")
        minor = payload.get("minor_topic")
    if major not in labels:
        raise ValueError("invalid major_topic: %s" % major)
    minors = labels[major].get("minors", [])
    if minor not in minors:
        raise ValueError("invalid minor_topic %s for major_topic %s" % (minor, major))
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    keywords = payload.get("topic_keywords", [])
    proper = payload.get("proper_nouns", [])
    if not isinstance(keywords, list):
        keywords = []
    if not isinstance(proper, list):
        proper = []
    return {
        "major_topic": major,
        "minor_topic": minor,
        "confidence": confidence,
        "topic_keywords": [str(x) for x in keywords],
        "proper_nouns": [str(x) for x in proper],
        "reason_short": str(payload.get("reason_short", ""))[:240],
        "secondary_topics": payload.get("secondary_topics", []),
    }


def heuristic_topic(text, context, taxonomy):
    full = " ".join([
        text or "",
        context.get("meeting_window_text", ""),
        context.get("speaker_window_text", ""),
        context.get("neighbor_window_text", ""),
        context.get("same_speaker_window_text", ""),
    ]).lower()
    major = "other"
    minor = "insufficient_context"
    confidence = 0.45

    rules = [
        (["remote", "control", "prototype", "industrial designer", "interface"], "technology_engineering", "product_design", 0.78),
        (["software", "code", "debug", "pipeline", "api", "database"], "technology_engineering", "software_engineering", 0.72),
        (["speech recognition", "asr", "model", "tagger", "dataset", "transcript"], "technology_engineering", "artificial_intelligence", 0.74),
        (["algorithm", "theorem", "proof", "mathematics", "physics", "computer science", "paper"], "academic_research", "computer_science", 0.70),
        (["budget", "cost", "price", "euro", "finance", "accounting"], "business_management", "finance_accounting", 0.72),
        (["market", "sales", "customer", "selling"], "business_management", "marketing_sales", 0.70),
        (["agenda", "schedule", "meeting", "action item", "decision"], "meeting_workflow", "coordination", 0.65),
        (["doctor", "patient", "diagnosis", "medicine", "health"], "health_medicine", "clinical_medicine", 0.70),
        (["law", "legal", "contract", "policy", "regulation"], "law_policy_government", "law", 0.70),
        (["lesson", "exam", "homework", "course", "teacher"], "education_training", "classroom_discussion", 0.68),
        (["news", "election", "government", "climate"], "news_current_events", "politics", 0.66),
        (["movie", "music", "game", "book", "art"], "culture_media_arts", "pop_culture", 0.65),
        (["food", "travel", "family", "shopping", "house"], "daily_life_social", "personal_experience", 0.62),
        (["refund", "billing", "account", "troubleshoot", "complaint"], "customer_service_support", "troubleshooting", 0.70),
    ]
    for cues, r_major, r_minor, conf in rules:
        if any(cue in full for cue in cues):
            major, minor, confidence = r_major, r_minor, conf
            break
    if len(full.strip()) < 20:
        major, minor, confidence = "other", "insufficient_context", 0.35

    return {
        "major_topic": major,
        "minor_topic": minor,
        "confidence": confidence,
        "topic_keywords": extract_keywords(full),
        "proper_nouns": extract_proper_nouns(" ".join([text or "", context.get("meeting_window_text", "")])),
        "reason_short": "Heuristic topic classification based on lexical cues.",
    }


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
    return default


def _guard_config(config):
    value = (config or {}).get("short_utterance_guard", {})
    if isinstance(value, dict):
        return value
    return {"enabled": value}


def topic_word_tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def is_non_content_utterance(text, config=None):
    guard = _guard_config(config)
    if not _as_bool(guard.get("enabled"), True):
        return False
    max_tokens = _as_int(guard.get("max_tokens"), 3)
    tokens = topic_word_tokens(text)
    if not tokens or len(tokens) > max_tokens:
        return False
    words = set(BACKCHANNEL_WORDS)
    for item in guard.get("extra_words", []):
        words.add(str(item).strip().lower())
    return all(token in words for token in tokens)


def resolve_topic_schema_path(config):
    model_conf = config.get("model") or {}
    use_json_schema = _as_bool(
        config.get("use_json_schema", model_conf.get("use_json_schema")),
        True,
    )
    if not use_json_schema:
        return None
    schema_path = config.get("schema_path")
    if schema_path is None:
        schema_path = os.path.join("configs", "topic_response_schema.json")
    if isinstance(schema_path, str) and schema_path.strip().lower() in ("", "none", "null", "false", "off"):
        return None
    return schema_path


def _topic_texts(record):
    sample = record["sample"]
    text_obj = sample.get("text", {})
    plain_text = text_obj.get("transcript", "") or ""
    meta = sample.get("native_metadata", {})
    if meta.get("granularity") == "meeting":
        llm_text = text_obj.get("speaker_labeled_transcript") or plain_text
    else:
        llm_text = plain_text
    return plain_text, llm_text


def _topic_limits(config):
    meeting_conf = config.get("meeting", {})
    single_call_max_chars = _as_int(
        meeting_conf.get("single_call_max_chars", config.get("single_call_max_chars", 80000)),
        80000,
    )
    chunk_chars = _as_int(
        meeting_conf.get("chunk_chars", config.get("chunk_chars", single_call_max_chars)),
        single_call_max_chars,
    )
    single_call_max_chars = max(1000, single_call_max_chars)
    chunk_chars = max(1000, chunk_chars)
    return single_call_max_chars, chunk_chars


def split_text_for_llm(text, max_chars):
    text = text or ""
    if len(text) <= int(max_chars):
        return [text]

    chunks = []
    current = []
    current_len = 0
    lines = text.splitlines()
    if not lines:
        lines = [text]

    def flush():
        if current:
            chunks.append("\n".join(current))
            del current[:]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > int(max_chars):
            flush()
            start = 0
            while start < len(line):
                chunks.append(line[start:start + int(max_chars)])
                start += int(max_chars)
            current_len = 0
            continue
        extra = len(line) + (1 if current else 0)
        if current and current_len + extra > int(max_chars):
            flush()
            current_len = 0
        current.append(line)
        current_len += extra
    flush()
    return chunks or [text[:int(max_chars)]]


def _unique(items, max_items=12):
    seen = set()
    out = []
    for item in items:
        value = str(item)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= int(max_items):
            break
    return out


def merge_chunk_payloads(payloads, taxonomy, chunk_lengths):
    cleaned = []
    total_weight = 0.0
    scores = {}
    support = {}
    chunk_votes = []
    all_keywords = []
    all_proper = []

    for idx, payload in enumerate(payloads):
        clean = validate_topic_payload(payload, taxonomy)
        weight = float(max(1, int(chunk_lengths[idx])))
        total_weight += weight
        key = (clean["major_topic"], clean["minor_topic"])
        scores[key] = scores.get(key, 0.0) + clean["confidence"] * weight
        support[key] = support.get(key, 0.0) + weight
        cleaned.append(clean)
        all_keywords.extend(clean["topic_keywords"])
        all_proper.extend(clean["proper_nouns"])
        chunk_votes.append({
            "chunk_index": idx + 1,
            "major_topic": clean["major_topic"],
            "minor_topic": clean["minor_topic"],
            "confidence": clean["confidence"],
            "char_count": int(chunk_lengths[idx]),
        })

    if not cleaned:
        raise ValueError("no chunk payloads to merge")

    best_key = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = scores[best_key] / total_weight if total_weight else 0.0
    secondary = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        if key == best_key:
            continue
        secondary.append({
            "major_topic": key[0],
            "minor_topic": key[1],
            "support_ratio": support[key] / total_weight if total_weight else 0.0,
            "weighted_confidence": score / total_weight if total_weight else 0.0,
        })

    return {
        "major_topic": best_key[0],
        "minor_topic": best_key[1],
        "confidence": max(0.0, min(1.0, confidence)),
        "topic_keywords": _unique(all_keywords),
        "proper_nouns": _unique(all_proper),
        "reason_short": "Merged chunk-level topic predictions by confidence-weighted transcript length.",
        "secondary_topics": secondary,
        "_chunk_votes": chunk_votes,
    }


def _llm_topic_payload(llm_text, context, taxonomy, config, schema_path, dataset_metadata=None):
    client = LLMClient(config.get("model") or {})
    single_call_max_chars, chunk_chars = _topic_limits(config)
    if len(llm_text or "") <= single_call_max_chars:
        prompt = build_topic_prompt(taxonomy, llm_text, context, dataset_metadata or {})
        payload = client.complete_json(prompt, schema_path=schema_path)
        return payload, {
            "llm_call_count": 1,
            "chunk_count": 1,
            "single_call_max_chars": single_call_max_chars,
            "chunk_chars": chunk_chars,
            "chunking_strategy": "single_call",
            "input_char_count": len(llm_text or ""),
        }

    chunks = split_text_for_llm(llm_text, chunk_chars)
    payloads = []
    chunk_lengths = []
    for idx, chunk in enumerate(chunks):
        chunk_context = dict(context or {})
        chunk_context.update({
            "evidence_scope": "meeting_chunk",
            "chunk_index": idx + 1,
            "chunk_count": len(chunks),
            "chunk_char_count": len(chunk),
        })
        prompt = build_topic_prompt(taxonomy, chunk, chunk_context, dataset_metadata or {})
        payloads.append(client.complete_json(prompt, schema_path=schema_path))
        chunk_lengths.append(len(chunk))

    merged = merge_chunk_payloads(payloads, taxonomy, chunk_lengths)
    chunk_votes = merged.pop("_chunk_votes", [])
    return merged, {
        "llm_call_count": len(chunks),
        "chunk_count": len(chunks),
        "single_call_max_chars": single_call_max_chars,
        "chunk_chars": chunk_chars,
        "chunking_strategy": "chunk_then_weighted_vote",
        "input_char_count": len(llm_text or ""),
        "chunk_char_counts": chunk_lengths,
        "chunk_topic_votes": chunk_votes,
    }


def tag(record, config=None, context=None):
    config = config or {}
    context = context or {}
    taxonomy_path = config.get("taxonomy_path") or os.path.join("configs", "topic_taxonomy_general.yaml")
    taxonomy = load_taxonomy(taxonomy_path)
    provider = (config.get("model") or {}).get("provider", config.get("method", "heuristic"))
    text, llm_text = _topic_texts(record)
    meta = record["sample"].get("native_metadata", {})
    native_granularity = meta.get("granularity") or (
        "utterance" if meta.get("utt_id") or meta.get("utterances") else "segment"
    )
    target_granularity = context.get("target_granularity", native_granularity)
    single_call_max_chars, chunk_chars = _topic_limits(config)
    schema_path = resolve_topic_schema_path(config)

    if target_granularity == "utterance" and is_non_content_utterance(text, config):
        value = {
            "major_topic": "other",
            "minor_topic": "insufficient_context",
        }
        details = {
            "taxonomy": taxonomy.get("version"),
            "prompt_version": TOPIC_PROMPT_VERSION,
            "provider": provider,
            "evidence_scope": context.get("evidence_scope", "sample"),
            "evidence_sample_count": context.get("evidence_sample_count", 1),
            "granularity": native_granularity,
            "target_granularity": target_granularity,
            "input_char_count": len(llm_text or ""),
            "guard": "short_non_content_utterance",
            "topic_keywords": [],
            "proper_nouns": [],
            "reason_short": "Target utterance is a short acknowledgement, backchannel, filler, or non-content response.",
            "secondary_topics": [],
        }
        return make_tag(
            value,
            0.95,
            "deterministic_non_content_utterance_guard",
            TOOL_VERSION,
            "L1",
            details,
        )

    cache_conf = config.get("cache", {})
    cache = JsonlCache(cache_conf.get("path")) if cache_conf.get("enabled", True) else None
    prompt = build_topic_prompt(taxonomy, llm_text, context, record.get("corpus", {}).get("native_metadata", {}))
    cache_key = stable_hash({
        "tool": TOOL_VERSION,
        "prompt_version": TOPIC_PROMPT_VERSION,
        "taxonomy": taxonomy.get("version"),
        "provider": provider,
        "single_call_max_chars": single_call_max_chars,
        "chunk_chars": chunk_chars,
        "use_json_schema": bool(schema_path),
        "schema_path": schema_path or "",
        "prompt": prompt,
    })

    payload = None
    cache_payload = True
    method = "heuristic_hierarchical_classification"
    reliability = "L1"
    model_details = {
        "taxonomy": taxonomy.get("version"),
        "prompt_version": TOPIC_PROMPT_VERSION,
        "provider": provider,
        "cache_key": cache_key,
        "evidence_scope": context.get("evidence_scope", "sample"),
        "evidence_sample_count": context.get("evidence_sample_count", 1),
        "granularity": native_granularity,
        "target_granularity": target_granularity,
        "single_call_max_chars": single_call_max_chars,
        "chunk_chars": chunk_chars,
        "use_json_schema": bool(schema_path),
        "input_char_count": len(llm_text or ""),
    }

    if cache:
        payload = cache.get(cache_key)
        if payload:
            model_details["cache_hit"] = True
            if provider in ("heuristic", "dry_run"):
                method = "heuristic_hierarchical_classification_cached"
                reliability = "L1"
            else:
                method = "llm_hierarchical_classification_cached"
                reliability = "L2"

    if payload is None:
        if provider in ("heuristic", "dry_run"):
            payload = heuristic_topic(text, context, taxonomy)
            if provider == "dry_run":
                model_details["dry_run"] = True
        else:
            try:
                payload, llm_details = _llm_topic_payload(
                    llm_text,
                    context,
                    taxonomy,
                    config,
                    schema_path,
                    record.get("corpus", {}).get("native_metadata", {}),
                )
                model_details.update(llm_details)
                if llm_details.get("chunk_count", 1) > 1:
                    method = "llm_chunked_hierarchical_classification"
                else:
                    method = "llm_hierarchical_classification"
                reliability = "L2"
            except (LLMError, ValueError) as exc:
                model_details["llm_error"] = str(exc)
                cache_payload = False
                fallback = config.get("fallback", "heuristic")
                if fallback != "heuristic":
                    raise
                payload = heuristic_topic(text, context, taxonomy)
                method = "heuristic_fallback_after_llm_error"
                reliability = "L1"
        if cache and cache_payload:
            cache.set(cache_key, payload)

    clean = validate_topic_payload(payload, taxonomy)
    value = {
        "major_topic": clean["major_topic"],
        "minor_topic": clean["minor_topic"],
    }
    details = dict(model_details)
    details.update({
        "topic_keywords": clean["topic_keywords"],
        "proper_nouns": clean["proper_nouns"],
        "reason_short": clean["reason_short"],
        "secondary_topics": clean["secondary_topics"],
    })
    return make_tag(value, clean["confidence"], method, TOOL_VERSION, reliability, details)
