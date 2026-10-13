import json
import os

from sure_tagger.schemas import make_error


REQUIRED_FIELDS = ["utt_id", "audio_id", "speaker", "start", "end", "text", "words"]


def _safe_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % field_name)


def _jsonl_files(root, meetings=None):
    meetings_set = set(meetings or [])
    if os.path.isfile(root):
        base = os.path.basename(root)
        audio_id = base[:-6] if base.endswith(".jsonl") else base
        if meetings_set and audio_id not in meetings_set:
            return []
        return [root]

    if not os.path.isdir(root):
        raise ValueError("AMI utterance root is not a directory or file: %s" % root)

    paths = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        audio_id = name[:-6]
        if meetings_set and audio_id not in meetings_set:
            continue
        paths.append(os.path.join(root, name))
    return paths


def iter_utterance_jsonl_files(root, meetings=None):
    for path in _jsonl_files(root, meetings=meetings):
        yield path


def _require_raw_fields(raw):
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError("utterance missing required fields: %s" % ", ".join(missing))


def _normalize_words(words):
    if not isinstance(words, list):
        raise ValueError("words must be a list")
    normalized = []
    for index, item in enumerate(words):
        if not isinstance(item, dict):
            raise ValueError("words[%s] must be an object" % index)
        word = {
            "w": str(item.get("w", "")),
        }
        if "start" in item and item.get("start") is not None:
            word["start"] = _safe_float(item.get("start"), "words[%s].start" % index)
        if "end" in item and item.get("end") is not None:
            word["end"] = _safe_float(item.get("end"), "words[%s].end" % index)
        normalized.append(word)
    return normalized


def record_from_utterance(raw, source_path, line_no):
    _require_raw_fields(raw)
    utt_id = str(raw["utt_id"])
    audio_id = str(raw["audio_id"])
    speaker_id = str(raw["speaker"])
    start_sec = _safe_float(raw["start"], "start")
    end_sec = _safe_float(raw["end"], "end")
    if end_sec < start_sec:
        raise ValueError("end must be greater than or equal to start")

    transcript = str(raw.get("text", ""))
    words = _normalize_words(raw.get("words", []))

    return {
        "corpus": {
            "dataset_name": "AMI",
            "source_urls": {
                "article": [],
                "github": [],
                "huggingface": [],
                "dataset_card": [],
            },
            "native_metadata": {},
        },
        "sample": {
            "sample_id": utt_id,
            "audio": {"path": ""},
            "text": {"transcript": transcript},
            "native_metadata": {
                "utt_id": utt_id,
                "audio_id": audio_id,
                "speaker": speaker_id,
                "start": start_sec,
                "end": end_sec,
                "text": transcript,
                "words": words,
            },
        },
    }


def build_utterance_manifest(root, output_writer, bad_writer=None, limit=None, meetings=None):
    meetings_set = set(meetings or [])
    stats = {
        "records": 0,
        "bad_records": 0,
        "source_files": 0,
        "filtered_meetings": len(meetings_set),
        "empty_text_count": 0,
        "duration_missing_count": 0,
        "total_text_words": 0,
        "meetings_seen": 0,
        "speakers_seen": [],
    }
    seen_meetings = set()
    seen_speakers = set()

    for path in iter_utterance_jsonl_files(root, meetings=meetings):
        stats["source_files"] += 1
        with open(path, "r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                sample_id = "%s:%s" % (os.path.basename(path), line_no)
                try:
                    raw = json.loads(line)
                    sample_id = str(raw.get("utt_id", sample_id))
                    record = record_from_utterance(raw, path, line_no)
                    text = record["sample"]["text"].get("transcript", "")
                    if not text.strip():
                        stats["empty_text_count"] += 1
                    metadata = record["sample"]["native_metadata"]
                    start_sec = metadata.get("start")
                    end_sec = metadata.get("end")
                    if start_sec is None or end_sec is None:
                        stats["duration_missing_count"] += 1
                    stats["total_text_words"] += len(text.split())
                    seen_meetings.add(metadata.get("audio_id", ""))
                    seen_speakers.add(metadata.get("speaker", ""))
                    output_writer.write(record)
                    stats["records"] += 1
                    if limit and stats["records"] >= int(limit):
                        stats["meetings_seen"] = len([m for m in seen_meetings if m])
                        stats["speakers_seen"] = sorted([s for s in seen_speakers if s])
                        stats["avg_words_per_utterance"] = (
                            float(stats["total_text_words"]) / float(stats["records"])
                            if stats["records"]
                            else 0.0
                        )
                        return stats
                except Exception as exc:
                    stats["bad_records"] += 1
                    if bad_writer:
                        bad_writer.write(
                            make_error(
                                sample_id,
                                "build_utterance_manifest",
                                exc.__class__.__name__,
                                exc,
                                path,
                            )
                        )

    stats["meetings_seen"] = len([m for m in seen_meetings if m])
    stats["speakers_seen"] = sorted([s for s in seen_speakers if s])
    stats["avg_words_per_utterance"] = (
        float(stats["total_text_words"]) / float(stats["records"])
        if stats["records"]
        else 0.0
    )
    return stats
