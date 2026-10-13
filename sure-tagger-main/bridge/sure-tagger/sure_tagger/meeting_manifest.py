from collections import OrderedDict

from sure_tagger.io.jsonl import read_jsonl
from sure_tagger.schemas import make_error, validate_manifest_record
from sure_tagger.text.normalize import normalize_transcript


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sample_sort_key(record):
    sample = record.get("sample", {})
    meta = sample.get("native_metadata", {})
    audio = sample.get("audio", {})
    start = _safe_float(audio.get("start_sec"), -1.0)
    end = _safe_float(audio.get("end_sec"), -1.0)
    return (
        meta.get("meeting_id", ""),
        start,
        end,
        meta.get("speaker_id", ""),
        sample.get("sample_id", ""),
    )


def _format_time(value):
    value = _safe_float(value)
    if value is None:
        return "?:??:??"
    total = int(round(value))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return "%d:%02d:%02d" % (h, m, s)


def _record_text(record):
    return record.get("sample", {}).get("text", {}).get("transcript", "") or ""


def _speaker_line(record):
    sample = record.get("sample", {})
    meta = sample.get("native_metadata", {})
    audio = sample.get("audio", {})
    text = _record_text(record).strip()
    if not text:
        return ""
    speaker = meta.get("speaker_id") or "unknown_speaker"
    start = _format_time(audio.get("start_sec"))
    end = _format_time(audio.get("end_sec"))
    return "[%s-%s] %s: %s" % (start, end, speaker, text)


def _first_value(records, getter, default=""):
    for rec in records:
        value = getter(rec)
        if value:
            return value
    return default


def aggregate_meeting_records(records, source_manifest_path=""):
    groups = OrderedDict()
    for record in sorted(records, key=_sample_sort_key):
        sample = record.get("sample", {})
        meta = sample.get("native_metadata", {})
        meeting_id = meta.get("meeting_id") or sample.get("sample_id")
        groups.setdefault(meeting_id, []).append(record)

    meeting_records = []
    for meeting_id, group in groups.items():
        ordered = sorted(group, key=_sample_sort_key)
        first = ordered[0]
        corpus = dict(first.get("corpus", {}))
        corpus.setdefault("native_metadata", {})
        corpus["native_metadata"] = dict(corpus.get("native_metadata", {}))

        texts = []
        speaker_lines = []
        speakers = set()
        starts = []
        ends = []
        nonword_event_count = 0
        source_paths = set()

        for rec in ordered:
            sample = rec.get("sample", {})
            meta = sample.get("native_metadata", {})
            audio = sample.get("audio", {})
            text = _record_text(rec).strip()
            if text:
                texts.append(text)
            line = _speaker_line(rec)
            if line:
                speaker_lines.append(line)
            if meta.get("speaker_id"):
                speakers.add(meta.get("speaker_id"))
            start = _safe_float(audio.get("start_sec"))
            end = _safe_float(audio.get("end_sec"))
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
            events = meta.get("nonword_events", [])
            if isinstance(events, list):
                nonword_event_count += len(events)
            source_path = sample.get("provenance", {}).get("source_path")
            if source_path:
                source_paths.add(source_path)

        transcript = "\n".join(texts)
        speaker_labeled = "\n".join(speaker_lines)
        audio_path = _first_value(
            ordered,
            lambda rec: rec.get("sample", {}).get("audio", {}).get("path"),
            "",
        )
        dataset_name = corpus.get("dataset_name", "dataset")
        sample_id = "%s:%s" % (dataset_name, meeting_id)
        first_meta = first.get("sample", {}).get("native_metadata", {})
        start_sec = min(starts) if starts else None
        end_sec = max(ends) if ends else None

        meeting_record = {
            "corpus": corpus,
            "sample": {
                "sample_id": sample_id,
                "audio": {
                    "path": audio_path,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                },
                "text": {
                    "transcript": transcript,
                    "normalized_transcript": normalize_transcript(transcript),
                    "speaker_labeled_transcript": speaker_labeled,
                },
                "native_metadata": {
                    "granularity": "meeting",
                    "meeting_id": meeting_id,
                    "meeting_type": first_meta.get("meeting_type", ""),
                    "source_sample_count": len(ordered),
                    "text_segment_count": len(texts),
                    "speaker_ids": sorted(speakers),
                    "nonword_event_count": nonword_event_count,
                    "source_manifest": source_manifest_path,
                    "source_path_count": len(source_paths),
                },
                "provenance": {
                    "source_path": source_manifest_path,
                    "source_split": "",
                },
            },
        }
        meeting_records.append(meeting_record)
    return meeting_records


def build_meeting_manifest(input_manifest, output_writer, bad_writer=None, limit=None, meetings=None):
    meetings_set = set(meetings or [])
    records = []
    stats = {
        "input_records": 0,
        "records": 0,
        "bad_records": 0,
        "meetings_seen": 0,
        "filtered_meetings": len(meetings_set),
        "source_manifest": input_manifest,
    }

    for record in read_jsonl(input_manifest):
        stats["input_records"] += 1
        sample_id = record.get("sample", {}).get("sample_id", "unknown")
        try:
            validate_manifest_record(record)
            meeting_id = record["sample"]["native_metadata"].get("meeting_id", "")
            if meetings_set and meeting_id not in meetings_set:
                continue
            records.append(record)
        except Exception as exc:
            stats["bad_records"] += 1
            if bad_writer:
                bad_writer.write(make_error(sample_id, "build_meeting_manifest", exc.__class__.__name__, exc, input_manifest))

    meeting_records = aggregate_meeting_records(records, source_manifest_path=input_manifest)
    stats["meetings_seen"] = len(meeting_records)
    for meeting_record in meeting_records:
        output_writer.write(meeting_record)
        stats["records"] += 1
        if limit and stats["records"] >= int(limit):
            break
    return stats
