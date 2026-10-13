#!/usr/bin/env python3
"""Build the masked AMI structural ground-truth manifest."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
import wave


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_UNIVERSE = SCRIPT_DIR / "field_universe.structural-v0.1.json"
DEFAULT_MANIFEST = COMPARE_ROOT / "dataset" / "Ami" / "data" / "manifest.jsonl"
DEFAULT_OUTPUT = COMPARE_ROOT / "evaluation" / "ami-v0" / "ground_truth.jsonl"

TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\w\s]",
    re.UNICODE,
)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("%s:%d is invalid JSON: %s" % (path, line_number, exc))
            if not isinstance(row, dict):
                raise ValueError("%s:%d must contain a JSON object" % (path, line_number))
            rows.append(row)
    return rows


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_jsonl(path, rows):
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, payload)


def write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_specs(universe):
    fields = universe.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("field universe must contain a non-empty fields array")
    by_path = {}
    for index, field in enumerate(fields):
        if not isinstance(field, dict) or not isinstance(field.get("path"), str):
            raise ValueError("fields[%d] has no path" % index)
        path = field["path"]
        if path in by_path:
            raise ValueError("duplicate field path: %s" % path)
        by_path[path] = field
    return by_path


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _normalize_segments(utterances, duration_sec, rule):
    minimum = float(rule["min_segment_duration_sec"])
    normalized = []
    for index, item in enumerate(utterances):
        if not isinstance(item, dict):
            raise ValueError("utterances[%d] must be an object" % index)
        speaker = item.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("utterances[%d].speaker is invalid" % index)
        start = max(0.0, _finite_number(item.get("start"), "utterance start"))
        end = min(duration_sec, _finite_number(item.get("end"), "utterance end"))
        if end <= start:
            raise ValueError("utterances[%d] has an empty interval" % index)
        if end - start < minimum:
            continue
        normalized.append(
            {
                "speaker_id": speaker.strip().lower().replace(" ", "_"),
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]),
    )


def _merge_per_speaker(segments, duration_sec, rule):
    grouped = {}
    for segment in segments:
        grouped.setdefault(segment["speaker_id"], []).append(segment)
    maximum_gap = float(rule["merge_same_speaker_gap_sec"])
    merged_by_speaker = {}
    for speaker_id, items in grouped.items():
        merged = []
        for item in sorted(items, key=lambda value: (value["start_sec"], value["end_sec"])):
            current = dict(item)
            current["start_sec"] = max(0.0, float(current["start_sec"]))
            current["end_sec"] = min(duration_sec, float(current["end_sec"]))
            if not merged or current["start_sec"] - merged[-1]["end_sec"] > maximum_gap:
                merged.append(current)
            else:
                merged[-1]["end_sec"] = max(merged[-1]["end_sec"], current["end_sec"])
        merged_by_speaker[speaker_id] = merged
    return merged_by_speaker


def _union_duration(intervals):
    ordered = sorted(
        (float(item[0]), float(item[1])) for item in intervals if item[1] > item[0]
    )
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return round(sum(end - start for start, end in merged), 6)


def _overlap_intervals(merged_by_speaker, duration_sec, minimum_duration_sec):
    events = []
    for speaker_id, segments in merged_by_speaker.items():
        for segment in segments:
            start = max(0.0, float(segment["start_sec"]))
            end = min(duration_sec, float(segment["end_sec"]))
            if end <= start:
                continue
            events.append((start, 1, speaker_id))
            events.append((end, -1, speaker_id))
    events.sort(key=lambda item: (item[0], -item[1]))
    active = set()
    previous_time = None
    overlaps = []
    for time_sec, delta, speaker_id in events:
        if previous_time is not None and time_sec > previous_time and len(active) >= 2:
            if time_sec - previous_time >= minimum_duration_sec:
                overlaps.append((previous_time, time_sec))
        if delta > 0:
            active.add(speaker_id)
        else:
            active.discard(speaker_id)
        previous_time = time_sec
    return overlaps


def _speaker_change_count(merged_by_speaker, maximum_gap_sec):
    merged = []
    for speaker_id, items in merged_by_speaker.items():
        for item in items:
            value = dict(item)
            value["speaker_id"] = speaker_id
            merged.append(value)
    merged.sort(
        key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"])
    )
    points = []
    previous = None
    for segment in merged:
        if previous is not None and segment["speaker_id"] != previous["speaker_id"]:
            gap = float(segment["start_sec"]) - float(previous["end_sec"])
            if gap <= maximum_gap_sec:
                point = float(segment["start_sec"])
                if not points or abs(point - points[-1]) > 1e-6:
                    points.append(point)
        previous = segment
    return len(points)


def derive_speaker_fields(utterances, duration_sec, rule):
    """Derive aggregate speaker fields with the frozen benchmark rule."""
    duration_sec = _finite_number(duration_sec, "duration_sec")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("AMI sample must contain at least one utterance")
    segments = _normalize_segments(utterances, duration_sec, rule)
    if not segments:
        raise ValueError("all AMI utterances were removed by the minimum duration rule")
    merged_by_speaker = _merge_per_speaker(segments, duration_sec, rule)
    minimum = float(rule["min_segment_duration_sec"])
    active_speakers = []
    for speaker_id, items in merged_by_speaker.items():
        active_duration = sum(item["end_sec"] - item["start_sec"] for item in items)
        if active_duration >= minimum:
            active_speakers.append(speaker_id)
    speech_intervals = [
        (item["start_sec"], item["end_sec"])
        for items in merged_by_speaker.values()
        for item in items
    ]
    speech_union = _union_duration(speech_intervals)
    overlaps = _overlap_intervals(
        merged_by_speaker,
        duration_sec,
        float(rule["overlap_min_duration_sec"]),
    )
    overlap_duration = round(sum(end - start for start, end in overlaps), 6)
    overlap_ratio = round(overlap_duration / speech_union, 6) if speech_union > 0 else None
    change_count = _speaker_change_count(
        merged_by_speaker, float(rule["speaker_change_max_gap_sec"])
    )
    speaker_count = len(active_speakers)
    return {
        "speaker.speaker_count": speaker_count,
        "speaker.multi_speaker": speaker_count >= 2,
        "speaker.speaker_change_count": change_count,
        "speaker.speaker_change": change_count > 0,
        "speaker.overlap_ratio": overlap_ratio,
        "speaker.speaker_overlap": bool(overlaps),
    }


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def is_word_token(token):
    if not token:
        return False
    if re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", token):
        return True
    if re.match(r"^\d+(?:\.\d+)?$", token):
        return True
    return len(token) == 1 and "\u4e00" <= token <= "\u9fff"


def word_tokens(text):
    return [token for token in tokenize(text) if is_word_token(token)]


def _repetition_count(tokens, maximum_ngram):
    spans = 0
    token_count = len(tokens)
    for size in range(int(maximum_ngram), 0, -1):
        index = 0
        while index + 2 * size <= token_count:
            if tokens[index : index + size] == tokens[index + size : index + 2 * size]:
                spans += 1
                index += 2 * size
            else:
                index += 1
    return spans


def derive_text_fields(text, rule):
    """Derive lexical fields without importing either evaluated system."""
    if not isinstance(text, str):
        raise ValueError("reference transcript must be a string")
    words = word_tokens(text)
    punctuation_count = sum(
        1 for character in text if unicodedata.category(character).startswith("P")
    )
    stripped = text.strip()
    terminal = bool(stripped and stripped[-1] in ".\u3002?!\uff1f\uff01")
    normalized = [token.lower() for token in words]
    repetition_count = _repetition_count(
        normalized, int(rule["repetition_max_ngram"])
    )
    fillers = set(str(value).lower() for value in rule["filler_lexicon"])
    filler_count = sum(token.lower() in fillers for token in words)
    return {
        "language_content.word_count": len(words),
        "language_content.punctuation.punctuation_count": punctuation_count,
        "language_content.punctuation.has_terminal_punctuation": terminal,
        "language_content.repetition.has_repetition": repetition_count > 0,
        "language_content.repetition.repetition_count": repetition_count,
        "language_content.filler": filler_count,
    }


def audio_header(path):
    try:
        with wave.open(str(path), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            frames = source.getnframes()
            compression = source.getcomptype()
    except (OSError, wave.Error) as exc:
        raise ValueError("cannot read WAV header %s: %s" % (path, exc))
    if compression != "NONE" or sample_rate <= 0 or channels <= 0:
        raise ValueError("unsupported WAV header: %s" % path)
    return {
        "duration_sec": frames / float(sample_rate),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "frames": frames,
    }


def _provenance(paths, source, strength, rule_version):
    return (
        {path: strength for path in paths},
        {path: source for path in paths},
        {path: rule_version for path in paths},
    )


def build_record(source_row, manifest_path, universe, fields, language):
    sample = source_row.get("sample")
    corpus = source_row.get("corpus")
    if not isinstance(sample, dict) or not isinstance(corpus, dict):
        raise ValueError("AMI source row must contain corpus and sample objects")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("AMI source row has no sample_id")
    audio_value = sample.get("audio", {}).get("path")
    if not isinstance(audio_value, str) or not audio_value:
        raise ValueError("%s has no audio path" % sample_id)
    audio_path = Path(audio_value)
    if not audio_path.is_absolute():
        audio_path = manifest_path.parent / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))
    header = audio_header(audio_path)
    native = sample.get("native_metadata")
    if not isinstance(native, dict):
        raise ValueError("%s has no native metadata" % sample_id)
    native_start = _finite_number(native.get("start"), "%s native start" % sample_id)
    native_end = _finite_number(native.get("end"), "%s native end" % sample_id)
    native_duration = native_end - native_start
    frame_tolerance = max(0.001, 1.0 / float(header["sample_rate_hz"]))
    if abs(native_duration - header["duration_sec"]) > frame_tolerance:
        raise ValueError(
            "%s native/header duration mismatch: %.9f vs %.9f"
            % (sample_id, native_duration, header["duration_sec"])
        )
    utterances = native.get("utterances")
    transcript = sample.get("text", {}).get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("%s has no reference transcript" % sample_id)

    values = {path: None for path in fields}
    values.update(
        {
            "basic_acoustic.duration_sec": round(header["duration_sec"], 9),
            "basic_acoustic.sample_rate_hz": header["sample_rate_hz"],
            "basic_acoustic.channels": header["channels"],
            "sound_field_scene.speech_music_events": ["speech"],
            "speaker.asr_transcript": transcript,
            "language_content.language": language,
        }
    )
    values.update(
        derive_speaker_fields(utterances, header["duration_sec"], universe["speaker_rule"])
    )
    values.update(derive_text_fields(transcript, universe["text_rule"]))
    values["speaker.speaker_present"] = values["speaker.speaker_count"] > 0
    values["deterministic_derived.speaker_change_rate_per_min"] = round(
        values["speaker.speaker_change_count"] * 60.0 / values["basic_acoustic.duration_sec"],
        6,
    )
    values["deterministic_derived.filler_ratio"] = (
        round(
            float(values["language_content.filler"])
            / float(values["language_content.word_count"]),
            6,
        )
        if values["language_content.word_count"] > 0
        else None
    )

    evaluable = [
        "basic_acoustic.duration_sec",
        "basic_acoustic.sample_rate_hz",
        "basic_acoustic.channels",
        "speaker.speaker_count",
        "speaker.speaker_present",
        "speaker.asr_transcript",
        "speaker.multi_speaker",
        "speaker.speaker_change_count",
        "speaker.speaker_change",
        "speaker.overlap_ratio",
        "speaker.speaker_overlap",
        "language_content.language",
        "language_content.word_count",
        "language_content.punctuation.punctuation_count",
        "language_content.punctuation.has_terminal_punctuation",
        "language_content.repetition.has_repetition",
        "language_content.repetition.repetition_count",
        "language_content.filler",
        "deterministic_derived.speaker_change_rate_per_min",
    ]
    if values["deterministic_derived.filler_ratio"] is not None:
        evaluable.append("deterministic_derived.filler_ratio")
    gt_mask = {path: path in evaluable for path in fields}
    direct = [
        "basic_acoustic.duration_sec",
        "basic_acoustic.sample_rate_hz",
        "basic_acoustic.channels",
        "speaker.asr_transcript",
        "language_content.language",
    ]
    derived = [path for path in evaluable if path not in direct]
    strength_a, source_a, rule_a = _provenance(
        direct[:3], "system_input_audio_header", "D0", "audio-header-v1"
    )
    strength_b, source_b, rule_b = _provenance(
        direct[3:4], "AMI_reference_transcript", "D0", "AMI-native"
    )
    strength_c, source_c, rule_c = _provenance(
        direct[4:], "AMI_corpus_metadata", "D0", "language-map-v0.1"
    )
    speaker_paths = [path for path in derived if path.startswith("speaker.")]
    lexical_paths = [path for path in derived if path.startswith("language_content.")]
    strength_d, source_d, rule_d = _provenance(
        speaker_paths, "AMI_words_and_segments_release", "D1", "speaker-gt-v0.1"
    )
    strength_e, source_e, rule_e = _provenance(
        lexical_paths, "AMI_reference_transcript", "D1", "text-rules-v0.1"
    )
    strength_f, source_f, rule_f = _provenance(
        ["deterministic_derived.speaker_change_rate_per_min"],
        "AMI_words_and_segments_release+system_input_audio_header",
        "D1",
        "speaker-change-rate-v0.1",
    )
    filler_ratio_paths = (
        ["deterministic_derived.filler_ratio"]
        if "deterministic_derived.filler_ratio" in evaluable
        else []
    )
    strength_g, source_g, rule_g = _provenance(
        filler_ratio_paths,
        "AMI_reference_transcript",
        "D1",
        "text-rules-v0.1",
    )
    strength = {}
    source_by_field = {}
    rule_by_field = {}
    for target, parts in (
        (strength, (strength_a, strength_b, strength_c, strength_d, strength_e, strength_f, strength_g)),
        (source_by_field, (source_a, source_b, source_c, source_d, source_e, source_f, source_g)),
        (rule_by_field, (rule_a, rule_b, rule_c, rule_d, rule_e, rule_f, rule_g)),
    ):
        for part in parts:
            target.update(part)

    meeting_id = native.get("audio_id")
    if not isinstance(meeting_id, str) or not meeting_id:
        raise ValueError("%s has no native audio_id" % sample_id)
    split = "legacy_200" if meeting_id in ("EN2001a", "EN2001b") else "diverse_100"
    count = values["speaker.speaker_change_count"]
    duration = values["basic_acoustic.duration_sec"]
    filler = values["language_content.filler"]
    word_count = values["language_content.word_count"]
    contract_reference = {
        "speaker.speaker_present": values["speaker.speaker_count"] > 0,
        "deterministic_derived.speaker_change_rate_per_min": round(
            count * 60.0 / duration, 6
        ),
        "deterministic_derived.filler_ratio": (
            round(float(filler) / float(word_count), 6) if word_count > 0 else None
        ),
    }
    return {
        "schema_version": "structural-benchmark-sample-v0.1",
        "field_universe_version": universe["universe_version"],
        "sample_id": sample_id,
        "dataset": "AMI",
        "group_id": meeting_id,
        "split": split,
        "input_mode": "shared_audio",
        "system_input_audio": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "duration_sec": round(header["duration_sec"], 9),
            "sample_rate_hz": header["sample_rate_hz"],
            "channels": header["channels"],
            "frames": header["frames"],
        },
        "ground_truth": {
            "values": values,
            "gt_mask": gt_mask,
            "evaluable_fields": sorted(evaluable),
            "label_mask_by_field": {
                "sound_field_scene.speech_music_events": {
                    "speech": True,
                    "singing": False,
                    "music": False,
                }
            },
            "evaluable_labels_by_field": {
                "sound_field_scene.speech_music_events": ["speech"]
            },
            "weak_positive_labels_by_field": {},
            "strength_by_field": strength,
            "source_by_field": source_by_field,
            "rule_version_by_field": rule_by_field,
        },
        "contract_expectations": {
            "common_reference": contract_reference,
            "rules": [item["name"] for item in universe["contract_rules"]],
        },
        "annotation": {
            "status": "partial",
            "source": "AMI release manifest/annotation",
            "rule_version": "gt-rules-v0.1",
        },
    }


def validate_built_records(records, universe, fields):
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("built ground truth has duplicate sample IDs")
    expected_total = int(universe["benchmark"]["expected_sample_count"])
    if len(records) != expected_total:
        raise ValueError("expected %d AMI samples, built %d" % (expected_total, len(records)))
    split_counts = {}
    for row in records:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        ground_truth = row["ground_truth"]
        if set(ground_truth["values"]) != set(fields):
            raise ValueError("%s has an incomplete field universe" % row["sample_id"])
        expected_evaluable = sorted(
            path for path, enabled in ground_truth["gt_mask"].items() if enabled
        )
        if expected_evaluable != ground_truth["evaluable_fields"]:
            raise ValueError("%s gt_mask/evaluable_fields mismatch" % row["sample_id"])
        for path in expected_evaluable:
            if ground_truth["values"][path] is None:
                raise ValueError("%s has null evaluable field %s" % (row["sample_id"], path))
            for provenance_key in (
                "strength_by_field",
                "source_by_field",
                "rule_version_by_field",
            ):
                if path not in ground_truth[provenance_key]:
                    raise ValueError(
                        "%s field %s lacks %s"
                        % (row["sample_id"], path, provenance_key)
                    )
    expected_splits = universe["benchmark"]["splits"]
    if split_counts != expected_splits:
        raise ValueError(
            "AMI split counts differ: expected %s, built %s"
            % (expected_splits, split_counts)
        )
    return split_counts


def build(manifest_path, universe_path, output_path, language="en"):
    manifest_path = Path(manifest_path).resolve()
    universe = load_json(universe_path)
    fields = field_specs(universe)
    source_rows = load_jsonl(manifest_path)
    records = [
        build_record(row, manifest_path, universe, fields, language)
        for row in source_rows
    ]
    split_counts = validate_built_records(records, universe, fields)
    write_jsonl(output_path, records)
    summary = {
        "schema_version": "structural-benchmark-build-summary-v0.1",
        "field_universe_version": universe["universe_version"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "output": str(Path(output_path).resolve()),
        "sample_count": len(records),
        "split_counts": split_counts,
        "evaluable_field_count": len(records[0]["ground_truth"]["evaluable_fields"]),
        "partial_label_fields": ["sound_field_scene.speech_music_events"],
    }
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--field-universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    summary = build(args.manifest, args.field_universe, args.output, args.language)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
