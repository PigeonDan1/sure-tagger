import os
import re
import xml.etree.ElementTree as ET

from sure_tagger.schemas import make_error
from sure_tagger.text.normalize import join_tokens, normalize_transcript


NITE_NS = "{http://nite.sourceforge.net/}"


def local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def nite_id(elem):
    return elem.attrib.get(NITE_NS + "id") or elem.attrib.get("nite:id") or elem.attrib.get("id")


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_meeting_speaker(filename, suffix):
    base = os.path.basename(filename)
    pattern = r"^(.+)\.([A-Z])\.%s$" % re.escape(suffix)
    m = re.match(pattern, base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def meeting_type(meeting_id):
    if re.match(r"^(ES|IS|TS)", meeting_id or ""):
        return "scenario"
    if re.match(r"^(EN|IN|IB)", meeting_id or ""):
        return "non-scenario"
    return "unknown"


def parse_words_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    items = []
    by_id = {}
    for order, elem in enumerate(list(root)):
        tag = local_name(elem.tag)
        item_id = nite_id(elem)
        if not item_id:
            continue
        attrs = dict(elem.attrib)
        item = {
            "id": item_id,
            "xml_tag": tag,
            "text": elem.text or "",
            "start": to_float(elem.attrib.get("starttime")),
            "end": to_float(elem.attrib.get("endtime")),
            "attrs": attrs,
            "order": order,
        }
        if tag == "w":
            item["kind"] = "punctuation" if elem.attrib.get("punc") == "true" else "word"
        elif tag in ("vocalsound", "nonvocalsound"):
            item["kind"] = tag
            item["event_type"] = elem.attrib.get("type", "")
        else:
            item["kind"] = "event"
        items.append(item)
        by_id[item_id] = item
    return {"path": path, "items": items, "by_id": by_id}


def parse_segments_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    segments = []
    for elem in list(root):
        if local_name(elem.tag) != "segment":
            continue
        seg_id = nite_id(elem)
        children = []
        for child in list(elem):
            if local_name(child.tag) == "child":
                href = child.attrib.get("href")
                if href:
                    children.append(href)
        segments.append({
            "id": seg_id,
            "start": to_float(elem.attrib.get("transcriber_start") or elem.attrib.get("starttime")),
            "end": to_float(elem.attrib.get("transcriber_end") or elem.attrib.get("endtime")),
            "channel": elem.attrib.get("channel", ""),
            "children": children,
        })
    return segments


def href_file(href):
    if "#" not in href:
        return None
    return href.split("#", 1)[0]


def href_ids(href):
    if "#" not in href:
        return []
    frag = href.split("#", 1)[1]
    ids = re.findall(r"id\(([^)]+)\)", frag)
    return ids


def resolve_href(href, words_data):
    ids = href_ids(href)
    if not ids:
        raise ValueError("No ids in href: %s" % href)
    by_id = words_data["by_id"]
    items = words_data["items"]
    if len(ids) == 1:
        item = by_id.get(ids[0])
        if item is None:
            raise KeyError(ids[0])
        return [item]
    start = by_id.get(ids[0])
    end = by_id.get(ids[1])
    if start is None:
        raise KeyError(ids[0])
    if end is None:
        raise KeyError(ids[1])
    lo = min(start["order"], end["order"])
    hi = max(start["order"], end["order"])
    return items[lo:hi + 1]


def choose_audio_path(root, meeting_id):
    candidates = [
        os.path.join(root, "amicorpus-Mix-Headset", meeting_id, "audio", "%s.Mix-Headset.wav" % meeting_id),
        os.path.join(root, "amicorpus", "beamformed", meeting_id, "%s_MDM8.wav" % meeting_id),
    ]
    audio_dir = os.path.join(root, "amicorpus", meeting_id, "audio")
    if os.path.isdir(audio_dir):
        for name in sorted(os.listdir(audio_dir)):
            if name.endswith(".wav") and ".Headset-" in name:
                candidates.append(os.path.join(audio_dir, name))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def recover_text_and_events(items):
    tokens = []
    events = []
    for item in items:
        if item["kind"] in ("word", "punctuation"):
            tokens.append(item.get("text", ""))
        else:
            events.append({
                "id": item["id"],
                "kind": item["kind"],
                "event_type": item.get("event_type", ""),
                "start_sec": item.get("start"),
                "end_sec": item.get("end"),
            })
    return normalize_transcript(join_tokens(tokens)), events


def fallback_times(segment, items):
    start = segment.get("start")
    end = segment.get("end")
    timed = [i for i in items if i.get("start") is not None and i.get("end") is not None]
    if start is None and timed:
        start = timed[0]["start"]
    if end is None and timed:
        end = timed[-1]["end"]
    return start, end


def build_manifest(root, output_writer, bad_writer=None, limit=None, meetings=None):
    annotations = os.path.join(root, "amicorpus", "annotations")
    words_dir = os.path.join(annotations, "words")
    segments_dir = os.path.join(annotations, "segments")
    segment_files = sorted([
        os.path.join(segments_dir, name)
        for name in os.listdir(segments_dir)
        if name.endswith(".segments.xml")
    ])
    words_cache = {}
    stats = {"records": 0, "bad_records": 0, "segment_files": 0, "warnings": 0}
    meetings_set = set(meetings or [])

    for seg_path in segment_files:
        meeting_id, speaker_id = parse_meeting_speaker(seg_path, "segments.xml")
        if not meeting_id or not speaker_id:
            continue
        if meetings_set and meeting_id not in meetings_set:
            continue
        stats["segment_files"] += 1
        word_path = os.path.join(words_dir, "%s.%s.words.xml" % (meeting_id, speaker_id))
        if word_path not in words_cache:
            try:
                words_cache[word_path] = parse_words_file(word_path)
            except Exception as exc:
                stats["bad_records"] += 1
                if bad_writer:
                    bad_writer.write(make_error("AMI:%s:%s" % (meeting_id, speaker_id), "build_manifest", "words_parse_error", exc, word_path))
                continue
        words_data = words_cache[word_path]
        try:
            segments = parse_segments_file(seg_path)
        except Exception as exc:
            stats["bad_records"] += 1
            if bad_writer:
                bad_writer.write(make_error("AMI:%s:%s" % (meeting_id, speaker_id), "build_manifest", "segments_parse_error", exc, seg_path))
            continue

        for seg in segments:
            sample_id = "AMI:%s:%s:%s" % (meeting_id, speaker_id, seg["id"])
            try:
                resolved = []
                raw_hrefs = []
                for href in seg["children"]:
                    raw_hrefs.append(href)
                    ref_file = href_file(href)
                    ref_words_data = words_data
                    if ref_file and ref_file != os.path.basename(word_path):
                        ref_path = os.path.join(words_dir, ref_file)
                        if ref_path not in words_cache:
                            words_cache[ref_path] = parse_words_file(ref_path)
                        ref_words_data = words_cache[ref_path]
                    resolved.extend(resolve_href(href, ref_words_data))
                transcript, events = recover_text_and_events(resolved)
                start_sec, end_sec = fallback_times(seg, resolved)
                if start_sec is None or end_sec is None:
                    stats["warnings"] += 1
                record = {
                    "corpus": {
                        "dataset_name": "AMI",
                        "source_urls": {
                            "article": [],
                            "github": [],
                            "huggingface": [],
                            "dataset_card": [],
                        },
                        "native_metadata": {
                            "annotation_release": "AMI Manual Annotations 1.7",
                        },
                    },
                    "sample": {
                        "sample_id": sample_id,
                        "audio": {
                            "path": choose_audio_path(root, meeting_id),
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                        },
                        "text": {
                            "transcript": transcript,
                            "normalized_transcript": normalize_transcript(transcript),
                        },
                        "native_metadata": {
                            "meeting_id": meeting_id,
                            "speaker_id": speaker_id,
                            "segment_id": seg["id"],
                            "meeting_type": meeting_type(meeting_id),
                            "source_words_file": os.path.relpath(word_path, os.path.join(root, "amicorpus")),
                            "source_segments_file": os.path.relpath(seg_path, os.path.join(root, "amicorpus")),
                            "raw_hrefs": raw_hrefs,
                            "nonword_events": events,
                        },
                        "provenance": {
                            "source_path": seg_path,
                            "source_split": "",
                        },
                    },
                }
                output_writer.write(record)
                stats["records"] += 1
                if limit and stats["records"] >= int(limit):
                    return stats
            except Exception as exc:
                stats["bad_records"] += 1
                if bad_writer:
                    bad_writer.write(make_error(sample_id, "build_manifest", "segment_resolve_error", exc, seg_path))
    return stats
