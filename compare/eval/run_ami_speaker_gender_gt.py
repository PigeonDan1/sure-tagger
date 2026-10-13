#!/usr/bin/env python3
"""Build per-utterance AMI speaker gender GT from official metadata and clips.

For every active speaker, this script recovers the utterance's absolute meeting
interval, cuts the matching close-talking headset, and asks Qwen3.5-Omni-Plus
for an auditable perceived-gender label. Official AMI participant metadata is
preferred for the final GT when available; the Qwen label is the fallback.
"""

from __future__ import print_function

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
import xml.etree.ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
SHARE_ROOT = COMPARE_ROOT.parent
DEFAULT_MANIFEST = COMPARE_ROOT / "dataset" / "benchmark" / "t1-ami" / "manifest.jsonl"
DEFAULT_ANNOTATIONS_ROOT = SHARE_ROOT / "tagger" / "dataset" / "all"
DEFAULT_HEADSET_ROOT = COMPARE_ROOT / "dataset" / "Ami-seperate-headset"
DEFAULT_OUTPUT_DIR = (
    COMPARE_ROOT / "dataset" / "benchmark" / "t1-ami" / "speaker_gender_gt"
)
DEFAULT_API_FILES = (
    COMPARE_ROOT / "api" / "qwen-api5.txt",
    COMPARE_ROOT / "api" / "qwen-api6.txt",
    COMPARE_ROOT / "api" / "qwen-omni-captioner-api.txt",
)
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
KEY_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._-]{20,}")
GENDER_PROMPT = """You are labeling the perceived gender of the target speaker in a close-talking headset recording.

The recording contains one or more concatenated excerpts where the target headset wearer is annotated as speaking. It can contain faint distant crosstalk from other people. Judge only the clearest, loudest, on-mic speaker. Based on the voice, choose the more likely label: male or female. You must choose one even when the speech is brief.

Return exactly one JSON object and nothing else:
{"gender":"male"}
or
{"gender":"female"}"""
USER_PROMPT = "Determine the gender of the clearest on-mic speaker in this headset clip."
ORIGIN_TOLERANCE_SEC = 0.0001
MIN_NON_OVERLAP_SEC = 1.0
CLIP_EXTRACTION_VERSION = "speaker-speech-v1"
NITE_ID = "{http://nite.sourceforge.net/}id"


def utc_now():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, value):
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def write_jsonl(path, values):
    atomic_write_text(
        path,
        "".join(canonical_json(value) + "\n" for value in values),
    )


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise ValueError("%s:%d: %s" % (path, line_number, error))
            if not isinstance(value, dict):
                raise ValueError("%s:%d must contain an object" % (path, line_number))
            rows.append(value)
    return rows


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def read_api_key(path):
    raw = Path(path).read_text(encoding="utf-8")
    match = KEY_RE.search(raw)
    if not match:
        raise ValueError("API file does not contain a Bailian key: %s" % path)
    return match.group(0)


def redact(value, secret):
    return str(value).replace(secret, "<redacted>") if secret else str(value)


def parse_gender_response(raw_text):
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    decoded = None
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            decoded = value
            break
    if not isinstance(decoded, dict) or set(decoded) != {"gender"}:
        raise ValueError("response must be a JSON object with only the gender key")
    gender = decoded["gender"]
    if not isinstance(gender, str) or gender.strip().lower() not in ("male", "female"):
        raise ValueError("gender must be male or female")
    return gender.strip().lower()


def _xml_id(element):
    return element.attrib.get(NITE_ID) or element.attrib.get("nite:id")


def load_headset_metadata(headset_root):
    resource_root = Path(headset_root) / "annotations" / "corpusResources"
    meetings_path = resource_root / "meetings.xml"
    participants_path = resource_root / "participants.xml"
    if not meetings_path.is_file() or not participants_path.is_file():
        raise FileNotFoundError("AMI corpusResources metadata is incomplete")

    participant_sex = {}
    for element in ET.parse(str(participants_path)).getroot().iter("participant"):
        participant_id = _xml_id(element)
        sex = element.attrib.get("sex", "").strip().upper()
        if participant_id and sex in ("M", "F"):
            participant_sex[participant_id] = "male" if sex == "M" else "female"

    meetings = {}
    for meeting in ET.parse(str(meetings_path)).getroot().iter("meeting"):
        meeting_id = meeting.attrib.get("observation")
        if not meeting_id:
            continue
        speakers = {}
        for speaker in meeting.findall("speaker"):
            source_speaker = speaker.attrib.get("nxt_agent")
            channel = speaker.attrib.get("channel")
            participant_id = speaker.attrib.get("global_name")
            if not source_speaker or channel is None or not participant_id:
                continue
            speakers[source_speaker] = {
                "channel": int(channel),
                "participant_id": participant_id,
                "official_metadata_gender": participant_sex.get(participant_id),
            }
        meetings[meeting_id] = speakers
    return meetings


def speaker_order(utterances):
    first_start = {}
    for utterance in utterances:
        speaker = utterance["speaker"]
        start = float(utterance["start"])
        first_start[speaker] = min(start, first_start.get(speaker, start))
    return sorted(first_start, key=lambda speaker: (first_start[speaker], speaker))


def merge_intervals(intervals):
    merged = []
    for interval in sorted(intervals, key=lambda value: (value["start_sec"], value["end_sec"])):
        start = float(interval["start_sec"])
        end = float(interval["end_sec"])
        if end <= start:
            continue
        if merged and start <= merged[-1]["end_sec"]:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], end)
        else:
            merged.append({"start_sec": start, "end_sec": end})
    return merged


def subtract_intervals(intervals, blockers):
    remaining = []
    blockers = merge_intervals(blockers)
    for interval in merge_intervals(intervals):
        start = interval["start_sec"]
        end = interval["end_sec"]
        cursor = start
        for blocker in blockers:
            blocker_start = blocker["start_sec"]
            blocker_end = blocker["end_sec"]
            if blocker_end <= cursor:
                continue
            if blocker_start >= end:
                break
            if blocker_start > cursor:
                remaining.append(
                    {"start_sec": cursor, "end_sec": min(blocker_start, end)}
                )
            cursor = max(cursor, blocker_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append({"start_sec": cursor, "end_sec": end})
    return remaining


def intervals_duration(intervals):
    return sum(item["end_sec"] - item["start_sec"] for item in intervals)


def recover_origin(local_utterances, absolute_by_id):
    offsets = []
    for local in local_utterances:
        utterance_id = local.get("utt_id")
        absolute = absolute_by_id.get(utterance_id)
        if absolute is None:
            raise ValueError("source annotation is missing utt_id %s" % utterance_id)
        if absolute.get("speaker") != local.get("speaker"):
            raise ValueError("speaker differs for utt_id %s" % utterance_id)
        offsets.extend(
            (
                float(absolute["start"]) - float(local["start"]),
                float(absolute["end"]) - float(local["end"]),
            )
        )
    if not offsets:
        raise ValueError("utterance has no source offsets")
    low = min(offsets)
    high = max(offsets)
    if high - low > ORIGIN_TOLERANCE_SEC:
        raise ValueError("inconsistent clip origin offsets: %.6f..%.6f" % (low, high))
    return round(sum(offsets) / len(offsets), 6)


def wave_info(path):
    with wave.open(str(path), "rb") as stream:
        return {
            "channels": stream.getnchannels(),
            "sample_width_bytes": stream.getsampwidth(),
            "sample_rate_hz": stream.getframerate(),
            "frame_count": stream.getnframes(),
            "compression": stream.getcomptype(),
        }


def prepare_tasks(manifest_path, annotations_root, headset_root, output_dir, limit=None):
    manifest_rows = read_jsonl(manifest_path)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    headset_metadata = load_headset_metadata(headset_root)
    annotation_cache = {}
    source_info_cache = {}
    tasks = []
    sample_rows = []
    seen_samples = set()

    for manifest_line, row in enumerate(manifest_rows, 1):
        sample = row.get("sample", {})
        sample_id = sample.get("sample_id")
        native = sample.get("native_metadata", {})
        meeting_id = native.get("audio_id")
        utterances = native.get("utterances")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("manifest line %d has no sample_id" % manifest_line)
        if sample_id in seen_samples:
            raise ValueError("duplicate sample_id: %s" % sample_id)
        seen_samples.add(sample_id)
        if not isinstance(meeting_id, str) or not isinstance(utterances, list) or not utterances:
            raise ValueError("%s has invalid AMI native metadata" % sample_id)
        if meeting_id not in headset_metadata:
            raise ValueError("meeting metadata not found: %s" % meeting_id)

        if meeting_id not in annotation_cache:
            annotation_path = Path(annotations_root) / (meeting_id + ".jsonl")
            absolute_rows = read_jsonl(annotation_path)
            annotation_cache[meeting_id] = {
                item["utt_id"]: item for item in absolute_rows
            }
        absolute_by_id = annotation_cache[meeting_id]
        origin_sec = recover_origin(utterances, absolute_by_id)
        duration_sec = float(native.get("end")) - float(native.get("start", 0.0))
        if origin_sec < 0 or duration_sec <= 0:
            raise ValueError("%s has invalid recovered interval" % sample_id)

        ordered_speakers = speaker_order(utterances)
        speaker_tasks = []
        for speaker_index, source_speaker in enumerate(ordered_speakers, 1):
            mapping = headset_metadata[meeting_id].get(source_speaker)
            if mapping is None:
                raise ValueError("%s has no headset mapping for %s" % (meeting_id, source_speaker))
            channel = mapping["channel"]
            source_audio = (
                Path(headset_root)
                / meeting_id
                / "audio"
                / ("%s.Headset-%d.wav" % (meeting_id, channel))
            )
            if not source_audio.is_file():
                raise FileNotFoundError(str(source_audio))
            if source_audio not in source_info_cache:
                source_info_cache[source_audio] = wave_info(source_audio)
            info = source_info_cache[source_audio]
            if info["compression"] != "NONE" or info["channels"] not in (1, 2):
                raise ValueError(
                    "headset source must be uncompressed mono/stereo PCM: %s" % source_audio
                )
            end_frame = round((origin_sec + duration_sec) * info["sample_rate_hz"])
            if end_frame > info["frame_count"]:
                raise ValueError("%s extends beyond %s" % (sample_id, source_audio))

            speaker_id = "speaker_%d" % speaker_index
            first_start = min(
                float(item["start"])
                for item in utterances
                if item["speaker"] == source_speaker
            )
            intervals = [
                {"start_sec": float(item["start"]), "end_sec": float(item["end"])}
                for item in utterances
                if item["speaker"] == source_speaker
            ]
            all_speech_intervals = merge_intervals(intervals)
            other_speech_intervals = [
                {"start_sec": float(item["start"]), "end_sec": float(item["end"])}
                for item in utterances
                if item["speaker"] != source_speaker
            ]
            non_overlap_intervals = subtract_intervals(
                all_speech_intervals, other_speech_intervals
            )
            non_overlap_duration_sec = intervals_duration(non_overlap_intervals)
            if non_overlap_duration_sec >= MIN_NON_OVERLAP_SEC:
                selected_intervals = non_overlap_intervals
                clip_selection_rule = "non_overlap_speech"
            else:
                selected_intervals = all_speech_intervals
                clip_selection_rule = "all_speech_non_overlap_under_1s"
            task_key = sample_id + "__" + speaker_id
            clip_path = Path(output_dir) / "speaker_speech_clips" / (task_key + ".wav")
            task = {
                "task_key": task_key,
                "sample_id": sample_id,
                "manifest_line": manifest_line,
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "source_speaker": source_speaker,
                "first_speech_start_sec": first_start,
                "speech_intervals": intervals,
                "non_overlap_speech_intervals": non_overlap_intervals,
                "selected_speech_intervals": selected_intervals,
                "non_overlap_speech_duration_sec": non_overlap_duration_sec,
                "selected_speech_duration_sec": intervals_duration(selected_intervals),
                "clip_selection_rule": clip_selection_rule,
                "headset_channel": channel,
                "participant_id": mapping["participant_id"],
                "official_metadata_gender": mapping["official_metadata_gender"],
                "source_audio": source_audio,
                "clip_path": clip_path,
                "source_start_sec": origin_sec,
                "duration_sec": duration_sec,
            }
            tasks.append(task)
            speaker_tasks.append(task)
        sample_rows.append((sample_id, speaker_tasks))
    return tasks, sample_rows


def ensure_clip(task, overwrite=False):
    clip_path = task["clip_path"]
    source_path = task["source_audio"]
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        frame_ranges = []
        for interval in task["selected_speech_intervals"]:
            start_frame = int(
                round(
                    (task["source_start_sec"] + interval["start_sec"])
                    * params.framerate
                )
            )
            end_frame = int(
                round(
                    (task["source_start_sec"] + interval["end_sec"])
                    * params.framerate
                )
            )
            if start_frame < 0 or end_frame <= start_frame or end_frame > source.getnframes():
                raise ValueError("invalid frame range for %s" % task["task_key"])
            frame_ranges.append((start_frame, end_frame))
        frame_count = sum(end - start for start, end in frame_ranges)
        if not frame_ranges or frame_count <= 0:
            raise ValueError("no selected speech for %s" % task["task_key"])

        if clip_path.is_file() and not overwrite:
            info = wave_info(clip_path)
            if (
                info["channels"] == params.nchannels
                and info["sample_width_bytes"] == params.sampwidth
                and info["sample_rate_hz"] == params.framerate
                and info["frame_count"] == frame_count
                and info["compression"] == params.comptype
            ):
                return clip_path

        clip_path.parent.mkdir(parents=True, exist_ok=True)
        frame_parts = []
        for start_frame, end_frame in frame_ranges:
            source.setpos(start_frame)
            frame_parts.append(source.readframes(end_frame - start_frame))
        frames = b"".join(frame_parts)
        expected_bytes = frame_count * params.nchannels * params.sampwidth
        if len(frames) != expected_bytes:
            raise IOError("short headset read for %s" % task["task_key"])
        descriptor, temporary = tempfile.mkstemp(
            prefix=".%s." % clip_path.name,
            suffix=".tmp.wav",
            dir=str(clip_path.parent),
        )
        os.close(descriptor)
        try:
            with wave.open(temporary, "wb") as sink:
                sink.setparams(params)
                sink.writeframes(frames)
            os.replace(temporary, str(clip_path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return clip_path


def ensure_upload_audio(task, clip_path, overwrite=False):
    """Make a compact, speech-preserving upload copy while retaining the GT WAV."""
    upload_path = clip_path.parent.parent / "speaker_speech_uploads" / (task["task_key"] + ".mp3")
    if (
        upload_path.is_file()
        and not overwrite
        and upload_path.stat().st_size > 1000
        and upload_path.stat().st_mtime_ns >= clip_path.stat().st_mtime_ns
    ):
        return upload_path
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % upload_path.name,
        suffix=".tmp.mp3",
        dir=str(upload_path.parent),
    )
    os.close(descriptor)
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            "-f",
            "mp3",
            temporary,
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("ffmpeg upload conversion failed: %s" % completed.stderr[-1000:])
        os.replace(temporary, str(upload_path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return upload_path


def _delta_text(delta):
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


def request_gender(base_url, api_key, model, audio_bytes, timeout, max_tokens):
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": GENDER_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/mpeg;base64," + encoded,
                            "format": "mp3",
                        },
                    },
                ],
            },
        ],
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "ami-speaker-gender-gt/1.0",
        },
        method="POST",
    )
    parts = []
    usage = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError("HTTP %s: %s" % (error.code, redact(detail, api_key)))
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
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
                parts.append(_delta_text(choices[0].get("delta")))
    raw_text = "".join(parts).strip()
    if not raw_text:
        raise ValueError("Qwen returned empty text")
    return raw_text, usage


def process_task(task, api_key, api_file, args, fingerprint, overwrite_clip=False):
    started = time.monotonic()
    clip_path = ensure_clip(task, overwrite=overwrite_clip)
    clip_hash = sha256_file(clip_path)
    upload_path = ensure_upload_audio(task, clip_path, overwrite=overwrite_clip)
    upload_hash = sha256_file(upload_path)
    upload_bytes = upload_path.read_bytes()
    errors = []
    for attempt in range(1, args.retries + 2):
        try:
            raw_text, usage = request_gender(
                args.base_url,
                api_key,
                args.model,
                upload_bytes,
                args.timeout,
                args.max_tokens,
            )
            gender = parse_gender_response(raw_text)
            return {
                "task_key": task["task_key"],
                "sample_id": task["sample_id"],
                "speaker_id": task["speaker_id"],
                "source_speaker": task["source_speaker"],
                "status": "success",
                "gender": gender,
                "raw_text": raw_text,
                "usage": usage,
                "attempts": attempt,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "model": args.model,
                "base_url": args.base_url,
                "api_file": str(api_file),
                "prompt_sha256": sha256_text(GENDER_PROMPT),
                "config_fingerprint": fingerprint,
                "headset_clip": str(clip_path),
                "headset_clip_sha256": clip_hash,
                "clip_extraction_version": CLIP_EXTRACTION_VERSION,
                "clip_selection_rule": task["clip_selection_rule"],
                "non_overlap_speech_intervals": task["non_overlap_speech_intervals"],
                "selected_speech_intervals": task["selected_speech_intervals"],
                "non_overlap_speech_duration_sec": round(
                    task["non_overlap_speech_duration_sec"], 6
                ),
                "selected_speech_duration_sec": round(
                    task["selected_speech_duration_sec"], 6
                ),
                "api_audio": str(upload_path),
                "api_audio_format": "mp3_16khz_mono_64kbps",
                "api_audio_sha256": upload_hash,
                "created_at": utc_now(),
            }
        except Exception as error:  # network and output-format errors are retryable
            errors.append(redact(error, api_key))
            if attempt <= args.retries:
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))
    return {
        "task_key": task["task_key"],
        "sample_id": task["sample_id"],
        "speaker_id": task["speaker_id"],
        "source_speaker": task["source_speaker"],
        "status": "failed",
        "errors": errors,
        "attempts": args.retries + 1,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "model": args.model,
        "base_url": args.base_url,
        "api_file": str(api_file),
        "prompt_sha256": sha256_text(GENDER_PROMPT),
        "config_fingerprint": fingerprint,
        "headset_clip": str(clip_path),
        "headset_clip_sha256": clip_hash,
        "clip_extraction_version": CLIP_EXTRACTION_VERSION,
        "clip_selection_rule": task["clip_selection_rule"],
        "non_overlap_speech_intervals": task["non_overlap_speech_intervals"],
        "selected_speech_intervals": task["selected_speech_intervals"],
        "non_overlap_speech_duration_sec": round(
            task["non_overlap_speech_duration_sec"], 6
        ),
        "selected_speech_duration_sec": round(task["selected_speech_duration_sec"], 6),
        "api_audio": str(upload_path),
        "api_audio_format": "mp3_16khz_mono_64kbps",
        "api_audio_sha256": upload_hash,
        "created_at": utc_now(),
    }


def checkpoint_path(output_dir, task_key):
    return Path(output_dir) / "items" / (task_key + ".json")


def reusable_checkpoint(output_dir, task, fingerprint):
    record = read_json(checkpoint_path(output_dir, task["task_key"]))
    if not isinstance(record, dict):
        return None
    if (
        record.get("status") == "success"
        and record.get("task_key") == task["task_key"]
        and record.get("config_fingerprint") == fingerprint
        and record.get("gender") in ("male", "female")
    ):
        clip_path = Path(record.get("headset_clip", ""))
        if clip_path.is_file() and sha256_file(clip_path) == record.get("headset_clip_sha256"):
            return record
    return None


def public_task_details(task, result, output_dir):
    output_dir = Path(output_dir)
    result_clip = Path(result.get("headset_clip", task["clip_path"]))
    try:
        clip_relative = result_clip.relative_to(output_dir).as_posix()
    except ValueError:
        clip_relative = str(result_clip)
    item_relative = checkpoint_path(output_dir, task["task_key"]).relative_to(output_dir).as_posix()
    official = task["official_metadata_gender"]
    qwen_gender = result.get("gender")
    gender = official or qwen_gender
    return {
        "speaker_id": task["speaker_id"],
        "gender": gender,
        "gender_source": (
            "ami_official_metadata" if official else result.get("model", DEFAULT_MODEL)
        ),
        "qwen_gender": qwen_gender,
        "first_speech_start_sec": task["first_speech_start_sec"],
        "source_speaker": task["source_speaker"],
        "meeting_id": task["meeting_id"],
        "headset_channel": task["headset_channel"],
        "headset_clip": clip_relative,
        "clip_extraction_version": result.get(
            "clip_extraction_version", "full-sample-interval-v1"
        ),
        "clip_selection_rule": result.get(
            "clip_selection_rule", "full_sample_interval"
        ),
        "non_overlap_speech_intervals": result.get(
            "non_overlap_speech_intervals"
        ),
        "selected_speech_intervals": result.get("selected_speech_intervals"),
        "non_overlap_speech_duration_sec": result.get(
            "non_overlap_speech_duration_sec"
        ),
        "selected_speech_duration_sec": result.get("selected_speech_duration_sec"),
        "qwen_item": item_relative,
        "qwen_status": result.get("status"),
        "speech_intervals": task["speech_intervals"],
        "source_interval": {
            "start_sec": task["source_start_sec"],
            "end_sec": round(task["source_start_sec"] + task["duration_sec"], 6),
        },
        "official_metadata_gender": official,
        "qwen_matches_official_metadata": (
            qwen_gender == official
            if qwen_gender in ("male", "female") and official
            else None
        ),
    }


def finalize(output_dir, sample_rows, results, config, started_at):
    output_dir = Path(output_dir)
    gt_rows = []
    detail_rows = []
    qwen_rows = []
    mismatch_rows = []
    failed = []
    male = 0
    female = 0
    fallback_results = []
    fallback_selection_counts = {}
    for sample_id, tasks in sample_rows:
        speakers = []
        gt_speakers = []
        for task in tasks:
            result = results[task["task_key"]]
            detail = public_task_details(task, result, output_dir)
            speakers.append(detail)
            if result.get("status") == "success":
                gender = detail["gender"]
                gt_speakers.append(
                    {"speaker_id": task["speaker_id"], "gender": gender}
                )
                male += gender == "male"
                female += gender == "female"
                if detail["qwen_matches_official_metadata"] is False:
                    mismatch_rows.append(
                        {
                            "sample_id": sample_id,
                            "speaker_id": task["speaker_id"],
                            "qwen_gender": detail["qwen_gender"],
                            "official_metadata_gender": task["official_metadata_gender"],
                            "source_speaker": task["source_speaker"],
                            "participant_id": task["participant_id"],
                            "headset_clip": detail["headset_clip"],
                        }
                    )
            else:
                failed.append(task["task_key"])
            qwen_rows.append(result)
            if task["official_metadata_gender"] not in ("male", "female"):
                fallback_results.append(result)
                rule = task["clip_selection_rule"]
                fallback_selection_counts[rule] = (
                    fallback_selection_counts.get(rule, 0) + 1
                )
        gt_rows.append({"sample_id": sample_id, "speakers": gt_speakers})
        detail_rows.append(
            {
                "sample_id": sample_id,
                "speaker_order_rule": "ascending first_speech_start_sec, then source speaker ID",
                "speakers": speakers,
            }
        )

    write_jsonl(output_dir / "speaker_gender_gt.jsonl", gt_rows)
    write_jsonl(output_dir / "speaker_gender_details.jsonl", detail_rows)
    write_jsonl(output_dir / "qwen_results.jsonl", qwen_rows)
    write_jsonl(output_dir / "qwen_official_mismatches.jsonl", mismatch_rows)
    baseline_path = output_dir / "qwen_full_interval_results.jsonl"
    fallback_relabel_changed_count = None
    if baseline_path.is_file():
        baseline_gender = {
            item["task_key"]: item.get("gender") for item in read_jsonl(baseline_path)
        }
        fallback_relabel_changed_count = sum(
            baseline_gender.get(item["task_key"]) != item.get("gender")
            for item in fallback_results
        )

    summary = {
        "status": "complete" if not failed else "incomplete",
        "sample_count": len(sample_rows),
        "speaker_clip_count": len(qwen_rows),
        "success_count": len(qwen_rows) - len(failed),
        "failed_count": len(failed),
        "failed_task_keys": failed,
        "gender_counts": {"male": male, "female": female},
        "official_metadata_covered_count": sum(
            item["official_metadata_gender"] in ("male", "female")
            for _, tasks in sample_rows
            for item in tasks
        ),
        "qwen_fallback_count": sum(
            item["official_metadata_gender"] not in ("male", "female")
            for _, tasks in sample_rows
            for item in tasks
        ),
        "official_metadata_compared_count": sum(
            item["official_metadata_gender"] in ("male", "female")
            for _, tasks in sample_rows
            for item in tasks
        ),
        "qwen_official_mismatch_count": len(mismatch_rows),
        "official_metadata_override_count": len(mismatch_rows),
        "official_metadata_mapping_rule": "order source A/B/C/D/E speakers by first speech in each sample; map meeting_id plus nxt_agent through meetings.xml global_name to participants.xml sex",
        "qwen_fallback_relabel_success_count": sum(
            item.get("status") == "success" for item in fallback_results
        ),
        "qwen_fallback_relabel_changed_count": fallback_relabel_changed_count,
        "qwen_fallback_clip_selection_counts": fallback_selection_counts,
        "qwen_fallback_gender_counts": {
            "male": sum(item.get("gender") == "male" for item in fallback_results),
            "female": sum(
                item.get("gender") == "female" for item in fallback_results
            ),
        },
        "qwen_gender_counts": {
            "male": sum(item.get("gender") == "male" for item in qwen_rows),
            "female": sum(item.get("gender") == "female" for item in qwen_rows),
        },
        "model": config["model"],
        "api_count": config["api_count"],
        "workers_per_api": config["workers_per_api"],
        "worker_count": config["worker_count"],
        "speaker_order_rule": "ascending first_speech_start_sec, then source speaker ID",
        "label_source": "AMI official participant sex metadata where available; Qwen3.5-Omni-Plus fallback otherwise",
        "official_metadata_usage": "preferred for final GT when available; Qwen labels retained only where official metadata is unavailable",
        "started_at": started_at,
        "completed_at": utc_now(),
        "outputs": {
            "ground_truth": str((output_dir / "speaker_gender_gt.jsonl").resolve()),
            "details": str((output_dir / "speaker_gender_details.jsonl").resolve()),
            "qwen_results": str((output_dir / "qwen_results.jsonl").resolve()),
            "qwen_full_interval_results": str(baseline_path.resolve()),
            "mismatches": str((output_dir / "qwen_official_mismatches.jsonl").resolve()),
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def acquire_lock(output_dir):
    lock_path = Path(output_dir) / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        old = read_json(lock_path)
        pid = old.get("pid") if isinstance(old, dict) else None
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
            except OSError:
                lock_path.unlink()
                return acquire_lock(output_dir)
        raise RuntimeError("another gender batch appears to be running: %s" % lock_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}) + "\n")
    return lock_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS_ROOT)
    parser.add_argument("--headset-root", type=Path, default=DEFAULT_HEADSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--api-file",
        action="append",
        type=Path,
        dest="api_files",
        default=None,
        help="Bailian API key file; repeat for multiple APIs",
    )
    parser.add_argument("--workers-per-api", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--missing-official-only",
        action="store_true",
        help="rerun only speakers without official AMI participant sex metadata",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.api_files is None:
        args.api_files = list(DEFAULT_API_FILES)
    if args.workers_per_api < 1 or args.timeout < 1 or args.max_tokens < 1 or args.retries < 0:
        raise ValueError("workers, timeout, and max-tokens must be positive; retries cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    args.manifest = args.manifest.resolve()
    args.annotations_root = args.annotations_root.resolve()
    args.headset_root = args.headset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    api_files = [path.resolve() for path in args.api_files]
    for required in [args.manifest] + api_files:
        if not required.is_file():
            raise FileNotFoundError(str(required))

    tasks, sample_rows = prepare_tasks(
        args.manifest,
        args.annotations_root,
        args.headset_root,
        args.output_dir,
        limit=args.limit,
    )
    fingerprint_value = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_sha256": sha256_text(GENDER_PROMPT),
        "user_prompt_sha256": sha256_text(USER_PROMPT),
        "max_tokens": args.max_tokens,
        "clip_extraction_version": CLIP_EXTRACTION_VERSION,
        "min_non_overlap_sec": MIN_NON_OVERLAP_SEC,
    }
    fingerprint = sha256_text(canonical_json(fingerprint_value))
    config = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(sample_rows),
        "speaker_clip_count": len(tasks),
        "processed_speaker_clip_count": sum(
            not args.missing_official_only
            or task["official_metadata_gender"] not in ("male", "female")
            for task in tasks
        ),
        "annotations_root": str(args.annotations_root),
        "headset_root": str(args.headset_root),
        "output_dir": str(args.output_dir),
        "api_files": [str(path) for path in api_files],
        "api_count": len(api_files),
        "workers_per_api": args.workers_per_api,
        "worker_count": len(api_files) * args.workers_per_api,
        "model": args.model,
        "base_url": args.base_url,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "api_audio_encoding": "mp3_16khz_mono_64kbps",
        "clip_extraction_version": CLIP_EXTRACTION_VERSION,
        "clip_selection_rule": "use non-overlap speech when its total duration is at least 1 second; otherwise use all target-speaker speech",
        "missing_official_only": args.missing_official_only,
        "config_fingerprint": fingerprint,
        "prompt_sha256": sha256_text(GENDER_PROMPT),
        "created_at": utc_now(),
    }
    if args.dry_run:
        print(json.dumps(dict(config, status="dry-run"), ensure_ascii=False, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_config = read_json(args.output_dir / "run_config.json")
    if (
        not args.overwrite
        and isinstance(old_config, dict)
        and (
            old_config.get("manifest_sha256") != config["manifest_sha256"]
            or old_config.get("speaker_clip_count") != config["speaker_clip_count"]
            or old_config.get("config_fingerprint") != fingerprint
        )
    ):
        raise RuntimeError("existing run config differs; use a new output directory or --overwrite")
    atomic_write_json(args.output_dir / "run_config.json", config)
    api_keys = [read_api_key(path) for path in api_files]
    lock_path = acquire_lock(args.output_dir)
    started_at = utc_now()
    selected_tasks = [
        task
        for task in tasks
        if not args.missing_official_only
        or task["official_metadata_gender"] not in ("male", "female")
    ]
    results = {}
    if args.missing_official_only:
        old_results = {
            item["task_key"]: item
            for item in read_jsonl(args.output_dir / "qwen_results.jsonl")
        }
        for task in tasks:
            if task["official_metadata_gender"] in ("male", "female"):
                old = old_results.get(task["task_key"])
                if not isinstance(old, dict) or old.get("status") != "success":
                    raise RuntimeError(
                        "missing prior Qwen audit result for official speaker: %s"
                        % task["task_key"]
                    )
                results[task["task_key"]] = old
    resumed = 0
    submitted = 0
    completed = 0
    print(
        "AMI gender batch: samples=%d clips=%d APIs=%d workers=%d (%d/API) output=%s"
        % (
            len(sample_rows),
            len(tasks),
            len(api_files),
            len(api_files) * args.workers_per_api,
            args.workers_per_api,
            args.output_dir,
        ),
        flush=True,
    )
    try:
        with contextlib.ExitStack() as stack:
            pools = [
                stack.enter_context(
                    concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.workers_per_api,
                        thread_name_prefix="qwen%d" % (index + 1),
                    )
                )
                for index in range(len(api_files))
            ]
            futures = {}
            for task_index, task in enumerate(selected_tasks):
                old = None if args.overwrite else reusable_checkpoint(
                    args.output_dir, task, fingerprint
                )
                if old is not None:
                    results[task["task_key"]] = old
                    resumed += 1
                    completed += 1
                    continue
                api_index = task_index % len(api_files)
                future = pools[api_index].submit(
                    process_task,
                    task,
                    api_keys[api_index],
                    api_files[api_index],
                    args,
                    fingerprint,
                    args.overwrite,
                )
                futures[future] = task
                submitted += 1

            print("resumed=%d submitted=%d" % (resumed, submitted), flush=True)
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "task_key": task["task_key"],
                        "sample_id": task["sample_id"],
                        "speaker_id": task["speaker_id"],
                        "source_speaker": task["source_speaker"],
                        "status": "failed",
                        "errors": [str(error)],
                        "config_fingerprint": fingerprint,
                        "created_at": utc_now(),
                    }
                results[task["task_key"]] = result
                atomic_write_json(checkpoint_path(args.output_dir, task["task_key"]), result)
                completed += 1
                if completed % 20 == 0 or completed == len(selected_tasks):
                    selected_results = [
                        results[item["task_key"]]
                        for item in selected_tasks
                        if item["task_key"] in results
                    ]
                    success = sum(
                        item.get("status") == "success" for item in selected_results
                    )
                    print(
                        "progress=%d/%d success=%d failed=%d"
                        % (
                            completed,
                            len(selected_tasks),
                            success,
                            completed - success,
                        ),
                        flush=True,
                    )

        failed_selected = [
            task["task_key"]
            for task in selected_tasks
            if results.get(task["task_key"], {}).get("status") != "success"
        ]
        if failed_selected:
            print(
                json.dumps(
                    {
                        "status": "incomplete",
                        "failed_count": len(failed_selected),
                        "failed_task_keys": failed_selected,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            return 1

        summary = finalize(args.output_dir, sample_rows, results, config, started_at)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if summary["status"] == "complete" else 1
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
