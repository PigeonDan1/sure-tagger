#!/usr/bin/env python3
"""Cut an AMI meeting into silence-bounded, multi-utt test samples."""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import wave


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.input_schema import validate_input_record  # noqa: E402


DEFAULT_ANNOTATIONS = ROOT / "EN2001a.jsonl"
DEFAULT_SOURCE_AUDIO = (
    ROOT
    / "AMI"
    / "amicorpus-Mix-Headset"
    / "EN2001a"
    / "audio"
    / "EN2001a.Mix-Headset.wav"
)
DEFAULT_OUTPUT_DIR = ROOT / "ami_en2001a_utterances"
DEFAULT_MIN_DURATION_SEC = 10.0
DEFAULT_TARGET_DURATION_SEC = 20.0
DEFAULT_MAX_DURATION_SEC = 30.0
ENDPOINT_PADDING_SEC = 0.5
OUT_OF_RANGE_PENALTY = 1000.0
ROUND_DIGITS = 6


def load_utterances(path):
    utterances = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_utterance(record, line_number)
            utt_id = record["utt_id"]
            if utt_id in seen_ids:
                raise ValueError("line %d: duplicate utt_id %r" % (line_number, utt_id))
            seen_ids.add(utt_id)
            utterances.append(record)
    if not utterances:
        raise ValueError("annotation file contains no utterances: %s" % path)
    return utterances


def _validate_utterance(record, line_number):
    required = {"utt_id", "audio_id", "speaker", "start", "end", "text", "words"}
    if not isinstance(record, dict):
        raise ValueError("line %d: utterance must be an object" % line_number)
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(
            "line %d: missing fields: %s" % (line_number, ", ".join(missing))
        )
    for key in ("utt_id", "audio_id", "speaker", "text"):
        if not isinstance(record[key], str):
            raise ValueError("line %d: %s must be a string" % (line_number, key))
    if not record["utt_id"] or Path(record["utt_id"]).name != record["utt_id"]:
        raise ValueError("line %d: unsafe utt_id %r" % (line_number, record["utt_id"]))
    start = record["start"]
    end = record["end"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
    ):
        raise ValueError("line %d: invalid interval %r-%r" % (line_number, start, end))
    if not isinstance(record["words"], list):
        raise ValueError("line %d: words must be an array" % line_number)
    for word_index, word in enumerate(record["words"]):
        if not isinstance(word, dict):
            raise ValueError(
                "line %d: words[%d] must be an object" % (line_number, word_index)
            )
        for field in ("start", "end"):
            value = word.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "line %d: words[%d].%s must be finite"
                    % (line_number, word_index, field)
                )


def merge_activity_intervals(utterances):
    intervals = sorted((row["start"], row["end"]) for row in utterances)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def choose_silence_boundaries(
    utterances,
    min_duration_sec=DEFAULT_MIN_DURATION_SEC,
    target_duration_sec=DEFAULT_TARGET_DURATION_SEC,
    max_duration_sec=DEFAULT_MAX_DURATION_SEC,
):
    if not (0 < min_duration_sec <= target_duration_sec <= max_duration_sec):
        raise ValueError("durations must satisfy 0 < min <= target <= max")

    activity = merge_activity_intervals(utterances)
    candidates = [max(0.0, activity[0][0] - ENDPOINT_PADDING_SEC)]
    candidates.extend(
        (left[1] + right[0]) / 2.0
        for left, right in zip(activity, activity[1:])
    )
    candidates.append(activity[-1][1] + ENDPOINT_PADDING_SEC)

    # Every candidate is in global non-speech. Dynamic programming chooses a
    # subset near the target duration. A large penalty keeps normal segments in
    # range, while still permitting an indivisible long utt/overlap component.
    best_cost = [None] * len(candidates)
    previous = [None] * len(candidates)
    best_cost[0] = 0.0
    for end_index in range(1, len(candidates)):
        for start_index in range(end_index):
            if best_cost[start_index] is None:
                continue
            duration = candidates[end_index] - candidates[start_index]
            if duration < min_duration_sec:
                continue
            cost = best_cost[start_index] + _duration_penalty(
                duration,
                target_duration_sec,
                max_duration_sec,
            )
            if best_cost[end_index] is None or cost < best_cost[end_index]:
                best_cost[end_index] = cost
                previous[end_index] = start_index

    if best_cost[-1] is None:
        total_duration = candidates[-1] - candidates[0]
        if total_duration < min_duration_sec:
            return [candidates[0], candidates[-1]]
        raise ValueError("could not satisfy the minimum segment duration")

    selected = []
    index = len(candidates) - 1
    while index is not None:
        selected.append(candidates[index])
        index = previous[index]
    selected.reverse()
    return selected


def _duration_penalty(duration, target, maximum):
    cost = (duration - target) ** 2
    if duration > maximum:
        cost += OUT_OF_RANGE_PENALTY * (duration - maximum) ** 2
    return cost


def build_segment_specs(utterances, boundaries, frame_rate):
    boundary_frames = [int(round(value * frame_rate)) for value in boundaries]
    if any(right <= left for left, right in zip(boundary_frames, boundary_frames[1:])):
        raise ValueError("silence boundaries collapse after frame quantization")

    specs = []
    assigned_ids = []
    for index, (start_frame, end_frame) in enumerate(
        zip(boundary_frames, boundary_frames[1:])
    ):
        source_start = boundaries[index]
        source_end = boundaries[index + 1]
        members = [
            row
            for row in utterances
            if row["start"] >= source_start and row["end"] <= source_end
        ]
        if not members:
            raise ValueError("segment %d contains no utterances" % index)
        assigned_ids.extend(row["utt_id"] for row in members)
        specs.append(
            {
                "segment_id": "%s_utterance_%05d" % (members[0]["audio_id"], index),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "utterances": members,
            }
        )

    expected_ids = [row["utt_id"] for row in utterances]
    if sorted(assigned_ids) != sorted(expected_ids):
        raise ValueError("silence segmentation did not assign every utt_id exactly once")
    return specs


def make_manifest_record(spec, frame_rate):
    origin_sec = spec["start_frame"] / float(frame_rate)
    duration_sec = (spec["end_frame"] - spec["start_frame"]) / float(frame_rate)
    shifted_utterances = [
        _shift_utterance(row, origin_sec) for row in spec["utterances"]
    ]
    record = {
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
            "sample_id": spec["segment_id"],
            "audio": {"path": "audio/%s.wav" % spec["segment_id"]},
            "text": {
                "transcript": " ".join(row["text"] for row in spec["utterances"])
            },
            "native_metadata": {
                "audio_id": spec["utterances"][0]["audio_id"],
                "start": 0.0,
                "end": _round_time(duration_sec),
                "utterances": shifted_utterances,
            },
        },
    }
    validate_input_record(record)
    return record


def _shift_utterance(utterance, origin_sec):
    shifted = dict(utterance)
    shifted["start"] = _round_time(utterance["start"] - origin_sec)
    shifted["end"] = _round_time(utterance["end"] - origin_sec)
    shifted["words"] = []
    for word in utterance["words"]:
        shifted_word = dict(word)
        shifted_word["start"] = _round_time(word["start"] - origin_sec)
        shifted_word["end"] = _round_time(word["end"] - origin_sec)
        shifted["words"].append(shifted_word)
    return shifted


def _round_time(value):
    rounded = round(float(value), ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def prepare_dataset(
    annotations,
    source_audio,
    output_dir,
    overwrite=False,
    min_duration_sec=DEFAULT_MIN_DURATION_SEC,
    target_duration_sec=DEFAULT_TARGET_DURATION_SEC,
    max_duration_sec=DEFAULT_MAX_DURATION_SEC,
):
    annotations = Path(annotations)
    source_audio = Path(source_audio)
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.jsonl"
    utterances = load_utterances(annotations)

    if not source_audio.is_file():
        raise FileNotFoundError("source audio not found: %s" % source_audio)
    audio_ids = {row["audio_id"] for row in utterances}
    if len(audio_ids) != 1:
        raise ValueError("annotations must describe exactly one audio_id")
    if not overwrite and (manifest_path.exists() or audio_dir.exists()):
        raise FileExistsError(
            "output already exists; rerun with --overwrite: %s" % output_dir
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_audio), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError("source WAV must contain uncompressed PCM audio")
        params = source.getparams()
        frame_rate = source.getframerate()
        source_frames = source.getnframes()
        source_duration = source_frames / float(frame_rate)
        if max(row["end"] for row in utterances) > source_duration:
            raise ValueError("an utterance extends beyond the source audio")

        boundaries = choose_silence_boundaries(
            utterances,
            min_duration_sec=min_duration_sec,
            target_duration_sec=target_duration_sec,
            max_duration_sec=max_duration_sec,
        )
        # The endpoint padding may extend beyond a recording that ends shortly
        # after its final annotation. Do not ask wave.readframes() for frames
        # that cannot exist; the source-duration check above still rejects
        # annotations that themselves exceed the recording.
        boundaries[-1] = min(boundaries[-1], source_duration)
        specs = build_segment_specs(utterances, boundaries, frame_rate)

        with tempfile.TemporaryDirectory(
            prefix=".ami_utterances_", dir=str(output_dir)
        ) as staging_root:
            staging_root = Path(staging_root)
            staging_audio = staging_root / "audio"
            staging_manifest = staging_root / "manifest.jsonl"
            staging_audio.mkdir()
            with staging_manifest.open("w", encoding="utf-8") as manifest:
                for spec in specs:
                    frame_count = spec["end_frame"] - spec["start_frame"]
                    source.setpos(spec["start_frame"])
                    frames = source.readframes(frame_count)
                    expected_bytes = frame_count * params.nchannels * params.sampwidth
                    if len(frames) != expected_bytes:
                        raise IOError("short read for %s" % spec["segment_id"])

                    output_audio = staging_audio / (spec["segment_id"] + ".wav")
                    with wave.open(str(output_audio), "wb") as sink:
                        sink.setparams(params)
                        sink.writeframes(frames)
                    record = make_manifest_record(spec, frame_rate)
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

            _replace_generated_audio(audio_dir, staging_audio)
            staging_manifest.replace(manifest_path)

    durations = [
        (spec["end_frame"] - spec["start_frame"]) / float(frame_rate)
        for spec in specs
    ]
    return {
        "annotation_path": str(annotations),
        "source_audio_path": str(source_audio),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "source_utt_id_count": len(utterances),
        "utterance_count": len(specs),
        "speaker_count": len({row["speaker"] for row in utterances}),
        "sample_rate_hz": frame_rate,
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
        "mean_duration_sec": sum(durations) / len(durations),
        "over_max_duration_count": sum(d > max_duration_sec for d in durations),
    }


def _replace_generated_audio(audio_dir, staging_audio):
    if audio_dir.exists():
        unexpected = [
            path for path in audio_dir.iterdir() if not path.is_file() or path.suffix != ".wav"
        ]
        if unexpected:
            raise ValueError(
                "refusing to replace audio directory containing non-WAV entries: %s"
                % ", ".join(str(path) for path in unexpected)
            )
        for path in audio_dir.iterdir():
            path.unlink()
        audio_dir.rmdir()
    staging_audio.replace(audio_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--source-audio", type=Path, default=DEFAULT_SOURCE_AUDIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-duration-sec", type=float, default=DEFAULT_MIN_DURATION_SEC)
    parser.add_argument(
        "--target-duration-sec", type=float, default=DEFAULT_TARGET_DURATION_SEC
    )
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the previously generated WAV directory and manifest.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary = prepare_dataset(
        args.annotations,
        args.source_audio,
        args.output_dir,
        overwrite=args.overwrite,
        min_duration_sec=args.min_duration_sec,
        target_duration_sec=args.target_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
