#!/usr/bin/env python3
"""Run the locally pinned Brouhaha VAD on a frozen utterance manifest.

This is a diagnostic VAD track. It uses the official Brouhaha checkpoint in
its isolated CPU runtime and never passes native metadata to model inference.
"""

from __future__ import print_function

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.pipelines.speaker_evidence import resolve_sample_audio_path
from tagger.tools.acoustic_io import probe_audio_info
from tagger.tools.basic_acoustic.brouhaha_signal_estimator import (
    BrouhahaConfig,
    BrouhahaSubprocessClient,
)
from tagger.tools.basic_acoustic.brouhaha_vad_silence_detector import (
    clip_segments_to_duration,
    extract_speech_segments,
)
from tagger.tools.subprocess_runner import close_subprocess_workers


DEFAULT_PYTHON = ".runtime/fireredvad_rebuild_py310/bin/python"
DEFAULT_MODEL = "models/brouhaha/brouhaha-vad/models/best/checkpoints/best.ckpt"
DEFAULT_REPO = "models/brouhaha/brouhaha-vad"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repo-dir", default=DEFAULT_REPO)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                records.append((line_number, json.loads(line)))

    completed = {}
    if args.resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        completed[row["sample_id"]] = row

    context = {
        "_subprocess_workers_lock": threading.Lock(),
        "_subprocess_worker_slots": {"brouhaha_estimate": args.workers},
    }
    config = BrouhahaConfig(
        model_path=args.model,
        repo_dir=args.repo_dir,
        use_gpu=False,
        subprocess_python=args.python,
    )
    client = BrouhahaSubprocessClient(config)
    pending = []
    results = []
    for line_number, record in records:
        sample_id = record["sample"]["sample_id"]
        if sample_id in completed:
            results.append(completed[sample_id])
        else:
            pending.append((line_number, record))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_one, item, manifest_path.parent, client, context): item
                for item in pending
            }
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                _append_jsonl(output_path, row)
    finally:
        close_subprocess_workers(context)

    results.sort(key=lambda row: row.get("sample_id", ""))
    _write_jsonl_atomic(output_path, results)
    summary = {
        "model_id": "brouhaha_vad",
        "model_label": "Brouhaha VAD v0.9.0",
        "manifest": str(manifest_path),
        "output": str(output_path),
        "n_samples": len(results),
        "n_ok": sum(row.get("status") == "ok" for row in results),
        "n_failed": sum(row.get("status") != "ok" for row in results),
        "workers": args.workers,
        "device": "cpu",
        "model": config.to_record(),
        "native_metadata_entered_inference": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["n_failed"] == 0 else 1


def _run_one(item, manifest_dir, client, context):
    line_number, record = item
    sample = record["sample"]
    sample_id = sample["sample_id"]
    try:
        audio_path = resolve_sample_audio_path(sample["audio"]["path"], manifest_dir)
        duration_sec = float(probe_audio_info(audio_path).duration_sec)
        output = client.estimate(audio_path, context=context)
        raw_segments = extract_speech_segments(output)
        speech_segments = clip_segments_to_duration(raw_segments, duration_sec)
        return {
            "sample_id": sample_id,
            "status": "ok",
            "audio_path": str(audio_path),
            "duration_sec": duration_sec,
            "speech_segments": speech_segments,
            "model_id": "brouhaha_vad",
            "model_version": "github:marianne-m/brouhaha-vad@9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891",
            "native_metadata_entered_inference": False,
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-sample failure.
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "manifest_line": line_number,
            "model_id": "brouhaha_vad",
        }


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl_atomic(path, rows):
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
