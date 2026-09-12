#!/usr/bin/env python3
"""Run the pinned pyannote Community-1 pipeline on a frozen manifest.

The runner deliberately preloads audio as a waveform dictionary. This avoids
the optional TorchCodec/FFmpeg path and makes the model input contract explicit.
One process owns one pipeline/GPU; use separate processes for multi-GPU runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.tools.speaker_v2.timeline import summarize_timeline  # noqa: E402


MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
MODEL_VERSION = "%s@%s" % (MODEL_ID, MODEL_REVISION)
MODEL_SHA256 = {
    "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
    "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-activity-sec", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _verify_assets(model_dir)

    records = []
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                records.append((line_number, json.loads(line)))
                if args.max_samples is not None and len(records) >= args.max_samples:
                    break

    completed = {}
    if args.resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        completed[row["sample_id"]] = row

    # Community-1 checkpoints are trusted, pinned local artifacts. The model
    # bundle predates PyTorch 2.6's weights_only default and contains pyannote
    # Specifications objects needed to reconstruct the task.
    import torch
    from lightning.fabric.utilities import cloud_io
    from lightning.pytorch.core import saving

    original_load = cloud_io._load

    def trusted_local_load(path_or_url, map_location=None, weights_only=None):
        return original_load(path_or_url, map_location=map_location, weights_only=False)

    cloud_io._load = trusted_local_load
    saving.pl_load = trusted_local_load
    import pyannote.audio.core.model as model_core

    model_core.pl_load = trusted_local_load
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(model_dir))
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        pipeline.to(torch.device(args.device))

    rows = []
    for line_number, record in records:
        sample = record["sample"]
        sample_id = sample["sample_id"]
        if sample_id in completed:
            rows.append(completed[sample_id])
            continue
        started = time.time()
        try:
            audio_path = _resolve_audio(sample["audio"]["path"], manifest_path.parent)
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.shape[1] == 0:
                raise ValueError("empty audio")
            # Pipeline input is (channel, time), and Community-1 expects 16 kHz.
            tensor = torch.from_numpy(waveform.T.copy())
            if int(sample_rate) != 16000:
                raise ValueError("Community-1 expects 16 kHz audio, got %s" % sample_rate)
            with torch.inference_mode():
                diarization = pipeline({"waveform": tensor, "sample_rate": int(sample_rate)})
            duration_sec = float(waveform.shape[0]) / float(sample_rate)
            raw_segments = _annotation_segments(diarization.speaker_diarization)
            exclusive_segments = _annotation_segments(diarization.exclusive_speaker_diarization)
            raw_summary = summarize_timeline(raw_segments, duration_sec, args.min_activity_sec)
            exclusive_summary = summarize_timeline(exclusive_segments, duration_sec, args.min_activity_sec)
            rows.append({
                "sample_id": sample_id,
                "status": "ok",
                "model_id": MODEL_ID.replace("/", "_"),
                "model_label": "pyannote Community-1",
                "model_version": MODEL_VERSION,
                "model_revision": MODEL_REVISION,
                "model_sha256": MODEL_SHA256,
                "audio_path": str(audio_path),
                "duration_sec": duration_sec,
                "device": str(args.device),
                "raw_segments": raw_segments,
                "exclusive_segments": exclusive_segments,
                "raw_timeline_summary": raw_summary,
                "exclusive_timeline_summary": exclusive_summary,
                "native_metadata_entered_inference": False,
                "runtime_sec": round(time.time() - started, 6),
            })
        except Exception as exc:  # preserve per-sample failures for diagnostics
            rows.append({
                "sample_id": sample_id,
                "status": "failed",
                "model_id": MODEL_ID.replace("/", "_"),
                "model_label": "pyannote Community-1",
                "model_version": MODEL_VERSION,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
                "manifest_line": line_number,
                "device": str(args.device),
                "runtime_sec": round(time.time() - started, 6),
            })
        _append_jsonl(output_path, rows[-1])

    rows.sort(key=lambda row: row.get("sample_id", ""))
    _write_jsonl_atomic(output_path, rows)
    summary = {
        "model_id": MODEL_ID.replace("/", "_"),
        "model_label": "pyannote Community-1",
        "model_version": MODEL_VERSION,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "manifest": str(manifest_path),
        "output": str(output_path),
        "n_samples": len(rows),
        "n_ok": sum(row.get("status") == "ok" for row in rows),
        "n_failed": sum(row.get("status") != "ok" for row in rows),
        "device": str(args.device),
        "native_metadata_entered_inference": False,
        "claim_scope": ["C", "M", "O", "X", "D"],
    }
    output_path.with_name(output_path.stem + "_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["n_failed"] == 0 else 1


def _annotation_segments(annotation):
    segments = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end > start:
            segments.append({
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "speaker_id": str(label),
            })
    return segments


def _resolve_audio(path, manifest_dir):
    value = Path(str(path)).expanduser()
    resolved = value.resolve() if value.is_absolute() else (manifest_dir / value).resolve()
    if not resolved.is_file():
        raise IOError("sample audio does not exist: %s" % resolved)
    return resolved


def _verify_assets(model_dir):
    for relative, expected in MODEL_SHA256.items():
        path = model_dir / relative
        if not path.is_file():
            raise IOError("missing pinned model asset: %s" % path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise IOError("sha256 mismatch for %s: %s != %s" % (path, digest, expected))


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
