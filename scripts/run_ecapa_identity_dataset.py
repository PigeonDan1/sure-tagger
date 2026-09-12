#!/usr/bin/env python3
"""Run SpeechBrain ECAPA-TDNN identity diagnostics on a frozen manifest.

The model sees only sample-local audio crops. Native speaker IDs are used
after inference to label same/different pairs for the diagnostic scorer.
"""

from __future__ import print_function

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import threading
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

DEFAULT_MODEL_DIR = "models/speaker/speechbrain_ecapa_voxceleb"
DEFAULT_PYTHON = ".runtime/fireredvad_rebuild_py310/bin/python"
MODEL_VERSION = "speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
_MODEL_INIT_LOCK = threading.Lock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-regions-per-speaker", type=int, default=2)
    parser.add_argument("--min-region-duration-sec", type=float, default=0.80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_regions_per_speaker < 1:
        raise ValueError("workers and max-regions-per-speaker must be positive")

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    thread_local = threading.local()
    pending = []
    results = []
    for line_number, record in records:
        sample_id = record["sample"]["sample_id"]
        if sample_id in completed:
            results.append(completed[sample_id])
        else:
            pending.append((line_number, record))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                item,
                manifest_path.parent,
                Path(args.model_dir).resolve(),
                thread_local,
                args.max_regions_per_speaker,
                args.min_region_duration_sec,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            _append_jsonl(output_path, row)

    results.sort(key=lambda row: row.get("sample_id", ""))
    _write_jsonl_atomic(output_path, results)
    pairs = [pair for row in results for pair in row.get("pairs", [])]
    summary = {
        "model_id": "speechbrain_ecapa_voxceleb",
        "model_label": "SpeechBrain ECAPA-TDNN (VoxCeleb)",
        "status": "diagnostic_measured_uncalibrated",
        "manifest": str(manifest_path),
        "results": str(output_path),
        "sample_count": len(results),
        "success_count": sum(row.get("status") == "ok" for row in results),
        "failure_count": sum(row.get("status") != "ok" for row in results),
        "pair_count": len(pairs),
        "same_pair_count": sum(pair.get("gold_same") for pair in pairs),
        "different_pair_count": sum(not pair.get("gold_same") for pair in pairs),
        "workers": args.workers,
        "device": "cpu",
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_version": MODEL_VERSION,
        "native_metadata_entered_inference": False,
        "claim_scope": ["I"],
    }
    summary_path = output_path.with_name("ecapa_identity_run_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


def _get_model(thread_local, model_dir):
    model = getattr(thread_local, "ecapa_model", None)
    if model is None:
        from speechbrain.pretrained import EncoderClassifier

        source = str(model_dir)
        # SpeechBrain 0.5.x creates optional-file symlinks in savedir. Serialize
        # this one-time per-worker initialization to avoid concurrent races.
        with _MODEL_INIT_LOCK:
            model = EncoderClassifier.from_hparams(
                source=source,
                savedir=str(model_dir / "runtime_cache"),
                overrides={"pretrained_path": source},
                run_opts={"device": "cpu"},
            )
        model.eval()
        thread_local.ecapa_model = model
    return model


def _run_one(item, manifest_dir, model_dir, thread_local, max_regions, min_duration):
    line_number, record = item
    sample = record["sample"]
    sample_id = sample["sample_id"]
    try:
        audio_path = _resolve_audio(sample["audio"]["path"], manifest_dir)
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if sample_rate != 16000:
            raise ValueError("ECAPA expects 16 kHz audio, got %s" % sample_rate)
        regions = _select_regions(sample, max_regions, min_duration)
        pairs = []
        model = _get_model(thread_local, model_dir)
        with tempfile.TemporaryDirectory(prefix="sure_tagger_ecapa_") as tmpdir:
            embeddings = {}
            for index, region in enumerate(regions):
                start = max(0, int(round(region["start_sec"] * sample_rate)))
                end = min(len(audio), int(round(region["end_sec"] * sample_rate)))
                if end <= start:
                    continue
                path = Path(tmpdir) / ("region_%04d.wav" % index)
                sf.write(str(path), audio[start:end], sample_rate, subtype="PCM_16")
                signal, _ = sf.read(str(path), dtype="float32")
                import torch
                tensor = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    embedding = model.encode_batch(tensor, normalize=False).squeeze()
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1).cpu().numpy()
                embeddings[region["region_id"]] = embedding
            for left in range(len(regions)):
                for right in range(left + 1, len(regions)):
                    a = regions[left]; b = regions[right]
                    if a["region_id"] not in embeddings or b["region_id"] not in embeddings:
                        continue
                    score = float(np.dot(embeddings[a["region_id"]], embeddings[b["region_id"]]))
                    pairs.append({
                        "region_ids": [a["region_id"], b["region_id"]],
                        "speaker_ids": [a["speaker_id"], b["speaker_id"]],
                        "gold_same": a["speaker_id"] == b["speaker_id"],
                        "score": round(score, 8),
                    })
        return {
            "sample_id": sample_id,
            "status": "ok",
            "model_id": "speechbrain_ecapa_voxceleb",
            "model_version": MODEL_VERSION,
            "audio_path": str(audio_path),
            "regions": regions,
            "pairs": pairs,
            "native_metadata_entered_inference": False,
        }
    except Exception as exc:  # noqa: BLE001 - preserve per-sample failures.
        return {
            "sample_id": sample_id,
            "status": "failed",
            "model_id": "speechbrain_ecapa_voxceleb",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "manifest_line": line_number,
        }


def _select_regions(sample, max_regions, min_duration):
    utterances = sample.get("native_metadata", {}).get("utterances", [])
    all_intervals = [(float(item["start"]), float(item["end"]), str(item["speaker"])) for item in utterances]
    by_speaker = {}
    for index, utt in enumerate(utterances):
        start = float(utt["start"]); end = float(utt["end"])
        if end - start < min_duration:
            continue
        # Identity pairs use clean, sample-local regions. Exclude an entire
        # utterance if any other native speaker is active during it.
        if any(
            other_speaker != str(utt["speaker"])
            and max(start, other_start) < min(end, other_end)
            for other_start, other_end, other_speaker in all_intervals
        ):
            continue
        item = {
            "region_id": "native_%06d" % index,
            "speaker_id": str(utt["speaker"]),
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
        }
        by_speaker.setdefault(item["speaker_id"], []).append(item)
    selected = []
    for speaker_id, items in by_speaker.items():
        selected.extend(sorted(items, key=lambda x: (-x["duration_sec"], x["start_sec"]))[:max_regions])
    return sorted(selected, key=lambda x: (x["start_sec"], x["end_sec"], x["region_id"]))


def _resolve_audio(path, manifest_dir):
    value = Path(str(path)).expanduser()
    return value.resolve() if value.is_absolute() else (manifest_dir / value).resolve()


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
