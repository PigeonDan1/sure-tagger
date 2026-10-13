#!/usr/bin/env python3
"""Prepare an audio-only benchmark runtime and deterministic worker shards.

The benchmark manifest is intentionally kept separate from the runtime input.
Only the audio path and an external sample ID survive into the records consumed
by sure-tagger. The source benchmark manifest, metadata, and GT are never
copied into a runtime record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import wave
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"
DEFAULT_SOURCE_MANIFEST = DEFAULT_BENCHMARK_ROOT / "benchmark.manifest.jsonl"
DEFAULT_OUTPUT_ROOT = COMPARE_ROOT / "evaluation" / "benchmark-v1"
DEFAULT_SHARD_COUNT = 7
DEFAULT_GPU_COUNT = 7
DEFAULT_WORKERS_PER_GPU = 1
SOURCE_URL_KEYS = ("article", "dataset_card", "github", "huggingface")


class PreparationError(RuntimeError):
    """Raised when the frozen benchmark cannot be made into a safe runtime."""


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _atomic_write(path, "".join(_json_line(row) + "\n" for row in rows))


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_info(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            sample_width = source.getsampwidth()
    except (OSError, EOFError, wave.Error) as exc:
        raise PreparationError("invalid WAV %s: %s" % (path, exc))
    if channels <= 0 or sample_rate <= 0 or frames < 0 or sample_width <= 0:
        raise PreparationError("invalid WAV header: %s" % path)
    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "sample_width_bytes": sample_width,
        "duration_sec": frames / float(sample_rate),
    }


def _safe_sample_id(value: Any, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PreparationError("line %d has an invalid sample_id" % line_number)
    # IDs become flat filenames. Reject traversal, separators, and control chars.
    if value in (".", "..") or Path(value).name != value:
        raise PreparationError("line %d sample_id is not a safe flat filename: %r" % (line_number, value))
    if any(ord(char) < 32 for char in value) or "\x00" in value:
        raise PreparationError("line %d sample_id contains a control character" % line_number)
    return value


def _resolve_source_audio(value: Any, benchmark_root: Path, line_number: int) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError("line %d has no sample.audio.path" % line_number)
    raw = Path(value)
    candidate = raw if raw.is_absolute() else benchmark_root / raw
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PreparationError("line %d audio path cannot be resolved: %s (%s)" % (line_number, candidate, exc))
    root = benchmark_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PreparationError("line %d audio path escapes benchmark root: %s" % (line_number, value))
    if not resolved.is_file():
        raise PreparationError("line %d audio path is not a file: %s" % (line_number, resolved))
    if resolved.suffix.lower() != ".wav":
        raise PreparationError("line %d audio must be a WAV file: %s" % (line_number, resolved))
    return resolved


def _load_entries(source_manifest: Path, benchmark_root: Path, expected_count: int) -> List[Dict[str, Any]]:
    if not source_manifest.is_file():
        raise PreparationError("source manifest not found: %s" % source_manifest)
    entries: List[Dict[str, Any]] = []
    seen = set()
    seen_audio = set()
    with source_manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise PreparationError("%s:%d is invalid JSON: %s" % (source_manifest, line_number, exc))
            if not isinstance(record, dict):
                raise PreparationError("%s:%d must contain an object" % (source_manifest, line_number))
            sample = record.get("sample")
            if not isinstance(sample, dict):
                raise PreparationError("%s:%d has no sample object" % (source_manifest, line_number))
            sample_id = _safe_sample_id(sample.get("sample_id"), line_number)
            if sample_id in seen:
                raise PreparationError("duplicate sample_id: %s" % sample_id)
            seen.add(sample_id)
            audio = sample.get("audio")
            audio_value = audio.get("path") if isinstance(audio, dict) else None
            audio_path = _resolve_source_audio(audio_value, benchmark_root, line_number)
            if audio_path in seen_audio:
                raise PreparationError("duplicate audio path: %s" % audio_path)
            seen_audio.add(audio_path)
            info = _wav_info(audio_path)
            entries.append(
                {
                    "global_index": len(entries),
                    "sample_id": sample_id,
                    "source_path": audio_path,
                    "source_bytes": audio_path.stat().st_size,
                    "source_sha256": _sha256(audio_path),
                    **info,
                }
            )
    if expected_count >= 0 and len(entries) != expected_count:
        raise PreparationError(
            "expected %d source rows, found %d in %s" % (expected_count, len(entries), source_manifest)
        )
    if not entries:
        raise PreparationError("source manifest is empty: %s" % source_manifest)
    return entries


def _runtime_record(entry: Dict[str, Any], staged_path: Path) -> Dict[str, Any]:
    # These empty fields are required by sure-tagger's closed raw-only schema.
    return {
        "corpus": {
            "dataset_name": "",
            "native_metadata": {},
            "source_urls": {key: [] for key in SOURCE_URL_KEYS},
        },
        "sample": {
            "audio": {"path": str(staged_path)},
            "native_metadata": {},
            "sample_id": entry["sample_id"],
            "text": {"transcript": ""},
        },
    }


def _ensure_staged_link(target: Path, source: Path, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(target)):
        if target.is_symlink() and os.path.realpath(str(target)) == str(source):
            return
        if not force:
            raise PreparationError(
                "staging target exists and does not point at the source: %s (use --force to replace)" % target
            )
        if target.is_dir() and not target.is_symlink():
            raise PreparationError("refusing to replace directory: %s" % target)
        target.unlink()
    os.symlink(str(source), str(target))


def _assign_shards(entries: Sequence[Dict[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    # Largest-first greedy placement is deterministic and keeps worker wall time close.
    for entry in sorted(entries, key=lambda item: (-item["duration_sec"], item["global_index"])):
        shard = min(
            range(shard_count),
            key=lambda index: (loads[index], len(buckets[index]), index),
        )
        buckets[shard].append(entry)
        loads[shard] += entry["duration_sec"]
    for bucket in buckets:
        bucket.sort(key=lambda item: item["global_index"])
    return buckets


def _validate_runtime_record(record: Any, expected_audio: Path, expected_id: str) -> None:
    if not isinstance(record, dict) or set(record) != {"corpus", "sample"}:
        raise PreparationError("runtime record has non-closed top-level keys")
    corpus = record["corpus"]
    sample = record["sample"]
    if not isinstance(corpus, dict) or set(corpus) != {"dataset_name", "native_metadata", "source_urls"}:
        raise PreparationError("runtime record corpus schema mismatch")
    if corpus["dataset_name"] != "" or corpus["native_metadata"] != {}:
        raise PreparationError("runtime record carries corpus metadata")
    if set(corpus["source_urls"]) != set(SOURCE_URL_KEYS) or any(corpus["source_urls"][key] != [] for key in SOURCE_URL_KEYS):
        raise PreparationError("runtime record carries source URLs")
    if not isinstance(sample, dict) or set(sample) != {"audio", "native_metadata", "sample_id", "text"}:
        raise PreparationError("runtime record sample schema mismatch")
    if sample["sample_id"] != expected_id or sample["native_metadata"] != {}:
        raise PreparationError("runtime record carries sample metadata or has an ID mismatch")
    if sample["text"] != {"transcript": ""}:
        raise PreparationError("runtime record carries transcript text")
    if sample["audio"] != {"path": str(expected_audio)}:
        raise PreparationError("runtime record audio path mismatch")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise PreparationError("%s:%d is invalid JSON: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise PreparationError("%s:%d is not an object" % (path, line_number))
            rows.append(value)
    return rows


def _validate_output(output_root: Path, entries: Sequence[Dict[str, Any]], shard_count: int, gpu_count: int, workers_per_gpu: int) -> Dict[str, Any]:
    manifest_path = output_root / "runtime.manifest.jsonl"
    if not manifest_path.is_file():
        raise PreparationError("runtime manifest not found: %s" % manifest_path)
    rows = _read_jsonl(manifest_path)
    if len(rows) != len(entries):
        raise PreparationError("runtime manifest row count mismatch: %d vs %d" % (len(rows), len(entries)))
    expected_ids = [entry["sample_id"] for entry in entries]
    expected_id_set = set(expected_ids)
    observed_ids = []
    for entry, row in zip(entries, rows):
        staged = output_root / "audio" / (entry["sample_id"] + ".wav")
        _validate_runtime_record(row, staged, entry["sample_id"])
        if not staged.is_file() or not staged.is_symlink():
            raise PreparationError("staged audio is not a symlinked file: %s" % staged)
        if os.path.realpath(str(staged)) != str(entry["source_path"]):
            raise PreparationError("staged audio target mismatch: %s" % staged)
        observed_ids.append(row["sample"]["sample_id"])
    if observed_ids != expected_ids:
        raise PreparationError("runtime manifest order or IDs differ from source manifest")

    seen = set()
    shard_rows = []
    for shard in range(shard_count):
        path = output_root / "shards" / ("manifest_%02d.jsonl" % shard)
        if not path.is_file():
            raise PreparationError("missing shard manifest: %s" % path)
        rows_for_shard = _read_jsonl(path)
        shard_rows.append(len(rows_for_shard))
        for row in rows_for_shard:
            sample = row.get("sample")
            sid = sample.get("sample_id") if isinstance(sample, dict) else None
            if sid in seen:
                raise PreparationError("duplicate sample_id across shards: %s" % sid)
            seen.add(sid)
            if sid not in expected_id_set:
                raise PreparationError("unexpected sample_id in shard: %s" % sid)
            expected = output_root / "audio" / (sid + ".wav")
            _validate_runtime_record(row, expected, sid)
    if seen != expected_id_set:
        raise PreparationError("shards do not cover exactly the source IDs")
    if shard_count != gpu_count * workers_per_gpu:
        raise PreparationError("shard count must equal gpu_count * workers_per_gpu")
    return {
        "sample_count": len(entries),
        "shard_count": shard_count,
        "shard_counts": shard_rows,
        "gpu_count": gpu_count,
        "workers_per_gpu": workers_per_gpu,
        "audio_only": True,
    }


def _launcher_text(sure_tagger_root: Path, python_bin: str, shard_count: int, gpu_count: int, workers_per_gpu: int) -> str:
    # Keep this launcher generated and auditable; preparation itself never
    # starts it. The canonical runner owns the serial core/aux lifecycle and
    # cannot accidentally put two model stacks on one GPU.
    del sure_tagger_root, python_bin, shard_count, gpu_count, workers_per_gpu
    canonical = str((SCRIPT_DIR / "run_sure_tagger_7.sh").resolve())
    return """#!/usr/bin/env bash
set -u
set -o pipefail

# Generated compatibility entry point; one top-level worker per GPU.
exec "%s" "$@"
""" % canonical


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    shard_count = int(args.shard_count)
    gpu_count = int(args.gpu_count)
    workers_per_gpu = int(args.workers_per_gpu)
    if shard_count < 1 or gpu_count < 1 or workers_per_gpu < 1:
        raise PreparationError("shard/gpu/worker counts must be positive")
    if shard_count != gpu_count * workers_per_gpu:
        raise PreparationError("--shard-count must equal --gpu-count * --workers-per-gpu")

    launcher = output_root / "run_sure_tagger_7.sh"
    launcher_text = _launcher_text(
        Path(args.sure_tagger_root).resolve(),
        args.python_bin,
        shard_count,
        gpu_count,
        workers_per_gpu,
    )
    if launcher.exists() and not args.force:
        try:
            launcher_matches = launcher.read_text(encoding="utf-8") == launcher_text
        except OSError as exc:
            raise PreparationError("cannot inspect existing launcher %s: %s" % (launcher, exc))
        if not launcher_matches:
            raise PreparationError("launcher exists with different content; use --force to regenerate: %s" % launcher)

    entries = _load_entries(source_manifest, benchmark_root, int(args.expected_count))
    output_root.mkdir(parents=True, exist_ok=True)
    launcher = output_root / "run_sure_tagger_7.sh"
    audio_dir = output_root / "audio"
    shard_dir = output_root / "shards"
    sure_dir = output_root / "sure_tagger"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    (sure_dir / "logs").mkdir(parents=True, exist_ok=True)
    (sure_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    expected_audio_names = {entry["sample_id"] + ".wav" for entry in entries}
    extras = sorted(path.name for path in audio_dir.glob("*.wav") if path.name not in expected_audio_names)
    if extras:
        raise PreparationError("audio staging contains unexpected WAV files: %s" % ", ".join(extras[:10]))
    runtime_rows = []
    for entry in entries:
        staged = audio_dir / (entry["sample_id"] + ".wav")
        _ensure_staged_link(staged, entry["source_path"], bool(args.force))
        runtime_rows.append(_runtime_record(entry, staged))

    _write_jsonl(output_root / "runtime.manifest.jsonl", runtime_rows)
    buckets = _assign_shards(entries, shard_count)
    shard_summaries = []
    sample_index = []
    for shard, bucket in enumerate(buckets):
        rows = [_runtime_record(entry, audio_dir / (entry["sample_id"] + ".wav")) for entry in bucket]
        _write_jsonl(shard_dir / ("manifest_%02d.jsonl" % shard), rows)
        shard_duration = sum(entry["duration_sec"] for entry in bucket)
        shard_summaries.append(
            {
                "shard": shard,
                "gpu": shard // workers_per_gpu,
                "worker_slot": shard % workers_per_gpu,
                "manifest": str((shard_dir / ("manifest_%02d.jsonl" % shard)).resolve()),
                "output": str((sure_dir / ("tags_%02d.jsonl" % shard)).resolve()),
                "artifact_dir": str((sure_dir / "artifacts" / ("shard_%02d" % shard)).resolve()),
                "log": str((sure_dir / "logs" / ("shard_%02d.log" % shard)).resolve()),
                "sample_count": len(bucket),
                "duration_sec": round(shard_duration, 6),
            }
        )
        for rank, entry in enumerate(bucket):
            sample_index.append(
                {
                    "global_index": entry["global_index"],
                    "sample_id": entry["sample_id"],
                    "audio": str(audio_dir / (entry["sample_id"] + ".wav")),
                    "source_audio": str(entry["source_path"]),
                    "source_sha256": entry["source_sha256"],
                    "bytes": entry["source_bytes"],
                    "duration_sec": round(entry["duration_sec"], 6),
                    "sample_rate_hz": entry["sample_rate_hz"],
                    "channels": entry["channels"],
                    "shard": shard,
                    "shard_rank": rank,
                    "gpu": shard // workers_per_gpu,
                    "worker_slot": shard % workers_per_gpu,
                }
            )
    sample_index.sort(key=lambda item: item["global_index"])
    _write_json(
        output_root / "staging.index.json",
        {
            "audio_only": True,
            "benchmark_root": str(benchmark_root),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "sample_count": len(entries),
            "shard_count": shard_count,
            "gpu_count": gpu_count,
            "workers_per_gpu": workers_per_gpu,
            "audio_dir": str(audio_dir),
            "runtime_manifest": str((output_root / "runtime.manifest.jsonl").resolve()),
            "shards": shard_summaries,
            "samples": sample_index,
        },
    )
    validation_summary = _validate_output(
        output_root, entries, shard_count, gpu_count, workers_per_gpu
    )
    validation_summary.update(
        {
            "output_root": str(output_root),
            "runtime_manifest": str((output_root / "runtime.manifest.jsonl").resolve()),
            "staging_index": str((output_root / "staging.index.json").resolve()),
            "launcher": str(launcher.resolve()),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
        }
    )
    _write_json(output_root / "prepare.summary.json", validation_summary)

    if not launcher.exists() or launcher.read_text(encoding="utf-8") != launcher_text:
        if launcher.exists() and not args.force:
            raise PreparationError("launcher exists with different content; use --force to regenerate: %s" % launcher)
        _atomic_write(launcher, launcher_text, mode=0o755)
    result = _validate_output(output_root, entries, shard_count, gpu_count, workers_per_gpu)
    result.update(
        {
            "output_root": str(output_root),
            "runtime_manifest": str((output_root / "runtime.manifest.jsonl").resolve()),
            "staging_index": str((output_root / "staging.index.json").resolve()),
            "launcher": str(launcher.resolve()),
            "shards": [item["sample_count"] for item in shard_summaries],
        }
    )
    return result


def validate_only(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    entries = _load_entries(source_manifest, benchmark_root, int(args.expected_count))
    result = _validate_output(
        output_root,
        entries,
        int(args.shard_count),
        int(args.gpu_count),
        int(args.workers_per_gpu),
    )
    result.update({"output_root": str(output_root), "validated_only": True})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sure-tagger-root", default="/hpc_stor03/sjtu_home/weihan.chen/share/sure-tagger-main")
    parser.add_argument("--python-bin", default="/usr/bin/python3.9")
    parser.add_argument("--expected-count", type=int, default=650)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--gpu-count", type=int, default=DEFAULT_GPU_COUNT)
    parser.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    parser.add_argument("--force", action="store_true", help="Replace generated links/launcher when their targets differ.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing runtime without changing files.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_only(args) if args.validate_only else prepare(args)
    except (OSError, PreparationError) as exc:
        print("prepare benchmark runtime: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
