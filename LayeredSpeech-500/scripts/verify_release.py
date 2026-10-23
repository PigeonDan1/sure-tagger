#!/usr/bin/env python3
"""Validate portability and publication hygiene for the release directory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
PRIVATE_MARKERS = ("api" + "_key", "DASH" + "SCOPE_API_KEY", "OPENAI" + "_API_KEY")
PRIVATE_PATH_MARKERS = ("/" + "mnt/", "/" + "hpc", "/" + "home/", "/" + "tmp/")
REQUIRED_FILES = (
    "LICENSE", "NOTICE", "README.md", "pyproject.toml", "uv.lock",
    "data/scene_scripts_v4.jsonl", "data/asset_hashes.json",
    "data/models.lock.yaml", "data/model_hashes.json", "manifests/summary.json",
)


def main() -> None:
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"missing release file: {relative}")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" not in license_text:
        failures.append("LICENSE does not contain the Apache-2.0 terms")
    if len(license_text.splitlines()) < 170:
        failures.append("LICENSE appears truncated")
    try:
        assets = json.loads((ROOT / "data/asset_hashes.json").read_text(encoding="utf-8"))
        if len(assets["files"]) != 315:
            failures.append("asset hash inventory must contain 315 files")
        if len({item["path"] for item in assets["files"]}) != len(assets["files"]):
            failures.append("asset hash inventory contains duplicate paths")
        for item in assets["files"]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts:
                failures.append(f"unsafe asset path: {path}")
            if len(item["sha256"]) != 64 or item["bytes"] <= 0:
                failures.append(f"invalid asset hash entry: {path}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        failures.append(f"invalid asset hash inventory: {exc}")
    try:
        model_inventory = json.loads((ROOT / "data/model_hashes.json").read_text(encoding="utf-8"))
        if set(model_inventory["models"]) != {"dots_tts_base", "dots_tts_edit"}:
            failures.append("model hash inventory must cover both Dots checkpoints")
        for files in model_inventory["models"].values():
            for item in files:
                if len(item["sha256"]) != 64 or item["bytes"] <= 0:
                    failures.append(f"invalid model hash entry: {item['path']}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        failures.append(f"invalid model hash inventory: {exc}")
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            failures.append(f"symbolic link in release: {path.relative_to(ROOT)}")
            continue
        if not path.is_file() or any(
            part in {".git", ".venv", "__pycache__", "runs", "models", "external_assets"}
            for part in path.parts
        ):
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size > 20_000_000:
            failures.append(f"unexpectedly large file: {path.relative_to(ROOT)}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in PRIVATE_MARKERS + PRIVATE_PATH_MARKERS):
            failures.append(f"private path/credential marker: {path.relative_to(ROOT)}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        failures.append(f"invalid JSONL {path.relative_to(ROOT)}:{line_number}: {exc}")
    try:
        scenes = [
            json.loads(line)
            for line in (ROOT / "data/scene_scripts_v4.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads((ROOT / "manifests/summary.json").read_text(encoding="utf-8"))
        if len(scenes) != summary.get("scene_count") or len(scenes) != 500:
            failures.append("scene count differs between script and published summary")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid scene summary: {exc}")
    if failures:
        raise SystemExit("release verification failed:\n- " + "\n- ".join(failures[:40]))
    print("release verification passed")


if __name__ == "__main__":
    main()
