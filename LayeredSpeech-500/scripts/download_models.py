#!/usr/bin/env python3
"""Download pinned Dots checkpoints after the user accepts upstream terms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/models.lock.yaml"
HASHES = ROOT / "data/model_hashes.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_model_files(model_id: str, checkpoint_dir: Path) -> list[str]:
    inventory = json.loads(HASHES.read_text(encoding="utf-8"))["models"][model_id]
    errors = []
    for item in inventory:
        path = checkpoint_dir / item["path"]
        if not path.is_file():
            errors.append(f"missing {model_id} file: {item['path']}")
        elif path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            errors.append(f"checkpoint hash mismatch: {model_id}/{item['path']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-model-terms", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    selected = set(args.models or [item["id"] for item in config["models"]])
    known = {item["id"] for item in config["models"]}
    if unknown := selected - known:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")
    if not args.dry_run and not args.accept_model_terms:
        raise SystemExit("Read docs/MODEL_LICENSES.md and pass --accept-model-terms.")
    if not args.dry_run:
        from huggingface_hub import HfApi, snapshot_download

        args.root.mkdir(parents=True, exist_ok=True)
    for item in config["models"]:
        if item["id"] not in selected:
            continue
        target = args.root / item["id"]
        print(f"{item['id']}: {item['repository']}@{item['revision']} -> {target / 'checkpoints'}")
        if args.dry_run:
            continue
        info = HfApi().model_info(item["repository"], revision=item["revision"])
        if info.sha != item["revision"]:
            raise RuntimeError(f"unexpected model revision for {item['id']}: {info.sha}")
        checkpoint_dir = target / "checkpoints"
        snapshot_download(
            repo_id=item["repository"],
            revision=item["revision"],
            local_dir=checkpoint_dir,
        )
        errors = check_model_files(item["id"], checkpoint_dir)
        if errors:
            raise RuntimeError("checkpoint verification failed:\n- " + "\n- ".join(errors))
        (target / "MODEL_SOURCE.json").write_text(
            json.dumps({
                "repository": item["repository"],
                "revision": info.sha,
                "license": item["license"],
            }, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
