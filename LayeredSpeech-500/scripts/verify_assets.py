#!/usr/bin/env python3
"""Verify user-supplied inputs against the published file inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/asset_hashes.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(root: Path, manifest: Path = MANIFEST, group: str | None = None) -> list[str]:
    inventory = json.loads(manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = 0
    for item in inventory["files"]:
        if group is not None and item["group"] != group:
            continue
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"invalid manifest path: {item['path']}")
            continue
        path = root / relative
        checked += 1
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size != item["bytes"]:
            errors.append(f"size mismatch: {relative}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {relative}")
    if not checked:
        errors.append(f"no assets matched group {group!r}")
    if not errors:
        print(f"Verified {checked} files under {root}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "external_assets")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--group", choices=("reference_voice", "reference_transcript", "measured_rir", "music")
    )
    args = parser.parse_args()
    errors = verify_assets(args.root.expanduser().resolve(), args.manifest, args.group)
    if errors:
        raise SystemExit("Asset verification failed:\n- " + "\n- ".join(errors[:25]) + (
            f"\n... and {len(errors) - 25} more" if len(errors) > 25 else ""
        ))


if __name__ == "__main__":
    main()
