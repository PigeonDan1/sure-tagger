#!/usr/bin/env python3
"""Build the closed raw-only sure-tagger manifest for the 500 rendered clips."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("RELEASE_OUTPUT_ROOT", ROOT / "runs/v4_multilingual"))
SOURCE = DATASET / "tagger" / "tagger_manifest.jsonl"
OUTPUT = DATASET / "tagger" / "input_manifest.jsonl"


def main() -> None:
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        audio_path = Path(source["audio_path"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        rows.append({
            "corpus": {
                "dataset_name": "outputs_l1_l5_dots_v4_multilingual",
                "source_urls": {
                    "article": [],
                    "github": [],
                    "huggingface": [],
                    "dataset_card": [],
                },
                "native_metadata": {},
            },
            "sample": {
                "sample_id": source["id"],
                "audio": {"path": str(audio_path)},
                "text": {"transcript": source.get("transcript", "")},
                "native_metadata": {},
            },
        })
    if len(rows) != 500:
        raise ValueError(f"expected 500 input rows, got {len(rows)}")
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "sample_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
