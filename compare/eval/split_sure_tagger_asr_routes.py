#!/usr/bin/env python3
"""Split prepared shards by the already measured MOSS/DiariZen LID route."""

from __future__ import print_function

import argparse
import json
from pathlib import Path


def is_moss_language(value):
    if not isinstance(value, str):
        return False
    value = value.strip().lower()
    return (
        value == "zh"
        or value.startswith(("zh-", "zh_", "zh "))
        or value == "en"
        or value.startswith(("en-", "en_"))
        or value in ("english", "en english")
    )


def split(root, gpu_count=7):
    root = Path(root).resolve()
    output = root / "routes"
    for route in ("moss", "official"):
        (output / route / "manifests").mkdir(parents=True, exist_ok=True)
        (output / route / "baseline").mkdir(parents=True, exist_ok=True)

    counts = {"moss": 0, "official": 0}
    for gpu in range(int(gpu_count)):
        name = "%02d" % gpu
        manifest = root / "manifests" / ("manifest_%s.jsonl" % name)
        baseline = root / "baseline" / ("tags_%s.jsonl" % name)
        buckets = {"moss": [], "official": []}
        with manifest.open("r", encoding="utf-8") as stream:
            manifest_rows = [json.loads(line) for line in stream if line.strip()]
        with baseline.open("r", encoding="utf-8") as stream:
            baseline_rows = [json.loads(line) for line in stream if line.strip()]
        if len(manifest_rows) != len(baseline_rows):
            raise ValueError("manifest/baseline mismatch for shard %s" % name)
        for record, tags in zip(manifest_rows, baseline_rows):
            language = tags.get("language_content", {}).get("language")
            route = "moss" if is_moss_language(language) else "official"
            buckets[route].append((record, tags))
            counts[route] += 1
        for route, rows in buckets.items():
            (output / route / "manifests" / ("manifest_%s.jsonl" % name)).write_text(
                "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record, _tags in rows),
                encoding="utf-8",
            )
            (output / route / "baseline" / ("tags_%s.jsonl" % name)).write_text(
                "".join(json.dumps(tags, ensure_ascii=False, sort_keys=True) + "\n" for _record, tags in rows),
                encoding="utf-8",
            )
    summary = {"gpu_count": int(gpu_count), "route_counts": counts}
    (output / "split.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu-count", type=int, default=7)
    args = parser.parse_args(argv)
    split(args.root, args.gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
