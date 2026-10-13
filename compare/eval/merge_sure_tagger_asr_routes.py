#!/usr/bin/env python3
"""Merge route-specific tags back into the original shard order."""

from __future__ import print_function

import argparse
import json
from pathlib import Path


def load_ids(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line)["sample"]["sample_id"])
    return rows


def load_route(path, ids):
    rows = [json.loads(line) for line in Path(path).open("r", encoding="utf-8") if line.strip()]
    if len(rows) != len(ids):
        raise ValueError("route output count mismatch: %s expected %d got %d" % (path, len(ids), len(rows)))
    return dict(zip(ids, rows))


def merge(root, final_root, gpu_count=7):
    root = Path(root).resolve()
    final_root = Path(final_root).resolve()
    final_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for gpu in range(int(gpu_count)):
        name = "%02d" % gpu
        manifest = root / "manifests" / ("manifest_%s.jsonl" % name)
        ids = load_ids(manifest)
        moss_manifest = root / "routes/moss/manifests" / ("manifest_%s.jsonl" % name)
        official_manifest = root / "routes/official/manifests" / ("manifest_%s.jsonl" % name)
        moss_ids = load_ids(moss_manifest)
        official_ids = load_ids(official_manifest)
        values = {}
        values.update(load_route(root / "routes/moss/outputs" / ("tags_%s.jsonl" % name), moss_ids))
        values.update(load_route(root / "routes/official/outputs" / ("tags_%s.jsonl" % name), official_ids))
        if set(values) != set(ids):
            raise ValueError("route coverage mismatch for shard %s" % name)
        output = final_root / ("tags_%s.jsonl" % name)
        output.write_text(
            "".join(json.dumps(values[sample_id], ensure_ascii=False, sort_keys=True) + "\n" for sample_id in ids),
            encoding="utf-8",
        )
        total += len(ids)
    print(json.dumps({"gpu_count": int(gpu_count), "sample_count": total, "merged": True}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--gpu-count", type=int, default=7)
    args = parser.parse_args(argv)
    merge(args.root, args.final_root, args.gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
