#!/usr/bin/env python3
"""Start the captioner DeepSeek batch after the Qwen batch is complete.

The watcher is intentionally conservative: it requires the Qwen aggregate to
contain one valid row for every benchmark sample before making any downstream
API request.  It is safe to leave running in the background and uses an OS
lock, so a stale lock file cannot prevent a later watcher from taking over.
"""

from __future__ import print_function

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


COMPARE_ROOT = Path("/mnt/cloudstorfs/sjtu_home/weihan.chen/share/compare")
RAW_DIR = COMPARE_ROOT / "evaluation" / "benchmark-v1" / "captioner_raw"
OUT_DIR = COMPARE_ROOT / "evaluation" / "benchmark-v1" / "captioner-deepseek"
LOCK_PATH = OUT_DIR / ".captioner-watcher-v2.lock"
RUN_LOG = OUT_DIR / "run_full_20260903.log"
POLL_SECONDS = 60
EXPECTED_COUNT = 650
PYTHON = Path(
    "/mnt/cloudstorfs/sjtu_home/weihan.chen/share/sure-tagger-main/.runtime/"
    "campplus_sv_py311_torch280_cu128_v1/bin/python"
)
RUNNER = COMPARE_ROOT / "eval" / "run_captioner_deepseek_batch.py"
MANIFEST = RAW_DIR / "results.jsonl"
SUMMARY = RAW_DIR / "summary.json"


def log(message):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as stream:
        stream.write("[watcher] %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))
        stream.flush()


def complete_qwen_output():
    if not SUMMARY.is_file() or not MANIFEST.is_file():
        return False
    try:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if summary.get("sample_count") != EXPECTED_COUNT:
        return False
    if summary.get("success_count", 0) + summary.get("failed", 0) != EXPECTED_COUNT:
        # Current Qwen summaries use failed_count; accept either spelling for
        # compatibility with an older aggregate writer.
        if summary.get("success_count", 0) + summary.get("failed_count", 0) != EXPECTED_COUNT:
            return False
    try:
        rows = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return False
    if len(rows) != EXPECTED_COUNT or not all(isinstance(row, dict) for row in rows):
        return False
    ids = [row.get("sample_id") for row in rows]
    return len(set(ids)) == EXPECTED_COUNT and all(isinstance(value, str) and value for value in ids)


def running_caption_batch():
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], universal_newlines=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    needle = "run_captioner_deepseek_batch.py"
    return any(needle in line for line in output.splitlines())


def launch():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(RUNNER),
        "--input-manifest",
        str(MANIFEST),
        "--output-dir",
        str(OUT_DIR),
        "--workers",
        "10",
        "--retries",
        "4",
        "--timeout",
        "300",
        "--max-tokens",
        "20000",
        "--thinking",
        "enabled",
        # The directory only contains the historical one-row smoke config;
        # overwrite is therefore safe and prevents a fingerprint mismatch.
        "--overwrite",
    ]
    with RUN_LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log("launched pid=%d workers=10 manifest_rows=%d" % (process.pid, EXPECTED_COUNT))
    return process


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        return 0
    log("watcher started pid=%d" % os.getpid())
    try:
        while True:
            if complete_qwen_output():
                if running_caption_batch():
                    log("captioner DeepSeek process already running; exiting watcher")
                    return 0
                process = launch()
                return process.wait()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log("watcher interrupted")
        return 130
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
