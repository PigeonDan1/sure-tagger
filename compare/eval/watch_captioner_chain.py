#!/usr/bin/env python3
"""Bridge the audio captioner batch to its text-only DeepSeek batch.

Qwen writes a complete aggregate only after all manifest rows have been
attempted.  This watcher waits for that aggregate, gives transient OSS upload
failures a bounded resume pass, and then starts the downstream extractor with
the aggregate as its sole input.  No partial caption manifest is ever sent.
"""

from __future__ import print_function

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


COMPARE_ROOT = Path("/mnt/cloudstorfs/sjtu_home/weihan.chen/share/compare")
RUNTIME_ROOT = COMPARE_ROOT / "evaluation" / "benchmark-v1"
RAW_DIR = RUNTIME_ROOT / "captioner_raw"
CAPTION_OUT = RUNTIME_ROOT / "captioner-deepseek"
MANIFEST = RAW_DIR / "results.jsonl"
SUMMARY = RAW_DIR / "summary.json"
LOCK_PATH = CAPTION_OUT / ".captioner-chain-watcher.lock"
WATCH_LOG = CAPTION_OUT / "chain_watcher.log"
QWEN_LOG = RAW_DIR / "run_native_resume_after_chain.log"
CAPTION_LOG = CAPTION_OUT / "run_full_20260903.log"
AMI_OUT = RUNTIME_ROOT / "captioner-deepseek-ami"
AMI_MANIFEST = RUNTIME_ROOT / "captioner_ami_deepseek_input.jsonl"
AMI_META = RUNTIME_ROOT / "captioner_ami_deepseek_input.meta.json"
AMI_STAGE_INDEX = RUNTIME_ROOT / "staging.index.json"
AMI_LOG = AMI_OUT / "run_ami_refresh.log"
AMI_RETRY_ROOT = RUNTIME_ROOT / "captioner_ami_qwen_retries"
EXPECTED_COUNT = 650
AMI_COUNT = 350
POLL_SECONDS = 60
MAX_RESUME_ROUNDS = 2
MAX_AMI_RETRY_ROUNDS = 10
# A DeepSeek batch retries transport failures internally.  A second pass is
# still useful after an interrupted process (or a provider outage that
# affected a whole worker pool), but bound supervisor-level passes so a
# deterministic failure cannot keep the watcher alive forever.
MAX_CAPTION_RESUME_ROUNDS = 2

# Qwen item checkpoints only persist provider error text.  Keep the retry
# policy conservative: an AMI item is retried only when every recorded error
# is an explicit transport/provider-transient failure.  Deterministic model,
# authentication, input, and validation failures remain audit-only terminal
# rows.
NETWORK_ERROR_RE = re.compile(
    r"(?i)(?:"
    r"\bhttp\s*(?:408|409|425|429|5\d\d)\b"
    r"|timeout|timed\s*out|"
    r"remote\s*(?:disconnected|end\s+closed)"
    r"|connection\s*(?:aborted|reset|refused|closed|error|failure|timed\s*out)"
    r"|connection(?:aborted|reset|refused|closed|error)"
    r"|connect(?:timeout|error)"
    r"|broken\s+pipe|socket\s+error|"
    r"ssl(?:error)?|ssleof|unexpected[_\s-]*eof|\beof\b|"
    r"temporary\s+failure|name\s+(?:or\s+service|resolution)|"
    r"max\s+retries\s+exceeded|network\s+unreachable|no\s+route|"
    r"gaierror"
    r")"
)
HTTP_STATUS_RE = re.compile(r"(?i)\b(?:http|status|code)\s*[:=]?\s*(\d{3})\b")


def error_is_network(error):
    """Classify one persisted error with HTTP status taking precedence."""

    match = HTTP_STATUS_RE.search(str(error))
    if match:
        try:
            status = int(match.group(1))
        except (TypeError, ValueError):
            status = None
        if isinstance(status, int):
            return status in (408, 409, 425, 429) or status >= 500
    return NETWORK_ERROR_RE.search(str(error)) is not None


def errors_are_retryable(errors):
    """Return true only when every recorded failure is network/transient.

    The batch runner historically persisted retry attempts as one ``; ``-joined
    string.  Split that legacy form before classification so a network error
    cannot mask a deterministic error from another attempt.
    """

    if isinstance(errors, str):
        values = [errors]
    elif isinstance(errors, (list, tuple)):
        values = list(errors)
    else:
        values = []
    segments = []
    for value in values:
        segments.extend(
            part.strip()
            for part in str(value).split("; ")
            if part.strip()
        )
    return bool(segments) and all(error_is_network(error) for error in segments)


def deepseek_failure_retryable(record):
    """Return whether a failed AMI DeepSeek checkpoint should be retried."""

    if not isinstance(record, dict) or record.get("status") != "failed":
        return False
    error_type = record.get("error_type")
    if error_type == "network_error":
        # Require every persisted attempt to remain network/transient.  If a
        # future record omits messages entirely, the explicit type is the only
        # available evidence and is therefore trusted.
        errors = record.get("errors")
        if errors is None:
            errors = record.get("error")
        return True if errors is None else errors_are_retryable(errors)
    if error_type not in (None, "api_error"):
        return False
    errors = record.get("errors")
    if errors is None:
        errors = record.get("error")
    return errors_are_retryable(errors)

PYTHON = Path(
    "/mnt/cloudstorfs/sjtu_home/weihan.chen/share/sure-tagger-main/.runtime/"
    "campplus_sv_py311_torch280_cu128_v1/bin/python"
)
QWEN_RUNNER = COMPARE_ROOT / "eval" / "run_qwen_benchmark.py"
CAPTION_RUNNER = COMPARE_ROOT / "eval" / "run_captioner_deepseek_batch.py"
CAPTION_PROMPT = COMPARE_ROOT / "eval" / "captioner_deepseek_extraction_prompt.txt"
DEEPSEEK_API_FILE = COMPARE_ROOT / "api" / "deepseek-api.txt"
QWEN_MANIFEST = RUNTIME_ROOT / "runtime.manifest.jsonl"
BENCHMARK_ROOT = COMPARE_ROOT / "dataset" / "benchmark"


def log(message):
    CAPTION_OUT.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as stream:
        stream.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))
        stream.flush()


def process_lines():
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,args="], universal_newlines=True
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return output.splitlines()


def pids_for(needle):
    """Return PIDs whose parsed argv contains the requested executable script.

    A substring search over the complete ``ps`` line also matches wrapper
    shells whose ``-c`` command merely mentions the script.  That is enough to
    deadlock the handoff supervisor: the watcher waits for its own parent
    shell, while the parent waits for the watcher.  For Python script names,
    require an exact argv token and a Python interpreter earlier in argv.
    """

    result = []
    for line in process_lines():
        line = line.strip()
        if not line:
            continue
        try:
            tokens = shlex.split(line)
            pid = int(tokens[0])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid() or len(tokens) < 2:
            continue
        args = tokens[1:]
        if needle.endswith(".py"):
            # ``ps`` prints a shell's ``-c`` string without its original
            # quoting, so later words in that string can look like ordinary
            # argv items.  The first argv item is the real executable; only a
            # Python executable can own the benchmark script process we seek.
            if "python" not in Path(args[0]).name.lower():
                continue
            matched = False
            for index, token in enumerate(args):
                if Path(token).name != needle:
                    continue
                # The script must be an argv item of a Python invocation.
                # A ``bash -c`` command remains one token and therefore cannot
                # satisfy this check, even when its text names the script.
                prefix = args[:index]
                if any("python" in Path(value).name.lower() for value in prefix):
                    matched = True
                    break
            if not matched:
                continue
        elif needle not in line:
            continue
        result.append(pid)
    return result


def caption_pids(output_dir):
    """Find only the canonical full-benchmark captioner run.

    AMI is intentionally allowed to run as a separate subset in another
    output directory; treating any captioner runner as the full run would
    make the handoff silently skip the remaining datasets.
    """

    result = []
    expected_output = os.path.normpath(str(output_dir))
    if not os.path.isabs(expected_output):
        expected_output = os.path.normpath(str(COMPARE_ROOT / expected_output))
    for line in process_lines():
        line = line.strip()
        if "run_captioner_deepseek_batch.py" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        output_dir = None
        for index, token in enumerate(tokens[:-1]):
            if token == "--output-dir":
                output_dir = os.path.normpath(tokens[index + 1])
                if not os.path.isabs(output_dir):
                    output_dir = os.path.normpath(str(COMPARE_ROOT / output_dir))
                break
        if output_dir != expected_output:
            continue
        try:
            pid = int(line.split(None, 1)[0])
        except (ValueError, IndexError):
            continue
        if pid != os.getpid():
            result.append(pid)
    return result


def full_caption_pids():
    return caption_pids(CAPTION_OUT)


def ami_caption_pids():
    return caption_pids(AMI_OUT)


def valid_aggregate():
    """Return Qwen summary/counts only for a complete, internally keyed file."""

    if not SUMMARY.is_file() or not MANIFEST.is_file():
        return None
    try:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict) or summary.get("sample_count") != EXPECTED_COUNT:
        return None
    success = summary.get("success_count")
    failed = summary.get("failed_count")
    if not isinstance(success, int) or not isinstance(failed, int):
        return None
    if success + failed != EXPECTED_COUNT:
        return None
    if len(rows) != EXPECTED_COUNT or not all(isinstance(row, dict) for row in rows):
        return None
    ids = [row.get("sample_id") for row in rows]
    if len(set(ids)) != EXPECTED_COUNT or not all(isinstance(value, str) and value for value in ids):
        return None
    return {
        "success_count": success,
        "failed_count": failed,
        "summary_mtime": SUMMARY.stat().st_mtime,
    }


def qwen_command(manifest=None):
    # Match the active audio-only invocation.  Omitting --overwrite makes the
    # resume pass reuse valid successes and retry only failed/missing items.
    manifest = Path(manifest) if manifest is not None else QWEN_MANIFEST
    return [
        str(PYTHON),
        str(QWEN_RUNNER),
        "--native-oss",
        "--audio-only",
        "--workers",
        "5",
        "--max-inflight",
        "5",
        "--retries",
        "4",
        "--timeout",
        "1200",
        "--endpoint-ips",
        "8.152.159.24,8.140.217.18",
        "--manifest",
        str(manifest),
        "--benchmark-root",
        str(BENCHMARK_ROOT),
        "--output-dir",
        str(RAW_DIR),
    ]


def launch_qwen(round_number, manifest=None, log_path=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest) if manifest is not None else QWEN_MANIFEST
    log_path = Path(log_path) if log_path is not None else QWEN_LOG
    command = qwen_command(manifest)
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(
        "launched Qwen resume round=%s pid=%d workers=5 manifest=%s"
        % (round_number, process.pid, manifest.name)
    )
    return process


def wait_for_qwen(process=None):
    if process is not None:
        process.wait()
        return
    while pids_for("run_qwen_benchmark.py"):
        time.sleep(POLL_SECONDS)


def caption_command(use_overwrite):
    command = [
        str(PYTHON),
        str(CAPTION_RUNNER),
        "--input-manifest",
        str(MANIFEST),
        "--output-dir",
        str(CAPTION_OUT),
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
    ]
    if use_overwrite:
        command.append("--overwrite")
    return command


def _caption_config_matches_current():
    """Return whether existing caption checkpoints use the current request config."""

    config_path = CAPTION_OUT / "run_config.json"
    if not config_path.is_file():
        return False
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    current = current_deepseek_fingerprint()
    return bool(
        current
        and isinstance(previous, dict)
        and previous.get("config_fingerprint") == current
    )


def launch_caption(use_overwrite=None):
    CAPTION_OUT.mkdir(parents=True, exist_ok=True)
    item_dir = CAPTION_OUT / "items"
    item_count = len(list(item_dir.glob("*.json"))) if item_dir.is_dir() else 0
    if use_overwrite is None:
        # A fresh directory needs an initial materialization.  Existing
        # checkpoints are resumed unless their request fingerprint changed;
        # the batch runner otherwise (correctly) refuses to mix configurations.
        use_overwrite = item_count == 0 or not _caption_config_matches_current()
    command = caption_command(use_overwrite)
    with CAPTION_LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(
        "launched captioner DeepSeek pid=%d workers=10 manifest_rows=%d overwrite=%s"
        % (process.pid, EXPECTED_COUNT, str(use_overwrite).lower())
    )
    return process


def atomic_text(path, value):
    """Write a generated manifest without exposing a partial JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ami_runtime_rows():
    """Return the frozen runtime manifest rows belonging to the AMI subset."""

    stage = json.loads(AMI_STAGE_INDEX.read_text(encoding="utf-8"))
    stage_samples = {
        sample["sample_id"]: sample
        for sample in stage.get("samples", [])
        if "/t1-ami/" in sample.get("source_audio", "")
    }
    if len(stage_samples) != AMI_COUNT:
        raise ValueError(
            "expected %d AMI samples, got %d" % (AMI_COUNT, len(stage_samples))
        )

    rows = []
    seen = set()
    for line in QWEN_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        sample = value.get("sample") or {}
        sample_id = sample.get("sample_id")
        if sample_id not in stage_samples:
            continue
        if sample_id in seen:
            raise ValueError("duplicate AMI sample_id: %s" % sample_id)
        seen.add(sample_id)
        rows.append(value)
    if seen != set(stage_samples):
        raise ValueError("AMI runtime manifest coverage is %d/%d" % (len(seen), AMI_COUNT))
    return rows, stage_samples


def ami_qwen_status():
    """Count AMI Qwen items and return only clearly retryable rows.

    A failed checkpoint with a deterministic/provider-validation error is
    terminal for this chain.  It is retained in the AMI snapshot for audit,
    but must not cause another API request.  Missing or malformed checkpoints
    remain pending because they can be recovered after an interrupted batch.
    """

    rows, stage_samples = ami_runtime_rows()
    counts = {
        "success": 0,
        "failed": 0,
        "retryable": 0,
        "non_network": 0,
        "pending": 0,
    }
    retry_rows = []
    for row in rows:
        sample = row.get("sample") or {}
        sample_id = sample.get("sample_id")
        item_path = RAW_DIR / "items" / (sample_id + ".json")
        try:
            item = json.loads(item_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            counts["pending"] += 1
            retry_rows.append(row)
            continue
        caption = item.get("caption")
        valid_success = (
            item.get("status") == "success"
            and item.get("audio_sha256") == stage_samples[sample_id].get("source_sha256")
            and isinstance(caption, str)
            and bool(caption.strip())
        )
        if valid_success:
            counts["success"] += 1
            continue

        status = item.get("status")
        if status != "failed":
            # A malformed/in-progress checkpoint is not a terminal failure;
            # let a subsequent run materialize it instead of losing coverage.
            counts["pending"] += 1
            retry_rows.append(row)
            continue

        counts["failed"] += 1
        if errors_are_retryable(item.get("errors")):
            counts["retryable"] += 1
            retry_rows.append(row)
        else:
            counts["non_network"] += 1
    return counts, retry_rows


def full_qwen_status():
    """Classify full-benchmark checkpoints before a recovery round.

    Terminal non-network failures remain in the audit output, but only
    retryable or incomplete rows should cause a later Qwen invocation.  The
    full manifest is still used for a recovery invocation so its aggregate is
    rebuilt in the original 650-row order.
    """

    counts = {
        "success": 0,
        "failed": 0,
        "retryable": 0,
        "non_network": 0,
        "pending": 0,
    }
    try:
        rows = [
            json.loads(line)
            for line in QWEN_MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return None
    if len(rows) != EXPECTED_COUNT:
        return None
    retry_rows = []
    for row in rows:
        sample = row.get("sample") if isinstance(row, dict) else None
        sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            counts["pending"] += 1
            retry_rows.append(row)
            continue
        item_path = RAW_DIR / "items" / (sample_id + ".json")
        try:
            item = json.loads(item_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            counts["pending"] += 1
            retry_rows.append(row)
            continue
        if (
            item.get("status") == "success"
            and isinstance(item.get("caption"), str)
            and item.get("caption", "").strip()
        ):
            counts["success"] += 1
            continue
        if item.get("status") != "failed":
            counts["pending"] += 1
            retry_rows.append(row)
            continue
        counts["failed"] += 1
        if errors_are_retryable(item.get("errors")):
            counts["retryable"] += 1
            retry_rows.append(row)
        else:
            counts["non_network"] += 1
    return counts, retry_rows


def write_ami_retry_manifest(rows, round_number):
    """Materialize a deterministic audio-only manifest for one retry round."""

    AMI_RETRY_ROOT.mkdir(parents=True, exist_ok=True)
    path = AMI_RETRY_ROOT / ("round_%02d.manifest.jsonl" % round_number)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_text(path, content)
    return path


def ensure_ami_qwen_complete():
    """Retry AMI Qwen failures before allowing the full benchmark to proceed."""

    round_number = 0
    while True:
        # A currently active batch may still be writing item records.  Wait for
        # it before classifying records so a transient in-progress item is not
        # mistaken for a terminal failure.
        active = pids_for("run_qwen_benchmark.py")
        if active:
            log(
                "waiting for Qwen process(es) before AMI status: %s"
                % ",".join(map(str, active))
            )
            wait_for_qwen()
            continue

        counts, retry_rows = ami_qwen_status()
        log(
            "AMI Qwen status success=%d failed=%d retryable=%d "
            "non_network=%d pending=%d"
            % (
                counts["success"],
                counts["failed"],
                counts["retryable"],
                counts["non_network"],
                counts["pending"],
            )
        )
        if counts["retryable"] == 0 and counts["pending"] == 0:
            return counts
        round_number += 1
        if round_number > MAX_AMI_RETRY_ROUNDS:
            log(
                "AMI Qwen remains incomplete after %d retry rounds; refusing full run"
                % MAX_AMI_RETRY_ROUNDS
            )
            return None
        manifest = write_ami_retry_manifest(retry_rows, round_number)
        log_path = RAW_DIR / ("run_native_ami_retry_round_%02d.log" % round_number)
        process = launch_qwen(
            "ami-%d" % round_number, manifest=manifest, log_path=log_path
        )
        wait_for_qwen(process)


def build_ami_snapshot():
    """Rebuild the AMI caption manifest from the final Qwen item records."""

    stage = json.loads(AMI_STAGE_INDEX.read_text(encoding="utf-8"))
    stage_samples = {
        sample["sample_id"]: sample
        for sample in stage.get("samples", [])
        if "/t1-ami/" in sample.get("source_audio", "")
    }
    if len(stage_samples) != AMI_COUNT:
        raise ValueError("expected %d AMI samples, got %d" % (AMI_COUNT, len(stage_samples)))

    rows = []
    seen = set()
    for line in QWEN_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        sample = value.get("sample") or {}
        sample_id = sample.get("sample_id")
        if sample_id not in stage_samples:
            continue
        if sample_id in seen:
            raise ValueError("duplicate AMI sample_id: %s" % sample_id)
        seen.add(sample_id)
        item_path = RAW_DIR / "items" / (sample_id + ".json")
        item = json.loads(item_path.read_text(encoding="utf-8"))
        if item.get("audio_sha256") != stage_samples[sample_id].get("source_sha256"):
            raise ValueError("Qwen audio hash mismatch for %s" % sample_id)
        if item.get("status") == "success":
            caption = item.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("successful Qwen item has empty caption: %s" % sample_id)
            rows.append({"caption": caption, "sample_id": sample_id, "status": "success"})
        elif item.get("status") == "failed":
            rows.append(
                {
                    "errors": item.get("errors") or ["Qwen upstream failed"],
                    "sample_id": sample_id,
                    "status": "failed",
                }
            )
        else:
            raise ValueError("non-terminal Qwen status for %s" % sample_id)

    if len(rows) != AMI_COUNT or seen != set(stage_samples):
        raise ValueError("AMI Qwen coverage is %d/%d" % (len(rows), AMI_COUNT))
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_text(AMI_MANIFEST, content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    counts = {
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
    }
    meta = {
        "subset": "t1-ami",
        "sample_count": AMI_COUNT,
        "status_counts": counts,
        "input_manifest": str(AMI_MANIFEST.resolve()),
        "input_manifest_sha256": digest,
        "runtime_manifest": str(QWEN_MANIFEST.resolve()),
        "runtime_manifest_sha256": hashlib.sha256(QWEN_MANIFEST.read_bytes()).hexdigest(),
        "staging_index": str(AMI_STAGE_INDEX.resolve()),
        "staging_index_sha256": hashlib.sha256(AMI_STAGE_INDEX.read_bytes()).hexdigest(),
        "input_contract": "one_upstream_text_value_per_request",
        "qwen_snapshot_note": (
            "Built after the final Qwen aggregate; failed upstream rows are retained "
            "for audit and are never sent to DeepSeek."
        ),
    }
    atomic_text(AMI_META, json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return digest, counts


def _deepseek_output_complete(
    output_dir,
    expected_count,
    expected_digest,
    expected_fingerprint,
    expected_sample_ids=None,
):
    """Validate a DeepSeek aggregate and classify retryable failures.

    ``run_deepseek_batch`` writes one status row for every input row, including
    deterministic upstream/input failures.  Those failures are a valid
    terminal outcome for the chain.  Only incomplete rows and failures whose
    *entire* recorded history is network/transient require another request.
    """

    if not isinstance(expected_digest, str) or not expected_digest:
        return False
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        return False

    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    status_path = output_dir / "status.jsonl"
    if not summary_path.is_file() or not status_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in status_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict) or summary.get("sample_count") != expected_count:
        return False
    if summary.get("input_manifest_sha256") != expected_digest:
        return False
    if summary.get("config_fingerprint") != expected_fingerprint:
        return False
    if len(rows) != expected_count:
        return False
    if any(not isinstance(row, dict) for row in rows):
        return False
    sample_ids = [row.get("sample_id") for row in rows]
    if not all(
        isinstance(sample_id, str) and sample_id for sample_id in sample_ids
    ):
        return False
    if len(set(sample_ids)) != expected_count:
        return False
    if expected_sample_ids is not None:
        try:
            expected_ids = list(expected_sample_ids)
        except TypeError:
            return False
        if len(expected_ids) != expected_count or len(set(expected_ids)) != expected_count:
            return False
        if set(sample_ids) != set(expected_ids):
            return False
    success_count = sum(row.get("status") == "success" for row in rows)
    failed_count = sum(row.get("status") == "failed" for row in rows)
    if success_count + failed_count != expected_count:
        return False
    if (
        summary.get("success_count") != success_count
        or summary.get("failed_count") != failed_count
    ):
        return False
    if any(
        row.get("status") == "failed" and deepseek_failure_retryable(row)
        for row in rows
    ):
        return False
    return True


def ami_output_complete(expected_digest, expected_fingerprint):
    """Check whether AMI output matches the current input and request config.

    A status file with one row per sample is not sufficient: an earlier run
    can contain terminal upstream failures while still looking complete by
    row count.  Tie the completion decision to the freshly rebuilt manifest,
    the current DeepSeek request fingerprint, and complete per-sample status.
    Deterministic failures remain terminal audit rows; only pending rows and
    retryable network failures require another batch pass.
    """

    expected_ids = _manifest_sample_ids(AMI_MANIFEST, AMI_COUNT)
    if expected_ids is None:
        return False
    return _deepseek_output_complete(
        AMI_OUT,
        AMI_COUNT,
        expected_digest,
        expected_fingerprint,
        expected_sample_ids=expected_ids,
    )


def _manifest_sample_ids(path, expected_count):
    """Read the ordered sample IDs from a JSONL input manifest."""

    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, TypeError):
        return None
    if len(rows) != expected_count:
        return None
    ids = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            sample = row.get("sample")
            sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            return None
        ids.append(sample_id)
    if len(set(ids)) != expected_count:
        return None
    return ids


def full_output_complete(expected_digest, expected_fingerprint):
    """Return whether the full captioner DeepSeek output is terminal/current."""

    expected_ids = _manifest_sample_ids(MANIFEST, EXPECTED_COUNT)
    if expected_ids is None:
        return False
    return _deepseek_output_complete(
        CAPTION_OUT,
        EXPECTED_COUNT,
        expected_digest,
        expected_fingerprint,
        expected_sample_ids=expected_ids,
    )


def full_manifest_digest():
    """Return the current full captioner input digest, or ``None`` if absent."""

    try:
        return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None


def wait_for_full_caption():
    """Wait for an already-running full captioner process to finish."""

    pids = full_caption_pids()
    if not pids:
        return
    log(
        "waiting for existing full captioner DeepSeek pid(s)=%s"
        % ",".join(map(str, pids))
    )
    previous = tuple(pids)
    while pids:
        time.sleep(POLL_SECONDS)
        pids = full_caption_pids()
        current = tuple(pids)
        if current != previous and pids:
            log(
                "full captioner DeepSeek pid set changed to %s; continuing to wait"
                % ",".join(map(str, pids))
            )
        previous = current
    log("existing full captioner DeepSeek process(es) exited")


def current_deepseek_fingerprint():
    """Return the captioner request fingerprint used by the batch runner."""

    try:
        import sys

        eval_dir = str(COMPARE_ROOT / "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        import extract_captioner_deepseek_once as captioner_once

        config = captioner_once.read_api_file(DEEPSEEK_API_FILE)
        prompt_hash = hashlib.sha256(CAPTION_PROMPT.read_bytes()).hexdigest()
        value = {
            "system": "captioner",
            "model": config["model"],
            "endpoint": captioner_once.endpoint(config["base_url"]),
            "prompt_sha256": prompt_hash,
            "thinking": "enabled",
            "max_tokens": 20000,
            "timeout": 300,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        return None


def refresh_ami_deepseek():
    """Resume AMI DeepSeek only when final Qwen output adds new captions."""

    while ami_caption_pids():
        time.sleep(POLL_SECONDS)
    digest, counts = build_ami_snapshot()
    current_fingerprint = current_deepseek_fingerprint()
    if ami_output_complete(digest, current_fingerprint):
        log("AMI output matches current snapshot/config; no duplicate DeepSeek requests")
        return
    item_dir = AMI_OUT / "items"
    use_overwrite = not item_dir.is_dir() or not any(item_dir.glob("*.json"))
    previous_config_path = AMI_OUT / "run_config.json"
    if previous_config_path.is_file():
        try:
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_config = None
        current_fingerprint = current_deepseek_fingerprint()
        if (
            not isinstance(previous_config, dict)
            or not current_fingerprint
            or previous_config.get("config_fingerprint") != current_fingerprint
        ):
            # Existing checkpoints were produced with a different prompt,
            # model, endpoint, or request mode and must not be mixed in.
            use_overwrite = True
    command = [
        str(PYTHON),
        str(CAPTION_RUNNER),
        "--input-manifest",
        str(AMI_MANIFEST),
        "--output-dir",
        str(AMI_OUT),
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
    ]
    if use_overwrite:
        command.append("--overwrite")
    AMI_OUT.mkdir(parents=True, exist_ok=True)
    with AMI_LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(
        "AMI manifest changed success=%d failed=%d; started DeepSeek pid=%d overwrite=%s"
        % (counts["success"], counts["failed"], process.pid, str(use_overwrite).lower())
    )
    code = process.wait()
    log("AMI DeepSeek refresh pid=%d exited code=%d" % (process.pid, code))


def main():
    CAPTION_OUT.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        return 0

    log("chain watcher started pid=%d" % os.getpid())
    try:
        # Finish the AMI subset first.  This also waits for an already-running
        # Qwen batch and then retries every non-success AMI item before the
        # 650-row aggregate or any downstream DeepSeek request is considered.
        if ensure_ami_qwen_complete() is None:
            return 2

        # The AMI Qwen snapshot is now complete; refresh its downstream
        # DeepSeek results before spending time on the 650-row Qwen run.
        refresh_ami_deepseek()

        resume_round = 0
        while True:
            aggregate = valid_aggregate()
            if aggregate is not None:
                log(
                    "Qwen aggregate complete success=%d failed=%d"
                    % (aggregate["success_count"], aggregate["failed_count"])
                )
                if aggregate["failed_count"] == 0 or resume_round >= MAX_RESUME_ROUNDS:
                    break
                status = full_qwen_status()
                if status is not None:
                    counts, retry_rows = status
                    log(
                        "Qwen full status success=%d failed=%d retryable=%d "
                        "non_network=%d pending=%d"
                        % (
                            counts["success"],
                            counts["failed"],
                            counts["retryable"],
                            counts["non_network"],
                            counts["pending"],
                        )
                    )
                    if counts["retryable"] == 0 and counts["pending"] == 0:
                        log(
                            "Qwen failures are terminal non-network errors; "
                            "skipping recovery requests"
                        )
                        break
                resume_round += 1
                log(
                    "retrying Qwen failed rows round=%d/%d"
                    % (resume_round, MAX_RESUME_ROUNDS)
                )
                process = launch_qwen(resume_round)
                wait_for_qwen(process)
                continue

            if pids_for("run_qwen_benchmark.py"):
                time.sleep(POLL_SECONDS)
                continue
            if resume_round >= MAX_RESUME_ROUNDS:
                log("Qwen exited without a complete 650-row aggregate; refusing downstream launch")
                return 2
            resume_round += 1
            log("Qwen aggregate not ready; launching recovery round=%d" % resume_round)
            process = launch_qwen(resume_round)
            wait_for_qwen(process)

        # A supervisor can be started while the canonical full run is already
        # active.  Wait for that process and inspect the aggregate it leaves
        # behind; returning immediately here used to strand the handoff with
        # no completion check or network-only recovery pass.
        caption_round = 0
        while True:
            if full_caption_pids():
                wait_for_full_caption()

            digest = full_manifest_digest()
            fingerprint = current_deepseek_fingerprint()
            if digest and fingerprint and full_output_complete(digest, fingerprint):
                log(
                    "full captioner DeepSeek output is complete for current "
                    "manifest/config; no duplicate requests"
                )
                return 0

            if not digest or not fingerprint:
                log(
                    "full captioner output cannot be checked: manifest/config "
                    "fingerprint unavailable; refusing launch"
                )
                return 2
            if caption_round >= MAX_CAPTION_RESUME_ROUNDS:
                log(
                    "full captioner output remains incomplete after %d "
                    "recovery rounds; refusing further requests"
                    % MAX_CAPTION_RESUME_ROUNDS
                )
                return 2

            caption_round += 1
            item_dir = CAPTION_OUT / "items"
            has_checkpoints = item_dir.is_dir() and any(item_dir.glob("*.json"))
            use_overwrite = not has_checkpoints or not _caption_config_matches_current()
            log(
                "full captioner output incomplete; launching recovery round=%d/%d "
                "overwrite=%s (only pending/network checkpoints are eligible)"
                % (
                    caption_round,
                    MAX_CAPTION_RESUME_ROUNDS,
                    str(use_overwrite).lower(),
                )
            )
            process = launch_caption(use_overwrite=use_overwrite)
            code = process.wait()
            log(
                "full captioner DeepSeek recovery pid=%d exited code=%d"
                % (process.pid, code)
            )
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
