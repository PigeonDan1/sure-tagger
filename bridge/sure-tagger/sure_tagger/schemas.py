import time


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def require_keys(obj, keys, name):
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError("%s missing required keys: %s" % (name, ", ".join(missing)))


def validate_manifest_record(record):
    require_keys(record, ["corpus", "sample"], "manifest record")
    sample = record["sample"]
    require_keys(sample, ["sample_id", "audio", "text", "native_metadata"], "sample")
    require_keys(sample["audio"], ["path"], "sample.audio")
    require_keys(sample["text"], ["transcript"], "sample.text")
    return True


def make_tag(value, confidence, method, tool_version, reliability, details=None):
    tag = {
        "value": value,
        "confidence": confidence,
        "method": method,
        "tool_version": tool_version,
        "reliability": reliability,
    }
    if details is not None:
        tag["details"] = details
    return tag


def make_error(sample_id, stage, error_type, message, source_path=None):
    return {
        "sample_id": sample_id,
        "stage": stage,
        "error_type": error_type,
        "message": str(message),
        "source_path": source_path or "",
    }
