"""OpenASR FireRed transcription plus DiariZen speaker timeline adapter."""

from __future__ import print_function

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_EXECUTABLE = (
    PROJECT_ROOT / ".runtime" / "openasr" / "0.1.40" / "openasr"
)
DEFAULT_DIARIZEN_PACK = PROJECT_ROOT / "models" / (
    "diarizen-large-s80-v2/diarizen-large-s80-v2-fp16.oasr"
)
DEFAULT_REDIMNET_PACK = PROJECT_ROOT / "models" / (
    "redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr"
)
DEFAULT_FIRERED_ASR_PACK = PROJECT_ROOT / "models" / (
    "firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr"
)
DEFAULT_FORCED_ALIGNER_PACK = PROJECT_ROOT / "models" / (
    "qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr"
)

MODEL_ID = "OpenASR/diarizen-large-s80-v2"
MODEL_VERSION = "%s@655b0132a5539f55d5ad3fffc88a2206ab0b7238" % MODEL_ID
MODEL_PACK_SHA256 = (
    "1d4b2c2e9fe3a3f6eebfc8f2280bca04dcbc6f9b14e4d9cbccb0fdca40e88311"
)
MODEL_PACK_SIZE_BYTES = 139425120
REDIMNET_MODEL_ID = "OpenASR/redimnet2-b6-cn"
REDIMNET_MODEL_VERSION = "%s@4c8deac5d2f81fabe4a16a2dcd6184c3e47c02dd" % (
    REDIMNET_MODEL_ID,
)
REDIMNET_PACK_SHA256 = (
    "ef408d57e9f41d3e430e5c6aa9483bacaebf9531cc224217d395d0cfb37cf6d1"
)
REDIMNET_PACK_SIZE_BYTES = 27775520
FIRERED_ASR_MODEL_ID = "OpenASR/firered-aed-l-v2"
FIRERED_ASR_MODEL_VERSION = (
    "%s@878a7be29104370c25938af686f80602c3db73d2" % FIRERED_ASR_MODEL_ID
)
FIRERED_ASR_PACK_SHA256 = (
    "b24853f159ed85192b527601190bf31b904e3f62957b8a26418a78c07a5b5688"
)
FIRERED_ASR_PACK_SIZE_BYTES = 1062435296
FORCED_ALIGNER_MODEL_ID = "OpenASR/qwen3-forced-aligner-0.6b"
FORCED_ALIGNER_MODEL_VERSION = (
    "%s@3a82628cf7036c2ad0128f2d35bd1cced09b3100"
    % FORCED_ALIGNER_MODEL_ID
)
FORCED_ALIGNER_PACK_SHA256 = (
    "5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b"
)
FORCED_ALIGNER_PACK_SIZE_BYTES = 765395744
OPENASR_VERSION = "0.1.40"
OPENASR_EXECUTABLE_SHA256 = (
    "98d973d0d72b47198ecf17e4c00b2710d84c351b77d3c8024a7256bcea59a1a5"
)
OPENASR_EXECUTABLE_SIZE_BYTES = 42268384
TOOL_VERSION = "openasr_diarizen_firered.v0.1.40.3"
MODEL_SLOT_KEY = "openasr_transcribe_diarize"
_RUNTIME_VERIFICATION_CACHE = {}
_RUNTIME_VERIFICATION_LOCK = threading.Lock()

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_HEADER_RE = re.compile(
    r"(?<!\S)stage=batch\s+turns=(\d+)\s+speakers=(\d+)(?!\S)"
)
_TURN_RE = re.compile(
    r"(?<!\S)turn\s+start=(%s)\s+end=(%s)\s+speaker=(\S+)"
    r"\s+overlap=(true|false|1|0)(?!\S)" % (_NUMBER, _NUMBER),
    re.IGNORECASE,
)


class DiariZenTimelineError(RuntimeError):
    """Raised when OpenASR cannot produce a valid diarization result."""


class DiariZenTimelineConfig:
    def __init__(
        self,
        runtime_executable=DEFAULT_RUNTIME_EXECUTABLE,
        diarizen_pack=DEFAULT_DIARIZEN_PACK,
        redimnet_pack=DEFAULT_REDIMNET_PACK,
        firered_asr_pack=DEFAULT_FIRERED_ASR_PACK,
        forced_aligner_pack=DEFAULT_FORCED_ALIGNER_PACK,
        timeout_sec=900,
        openasr_version=OPENASR_VERSION,
        runtime_executable_sha256=OPENASR_EXECUTABLE_SHA256,
    ):
        self.runtime_executable = str(runtime_executable or "")
        self.diarizen_pack = str(diarizen_pack or "")
        self.redimnet_pack = str(redimnet_pack or "")
        self.firered_asr_pack = str(firered_asr_pack or "")
        self.forced_aligner_pack = str(forced_aligner_pack or "")
        self.timeout_sec = int(timeout_sec)
        self.openasr_version = str(openasr_version)
        self.runtime_executable_sha256 = str(runtime_executable_sha256)
        for name in (
            "runtime_executable",
            "diarizen_pack",
            "redimnet_pack",
            "firered_asr_pack",
            "forced_aligner_pack",
        ):
            if not getattr(self, name):
                raise ValueError("DiariZen %s is required" % name)
        if self.timeout_sec <= 0:
            raise ValueError("DiariZen timeout_sec must be positive")
        if self.openasr_version != OPENASR_VERSION:
            raise ValueError(
                "DiariZen OpenASR version must be pinned to %s"
                % OPENASR_VERSION
            )
        if self.runtime_executable_sha256 != OPENASR_EXECUTABLE_SHA256:
            raise ValueError(
                "DiariZen OpenASR executable SHA-256 must be pinned to %s"
                % OPENASR_EXECUTABLE_SHA256
            )

    @property
    def model_pack(self):
        """Compatibility alias for early deployment callers."""

        return self.diarizen_pack

    def to_record(self):
        return {
            "runtime_executable": self.runtime_executable,
            "diarizen_pack": self.diarizen_pack,
            "redimnet_pack": self.redimnet_pack,
            "firered_asr_pack": self.firered_asr_pack,
            "forced_aligner_pack": self.forced_aligner_pack,
            "timeout_sec": self.timeout_sec,
            "openasr_version": self.openasr_version,
            "runtime_executable_sha256": self.runtime_executable_sha256,
        }


class DiariZenTimelineClient:
    def __init__(self, config=None):
        self.config = config or DiariZenTimelineConfig()

    def transcribe_and_diarize(self, audio_path, context=None):
        executable = _resolve_executable(self.config.runtime_executable)
        audio_path = _required_file(audio_path, "audio")
        diarizen_pack = _required_file(
            self.config.diarizen_pack, "DiariZen model pack"
        )
        redimnet_pack = _required_file(
            self.config.redimnet_pack, "ReDimNet model pack"
        )
        firered_pack = _required_file(
            self.config.firered_asr_pack, "FireRed ASR model pack"
        )
        forced_aligner_pack = _required_file(
            self.config.forced_aligner_pack,
            "Qwen3 forced aligner model pack",
        )
        runtime_identity = _verify_openasr_runtime(
            executable, self.config.runtime_executable_sha256
        )
        argv = [
            executable,
            "transcribe",
            str(audio_path),
            "--model-pack",
            str(firered_pack),
            "--diarize",
            "--offline",
            "--no-punctuate",
            "--format",
            "json",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "OPENASR_DIARIZEN_PACK": str(diarizen_pack),
                "OPENASR_REDIMNET_PACK": str(redimnet_pack),
                "OPENASR_FORCED_ALIGNER_PACK": str(forced_aligner_pack),
                "OPENASR_DIARIZE_DEBUG": "1",
            }
        )
        try:
            limiter = _model_limiter(context)
            with limiter:
                completed = subprocess.run(
                    argv,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=self.config.timeout_sec,
                    env=environment,
                )
        except subprocess.TimeoutExpired as exc:
            raise DiariZenTimelineError(
                "OpenASR timed out after %s seconds" % self.config.timeout_sec
            ) from exc
        except OSError as exc:
            raise DiariZenTimelineError(
                "failed to start OpenASR runtime: %s" % exc
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            raise DiariZenTimelineError(
                "OpenASR exited with status %s%s"
                % (
                    completed.returncode,
                    ": %s" % detail if detail else "",
                )
            )
        return normalize_result(
            completed.stdout,
            completed.stderr,
            argv,
            runtime_identity=runtime_identity,
        )

    def diarize(self, audio_path, context=None):
        return self.transcribe_and_diarize(audio_path, context=context)


def diarize(audio_path, config, context=None, client=None):
    client = client or DiariZenTimelineClient(config)
    method = getattr(client, "transcribe_and_diarize", None)
    if callable(method):
        return method(audio_path, context=context)
    return client.diarize(audio_path, context=context)


def normalize_result(stdout, stderr, argv=None, runtime_identity=None):
    """Parse ASR and diarization independently after a successful command."""

    runtime = {
        "engine": "OpenASR",
        "engine_version": OPENASR_VERSION,
        "command": list(argv or []),
        "executable_verified": runtime_identity is not None,
    }
    if runtime_identity is not None:
        runtime.update(runtime_identity)
    try:
        text = _parse_stdout_text(stdout)
        runtime["asr_status"] = "usable" if text else "empty"
    except DiariZenTimelineError as exc:
        text = None
        runtime["asr_status"] = "invalid"
        runtime["asr_error"] = str(exc)
    try:
        segments, turn_count, speaker_count = _parse_debug_segments(stderr)
        runtime["timeline_status"] = "usable"
        runtime["debug_turn_count"] = turn_count
        runtime["debug_speaker_count"] = speaker_count
    except DiariZenTimelineError as exc:
        segments = None
        runtime["timeline_status"] = "invalid"
        runtime["timeline_error"] = str(exc)
    return {"text": text, "segments": segments, "runtime": runtime}


def _parse_stdout_text(stdout):
    try:
        raw = json.loads(
            stdout,
            parse_constant=lambda value: _raise_nonfinite_json(value),
        )
    except (TypeError, ValueError) as exc:
        raise DiariZenTimelineError(
            "OpenASR stdout must be one finite JSON object"
        ) from exc
    if not isinstance(raw, dict):
        raise DiariZenTimelineError("OpenASR stdout JSON must be an object")
    text = raw.get("text")
    if not isinstance(text, str):
        raise DiariZenTimelineError("OpenASR stdout text must be a string")
    return text.strip()


def _parse_debug_segments(stderr):
    headers = _HEADER_RE.findall(stderr or "")
    if len(headers) != 1:
        raise DiariZenTimelineError(
            "OpenASR debug trace must contain exactly one batch header"
        )
    expected_turns, expected_speakers = (
        int(headers[0][0]),
        int(headers[0][1]),
    )
    matches = list(_TURN_RE.finditer(stderr or ""))
    if len(matches) != expected_turns:
        raise DiariZenTimelineError(
            "OpenASR debug trace declares %s turns but contains %s"
            % (expected_turns, len(matches))
        )
    segments = []
    for index, match in enumerate(matches):
        start = float(match.group(1))
        end = float(match.group(2))
        speaker = match.group(3).strip()
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise DiariZenTimelineError(
                "OpenASR turn %s has invalid time bounds" % index
            )
        if not speaker:
            raise DiariZenTimelineError(
                "OpenASR turn %s has an empty speaker" % index
            )
        segments.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "speaker_id": speaker,
                "overlap": match.group(4).lower() in ("true", "1"),
            }
        )
    speaker_count = len({item["speaker_id"] for item in segments})
    if speaker_count != expected_speakers:
        raise DiariZenTimelineError(
            "OpenASR debug trace declares %s speakers but contains %s"
            % (expected_speakers, speaker_count)
        )
    if expected_turns == 0 and expected_speakers != 0:
        raise DiariZenTimelineError(
            "OpenASR zero-turn trace must declare zero speakers"
        )
    for index, segment in enumerate(segments):
        if not segment["overlap"]:
            continue
        has_other_speaker_overlap = any(
            other["speaker_id"] != segment["speaker_id"]
            and min(segment["end_sec"], other["end_sec"])
            > max(segment["start_sec"], other["start_sec"])
            for other in segments
        )
        if not has_other_speaker_overlap:
            raise DiariZenTimelineError(
                "OpenASR turn %s has overlap=true without an overlapping "
                "different-speaker turn" % index
            )
    segments.sort(
        key=lambda item: (
            item["start_sec"],
            item["end_sec"],
            item["speaker_id"],
            item["overlap"],
        )
    )
    return segments, expected_turns, expected_speakers


def _resolve_executable(value):
    executable = str(value or "").strip()
    if not executable:
        raise DiariZenTimelineError("OpenASR runtime executable is not configured")
    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved
    candidate = Path(executable).expanduser()
    if not candidate.is_file():
        raise DiariZenTimelineError(
            "OpenASR runtime executable is missing: %s" % executable
        )
    if not os.access(str(candidate), os.X_OK):
        raise DiariZenTimelineError(
            "OpenASR runtime executable is not executable: %s" % candidate
        )
    return str(candidate)


def _verify_openasr_runtime(executable, expected_sha256):
    path = Path(executable).resolve()
    stat = path.stat()
    cache_key = (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        str(expected_sha256),
    )
    with _RUNTIME_VERIFICATION_LOCK:
        cached = _RUNTIME_VERIFICATION_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
        if stat.st_size != OPENASR_EXECUTABLE_SIZE_BYTES:
            raise DiariZenTimelineError(
                "OpenASR runtime size mismatch: expected %s bytes, got %s"
                % (OPENASR_EXECUTABLE_SIZE_BYTES, stat.st_size)
            )
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != str(expected_sha256):
            raise DiariZenTimelineError(
                "OpenASR runtime SHA-256 mismatch: expected %s, got %s"
                % (expected_sha256, actual_sha256)
            )
        try:
            completed = subprocess.run(
                [str(path), "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DiariZenTimelineError(
                "failed to verify OpenASR runtime version: %s" % exc
            ) from exc
        version_output = (completed.stdout or "").strip()
        expected_output = "openasr %s" % OPENASR_VERSION
        if completed.returncode != 0 or version_output != expected_output:
            raise DiariZenTimelineError(
                "OpenASR runtime version mismatch: expected %r, got %r (status %s)"
                % (expected_output, version_output, completed.returncode)
            )
        identity = {
            "engine_version": OPENASR_VERSION,
            "executable_sha256": actual_sha256,
            "executable_size_bytes": stat.st_size,
            "executable_version_output": version_output,
            "executable_verified": True,
        }
        _RUNTIME_VERIFICATION_CACHE[cache_key] = dict(identity)
        return identity


def _required_file(value, label):
    path = Path(str(value or "")).expanduser()
    if not path.is_file():
        raise DiariZenTimelineError("%s is missing: %s" % (label, path))
    return path.resolve()


def _model_limiter(context):
    if context is None:
        return threading.BoundedSemaphore(1)
    lock = context.setdefault("_subprocess_workers_lock", threading.Lock())
    with lock:
        limiters = context.setdefault("_external_model_semaphores", {})
        if MODEL_SLOT_KEY not in limiters:
            slots_by_tool = context.get("_subprocess_worker_slots", {})
            slots = max(1, int(slots_by_tool.get(MODEL_SLOT_KEY, 1)))
            limiters[MODEL_SLOT_KEY] = threading.BoundedSemaphore(slots)
        return limiters[MODEL_SLOT_KEY]


def _raise_nonfinite_json(value):
    raise ValueError("non-finite JSON constant: %s" % value)
