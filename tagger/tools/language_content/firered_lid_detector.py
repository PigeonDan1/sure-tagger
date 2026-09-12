"""FireRed LID spoken-language detector.

FireRed LID (from FireRedASR2S) recognizes 100+ languages and 20+ Chinese
dialects/accents from audio. The detected ISO language code is public; model
confidence and duration stay internal evidence.
"""

from pathlib import Path
import sys

from tagger.local_config import (
    FIRERED_LID_MODEL_DIR,
    FIRERED_LID_PYTHON,
    FIRERED_LID_REPO_DIR,
)
from tagger.tools.base import TOOL_VERSION, ToolResult
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "firered_lid_detector"
METHOD = "FireRed LID"
ROUND_DIGITS = 6


class FireRedLidError(RuntimeError):
    """Raised when FireRed LID cannot produce a valid language result."""


class FireRedLidConfig:
    """Fixed FireRed LID inference configuration."""

    def __init__(
        self,
        model_dir=None,
        repo_dir=None,
        use_gpu=False,
        use_half=False,
        subprocess_python=None,
    ):
        configured_model_dir = getattr(FIRERED_LID_MODEL_DIR, "strip", lambda: "")()
        configured_repo_dir = getattr(FIRERED_LID_REPO_DIR, "strip", lambda: "")()
        configured_python = getattr(FIRERED_LID_PYTHON, "strip", lambda: "")()
        self.model_dir = str(model_dir or configured_model_dir)
        self.repo_dir = str(repo_dir or configured_repo_dir)
        self.use_gpu = bool(use_gpu)
        self.use_half = bool(use_half)
        self.subprocess_python = (
            configured_python if subprocess_python is None else subprocess_python
        )

    def cache_key(self):
        return (
            self.model_dir,
            self.repo_dir,
            self.use_gpu,
            self.use_half,
            self.subprocess_python,
        )

    def to_record(self):
        record = _subprocess_config(self)
        record.update(
            {
                "public_mapping": {
                    "language": "ISO language or zh-<region> code from FireRed LID",
                },
                "subprocess_python": self.subprocess_python,
            }
        )
        return record


class FireRedLidClient:
    """Thin adapter around the official fireredasr2s FireRed LID Python API."""

    def __init__(self, config=None):
        self.config = config or FireRedLidConfig()
        self._lid = None

    def detect_language(self, audio_path, context=None):
        lid = self._get_lid(context)
        try:
            results = lid.process(["sample"], [str(audio_path)])
        except Exception as exc:  # noqa: BLE001 - normalized to tool failure.
            raise FireRedLidError("FireRed LID inference failed") from exc
        if not isinstance(results, list) or len(results) != 1:
            raise FireRedLidError("FireRed LID returned a non-single result list")
        return results[0]

    def _get_lid(self, context=None):
        if context is None:
            if self._lid is None:
                self._lid = self._load_lid()
            return self._lid

        cache = context.setdefault("firered_lid_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_lid()
        return cache[key]

    def _load_lid(self):
        if not self.config.model_dir:
            raise FireRedLidError(
                "FireRed LID model dir is not configured; set "
                "tagger/local_config.py:FIRERED_LID_MODEL_DIR"
            )

        model_dir = Path(self.config.model_dir)
        missing_files = [
            str(model_dir / name)
            for name in ("cmvn.ark", "model.pth.tar", "dict.txt")
            if not (model_dir / name).exists()
        ]
        if missing_files:
            raise FireRedLidError(
                "FireRed LID model files are missing: %s" % ", ".join(missing_files)
            )

        repo_dir = Path(self.config.repo_dir)
        if not repo_dir.is_dir():
            raise FireRedLidError(
                "FireRed LID repo dir is missing: %s" % repo_dir
            )

        repo_path = str(repo_dir.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from fireredlid.lid import (
                FireRedLid,
                FireRedLidConfig as UpstreamConfig,
            )
        except ImportError as exc:
            raise FireRedLidError(
                "fireredlid is not importable; install FireRedASR2S repo first"
            ) from exc

        upstream_config = UpstreamConfig(
            use_gpu=self.config.use_gpu,
            use_half=self.config.use_half,
        )
        return FireRedLid.from_pretrained(str(model_dir), upstream_config)


class FireRedLidSubprocessClient:
    """Adapter that runs FireRed LID in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or FireRedLidConfig()

    def detect_language(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "firered_lid_detect",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
        )
        return result["output"]


def run(audio_path, context=None, config=None, client=None, **_kwargs):
    config = config or FireRedLidConfig()
    client = client or _default_client(config)
    cache = None
    cache_key = None
    if context is not None:
        cache = context.setdefault("firered_lid_result_by_audio", {})
        cache_key = (
            str(Path(audio_path).expanduser().resolve()),
            config.cache_key(),
        )
    if cache is not None and cache_key in cache:
        raw_output = cache[cache_key]
    else:
        raw_output = client.detect_language(audio_path, context)
        if cache is not None:
            cache[cache_key] = raw_output
    summary = validate_lid_output(raw_output)
    evidence = {
        "config": config.to_record(),
        "confidence": summary["confidence"],
        "dur_s": summary.get("dur_s"),
        "rtf": summary.get("rtf"),
    }
    if summary.get("raw_lang") != summary["lang"]:
        evidence["raw_language"] = summary.get("raw_lang")
    return ToolResult(
        tag_path="language_content.language",
        value=summary["lang"],
        tool_name=TOOL_NAME,
        method=METHOD,
        status="estimated",
        confidence=summary["confidence"],
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence=evidence,
    )


def validate_lid_output(raw_output):
    if not isinstance(raw_output, dict):
        raise FireRedLidError("FireRed LID output must be an object")

    raw_lang = raw_output.get("lang")
    if (
        not isinstance(raw_lang, str)
        or not raw_lang
        or raw_lang != raw_lang.strip()
    ):
        raise FireRedLidError("FireRed LID lang must be a non-empty string")
    lang = _canonical_language_code(raw_lang)

    confidence = raw_output.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or confidence != confidence
        or confidence < 0
        or confidence > 1
    ):
        raise FireRedLidError("FireRed LID confidence must be within [0, 1]")

    summary = {
        "lang": lang,
        "raw_lang": raw_lang,
        "confidence": round(float(confidence), ROUND_DIGITS),
    }
    for key in ("dur_s", "rtf"):
        value = raw_output.get(key)
        if value is None:
            continue
        # Upstream formats rtf as a display string, e.g. "0.0860".
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise FireRedLidError(
                    "FireRed LID %s must be numeric" % key
                )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FireRedLidError("FireRed LID %s must be a number" % key)
        summary[key] = round(float(value), ROUND_DIGITS)
    return summary


def _canonical_language_code(language):
    """Normalize FireRed's two-token Chinese labels to the public code form."""

    normalized = " ".join(str(language).strip().lower().split())
    parts = normalized.split(" ")
    if len(parts) == 2 and parts[0] == "zh" and parts[1]:
        return "zh-%s" % parts[1].replace("_", "-")
    if normalized.startswith("zh_"):
        return "zh-%s" % normalized[3:].replace("_", "-")
    return normalized


def _default_client(config):
    if config.subprocess_python:
        return FireRedLidSubprocessClient(config)
    return FireRedLidClient(config)


def _subprocess_config(config):
    return {
        "model_dir": config.model_dir,
        "repo_dir": config.repo_dir,
        "use_gpu": config.use_gpu,
        "use_half": config.use_half,
        "subprocess_python": "",
    }
