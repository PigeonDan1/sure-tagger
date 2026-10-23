"""Post-TTS speech editing backends.

``DotsTtsEditBackend`` runs the dots.tts.edit checkpoint in its own Python
environment and process.  The simulator sends it source WAV paths over a small
JSON-RPC protocol, keeping the base TTS and edit dependency stacks isolated.
"""

from __future__ import annotations

import html
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class SpeechEditResult:
    audio: np.ndarray
    metadata: dict[str, Any]


class SpeechEditBackend(Protocol):
    """A post-TTS backend that edits one source waveform at a time."""

    name: str

    def resolve_instruction(
        self, text: str, explicit_instruction: str | None
    ) -> str | None: ...

    def edit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        source_text: str,
        instruction: str,
        source_audio_path: Path | None = None,
    ) -> SpeechEditResult: ...

    def close(self) -> None: ...


def build_rate_instruction(text: str, factor: float | str | Decimal) -> str:
    """Build the native dots.tts.edit whole-utterance rate instruction."""
    try:
        value = Decimal(str(factor))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid dots.tts.edit rate factor: {factor!r}") from exc
    if not value.is_finite():
        raise ValueError("dots.tts.edit rate factor must be finite")
    hundredths = value.quantize(Decimal("0.01"))
    if value != hundredths:
        raise ValueError("dots.tts.edit rate factor must use increments of 0.01")
    if not (Decimal("0.55") <= value <= Decimal("1.95")) or value == Decimal("1.00"):
        raise ValueError("dots.tts.edit rate factor must be 0.55..0.99 or 1.01..1.95")
    escaped_text = html.escape(text, quote=False)
    return f'<rate, factor={hundredths:.2f}>{escaped_text}</rate>'


class DotsTtsEditBackend:
    """Post-process TTS audio with the local dots.tts.edit checkpoint."""

    name = "dots_tts_edit"

    def __init__(
        self,
        model_dir: Path,
        sample_rate: int,
        *,
        checkpoint_path: Path | None = None,
        python: Path | None = None,
        server: Path | None = None,
        device: str = "cuda:0",
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
        max_sequence_length: int | None = None,
        default_rate: float | str | Decimal | None = None,
        use_xvector: bool | str = "auto",
        speaker_scale: float = 1.5,
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 20260414,
        work_dir: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path or self.model_dir / "checkpoints").resolve()
        # Keep the venv launcher path intact. Resolving its ``python`` symlink to
        # /usr/bin/python would bypass pyvenv.cfg and lose edit-only packages.
        self.python = Path(python or self.model_dir / ".venv/bin/python").absolute()
        package_root = Path(__file__).resolve().parents[2]
        self.server = Path(server or package_root / "scripts/dots_tts_edit_rpc_server.py").resolve()
        for label, path in (
            ("dots.tts.edit Python", self.python),
            ("dots.tts.edit checkpoint", self.checkpoint_path),
            ("dots.tts.edit RPC server", self.server),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.sample_rate = int(sample_rate)
        self.device = str(device)
        self.precision = str(precision)
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.max_sequence_length = (
            int(max_sequence_length) if max_sequence_length is not None else None
        )
        self.default_rate = default_rate
        if default_rate is not None:
            build_rate_instruction("validation", default_rate)
        if isinstance(use_xvector, str):
            normalized_xvector = use_xvector.lower()
            if normalized_xvector not in {"auto", "on", "off"}:
                raise ValueError("use_xvector must be auto, on, off, true, or false")
            self.use_xvector: bool | str = {
                "auto": "auto",
                "on": True,
                "off": False,
            }[normalized_xvector]
        else:
            self.use_xvector = bool(use_xvector)
        self.speaker_scale = float(speaker_scale)
        self.ode_method = str(ode_method) if ode_method is not None else None
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.guidance_scale = (
            float(guidance_scale) if guidance_scale is not None else None
        )
        self.seed = int(seed)
        self.work_dir = Path(work_dir or self.model_dir / "artifacts/saasr_edit_outputs").resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._edit_index = 0

    def resolve_instruction(
        self, text: str, explicit_instruction: str | None
    ) -> str | None:
        if explicit_instruction is not None:
            resolved = explicit_instruction.strip()
            return resolved or None
        if self.default_rate is not None:
            return build_rate_instruction(text, self.default_rate)
        return None

    def _start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif self.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
        elif self.device != "cuda":
            raise ValueError(f"Unsupported dots.tts.edit device: {self.device}")
        self._proc = subprocess.Popen(
            [str(self.python), str(self.server)],
            cwd=str(self.model_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        parameters: dict[str, Any] = {
            "checkpoint_path": str(self.checkpoint_path),
            "precision": self.precision,
            "optimize": self.optimize,
            "max_generate_length": self.max_generate_length,
        }
        if self.max_sequence_length is not None:
            parameters["max_sequence_length"] = self.max_sequence_length
        self._call("initialize", parameters)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("dots.tts.edit server closed unexpectedly")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [dots_tts_edit] {line.rstrip()}", file=sys.stderr)
                continue
            if response.get("id") != self._request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"dots.tts.edit server error: {response['error']}")
            return response.get("result", {})

    def edit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        source_text: str,
        instruction: str,
        source_audio_path: Path | None = None,
    ) -> SpeechEditResult:
        self._start()
        edit_index = self._edit_index
        self._edit_index += 1
        if source_audio_path is None:
            resolved_source_path = self.work_dir / f"source_{edit_index:05d}.wav"
            sf.write(
                resolved_source_path,
                np.asarray(audio, dtype=np.float32),
                int(sample_rate),
                subtype="FLOAT",
            )
            used_source_file = False
        else:
            resolved_source_path = Path(source_audio_path).resolve()
            if not resolved_source_path.is_file():
                raise FileNotFoundError(f"Edit source audio does not exist: {resolved_source_path}")
            used_source_file = True
        output_path = self.work_dir / f"edit_{edit_index:05d}.wav"
        print(
            f"  [dots_tts_edit] editing {resolved_source_path.name} "
            f"(instruction={instruction[:80]!r})...",
            file=sys.stderr,
        )
        params: dict[str, Any] = {
            "source_audio_path": str(resolved_source_path),
            "output_path": str(output_path),
            "instruction": instruction,
            "source_text": source_text,
            "target_text": source_text,
            "use_xvector": self.use_xvector,
            "speaker_scale": self.speaker_scale,
            "seed": self.seed + edit_index,
        }
        if self.ode_method is not None:
            params["ode_method"] = self.ode_method
        if self.num_steps is not None:
            params["num_steps"] = self.num_steps
        if self.guidance_scale is not None:
            params["guidance_scale"] = self.guidance_scale
        metadata = self._call("edit", params)
        returned_path = Path(metadata.get("audio_path") or output_path)
        if not returned_path.is_file():
            raise FileNotFoundError(f"dots.tts.edit produced no audio at {returned_path}")
        edited, source_rate = sf.read(str(returned_path))
        edited = self._to_target_rate(np.asarray(edited, dtype=np.float32), int(source_rate))
        return SpeechEditResult(
            audio=edited,
            metadata={
                **metadata,
                "edit_source_audio_path": str(resolved_source_path),
                "used_existing_source_file": used_source_file,
            },
        )

    def _to_target_rate(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if source_rate == self.sample_rate:
            return audio.astype(np.float32)
        common = math.gcd(int(source_rate), self.sample_rate)
        resampled = resample_poly(
            audio.astype(np.float64), self.sample_rate // common, int(source_rate) // common
        )
        return resampled.astype(np.float32)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        self._proc = None


def build_speech_edit_backend(
    backend_config: dict[str, Any], config_path: Path, sample_rate: int
) -> SpeechEditBackend | None:
    """Build the optional post-TTS editor configured for a simulation."""
    if not backend_config:
        return None
    kind = str(backend_config.get("type", "none")).lower()
    if kind in {"none", "disabled", "off"}:
        return None
    if kind not in {"dots_tts_edit", "dots_edit", "dots.tts.edit"}:
        raise ValueError(f"Unknown speech edit backend type: {kind}")

    config_dir = config_path.parent

    def resolve_path(value: Any, default: Path) -> Path:
        path = Path(value).expanduser() if value is not None else default
        return path if path.is_absolute() else (config_dir / path).resolve()

    model_dir = resolve_path(
        backend_config.get("model_dir") or os.environ.get("DOTS_TTS_EDIT_MODEL_DIR"),
        config_dir / "../../dots-studio__dots.tts.edit",
    )
    checkpoint_path = resolve_path(
        backend_config.get("checkpoint_path"), model_dir / "checkpoints"
    )
    python_value = backend_config.get("python") or os.environ.get("DOTS_TTS_EDIT_PYTHON")
    python = resolve_path(python_value, model_dir / ".venv/bin/python")
    server = (
        resolve_path(backend_config["server"], model_dir)
        if backend_config.get("server")
        else None
    )
    work_dir = (
        resolve_path(backend_config["work_dir"], model_dir)
        if backend_config.get("work_dir")
        else None
    )
    return DotsTtsEditBackend(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        python=python,
        server=server,
        sample_rate=sample_rate,
        device=str(backend_config.get("device", "cuda:0")),
        precision=str(backend_config.get("precision", "bfloat16")),
        optimize=bool(backend_config.get("optimize", False)),
        max_generate_length=int(backend_config.get("max_generate_length", 500)),
        max_sequence_length=(
            int(backend_config["max_sequence_length"])
            if backend_config.get("max_sequence_length") is not None
            else None
        ),
        default_rate=backend_config.get("default_rate"),
        use_xvector=backend_config.get("use_xvector", "auto"),
        speaker_scale=float(backend_config.get("speaker_scale", 1.5)),
        ode_method=backend_config.get("ode_method"),
        num_steps=(
            int(backend_config["num_steps"])
            if backend_config.get("num_steps") is not None
            else None
        ),
        guidance_scale=(
            float(backend_config["guidance_scale"])
            if backend_config.get("guidance_scale") is not None
            else None
        ),
        seed=int(backend_config.get("seed", 20260414)),
        work_dir=work_dir,
    )
