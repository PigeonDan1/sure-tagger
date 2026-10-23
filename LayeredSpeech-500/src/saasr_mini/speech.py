"""Pluggable speech backends: turn dialogue text into mono waveforms.

The renderer only needs ``text -> np.ndarray`` per utterance. The default
:class:`FixtureSpeechBackend` produces deterministic, non-linguistic test audio
and never touches a model. :class:`DotsTtsBackend` drives the relocated
``rednote-hilab/dots.tts-base`` model through its bundled JSON-RPC server so the
two environments stay isolated (dots.tts needs torch + CUDA, this package does
not).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter, resample_poly

from .audio import rms, stable_seed


class SpeechBackend(Protocol):
    """Anything that turns (text, speaker_index) into a mono float32 waveform."""

    name: str

    def synthesize(
        self,
        text: str,
        speaker_index: int,
        *,
        instructions: str | None = None,
        language: str | None = None,
        prompt_language: str | None = None,
        max_generate_length: int | None = None,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


def _fixture_duration(text: str) -> float:
    words = max(1, len(text.split()))
    punctuation = sum(text.count(mark) for mark in ",.;?!")
    return min(4.8, 0.28 + words * 0.255 + punctuation * 0.045)


def _fixture_speech(text: str, speaker_index: int, sample_rate: int, seed: int) -> np.ndarray:
    """Create non-linguistic, speech-shaped test audio; this is deliberately not a TTS model."""
    duration = _fixture_duration(text)
    count = round(duration * sample_rate)
    time = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(stable_seed(f"fixture:{speaker_index}:{text}", seed))

    base_f0 = 92.0 + (speaker_index * 29 % 105)
    drift = 1.0 + 0.035 * np.sin(2 * np.pi * (1.7 + speaker_index * 0.03) * time)
    phase = 2 * np.pi * np.cumsum(base_f0 * drift) / sample_rate
    voiced = sum((1.0 / harmonic) * np.sin(harmonic * phase) for harmonic in range(1, 9))

    syllables = max(2, round(duration * (3.1 + speaker_index % 3 * 0.2)))
    envelope = np.zeros(count, dtype=np.float64)
    centers = np.linspace(0.12, max(0.13, duration - 0.12), syllables)
    centers += rng.uniform(-0.045, 0.045, size=syllables)
    for center in centers:
        width = rng.uniform(0.065, 0.13)
        envelope += np.exp(-0.5 * ((time - center) / width) ** 2)
    envelope = np.clip(envelope, 0.0, 1.0)

    noise = rng.normal(0.0, 1.0, count)
    b, a = butter(2, [250 / (sample_rate / 2), 3600 / (sample_rate / 2)], btype="band")
    breath = lfilter(b, a, noise)
    audio = envelope * (0.12 * voiced + 0.015 * breath)
    fade = min(round(0.02 * sample_rate), count // 2)
    audio[:fade] *= np.linspace(0.0, 1.0, fade)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade)
    audio *= (10 ** (-24 / 20)) / max(rms(audio), 1e-8)
    return audio.astype(np.float32)


class FixtureSpeechBackend:
    name = "non_linguistic_fixture"

    def __init__(self, sample_rate: int, seed: int) -> None:
        self.sample_rate = sample_rate
        self.seed = seed

    def synthesize(
        self,
        text: str,
        speaker_index: int,
        *,
        instructions: str | None = None,
        language: str | None = None,
        prompt_language: str | None = None,
        max_generate_length: int | None = None,
    ) -> np.ndarray:
        del instructions
        del language
        del prompt_language
        return _fixture_speech(text, speaker_index, self.sample_rate, self.seed)

    def close(self) -> None:
        return None


class DotsTtsBackend:
    """Voice-cloning TTS via the dots.tts JSON-RPC server (one persistent subprocess)."""

    name = "dots_tts"
    NATIVE_SAMPLE_RATE = 48_000

    def __init__(
        self,
        model_dir: Path,
        voices: dict[int, dict[str, Any]],
        sample_rate: int,
        *,
        python: Path | None = None,
        server: Path | None = None,
        device: str = "cuda:0",
        precision: str = "float16",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        ode_method: str = "euler",
        normalize_text: bool = False,
        seed: int | None = None,
        instructions: str | None = None,
        max_generate_length: int | None = None,
        work_dir: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.python = Path(python) if python else self.model_dir / ".venv/bin/python"
        if not self.python.exists():
            raise FileNotFoundError(f"dots.tts Python not found: {self.python}")
        server = Path(server) if server else self.model_dir / "server.py"
        if not server.exists():
            raise FileNotFoundError(f"dots.tts server not found: {server}")
        self.server = server
        self.voices = dict(voices)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.precision = precision
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.speaker_scale = float(speaker_scale)
        self.ode_method = str(ode_method)
        self.normalize_text = bool(normalize_text)
        self.seed = int(seed) if seed is not None else None
        self.default_instructions = instructions
        self.max_generate_length = max_generate_length
        self.work_dir = Path(work_dir) if work_dir else self.model_dir / "artifacts/saasr_outputs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._utterance_index = 0
        self.last_native_output_path: Path | None = None

    def _start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Each worker owns one physical GPU. Dots' runtime selects ``cuda``
        # and therefore relies on visibility filtering; an inherited cluster
        # default must not route every server back to GPU 0.
        device_index = self.device.split(":", 1)[1] if self.device.startswith("cuda:") else "0"
        env["CUDA_VISIBLE_DEVICES"] = device_index
        env["DEVICE"] = "cuda:0"
        env["PRECISION"] = self.precision
        env["MODEL_PATH"] = str(self.model_dir / "checkpoints")
        if self.max_generate_length is not None:
            # Keep the server-side runtime cap in sync with the backend.  The
            # JSON-RPC payload cannot change Dots' already-initialized runtime,
            # so without this environment variable long/unsupported language
            # outputs can run until the vocoder exhausts GPU memory.
            env["MAX_GENERATE_LENGTH"] = str(int(self.max_generate_length))
        self._proc = subprocess.Popen(
            [str(self.python), str(self.server)],
            cwd=str(self.model_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: dots.tts loguru logs stay visible
            text=True,
            bufsize=1,
            env=env,
        )
        self._call("initialize", {})

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("dots.tts server closed unexpectedly during synthesis")
            response = json.loads(line)
            if response.get("id") != self._request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"dots.tts server error: {response['error']}")
            return response.get("result", {})

    def synthesize(
        self,
        text: str,
        speaker_index: int,
        *,
        instructions: str | None = None,
        language: str | None = None,
        prompt_language: str | None = None,
        max_generate_length: int | None = None,
    ) -> np.ndarray:
        self.last_native_output_path = None
        voice = self.voices.get(speaker_index)
        if voice is None:
            raise ValueError(
                f"No reference voice configured for speaker_index {speaker_index}; "
                "add it to the voices file."
            )
        self._start()
        output_path = self.work_dir / f"utt_{self._utterance_index:05d}.wav"
        self._utterance_index += 1
        print(
            f"  [dots_tts] synthesizing speaker {speaker_index} ({len(text)} chars, "
            f"instruction={bool(instructions or self.default_instructions)})...",
            file=sys.stderr,
        )
        resolved_instructions = (
            self.default_instructions if instructions is None else instructions
        )
        result = self._call(
            "tools/call",
            {
                "name": "synthesize_speech",
                "arguments": {
                    "text": text,
                    "prompt_audio_path": str(voice["prompt_audio"]),
                    "prompt_text": voice.get("prompt_text", ""),
                    "language": language or voice.get("language", "en"),
                    "prompt_language": prompt_language or voice.get("prompt_language"),
                    "output_path": str(output_path),
                    "num_steps": self.num_steps,
                    "guidance_scale": self.guidance_scale,
                    "speaker_scale": self.speaker_scale,
                    "ode_method": self.ode_method,
                    "normalize_text": self.normalize_text,
                    **(
                        {"instructions": resolved_instructions}
                        if resolved_instructions
                        else {}
                    ),
                    **({"seed": self.seed} if self.seed is not None else {}),
                    **(
                        {"max_generate_length": int(max_generate_length or self.max_generate_length)}
                        if max_generate_length or self.max_generate_length
                        else {}
                    ),
                },
            },
        )
        audio_path = Path(result.get("audio_path") or output_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"dots.tts produced no audio at {audio_path}")
        self.last_native_output_path = audio_path.resolve()
        audio, source_rate = sf.read(str(audio_path))
        return self._to_target_rate(np.asarray(audio, dtype=np.float32), int(source_rate))

    def _to_target_rate(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if source_rate == self.sample_rate:
            return audio.astype(np.float32)
        common = math.gcd(int(source_rate), int(self.sample_rate))
        up = int(self.sample_rate) // common
        down = int(source_rate) // common
        resampled = resample_poly(audio.astype(np.float64), up, down)
        return resampled.astype(np.float32)

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    def release_for_edit(self) -> None:
        """Unload base TTS while retaining an object that can restart next scene."""
        self.close()


def build_speech_backend(
    backend_config: dict[str, Any], config_path: Path, sample_rate: int, seed: int
) -> SpeechBackend:
    """Instantiate the backend named in ``backend_config`` (or the fixture default)."""
    kind = str(backend_config.get("type", "fixture")).lower()
    if kind in {"fixture", "none", "non_linguistic_fixture"}:
        return FixtureSpeechBackend(sample_rate, seed)
    if kind in {"dots_tts", "dots_tts_base", "dots"}:
        return _build_dots_tts(backend_config, config_path, sample_rate)
    raise ValueError(f"Unknown speech backend type: {kind}")


def _build_dots_tts(
    backend_config: dict[str, Any], config_path: Path, sample_rate: int
) -> DotsTtsBackend:
    config_dir = config_path.parent
    model_dir = Path(
        backend_config.get("model_dir")
        or os.environ.get("DOTS_TTS_MODEL_DIR")
        or (config_dir / "../rednote-hilab__dots.tts-base")
    ).expanduser()
    if not model_dir.is_absolute():
        model_dir = (config_dir / model_dir).resolve()

    python = backend_config.get("python") or os.environ.get("DOTS_TTS_PYTHON")
    python_path = Path(python).expanduser() if python else None

    voices = _load_voices(backend_config.get("voices"), config_dir)
    return DotsTtsBackend(
        model_dir=model_dir,
        python=python_path,
        voices=voices,
        sample_rate=sample_rate,
        device=str(backend_config.get("device", "cuda:0")),
        precision=str(backend_config.get("precision", "float16")),
        num_steps=int(backend_config.get("num_steps", 10)),
        guidance_scale=float(backend_config.get("guidance_scale", 1.2)),
        speaker_scale=float(backend_config.get("speaker_scale", 1.5)),
        ode_method=str(backend_config.get("ode_method", "euler")),
        normalize_text=bool(backend_config.get("normalize_text", False)),
        seed=(int(backend_config["seed"]) if backend_config.get("seed") is not None else None),
        instructions=(
            str(backend_config["instructions"])
            if backend_config.get("instructions") is not None
            else None
        ),
        max_generate_length=backend_config.get("max_generate_length"),
        work_dir=Path(backend_config["work_dir"]) if backend_config.get("work_dir") else None,
    )


def _load_voices(
    voices_spec: Any, config_dir: Path
) -> dict[int, dict[str, Any]]:
    """Load a per-speaker voice map from a file or an inline mapping.

    ``prompt_audio`` paths are resolved relative to the voices file's directory
    (or the config directory when the mapping is inline).
    """
    if isinstance(voices_spec, str):
        voices_path = Path(voices_spec).expanduser()
        if not voices_path.is_absolute():
            voices_path = (config_dir / voices_path).resolve()
        text = voices_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if voices_path.suffix in {".yaml", ".yml"} else json.loads(text)
        base = voices_path.parent
    elif isinstance(voices_spec, dict):
        raw = voices_spec
        base = config_dir
    else:
        raise ValueError("speech_backend.voices must be a file path or an inline mapping")

    voices: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or "prompt_audio" not in value:
            raise ValueError(f"voice for speaker {key} is missing prompt_audio")
        prompt_audio = Path(value["prompt_audio"]).expanduser()
        if not prompt_audio.is_absolute():
            prompt_audio = (base / prompt_audio).resolve()
        voices[int(key)] = {
            "prompt_audio": str(prompt_audio),
            "prompt_text": str(value.get("prompt_text", "")),
            "language": str(value.get("language", "en")),
        }
    return voices
