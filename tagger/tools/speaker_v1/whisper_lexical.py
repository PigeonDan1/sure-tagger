"""Offline Whisper lexical-clock adapter for one utterance at a time."""

import time

from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_VERSION = "whisper_lexical_clock_v2.0-shadow.1"
CHECKPOINT_SHA256 = (
    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
)


class WhisperLexicalError(RuntimeError):
    pass


class WhisperLexicalConfig:
    def __init__(
        self,
        model_path,
        subprocess_python="",
        device="auto",
        language=None,
        word_timestamps=True,
        portable_word_timing=True,
        timeout_sec=600,
    ):
        self.model_path = str(model_path)
        self.subprocess_python = str(subprocess_python or "")
        self.device = str(device or "auto")
        self.language = str(language) if language else None
        self.word_timestamps = bool(word_timestamps)
        self.portable_word_timing = bool(portable_word_timing)
        self.timeout_sec = int(timeout_sec)

    def to_record(self):
        return {
            "model_path": self.model_path,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
            "portable_word_timing": self.portable_word_timing,
            "timeout_sec": self.timeout_sec,
        }


class WhisperLexicalClient:
    def __init__(self, config):
        self.config = config
        self._model = None
        self._device = None

    def transcribe(self, audio_path, context=None):
        del context
        try:
            import librosa
            import torch
        except ImportError as exc:
            raise WhisperLexicalError(
                "Whisper requires librosa and torch in its model environment"
            ) from exc
        model, device = self._get_model()
        if self.config.word_timestamps and self.config.portable_word_timing:
            _configure_portable_word_timing()
        try:
            audio, sample_rate = librosa.load(
                str(audio_path), sr=16000, mono=True
            )
        except Exception as exc:
            raise WhisperLexicalError("failed to decode Whisper audio") from exc
        started = time.time()
        try:
            prediction = model.transcribe(
                audio,
                task="transcribe",
                language=self.config.language,
                fp16=device.startswith("cuda"),
                verbose=None,
                word_timestamps=self.config.word_timestamps,
            )
        except Exception as exc:
            raise WhisperLexicalError("Whisper inference failed") from exc

        units = []
        for segment_index, segment in enumerate(prediction.get("segments", [])):
            words = segment.get("words") or []
            if self.config.word_timestamps and words:
                for word_index, word in enumerate(words):
                    if word.get("start") is None or word.get("end") is None:
                        continue
                    units.append(
                        {
                            "unit_id": "word_%06d_%04d"
                            % (segment_index, word_index),
                            "start_sec": round(float(word["start"]), 6),
                            "end_sec": round(float(word["end"]), 6),
                            "text": str(word.get("word", "")).strip(),
                            "timestamp_method": "attention_dtw_word_interval",
                            "probability": _optional_round(
                                word.get("probability")
                            ),
                        }
                    )
            elif segment.get("start") is not None and segment.get("end") is not None:
                units.append(
                    {
                        "unit_id": "segment_%06d" % segment_index,
                        "start_sec": round(float(segment["start"]), 6),
                        "end_sec": round(float(segment["end"]), 6),
                        "text": str(segment.get("text", "")).strip(),
                        "timestamp_method": "whisper_decode_segment_interval",
                        "avg_logprob": _optional_round(
                            segment.get("avg_logprob")
                        ),
                        "no_speech_prob": _optional_round(
                            segment.get("no_speech_prob")
                        ),
                    }
                )
        elapsed = time.time() - started
        return {
            "text": str(prediction.get("text", "")).strip(),
            "language": prediction.get("language"),
            "lexical_units": units,
            "runtime": {
                "device": device,
                "dtype": "float16" if device.startswith("cuda") else "float32",
                "elapsed_sec": round(elapsed, 6),
                "audio_duration_sec": round(len(audio) / float(sample_rate), 6),
                "word_timing_backend": (
                    "torch_sort_numba_cpu_dtw"
                    if self.config.word_timestamps
                    and self.config.portable_word_timing
                    else "openai_whisper_default"
                ),
            },
        }

    def _get_model(self):
        if self._model is not None:
            return self._model, self._device
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise WhisperLexicalError(
                "Whisper requires openai-whisper and torch"
            ) from exc
        if self.config.device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise WhisperLexicalError("CUDA was requested but is unavailable")
        try:
            self._model = whisper.load_model(
                self.config.model_path, device=device
            )
        except Exception as exc:
            raise WhisperLexicalError("Whisper model loading failed") from exc
        self._device = device
        return self._model, self._device


class WhisperLexicalSubprocessClient:
    def __init__(self, config):
        self.config = config

    def transcribe(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "whisper_lexical_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


def transcribe(audio_path, config, context=None):
    client = (
        WhisperLexicalSubprocessClient(config)
        if config.subprocess_python
        else WhisperLexicalClient(config)
    )
    return client.transcribe(audio_path, context=context)


def _subprocess_config(config):
    result = config.to_record()
    result["subprocess_python"] = ""
    result.pop("timeout_sec", None)
    return result


def _optional_round(value):
    if value is None:
        return None
    return round(float(value), 6)


def _configure_portable_word_timing():
    """Skip optional Triton JIT when the shared runtime has no C headers."""

    import numpy as np
    import torch.nn.functional as functional
    import whisper.timing as timing

    if getattr(timing, "_sure_tagger_portable_timing", False):
        return

    def median_filter_portable(value, filter_width):
        pad_width = filter_width // 2
        if value.shape[-1] <= pad_width:
            return value
        dimensions = value.ndim
        if dimensions <= 2:
            value = value[None, None, :]
        value = functional.pad(
            value,
            (filter_width // 2, filter_width // 2, 0, 0),
            mode="reflect",
        )
        result = value.unfold(-1, filter_width, 1).sort()[0][
            ..., filter_width // 2
        ]
        if dimensions <= 2:
            result = result[0, 0]
        return result

    def dtw_portable(value):
        return timing.dtw_cpu(value.double().cpu().numpy().astype(np.float32))

    timing.median_filter = median_filter_portable
    timing.dtw = dtw_portable
    timing._sure_tagger_portable_timing = True
