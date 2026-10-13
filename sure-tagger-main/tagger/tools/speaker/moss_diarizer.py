"""MOSS-Transcribe-Diarize adapter."""

import json
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union
import urllib.error
import urllib.request
import wave

from tagger.tools.base import ToolResult
from tagger.tools.speaker.metrics import normalize_segments
from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_NAME = "moss_diarizer"
TOOL_VERSION = "moss_diarizer_v0.2.0"


class MossDiarizeError(RuntimeError):
    pass


class MossDiarizeConfig:
    def __init__(
        self,
        endpoint="",
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        timeout_sec=900,
        max_new_tokens=65536,
        api_key="",
        subprocess_python="",
        device="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        prompt="",
    ):
        self.endpoint = endpoint or ""
        self.model = model or "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        self.timeout_sec = int(timeout_sec)
        self.max_new_tokens = int(max_new_tokens)
        self.api_key = api_key or ""
        self.subprocess_python = subprocess_python or ""
        self.device = device or "auto"
        self.torch_dtype = torch_dtype or "auto"
        self.trust_remote_code = bool(trust_remote_code)
        self.prompt = prompt or ""

    def cache_key(self):
        return (
            self.model,
            self.max_new_tokens,
            self.device,
            self.torch_dtype,
            self.trust_remote_code,
            self.prompt,
            self.subprocess_python,
        )

    def to_record(self):
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "max_new_tokens": self.max_new_tokens,
            "api_key_configured": bool(self.api_key),
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "prompt_configured": bool(self.prompt),
        }


class MossDiarizeClient:
    """Adapter around the local OpenMOSS Transformers implementation."""

    def __init__(self, config=None):
        self.config = config or MossDiarizeConfig()
        self._runtime = None

    def diarize(self, audio_path, context=None):
        runtime = self._get_runtime(context)
        messages = _build_transcription_messages(
            runtime["build_transcription_messages"],
            audio_path,
            self.config.prompt,
        )
        try:
            result = runtime["generate_transcription"](
                runtime["model"],
                runtime["processor"],
                messages,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                device=runtime["device"],
                dtype=runtime["dtype"],
            )
        except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
            raise MossDiarizeError(
                "local MOSS diarize inference failed: %s" % exc
            ) from exc

        if isinstance(result, dict):
            text = result.get("text", "")
            payload = dict(result)
        else:
            text = str(result)
            payload = {"text": text}
        if not isinstance(text, str):
            text = str(text)
            payload["text"] = text

        segments = _segments_from_openmoss_parse(
            runtime["parse_transcript"],
            text,
        )
        if segments:
            payload["segments"] = segments
        return payload

    def _get_runtime(self, context=None):
        if context is None:
            if self._runtime is None:
                self._runtime = self._load_runtime()
            return self._runtime

        cache = context.setdefault("moss_diarize_runtime_by_config", {})
        key = self.config.cache_key()
        if key not in cache:
            cache[key] = self._load_runtime()
        return cache[key]

    def _load_runtime(self):
        (
            torch,
            AutoModelForCausalLM,
            AutoProcessor,
            parse_transcript,
            build_transcription_messages,
            generate_transcription,
            resolve_device,
        ) = _load_openmoss_dependencies()
        try:
            device = resolve_device(self.config.device)
            dtype = _resolve_torch_dtype(torch, self.config.torch_dtype, device)
            model = _load_openmoss_model(AutoModelForCausalLM, self.config)
            model = model.to(dtype=dtype).to(device).eval()
            processor = AutoProcessor.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
            raise MossDiarizeError(
                "local MOSS diarize runtime loading failed: %s" % exc
            ) from exc
        return {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype,
            "parse_transcript": parse_transcript,
            "build_transcription_messages": build_transcription_messages,
            "generate_transcription": generate_transcription,
        }


class MossDiarizeSubprocessClient:
    """Adapter that runs local MOSS in its configured Python environment."""

    def __init__(self, config=None):
        self.config = config or MossDiarizeConfig()

    def diarize(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "moss_diarize_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


class MossDiarizeHttpClient:
    """Legacy OpenAI-compatible HTTP adapter."""

    def __init__(self, config=None):
        self.config = config or MossDiarizeConfig()

    def diarize(self, audio_path, context=None):
        del context
        return call_moss_http(audio_path, self.config)


def run(audio_path, duration_sec=None, context=None, config=None, client=None, **_kwargs):
    # type: (Union[str, Path], Optional[float], Optional[Dict[str, Any]], Optional[MossDiarizeConfig], Any, Any) -> ToolResult
    config = config or MossDiarizeConfig()
    client = client or _default_client(config)
    payload = client.diarize(audio_path, context=context)
    segments = parse_moss_output(payload)
    if duration_sec is None:
        duration_sec = _duration_from_payload(payload)
    if duration_sec is not None:
        segments = normalize_segments(segments, float(duration_sec))
    if not segments:
        raise MossDiarizeError("MOSS diarize returned no speaker segments")
    value = {
        "metadata_version": "moss_diarize_timeline_v0.1",
        "segments": segments,
        "raw_text": payload.get("text", "") if isinstance(payload, dict) else "",
    }
    return ToolResult(
        tag_path="speaker.diarization_timeline",
        value=value,
        tool_name=TOOL_NAME,
        method="moss_transcribe_diarize",
        status="estimated",
        confidence=0.85,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence={
            "segment_count": len(segments),
            "model": config.model,
            "config": config.to_record(),
        },
    )


def _default_client(config):
    if config.subprocess_python:
        return MossDiarizeSubprocessClient(config)
    if config.endpoint:
        return MossDiarizeHttpClient(config)
    return MossDiarizeClient(config)


def _subprocess_config(config):
    return {
        "endpoint": "",
        "model": config.model,
        "timeout_sec": config.timeout_sec,
        "max_new_tokens": config.max_new_tokens,
        "api_key": "",
        "subprocess_python": "",
        "device": config.device,
        "torch_dtype": config.torch_dtype,
        "trust_remote_code": config.trust_remote_code,
        "prompt": config.prompt,
    }


def _load_openmoss_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
            resolve_device,
        )
    except ImportError as exc:
        raise MossDiarizeError(
            "local MOSS requires the OpenMOSS package, transformers, and torch "
            "in the configured Python environment"
        ) from exc
    return (
        torch,
        AutoModelForCausalLM,
        AutoProcessor,
        parse_transcript,
        build_transcription_messages,
        generate_transcription,
        resolve_device,
    )


def _load_openmoss_model(AutoModelForCausalLM, config):
    kwargs = {"trust_remote_code": config.trust_remote_code}
    try:
        return AutoModelForCausalLM.from_pretrained(
            config.model,
            dtype="auto",
            **kwargs
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype="auto",
            **kwargs
        )


def _resolve_torch_dtype(torch, dtype_name, device):
    name = str(dtype_name or "auto").lower()
    if name == "auto":
        device_type = getattr(device, "type", str(device))
        return torch.bfloat16 if device_type == "cuda" else torch.float32
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise MossDiarizeError("unsupported MOSS torch dtype: %s" % dtype_name)
    return dtype


def _build_transcription_messages(build_transcription_messages, audio_path, prompt):
    if prompt:
        try:
            return build_transcription_messages(str(audio_path), prompt=prompt)
        except TypeError:
            try:
                return build_transcription_messages(str(audio_path), prompt)
            except TypeError as exc:
                raise MossDiarizeError(
                    "OpenMOSS build_transcription_messages does not accept prompt"
                ) from exc
    return build_transcription_messages(str(audio_path))


def _segments_from_openmoss_parse(parse_transcript, text):
    try:
        parsed = parse_transcript(text)
    except Exception as exc:  # noqa: BLE001 - normalized to a tool error.
        raise MossDiarizeError("OpenMOSS transcript parsing failed") from exc
    segments = []
    for item in parsed:
        start = _object_value(item, "start", "start_sec")
        end = _object_value(item, "end", "end_sec")
        speaker = _object_value(item, "speaker", "speaker_id", "label")
        if start is None or end is None or speaker is None:
            continue
        segment = {
            "start_sec": start,
            "end_sec": end,
            "speaker_id": speaker,
        }
        text_value = _object_value(item, "text")
        if text_value:
            segment["text"] = text_value
        segments.append(segment)
    return segments


def _object_value(item, *names):
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def run_merged_channels(audio_path, duration_sec=None, context=None, config=None, client=None, **_kwargs):
    # type: (Union[str, Path], Optional[float], Optional[Dict[str, Any]], Optional[MossDiarizeConfig], Any, Any) -> ToolResult
    config = config or MossDiarizeConfig()
    path = Path(audio_path)
    with tempfile.TemporaryDirectory(prefix="sure_tagger_moss_headset_mix_") as tmpdir:
        mixed = mixdown_multichannel_wav(path, Path(tmpdir))
        result = run(
            mixed["path"],
            duration_sec=duration_sec,
            context=context,
            config=config,
            client=client,
        )
    value = dict(result.value)
    value["metadata_version"] = "moss_diarize_merged_headset_timeline_v0.1"
    return ToolResult(
        tag_path="speaker.diarization_timeline",
        value=value,
        tool_name=TOOL_NAME,
        method="moss_transcribe_diarize_merged_headset",
        status="estimated",
        confidence=result.confidence,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence={
            "input_channel_count": mixed["channel_count"],
            "mixdown_method": mixed["method"],
            "segment_count": len(value.get("segments", [])),
            "model": config.model,
        },
    )


def run_channel_purity_check(audio_path, duration_sec=None, context=None, config=None, client=None, **_kwargs):
    # type: (Union[str, Path], Optional[float], Optional[Dict[str, Any]], Optional[MossDiarizeConfig], Any, Any) -> ToolResult
    config = config or MossDiarizeConfig()
    path = Path(audio_path)
    channel_results = []
    with tempfile.TemporaryDirectory(prefix="sure_tagger_moss_channel_qa_") as tmpdir:
        split = split_multichannel_wav(path, Path(tmpdir))
        for channel_index, channel_path in enumerate(split["paths"]):
            channel_context = dict(context or {})
            channel_context.update({
                "speaker_route_phase": "channel_purity_check",
                "source_channel_id": "ch%s" % channel_index,
            })
            result = run(
                channel_path,
                duration_sec=duration_sec,
                context=channel_context,
                config=config,
                client=client,
            )
            speaker_ids = sorted(set(
                str(item["speaker_id"])
                for item in result.value.get("segments", [])
                if item.get("speaker_id") is not None
            ))
            channel_results.append({
                "channel_id": "ch%s" % channel_index,
                "speaker_count": len(speaker_ids),
                "speaker_ids": speaker_ids,
            })
    all_single_speaker = bool(channel_results) and all(
        item["speaker_count"] == 1 for item in channel_results
    )
    value = {
        "metadata_version": "moss_channel_purity_v0.1",
        "all_channels_single_speaker": all_single_speaker,
        "channels": channel_results,
    }
    return ToolResult(
        tag_path="speaker.channel_purity",
        value=value,
        tool_name=TOOL_NAME,
        method="moss_per_channel_speaker_purity",
        status="estimated",
        confidence=0.85,
        tool_type="model",
        tool_version=TOOL_VERSION,
        evidence={
            "input_channel_count": split["channel_count"],
            "channel_split_method": split["method"],
            "all_channels_single_speaker": all_single_speaker,
            "model": config.model,
        },
    )


def call_moss_http(audio_path, config):
    # type: (Union[str, Path], MossDiarizeConfig) -> Dict[str, Any]
    if not config.endpoint:
        raise MossDiarizeError("MOSS diarize endpoint is not configured")
    path = Path(audio_path)
    if not path.exists():
        raise MossDiarizeError("audio file does not exist: %s" % path)
    boundary = "----suretagger%s" % uuid.uuid4().hex
    fields = {
        "model": config.model,
        "response_format": "verbose_json",
        "max_new_tokens": str(config.max_new_tokens),
    }
    with path.open("rb") as source:
        file_bytes = source.read()
    body = _multipart_body(boundary, fields, "file", path.name, file_bytes)
    headers = {
        "Content-Type": "multipart/form-data; boundary=%s" % boundary,
    }
    if config.api_key:
        headers["Authorization"] = "Bearer %s" % config.api_key
    request = urllib.request.Request(
        config.endpoint,
        data=body,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise MossDiarizeError("MOSS diarize HTTP %s: %s" % (exc.code, detail))
    except urllib.error.URLError as exc:
        raise MossDiarizeError("MOSS diarize request failed: %s" % exc)
    except ValueError as exc:
        raise MossDiarizeError("MOSS diarize returned invalid JSON") from exc


def parse_moss_output(payload):
    # type: (Any) -> List[Dict[str, Any]]
    if isinstance(payload, dict):
        for key in ("segments", "speaker_segments", "diarization_segments"):
            segments = _segments_from_list(payload.get(key))
            if segments:
                return segments
        for key in ("chunks", "words"):
            segments = _segments_from_list(payload.get(key))
            if segments:
                return segments
        text = payload.get("text") or payload.get("transcript") or payload.get("output_text")
        if text:
            segments = parse_moss_text(str(text))
            if segments:
                return segments
    if isinstance(payload, list):
        segments = _segments_from_list(payload)
        if segments:
            return segments
    if isinstance(payload, str):
        return parse_moss_text(payload)
    return []


def parse_moss_text(text):
    # type: (str) -> List[Dict[str, Any]]
    text = text or ""
    patterns = [
        re.compile(
            r"\[(?P<start>[0-9:.]+)\]\s*\[(?P<speaker>[A-Za-z_]*\d+|S\d+)\]\s*(?P<text>.*?)\s*\[(?P<end>[0-9:.]+)\]",
            re.DOTALL,
        ),
        re.compile(
            r"\[(?P<start>[0-9:.]+)\s*[-,]\s*(?P<end>[0-9:.]+)\]\s*\[(?P<speaker>[A-Za-z_]*\d+|S\d+)\]\s*(?P<text>.*?)(?=\n|\Z)",
            re.DOTALL,
        ),
    ]
    segments = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = parse_time(match.group("start"))
            end = parse_time(match.group("end"))
            if start is None or end is None or end <= start:
                continue
            segments.append({
                "start_sec": start,
                "end_sec": end,
                "speaker_id": match.group("speaker"),
                "text": match.group("text").strip(),
            })
        if segments:
            return segments
    return []


def parse_time(value):
    # type: (str) -> Optional[float]
    raw = str(value).strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        return None
    return None


def _segments_from_list(items):
    segments = []
    if not isinstance(items, list):
        return segments
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker = (
            item.get("speaker_id")
            or item.get("speaker")
            or item.get("label")
            or item.get("speaker_label")
        )
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if speaker is None or start is None or end is None:
            continue
        segment = {
            "start_sec": start,
            "end_sec": end,
            "speaker_id": speaker,
        }
        if item.get("text"):
            segment["text"] = item.get("text")
        segments.append(segment)
    return segments


def _duration_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    for key in ("duration_sec", "duration"):
        try:
            value = payload.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _multipart_body(boundary, fields, file_field, filename, file_bytes):
    parts = []
    for name, value in fields.items():
        parts.append("--%s\r\n" % boundary)
        parts.append('Content-Disposition: form-data; name="%s"\r\n\r\n' % name)
        parts.append(str(value))
        parts.append("\r\n")
    parts.append("--%s\r\n" % boundary)
    parts.append(
        'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
        % (file_field, filename)
    )
    parts.append("Content-Type: audio/wav\r\n\r\n")
    prefix = "".join(parts).encode("utf-8")
    suffix = ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    return prefix + file_bytes + suffix


def _ffprobe_audio_info(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(command)
    except OSError as exc:
        raise MossDiarizeError("ffprobe is required for non-PCM channel split") from exc
    except subprocess.CalledProcessError as exc:
        raise MossDiarizeError("ffprobe failed for %s" % path) from exc
    payload = json.loads(output.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise MossDiarizeError("ffprobe found no audio stream: %s" % path)
    return streams[0]


def mixdown_multichannel_wav(audio_path, output_dir):
    # type: (Union[str, Path], Path) -> Dict[str, Any]
    path = Path(audio_path)
    try:
        return _mixdown_multichannel_wav_python(path, output_dir)
    except wave.Error:
        return _mixdown_multichannel_wav_ffmpeg(path, output_dir)


def split_multichannel_wav(audio_path, output_dir):
    # type: (Union[str, Path], Path) -> Dict[str, Any]
    path = Path(audio_path)
    try:
        return _split_multichannel_wav_python(path, output_dir)
    except wave.Error:
        return _split_multichannel_wav_ffmpeg(path, output_dir)


def _split_multichannel_wav_python(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    output_paths = []
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        if channels < 2:
            raise MossDiarizeError("MOSS channel purity check requires at least two channels")
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        try:
            for channel_index in range(channels):
                output_path = output_dir / (
                    "%s.channel_%03d.wav" % (path.stem, channel_index)
                )
                writer = wave.open(str(output_path), "wb")
                writer.setnchannels(1)
                writer.setsampwidth(sample_width)
                writer.setframerate(sample_rate)
                writers.append(writer)
                output_paths.append(output_path)
            frame_size = channels * sample_width
            while True:
                raw = source.readframes(16000)
                if not raw:
                    break
                frame_count = len(raw) // frame_size
                for channel_index, writer in enumerate(writers):
                    mono = bytearray(frame_count * sample_width)
                    for frame_index in range(frame_count):
                        source_offset = frame_index * frame_size + channel_index * sample_width
                        target_offset = frame_index * sample_width
                        mono[target_offset:target_offset + sample_width] = raw[
                            source_offset:source_offset + sample_width
                        ]
                    writer.writeframes(mono)
        finally:
            for writer in writers:
                writer.close()
    return {
        "paths": output_paths,
        "channel_count": channels,
        "sample_rate_hz": sample_rate,
        "method": "python_wave_channel_split",
    }


def _split_multichannel_wav_ffmpeg(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _ffprobe_audio_info(path)
    channels = int(info["channels"])
    sample_rate = int(info["sample_rate"])
    if channels < 2:
        raise MossDiarizeError("MOSS channel purity check requires at least two channels")
    output_paths = []
    for channel_index in range(channels):
        output_path = output_dir / (
            "%s.channel_%03d.wav" % (path.stem, channel_index)
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "pan=mono|c0=c%s" % channel_index,
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
        try:
            subprocess.check_call(command)
        except OSError as exc:
            raise MossDiarizeError("ffmpeg is required for MOSS channel purity check") from exc
        except subprocess.CalledProcessError as exc:
            raise MossDiarizeError("ffmpeg channel split failed for %s" % path) from exc
        output_paths.append(output_path)
    return {
        "paths": output_paths,
        "channel_count": channels,
        "sample_rate_hz": sample_rate,
        "method": "ffmpeg_channel_split",
    }


def _mixdown_multichannel_wav_python(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("%s.merged_mono.wav" % path.stem)
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        if channels < 2:
            raise MossDiarizeError("merged-headset MOSS requires at least two channels")
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        if sample_width not in (1, 2, 4):
            raise MossDiarizeError("unsupported sample width for headset mixdown: %s" % sample_width)
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            frame_size = channels * sample_width
            while True:
                raw = source.readframes(16000)
                if not raw:
                    break
                frame_count = len(raw) // frame_size
                out = bytearray(frame_count * sample_width)
                for frame_index in range(frame_count):
                    total = 0
                    for channel_index in range(channels):
                        offset = frame_index * frame_size + channel_index * sample_width
                        total += _read_pcm_sample(raw, offset, sample_width)
                    mixed = int(round(float(total) / float(channels)))
                    target = frame_index * sample_width
                    out[target:target + sample_width] = _write_pcm_sample(mixed, sample_width)
                writer.writeframes(out)
    return {
        "path": output_path,
        "channel_count": channels,
        "sample_rate_hz": sample_rate,
        "method": "mean_channels_python_wave",
    }


def _mixdown_multichannel_wav_ffmpeg(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _ffprobe_audio_info(path)
    channels = int(info["channels"])
    sample_rate = int(info["sample_rate"])
    if channels < 2:
        raise MossDiarizeError("merged-headset MOSS requires at least two channels")
    output_path = output_dir / ("%s.merged_mono.wav" % path.stem)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    try:
        subprocess.check_call(command)
    except OSError as exc:
        raise MossDiarizeError("ffmpeg is required for non-PCM headset mixdown") from exc
    except subprocess.CalledProcessError as exc:
        raise MossDiarizeError("ffmpeg headset mixdown failed for %s" % path) from exc
    return {
        "path": output_path,
        "channel_count": channels,
        "sample_rate_hz": sample_rate,
        "method": "ffmpeg_downmix_mono",
    }


def _read_pcm_sample(raw, offset, sample_width):
    if sample_width == 1:
        return int(raw[offset]) - 128
    return int.from_bytes(raw[offset:offset + sample_width], "little", signed=True)


def _write_pcm_sample(value, sample_width):
    if sample_width == 1:
        value = max(-128, min(127, int(value)))
        return bytes([value + 128])
    if sample_width == 2:
        value = max(-32768, min(32767, int(value)))
        return int(value).to_bytes(2, "little", signed=True)
    if sample_width == 4:
        value = max(-2147483648, min(2147483647, int(value)))
        return int(value).to_bytes(4, "little", signed=True)
    raise MossDiarizeError("unsupported sample width for headset mixdown: %s" % sample_width)
