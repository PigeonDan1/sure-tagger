"""Shared audio preparation for FireRed VAD and AED tools."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from tagger.tools.acoustic_io import get_audio_info


SUPPORTED_SAMPLE_RATE_HZ = 16000


def prepare_firered_audio(
    audio_path,
    context=None,
    normalize_to_16k_mono_pcm=True,
    error_class=RuntimeError,
    tool_label="FireRed",
    temp_prefix="firered_",
):
    info = get_audio_info(audio_path, context)
    if (
        not normalize_to_16k_mono_pcm
        or (
            info.sample_rate_hz == SUPPORTED_SAMPLE_RATE_HZ
            and info.channels == 1
            and info.sample_width_bytes == 2
        )
    ):
        return str(audio_path), None

    if shutil.which("ffmpeg") is None:
        raise error_class(
            "ffmpeg is required to convert audio to 16kHz 16-bit mono PCM WAV"
        )

    tmpdir = tempfile.TemporaryDirectory(prefix=temp_prefix)
    converted_path = Path(tmpdir.name) / "input_16k_mono_pcm.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ar",
        str(SUPPORTED_SAMPLE_RATE_HZ),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(converted_path),
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        tmpdir.cleanup()
        raise error_class("ffmpeg conversion failed for %s" % tool_label) from exc

    return str(converted_path), tmpdir
