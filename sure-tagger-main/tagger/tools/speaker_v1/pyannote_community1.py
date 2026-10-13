"""Pinned, offline pyannote Community-1 timeline adapter for speaker v2."""

from __future__ import print_function

import hashlib
from pathlib import Path
import threading
import time

from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_VERSION = "pyannote_community1_v2.0-shadow.1"
MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
MODEL_VERSION = "%s@%s" % (MODEL_ID, MODEL_REVISION)
MODEL_SHA256 = {
    "config.yaml": "5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be",
    "embedding/pytorch_model.bin": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929",
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
    "segmentation/pytorch_model.bin": "7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e",
}
SUPPORTED_SAMPLE_RATE_HZ = 16000
_ASSET_HASH_CACHE = {}
_ASSET_HASH_LOCK = threading.Lock()
_TRUSTED_LOAD_PATCHED = False


class PyannoteCommunity1Error(RuntimeError):
    pass


class PyannoteCommunity1Config:
    def __init__(
        self,
        model_dir,
        subprocess_python="",
        device="cuda:0",
        min_activity_sec=0.10,
        timeout_sec=600,
        calibration_profile_id=None,
        joint_negative_profile_id=None,
        license_review_status="pending",
    ):
        self.model_dir = str(model_dir)
        self.subprocess_python = str(subprocess_python or "")
        self.device = str(device or "cuda:0")
        self.min_activity_sec = float(min_activity_sec)
        self.timeout_sec = int(timeout_sec)
        self.calibration_profile_id = (
            str(calibration_profile_id) if calibration_profile_id else None
        )
        self.joint_negative_profile_id = (
            str(joint_negative_profile_id) if joint_negative_profile_id else None
        )
        self.license_review_status = str(license_review_status or "pending")
        if self.min_activity_sec <= 0:
            raise ValueError("pyannote min_activity_sec must be positive")
        if self.timeout_sec <= 0:
            raise ValueError("pyannote timeout_sec must be positive")
        if self.license_review_status not in ("pending", "approved"):
            raise ValueError(
                "pyannote license_review_status must be pending or approved"
            )

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "min_activity_sec": self.min_activity_sec,
            "timeout_sec": self.timeout_sec,
            "calibration_profile_id": self.calibration_profile_id,
            "joint_negative_profile_id": self.joint_negative_profile_id,
            "license_review_status": self.license_review_status,
        }


class PyannoteCommunity1Client:
    def __init__(self, config):
        self.config = config
        self._pipeline = None

    def diarize(self, audio_path, context=None):
        del context
        try:
            import numpy as np
            import soundfile as sf
            import torch
        except ImportError as exc:
            raise PyannoteCommunity1Error(
                "Community-1 requires numpy, soundfile, and torch"
            ) from exc

        try:
            waveform, sample_rate = sf.read(
                str(audio_path), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise PyannoteCommunity1Error(
                "failed to decode Community-1 audio"
            ) from exc
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise PyannoteCommunity1Error("Community-1 audio is empty")
        if int(sample_rate) != SUPPORTED_SAMPLE_RATE_HZ:
            raise PyannoteCommunity1Error(
                "Community-1 expects 16 kHz audio, got %s" % sample_rate
            )

        pipeline = self._get_pipeline()
        tensor = torch.from_numpy(waveform.T.copy())
        started = time.time()
        try:
            with torch.inference_mode():
                output = pipeline(
                    {"waveform": tensor, "sample_rate": int(sample_rate)}
                )
        except Exception as exc:
            raise PyannoteCommunity1Error(
                "Community-1 inference failed: %s" % exc
            ) from exc
        return {
            "raw_segments": annotation_segments(output.speaker_diarization),
            "exclusive_segments": annotation_segments(
                output.exclusive_speaker_diarization
            ),
            "runtime": {
                "elapsed_sec": round(time.time() - started, 6),
                "device": self.config.device,
                "sample_rate_hz": int(sample_rate),
                "channel_count": int(waveform.shape[1]),
                "audio_decode": "soundfile_preloaded_waveform",
                "torchcodec_bypassed": True,
            },
        }

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch

            _patch_trusted_local_checkpoint_load()
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PyannoteCommunity1Error(
                "Community-1 requires pyannote.audio and torch"
            ) from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise PyannoteCommunity1Error(
                "CUDA was requested for Community-1 but is unavailable"
            )
        try:
            self._pipeline = Pipeline.from_pretrained(self.config.model_dir)
            if self.config.device.startswith("cuda"):
                self._pipeline.to(torch.device(self.config.device))
        except Exception as exc:
            raise PyannoteCommunity1Error(
                "Community-1 model loading failed: %s" % exc
            ) from exc
        return self._pipeline


class PyannoteCommunity1SubprocessClient:
    def __init__(self, config):
        self.config = config

    def diarize(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "pyannote_community1_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


def diarize(audio_path, config, context=None, client=None):
    client = client or (
        PyannoteCommunity1SubprocessClient(config)
        if config.subprocess_python
        else PyannoteCommunity1Client(config)
    )
    return client.diarize(audio_path, context=context)


def annotation_segments(annotation):
    segments = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        start_sec = float(segment.start)
        end_sec = float(segment.end)
        if end_sec <= start_sec:
            continue
        segments.append(
            {
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "speaker_id": str(label),
            }
        )
    return segments


def verify_model_assets(model_dir):
    model_dir = Path(str(model_dir)).expanduser().resolve()
    verified = {}
    for relative_path, expected in sorted(MODEL_SHA256.items()):
        path = model_dir / relative_path
        if not path.is_file():
            raise PyannoteCommunity1Error(
                "missing pinned Community-1 asset: %s" % path
            )
        stat = path.stat()
        cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        with _ASSET_HASH_LOCK:
            actual = _ASSET_HASH_CACHE.get(cache_key)
        if actual is None:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            actual = digest.hexdigest()
            with _ASSET_HASH_LOCK:
                _ASSET_HASH_CACHE[cache_key] = actual
        if actual != expected:
            raise PyannoteCommunity1Error(
                "Community-1 SHA256 mismatch for %s: expected %s, got %s"
                % (path, expected, actual)
            )
        verified[relative_path] = actual
    return verified


def _subprocess_config(config):
    result = config.to_record()
    result["subprocess_python"] = ""
    result.pop("timeout_sec", None)
    return result


def _patch_trusted_local_checkpoint_load():
    global _TRUSTED_LOAD_PATCHED
    if _TRUSTED_LOAD_PATCHED:
        return
    from lightning.fabric.utilities import cloud_io
    from lightning.pytorch.core import saving
    import pyannote.audio.core.model as model_core

    original_load = cloud_io._load

    def trusted_local_load(path_or_url, map_location=None, weights_only=None):
        del weights_only
        return original_load(
            path_or_url, map_location=map_location, weights_only=False
        )

    cloud_io._load = trusted_local_load
    saving.pl_load = trusted_local_load
    model_core.pl_load = trusted_local_load
    _TRUSTED_LOAD_PATCHED = True
