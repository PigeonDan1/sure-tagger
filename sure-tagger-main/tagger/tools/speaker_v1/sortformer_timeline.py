"""NVIDIA Streaming Sortformer adapter for an independent speaker timeline."""

import math
import time

from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_VERSION = "sortformer_timeline_v2.0-shadow.1"
CHECKPOINT_SHA256 = (
    "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329"
)
MAXIMUM_SPEAKERS = 4


class SortformerTimelineError(RuntimeError):
    pass


class SortformerTimelineConfig:
    def __init__(
        self,
        model_path,
        subprocess_python="",
        device="cuda:0",
        chunk_len=340,
        chunk_right_context=40,
        fifo_len=40,
        spkcache_update_period=340,
        spkcache_len=188,
        timeout_sec=600,
    ):
        self.model_path = str(model_path)
        self.subprocess_python = str(subprocess_python or "")
        self.device = str(device or "cuda:0")
        self.chunk_len = int(chunk_len)
        self.chunk_right_context = int(chunk_right_context)
        self.fifo_len = int(fifo_len)
        self.spkcache_update_period = int(spkcache_update_period)
        self.spkcache_len = int(spkcache_len)
        self.timeout_sec = int(timeout_sec)

    def to_record(self):
        return {
            "model_path": self.model_path,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "chunk_len": self.chunk_len,
            "chunk_right_context": self.chunk_right_context,
            "fifo_len": self.fifo_len,
            "spkcache_update_period": self.spkcache_update_period,
            "spkcache_len": self.spkcache_len,
            "timeout_sec": self.timeout_sec,
        }


class SortformerTimelineClient:
    def __init__(self, config):
        self.config = config
        self._model = None

    def diarize(self, audio_path, context=None):
        del context
        try:
            import torch
        except ImportError as exc:
            raise SortformerTimelineError(
                "Sortformer requires torch in its NeMo environment"
            ) from exc
        model = self._get_model()
        started = time.time()
        try:
            with torch.inference_mode():
                raw_segments, raw_probabilities = model.diarize(
                    audio=[str(audio_path)],
                    batch_size=1,
                    include_tensor_outputs=True,
                )
        except Exception as exc:
            raise SortformerTimelineError("Sortformer inference failed") from exc
        segments = _parse_segments(raw_segments)
        probabilities = _probability_payload(raw_probabilities)
        return {
            "segments": segments,
            "probabilities": probabilities,
            "runtime": {
                "elapsed_sec": round(time.time() - started, 6),
                "device": self.config.device,
            },
        }

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as exc:
            raise SortformerTimelineError(
                "Sortformer requires NeMo ASR and torch"
            ) from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise SortformerTimelineError("CUDA was requested but is unavailable")
        try:
            model = SortformerEncLabelModel.restore_from(
                restore_path=self.config.model_path,
                map_location=self.config.device,
                strict=False,
            )
            model.eval()
            modules = model.sortformer_modules
            modules.chunk_len = self.config.chunk_len
            modules.chunk_right_context = self.config.chunk_right_context
            modules.fifo_len = self.config.fifo_len
            modules.spkcache_update_period = self.config.spkcache_update_period
            modules.spkcache_len = self.config.spkcache_len
            modules._check_streaming_parameters()
        except Exception as exc:
            raise SortformerTimelineError("Sortformer model loading failed") from exc
        self._model = model
        return self._model


class SortformerTimelineSubprocessClient:
    def __init__(self, config):
        self.config = config

    def diarize(self, audio_path, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "sortformer_timeline_estimate",
            {
                "audio_path": str(audio_path),
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


def diarize(audio_path, config, context=None):
    client = (
        SortformerTimelineSubprocessClient(config)
        if config.subprocess_python
        else SortformerTimelineClient(config)
    )
    return client.diarize(audio_path, context=context)


def _parse_segments(raw_segments):
    items = raw_segments
    if isinstance(items, (list, tuple)) and len(items) == 1:
        items = items[0]
    parsed = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        fields = item.strip().rsplit(maxsplit=1)
        if len(fields) != 2:
            continue
        times, speaker_id = fields
        time_fields = times.split()
        if len(time_fields) != 2:
            continue
        start_sec = float(time_fields[0])
        end_sec = float(time_fields[1])
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            continue
        if end_sec <= start_sec:
            continue
        parsed.append(
            {
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "speaker_id": str(speaker_id),
            }
        )
    return parsed


def _probability_payload(raw_probabilities):
    try:
        import torch
    except ImportError as exc:
        raise SortformerTimelineError("probability export requires torch") from exc
    value = raw_probabilities
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise SortformerTimelineError("unexpected Sortformer probability shape")
    values = [
        [round(float(item), 6) for item in row]
        for row in tensor.tolist()
    ]
    return {
        "frame_stride_sec": 0.08,
        "shape": list(tensor.shape),
        "values": values,
        "per_slot_max": [
            round(float(item), 6) for item in tensor.max(dim=0).values.tolist()
        ],
        "active_slots_ge_0_5": int(
            ((tensor >= 0.5).sum(dim=0) > 0).sum().item()
        ),
    }


def _subprocess_config(config):
    result = config.to_record()
    result["subprocess_python"] = ""
    result.pop("timeout_sec", None)
    return result
