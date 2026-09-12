"""CAM++ speaker identity adapter for sample-local candidate regions."""

from pathlib import Path
import tempfile

from tagger.tools.subprocess_runner import run_subprocess_tool


TOOL_VERSION = "campplus_identity_v2.0-shadow.2"
CHECKPOINT_SHA256 = (
    "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"
)


class CampPlusIdentityError(RuntimeError):
    pass


class CampPlusIdentityConfig:
    def __init__(
        self,
        model_dir,
        subprocess_python="",
        device="cpu",
        threshold=0.5,
        min_region_duration_sec=0.80,
        max_regions_per_speaker=2,
        torch_num_threads=4,
        timeout_sec=300,
    ):
        self.model_dir = str(model_dir)
        self.subprocess_python = str(subprocess_python or "")
        self.device = str(device or "cpu")
        self.threshold = float(threshold)
        self.min_region_duration_sec = float(min_region_duration_sec)
        self.max_regions_per_speaker = int(max_regions_per_speaker)
        self.torch_num_threads = int(torch_num_threads)
        self.timeout_sec = int(timeout_sec)

    def to_record(self):
        return {
            "model_dir": self.model_dir,
            "subprocess_python": self.subprocess_python,
            "device": self.device,
            "threshold": self.threshold,
            "min_region_duration_sec": self.min_region_duration_sec,
            "max_regions_per_speaker": self.max_regions_per_speaker,
            "torch_num_threads": self.torch_num_threads,
            "timeout_sec": self.timeout_sec,
        }


class CampPlusIdentityClient:
    def __init__(self, config):
        self.config = config
        self._pipeline = None

    def compare_regions(self, audio_path, regions, context=None):
        del context
        verifier = self._get_pipeline()
        with tempfile.TemporaryDirectory(prefix="sure_tagger_campplus_") as tmpdir:
            crop_paths = _write_region_crops(audio_path, regions, Path(tmpdir))
            comparisons = []
            for left in range(len(regions)):
                for right in range(left + 1, len(regions)):
                    left_region = regions[left]
                    right_region = regions[right]
                    if left_region["speaker_id"] == right_region["speaker_id"]:
                        comparison_kind = "within_source_cluster"
                    else:
                        comparison_kind = "cross_source_cluster"
                    result = verifier(
                        [str(crop_paths[left]), str(crop_paths[right])],
                        output_emb=True,
                    )
                    outputs = result.get("outputs", {})
                    score = float(outputs.get("score"))
                    decision = "same" if score >= self.config.threshold else "different"
                    comparisons.append(
                        {
                            "comparison_kind": comparison_kind,
                            "region_ids": [
                                left_region["region_id"],
                                right_region["region_id"],
                            ],
                            "speaker_pair": [
                                left_region["speaker_id"],
                                right_region["speaker_id"],
                            ],
                            "score": round(score, 8),
                            "decision": decision,
                            "threshold": self.config.threshold,
                            "model_output_text": outputs.get("text"),
                        }
                    )
        return {"regions": regions, "comparisons": comparisons}

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from modelscope.pipelines import pipeline
            except ImportError as exc:
                raise CampPlusIdentityError(
                    "CAM++ requires torch and modelscope in its model environment"
                ) from exc
            torch.set_num_threads(self.config.torch_num_threads)
            try:
                self._pipeline = pipeline(
                    task="speaker-verification",
                    model=self.config.model_dir,
                    device=self.config.device,
                )
            except Exception as exc:
                raise CampPlusIdentityError("CAM++ model loading failed") from exc
        return self._pipeline


class CampPlusIdentitySubprocessClient:
    def __init__(self, config):
        self.config = config

    def compare_regions(self, audio_path, regions, context=None):
        result = run_subprocess_tool(
            self.config.subprocess_python,
            "campplus_identity_estimate",
            {
                "audio_path": str(audio_path),
                "regions": regions,
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


def compare(audio_path, timeline_summary, config, context=None):
    regions = select_candidate_regions(timeline_summary, config)
    if len(regions) < 2:
        raise CampPlusIdentityError(
            "fewer than two sufficiently long candidate regions"
        )
    client = (
        CampPlusIdentitySubprocessClient(config)
        if config.subprocess_python
        else CampPlusIdentityClient(config)
    )
    return client.compare_regions(audio_path, regions, context=context)


def select_candidate_regions(timeline_summary, config):
    by_speaker = {}
    source_segments = timeline_summary.get(
        "activity_segments", timeline_summary.get("segments", [])
    )
    excluded_intervals = _union_intervals(
        timeline_summary.get(
            "overlap_activity_segments",
            timeline_summary.get("overlap_segments", []),
        )
    )
    for index, segment in enumerate(source_segments):
        speaker_id = str(segment["speaker_id"])
        clean_regions = _subtract_intervals(
            float(segment["start_sec"]),
            float(segment["end_sec"]),
            excluded_intervals,
        )
        for fragment_index, (start_sec, end_sec) in enumerate(clean_regions):
            duration = end_sec - start_sec
            if duration < config.min_region_duration_sec:
                continue
            record = {
                "region_id": "region_%06d_%03d" % (index, fragment_index),
                "speaker_id": speaker_id,
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "duration_sec": round(duration, 6),
                "selection": "source_timeline_nonoverlap_candidate",
                "source_segment_index": index,
                "overlap_exclusion_applied": bool(excluded_intervals),
            }
            by_speaker.setdefault(speaker_id, []).append(record)
    selected = []
    for speaker_id in sorted(by_speaker):
        ranked = sorted(
            by_speaker[speaker_id],
            key=lambda item: (-item["duration_sec"], item["start_sec"]),
        )
        selected.extend(ranked[: config.max_regions_per_speaker])
    return sorted(selected, key=lambda item: (item["start_sec"], item["end_sec"]))


def _union_intervals(items):
    parsed = sorted(
        (
            float(item["start_sec"]),
            float(item["end_sec"]),
        )
        for item in items
        if float(item["end_sec"]) > float(item["start_sec"])
    )
    merged = []
    for start_sec, end_sec in parsed:
        if not merged or start_sec > merged[-1][1]:
            merged.append([start_sec, end_sec])
        else:
            merged[-1][1] = max(merged[-1][1], end_sec)
    return merged


def _subtract_intervals(start_sec, end_sec, excluded_intervals):
    if end_sec <= start_sec:
        return []
    cursor = start_sec
    clean = []
    for excluded_start, excluded_end in excluded_intervals:
        if excluded_end <= cursor:
            continue
        if excluded_start >= end_sec:
            break
        if excluded_start > cursor:
            clean.append((cursor, min(excluded_start, end_sec)))
        cursor = max(cursor, excluded_end)
        if cursor >= end_sec:
            break
    if cursor < end_sec:
        clean.append((cursor, end_sec))
    return clean


def _write_region_crops(audio_path, regions, output_dir):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise CampPlusIdentityError(
            "CAM++ region cropping requires numpy and soundfile"
        ) from exc
    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise CampPlusIdentityError("failed to decode CAM++ source audio") from exc
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    paths = []
    for index, region in enumerate(regions):
        start_frame = max(0, int(round(region["start_sec"] * sample_rate)))
        end_frame = min(len(audio), int(round(region["end_sec"] * sample_rate)))
        if end_frame <= start_frame:
            raise CampPlusIdentityError("CAM++ candidate region is empty")
        path = output_dir / ("region_%06d.wav" % index)
        sf.write(str(path), audio[start_frame:end_frame], sample_rate, subtype="PCM_16")
        paths.append(path)
    return paths


def _subprocess_config(config):
    result = config.to_record()
    result["subprocess_python"] = ""
    result.pop("timeout_sec", None)
    return result
