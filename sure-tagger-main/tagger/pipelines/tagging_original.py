"""Archived sample-level tagging pipeline retained as a source snapshot.

The active entry point is ``tagger.pipelines.tagging``. This module keeps the
former topic and PANNs registrations locally so their removal from active
registries does not change the archived source contract.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from tagger.input_schema import InputSchemaError, validate_input_record
from tagger.tools.base import ToolResult
from tagger.tools.subprocess_runner import close_subprocess_workers
from tagger.tools.audio_quality.brouhaha_signal_estimator import BrouhahaConfig
from tagger.tools.audio_quality.dnsmos_quality_estimator import DnsmosConfig
from tagger.tools.basic_acoustic.firered_vad_silence_detector import FireRedVadConfig
from tagger.tools.basic_acoustic.registry import (
    AUDIO_PROBE_TOOL,
    FIRERED_VAD_SILENCE_TOOL,
    NATIVE_METADATA_VAD_TOOL,
    NONSPEECH_RATIO_TOOL,
)
from tagger.tools.audio_quality.registry import (
    BROUHAHA_ACOUSTIC_TOOL,
    DNSMOS_QUALITY_TOOL,
)
from tagger.tools.room_acoustic.registry import (
    C50_TOOL,
    RECRIR_RIR_TOOL,
    RT60_TOOL,
)
from tagger.tools.sound_field_scene.registry import FIRERED_AED_TOOL
from tagger.tools.sound_field_scene.firered_aed_detector import (
    EVENT_NAMES,
    FireRedAedConfig,
)
from tagger.tools.sound_field_scene.panns_background_detector import (
    TOP_EVENTS_LIMIT,
    PannsBackgroundConfig,
)
from tagger.tools.sound_field_scene import panns_background_detector as legacy_panns
from tagger.tools.room_acoustic.rir_estimator import (
    RecRirConfig,
    validate_rir_payload,
)
from tagger.tools.speaker.artifacts import write_speaker_artifact
from tagger.tools.speaker.config import SpeakerLayerConfig, default_speaker_layer_config
from tagger.tools.speaker.metrics import (
    SpeakerMetricsConfig,
    build_metadata_from_channel_activity,
    build_metadata_from_timeline,
)
from tagger.tools.speaker.registry import (
    CHANNEL_ACTIVITY_TOOL,
    MOSS_DIARIZE_TOOL,
    NATIVE_METADATA_DIARIZE_TOOL,
    SPEAKER_METRICS_TOOL,
)
from tagger.tools.language_content.registry import (
    DETERMINISTIC_LANGUAGE_CONTENT_TOOL,
)
from tagger.tools.language_content.topic import TopicConfig
from tagger.tools.language_content import topic as legacy_topic


TOPIC_LANGUAGE_CONTENT_TOOL = {
    "tool_name": legacy_topic.TOOL_NAME,
    "tag_path": "language_content.topic",
    "run": legacy_topic.run,
}

PANNS_BACKGROUND_TOOL = {
    "tool_name": legacy_panns.TOOL_NAME,
    "tag_path": "sound_field_scene.sound",
    "run": legacy_panns.run,
}


BASIC_ACOUSTIC_FIELDS = {
    "duration_sec": None,
    "sample_rate_hz": None,
    "channels": None,
    "nonspeech_ratio": None,
    "silence_segments": None,
    "snr_db": None,
    "c50": None,
    "dnsmos_sig": None,
    "dnsmos_bak": None,
    "dnsmos_ovrl": None,
    "dnsmos_p808": None,
}

SOUND_FIELD_SCENE_FIELDS = {
    "far_field": None,
    "rt60": None,
    "c50": None,
    "audio_events": None,
    "music": None,
    "sound": None,
}

SPEAKER_FIELDS = {
    "multi_speaker": None,
    "speaker_change": None,
    "speaker_overlap": None,
}

LANGUAGE_CONTENT_FIELDS = {
    "topic": None,
    "language": None,
    "word_count": None,
    "punctuation": None,
    "repetition": None,
    "filler": None,
}

STAGE_LANGUAGE_DETERMINISTIC = "language_deterministic"
STAGE_TOPIC = "topic"
STAGE_AUDIO_PROBE = "audio_probe"
STAGE_SILENCE = "silence"
STAGE_SPEAKER = "speaker"
STAGE_BROUHAHA = "brouhaha"
STAGE_DNSMOS = "dnsmos"
STAGE_FIRERED_AED = "firered_aed"
STAGE_PANNS = "panns"
STAGE_RECRIR = "recrir"

FULL_STAGES = [
    STAGE_LANGUAGE_DETERMINISTIC,
    STAGE_TOPIC,
    STAGE_AUDIO_PROBE,
    STAGE_SILENCE,
    STAGE_SPEAKER,
    STAGE_BROUHAHA,
    STAGE_DNSMOS,
    STAGE_FIRERED_AED,
    STAGE_PANNS,
    STAGE_RECRIR,
]

AUDIO_STAGES = set(
    [
        STAGE_AUDIO_PROBE,
        STAGE_SILENCE,
        STAGE_SPEAKER,
        STAGE_BROUHAHA,
        STAGE_DNSMOS,
        STAGE_FIRERED_AED,
        STAGE_PANNS,
        STAGE_RECRIR,
    ]
)

STAGE_TAG_PATHS = {
    STAGE_LANGUAGE_DETERMINISTIC: [
        "language_content.language",
        "language_content.word_count",
        "language_content.punctuation",
        "language_content.repetition",
        "language_content.filler",
    ],
    STAGE_TOPIC: ["language_content.topic"],
    STAGE_AUDIO_PROBE: [
        "basic_acoustic.duration_sec",
        "basic_acoustic.sample_rate_hz",
        "basic_acoustic.channels",
    ],
    STAGE_SILENCE: [
        "basic_acoustic.silence_segments",
        "basic_acoustic.nonspeech_ratio",
    ],
    STAGE_SPEAKER: [
        "speaker.multi_speaker",
        "speaker.speaker_change",
        "speaker.speaker_overlap",
    ],
    STAGE_BROUHAHA: ["basic_acoustic.snr_db", "basic_acoustic.c50"],
    STAGE_DNSMOS: [
        "basic_acoustic.dnsmos_sig",
        "basic_acoustic.dnsmos_bak",
        "basic_acoustic.dnsmos_ovrl",
        "basic_acoustic.dnsmos_p808",
    ],
    STAGE_FIRERED_AED: [
        "sound_field_scene.audio_events",
        "sound_field_scene.music",
    ],
    STAGE_PANNS: ["sound_field_scene.sound"],
    STAGE_RECRIR: ["sound_field_scene.rt60", "sound_field_scene.c50"],
}

IMPLEMENTED_TAG_PATHS = set(
    path
    for tag_paths in STAGE_TAG_PATHS.values()
    for path in tag_paths
)

def run_manifest(
    manifest_path,
    output_path,
    firered_vad_config=None,
    brouhaha_config=None,
    recrir_config=None,
    speaker_config=None,
    moss_client=None,
    channel_activity_client=None,
    artifact_dir=None,
    dnsmos_config=None,
    firered_aed_config=None,
    panns_config=None,
    topic_config=None,
    topic_client=None,
    sample_ids=None,
    existing_tags_path=None,
    selected_tag_paths=None,
    missing_only=False,
):
    # type: (Union[str, Path], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerLayerConfig], Any, Any, Optional[Union[str, Path]], Optional[DnsmosConfig], Optional[FireRedAedConfig], Optional[PannsBackgroundConfig], Optional[TopicConfig], Any, Optional[List[str]], Optional[Union[str, Path]], Optional[List[str]], bool) -> Dict[str, Any]
    manifest = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = _resolve_artifact_root(output, artifact_dir)
    selected_sample_ids = set(str(item) for item in (sample_ids or []))
    existing_tags = (
        _load_existing_tags(existing_tags_path) if existing_tags_path is not None else None
    )
    requested_tag_paths = _normalize_selected_tag_paths(selected_tag_paths)

    count = 0
    processed_count = 0
    internal_warning_count = 0
    tool_context = {}  # type: Dict[str, Any]
    try:
        with manifest.open("r", encoding="utf-8") as source, output.open(
            "w", encoding="utf-8"
        ) as sink:
            for row_index, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample", {}).get("sample_id", ""))
                selected = not selected_sample_ids or sample_id in selected_sample_ids
                base_tags = (
                    _tags_from_existing_row(existing_tags, row_index)
                    if existing_tags is not None
                    else None
                )
                if not selected:
                    if base_tags is not None:
                        sink.write(
                            json.dumps(base_tags, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        count += 1
                    continue
                row_tag_paths = requested_tag_paths
                if base_tags is not None and row_tag_paths is None:
                    row_tag_paths = _missing_tag_paths(base_tags)
                if base_tags is not None and missing_only and row_tag_paths is not None:
                    row_tag_paths = _filter_missing_tag_paths(base_tags, row_tag_paths)
                try:
                    internal = _tag_record_internal(
                        record,
                        manifest_dir=manifest.parent,
                        firered_vad_config=firered_vad_config,
                        brouhaha_config=brouhaha_config,
                        recrir_config=recrir_config,
                        speaker_config=speaker_config,
                        moss_client=moss_client,
                        channel_activity_client=channel_activity_client,
                        artifact_dir=artifact_root,
                        artifact_record_index=row_index,
                        tool_context=tool_context,
                        dnsmos_config=dnsmos_config,
                        firered_aed_config=firered_aed_config,
                        panns_config=panns_config,
                        topic_config=topic_config,
                        topic_client=topic_client,
                        initial_tags=base_tags,
                        selected_tag_paths=row_tag_paths,
                    )
                except InputSchemaError as exc:
                    raise InputSchemaError("line %s: %s" % (row_index, exc))
                internal_warning_count += len(internal["warnings"])
                sink.write(
                    json.dumps(internal["tags"], ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                count += 1
                processed_count += 1
    finally:
        close_subprocess_workers(tool_context)

    return {
        "manifest_path": str(manifest),
        "output_path": str(output),
        "artifact_dir": str(artifact_root),
        "sample_count": count,
        "processed_sample_count": processed_count,
        "internal_warning_count": internal_warning_count,
    }


def tag_record(
    record,
    manifest_dir,
    firered_vad_config=None,
    brouhaha_config=None,
    recrir_config=None,
    speaker_config=None,
    recrir_client=None,
    moss_client=None,
    channel_activity_client=None,
    artifact_dir=None,
    dnsmos_config=None,
    dnsmos_client=None,
    firered_aed_config=None,
    firered_aed_client=None,
    panns_config=None,
    panns_client=None,
    topic_config=None,
    topic_client=None,
    initial_tags=None,
    selected_tag_paths=None,
):
    # type: (Dict[str, Any], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerLayerConfig], Any, Any, Any, Any, Optional[Union[str, Path]], Optional[DnsmosConfig], Any, Optional[FireRedAedConfig], Any, Optional[PannsBackgroundConfig], Any, Optional[TopicConfig], Any, Optional[Dict[str, Any]], Optional[List[str]]) -> Dict[str, Any]
    return _tag_record_internal(
        record,
        manifest_dir,
        firered_vad_config=firered_vad_config,
        brouhaha_config=brouhaha_config,
        recrir_config=recrir_config,
        speaker_config=speaker_config,
        recrir_client=recrir_client,
        moss_client=moss_client,
        channel_activity_client=channel_activity_client,
        artifact_dir=artifact_dir,
        dnsmos_config=dnsmos_config,
        dnsmos_client=dnsmos_client,
        firered_aed_config=firered_aed_config,
        firered_aed_client=firered_aed_client,
        panns_config=panns_config,
        panns_client=panns_client,
        topic_config=topic_config,
        topic_client=topic_client,
        initial_tags=initial_tags,
        selected_tag_paths=selected_tag_paths,
    )["tags"]


def _tag_record_internal(
    record,
    manifest_dir,
    firered_vad_config=None,
    brouhaha_config=None,
    recrir_config=None,
    speaker_config=None,
    recrir_client=None,
    moss_client=None,
    channel_activity_client=None,
    artifact_dir=None,
    artifact_record_index=None,
    tool_context=None,
    dnsmos_config=None,
    dnsmos_client=None,
    firered_aed_config=None,
    firered_aed_client=None,
    panns_config=None,
    panns_client=None,
    topic_config=None,
    topic_client=None,
    initial_tags=None,
    selected_tag_paths=None,
):
    # type: (Dict[str, Any], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerLayerConfig], Any, Any, Any, Optional[Union[str, Path]], Optional[int], Optional[Dict[str, Any]], Optional[DnsmosConfig], Any, Optional[FireRedAedConfig], Any, Optional[PannsBackgroundConfig], Any, Optional[TopicConfig], Any, Optional[Dict[str, Any]], Optional[List[str]]) -> Dict[str, Any]
    validate_input_record(record)
    sample = record["sample"]
    sample_id = sample["sample_id"]
    audio_path = resolve_audio_path(sample, manifest_dir)

    tags = _merge_tags(initial_tags)
    internal_results = []  # type: List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]
    stages = _stages_for_tag_paths(selected_tag_paths, tags)
    stages = _apply_no_transcript_speech_stage_guard(sample, tags, stages, warnings)

    if STAGE_LANGUAGE_DETERMINISTIC in stages or STAGE_TOPIC in stages:
        _run_language_content_tools(
            record,
            tags,
            internal_results,
            warnings,
            sample_id,
            topic_config=topic_config,
            topic_client=topic_client,
            stages=stages,
        )

    needs_audio = bool(stages & AUDIO_STAGES)
    if not needs_audio:
        pass
    elif audio_path is None:
        warnings.append(
            {
                "type": "missing_audio_path",
                "message": "sample.audio.path is empty",
                "sample_id": sample_id,
            }
        )
    elif not audio_path.exists():
        warnings.append(
            {
                "type": "missing_audio_file",
                "message": "audio file does not exist",
                "sample_id": sample_id,
                "audio_path": str(audio_path),
            }
        )
    else:
        tool_context = tool_context if tool_context is not None else {}
        if _needs_audio_probe(stages, tags):
            _run_audio_probe(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
            )
        if STAGE_SILENCE in stages:
            _run_silence_tools(
                audio_path,
                sample,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                firered_vad_config=firered_vad_config,
            )
        if STAGE_SPEAKER in stages:
            _run_speaker_tools(
                audio_path,
                sample,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                speaker_config=speaker_config,
                moss_client=moss_client,
                channel_activity_client=channel_activity_client,
                artifact_dir=artifact_dir,
                artifact_record_index=artifact_record_index,
            )
        if STAGE_BROUHAHA in stages:
            _run_brouhaha_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                brouhaha_config=brouhaha_config,
            )
        if STAGE_DNSMOS in stages:
            _run_dnsmos_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                dnsmos_config=dnsmos_config,
                dnsmos_client=dnsmos_client,
            )
        if STAGE_FIRERED_AED in stages:
            _run_firered_aed_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                firered_aed_config=firered_aed_config,
                firered_aed_client=firered_aed_client,
            )
        if STAGE_PANNS in stages:
            _run_panns_background_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                panns_config=panns_config,
                panns_client=panns_client,
            )
        if STAGE_RECRIR in stages:
            _run_recrir_tools(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                recrir_config=recrir_config,
                recrir_client=recrir_client,
                artifact_dir=artifact_dir,
                artifact_record_index=artifact_record_index,
            )

    warnings.extend(
        compare_native_metadata_basic_acoustic_fields(sample, tags["basic_acoustic"])
    )
    warnings.extend(
        compare_native_metadata_sound_field_scene_fields(
            sample,
            tags["sound_field_scene"],
        )
    )
    warnings.extend(audit_basic_acoustic(tags["basic_acoustic"]))
    warnings.extend(audit_sound_field_scene(tags["sound_field_scene"]))
    warnings.extend(audit_speaker(tags["speaker"]))
    warnings.extend(audit_language_content(tags["language_content"]))

    return {
        "tags": tags,
        "internal_results": internal_results,
        "warnings": warnings,
    }


def empty_tags():
    # type: () -> Dict[str, Any]
    return {
        "basic_acoustic": dict(BASIC_ACOUSTIC_FIELDS),
        "sound_field_scene": dict(SOUND_FIELD_SCENE_FIELDS),
        "speaker": dict(SPEAKER_FIELDS),
        "language_content": dict(LANGUAGE_CONTENT_FIELDS),
    }


def _merge_tags(initial_tags=None):
    # type: (Optional[Dict[str, Any]]) -> Dict[str, Any]
    tags = empty_tags()
    if not isinstance(initial_tags, dict):
        return tags
    for group, fields in tags.items():
        incoming = initial_tags.get(group)
        if not isinstance(incoming, dict):
            continue
        for field in fields:
            if field in incoming:
                tags[group][field] = incoming[field]
    return tags


def _load_existing_tags(path):
    # type: (Union[str, Path]) -> List[Dict[str, Any]]
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for index, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(_merge_tags(json.loads(line)))
            except ValueError as exc:
                raise ValueError("existing tags line %s is invalid JSON" % index) from exc
    return rows


def _tags_from_existing_row(existing_tags, row_index):
    # type: (Optional[List[Dict[str, Any]]], int) -> Optional[Dict[str, Any]]
    if existing_tags is None:
        return None
    if row_index > len(existing_tags):
        raise ValueError("existing tags has no row %s" % row_index)
    return _merge_tags(existing_tags[row_index - 1])


def _normalize_selected_tag_paths(selected_tag_paths):
    # type: (Optional[List[str]]) -> Optional[List[str]]
    if selected_tag_paths is None:
        return None
    values = []
    for item in selected_tag_paths:
        for part in str(item).split(","):
            value = part.strip()
            if value:
                values.append(value)
    return values


def _missing_tag_paths(tags):
    # type: (Dict[str, Any]) -> List[str]
    paths = []
    for group, fields in _merge_tags(tags).items():
        for field, value in fields.items():
            path = "%s.%s" % (group, field)
            if value is None and path in IMPLEMENTED_TAG_PATHS:
                paths.append(path)
    return paths


def _filter_missing_tag_paths(tags, tag_paths):
    # type: (Dict[str, Any], List[str]) -> List[str]
    result = []
    for path in tag_paths:
        value = _tag_value(tags, path)
        if value is None:
            result.append(path)
    return result


def _apply_no_transcript_speech_stage_guard(sample, tags, stages, warnings):
    # type: (Dict[str, Any], Dict[str, Any], set, List[Dict[str, Any]]) -> set
    if _has_transcript(sample):
        return stages
    skipped = stages & set(
        [
            STAGE_LANGUAGE_DETERMINISTIC,
            STAGE_TOPIC,
            STAGE_SILENCE,
            STAGE_SPEAKER,
            STAGE_DNSMOS,
            STAGE_RECRIR,
        ]
    )
    if not skipped:
        return stages
    stages = set(stages) - skipped
    _null_language_content_tags(tags)
    _null_silence_tags(tags)
    _null_dnsmos_tags(tags)
    _null_recrir_tags(tags)
    _set_no_transcript_speaker_tags(tags)
    warnings.append(
        {
            "type": "speech_dependent_stages_skipped_no_transcript",
            "message": (
                "sample.text.transcript is empty; treating the sample as non-speech "
                "noise and skipping language, VAD, DNSMOS, speaker model, and "
                "Rec-RIR stages"
            ),
            "skipped_stages": sorted(skipped),
        }
    )
    return stages


def _has_transcript(sample):
    # type: (Dict[str, Any]) -> bool
    transcript = sample.get("text", {}).get("transcript")
    return bool(isinstance(transcript, str) and transcript.strip())


def _tag_value(tags, tag_path):
    # type: (Dict[str, Any], str) -> Any
    if "." not in tag_path:
        return None
    group, field = tag_path.split(".", 1)
    value = tags.get(group, {})
    if not isinstance(value, dict):
        return None
    return value.get(field)


def _stages_for_tag_paths(selected_tag_paths, tags=None):
    # type: (Optional[List[str]], Optional[Dict[str, Any]]) -> set
    if selected_tag_paths is None:
        return set(FULL_STAGES)
    stages = set()
    for value in _normalize_selected_tag_paths(selected_tag_paths) or []:
        if value in ("all", "*"):
            return set(FULL_STAGES)
        if value in STAGE_TAG_PATHS:
            stages.add(value)
            continue
        if value in ("basic_acoustic", "sound_field_scene", "speaker", "language_content"):
            stages.update(_stages_for_group(value))
            continue
        matched = False
        for stage, tag_paths in STAGE_TAG_PATHS.items():
            if value in tag_paths:
                stages.add(stage)
                matched = True
        if not matched:
            raise ValueError("unknown tag path or stage: %s" % value)
    _add_dependency_stages(stages, tags or empty_tags())
    return stages


def _stages_for_group(group):
    stages = set()
    for stage, tag_paths in STAGE_TAG_PATHS.items():
        if any(path.startswith(group + ".") for path in tag_paths):
            stages.add(stage)
    return stages


def _add_dependency_stages(stages, tags):
    if any(stage in stages for stage in AUDIO_STAGES - set([STAGE_AUDIO_PROBE])):
        if _needs_audio_probe(stages, tags):
            stages.add(STAGE_AUDIO_PROBE)


def _needs_audio_probe(stages, tags):
    # type: (set, Dict[str, Any]) -> bool
    if STAGE_AUDIO_PROBE in stages:
        return True
    if not stages & AUDIO_STAGES:
        return False
    basic = tags.get("basic_acoustic", {})
    if STAGE_SPEAKER in stages:
        return (
            basic.get("duration_sec") is None
            or basic.get("channels") is None
        )
    if STAGE_SILENCE in stages or STAGE_FIRERED_AED in stages:
        return basic.get("duration_sec") is None
    return False


def resolve_audio_path(sample, manifest_dir):
    # type: (Dict[str, Any], Union[str, Path]) -> Optional[Path]
    raw_path = sample["audio"]["path"]
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    manifest_path = Path(manifest_dir) / path
    if manifest_path.exists():
        return manifest_path

    return cwd_path


def apply_result(tags, internal_results, result):
    # type: (Dict[str, Any], List[Dict[str, Any]], ToolResult) -> None
    prefix, field = result.tag_path.split(".", 1)
    tags[prefix][field] = result.value
    internal_results.append(result.to_record())


def _run_audio_probe(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
):
    try:
        results = AUDIO_PROBE_TOOL["run"](audio_path, context=tool_context)
        for result in results:
            apply_result(tags, internal_results, result)
    except Exception as exc:  # noqa: BLE001 - tool failures become internal warnings.
        warnings.append(
            {
                "type": "basic_acoustic_tool_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": AUDIO_PROBE_TOOL["tool_name"],
            }
        )
        for field in ("duration_sec", "sample_rate_hz", "channels"):
            tags["basic_acoustic"][field] = None


def _run_silence_tools(
    audio_path,
    sample,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    firered_vad_config=None,
):
    duration_sec = tags["basic_acoustic"]["duration_sec"]
    if duration_sec is None or duration_sec <= 0:
        warnings.append(
            {
                "type": "invalid_duration_for_silence",
                "message": "duration_sec must be positive before FireRed VAD",
                "sample_id": sample_id,
                "audio_path": str(audio_path),
            }
        )
        tags["basic_acoustic"]["silence_segments"] = None
        tags["basic_acoustic"]["nonspeech_ratio"] = None
        return

    metadata_result = None
    try:
        metadata_result = NATIVE_METADATA_VAD_TOOL["run"](
            sample,
            duration_sec=duration_sec,
            context=tool_context,
        )
        apply_result(tags, internal_results, metadata_result)
    except Exception:
        metadata_result = None

    if metadata_result is None:
        try:
            silence_result = FIRERED_VAD_SILENCE_TOOL["run"](
                audio_path,
                duration_sec=duration_sec,
                context=tool_context,
                config=firered_vad_config,
            )
            apply_result(tags, internal_results, silence_result)
        except Exception as exc:  # noqa: BLE001 - no non-FireRed fallback is allowed.
            warnings.append(
                {
                    "type": "firered_vad_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": FIRERED_VAD_SILENCE_TOOL["tool_name"],
                }
            )
            tags["basic_acoustic"]["silence_segments"] = None
            tags["basic_acoustic"]["nonspeech_ratio"] = None
            return

    try:
        ratio_result = NONSPEECH_RATIO_TOOL["run"](
            tags["basic_acoustic"]["silence_segments"],
            duration_sec=duration_sec,
        )
        apply_result(tags, internal_results, ratio_result)
    except Exception as exc:  # noqa: BLE001 - invalid FireRed output nulls both silence tags.
        warnings.append(
            {
                "type": "nonspeech_ratio_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": NONSPEECH_RATIO_TOOL["tool_name"],
            }
        )
        tags["basic_acoustic"]["silence_segments"] = None
        tags["basic_acoustic"]["nonspeech_ratio"] = None


def _run_language_content_tools(
    record,
    tags,
    internal_results,
    warnings,
    sample_id,
    topic_config=None,
    topic_client=None,
    stages=None,
):
    # type: (Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[TopicConfig], Any, Optional[set]) -> None
    stages = stages or set([STAGE_LANGUAGE_DETERMINISTIC, STAGE_TOPIC])
    sample = record["sample"]
    transcript = sample.get("text", {}).get("transcript", "")
    if STAGE_LANGUAGE_DETERMINISTIC in stages:
        try:
            for result in DETERMINISTIC_LANGUAGE_CONTENT_TOOL["run"](transcript):
                apply_result(tags, internal_results, result)
        except Exception as exc:  # noqa: BLE001 - tool failures become internal warnings.
            warnings.append(
                {
                    "type": "language_content_tool_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "tool_name": DETERMINISTIC_LANGUAGE_CONTENT_TOOL["tool_name"],
                }
            )
            for field in ("language", "word_count", "punctuation", "repetition", "filler"):
                tags["language_content"][field] = None

    config = topic_config or TopicConfig()
    if STAGE_TOPIC not in stages or not config.enabled:
        return
    try:
        result = TOPIC_LANGUAGE_CONTENT_TOOL["run"](
            record,
            context={},
            config=config,
            client=topic_client,
        )
        apply_result(tags, internal_results, result)
        if result.status == "failed":
            warnings.append(
                {
                    "type": "topic_tool_failed",
                    "message": result.evidence.get("error", "topic tool failed"),
                    "sample_id": sample_id,
                    "tool_name": TOPIC_LANGUAGE_CONTENT_TOOL["tool_name"],
                }
            )
    except Exception as exc:  # noqa: BLE001 - topic failures become internal warnings.
        warnings.append(
            {
                "type": "topic_tool_error",
                "message": str(exc),
                "sample_id": sample_id,
                "tool_name": TOPIC_LANGUAGE_CONTENT_TOOL["tool_name"],
            }
        )
        tags["language_content"]["topic"] = None


def _run_speaker_tools(
    audio_path,
    sample,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    speaker_config=None,
    moss_client=None,
    channel_activity_client=None,
    artifact_dir=None,
    artifact_record_index=None,
):
    # type: (Path, Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[SpeakerLayerConfig], Any, Any, Optional[Union[str, Path]], Optional[int]) -> None
    duration_sec = tags["basic_acoustic"].get("duration_sec")
    channels = tags["basic_acoustic"].get("channels")
    if duration_sec is None or duration_sec <= 0:
        warnings.append(
            {
                "type": "invalid_duration_for_speaker",
                "message": "duration_sec must be positive before speaker tools",
                "sample_id": sample_id,
                "audio_path": str(audio_path),
            }
        )
        _null_speaker_tags(tags)
        return

    config = speaker_config or default_speaker_layer_config(enable_moss=False)
    target_units = _speaker_target_units_from_native_metadata(sample)
    recording_id = _speaker_recording_id(sample, sample_id)
    input_kind = _speaker_input_kind(sample, channels)
    metrics_config = SpeakerMetricsConfig(
        min_segment_duration_sec=config.channel_activity_config.min_segment_duration_sec,
        merge_same_speaker_gap_sec=config.channel_activity_config.merge_gap_sec,
        min_speech_duration_sec=config.channel_activity_config.min_segment_duration_sec,
    )
    try:
        _run_native_metadata_speaker_route(
            sample,
            tags,
            internal_results,
            warnings,
            sample_id,
            duration_sec,
            metrics_config,
            recording_id,
            input_kind,
            target_units,
            artifact_dir,
            artifact_record_index,
        )
        return
    except Exception as exc:  # noqa: BLE001 - invalid metadata falls back to models.
        if _has_native_speaker_segment_metadata(sample):
            warnings.append(
                {
                    "type": "native_metadata_speaker_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": NATIVE_METADATA_DIARIZE_TOOL["tool_name"],
                }
            )

    separated_channel_input = (
        channels is not None
        and channels > 1
        and input_kind == "separated_headset_channels"
    )
    channel_candidate = config.enable_channel_activity and separated_channel_input
    channel_error = None
    moss_available = config.enable_moss or moss_client is not None

    if separated_channel_input:
        channels_are_single_speaker = bool(
            config.force_channel_activity or config.prefer_channel_activity
        )
        if (
            not channels_are_single_speaker
            and moss_available
            and config.run_moss_for_channel_qa
        ):
            try:
                purity_result = MOSS_DIARIZE_TOOL["run_channel_purity_check"](
                    audio_path,
                    duration_sec=duration_sec,
                    context=tool_context,
                    config=config.moss_config,
                    client=moss_client,
                )
                internal_results.append(purity_result.to_record())
                channels_are_single_speaker = bool(
                    purity_result.value.get("all_channels_single_speaker")
                )
            except Exception as exc:  # noqa: BLE001 - QA failures select the mixed route.
                warnings.append(
                    {
                        "type": "moss_channel_purity_check_error",
                        "message": str(exc),
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "tool_name": MOSS_DIARIZE_TOOL["tool_name"],
                    }
                )

        if channels_are_single_speaker and channel_candidate:
            try:
                _run_channel_activity_speaker_route(
                    audio_path,
                    tool_context,
                    tags,
                    internal_results,
                    warnings,
                    sample_id,
                    duration_sec,
                    config,
                    metrics_config,
                    recording_id,
                    input_kind,
                    target_units,
                    channel_activity_client,
                    artifact_dir,
                    artifact_record_index,
                )
                return
            except Exception as exc:  # noqa: BLE001 - speaker failures become internal warnings.
                channel_error = exc
                warnings.append(
                    {
                        "type": "channel_activity_error",
                        "message": str(exc),
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "tool_name": CHANNEL_ACTIVITY_TOOL["tool_name"],
                    }
                )

        if moss_available:
            try:
                _run_moss_merged_headset_speaker_route(
                    audio_path,
                    tool_context,
                    tags,
                    internal_results,
                    warnings,
                    sample_id,
                    duration_sec,
                    config,
                    metrics_config,
                    recording_id,
                    input_kind,
                    target_units,
                    moss_client,
                    artifact_dir,
                    artifact_record_index,
                )
                return
            except Exception as exc:  # noqa: BLE001 - speaker failures become internal warnings.
                warnings.append(
                    {
                        "type": "moss_merged_headset_diarize_error",
                        "message": str(exc),
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "tool_name": MOSS_DIARIZE_TOOL["tool_name"],
                    }
                )

        if channel_error is None and not moss_available:
            if channels_are_single_speaker and not config.enable_channel_activity:
                warning_type = "speaker_channel_activity_disabled"
                message = "Channel activity is disabled for asserted single-speaker channels"
            else:
                warning_type = "speaker_channel_purity_not_configured"
                message = (
                    "Separated-headset channel activity requires MOSS channel "
                    "purity verification or an explicit single-speaker-per-channel assertion"
                )
            warnings.append(
                {
                    "type": warning_type,
                    "message": message,
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                }
            )
        _null_speaker_tags(tags)
        return

    if not separated_channel_input and moss_available:
        try:
            _run_moss_speaker_route(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                duration_sec,
                config,
                metrics_config,
                recording_id,
                input_kind,
                target_units,
                moss_client,
                artifact_dir,
                artifact_record_index,
            )
            return
        except Exception as exc:  # noqa: BLE001 - speaker failures become internal warnings.
            warnings.append(
                {
                    "type": "moss_diarize_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": MOSS_DIARIZE_TOOL["tool_name"],
                }
            )

    if not moss_available:
        warnings.append(
            {
                "type": "speaker_diarization_not_configured",
                "message": "MOSS diarize is disabled and channel route did not apply",
                "sample_id": sample_id,
                "audio_path": str(audio_path),
            }
        )
    _null_speaker_tags(tags)


def _run_native_metadata_speaker_route(
    sample,
    tags,
    internal_results,
    warnings,
    sample_id,
    duration_sec,
    metrics_config,
    recording_id,
    input_kind,
    target_units,
    artifact_dir,
    artifact_record_index,
):
    native_result = NATIVE_METADATA_DIARIZE_TOOL["run"](
        sample,
        duration_sec=duration_sec,
        config=metrics_config,
    )
    internal_results.append(native_result.to_record())
    source_key = native_result.value.get("source_key")
    if source_key == "utterances" and not _has_explicit_target_units(sample):
        target_units = []
    metadata = build_metadata_from_timeline(
        native_result.value.get("segments", []),
        duration_sec,
        sample_id,
        recording_id=recording_id,
        input_kind=input_kind,
        primary_route="native_metadata_segments",
        target_units=target_units,
        config=metrics_config,
    )
    _write_speaker_metadata_artifact(
        metadata,
        internal_results,
        warnings,
        artifact_dir,
        sample_id,
        artifact_record_index,
    )
    for result in SPEAKER_METRICS_TOOL["run"](metadata):
        apply_result(tags, internal_results, result)


def _run_channel_activity_speaker_route(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    duration_sec,
    config,
    metrics_config,
    recording_id,
    input_kind,
    target_units,
    channel_activity_client,
    artifact_dir,
    artifact_record_index,
):
    channel_result = CHANNEL_ACTIVITY_TOOL["run"](
        audio_path,
        duration_sec=duration_sec,
        context=tool_context,
        config=config.channel_activity_config,
        client=channel_activity_client,
    )
    internal_results.append(channel_result.to_record())
    metadata = build_metadata_from_channel_activity(
        channel_result.value,
        duration_sec,
        sample_id,
        recording_id=recording_id,
        input_kind=input_kind,
        target_units=target_units,
        config=metrics_config,
    )
    _write_speaker_metadata_artifact(
        metadata,
        internal_results,
        warnings,
        artifact_dir,
        sample_id,
        artifact_record_index,
    )
    for result in SPEAKER_METRICS_TOOL["run"](metadata):
        apply_result(tags, internal_results, result)


def _run_moss_merged_headset_speaker_route(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    duration_sec,
    config,
    metrics_config,
    recording_id,
    input_kind,
    target_units,
    moss_client,
    artifact_dir,
    artifact_record_index,
):
    moss_result = MOSS_DIARIZE_TOOL["run_merged_channels"](
        audio_path,
        duration_sec=duration_sec,
        context=tool_context,
        config=config.moss_config,
        client=moss_client,
    )
    internal_results.append(moss_result.to_record())
    metadata = build_metadata_from_timeline(
        moss_result.value.get("segments", []),
        duration_sec,
        sample_id,
        recording_id=recording_id,
        input_kind=input_kind,
        primary_route="moss_diarize_merged_headset",
        target_units=target_units,
        config=metrics_config,
    )
    _write_speaker_metadata_artifact(
        metadata,
        internal_results,
        warnings,
        artifact_dir,
        sample_id,
        artifact_record_index,
    )
    for result in SPEAKER_METRICS_TOOL["run"](metadata):
        apply_result(tags, internal_results, result)


def _run_moss_speaker_route(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    duration_sec,
    config,
    metrics_config,
    recording_id,
    input_kind,
    target_units,
    moss_client,
    artifact_dir,
    artifact_record_index,
):
    moss_result = MOSS_DIARIZE_TOOL["run"](
        audio_path,
        duration_sec=duration_sec,
        context=tool_context,
        config=config.moss_config,
        client=moss_client,
    )
    internal_results.append(moss_result.to_record())
    metadata = build_metadata_from_timeline(
        moss_result.value.get("segments", []),
        duration_sec,
        sample_id,
        recording_id=recording_id,
        input_kind=input_kind,
        primary_route="moss_diarize",
        target_units=target_units,
        config=metrics_config,
    )
    _write_speaker_metadata_artifact(
        metadata,
        internal_results,
        warnings,
        artifact_dir,
        sample_id,
        artifact_record_index,
    )
    for result in SPEAKER_METRICS_TOOL["run"](metadata):
        apply_result(tags, internal_results, result)


def _write_speaker_metadata_artifact(
    metadata,
    internal_results,
    warnings,
    artifact_dir,
    sample_id,
    artifact_record_index,
):
    evidence = {
        "metadata_version": metadata.get("metadata_version"),
        "primary_route": metadata.get("primary_route"),
        "input_kind": metadata.get("input_kind"),
    }
    if artifact_dir is not None:
        try:
            artifact_path = write_speaker_artifact(
                metadata,
                Path(artifact_dir) / "speaker",
                _artifact_sample_key(sample_id, artifact_record_index),
                route=metadata.get("primary_route"),
            )
            evidence["artifact_path"] = str(artifact_path)
            evidence["artifact_format"] = "json.gz"
        except Exception as exc:  # noqa: BLE001 - artifact failure is internal.
            warnings.append(
                {
                    "type": "speaker_artifact_write_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                }
            )
    internal_results.append(
        ToolResult(
            tag_path="speaker.metadata",
            value=metadata,
            tool_name="speaker_metadata_builder",
            method="speaker_metadata_v0.1",
            status="estimated",
            confidence=1.0,
            tool_type="derived",
            evidence=evidence,
        ).to_record()
    )


def _speaker_target_units_from_native_metadata(sample):
    native_metadata = sample.get("native_metadata", {})
    for key in ("target_units", "utterances"):
        value = native_metadata.get(key)
        if isinstance(value, list):
            return value
    start = _metadata_float(native_metadata.get("start_sec", native_metadata.get("start")))
    end = _metadata_float(native_metadata.get("end_sec", native_metadata.get("end")))
    if start is not None and end is not None and end > start:
        unit_id = (
            sample.get("sample_id")
            or native_metadata.get("utt_id")
            or native_metadata.get("utterance_id")
        )
        return [
            {
                "unit_id": str(unit_id),
                "start_sec": 0.0,
                "end_sec": round(end - start, 6),
            }
        ]
    return []


def _has_explicit_target_units(sample):
    native_metadata = sample.get("native_metadata", {})
    return isinstance(native_metadata.get("target_units"), list)


def _has_native_speaker_segment_metadata(sample):
    native_metadata = sample.get("native_metadata", {})
    if not isinstance(native_metadata, dict):
        return False
    for key in ("speaker_segments", "diarization_segments", "segments", "utterances"):
        value = native_metadata.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _metadata_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _speaker_recording_id(sample, sample_id):
    native_metadata = sample.get("native_metadata", {})
    for key in ("recording_id", "meeting_id", "audio_id"):
        value = native_metadata.get(key)
        if value:
            return str(value)
    return sample_id


def _speaker_input_kind(sample, channels):
    native_metadata = sample.get("native_metadata", {})
    microphone_type = str(native_metadata.get("microphone_type", "")).lower()
    raw_path = str(sample.get("audio", {}).get("path", "")).lower()
    if "mix-headset" in microphone_type or "mix-headset" in raw_path:
        return "mix_headset"
    if "headset" in microphone_type or "headset" in raw_path:
        if channels is not None and channels > 1:
            return "separated_headset_channels"
        return "separated_headset_files"
    if channels is not None and channels > 1:
        return "separated_headset_channels"
    return "unknown_audio_layout"


def _null_speaker_tags(tags):
    for field in SPEAKER_FIELDS:
        tags["speaker"][field] = None


def _set_no_transcript_speaker_tags(tags):
    tags["speaker"]["multi_speaker"] = False
    tags["speaker"]["speaker_change"] = False
    tags["speaker"]["speaker_overlap"] = False


def _null_language_content_tags(tags):
    for field in LANGUAGE_CONTENT_FIELDS:
        tags["language_content"][field] = None


def _null_silence_tags(tags):
    tags["basic_acoustic"]["silence_segments"] = None
    tags["basic_acoustic"]["nonspeech_ratio"] = None


def _null_recrir_tags(tags):
    tags["sound_field_scene"]["rt60"] = None
    tags["sound_field_scene"]["c50"] = None


def _run_brouhaha_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    brouhaha_config=None,
):
    try:
        results = BROUHAHA_ACOUSTIC_TOOL["run"](
            audio_path,
            context=tool_context,
            config=brouhaha_config,
        )
        for result in results:
            apply_result(tags, internal_results, result)
            if result.status != "estimated":
                warnings.append(
                    {
                        "type": "brouhaha_output_invalid",
                        "message": result.evidence.get("error", "invalid output"),
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "tool_name": BROUHAHA_ACOUSTIC_TOOL["tool_name"],
                        "field": result.tag_path,
                    }
                )
    except Exception as exc:  # noqa: BLE001 - no non-Brouhaha fallback is allowed.
        warnings.append(
            {
                "type": "brouhaha_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": BROUHAHA_ACOUSTIC_TOOL["tool_name"],
            }
        )
        tags["basic_acoustic"]["snr_db"] = None
        tags["basic_acoustic"]["c50"] = None


def _run_dnsmos_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    dnsmos_config=None,
    dnsmos_client=None,
):
    try:
        results = DNSMOS_QUALITY_TOOL["run"](
            audio_path,
            context=tool_context,
            config=dnsmos_config,
            client=dnsmos_client,
        )
        for result in results:
            apply_result(tags, internal_results, result)
            if result.status != "estimated":
                warnings.append(
                    {
                        "type": "dnsmos_output_invalid",
                        "message": result.evidence.get("error", "invalid output"),
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "tool_name": DNSMOS_QUALITY_TOOL["tool_name"],
                        "field": result.tag_path,
                    }
                )
    except Exception as exc:  # noqa: BLE001 - no non-DNSMOS fallback is allowed.
        warnings.append(
            {
                "type": "dnsmos_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": DNSMOS_QUALITY_TOOL["tool_name"],
            }
        )
        _null_dnsmos_tags(tags)


def _run_firered_aed_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    firered_aed_config=None,
    firered_aed_client=None,
):
    duration_sec = tags["basic_acoustic"]["duration_sec"]
    if duration_sec is None or duration_sec <= 0:
        warnings.append(
            {
                "type": "invalid_duration_for_firered_aed",
                "message": "duration_sec must be positive before FireRed AED",
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": FIRERED_AED_TOOL["tool_name"],
            }
        )
        _null_firered_aed_tags(tags)
        return

    try:
        results = FIRERED_AED_TOOL["run"](
            audio_path,
            duration_sec=duration_sec,
            context=tool_context,
            config=firered_aed_config,
            client=firered_aed_client,
        )
        for result in results:
            apply_result(tags, internal_results, result)
    except Exception as exc:  # noqa: BLE001 - no non-FireRed fallback is allowed.
        warnings.append(
            {
                "type": "firered_aed_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": FIRERED_AED_TOOL["tool_name"],
            }
        )
        _null_firered_aed_tags(tags)


def _run_panns_background_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    panns_config=None,
    panns_client=None,
):
    try:
        result = PANNS_BACKGROUND_TOOL["run"](
            audio_path,
            context=tool_context,
            config=panns_config,
            client=panns_client,
        )
        apply_result(tags, internal_results, result)
    except Exception as exc:  # noqa: BLE001 - no non-PANNs fallback is allowed.
        warnings.append(
            {
                "type": "panns_background_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": PANNS_BACKGROUND_TOOL["tool_name"],
            }
        )
        _null_panns_background_tag(tags)


def _run_recrir_tools(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    recrir_config=None,
    recrir_client=None,
    artifact_dir=None,
    artifact_record_index=None,
):
    try:
        rir_result = RECRIR_RIR_TOOL["run"](
            audio_path,
            context=tool_context,
            config=recrir_config,
            client=recrir_client,
        )
        internal_results.append(rir_result.to_record())
        if rir_result.status != "estimated" or rir_result.value is None:
            warnings.append(
                {
                    "type": "recrir_output_invalid",
                    "message": rir_result.evidence.get("error", "invalid output"),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": RECRIR_RIR_TOOL["tool_name"],
                    "field": rir_result.tag_path,
                }
            )
            _null_rir_related_tags(tags)
            return
    except Exception as exc:  # noqa: BLE001 - no non-Rec-RIR fallback is allowed.
        warnings.append(
            {
                "type": "recrir_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": RECRIR_RIR_TOOL["tool_name"],
            }
        )
        _null_rir_related_tags(tags)
        return

    rir_payload = rir_result.value
    if artifact_dir is not None:
        try:
            artifact_path = write_rir_artifact(
                rir_payload,
                Path(artifact_dir) / "rir",
                _artifact_sample_key(sample_id, artifact_record_index),
            )
            internal_results[-1]["evidence"]["artifact_path"] = str(artifact_path)
            internal_results[-1]["evidence"]["artifact_format"] = "json.gz"
        except Exception as exc:  # noqa: BLE001 - artifact failure is internal.
            warnings.append(
                {
                    "type": "rir_artifact_write_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": RECRIR_RIR_TOOL["tool_name"],
                }
            )

    for tool, field in ((RT60_TOOL, "rt60"), (C50_TOOL, "c50")):
        try:
            result = tool["run"](
                rir_payload,
                context=tool_context,
            )
            apply_result(tags, internal_results, result)
        except Exception as exc:  # noqa: BLE001 - derived RIR tag failure is internal.
            warnings.append(
                {
                    "type": "rir_derived_tool_error",
                    "message": str(exc),
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": tool["tool_name"],
                    "field": tool["tag_path"],
                }
            )
            tags["sound_field_scene"][field] = None


def _null_rir_related_tags(tags):
    tags["sound_field_scene"]["rt60"] = None
    tags["sound_field_scene"]["c50"] = None


def _null_dnsmos_tags(tags):
    for field in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        tags["basic_acoustic"][field] = None


def _null_firered_aed_tags(tags):
    tags["sound_field_scene"]["audio_events"] = None
    tags["sound_field_scene"]["music"] = None


def _null_panns_background_tag(tags):
    tags["sound_field_scene"]["sound"] = None


def write_rir_artifact(rir_payload, artifact_dir, sample_key):
    # type: (Dict[str, Any], Union[str, Path], str) -> Path
    payload = validate_rir_payload(rir_payload)
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("%s.rir.json.gz" % _safe_artifact_stem(sample_key))
    with gzip.open(str(path), "wt", encoding="utf-8") as sink:
        json.dump(payload, sink, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return path


def _resolve_artifact_root(output_path, artifact_dir):
    # type: (Path, Optional[Union[str, Path]]) -> Path
    if artifact_dir is not None:
        return Path(artifact_dir)
    return output_path.parent / "artifacts"


def _artifact_sample_key(sample_id, record_index):
    # type: (str, Optional[int]) -> str
    if record_index is None:
        return sample_id
    return "%06d_%s" % (record_index, sample_id)


def _safe_artifact_stem(value):
    # type: (str) -> str
    raw = str(value)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not stem:
        stem = "sample"
    if len(stem) > 160:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        stem = "%s_%s" % (stem[:147].rstrip("._-"), digest)
    return stem


def compare_native_metadata_basic_acoustic_fields(sample, observed_basic_acoustic):
    # type: (Dict[str, Any], Dict[str, Any]) -> List[Dict[str, Any]]
    native_metadata = sample.get("native_metadata", {})
    warnings = []  # type: List[Dict[str, Any]]
    comparisons = [
        ("duration_sec", 1e-3),
        ("sample_rate_hz", 0),
        ("channels", 0),
        ("nonspeech_ratio", 1e-6),
        ("silence_segments", None),
        ("snr_db", 1e-6),
        ("c50", 1e-6),
        ("dnsmos_sig", 1e-6),
        ("dnsmos_bak", 1e-6),
        ("dnsmos_ovrl", 1e-6),
        ("dnsmos_p808", 1e-6),
    ]
    for field, tolerance in comparisons:
        native_value = native_metadata.get(field)
        observed_value = observed_basic_acoustic.get(field)
        if native_value is None or observed_value is None:
            continue
        if (
            tolerance is not None
            and isinstance(native_value, (int, float))
            and isinstance(observed_value, (int, float))
        ):
            mismatch = abs(float(native_value) - float(observed_value)) > tolerance
        else:
            mismatch = native_value != observed_value
        if mismatch:
            warnings.append(
                {
                    "type": "native_metadata_basic_acoustic_mismatch",
                    "field": field,
                    "native_metadata_value": native_value,
                    "observed_value": observed_value,
                    "message": "observed basic acoustic value differs from native metadata",
                }
            )
    return warnings


def compare_native_metadata_sound_field_scene_fields(sample, observed_sound_field):
    # type: (Dict[str, Any], Dict[str, Any]) -> List[Dict[str, Any]]
    native_metadata = sample.get("native_metadata", {})
    warnings = []  # type: List[Dict[str, Any]]
    comparisons = [
        ("far_field", None),
        ("rt60", 1e-6),
        ("c50", 1e-6),
        ("audio_events", None),
        ("music", None),
        ("sound", None),
    ]
    for field, tolerance in comparisons:
        native_value = native_metadata.get(field)
        observed_value = observed_sound_field.get(field)
        if native_value is None or observed_value is None:
            continue
        if (
            tolerance is not None
            and isinstance(native_value, (int, float))
            and isinstance(observed_value, (int, float))
        ):
            mismatch = abs(float(native_value) - float(observed_value)) > tolerance
        else:
            mismatch = native_value != observed_value
        if mismatch:
            warnings.append(
                {
                    "type": "native_metadata_sound_field_scene_mismatch",
                    "field": field,
                    "native_metadata_value": native_value,
                    "observed_value": observed_value,
                    "message": "observed sound-field value differs from native metadata",
                }
            )
    return warnings


def audit_basic_acoustic(basic_acoustic):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]
    duration_sec = basic_acoustic.get("duration_sec")
    sample_rate_hz = basic_acoustic.get("sample_rate_hz")
    channels = basic_acoustic.get("channels")
    nonspeech_ratio = basic_acoustic.get("nonspeech_ratio")
    silence_segments = basic_acoustic.get("silence_segments")
    snr_db = basic_acoustic.get("snr_db")
    c50 = basic_acoustic.get("c50")

    if duration_sec is not None and (
        isinstance(duration_sec, bool)
        or not isinstance(duration_sec, (int, float))
        or not _is_finite_number(duration_sec)
        or duration_sec < 0
    ):
        basic_acoustic["duration_sec"] = None
        warnings.append(
            {"type": "invalid_basic_acoustic_value", "field": "duration_sec"}
        )

    if sample_rate_hz is not None and (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        basic_acoustic["sample_rate_hz"] = None
        warnings.append(
            {"type": "invalid_basic_acoustic_value", "field": "sample_rate_hz"}
        )

    if channels is not None and (
        isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0
    ):
        basic_acoustic["channels"] = None
        warnings.append(
            {"type": "invalid_basic_acoustic_value", "field": "channels"}
        )

    if nonspeech_ratio is not None and (
        isinstance(nonspeech_ratio, bool)
        or not isinstance(nonspeech_ratio, (int, float))
        or not _is_finite_number(nonspeech_ratio)
        or nonspeech_ratio < 0
        or nonspeech_ratio > 1
    ):
        basic_acoustic["nonspeech_ratio"] = None
        basic_acoustic["silence_segments"] = None
        warnings.append(
            {"type": "invalid_basic_acoustic_value", "field": "nonspeech_ratio"}
        )

    if silence_segments is not None:
        if not _is_valid_silence_segments(
            silence_segments,
            basic_acoustic.get("duration_sec"),
        ):
            basic_acoustic["nonspeech_ratio"] = None
            basic_acoustic["silence_segments"] = None
            warnings.append(
                {
                    "type": "invalid_basic_acoustic_value",
                    "field": "silence_segments",
                }
            )

    if snr_db is not None and (
        isinstance(snr_db, bool)
        or not isinstance(snr_db, (int, float))
        or not _is_finite_number(snr_db)
    ):
        basic_acoustic["snr_db"] = None
        warnings.append({"type": "invalid_basic_acoustic_value", "field": "snr_db"})

    if c50 is not None and (
        isinstance(c50, bool)
        or not isinstance(c50, (int, float))
        or not _is_finite_number(c50)
    ):
        basic_acoustic["c50"] = None
        warnings.append({"type": "invalid_basic_acoustic_value", "field": "c50"})

    for field in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        value = basic_acoustic.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite_number(value)
            or value < 1.0
            or value > 5.0
        ):
            basic_acoustic[field] = None
            warnings.append(
                {"type": "invalid_basic_acoustic_value", "field": field}
            )

    return warnings


def audit_sound_field_scene(sound_field_scene):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    for field in ("far_field", "music"):
        value = sound_field_scene.get(field)
        if value is not None and not isinstance(value, bool):
            sound_field_scene[field] = None
            warnings.append(
                {"type": "invalid_sound_field_scene_value", "field": field}
            )

    audio_events = sound_field_scene.get("audio_events")
    if audio_events is not None and not _is_valid_label_list(
        audio_events,
        allowed=EVENT_NAMES,
        require_allowed_order=True,
    ):
        sound_field_scene["audio_events"] = None
        warnings.append(
            {"type": "invalid_sound_field_scene_value", "field": "audio_events"}
        )

    sound = sound_field_scene.get("sound")
    if sound is not None and not _is_valid_label_list(
        sound,
        max_items=TOP_EVENTS_LIMIT,
    ):
        sound_field_scene["sound"] = None
        warnings.append(
            {"type": "invalid_sound_field_scene_value", "field": "sound"}
        )

    music = sound_field_scene.get("music")
    audio_events = sound_field_scene.get("audio_events")
    if (
        music is not None
        and audio_events is not None
        and music != ("music" in audio_events)
    ):
        sound_field_scene["audio_events"] = None
        sound_field_scene["music"] = None
        warnings.append(
            {
                "type": "inconsistent_sound_field_scene_values",
                "fields": ["audio_events", "music"],
            }
        )

    rt60 = sound_field_scene.get("rt60")
    if rt60 is not None and (
        isinstance(rt60, bool)
        or not isinstance(rt60, (int, float))
        or not _is_finite_number(rt60)
        or rt60 < 0
    ):
        sound_field_scene["rt60"] = None
        warnings.append({"type": "invalid_sound_field_scene_value", "field": "rt60"})

    c50 = sound_field_scene.get("c50")
    if c50 is not None and (
        isinstance(c50, bool)
        or not isinstance(c50, (int, float))
        or not _is_finite_number(c50)
    ):
        sound_field_scene["c50"] = None
        warnings.append({"type": "invalid_sound_field_scene_value", "field": "c50"})

    return warnings


def _is_valid_label_list(
    value,
    allowed=None,
    max_items=None,
    require_allowed_order=False,
):
    if not isinstance(value, list):
        return False
    if max_items is not None and len(value) > max_items:
        return False
    if any(
        not isinstance(label, str) or not label or label != label.strip()
        for label in value
    ):
        return False
    if len(set(value)) != len(value):
        return False
    if allowed is not None:
        allowed = tuple(allowed)
        if any(label not in allowed for label in value):
            return False
        if require_allowed_order:
            return value == [label for label in allowed if label in value]
    return True


def audit_speaker(speaker):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    for field in SPEAKER_FIELDS:
        value = speaker.get(field)
        if value is not None and not isinstance(value, bool):
            speaker[field] = None
            warnings.append({"type": "invalid_speaker_value", "field": field})

    return warnings


def audit_language_content(language_content):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    topic = language_content.get("topic")
    if topic is not None and not isinstance(topic, str):
        language_content["topic"] = None
        warnings.append({"type": "invalid_language_content_value", "field": "topic"})

    language = language_content.get("language")
    if language is not None and (not isinstance(language, str) or not language):
        language_content["language"] = None
        warnings.append(
            {"type": "invalid_language_content_value", "field": "language"}
        )

    if not _is_non_negative_int(language_content.get("word_count")):
        if language_content.get("word_count") is not None:
            language_content["word_count"] = None
            warnings.append(
                {"type": "invalid_language_content_value", "field": "word_count"}
            )

    punctuation = language_content.get("punctuation")
    if punctuation is not None and not _is_valid_punctuation_value(punctuation):
        language_content["punctuation"] = None
        warnings.append(
            {"type": "invalid_language_content_value", "field": "punctuation"}
        )

    repetition = language_content.get("repetition")
    if repetition is not None and not _is_valid_repetition_value(repetition):
        language_content["repetition"] = None
        warnings.append(
            {"type": "invalid_language_content_value", "field": "repetition"}
        )

    filler = language_content.get("filler")
    if filler is not None and not _is_non_negative_int(filler):
        language_content["filler"] = None
        warnings.append({"type": "invalid_language_content_value", "field": "filler"})

    return warnings


def _is_non_negative_int(value):
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_valid_punctuation_value(value):
    if not isinstance(value, dict):
        return False
    if set(value.keys()) != set(["punctuation_count", "has_terminal_punctuation"]):
        return False
    return _is_non_negative_int(value["punctuation_count"]) and isinstance(
        value["has_terminal_punctuation"],
        bool,
    )


def _is_valid_repetition_value(value):
    if not isinstance(value, dict):
        return False
    if set(value.keys()) != set(["has_repetition", "repetition_count"]):
        return False
    return isinstance(value["has_repetition"], bool) and _is_non_negative_int(
        value["repetition_count"]
    )


def _is_valid_silence_segments(segments, duration_sec):
    if (
        duration_sec is None
        or duration_sec <= 0
        or not _is_finite_number(duration_sec)
        or not isinstance(segments, list)
    ):
        return False

    previous_end = None
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        if set(segment.keys()) != set(["start_sec", "end_sec"]):
            return False
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        if isinstance(start_sec, bool) or isinstance(end_sec, bool):
            return False
        if not isinstance(start_sec, (int, float)) or not isinstance(
            end_sec, (int, float)
        ):
            return False
        if not _is_finite_number(start_sec) or not _is_finite_number(end_sec):
            return False
        if start_sec < 0 or start_sec >= end_sec or end_sec > duration_sec:
            return False
        if previous_end is not None and start_sec < previous_end:
            return False
        previous_end = end_sec
    return True


def _is_finite_number(value):
    return value == value and value not in (float("inf"), float("-inf"))


def build_arg_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="phase1_asr_samples/manifest.jsonl",
        help="Input JSONL manifest using the closed raw-only schema.",
    )
    parser.add_argument(
        "--output",
        default="phase1_asr_samples/outputs/sample_tags.jsonl",
        help="Output tags-only JSONL path for sample tags.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help=(
            "Only process the given sample_id. Can be passed multiple times. "
            "Without --input-tags, the output contains only selected samples; "
            "with --input-tags, non-selected rows are preserved."
        ),
    )
    parser.add_argument(
        "--input-tags",
        default=None,
        help=(
            "Existing tags-only JSONL used as a base for supplement mode. "
            "Rows must align with the input manifest."
        ),
    )
    parser.add_argument(
        "--only-tags",
        default=None,
        help=(
            "Comma-separated tag paths, groups, or stage names to run, for "
            "example speaker,language_content.topic,basic_acoustic.nonspeech_ratio."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="With --input-tags, only run selected tag paths that are currently null.",
    )
    parser.add_argument(
        "--firered-vad-use-gpu",
        action="store_true",
        help="Use GPU in FireRedVAD config. Defaults to CPU.",
    )
    parser.add_argument(
        "--firered-aed-use-gpu",
        action="store_true",
        help="Use GPU in FireRed AED config. Defaults to CPU.",
    )
    parser.add_argument(
        "--panns-use-gpu",
        action="store_true",
        help="Use GPU for PANNs background-sound inference. Defaults to CPU.",
    )
    parser.add_argument(
        "--panns-threshold",
        type=float,
        default=0.30,
        help="PANNs background-sound probability threshold. Defaults to 0.30.",
    )
    parser.add_argument(
        "--brouhaha-use-gpu",
        action="store_true",
        help="Use GPU in Brouhaha config. Defaults to CPU when supported.",
    )
    parser.add_argument(
        "--recrir-use-gpu",
        action="store_true",
        help="Use GPU in Rec-RIR config. Defaults to CPU when supported.",
    )
    parser.add_argument(
        "--firered-vad-python",
        default=None,
        help="Python executable for FireRed VAD subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--firered-aed-python",
        default=None,
        help="Python executable for FireRed AED subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--panns-python",
        default=None,
        help="Python executable for PANNs subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--brouhaha-python",
        default=None,
        help="Python executable for Brouhaha subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--recrir-python",
        default=None,
        help="Python executable for Rec-RIR subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--dnsmos-python",
        default=None,
        help="Python executable for DNSMOS subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--dnsmos-personalized",
        action="store_true",
        help="Use the personalized DNSMOS primary model. Defaults to regular DNSMOS.",
    )
    parser.add_argument(
        "--moss-diarize-enable",
        action="store_true",
        help="Enable MOSS-Transcribe-Diarize speaker diarization.",
    )
    parser.add_argument(
        "--moss-diarize-python",
        default=None,
        help=(
            "Python executable for local MOSS-Transcribe-Diarize subprocess. "
            "Defaults to local_config.py."
        ),
    )
    parser.add_argument(
        "--moss-diarize-endpoint",
        default=None,
        help=(
            "Legacy OpenAI-compatible MOSS /v1/audio/transcriptions endpoint. "
            "Ignored when a MOSS subprocess Python is configured."
        ),
    )
    parser.add_argument(
        "--moss-diarize-model",
        default=None,
        help="MOSS diarize model name. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--moss-diarize-timeout-sec",
        type=int,
        default=None,
        help="MOSS diarize legacy HTTP timeout in seconds. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--moss-diarize-max-new-tokens",
        type=int,
        default=None,
        help="MOSS diarize max_new_tokens. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--moss-diarize-api-key",
        default=None,
        help="Optional bearer token for the legacy MOSS endpoint.",
    )
    parser.add_argument(
        "--moss-diarize-device",
        default=None,
        help="Local MOSS device, for example auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--moss-diarize-torch-dtype",
        default=None,
        help="Local MOSS torch dtype, for example auto, bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--moss-diarize-prompt",
        default=None,
        help="Optional local MOSS transcription prompt or hotword hint.",
    )
    parser.add_argument(
        "--speaker-channel-activity-disable",
        action="store_true",
        help="Disable multi-channel WAV channel-activity speaker route.",
    )
    parser.add_argument(
        "--speaker-prefer-moss",
        action="store_true",
        help=(
            "Skip per-channel MOSS purity QA and use merged-headset MOSS directly. "
            "Kept for compatibility."
        ),
    )
    parser.add_argument(
        "--speaker-force-channel-activity",
        "--speaker-single-speaker-per-channel",
        "--speaker-prefer-channel-activity",
        dest="speaker_force_channel_activity",
        action="store_true",
        help=(
            "Assert that dataset documentation guarantees one speaker per channel "
            "and run per-channel energy VAD without MOSS purity QA."
        ),
    )
    parser.add_argument(
        "--topic-enable",
        action="store_true",
        help="Enable OpenAI Responses topic classification for language_content.topic.",
    )
    parser.add_argument(
        "--topic-provider",
        default=None,
        help="Topic provider name. Currently supports openai_responses.",
    )
    parser.add_argument(
        "--topic-model",
        default=None,
        help="OpenAI Responses model for topic classification.",
    )
    parser.add_argument(
        "--topic-base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to local_config.py or OpenAI.",
    )
    parser.add_argument(
        "--topic-api-key",
        default=None,
        help="OpenAI Responses API key. Prefer OPENAI_API_KEY or api.txt.",
    )
    parser.add_argument(
        "--topic-api-key-path",
        default=None,
        help="Path to a file containing the OpenAI Responses API key.",
    )
    parser.add_argument(
        "--topic-model-provider",
        default=None,
        help="Optional model provider section name in ~/.codex/config.toml.",
    )
    parser.add_argument(
        "--topic-codex-config-path",
        default=None,
        help="Optional Codex TOML config path for OpenAI-compatible provider settings.",
    )
    parser.add_argument(
        "--topic-timeout-sec",
        type=int,
        default=None,
        help="OpenAI Responses timeout in seconds.",
    )
    parser.add_argument(
        "--topic-temperature",
        type=float,
        default=None,
        help="Optional temperature for OpenAI Responses topic classification.",
    )
    parser.add_argument(
        "--topic-use-json-schema",
        action="store_true",
        default=None,
        help="Request Responses JSON schema output instead of json_object.",
    )
    parser.add_argument(
        "--topic-cache-disable",
        action="store_true",
        help="Disable JSONL cache for topic responses.",
    )
    parser.add_argument(
        "--topic-cache-path",
        default=None,
        help="JSONL cache path for topic responses.",
    )
    parser.add_argument(
        "--topic-fallback",
        choices=["null", "heuristic", "error"],
        default=None,
        help="Topic behavior when the Responses call fails. Defaults to null.",
    )
    parser.add_argument(
        "--topic-short-guard-disable",
        action="store_true",
        help="Disable deterministic short-utterance topic guard.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Directory for non-public artifacts such as Rec-RIR waveforms. "
            "Defaults to OUTPUT_PARENT/artifacts."
        ),
    )
    return parser


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    args = build_arg_parser().parse_args(argv)
    firered_vad_config = FireRedVadConfig(
        use_gpu=args.firered_vad_use_gpu,
        subprocess_python=args.firered_vad_python,
    )
    firered_aed_config = FireRedAedConfig(
        use_gpu=args.firered_aed_use_gpu,
        subprocess_python=args.firered_aed_python,
    )
    panns_config = PannsBackgroundConfig(
        use_gpu=args.panns_use_gpu,
        threshold=args.panns_threshold,
        subprocess_python=args.panns_python,
    )
    brouhaha_config = BrouhahaConfig(
        use_gpu=args.brouhaha_use_gpu,
        subprocess_python=args.brouhaha_python,
    )
    recrir_config = RecRirConfig(
        use_gpu=args.recrir_use_gpu,
        subprocess_python=args.recrir_python,
    )
    dnsmos_config = DnsmosConfig(
        personalized=args.dnsmos_personalized,
        subprocess_python=args.dnsmos_python,
    )
    speaker_config = default_speaker_layer_config(
        enable_moss=args.moss_diarize_enable,
        moss_endpoint=args.moss_diarize_endpoint,
        moss_model=args.moss_diarize_model,
        moss_timeout_sec=args.moss_diarize_timeout_sec,
        moss_max_new_tokens=args.moss_diarize_max_new_tokens,
        moss_api_key=args.moss_diarize_api_key,
        moss_python=args.moss_diarize_python,
        moss_device=args.moss_diarize_device,
        moss_torch_dtype=args.moss_diarize_torch_dtype,
        moss_prompt=args.moss_diarize_prompt,
    )
    speaker_config.enable_channel_activity = not args.speaker_channel_activity_disable
    if args.speaker_prefer_moss:
        speaker_config.run_moss_for_channel_qa = False
    if args.speaker_force_channel_activity:
        speaker_config.force_channel_activity = True
        speaker_config.prefer_channel_activity = True
    topic_config = TopicConfig(
        enabled=True if args.topic_enable else None,
        provider=args.topic_provider,
        model=args.topic_model,
        base_url=args.topic_base_url,
        api_key=args.topic_api_key,
        api_key_path=args.topic_api_key_path,
        model_provider=args.topic_model_provider,
        codex_config_path=args.topic_codex_config_path,
        timeout_sec=args.topic_timeout_sec,
        temperature=args.topic_temperature,
        use_json_schema=args.topic_use_json_schema,
        cache_enabled=False if args.topic_cache_disable else None,
        cache_path=args.topic_cache_path,
        fallback=args.topic_fallback,
        short_guard_enabled=False if args.topic_short_guard_disable else None,
    )
    summary = run_manifest(
        args.manifest,
        args.output,
        firered_vad_config=firered_vad_config,
        brouhaha_config=brouhaha_config,
        recrir_config=recrir_config,
        speaker_config=speaker_config,
        artifact_dir=args.artifact_dir,
        dnsmos_config=dnsmos_config,
        firered_aed_config=firered_aed_config,
        panns_config=panns_config,
        topic_config=topic_config,
        sample_ids=args.sample_id,
        existing_tags_path=args.input_tags,
        selected_tag_paths=[args.only_tags] if args.only_tags else None,
        missing_only=args.missing_only,
    )
    public_summary = {
        "output_path": summary["output_path"],
        "sample_count": summary["sample_count"],
        "processed_sample_count": summary["processed_sample_count"],
    }
    print(json.dumps(public_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
