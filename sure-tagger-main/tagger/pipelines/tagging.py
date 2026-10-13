"""Sample-level ASR dataset tagging pipeline.

Input records must match the closed raw-only schema in development.md. Public
output is tags-only and contains only tag values.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from tagger.input_schema import InputSchemaError, validate_input_record
from tagger.local_config import (
    NEMOTRON_MODEL_DIR,
    NEMOTRON_PYTHON,
    NEMOTRON_TIMEOUT_SEC,
    FIRERED_ASR_MODEL_DIR,
    FIRERED_ASR_PYTHON,
    FIRERED_ASR_SOURCE_DIR,
    FIRERED_ASR_TIMEOUT_SEC,
    QWEN3_ASR_MODEL_DIR,
    QWEN3_FORCED_ALIGNER_MODEL_DIR,
    QWEN3_ASR_PYTHON,
    QWEN3_ASR_TIMEOUT_SEC,
)
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
from tagger.tools.sound_field_scene.registry import (
    DASS_NOISE_TYPE_TOOL,
    FIRERED_AED_TOOL,
)
from tagger.tools.sound_field_scene.dass_categories import (
    NOISE_COMPOSITION_AUDIT_MAX_ITEMS,
    PUBLIC_COMPOSITION_CATEGORIES,
    classify_dass_label,
)
from tagger.tools.sound_field_scene.dass_noise_type_detector import (
    DassNoiseTypeConfig,
)
from tagger.tools.sound_field_scene.firered_aed_detector import (
    EVENT_NAMES,
    FireRedAedConfig,
)
from tagger.tools.room_acoustic.rir_estimator import (
    RecRirConfig,
    validate_rir_payload,
)
from tagger.pipelines.speaker_evidence import (
    SpeakerEvidenceConfig,
    default_speaker_evidence_config,
    run_record as run_speaker_v2_record,
)
from tagger.tools.speaker_v2.profiles import (
    available_profiles as available_speaker_profiles,
)
from tagger.tools.speaker_v2.nemotron3_diarization import Nemotron3DiarizationConfig
from tagger.tools.speaker_v2.firered_asr import FireRedAsrConfig
from tagger.tools.speaker_v2.qwen3_asr import Qwen3AsrConfig
from tagger.tools.language_content.registry import (
    DETERMINISTIC_LANGUAGE_CONTENT_TOOL,
    FIRERED_LID_LANGUAGE_CONTENT_TOOL,
)
from tagger.tools.language_content.firered_lid_detector import FireRedLidConfig
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
    run as run_speaker_segment_lid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_config_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


BASIC_ACOUSTIC_FIELDS = {
    "duration_sec": None,
    "sample_rate_hz": None,
    "channels": None,
    "nonspeech_ratio": None,
    "silence_segments": None,
}

AUDIO_QUALITY_FIELDS = {
    "snr_db": None,
    "dnsmos_sig": None,
    "dnsmos_bak": None,
    "dnsmos_ovrl": None,
    "dnsmos_p808": None,
}

ROOM_ACOUSTIC_FIELDS = {
    "far_field": None,
    "rt60_sec": None,
    "c50_db": None,
}

SOUND_FIELD_SCENE_FIELDS = {
    "speech_music_events": None,
    "music_present": None,
    "external_noise_type": None,
    "noise_composition": None,
}

SPEAKER_FIELDS = {
    "speaker_count": None,
    "speaker_present": None,
    "asr_transcript": None,
    "multi_speaker": None,
    "speaker_change_count": None,
    "speaker_change": None,
    "overlap_ratio": None,
    "speaker_overlap": None,
    "profiles": None,
}

LANGUAGE_CONTENT_FIELDS = {
    "language": None,
    "word_count": None,
    "punctuation": None,
    "repetition": None,
    "filler": None,
}

DETERMINISTIC_DERIVED_FIELDS = {
    "speaker_change_rate_per_min": None,
    "filler_ratio": None,
}

STAGE_LANGUAGE_DETERMINISTIC = "language_deterministic"
STAGE_DETERMINISTIC_DERIVED = "deterministic_derived"
STAGE_AUDIO_PROBE = "audio_probe"
STAGE_SILENCE = "silence"
STAGE_SPEAKER = "speaker"
STAGE_BROUHAHA = "brouhaha"
STAGE_DNSMOS = "dnsmos"
STAGE_FIRERED_AED = "firered_aed"
STAGE_DASS = "dass"
STAGE_RECRIR = "recrir"
STAGE_FIRERED_LID = "firered_lid"

# Deprecated on 2026-08-25: the PANNs stage and its public
# ``sound_field_scene.sound`` field were removed — noise_composition
# supersedes them. The panns_background_detector module stays importable for
# future cross-validation evidence work, but nothing runs it and its output
# must not enter public tags.
FULL_STAGES = [
    STAGE_LANGUAGE_DETERMINISTIC,
    STAGE_AUDIO_PROBE,
    STAGE_SILENCE,
    STAGE_SPEAKER,
    STAGE_BROUHAHA,
    STAGE_DNSMOS,
    STAGE_FIRERED_AED,
    STAGE_DASS,
    STAGE_RECRIR,
    STAGE_FIRERED_LID,
    STAGE_DETERMINISTIC_DERIVED,
]

AUDIO_STAGES = set(
    [
        STAGE_AUDIO_PROBE,
        STAGE_SILENCE,
        STAGE_SPEAKER,
        STAGE_BROUHAHA,
        STAGE_DNSMOS,
        STAGE_FIRERED_AED,
        STAGE_DASS,
        STAGE_RECRIR,
        STAGE_FIRERED_LID,
    ]
)

STAGE_TAG_PATHS = {
    STAGE_LANGUAGE_DETERMINISTIC: [
        "language_content.word_count",
        "language_content.punctuation",
        "language_content.punctuation.punctuation_count",
        "language_content.punctuation.has_terminal_punctuation",
        "language_content.repetition",
        "language_content.repetition.has_repetition",
        "language_content.repetition.repetition_count",
        "language_content.filler",
    ],
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
        "speaker.speaker_count",
        "speaker.speaker_present",
        "speaker.asr_transcript",
        "speaker.multi_speaker",
        "speaker.speaker_change_count",
        "speaker.speaker_change",
        "speaker.overlap_ratio",
        "speaker.speaker_overlap",
        "speaker.profiles",
    ],
    STAGE_BROUHAHA: ["audio_quality.snr_db"],
    STAGE_DNSMOS: [
        "audio_quality.dnsmos_sig",
        "audio_quality.dnsmos_bak",
        "audio_quality.dnsmos_ovrl",
        "audio_quality.dnsmos_p808",
    ],
    STAGE_FIRERED_AED: [
        "sound_field_scene.speech_music_events",
        "sound_field_scene.music_present",
    ],
    STAGE_DASS: [
        "sound_field_scene.external_noise_type",
        "sound_field_scene.noise_composition",
    ],
    STAGE_RECRIR: ["room_acoustic.rt60_sec", "room_acoustic.c50_db"],
    STAGE_FIRERED_LID: ["language_content.language"],
    STAGE_DETERMINISTIC_DERIVED: [
        "deterministic_derived.speaker_change_rate_per_min",
        "deterministic_derived.filler_ratio",
    ],
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
    artifact_dir=None,
    dnsmos_config=None,
    firered_aed_config=None,
    dass_config=None,
    firered_lid_config=None,
    sample_ids=None,
    existing_tags_path=None,
    selected_tag_paths=None,
    missing_only=False,
    speaker_segment_lid_config=None,
    workers=1,
):
    # type: (Union[str, Path], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerEvidenceConfig], Optional[Union[str, Path]], Optional[DnsmosConfig], Optional[FireRedAedConfig], Optional[DassNoiseTypeConfig], Optional[FireRedLidConfig], Optional[List[str]], Optional[Union[str, Path]], Optional[List[str]], bool, Optional[SpeakerSegmentLidConfig], int) -> Dict[str, Any]
    manifest = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = _resolve_artifact_root(output, artifact_dir)
    selected_sample_ids = set(str(item) for item in (sample_ids or []))
    existing_tags = (
        _load_existing_tags(existing_tags_path) if existing_tags_path is not None else None
    )
    requested_tag_paths = _normalize_selected_tag_paths(selected_tag_paths)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    count = 0
    processed_count = 0
    internal_warning_count = 0
    tool_context = {}  # type: Dict[str, Any]

    def run_task(task):
        row_index, record, base_tags, row_tag_paths = task
        try:
            return _tag_record_internal(
                record,
                manifest_dir=manifest.parent,
                firered_vad_config=firered_vad_config,
                brouhaha_config=brouhaha_config,
                recrir_config=recrir_config,
                speaker_config=speaker_config,
                artifact_dir=artifact_root,
                artifact_record_index=row_index,
                tool_context=tool_context,
                dnsmos_config=dnsmos_config,
                firered_aed_config=firered_aed_config,
                dass_config=dass_config,
                firered_lid_config=firered_lid_config,
                initial_tags=base_tags,
                selected_tag_paths=row_tag_paths,
                speaker_segment_lid_config=speaker_segment_lid_config,
            )
        except InputSchemaError as exc:
            raise InputSchemaError("line %s: %s" % (row_index, exc))

    try:
        rows = []
        tasks = []
        with manifest.open("r", encoding="utf-8") as source:
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
                        rows.append((False, base_tags))
                    continue
                row_tag_paths = requested_tag_paths
                if base_tags is not None and row_tag_paths is None:
                    row_tag_paths = _missing_tag_paths(base_tags)
                if base_tags is not None and missing_only and row_tag_paths is not None:
                    row_tag_paths = _filter_missing_tag_paths(base_tags, row_tag_paths)
                task_index = len(tasks)
                tasks.append((row_index, record, base_tags, row_tag_paths))
                rows.append((True, task_index))

        if workers == 1:
            results = [run_task(task) for task in tasks]
        else:
            # Every model adapter still has one shared subprocess worker. Sample
            # threads overlap different model stages without duplicating model
            # weights on the GPU.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(run_task, tasks))

        with output.open("w", encoding="utf-8") as sink:
            for is_processed, value in rows:
                internal = results[value] if is_processed else None
                tags = internal["tags"] if internal is not None else value
                if internal is not None:
                    internal_warning_count += len(internal["warnings"])
                    processed_count += 1
                sink.write(
                    json.dumps(tags, ensure_ascii=False, sort_keys=True) + "\n"
                )
                count += 1
    finally:
        close_subprocess_workers(tool_context)

    return {
        "manifest_path": str(manifest),
        "output_path": str(output),
        "artifact_dir": str(artifact_root),
        "sample_count": count,
        "processed_sample_count": processed_count,
        "internal_warning_count": internal_warning_count,
        "workers": workers,
    }


def tag_record(
    record,
    manifest_dir,
    firered_vad_config=None,
    brouhaha_config=None,
    recrir_config=None,
    speaker_config=None,
    recrir_client=None,
    artifact_dir=None,
    dnsmos_config=None,
    dnsmos_client=None,
    firered_aed_config=None,
    firered_aed_client=None,
    dass_config=None,
    dass_client=None,
    firered_lid_config=None,
    initial_tags=None,
    selected_tag_paths=None,
    speaker_segment_lid_config=None,
):
    # type: (Dict[str, Any], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerEvidenceConfig], Any, Optional[Union[str, Path]], Optional[DnsmosConfig], Any, Optional[FireRedAedConfig], Any, Optional[DassNoiseTypeConfig], Any, Optional[FireRedLidConfig], Optional[Dict[str, Any]], Optional[List[str]], Optional[SpeakerSegmentLidConfig]) -> Dict[str, Any]
    return _tag_record_internal(
        record,
        manifest_dir,
        firered_vad_config=firered_vad_config,
        brouhaha_config=brouhaha_config,
        recrir_config=recrir_config,
        speaker_config=speaker_config,
        recrir_client=recrir_client,
        artifact_dir=artifact_dir,
        dnsmos_config=dnsmos_config,
        dnsmos_client=dnsmos_client,
        firered_aed_config=firered_aed_config,
        firered_aed_client=firered_aed_client,
        dass_config=dass_config,
        dass_client=dass_client,
        firered_lid_config=firered_lid_config,
        initial_tags=initial_tags,
        selected_tag_paths=selected_tag_paths,
        speaker_segment_lid_config=speaker_segment_lid_config,
    )["tags"]


def _tag_record_internal(
    record,
    manifest_dir,
    firered_vad_config=None,
    brouhaha_config=None,
    recrir_config=None,
    speaker_config=None,
    recrir_client=None,
    artifact_dir=None,
    artifact_record_index=None,
    tool_context=None,
    dnsmos_config=None,
    dnsmos_client=None,
    firered_aed_config=None,
    firered_aed_client=None,
    dass_config=None,
    dass_client=None,
    firered_lid_config=None,
    initial_tags=None,
    selected_tag_paths=None,
    speaker_segment_lid_config=None,
):
    # type: (Dict[str, Any], Union[str, Path], Optional[FireRedVadConfig], Optional[BrouhahaConfig], Optional[RecRirConfig], Optional[SpeakerEvidenceConfig], Any, Optional[Union[str, Path]], Optional[int], Optional[Dict[str, Any]], Optional[DnsmosConfig], Any, Optional[FireRedAedConfig], Any, Optional[DassNoiseTypeConfig], Any, Optional[FireRedLidConfig], Optional[Dict[str, Any]], Optional[List[str]], Optional[SpeakerSegmentLidConfig]) -> Dict[str, Any]
    validate_input_record(record)
    sample = record["sample"]
    sample_id = sample["sample_id"]
    audio_path = resolve_audio_path(sample, manifest_dir)

    tags = _merge_tags(initial_tags)
    internal_results = []  # type: List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]
    stages = _stages_for_tag_paths(selected_tag_paths, tags)
    speaker_stage_requested = STAGE_SPEAKER in stages
    segment_lid_config = speaker_segment_lid_config or SpeakerSegmentLidConfig()
    has_input_transcript = _has_transcript(sample)
    speaker_asr_required = (
        not has_input_transcript
        and STAGE_LANGUAGE_DETERMINISTIC in stages
    )
    speaker_lid_required = (
        segment_lid_config.enabled
        and segment_lid_config.language_only_start_speaker_stage
        and not has_input_transcript
        and STAGE_FIRERED_LID in stages
        and STAGE_SPEAKER not in stages
    )
    if speaker_asr_required or speaker_lid_required:
        stages.add(STAGE_SPEAKER)
        _add_dependency_stages(stages, tags)

    deferred_firered_lid = (
        STAGE_SPEAKER in stages
        and STAGE_FIRERED_LID in stages
    )
    if deferred_firered_lid:
        stages.discard(STAGE_FIRERED_LID)

    if not speaker_asr_required:
        stages = _apply_no_transcript_speech_stage_guard(sample, tags, stages, warnings)

        if STAGE_LANGUAGE_DETERMINISTIC in stages:
            _run_language_content_tools(
                record,
                tags,
                internal_results,
                warnings,
                sample_id,
                stages=stages,
            )

    needs_audio = bool(stages & AUDIO_STAGES)
    speaker_output = _empty_speaker_output()
    speaker_asr_transcript = ""
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
            speaker_tags_before_internal_stage = dict(tags["speaker"])
            speaker_output = _run_speaker_tools(
                audio_path,
                record,
                manifest_dir,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                speaker_config=speaker_config,
                speaker_segment_lid_config=segment_lid_config,
                artifact_dir=artifact_dir,
                artifact_record_index=artifact_record_index,
            )
            speaker_asr_transcript = speaker_output["transcript"]
            if (
                speaker_lid_required
                and not speaker_asr_required
                and not speaker_stage_requested
            ):
                tags["speaker"].update(speaker_tags_before_internal_stage)
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
        if STAGE_DASS in stages:
            _run_dass_noise_type_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                dass_config=dass_config,
                dass_client=dass_client,
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
        if STAGE_FIRERED_LID in stages:
            _run_firered_lid_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                firered_lid_config=firered_lid_config,
            )

    if deferred_firered_lid:
        _run_speaker_aware_lid_tool(
            audio_path,
            speaker_output,
            tool_context,
            tags,
            internal_results,
            warnings,
            sample_id,
            firered_lid_config=firered_lid_config,
            segment_lid_config=segment_lid_config,
            artifact_dir=artifact_dir,
            artifact_record_index=artifact_record_index,
        )
    if speaker_asr_required:
        stages = _apply_no_transcript_speech_stage_guard(
            sample,
            tags,
            stages,
            warnings,
            allow_language_from_speaker_asr=bool(speaker_asr_transcript),
        )
        if (
            speaker_asr_transcript
            and (
                STAGE_LANGUAGE_DETERMINISTIC in stages
                or deferred_firered_lid
            )
        ):
            _run_language_content_tools(
                record,
                tags,
                internal_results,
                warnings,
                sample_id,
                stages=stages,
                transcript_override=speaker_asr_transcript,
            )

    warnings.extend(
        compare_native_metadata_basic_acoustic_fields(sample, tags["basic_acoustic"])
    )
    warnings.extend(
        compare_native_metadata_audio_quality_fields(sample, tags["audio_quality"])
    )
    warnings.extend(
        compare_native_metadata_room_acoustic_fields(sample, tags["room_acoustic"])
    )
    warnings.extend(
        compare_native_metadata_sound_field_scene_fields(
            sample,
            tags["sound_field_scene"],
        )
    )
    warnings.extend(audit_basic_acoustic(tags["basic_acoustic"]))
    warnings.extend(audit_audio_quality(tags["audio_quality"]))
    warnings.extend(audit_room_acoustic(tags["room_acoustic"]))
    warnings.extend(audit_sound_field_scene(tags["sound_field_scene"]))
    warnings.extend(audit_speaker(tags["speaker"]))
    warnings.extend(audit_language_content(tags["language_content"]))
    _derive_deterministic_tags(tags)
    warnings.extend(audit_deterministic_derived(tags["deterministic_derived"]))

    return {
        "tags": tags,
        "internal_results": internal_results,
        "warnings": warnings,
    }


def empty_tags():
    # type: () -> Dict[str, Any]
    return {
        "basic_acoustic": dict(BASIC_ACOUSTIC_FIELDS),
        "audio_quality": dict(AUDIO_QUALITY_FIELDS),
        "room_acoustic": dict(ROOM_ACOUSTIC_FIELDS),
        "sound_field_scene": dict(SOUND_FIELD_SCENE_FIELDS),
        "speaker": dict(SPEAKER_FIELDS),
        "language_content": dict(LANGUAGE_CONTENT_FIELDS),
        "deterministic_derived": dict(DETERMINISTIC_DERIVED_FIELDS),
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


def _apply_no_transcript_speech_stage_guard(
    sample,
    tags,
    stages,
    warnings,
    allow_language_from_speaker_asr=False,
):
    # type: (Dict[str, Any], Dict[str, Any], set, List[Dict[str, Any]]) -> set
    if _has_transcript(sample):
        return stages
    skipped = (
        set()
        if allow_language_from_speaker_asr
        else stages & set([STAGE_LANGUAGE_DETERMINISTIC])
    )
    if not skipped:
        return stages
    stages = set(stages) - skipped
    for field in ("word_count", "punctuation", "repetition", "filler"):
        tags["language_content"][field] = None
    message = (
        "sample.text.transcript is empty; language-content text stages use the "
        "speaker-v2 ASR transcript"
        if allow_language_from_speaker_asr
        else "sample.text.transcript is empty; skipping language-content stages "
        "while continuing audio and non-language stages"
    )
    warnings.append(
        {
            "type": "speech_dependent_stages_skipped_no_transcript",
            "message": message,
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
        if value in (
            "basic_acoustic",
            "audio_quality",
            "room_acoustic",
            "sound_field_scene",
            "speaker",
            "language_content",
            "deterministic_derived",
        ):
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
    if STAGE_DETERMINISTIC_DERIVED in stages:
        # Derived values are computed after their source tags are finalized.
        # Only request the source stages whose values are not already present.
        if (
            _tag_value(tags, "speaker.speaker_change_count") is None
            or _tag_value(tags, "basic_acoustic.duration_sec") is None
        ):
            stages.add(STAGE_SPEAKER)
        if (
            _tag_value(tags, "language_content.filler") is None
            or _tag_value(tags, "language_content.word_count") is None
        ):
            stages.add(STAGE_LANGUAGE_DETERMINISTIC)
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
    stages=None,
    transcript_override=None,
):
    # type: (Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[set], Optional[str]) -> None
    stages = stages or set([STAGE_LANGUAGE_DETERMINISTIC])
    sample = record["sample"]
    transcript = (
        sample.get("text", {}).get("transcript", "")
        if transcript_override is None
        else transcript_override
    )
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
            for field in ("word_count", "punctuation", "repetition", "filler"):
                tags["language_content"][field] = None


def _run_firered_lid_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    firered_lid_config=None,
):
    try:
        result = FIRERED_LID_LANGUAGE_CONTENT_TOOL["run"](
            audio_path,
            context=tool_context,
            config=firered_lid_config,
        )
        apply_result(tags, internal_results, result)
    except Exception as exc:  # noqa: BLE001 - no non-FireRed LID fallback is allowed.
        warnings.append(
            {
                "type": "firered_lid_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": FIRERED_LID_LANGUAGE_CONTENT_TOOL["tool_name"],
            }
        )
        tags["language_content"]["language"] = None


def _run_speaker_aware_lid_tool(
    audio_path,
    speaker_output,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    firered_lid_config=None,
    segment_lid_config=None,
    artifact_dir=None,
    artifact_record_index=None,
):
    """Publish speaker-aggregate LID or the speaker route's full-audio LID."""

    config = segment_lid_config or SpeakerSegmentLidConfig()
    full_audio_result = {
        "language": speaker_output.get("detected_language"),
        "confidence": speaker_output.get("language_confidence"),
        "error": speaker_output.get("language_error"),
        "evidence_id": speaker_output.get("language_evidence_id"),
        "language_route": speaker_output.get("language_route"),
    }
    if speaker_output.get("language_published_in_speaker_v2"):
        _reuse_speaker_lid_result(
            speaker_output,
            tags,
            internal_results,
            warnings,
            sample_id,
        )
        return
    if not config.enabled:
        if speaker_output.get("detected_language"):
            _reuse_speaker_lid_result(
                speaker_output,
                tags,
                internal_results,
                warnings,
                sample_id,
            )
            return
        if audio_path is not None and Path(audio_path).exists():
            _run_firered_lid_tool(
                audio_path,
                tool_context,
                tags,
                internal_results,
                warnings,
                sample_id,
                firered_lid_config=firered_lid_config,
            )
            return
        tags["language_content"]["language"] = None
        return
    try:
        result = run_speaker_segment_lid(
            audio_path,
            speaker_output.get("selected_timeline"),
            full_audio_result,
            context=tool_context,
            config=config,
            lid_config=firered_lid_config,
            artifact_dir=artifact_dir,
            sample_id=_speaker_artifact_sample_key(
                sample_id,
                artifact_record_index,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve the full-audio fallback.
        warnings.append(
            {
                "type": "speaker_segment_lid_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path) if audio_path is not None else None,
                "tool_name": "firered_lid_detector",
            }
        )
        result = None

    if result is not None and result.value is not None:
        apply_result(tags, internal_results, result)
        fallback_reason = result.evidence.get("fallback_reason")
        if fallback_reason in (
            "speaker_lid_failed",
            "speaker_audio_extraction_failed",
        ):
            warnings.append(
                {
                    "type": "speaker_segment_lid_fallback",
                    "message": fallback_reason,
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "tool_name": "firered_lid_detector",
                }
            )
        return

    if result is not None:
        internal_results.append(result.to_record())
    if not config.fallback_to_full_audio:
        tags["language_content"]["language"] = None
        return
    if audio_path is not None and Path(audio_path).exists():
        _run_firered_lid_tool(
            audio_path,
            tool_context,
            tags,
            internal_results,
            warnings,
            sample_id,
            firered_lid_config=firered_lid_config,
        )
        return
    _reuse_speaker_lid_result(
        speaker_output,
        tags,
        internal_results,
        warnings,
        sample_id,
    )


def _reuse_speaker_lid_result(
    speaker_output,
    tags,
    internal_results,
    warnings,
    sample_id,
):
    """Publish the FireRed LID observation already used by speaker routing."""

    language = speaker_output.get("detected_language")
    if not isinstance(language, str) or not language.strip():
        tags["language_content"]["language"] = None
        error = speaker_output.get("language_error")
        if error:
            warnings.append(
                {
                    "type": "firered_lid_error",
                    "message": str(error),
                    "sample_id": sample_id,
                    "tool_name": "firered_lid_detector",
                    "reused_from": "speaker_v2",
                }
            )
        return

    language = language.strip()
    confidence = speaker_output.get("language_confidence")
    tags["language_content"]["language"] = language
    internal_results.append(
        ToolResult(
            tag_path="language_content.language",
            value=language,
            tool_name="firered_lid_detector",
            method="FireRed LID (reused from speaker-v2 routing)",
            status="estimated",
            confidence=confidence,
            tool_type="model",
            tool_version="speaker_v2.firered_lid_reuse.1",
            evidence={
                "reused_from": "speaker_v2",
                "speaker_lid_evidence_id": speaker_output.get(
                    "language_evidence_id"
                ),
                "language_route": speaker_output.get("language_route"),
            },
        ).to_record()
    )


def _run_speaker_tools(
    audio_path,
    record,
    manifest_dir,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    speaker_config=None,
    speaker_segment_lid_config=None,
    artifact_dir=None,
    artifact_record_index=None,
):
    # type: (Path, Dict[str, Any], Union[str, Path], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[SpeakerEvidenceConfig], Optional[Union[str, Path]], Optional[int]) -> str
    duration_sec = tags["basic_acoustic"].get("duration_sec")
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
        return _empty_speaker_output()

    config = speaker_config or default_speaker_evidence_config()
    artifact_root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else Path(manifest_dir) / "artifacts"
    )
    try:
        result = run_speaker_v2_record(
            record,
            manifest_dir,
            manifest_dir,
            config,
            context=tool_context,
            artifact_root=artifact_root,
            artifact_sample_id=_speaker_artifact_sample_key(
                sample_id,
                artifact_record_index,
            ),
            speaker_segment_lid_config=speaker_segment_lid_config,
        )
        speaker = result.get("speaker")
        if not isinstance(speaker, dict):
            raise ValueError("speaker-v2 result is missing the public speaker object")
        missing_fields = sorted(
            set(
                field
                for field in SPEAKER_FIELDS
                if field not in ("profiles", "speaker_present", "asr_transcript")
            )
            - set(speaker)
        )
        if missing_fields:
            raise ValueError(
                "speaker-v2 result is missing fields: %s"
                % ", ".join(missing_fields)
            )
        for field in SPEAKER_FIELDS:
            tags["speaker"][field] = speaker.get(field)
        asr_transcript = result.get("speaker_asr_transcript", "")
        if not isinstance(asr_transcript, str):
            asr_transcript = ""
        asr_transcript = asr_transcript.strip()
        tags["speaker"]["asr_transcript"] = asr_transcript or None
        tags["speaker"]["speaker_present"] = _speaker_present_from_count(
            tags["speaker"].get("speaker_count")
        )
        internal_results.append(
            ToolResult(
                tag_path="speaker",
                value=dict(tags["speaker"]),
                tool_name="speaker_v2",
                method="speaker_evidence_v2_direct",
                status="estimated",
                confidence=1.0,
                tool_type="model",
                tool_version="speaker_v2.direct.1",
                evidence={
                    "run_profile": result.get("run_profile"),
                    "policy_version": result.get("policy_version"),
                    "policy_hash": result.get("policy_hash"),
                    "fusion_artifact": result.get("fusion_artifact"),
                    "artifacts": result.get("artifacts"),
                    "asr_selected_source": result.get("asr_selected_source"),
                    "language_route": result.get("language_route"),
                    "language_route_reason": result.get(
                        "language_route_reason"
                    ),
                    "asr_route": result.get("asr_route"),
                    "selected_timeline_source": (
                        (result.get("selected_timeline") or {}).get("source")
                        if isinstance(result.get("selected_timeline"), dict)
                        else None
                    ),
                },
            ).to_record()
        )
        return {
            "transcript": asr_transcript,
            "detected_language": result.get("detected_language"),
            "language_confidence": result.get("language_confidence"),
            "language_error": result.get("language_error"),
            "language_evidence_id": result.get("language_evidence_id"),
            "language_route": result.get("language_route"),
            "language_source": result.get("language_source"),
            "language_published_in_speaker_v2": bool(
                result.get("language_published_in_speaker_v2")
            ),
            "selected_timeline": result.get("selected_timeline"),
        }
    except Exception as exc:  # noqa: BLE001 - speaker failures become internal warnings.
        warnings.append(
            {
                "type": "speaker_v2_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": "speaker_v2",
            }
        )
        _null_speaker_tags(tags)
        return _empty_speaker_output()


def _empty_speaker_output():
    return {
        "transcript": "",
        "detected_language": None,
        "language_confidence": None,
        "language_error": None,
        "language_evidence_id": None,
        "language_route": None,
        "language_source": None,
        "language_published_in_speaker_v2": False,
        "selected_timeline": None,
    }


def _null_speaker_tags(tags):
    for field in SPEAKER_FIELDS:
        tags["speaker"][field] = None


def _set_no_transcript_speaker_tags(tags):
    tags["speaker"]["speaker_count"] = 0
    tags["speaker"]["speaker_present"] = False
    tags["speaker"]["asr_transcript"] = None
    tags["speaker"]["multi_speaker"] = False
    tags["speaker"]["speaker_change_count"] = 0
    tags["speaker"]["speaker_change"] = False
    tags["speaker"]["overlap_ratio"] = 0.0
    tags["speaker"]["speaker_overlap"] = False
    tags["speaker"]["profiles"] = []


def _null_language_content_tags(tags):
    for field in LANGUAGE_CONTENT_FIELDS:
        tags["language_content"][field] = None


def _null_silence_tags(tags):
    tags["basic_acoustic"]["silence_segments"] = None
    tags["basic_acoustic"]["nonspeech_ratio"] = None


def _null_recrir_tags(tags):
    tags["room_acoustic"]["rt60_sec"] = None
    tags["room_acoustic"]["c50_db"] = None


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
        public_paths = set(STAGE_TAG_PATHS[STAGE_BROUHAHA])
        for result in results:
            if result.tag_path in public_paths:
                apply_result(tags, internal_results, result)
            else:
                # Brouhaha C50 (internal.brouhaha_c50_db) stays internal
                # evidence for cross-validation against room_acoustic.c50_db.
                internal_results.append(result.to_record())
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
        tags["audio_quality"]["snr_db"] = None


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


def _run_dass_noise_type_tool(
    audio_path,
    tool_context,
    tags,
    internal_results,
    warnings,
    sample_id,
    dass_config=None,
    dass_client=None,
):
    try:
        results = DASS_NOISE_TYPE_TOOL["run"](
            audio_path,
            context=tool_context,
            config=dass_config,
            client=dass_client,
            music_present=tags["sound_field_scene"]["music_present"],
        )
        for result in results:
            apply_result(tags, internal_results, result)
    except Exception as exc:  # noqa: BLE001 - no non-DASS fallback is allowed.
        warnings.append(
            {
                "type": "dass_noise_type_error",
                "message": str(exc),
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "tool_name": DASS_NOISE_TYPE_TOOL["tool_name"],
            }
        )
        _null_dass_tags(tags)


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

    for tool, field in ((RT60_TOOL, "rt60_sec"), (C50_TOOL, "c50_db")):
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
            tags["room_acoustic"][field] = None


def _null_rir_related_tags(tags):
    tags["room_acoustic"]["rt60_sec"] = None
    tags["room_acoustic"]["c50_db"] = None


def _null_dnsmos_tags(tags):
    for field in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        tags["audio_quality"][field] = None


def _null_firered_aed_tags(tags):
    tags["sound_field_scene"]["speech_music_events"] = None
    tags["sound_field_scene"]["music_present"] = None


def _null_dass_tags(tags):
    tags["sound_field_scene"]["external_noise_type"] = None
    tags["sound_field_scene"]["noise_composition"] = None


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


def _speaker_artifact_sample_key(sample_id, record_index):
    # type: (str, Optional[int]) -> str
    row_number = 1 if record_index is None else record_index
    return "%s-sample-%d" % (sample_id, row_number)


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


def compare_native_metadata_audio_quality_fields(sample, observed_audio_quality):
    # type: (Dict[str, Any], Dict[str, Any]) -> List[Dict[str, Any]]
    native_metadata = sample.get("native_metadata", {})
    warnings = []  # type: List[Dict[str, Any]]
    comparisons = [
        ("snr_db", 1e-6),
        ("dnsmos_sig", 1e-6),
        ("dnsmos_bak", 1e-6),
        ("dnsmos_ovrl", 1e-6),
        ("dnsmos_p808", 1e-6),
    ]
    for field, tolerance in comparisons:
        native_value = native_metadata.get(field)
        observed_value = observed_audio_quality.get(field)
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
                    "type": "native_metadata_audio_quality_mismatch",
                    "field": field,
                    "native_metadata_value": native_value,
                    "observed_value": observed_value,
                    "message": "observed audio quality value differs from native metadata",
                }
            )
    return warnings


def compare_native_metadata_room_acoustic_fields(sample, observed_room_acoustic):
    # type: (Dict[str, Any], Dict[str, Any]) -> List[Dict[str, Any]]
    native_metadata = sample.get("native_metadata", {})
    warnings = []  # type: List[Dict[str, Any]]
    # Native metadata keys stay in dataset terms; observed keys use the new
    # naming (rt60_sec / c50_db).
    comparisons = [
        ("far_field", "far_field", None),
        ("rt60", "rt60_sec", 1e-6),
        ("c50", "c50_db", 1e-6),
    ]
    for native_key, observed_key, tolerance in comparisons:
        native_value = native_metadata.get(native_key)
        observed_value = observed_room_acoustic.get(observed_key)
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
                    "type": "native_metadata_room_acoustic_mismatch",
                    "field": observed_key,
                    "native_metadata_value": native_value,
                    "observed_value": observed_value,
                    "message": "observed room acoustic value differs from native metadata",
                }
            )
    return warnings


def compare_native_metadata_sound_field_scene_fields(sample, observed_sound_field):
    # type: (Dict[str, Any], Dict[str, Any]) -> List[Dict[str, Any]]
    native_metadata = sample.get("native_metadata", {})
    warnings = []  # type: List[Dict[str, Any]]
    # Native metadata keys stay in dataset terms (audio_events / music);
    # observed keys use the new naming (speech_music_events / music_present).
    comparisons = [
        ("audio_events", "speech_music_events", None),
        ("music", "music_present", None),
        # ("sound", "sound", None) removed with the PANNs stage (2026-08-25).
        ("external_noise_type", "external_noise_type", None),
    ]
    for native_key, observed_key, tolerance in comparisons:
        native_value = native_metadata.get(native_key)
        observed_value = observed_sound_field.get(observed_key)
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
                    "field": observed_key,
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

    return warnings


def audit_audio_quality(audio_quality):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    snr_db = audio_quality.get("snr_db")
    if snr_db is not None and (
        isinstance(snr_db, bool)
        or not isinstance(snr_db, (int, float))
        or not _is_finite_number(snr_db)
    ):
        audio_quality["snr_db"] = None
        warnings.append({"type": "invalid_audio_quality_value", "field": "snr_db"})

    for field in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"):
        value = audio_quality.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite_number(value)
            or value < 1.0
            or value > 5.0
        ):
            audio_quality[field] = None
            warnings.append(
                {"type": "invalid_audio_quality_value", "field": field}
            )

    return warnings


def audit_room_acoustic(room_acoustic):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    far_field = room_acoustic.get("far_field")
    if far_field is not None and not isinstance(far_field, bool):
        room_acoustic["far_field"] = None
        warnings.append(
            {"type": "invalid_room_acoustic_value", "field": "far_field"}
        )

    rt60_sec = room_acoustic.get("rt60_sec")
    if rt60_sec is not None and (
        isinstance(rt60_sec, bool)
        or not isinstance(rt60_sec, (int, float))
        or not _is_finite_number(rt60_sec)
        or rt60_sec < 0
    ):
        room_acoustic["rt60_sec"] = None
        warnings.append(
            {"type": "invalid_room_acoustic_value", "field": "rt60_sec"}
        )

    c50_db = room_acoustic.get("c50_db")
    if c50_db is not None and (
        isinstance(c50_db, bool)
        or not isinstance(c50_db, (int, float))
        or not _is_finite_number(c50_db)
    ):
        room_acoustic["c50_db"] = None
        warnings.append(
            {"type": "invalid_room_acoustic_value", "field": "c50_db"}
        )

    return warnings


def audit_sound_field_scene(sound_field_scene):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    music_present = sound_field_scene.get("music_present")
    if music_present is not None and not isinstance(music_present, bool):
        sound_field_scene["music_present"] = None
        warnings.append(
            {"type": "invalid_sound_field_scene_value", "field": "music_present"}
        )

    speech_music_events = sound_field_scene.get("speech_music_events")
    if speech_music_events is not None and not _is_valid_label_list(
        speech_music_events,
        allowed=EVENT_NAMES,
        require_allowed_order=True,
    ):
        sound_field_scene["speech_music_events"] = None
        warnings.append(
            {
                "type": "invalid_sound_field_scene_value",
                "field": "speech_music_events",
            }
        )

    if "sound" in sound_field_scene:
        # Deprecated on 2026-08-25: the PANNs stage and its public field
        # were removed (noise_composition supersedes them). A non-null value
        # can only come from stale input, so it is dropped (with a warning);
        # the key is deleted so the deprecated field never reaches output.
        if sound_field_scene["sound"] is not None:
            warnings.append(
                {"type": "deprecated_sound_field_scene_value", "field": "sound"}
            )
        del sound_field_scene["sound"]

    external_noise_type = sound_field_scene.get("external_noise_type")
    if external_noise_type is not None and not _is_valid_label_list(
        external_noise_type,
        allowed=PUBLIC_COMPOSITION_CATEGORIES,
        max_items=len(PUBLIC_COMPOSITION_CATEGORIES),
    ):
        sound_field_scene["external_noise_type"] = None
        warnings.append(
            {
                "type": "invalid_sound_field_scene_value",
                "field": "external_noise_type",
            }
        )

    noise_composition = sound_field_scene.get("noise_composition")
    if noise_composition is not None and not _is_valid_noise_composition(
        noise_composition
    ):
        sound_field_scene["noise_composition"] = None
        warnings.append(
            {
                "type": "invalid_sound_field_scene_value",
                "field": "noise_composition",
            }
        )

    music_present = sound_field_scene.get("music_present")
    speech_music_events = sound_field_scene.get("speech_music_events")
    if (
        music_present is not None
        and speech_music_events is not None
        and music_present != ("music" in speech_music_events)
    ):
        sound_field_scene["speech_music_events"] = None
        sound_field_scene["music_present"] = None
        warnings.append(
            {
                "type": "inconsistent_sound_field_scene_values",
                "fields": ["speech_music_events", "music_present"],
            }
        )

    return warnings


def _is_valid_noise_composition(value):
    if not isinstance(value, dict):
        return False
    if set(value.keys()) != set(PUBLIC_COMPOSITION_CATEGORIES):
        return False
    for category in PUBLIC_COMPOSITION_CATEGORIES:
        labels = value[category]
        if not _is_valid_label_list(
            labels,
            max_items=NOISE_COMPOSITION_AUDIT_MAX_ITEMS,
        ):
            return False
        if any(
            classify_dass_label(label) != category for label in labels
        ):
            return False
    return True


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


_SPEAKER_PROFILE_RATE_BANDS = (None, "slow", "normal", "fast", "variable")
_SPEAKER_PROFILE_RATE_UNITS = (None, "zh_char_per_sec", "word_per_min")
_SPEAKER_PROFILE_PITCH_VALUES = (None, "low", "mid", "high", "variable")
_SPEAKER_PROFILE_VOLUME_VALUES = (None, "low", "normal", "loud", "variable")
_SPEAKER_PROFILE_GENDER_VALUES = (None, "male", "female")


def _is_valid_speaker_profile(item):
    # Mirrors the speaker-v2 resolver profile normalization rules.
    if not isinstance(item, dict):
        return False
    speaker_id = item.get("speaker_id")
    if not isinstance(speaker_id, str) or not re.match(
        r"^speaker_[1-9][0-9]*$", speaker_id
    ):
        return False
    rate = item.get("speech_rate")
    if not isinstance(rate, dict):
        return False
    band = rate.get("band")
    unit = rate.get("unit")
    number = rate.get("value")
    if band not in _SPEAKER_PROFILE_RATE_BANDS:
        return False
    if unit not in _SPEAKER_PROFILE_RATE_UNITS:
        return False
    if number is not None and (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not _is_finite_number(number)
        or unit is None
    ):
        return False
    if item.get("pitch") not in _SPEAKER_PROFILE_PITCH_VALUES:
        return False
    if item.get("speaker_volume") not in _SPEAKER_PROFILE_VOLUME_VALUES:
        return False
    if item.get("gender") not in _SPEAKER_PROFILE_GENDER_VALUES:
        return False
    return True


def audit_speaker(speaker):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

    for field in ("multi_speaker", "speaker_change", "speaker_overlap"):
        value = speaker.get(field)
        if value is not None and not isinstance(value, bool):
            speaker[field] = None
            warnings.append({"type": "invalid_speaker_value", "field": field})

    for field in ("speaker_count", "speaker_change_count"):
        value = speaker.get(field)
        if value is not None and not _is_non_negative_int(value):
            speaker[field] = None
            warnings.append({"type": "invalid_speaker_value", "field": field})

    expected_speaker_present = _speaker_present_from_count(
        speaker.get("speaker_count")
    )
    speaker_present = speaker.get("speaker_present")
    if speaker_present is not None and (
        not isinstance(speaker_present, bool)
        or speaker_present != expected_speaker_present
    ):
        warnings.append(
            {"type": "invalid_speaker_value", "field": "speaker_present"}
        )
    speaker["speaker_present"] = expected_speaker_present

    asr_transcript = speaker.get("asr_transcript")
    if asr_transcript is not None:
        if not isinstance(asr_transcript, str) or not asr_transcript.strip():
            speaker["asr_transcript"] = None
            warnings.append(
                {"type": "invalid_speaker_value", "field": "asr_transcript"}
            )
        else:
            speaker["asr_transcript"] = asr_transcript.strip()

    overlap_ratio = speaker.get("overlap_ratio")
    if overlap_ratio is not None and (
        isinstance(overlap_ratio, bool)
        or not isinstance(overlap_ratio, (int, float))
        or not _is_finite_number(overlap_ratio)
        or not 0 <= overlap_ratio <= 1
    ):
        speaker["overlap_ratio"] = None
        warnings.append(
            {"type": "invalid_speaker_value", "field": "overlap_ratio"}
        )

    profiles = speaker.get("profiles")
    if profiles is not None:
        valid = isinstance(profiles, list) and all(
            _is_valid_speaker_profile(item) for item in profiles
        )
        if valid:
            profile_ids = [item.get("speaker_id") for item in profiles]
            valid = len(set(profile_ids)) == len(profile_ids)
        if not valid:
            speaker["profiles"] = None
            warnings.append(
                {"type": "invalid_speaker_value", "field": "profiles"}
            )

    return warnings


def _speaker_present_from_count(speaker_count):
    if not _is_non_negative_int(speaker_count):
        return None
    return speaker_count > 0


def audit_language_content(language_content):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]

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


def _derive_deterministic_tags(tags):
    """Populate contract-only values from finalized structural source tags."""
    derived = tags["deterministic_derived"]
    change_count = tags["speaker"].get("speaker_change_count")
    duration_sec = tags["basic_acoustic"].get("duration_sec")
    if (
        _is_non_negative_int(change_count)
        and isinstance(duration_sec, (int, float))
        and not isinstance(duration_sec, bool)
        and _is_finite_number(duration_sec)
        and duration_sec > 0
    ):
        derived["speaker_change_rate_per_min"] = round(
            float(change_count) * 60.0 / float(duration_sec), 6
        )
    else:
        derived["speaker_change_rate_per_min"] = None

    filler = tags["language_content"].get("filler")
    word_count = tags["language_content"].get("word_count")
    if (
        _is_non_negative_int(filler)
        and _is_non_negative_int(word_count)
        and word_count > 0
    ):
        derived["filler_ratio"] = round(float(filler) / float(word_count), 6)
    else:
        derived["filler_ratio"] = None


def audit_deterministic_derived(derived):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    warnings = []  # type: List[Dict[str, Any]]
    for field in DETERMINISTIC_DERIVED_FIELDS:
        value = derived.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite_number(value)
            or value < 0
        ):
            derived[field] = None
            warnings.append(
                {"type": "invalid_deterministic_derived_value", "field": field}
            )
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
            "example speaker.asr_transcript,language_content.word_count,"
            "basic_acoustic.nonspeech_ratio."
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
        "--firered-aed-min-singing-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum singing event ratio for singing to enter "
            "sound_field_scene.speech_music_events. Defaults to 0.10, "
            "matching --firered-aed-min-music-ratio; calibrated on "
            "caption_pairs_3000 where singing false positives on speech "
            "stay below 0.10."
        ),
    )
    parser.add_argument(
        "--firered-aed-min-music-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum music event ratio for sound_field_scene.music_present "
            "to be true. Defaults to 0.10, calibrated on caption_pairs_3000: "
            "speech frames with music confidence above the frame threshold "
            "produce short segments whose ratio stays below 0.10, while "
            "real music occupies far more of the clip."
        ),
    )
    dass_device_group = parser.add_mutually_exclusive_group()
    dass_device_group.add_argument(
        "--dass-use-gpu",
        dest="dass_use_gpu",
        action="store_true",
        help="Use GPU for DASS noise-type inference (the default).",
    )
    dass_device_group.add_argument(
        "--dass-use-cpu",
        dest="dass_use_gpu",
        action="store_false",
        help="Run DASS inference on CPU instead of the default GPU.",
    )
    parser.set_defaults(dass_use_gpu=True)
    parser.add_argument(
        "--dass-threshold",
        type=float,
        default=0.25,
        help=(
            "DASS noise-type probability threshold. Defaults to 0.25 "
            "(calibrated on phase2: DASS-medium scores are soft, clean "
            "speech stays below 0.15). Categories present in the full "
            "527-class vector at or above this threshold populate "
            "sound_field_scene.external_noise_type."
        ),
    )
    parser.add_argument(
        "--no-exclusion",
        action="store_true",
        help=(
            "Disable DASS class exclusion entirely. By default primary speech, "
            "silence, acoustic-scene, reverberation, and echo labels are "
            "excluded from the ranked top events evidence; with this flag "
            "every AudioSet class stays eligible so the raw class "
            "distribution remains visible."
        ),
    )
    parser.add_argument(
        "--dass-composition-threshold",
        type=float,
        default=0.25,
        help=(
            "DASS noise-composition per-category probability threshold. "
            "Defaults to 0.25, aligned with --dass-threshold so every "
            "present category has a non-empty composition bucket."
        ),
    )
    parser.add_argument(
        "--dass-composition-top-k",
        type=int,
        default=3,
        help=(
            "Maximum DASS noise-composition labels kept per category. "
            "Defaults to 3."
        ),
    )
    parser.add_argument(
        "--dass-python",
        default=None,
        help="Python executable for DASS subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--firered-lid-python",
        default=None,
        help="Python executable for FireRed LID subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--firered-lid-model",
        default=None,
        help="FireRed LID model directory. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--firered-lid-repo",
        default=None,
        help="FireRed LID source directory. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--firered-lid-use-gpu",
        action="store_true",
        help="Use GPU in FireRed LID config. Defaults to CPU.",
    )
    parser.add_argument(
        "--firered-lid-half",
        action="store_true",
        help="Use half precision in standalone FireRed LID.",
    )
    parser.add_argument(
        "--speaker-segment-lid",
        dest="speaker_segment_lid_enabled",
        action="store_true",
        help=(
            "Use speaker-aware FireRed LID on longest clean segments to decide "
            "language. Disabled by default; the pipeline uses full-audio LID."
        ),
    )
    parser.add_argument(
        "--no-speaker-segment-lid",
        dest="speaker_segment_lid_enabled",
        action="store_false",
        help="Disable speaker-aware FireRed LID and publish full-audio LID.",
    )
    parser.add_argument(
        "--no-language-only-speaker-stage",
        dest="language_only_start_speaker_stage",
        action="store_false",
        help=(
            "Do not start speaker-v2 when language is the only requested tag; "
            "speaker-aware LID can still reuse an already requested speaker stage."
        ),
    )
    parser.add_argument(
        "--speaker-segment-lid-min-speaker-sec",
        type=float,
        default=2.0,
        help=(
            "Minimum longest clean speech segment duration per speaker. "
            "Defaults to 2.0."
        ),
    )
    parser.add_argument(
        "--speaker-segment-lid-min-total-sec",
        type=float,
        default=3.0,
        help=(
            "Minimum total duration of the selected longest clean segments "
            "across eligible speakers. Defaults to 3.0."
        ),
    )
    parser.add_argument(
        "--speaker-segment-lid-shadow-only",
        action="store_true",
        help="Compute speaker-aware LID evidence but publish the full-audio result.",
    )
    parser.add_argument(
        "--speaker-segment-lid-keep-artifacts",
        action="store_true",
        help="Keep per-speaker selected longest-segment WAV files under the artifact directory.",
    )
    parser.set_defaults(
        speaker_segment_lid_enabled=False,
        language_only_start_speaker_stage=True,
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
        "--dnsmos-use-gpu",
        action="store_true",
        help="Use CUDAExecutionProvider for DNSMOS; fail instead of falling back to CPU.",
    )
    parser.add_argument(
        "--dnsmos-personalized",
        action="store_true",
        help="Use the personalized DNSMOS primary model. Defaults to regular DNSMOS.",
    )
    parser.add_argument(
        "--speaker-profile",
        choices=available_speaker_profiles(),
        default="quality-shadow",
        help="Speaker-v2 model and claim-routing profile. Defaults to quality-shadow.",
    )
    parser.add_argument(
        "--firered-asr-model",
        default=_project_config_path(FIRERED_ASR_MODEL_DIR),
        help="FireRedASR2-AED model directory for other-language transcription.",
    )
    parser.add_argument(
        "--firered-asr-source",
        default=_project_config_path(FIRERED_ASR_SOURCE_DIR),
        help="FireRedASR2S source directory.",
    )
    parser.add_argument(
        "--firered-asr-python",
        default=_project_config_path(FIRERED_ASR_PYTHON),
        help="Python executable for the independent FireRed ASR worker.",
    )
    parser.add_argument(
        "--firered-asr-device",
        choices=("auto", "cpu", "cuda", "cuda:0"),
        default="cuda:0",
        help="Device visible inside the FireRed ASR subprocess.",
    )
    parser.add_argument(
        "--firered-asr-half",
        action="store_true",
        help="Use half precision for FireRedASR2-AED.",
    )
    parser.add_argument(
        "--firered-asr-timeout-sec",
        type=int,
        default=FIRERED_ASR_TIMEOUT_SEC,
        help="Timeout for one independent FireRed ASR request.",
    )
    parser.add_argument(
        "--qwen3-asr-model",
        default=_project_config_path(QWEN3_ASR_MODEL_DIR),
        help="Qwen3-ASR-0.6B model directory for non-Chinese/English transcription.",
    )
    parser.add_argument(
        "--qwen3-forced-aligner-model",
        default=_project_config_path(QWEN3_FORCED_ALIGNER_MODEL_DIR),
        help="Qwen3-ForcedAligner model directory used for native timestamps.",
    )
    parser.add_argument(
        "--qwen3-asr-python",
        default=_project_config_path(QWEN3_ASR_PYTHON),
        help="Python executable for the Qwen3 ASR worker.",
    )
    parser.add_argument(
        "--qwen3-asr-device",
        default="cuda:0",
        help="Device visible inside the Qwen3 ASR subprocess.",
    )
    parser.add_argument(
        "--qwen3-asr-torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Torch dtype for Qwen3-ASR-0.6B.",
    )
    parser.add_argument(
        "--qwen3-asr-max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens for one Qwen3 ASR request.",
    )
    parser.add_argument(
        "--qwen3-asr-timeout-sec",
        type=int,
        default=QWEN3_ASR_TIMEOUT_SEC,
        help="Timeout for one Qwen3 ASR request.",
    )
    parser.add_argument(
        "--nemotron-model-dir",
        default=_project_config_path(NEMOTRON_MODEL_DIR),
        help="Local Nemotron 3 Diarization model directory.",
    )
    parser.add_argument(
        "--nemotron-python",
        default=_project_config_path(NEMOTRON_PYTHON),
        help="Python executable for the Nemotron worker.",
    )
    parser.add_argument(
        "--nemotron-device",
        default="cuda:0",
        help="Device visible inside the Nemotron subprocess.",
    )
    parser.add_argument(
        "--nemotron-timeout-sec",
        type=int,
        default=NEMOTRON_TIMEOUT_SEC,
        help="Timeout for one Nemotron request.",
    )
    parser.add_argument(
        "--speaker-v2-skip-model-verification",
        action="store_true",
        help="Skip pinned speaker-v2 model asset hash verification.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Directory for non-public artifacts such as Rec-RIR waveforms. "
            "Defaults to OUTPUT_PARENT/artifacts."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel sample tasks sharing one subprocess per model. "
            "This pipelines model stages without duplicating GPU weights."
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
        min_singing_ratio=args.firered_aed_min_singing_ratio,
        min_music_ratio=args.firered_aed_min_music_ratio,
        subprocess_python=args.firered_aed_python,
    )
    dass_config = DassNoiseTypeConfig(
        use_gpu=args.dass_use_gpu,
        threshold=args.dass_threshold,
        exclude_classes=not args.no_exclusion,
        composition_threshold=args.dass_composition_threshold,
        composition_top_k=args.dass_composition_top_k,
        subprocess_python=args.dass_python,
    )
    firered_lid_config = FireRedLidConfig(
        model_dir=args.firered_lid_model,
        repo_dir=args.firered_lid_repo,
        use_gpu=args.firered_lid_use_gpu,
        use_half=args.firered_lid_half,
        subprocess_python=args.firered_lid_python,
    )
    speaker_segment_lid_config = SpeakerSegmentLidConfig(
        enabled=args.speaker_segment_lid_enabled,
        language_only_start_speaker_stage=(
            args.language_only_start_speaker_stage
        ),
        min_speaker_duration_sec=args.speaker_segment_lid_min_speaker_sec,
        min_total_duration_sec=args.speaker_segment_lid_min_total_sec,
        shadow_only=args.speaker_segment_lid_shadow_only,
        keep_artifacts=args.speaker_segment_lid_keep_artifacts,
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
        use_gpu=args.dnsmos_use_gpu,
    )
    firered_asr_config = FireRedAsrConfig(
        model_dir=args.firered_asr_model,
        source_dir=args.firered_asr_source,
        subprocess_python=args.firered_asr_python,
        device=args.firered_asr_device,
        use_half=args.firered_asr_half,
        timeout_sec=args.firered_asr_timeout_sec,
        enable_lid=False,
    )
    qwen3_asr_config = Qwen3AsrConfig(
        model_dir=args.qwen3_asr_model,
        forced_aligner_model_dir=args.qwen3_forced_aligner_model,
        subprocess_python=args.qwen3_asr_python,
        device=args.qwen3_asr_device,
        torch_dtype=args.qwen3_asr_torch_dtype,
        max_new_tokens=args.qwen3_asr_max_new_tokens,
        timeout_sec=args.qwen3_asr_timeout_sec,
        return_time_stamps=True,
    )
    nemotron_config = Nemotron3DiarizationConfig(
        model_dir=args.nemotron_model_dir,
        subprocess_python=args.nemotron_python,
        timeout_sec=args.nemotron_timeout_sec,
        device=args.nemotron_device,
    )
    speaker_config = default_speaker_evidence_config(
        profile_id=args.speaker_profile,
        vad_config=firered_vad_config,
        brouhaha_config=brouhaha_config,
        firered_asr_config=firered_asr_config,
        qwen3_asr_config=qwen3_asr_config,
        nemotron_config=nemotron_config,
        firered_lid_config=firered_lid_config,
        speaker_segment_lid_config=speaker_segment_lid_config,
        verify_model_assets=not args.speaker_v2_skip_model_verification,
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
        dass_config=dass_config,
        firered_lid_config=firered_lid_config,
        sample_ids=args.sample_id,
        existing_tags_path=args.input_tags,
        selected_tag_paths=[args.only_tags] if args.only_tags else None,
        missing_only=args.missing_only,
        speaker_segment_lid_config=speaker_segment_lid_config,
        workers=args.workers,
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
