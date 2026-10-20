#!/usr/bin/env python3
"""Scene clustering for the sampled multi-data snapshot.

Implements cluster_rules_v0.5 from compare/docs/multi-data-scene-classification.md.

By default clustering uses the calibrated sure-tagger audio feature whitelist
and d_tag. With --use-all-sure-tagger-labels, all non-null sure-tagger labels
are discovered from the snapshot and included in a reproducible mixed-type
feature vector. With --use-wer-input, one ASR system's aligned WER/CER
error_rate is added as an additional, equally weighted distance axis.
K-medoids candidates are searched on the same tags-only d_tag. WER/CER may
filter candidates after the distance is frozen, but must not be mixed into
assignment.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
HERE = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import cluster_methods
DEFAULT_TAGS = HERE / "sampled-100-per-source" / "tags.jsonl"
DEFAULT_INDEX_MAP = HERE / "sampled-100-per-source" / "index_map.jsonl"
DEFAULT_RUN_METADATA = HERE / "sampled-100-per-source" / "tag_run_metadata.json"
DEFAULT_SAMPLED_MANIFEST = HERE / "sampled-100-per-source" / "manifest.jsonl"
DEFAULT_WER_CER = HERE / "sampled-100-per-source" / "sample_wer_cer.jsonl"
DEFAULT_SIDECAR = HERE / "wer-cer-results" / "sample_sidecar.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "result" / "scene-clusters"

MODEL_IDS = (
    "nvidia__parakeet-tdt-0.6b-v2",
    "Qwen__Qwen3-ASR-1.7B",
    "OpenMOSS-Team__MOSS-Transcribe-preview-2B",
    "FunAudioLLM__SenseVoiceSmall",
    "openai__whisper-large-v3-turbo",
    "ibm-granite__granite-speech-4.1-2b",
    "fun-asr-flash-2026-06-15",
)
BILINGUAL_MODELS = (
    "Qwen__Qwen3-ASR-1.7B",
    "FunAudioLLM__SenseVoiceSmall",
    "openai__whisper-large-v3-turbo",
)

CLUSTER_RULES_VERSION = "cluster_rules_v0.5"
ALL_LABELS_RULES_VERSION = "cluster_rules_v0.5+all_sure_tagger_labels_v1"
WER_INPUT_RULES_VERSION = "cluster_rules_v0.5+wer_input_experiment_v1"
ALL_LABELS_WER_INPUT_RULES_VERSION = (
    "cluster_rules_v0.5+all_sure_tagger_labels_v1+wer_input_experiment_v1"
)
CLUSTER_NAME_VERSION = "cluster_names_v0.4"
SCORING_PROTOCOL = "sure.eval.protocol.v1/standard_system"
MODE = "model_agnostic"
ASSIGNMENT_BASIS = "audio_tags_only"
ALL_LABELS_ASSIGNMENT_BASIS = "all_sure_tagger_labels"
ALL_LABELS_WER_ASSIGNMENT_BASIS = "all_sure_tagger_labels_plus_wer_cer"

NUMERIC_FEATURES = (
    ("basic_acoustic.duration_sec", "spatial"),
    ("basic_acoustic.nonspeech_ratio", "quality"),
    ("basic_acoustic.silence_segment_count", "spatial"),
    ("basic_acoustic.silence_total_sec", "spatial"),
    ("audio_quality.snr_db", "quality"),
    ("audio_quality.dnsmos_ovrl", "quality"),
    ("audio_quality.dnsmos_bak", "quality"),
    ("audio_quality.dnsmos_sig", "quality"),
    ("audio_quality.dnsmos_p808", "quality"),
    ("room_acoustic.rt60_sec", "spatial"),
    ("room_acoustic.c50_db", "spatial"),
    ("speaker.speaker_count", "interaction"),
    ("speaker.overlap_ratio", "interaction"),
    ("speaker.speaker_change_count", "interaction"),
    ("deterministic_derived.speaker_change_rate_per_min", "interaction"),
)
BINARY_FEATURES = (
    ("sound_field_scene.music_present", "content"),
    ("speaker.speaker_overlap", "interaction"),
    ("speaker.multi_speaker", "interaction"),
    ("speaker.speaker_change", "interaction"),
    ("speaker.speaker_present", "interaction"),
)
SET_FEATURES = (
    ("sound_field_scene.speech_music_events", "content"),
    ("sound_field_scene.external_noise_type", "content"),
)
AXES = ("content", "interaction", "quality", "spatial")
ALL_LABELS_AXES = ("content", "interaction", "quality", "spatial", "language")
ERROR_AXIS = "wer_cer_error_rate"
DEFAULT_WER_INPUT_MODEL = "fun-asr-flash-2026-06-15"
NOISE_KINDS = (
    "mechanical",
    "music",
    "animal",
    "nature",
    "channel_environment",
    "formless",
)
MISSING_WEIGHT_CAP = 0.15
EPS = 1e-8
INF = 1e9
_EMPTY_LIST = object()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%s: invalid JSON: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("%s:%s: expected a JSON object" % (path, line_number))
            rows.append(value)
    return rows


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_text_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nested_get(row, path):
    current = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def median(values):
    if not values:
        return None
    return statistics.median(values)


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def iqr(values):
    if len(values) < 4:
        return None
    return quantile(values, 0.75) - quantile(values, 0.25)


def mad(values, center=None):
    if not values:
        return None
    if center is None:
        center = median(values)
    deviations = [abs(value - center) for value in values]
    return median(deviations)


def robust_scale(values):
    finite = [float(value) for value in values if is_number(value)]
    if not finite:
        return None, None, True
    center = median(finite)
    spread = iqr(finite)
    if spread is None or spread <= EPS:
        spread = (mad(finite, center) or 0.0) * 1.4826
    if spread <= EPS:
        std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        spread = 1.4826 * std if std > EPS else None
    if spread is None or spread <= EPS:
        return center, None, True
    return center, float(spread), False


def clip01(value):
    if value is None or not math.isfinite(value):
        return None
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def ami_meeting_id(key):
    if not isinstance(key, str):
        return None
    parts = key.split("_")
    if len(parts) >= 2 and parts[0] == "AMI":
        return parts[1]
    return None


def seniortalk_speaker(key):
    if not isinstance(key, str):
        return None
    head = key.split("__", 1)[0]
    return head.split("S", 1)[0] or head


def group_id_for(manifest_row):
    dataset = str(manifest_row.get("dataset") or "unknown")
    meta = manifest_row.get("metadata") if isinstance(manifest_row.get("metadata"), dict) else {}
    key = manifest_row.get("key")
    if dataset == "esb_tedlium__v1.0.1":
        token = meta.get("talk_id") or manifest_row.get("source_audio") or key
    elif dataset == "slidespeech_test":
        token = manifest_row.get("source_audio") or key
    elif dataset == "esb_ami__v1.0.1":
        token = ami_meeting_id(key) or key
    elif dataset == "BAAI_ChildMandarin_test__v1.0.1":
        token = manifest_row.get("speaker_id") or meta.get("speaker_id") or key
    elif dataset == "speech_asr_aishell1_testsets":
        token = meta.get("speaker_id") or key
    elif dataset == "BAAI_SeniorTalk_sentence_test__v1.0.1":
        token = seniortalk_speaker(key) or key
    elif dataset == "esb_spgispeech_test__v1.0.1":
        token = str(key).rsplit("_", 1)[0] if key else None
    else:
        token = manifest_row.get("source_audio") or meta.get("parent_sample_id") or key
    return "%s::%s" % (dataset, token)


def as_float(value):
    if is_number(value):
        return float(value)
    return None


def as_bool(value):
    if isinstance(value, bool):
        return value
    return None


def as_set(value):
    if value is None:
        return None
    if isinstance(value, list):
        return frozenset(str(item) for item in value)
    return None


def default_feature_specs():
    """Return the frozen, backwards-compatible audio tag feature schema."""
    return {
        "numeric": tuple(NUMERIC_FEATURES),
        "binary": tuple(BINARY_FEATURES),
        "categorical": tuple(),
        "sets": tuple(SET_FEATURES) + (("sound_field_scene.noise_detail", "content"),),
        "axes": tuple(AXES),
        "all_labels": False,
    }


def axis_for_all_label(path):
    """Map a discovered sure-tagger path to a semantic distance axis."""
    root = path.split(".", 1)[0]
    if "asr_transcript" in path or "transcript" in path:
        return "language"
    if root == "sound_field_scene":
        return "content"
    if root == "speaker":
        return "interaction"
    if root == "audio_quality":
        return "quality"
    if root == "room_acoustic":
        return "spatial"
    if root == "basic_acoustic":
        return "spatial"
    if root == "language_content":
        return "language"
    if root == "deterministic_derived":
        if "speaker" in path or "change" in path:
            return "interaction"
        return "language"
    # Descriptive fields are still sure-tagger labels. They describe audible
    # content or interaction and therefore share the content axis rather than
    # creating an unbounded sixth axis.
    if root == "descriptive":
        return "content"
    return "content"


def _collect_all_label_values(value, path, values, list_paths):
    """Flatten one tag row while preserving list wildcards and null/missing."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "%s.%s" % (path, key) if path else str(key)
            _collect_all_label_values(child, child_path, values, list_paths)
        return
    if isinstance(value, list):
        list_paths.add(path)
        # Container size is useful for profiles/transcript arrays and lets an
        # empty list remain distinct from an unavailable (null) container.
        values.setdefault(path + ".__count", []).append(float(len(value)))
        if not value:
            values.setdefault(path, []).append(_EMPTY_LIST)
        for child in value:
            if isinstance(child, dict):
                _collect_all_label_values(child, path + "[]", values, list_paths)
            elif isinstance(child, list):
                _collect_all_label_values(child, path + "[]", values, list_paths)
            elif child is not None:
                values.setdefault(path, []).append(child)
        return
    if value is not None:
        values.setdefault(path, []).append(value)


def _collect_all_label_paths(value, path, paths):
    """Collect scalar schema paths, including fields that are null everywhere."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "%s.%s" % (path, key) if path else str(key)
            _collect_all_label_paths(child, child_path, paths)
        return
    if isinstance(value, list):
        for child in value:
            if isinstance(child, dict):
                _collect_all_label_paths(child, path + "[]", paths)
            elif isinstance(child, list):
                _collect_all_label_paths(child, path + "[]", paths)
        return
    if path:
        paths.add(path)


def _normalise_all_label_values(tag_row):
    """Return typed feature maps for every scalar/list leaf in a tag row."""
    values = {}
    list_paths = set()
    _collect_all_label_values(tag_row, "", values, list_paths)
    numeric = {}
    binary = {}
    categorical = {}
    sets = {}
    for path, raw_values in values.items():
        if not path or not raw_values:
            continue
        finite_values = [value for value in raw_values if is_number(value)]
        non_null = [value for value in raw_values if value is not None and value is not _EMPTY_LIST]
        has_empty_list = any(value is _EMPTY_LIST for value in raw_values)
        if has_empty_list and not non_null:
            sets[path] = frozenset()
            continue
        if non_null and len(finite_values) == len(non_null):
            numeric[path] = sum(float(value) for value in finite_values) / float(len(finite_values))
            continue
        if (
            non_null
            and all(isinstance(value, bool) for value in non_null)
            and path not in list_paths
            and "[]" not in path
        ):
            binary[path] = non_null[-1]
            continue
        if (
            non_null
            and all(isinstance(value, str) for value in non_null)
            and path not in list_paths
            and "[]" not in path
        ):
            categorical[path] = non_null[-1]
            continue
        # Lists and repeated values (for example profile fields) are treated
        # as nominal sets and compared with Jaccard distance.
        sets[path] = frozenset(
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value)
            for value in non_null
        )
    return {"numeric": numeric, "binary": binary, "categorical": categorical, "sets": sets}


def discover_all_feature_specs(tag_rows):
    """Discover stable mixed-type feature paths over the complete snapshot."""
    discovered = {"numeric": set(), "binary": set(), "categorical": set(), "sets": set()}
    all_paths = set()
    for tag_row in tag_rows:
        row_features = _normalise_all_label_values(tag_row)
        _collect_all_label_paths(tag_row, "", all_paths)
        for kind in discovered:
            discovered[kind].update(row_features[kind])
    # A field that is null in every row cannot change a distance, but retaining
    # it in the recorded schema makes "all labels" auditable and keeps its
    # missing-only status explicit in the output feature specification.
    known_paths = set().union(*(discovered[kind] for kind in discovered))
    discovered["categorical"].update(all_paths - known_paths)
    # A malformed snapshot may emit different JSON types for one path. Keep
    # one deterministic owner per path so a field cannot be counted twice in
    # the distance vector.
    owners = {}
    for kind in ("numeric", "binary", "categorical", "sets"):
        for path in sorted(discovered[kind]):
            owners.setdefault(path, kind)
    for kind in discovered:
        discovered[kind] = {path for path in discovered[kind] if owners.get(path) == kind}
    specs = {}
    for kind in ("numeric", "binary", "categorical", "sets"):
        specs[kind] = tuple(
            (path, axis_for_all_label(path)) for path in sorted(discovered[kind])
        )
    specs["axes"] = tuple(ALL_LABELS_AXES)
    specs["all_labels"] = True
    return specs


def extract_tag_features(tag_row, feature_specs=None):
    if feature_specs is not None and feature_specs.get("all_labels"):
        normalised = _normalise_all_label_values(tag_row)
        features = {"numeric": {}, "binary": {}, "categorical": {}, "sets": {}}
        for kind in features:
            for path, _axis in feature_specs[kind]:
                features[kind][path] = normalised[kind].get(path)
        return features
    if feature_specs is None:
        feature_specs = default_feature_specs()
    numeric = {}
    binary = {}
    sets = {}
    for path, _axis in feature_specs["numeric"]:
        numeric[path] = as_float(nested_get(tag_row, path))
    for path, _axis in feature_specs["binary"]:
        binary[path] = as_bool(nested_get(tag_row, path))
    for path, _axis in feature_specs["sets"]:
        sets[path] = as_set(nested_get(tag_row, path))
    # The default schema computes this compact set from noise composition; it
    # is retained as a compatibility feature even though all-label mode also
    # includes every raw noise_composition child path.
    if "sound_field_scene.noise_detail" in sets:
        sets["sound_field_scene.noise_detail"] = None
    composition = nested_get(tag_row, "sound_field_scene.noise_composition")
    noise_detail = []
    if isinstance(composition, dict):
        for kind in NOISE_KINDS:
            items = composition.get(kind)
            if not isinstance(items, list):
                continue
            for item in items:
                if item:
                    noise_detail.append("%s:%s" % (kind, item))
    if "sound_field_scene.noise_detail" in sets:
        sets["sound_field_scene.noise_detail"] = frozenset(noise_detail)
    segs = nested_get(tag_row, "basic_acoustic.silence_segments")
    if segs is None:
        numeric["basic_acoustic.silence_segment_count"] = None
        numeric["basic_acoustic.silence_total_sec"] = None
    elif isinstance(segs, list):
        total = 0.0
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            start = as_float(seg.get("start_sec"))
            end = as_float(seg.get("end_sec"))
            if start is not None and end is not None and end > start:
                total += end - start
        numeric["basic_acoustic.silence_segment_count"] = float(len(segs))
        numeric["basic_acoustic.silence_total_sec"] = total
    else:
        numeric["basic_acoustic.silence_segment_count"] = None
        numeric["basic_acoustic.silence_total_sec"] = None
    # Preserve the historical default feature-map shape. The optional
    # all-label mode adds the categorical map explicitly above.
    return {"numeric": numeric, "binary": binary, "sets": sets}


def conflict_reasons(tag_row, features):
    reasons = []
    speaker_count = features["numeric"].get("speaker.speaker_count")
    multi = features["binary"].get("speaker.multi_speaker")
    overlap = features["binary"].get("speaker.speaker_overlap")
    overlap_ratio = features["numeric"].get("speaker.overlap_ratio")
    events = features["sets"].get("sound_field_scene.speech_music_events") or frozenset()
    if speaker_count is not None and multi is not None:
        if bool(speaker_count >= 2) != bool(multi):
            reasons.append("speaker_count_multi_speaker_mismatch")
    if overlap is True and overlap_ratio is not None and overlap_ratio <= EPS:
        reasons.append("overlap_flag_without_ratio")
    if speaker_count == 0 and "speech" in events:
        reasons.append("zero_speaker_with_speech_event")
    if features["binary"].get("speaker.speaker_present") is False and "speech" in events:
        reasons.append("no_speaker_with_speech_event")
    return reasons


def language_group(language):
    if not language:
        return "unknown"
    text = str(language).lower()
    if text.startswith("zh"):
        return "zh"
    if text == "en" or text.startswith("en-"):
        return "en"
    return "other_known"


TAG_ROOT_KEYS = (
    "audio_quality",
    "basic_acoustic",
    "deterministic_derived",
    "language_content",
    "room_acoustic",
    "sound_field_scene",
    "speaker",
)


def structural_tag_row(tag_row):
    """Keep sure-tagger structural groups; drop identity/descriptive extras."""
    if not isinstance(tag_row, dict):
        return tag_row
    if any(key in tag_row for key in TAG_ROOT_KEYS):
        return {key: tag_row[key] for key in TAG_ROOT_KEYS if key in tag_row}
    return tag_row


def load_tag_snapshot(args):
    tags = read_jsonl(args.tags)
    index_map = read_jsonl(args.index_map)
    sampled_manifest = read_jsonl(args.sampled_manifest)
    n = len(tags)
    if not (len(index_map) == len(sampled_manifest) == n):
        raise ValueError(
            "row-count mismatch: tags=%s index_map=%s sampled_manifest=%s"
            % (len(tags), len(index_map), len(sampled_manifest))
        )
    samples = []
    feature_specs = (
        discover_all_feature_specs([structural_tag_row(row) for row in tags])
        if getattr(args, "use_all_sure_tagger_labels", False)
        else default_feature_specs()
    )
    # Keep the discovered schema on args so the rest of the pipeline uses the
    # exact same columns and axes that were used while extracting each row.
    args.feature_specs = feature_specs
    seen_ids = set()
    for index, (tag_row, index_row, manifest_row) in enumerate(zip(tags, index_map, sampled_manifest)):
        if index_row.get("global_index") != index:
            raise ValueError("index_map is not contiguous at %s" % index)
        sample_id = index_row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("invalid sample_id at %s" % index)
        if sample_id in seen_ids:
            raise ValueError("duplicate sample_id %s" % sample_id)
        seen_ids.add(sample_id)
        tag_row = structural_tag_row(tag_row)
        features = extract_tag_features(tag_row, feature_specs)
        samples.append(
            {
                "global_index": index,
                "sample_id": sample_id,
                "dataset": manifest_row.get("dataset") or index_row.get("dataset"),
                "key": manifest_row.get("key") or index_row.get("key"),
                "language": manifest_row.get("language") or index_row.get("language"),
                "path": manifest_row.get("path"),
                "group_id": group_id_for(manifest_row),
                "tags": tag_row,
                "features": features,
                "conflicts": conflict_reasons(tag_row, features),
                "scores": None,
            }
        )
    return samples


def extract_error(score):
    empty = {
        "metric_name": None,
        "error_rate": None,
        "error_valid": False,
        "n_ref": None,
        "counts": None,
        "run_id": None,
        "scoring_config_id": None,
        "pipeline_id": None,
        "scoring_protocol": None,
    }
    if not isinstance(score, dict):
        return dict(empty)
    error_rate = as_float(score.get("error_rate"))
    metric_name = score.get("metric_name")
    counts = score.get("counts") if isinstance(score.get("counts"), dict) else None
    n_ref = counts.get("all") if counts and is_number(counts.get("all")) else None
    valid_flag = bool(score.get("wer_valid")) or bool(score.get("cer_valid"))
    error_valid = (
        error_rate is not None
        and error_rate >= 0
        and metric_name in ("wer", "cer")
        and valid_flag
        and n_ref is not None
        and n_ref > 0
    )
    return {
        "metric_name": metric_name if metric_name in ("wer", "cer") else None,
        "error_rate": error_rate if error_valid else None,
        "error_valid": error_valid,
        "n_ref": int(n_ref) if error_valid else None,
        "counts": counts if error_valid else None,
        "run_id": score.get("run_id"),
        "scoring_config_id": score.get("scoring_config_id"),
        "pipeline_id": score.get("pipeline_id"),
        "scoring_protocol": score.get("scoring_protocol"),
    }


def load_sidecar_by_sample_model(path):
    mapping = {}
    if not path.is_file():
        return mapping
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        model_id = row.get("model_id") or row.get("asr_system_id")
        if not sample_id or not model_id:
            continue
        key = (sample_id, model_id)
        if key in mapping:
            raise ValueError("duplicate sidecar key %s" % (key,))
        mapping[key] = row
    return mapping


def attach_error_diagnostics(samples, args):
    wer_cer = read_jsonl(args.wer_cer)
    if len(wer_cer) != len(samples):
        raise ValueError("sample_wer_cer.jsonl row count %s != %s" % (len(wer_cer), len(samples)))
    sidecar = load_sidecar_by_sample_model(args.sidecar)
    seen_pair = set()
    for index, sample in enumerate(samples):
        row = wer_cer[index]
        if row.get("sample_id") != sample["sample_id"] or row.get("global_index") != index:
            raise ValueError("sample_wer_cer alignment failed at %s" % index)
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(MODEL_IDS):
            raise ValueError("%s does not have %s scores" % (sample["sample_id"], len(MODEL_IDS)))
        score_by_model = {}
        for position, model_id in enumerate(MODEL_IDS):
            score = scores[position]
            if not isinstance(score, dict) or score.get("model_id") != model_id:
                raise ValueError("%s: score[%s] is not %s" % (sample["sample_id"], position, model_id))
            extracted = extract_error(score)
            sidecar_row = sidecar.get((sample["sample_id"], model_id), {})
            extracted["scoring_config_id"] = sidecar_row.get("scoring_config_id")
            extracted["pipeline_id"] = sidecar_row.get("pipeline_id")
            extracted["scoring_protocol"] = sidecar_row.get("scoring_protocol") or SCORING_PROTOCOL
            extracted["run_id"] = extracted.get("run_id") or sidecar_row.get("run_id")
            pair = (sample["sample_id"], model_id, extracted.get("scoring_config_id"))
            if pair in seen_pair and extracted.get("scoring_config_id"):
                raise ValueError("duplicate diagnostic key %s" % (pair,))
            seen_pair.add(pair)
            score_by_model[model_id] = extracted
        sample["scores"] = score_by_model
        sample["dataset"] = row.get("dataset") or sample["dataset"]
        sample["key"] = row.get("key") or sample["key"]
        sample["language"] = row.get("language") or sample["language"]
        sample["path"] = row.get("path") or sample["path"]


def fit_numeric_preprocess(samples, feature_specs=None):
    feature_specs = feature_specs or default_feature_specs()
    scales = {}
    for path, axis in feature_specs["numeric"]:
        values = [sample["features"]["numeric"][path] for sample in samples]
        center, spread, constant = robust_scale(values)
        scales[path] = {
            "axis": axis,
            "center": center,
            "spread": spread,
            "constant": constant,
            "missing_rate": sum(value is None for value in values) / float(max(len(values), 1)),
        }
    return scales


def fit_error_preprocess(samples, model_id):
    transformed = []
    metric_names = Counter()
    scoring_protocols = Counter()
    scoring_config_ids = set()
    for sample in samples:
        score = (sample.get("scores") or {}).get(model_id) or {}
        if not score.get("error_valid"):
            continue
        value = score.get("error_rate")
        if value is None:
            continue
        transformed.append(math.log1p(value))
        metric_names[score.get("metric_name")] += 1
        if score.get("scoring_protocol"):
            scoring_protocols[score["scoring_protocol"]] += 1
        if score.get("scoring_config_id"):
            scoring_config_ids.add(score["scoring_config_id"])
    center, spread, constant = robust_scale(transformed)
    if not transformed:
        raise ValueError("no valid WER/CER values for input model %s" % model_id)
    if constant or spread is None:
        raise ValueError("WER/CER input is constant for model %s" % model_id)
    return {
        "axis": ERROR_AXIS,
        "model_id": model_id,
        "transform": "log1p",
        "center": center,
        "spread": spread,
        "constant": constant,
        "n_valid": len(transformed),
        "missing_rate": 1.0 - len(transformed) / float(max(len(samples), 1)),
        "metric_name_counts": dict(metric_names),
        "scoring_protocol_counts": dict(scoring_protocols),
        "scoring_config_ids": sorted(scoring_config_ids),
    }


def z_numeric(value, spec):
    if value is None or spec["constant"] or spec["spread"] is None:
        return None
    return (value - spec["center"]) / spec["spread"]


def axis_distance(sample_i, sample_j, axis, scales, tau_num, feature_specs=None):
    feature_specs = feature_specs or default_feature_specs()
    parts = []
    missing_parts = []
    for path, feature_axis in feature_specs["numeric"]:
        if feature_axis != axis:
            continue
        spec = scales[path]
        zi = z_numeric(sample_i["features"]["numeric"][path], spec)
        zj = z_numeric(sample_j["features"]["numeric"][path], spec)
        mi = 1.0 if sample_i["features"]["numeric"][path] is None else 0.0
        mj = 1.0 if sample_j["features"]["numeric"][path] is None else 0.0
        missing_parts.append(0.0 if mi == mj else 1.0)
        if zi is not None and zj is not None:
            parts.append(min(abs(zi - zj) / tau_num, 1.0))
    for path, feature_axis in feature_specs["binary"]:
        if feature_axis != axis:
            continue
        vi = sample_i["features"]["binary"][path]
        vj = sample_j["features"]["binary"][path]
        mi = 1.0 if vi is None else 0.0
        mj = 1.0 if vj is None else 0.0
        missing_parts.append(0.0 if mi == mj else 1.0)
        if vi is not None and vj is not None:
            parts.append(0.0 if vi == vj else 1.0)
    for path, feature_axis in feature_specs["categorical"]:
        if feature_axis != axis:
            continue
        vi = sample_i["features"].get("categorical", {}).get(path)
        vj = sample_j["features"].get("categorical", {}).get(path)
        mi = 1.0 if vi is None else 0.0
        mj = 1.0 if vj is None else 0.0
        missing_parts.append(0.0 if mi == mj else 1.0)
        if vi is not None and vj is not None:
            parts.append(0.0 if vi == vj else 1.0)
    for path, feature_axis in feature_specs["sets"]:
        if feature_axis != axis:
            continue
        vi = sample_i["features"]["sets"].get(path)
        vj = sample_j["features"]["sets"].get(path)
        if vi is None or vj is None:
            missing_parts.append(1.0)
            continue
        missing_parts.append(0.0)
        union = vi | vj
        if not union:
            parts.append(0.0)
        else:
            parts.append(1.0 - (len(vi & vj) / float(len(union))))
    if not parts:
        return None
    observed = sum(parts) / float(len(parts))
    if missing_parts:
        missing = sum(missing_parts) / float(len(missing_parts))
        observed = (1.0 - MISSING_WEIGHT_CAP) * observed + MISSING_WEIGHT_CAP * missing
    return observed


def error_axis_distance(sample_i, sample_j, error_spec, tau_num):
    model_id = error_spec["model_id"]
    score_i = (sample_i.get("scores") or {}).get(model_id) or {}
    score_j = (sample_j.get("scores") or {}).get(model_id) or {}
    valid_i = bool(score_i.get("error_valid"))
    valid_j = bool(score_j.get("error_valid"))
    if not valid_i and not valid_j:
        return None
    if not valid_i or not valid_j:
        return MISSING_WEIGHT_CAP
    value_i = math.log1p(score_i["error_rate"])
    value_j = math.log1p(score_j["error_rate"])
    zi = (value_i - error_spec["center"]) / error_spec["spread"]
    zj = (value_j - error_spec["center"]) / error_spec["spread"]
    return min(abs(zi - zj) / tau_num, 1.0)


def tag_distance(sample_i, sample_j, scales, tau_num, error_spec=None, feature_specs=None):
    feature_specs = feature_specs or default_feature_specs()
    axis_values = []
    for axis in feature_specs["axes"]:
        if feature_specs.get("all_labels"):
            value = axis_distance(sample_i, sample_j, axis, scales, tau_num, feature_specs)
        else:
            # Keep the historical call shape for downstream users/tests that
            # monkey-patch axis_distance for the four default axes.
            value = axis_distance(sample_i, sample_j, axis, scales, tau_num)
        if value is not None:
            axis_values.append(value)
    if error_spec is not None:
        value = error_axis_distance(sample_i, sample_j, error_spec, tau_num)
        if value is not None:
            axis_values.append(value)
    if not axis_values:
        return None
    return sum(axis_values) / float(len(axis_values))


def build_tag_distance(samples, scales, tau_num, error_spec=None, feature_specs=None):
    n = len(samples)
    d_tag = [[None] * n for _ in range(n)]
    for i in range(n):
        d_tag[i][i] = 0.0
        for j in range(i + 1, n):
            value = tag_distance(
                samples[i], samples[j], scales, tau_num, error_spec, feature_specs
            )
            d_tag[i][j] = d_tag[j][i] = value
    return d_tag


def neighbor_counts(d_tag, min_neighbors=3):
    counts = []
    for i, row in enumerate(d_tag):
        counts.append(sum(1 for j, value in enumerate(row) if i != j and value is not None))
    review = [index for index, count in enumerate(counts) if count < min_neighbors]
    return counts, review


def submatrix(distance, indices):
    return [[distance[i][j] for j in indices] for i in indices]


def kmedoids(distance, k, seed, n_iter=25):
    rng = random.Random(seed)
    n = len(distance)
    finite_rows = [i for i in range(n) if any(distance[i][j] is not None for j in range(n))]
    if len(finite_rows) < k:
        return None, None
    medoids = [rng.choice(finite_rows)]
    while len(medoids) < k:
        weights = []
        candidates = []
        for index in finite_rows:
            if index in medoids:
                continue
            comparable = [distance[index][medoid] for medoid in medoids if distance[index][medoid] is not None]
            best = min(comparable) if comparable else 1.0
            candidates.append(index)
            weights.append(best ** 2)
        total = sum(weights)
        if total <= EPS:
            remain = [index for index in finite_rows if index not in medoids]
            if not remain:
                break
            medoids.append(rng.choice(remain))
            continue
        pick = rng.random() * total
        cumulative = 0.0
        chosen = candidates[-1]
        for index, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative >= pick:
                chosen = index
                break
        medoids.append(chosen)

    def assign(current):
        labels = [-1] * n
        for i in range(n):
            best_medoid = None
            best_distance = None
            for medoid in current:
                value = distance[i][medoid]
                if value is None:
                    continue
                if best_distance is None or value < best_distance:
                    best_distance = value
                    best_medoid = medoid
            if best_medoid is None:
                labels[i] = -1
            else:
                labels[i] = current.index(best_medoid)
        return labels

    labels = assign(medoids)
    for _ in range(n_iter):
        changed = False
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                groups[label].append(i)
        new_medoids = list(medoids)
        for label, members in groups.items():
            best = medoids[label]
            best_cost = None
            ordered = sorted(members, key=lambda index: (0, index))
            for candidate in ordered:
                cost = 0.0
                valid = 0
                for other in members:
                    value = distance[candidate][other]
                    if value is None:
                        continue
                    cost += value
                    valid += 1
                if valid == 0:
                    continue
                if best_cost is None or cost < best_cost - EPS or (
                    abs(cost - best_cost) <= EPS and candidate < best
                ):
                    best_cost = cost
                    best = candidate
            if best != new_medoids[label]:
                new_medoids[label] = best
                changed = True
        medoids = new_medoids
        new_labels = assign(medoids)
        if new_labels != labels:
            changed = True
            labels = new_labels
        if not changed:
            break
    return labels, medoids


def medoids_from_labels(labels, d_tag, k):
    medoids = []
    for label in range(k):
        members = [index for index, value in enumerate(labels) if value == label]
        if not members:
            return None
        best = members[0]
        best_cost = None
        for candidate in members:
            cost = 0.0
            valid = 0
            for other in members:
                value = d_tag[candidate][other]
                if value is None:
                    continue
                cost += value
                valid += 1
            if valid == 0:
                continue
            if best_cost is None or cost < best_cost - EPS or (
                abs((best_cost or 0) - cost) <= EPS and candidate < best
            ):
                best_cost = cost
                best = candidate
        medoids.append(best)
    return medoids


def cluster_sizes(labels, k):
    counts = Counter(label for label in labels if 0 <= label < k)
    return [counts.get(index, 0) for index in range(k)]


def c_tag_metric(labels, medoids, d_tag, k):
    values = []
    weights = []
    for label in range(k):
        members = [index for index, value in enumerate(labels) if value == label]
        if not members:
            continue
        medoid = medoids[label]
        dists = [d_tag[index][medoid] for index in members if d_tag[index][medoid] is not None]
        if not dists:
            continue
        values.append(median(dists))
        weights.append(len(members))
    if not values:
        return None
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / float(total)


def source_shares(samples, labels, k):
    result = []
    for label in range(k):
        members = [samples[index] for index, value in enumerate(labels) if value == label]
        if not members:
            result.append({"n": 0, "max_share": None, "n_sources": 0, "top_source": None})
            continue
        counts = Counter(sample["dataset"] for sample in members)
        top_source, top_n = counts.most_common(1)[0]
        result.append(
            {
                "n": len(members),
                "max_share": top_n / float(len(members)),
                "n_sources": len(counts),
                "top_source": top_source,
                "source_counts": dict(counts),
            }
        )
    return result


def nearest_medoid_accuracy(labels, d_tag, medoids, holdout_indices):
    holdout = set(holdout_indices)
    correct = 0
    total = 0
    per_cluster = defaultdict(lambda: [0, 0])
    for index in holdout_indices:
        true = labels[index]
        if true < 0:
            continue
        best = None
        best_d = None
        for label, medoid in enumerate(medoids):
            if medoid in holdout:
                continue
            value = d_tag[index][medoid]
            if value is None:
                continue
            if best_d is None or value < best_d:
                best_d = value
                best = label
        if best is None:
            continue
        total += 1
        per_cluster[true][1] += 1
        if best == true:
            correct += 1
            per_cluster[true][0] += 1
    if total == 0:
        return None, {}
    recalls = {}
    for label, pair in per_cluster.items():
        hit, count = pair
        recalls[str(label)] = None if count == 0 else hit / float(count)
    return correct / float(total), recalls


def q_tag_leave_one_source(samples, labels, d_tag, medoids, k):
    by_source = defaultdict(list)
    for index, sample in enumerate(samples):
        if 0 <= labels[index] < k:
            by_source[sample["dataset"]].append(index)
    raw_scores = []
    recall_accum = defaultdict(list)
    used_sources = 0
    for source, indices in by_source.items():
        holdout = set(indices)
        remaining_labels = {
            labels[index]
            for index in range(len(samples))
            if labels[index] >= 0 and index not in holdout
        }
        if len(remaining_labels) < 2:
            continue
        acc, recalls = nearest_medoid_accuracy(labels, d_tag, medoids, indices)
        if acc is None:
            continue
        used_sources += 1
        raw_scores.append(acc)
        for label, value in recalls.items():
            if value is not None:
                recall_accum[label].append(value)
    if not raw_scores:
        return None, {}, used_sources
    q_raw = sum(raw_scores) / float(len(raw_scores))
    q_chance = 1.0 / float(k)
    q_tag = clip01((q_raw - q_chance) / (1.0 - q_chance)) if q_chance < 1 else None
    recalls = {
        label: sum(values) / float(len(values))
        for label, values in recall_accum.items()
        if values
    }
    return q_tag, recalls, used_sources


def feature_signature(sample):
    feats = sample["features"]
    events = tuple(sorted(feats["sets"].get("sound_field_scene.speech_music_events") or ()))
    noise = tuple(sorted(feats["sets"].get("sound_field_scene.external_noise_type") or ()))
    return (
        events,
        noise,
        feats["binary"].get("sound_field_scene.music_present"),
        feats["binary"].get("speaker.speaker_overlap"),
        feats["numeric"].get("speaker.speaker_count"),
    )


def numeric_effect_anchor(members, samples, path, min_n=5, min_effect=0.45):
    # All-label discovery can omit default anchor fields, including fields
    # that are always null and therefore have no discovered numeric type.
    local = [
        sample["features"]["numeric"].get(path)
        for sample in members
        if sample["features"]["numeric"].get(path) is not None
    ]
    glob = [
        sample["features"]["numeric"].get(path)
        for sample in samples
        if sample["features"]["numeric"].get(path) is not None
    ]
    if len(local) < min_n or len(glob) < min_n:
        return None
    local_med = median(local)
    glob_med = median(glob)
    scale = iqr(glob) or ((mad(glob, glob_med) or 0.0) * 1.4826)
    if scale is None or scale <= EPS:
        return None
    effect = (local_med - glob_med) / scale
    if abs(effect) < min_effect:
        return None
    return {
        "feature": path,
        "value": round(local_med, 4),
        "lift": abs(effect),
        "n": len(local),
        "effect": effect,
        "direction": "higher" if effect > 0 else "lower",
    }


def tag_anchors(samples, labels, k, lift_min):
    global_counts = Counter()
    for sample in samples:
        sig = feature_signature(sample)
        global_counts[("events", sig[0])] += 1
        global_counts[("noise", sig[1])] += 1
        global_counts[("music", sig[2])] += 1
        global_counts[("overlap", sig[3])] += 1
        count_bin = None if sig[4] is None else ("0" if sig[4] <= 0 else "1" if sig[4] == 1 else "2+")
        global_counts[("speaker_bin", count_bin)] += 1
    n = len(samples)
    anchors = {}
    unanchored = 0
    numeric_paths = (
        "audio_quality.snr_db",
        "room_acoustic.rt60_sec",
        "basic_acoustic.duration_sec",
        "basic_acoustic.nonspeech_ratio",
        "speaker.overlap_ratio",
        "speaker.speaker_count",
    )
    for label in range(k):
        members = [samples[index] for index, value in enumerate(labels) if value == label]
        if not members:
            anchors[label] = []
            continue
        local = Counter()
        for sample in members:
            sig = feature_signature(sample)
            local[("events", sig[0])] += 1
            local[("noise", sig[1])] += 1
            local[("music", sig[2])] += 1
            local[("overlap", sig[3])] += 1
            count_bin = None if sig[4] is None else ("0" if sig[4] <= 0 else "1" if sig[4] == 1 else "2+")
            local[("speaker_bin", count_bin)] += 1
        found = []
        for key, count in local.items():
            if key[1] in (None, (), False) and count == 0:
                continue
            if key[0] in {"music", "overlap"} and key[1] is False:
                continue
            if key[0] == "noise" and key[1] == ():
                continue
            global_n = global_counts[key]
            if global_n <= 0:
                continue
            lift = (count / float(len(members))) / (global_n / float(n))
            distinctive = key[0] in {"music", "overlap", "noise", "events"} and key[1] not in (
                None,
                (),
                False,
                ("speech",),
            )
            speaker_multi = key == ("speaker_bin", "2+")
            if (lift >= lift_min or distinctive or speaker_multi) and count >= max(3, int(0.15 * len(members))):
                if lift >= 1.15 or distinctive or speaker_multi:
                    found.append({"feature": key[0], "value": key[1], "lift": lift, "n": count})
        for path in numeric_paths:
            item = numeric_effect_anchor(members, samples, path)
            if item is not None:
                found.append(item)
        found.sort(key=lambda item: (-item["lift"], -item["n"]))
        anchors[label] = found[:8]
        if not found:
            unanchored += len(members)
    return anchors, unanchored / float(max(sum(1 for label in labels if label >= 0), 1))


def cut_tag(labels, d_tag, k_graph):
    n = len(d_tag)
    neighbors = []
    for i in range(n):
        ranked = sorted(
            ((d_tag[i][j], j) for j in range(n) if i != j and d_tag[i][j] is not None),
        )
        neighbors.append([index for _dist, index in ranked[:k_graph]])
    edges = set()
    for i in range(n):
        for j in neighbors[i]:
            if i in neighbors[j]:
                edges.add(tuple(sorted((i, j))))
    if not edges:
        return None
    distances = [d_tag[i][j] for i, j in edges]
    sigma = median(distances) or 1.0
    if sigma <= EPS:
        return None
    num = 0.0
    den = 0.0
    for i, j in edges:
        weight = math.exp(-((d_tag[i][j] / sigma) ** 2))
        den += weight
        if labels[i] != labels[j]:
            num += weight
    if den <= EPS:
        return None
    return num / den


def profile_key(anchors):
    items = []
    for item in anchors or []:
        items.append("%s=%s" % (item.get("feature"), item.get("value")))
    return tuple(items[:4])


def evaluate_candidate(samples, labels, medoids, d_tag, args):
    k = len(medoids)
    n = len(samples)
    main = [index for index, label in enumerate(labels) if 0 <= label < k]
    n_main = len(main)
    n_non_main = n - n_main
    sizes = cluster_sizes(labels, k)
    b_small = sum(max(0, args.n_min - size) for size in sizes) / float(max(n_main, 1))
    b_non_main = n_non_main / float(n)
    c_tag = c_tag_metric(labels, medoids, d_tag, k)
    shares = source_shares(samples, labels, k)
    source_driven = any(
        (item["max_share"] or 0) >= args.source_max_share
        for item in shares
        if item["n"] >= args.n_min
    )
    q_tag, recalls, n_q_sources = q_tag_leave_one_source(samples, labels, d_tag, medoids, k)
    anchors, a_unanchored = tag_anchors(samples, labels, k, args.lift_min)
    k_graph = min(max(n_main - 1, 1), 20, max(5, int(round(math.sqrt(max(n_main, 1))))))
    cut = cut_tag(labels, d_tag, k_graph) if n_main > 5 else None
    b_anchor = a_unanchored + max(0.0, args.q_min - (q_tag or 0.0))
    hard_gates = {
        "min_k": k >= args.k_min,
        "c_tag": c_tag is not None,
    }
    diagnostic_flags = {
        "min_size": all(size >= args.n_min for size in sizes if size),
        "non_main": b_non_main <= args.b_non_main_max,
        "source_driven": not source_driven,
        "has_anchor": all(bool(anchors[label]) for label in range(k) if sizes[label] >= args.n_min),
        "q_tag": q_tag is not None and q_tag >= args.q_min,
        "cut_tag": cut is None or cut <= args.cut_tag_max,
    }
    passed = all(hard_gates.values())
    j_value = None
    if c_tag is not None:
        j_value = (
            args.lambda_t * c_tag
            + args.lambda_a * b_anchor
            + args.lambda_k * math.log(1.0 + k) / math.log(1.0 + max(args.k_max, 1))
            + args.lambda_b * b_small
            + args.lambda_r * b_non_main
        )
    return {
        "k": k,
        "sizes": sizes,
        "n_main": n_main,
        "n_non_main": n_non_main,
        "b_small": b_small,
        "b_non_main": b_non_main,
        "c_tag": c_tag,
        "q_tag": q_tag,
        "q_tag_sources": n_q_sources,
        "recalls": recalls,
        "a_unanchored": a_unanchored,
        "b_anchor": b_anchor,
        "cut_tag": cut,
        "source_shares": shares,
        "source_driven": source_driven,
        "anchors": {str(label): value for label, value in anchors.items()},
        "gates": hard_gates,
        "diagnostic_flags": diagnostic_flags,
        "passed": passed,
        "j": j_value,
    }


def expand_labels(active, local_labels, n):
    labels = [-1] * n
    medoids_local = []
    k = max(local_labels) + 1 if local_labels else 0
    groups = defaultdict(list)
    for local_index, label in enumerate(local_labels):
        if label >= 0:
            groups[label].append(active[local_index])
            labels[active[local_index]] = label
    return labels, k, groups


def candidate_from_labels(samples, labels, d_tag, args, method, extra):
    k = max(labels) + 1 if any(label >= 0 for label in labels) else 0
    if k < 2:
        return None
    medoids = medoids_from_labels(labels, d_tag, k)
    if medoids is None:
        return None
    metrics = evaluate_candidate(samples, labels, medoids, d_tag, args)
    item = {
        "method": method,
        "k": k,
        "labels": labels,
        "medoids": medoids,
        "metrics": metrics,
    }
    item.update(extra)
    return item


def search_candidates(samples, d_tag, active, args):
    methods = getattr(args, "cluster_method_list", None) or ["kmedoids"]
    distance_name = "d_tags_plus_wer_cer" if args.use_wer_input else "d_tag"
    print("searching %s on %s" % (",".join(methods), distance_name))
    candidates = []
    seen = set()
    for raw in cluster_methods.generate_method_partitions(d_tag, active, args, methods):
        key = (raw["method"], tuple(raw["labels"]), raw.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        item = candidate_from_labels(
            samples,
            raw["labels"],
            d_tag,
            args,
            raw["method"],
            {"seed": raw.get("seed"), "params": raw.get("params") or {}},
        )
        if item is None:
            continue
        candidates.append(item)
        status = "PASS" if item["metrics"]["passed"] else "fail"
        print(
            "  %s K=%s J=%s C_tag=%s Q_tag=%s %s"
            % (
                item["method"],
                item["k"],
                item["metrics"]["j"],
                item["metrics"]["c_tag"],
                item["metrics"]["q_tag"],
                status,
            )
        )
    return candidates


def select_candidate(candidates, wer_filter=False, min_snk=0.10):
    """Select a tags-only candidate. WER/CER may filter, never re-rank the distance."""
    feasible = [item for item in candidates if item["metrics"]["passed"] and item["metrics"]["j"] is not None]
    pool = feasible or [item for item in candidates if item["metrics"]["j"] is not None]
    if not pool:
        return None, False
    not_source = [item for item in pool if not item["metrics"].get("source_driven")]
    if not_source:
        pool = not_source
    if wer_filter:
        scored = [
            item
            for item in pool
            if ((item.get("posthoc") or {}).get("mean_snk") is not None)
        ]
        if scored:
            snks = sorted(item["posthoc"]["mean_snk"] for item in scored)
            threshold = max(min_snk, snks[int(0.3 * (len(snks) - 1))])
            filtered = [item for item in scored if item["posthoc"]["mean_snk"] >= threshold]
            if filtered:
                pool = filtered
    best_j = min(item["metrics"]["j"] for item in pool)
    band = [item for item in pool if item["metrics"]["j"] <= best_j * 1.05 + 0.01]

    def sort_key(item):
        snk = (item.get("posthoc") or {}).get("mean_snk")
        return (
            item["metrics"]["j"],
            item["metrics"]["k"],
            -(item["metrics"]["q_tag"] or 0.0),
            -(snk if snk is not None else -1.0),
            item["method"],
        )

    band.sort(key=sort_key)
    return band[0], bool(feasible)


def distinct_profiles(anchors_a, anchors_b):
    return profile_key(anchors_a) != profile_key(anchors_b)


def refine_split_merge(samples, labels, medoids, d_tag, args):
    history = []
    current_labels = list(labels)
    current_medoids = list(medoids)
    for _step in range(args.max_refine):
        metrics = evaluate_candidate(samples, current_labels, current_medoids, d_tag, args)
        k = len(current_medoids)
        accepted = None
        for label in range(k):
            members = [index for index, value in enumerate(current_labels) if value == label]
            if len(members) < 2 * args.n_min:
                continue
            local = kmedoids(submatrix(d_tag, members), 2, seed=0)
            if local[0] is None:
                continue
            local_labels, local_medoids = local
            groups = defaultdict(list)
            for local_index, local_label in enumerate(local_labels):
                if local_label >= 0:
                    groups[local_label].append(members[local_index])
            if len(groups) != 2:
                continue
            if any(len(group) < args.n_min for group in groups.values()):
                continue
            trial = list(current_labels)
            new_label = k
            for index in groups[1]:
                trial[index] = new_label
            trial_medoids = list(current_medoids)
            trial_medoids[label] = members[local_medoids[0]]
            trial_medoids.append(members[local_medoids[1]])
            trial_metrics = evaluate_candidate(samples, trial, trial_medoids, d_tag, args)
            if trial_metrics["j"] is None or metrics["j"] is None:
                continue
            if trial_metrics["j"] >= metrics["j"] - args.epsilon_refine:
                continue
            if not distinct_profiles(
                trial_metrics["anchors"].get(str(label), []),
                trial_metrics["anchors"].get(str(new_label), []),
            ):
                continue
            accepted = ("split", label, trial, trial_medoids, trial_metrics)
            break
        if accepted is None:
            for a in range(k):
                for b in range(a + 1, k):
                    size_a = metrics["sizes"][a]
                    size_b = metrics["sizes"][b]
                    if min(size_a, size_b) >= args.n_min and distinct_profiles(
                        metrics["anchors"].get(str(a), []),
                        metrics["anchors"].get(str(b), []),
                    ):
                        continue
                    trial = [b if value == a else value for value in current_labels]
                    remap = {}
                    next_id = 0
                    remapped = []
                    for value in trial:
                        if value < 0:
                            remapped.append(-1)
                            continue
                        if value not in remap:
                            remap[value] = next_id
                            next_id += 1
                        remapped.append(remap[value])
                    trial_medoids = medoids_from_labels(remapped, d_tag, next_id)
                    if trial_medoids is None:
                        continue
                    trial_metrics = evaluate_candidate(samples, remapped, trial_medoids, d_tag, args)
                    if trial_metrics["j"] is None or metrics["j"] is None:
                        continue
                    if trial_metrics["j"] > metrics["j"] + args.epsilon_refine:
                        continue
                    if trial_metrics["k"] < args.k_min:
                        continue
                    accepted = ("merge", (a, b), remapped, trial_medoids, trial_metrics)
                    break
                if accepted is not None:
                    break
        if accepted is None:
            break
        op, target, current_labels, current_medoids, new_metrics = accepted
        history.append({"op": op, "target": target, "k": new_metrics["k"], "j": new_metrics["j"]})
        print("  refine %s -> K=%s J=%s" % (op, new_metrics["k"], new_metrics["j"]))
    return current_labels, current_medoids, history


def canonicalize(samples, labels, d_tag):
    clusters = defaultdict(list)
    for index, label in enumerate(labels):
        if label >= 0:
            clusters[label].append(index)
    records = []
    for old_label, members in clusters.items():
        medoids = medoids_from_labels(
            [0 if index in set(members) else -1 for index in range(len(labels))],
            d_tag,
            1,
        )
        medoid = medoids[0] if medoids else members[0]
        profile = sha256_text(
            json.dumps(
                [
                    [str(part) for part in feature_signature(samples[index])]
                    for index in sorted(members, key=lambda item: samples[item]["sample_id"])
                ],
                ensure_ascii=False,
            )
        )
        records.append((samples[medoid]["sample_id"], profile, old_label, medoid, members))
    records.sort()
    new_labels = [-1] * len(labels)
    new_medoids = []
    mapping = {}
    for new_label, (_sid, _profile, old_label, medoid, members) in enumerate(records):
        mapping[old_label] = "C%02d" % new_label
        new_medoids.append(medoid)
        for index in members:
            new_labels[index] = new_label
    return new_labels, new_medoids, mapping


def secondary_tags(sample):
    events = sample["features"]["sets"].get("sound_field_scene.speech_music_events") or frozenset()
    content = []
    for item in ("speech", "singing", "music"):
        if item in events:
            content.append(item)
    if not content and not events:
        content.append("non_speech_candidate")
    speaker_count = sample["features"]["numeric"].get("speaker.speaker_count")
    overlap = sample["features"]["binary"].get("speaker.speaker_overlap")
    if speaker_count is None:
        interaction = "unknown"
    elif overlap:
        interaction = "overlap"
    elif speaker_count >= 2:
        interaction = "multi"
    else:
        interaction = "single"
    noise = sorted(sample["features"]["sets"].get("sound_field_scene.external_noise_type") or [])
    snr = sample["features"]["numeric"].get("audio_quality.snr_db")
    if snr is None:
        quality = "unknown"
    elif snr > 20:
        quality = "high"
    elif snr >= 10:
        quality = "medium"
    else:
        quality = "low"
    rt60 = sample["features"]["numeric"].get("room_acoustic.rt60_sec")
    if rt60 is None:
        room = "unknown"
    elif rt60 < 0.3:
        room = "dry"
    elif rt60 < 0.6:
        room = "ordinary"
    else:
        room = "reverberant"
    duration = sample["features"]["numeric"].get("basic_acoustic.duration_sec")
    nonspeech = sample["features"]["numeric"].get("basic_acoustic.nonspeech_ratio")
    temporal = []
    if duration is not None and duration < 3:
        temporal.append("short_fragment")
    if duration is not None and duration >= 30:
        temporal.append("long_form")
    if nonspeech is not None and nonspeech >= 0.5:
        temporal.append("sparse_speech")
    return {
        "content": content,
        "interaction": interaction,
        "language_group": language_group(sample["language"]),
        "noise": noise,
        "quality": quality,
        "room": room,
        "temporal": temporal,
    }


def majority_secondary(members):
    bag = defaultdict(Counter)
    for sample in members:
        tags = secondary_tags(sample)
        for key in ("interaction", "quality", "room"):
            bag[key][tags[key]] += 1
        for item in tags["content"]:
            bag["content"][item] += 1
        for item in tags["noise"]:
            bag["noise"][item] += 1
        for item in tags["temporal"]:
            bag["temporal"][item] += 1
        bag["language_group"][tags["language_group"]] += 1
    result = {}
    n = max(len(members), 1)
    thresh = max(3, int(0.15 * n))
    for key in ("interaction", "quality", "room", "language_group"):
        result[key] = bag[key].most_common(1)[0][0] if bag[key] else "unknown"
    for key in ("content", "noise", "temporal"):
        result[key] = [item for item, count in bag[key].most_common() if item and count >= thresh]
    return result


def name_from_profile(anchor_list, secondary, heterogeneous):
    texts = []
    interaction = secondary.get("interaction")
    content = secondary.get("content") or []
    noise = secondary.get("noise") or []
    quality = secondary.get("quality")
    room = secondary.get("room")
    temporal = secondary.get("temporal") or []
    if interaction == "overlap":
        texts.append("多人交互 / 重叠发言")
    elif interaction == "multi":
        texts.append("多人语音")
    elif interaction == "single":
        texts.append("单人语音")
    if "singing" in content:
        texts.append("含歌唱")
    elif "music" in content:
        texts.append("含音乐")
    if "mechanical" in noise:
        texts.append("机械背景")
    elif noise:
        texts.append("有外部噪声")
    if quality == "low":
        texts.append("低清晰度")
    if room == "reverberant":
        texts.append("较高混响")
    if "short_fragment" in temporal:
        texts.append("短片段")
    if "sparse_speech" in temporal:
        texts.append("稀疏语音")
    if not texts:
        if anchor_list:
            texts.append("可观测标签组合")
        else:
            texts.append("标签画像不足")
    name = "，".join(texts[:3])
    if heterogeneous:
        name += "（误差异质，待复核）"
    return name


def evidence_tags_for(sample):
    paths = []
    events = nested_get(sample["tags"], "sound_field_scene.speech_music_events")
    if events:
        paths.append("sound_field_scene.speech_music_events")
    noise = nested_get(sample["tags"], "sound_field_scene.external_noise_type")
    if noise:
        paths.append("sound_field_scene.external_noise_type")
    if nested_get(sample["tags"], "speaker.speaker_count") is not None:
        paths.append("speaker.speaker_count")
    if nested_get(sample["tags"], "audio_quality.snr_db") is not None:
        paths.append("audio_quality.snr_db")
    if nested_get(sample["tags"], "room_acoustic.rt60_sec") is not None:
        paths.append("room_acoustic.rt60_sec")
    return paths


def assignment_confidence(sample, metrics, label):
    if label < 0:
        return "low"
    if sample["conflicts"]:
        return "low"
    share = metrics["source_shares"][label]["max_share"] or 0
    if share >= 0.9:
        return "low"
    if not metrics["anchors"].get(str(label)):
        return "low"
    if metrics["q_tag"] is not None and metrics["q_tag"] >= 0.3:
        return "high"
    return "medium"


def percentile_spread(values, low=0.10, high=0.90):
    if len(values) < 2:
        return None
    hi = quantile(values, high)
    lo = quantile(values, low)
    if hi is None or lo is None:
        return None
    return hi - lo


def d_err_metric(labels, k, u_values, valid_mask, n_error_min):
    groups = defaultdict(list)
    valid_indices = [index for index, flag in enumerate(valid_mask) if flag]
    if not valid_indices:
        return None, 1.0, {}
    for index in valid_indices:
        label = labels[index]
        if 0 <= label < k:
            groups[label].append(u_values[index])
    cal_values = [u_values[index] for index in valid_indices]
    _center, scale, constant = robust_scale(cal_values)
    if constant or scale is None:
        return None, 1.0, {"metric_constant": True}
    eligible = {}
    for label, values in groups.items():
        if len(values) >= n_error_min:
            spread = percentile_spread(values)
            if spread is None:
                continue
            eligible[label] = (len(values), spread)
    if not eligible:
        return None, 1.0, {"insufficient_error_coverage": True}
    total = sum(item[0] for item in eligible.values())
    d_err = sum((count / float(total)) * (spread / scale) for count, spread in eligible.values())
    coverage = total / float(len(valid_indices))
    per_cluster = {}
    for label, pair in eligible.items():
        count, spread = pair
        per_cluster[str(label)] = {
            "n": count,
            "spread": spread,
            "median": median(groups[label]),
            "p05": quantile(groups[label], 0.05),
            "p95": quantile(groups[label], 0.95),
        }
    return d_err, 1.0 - coverage, per_cluster


def error_homogeneity_metrics(labels, raw_values, valid_mask, group_ids=None):
    """ANOVA eta^2 and MAD compactness on raw error_rate. Diagnostic only."""
    items = []
    for index, flag in enumerate(valid_mask):
        if not flag or raw_values[index] is None or labels[index] < 0:
            continue
        items.append(
            {
                "cluster_id": "C%02d" % labels[index],
                "error_rate": float(raw_values[index]),
                "group_id": None if group_ids is None else group_ids[index],
            }
        )
    n = len(items)
    if n < 2:
        return {"n_valid": n, "eta2": None, "used_for_assignment": False}
    values = [item["error_rate"] for item in items]
    mu = sum(values) / float(n)
    ss_total = sum((value - mu) ** 2 for value in values)
    groups = defaultdict(list)
    for item in items:
        groups[item["cluster_id"]].append(item["error_rate"])
    ss_within = 0.0
    ss_between = 0.0
    mad_global = mad(values)
    weighted_mad = 0.0
    per_cluster = {}
    for cluster_id, cluster_values in groups.items():
        cluster_mean = sum(cluster_values) / float(len(cluster_values))
        ss_within += sum((value - cluster_mean) ** 2 for value in cluster_values)
        ss_between += len(cluster_values) * ((cluster_mean - mu) ** 2)
        cluster_mad = mad(cluster_values)
        weighted_mad += (len(cluster_values) / float(n)) * (cluster_mad or 0.0)
        per_cluster[cluster_id] = {
            "n_valid": len(cluster_values),
            "median": median(cluster_values),
            "mad": cluster_mad,
            "iqr": iqr(cluster_values),
            "mad_ratio_vs_global": None if not mad_global else cluster_mad / mad_global,
        }
    eta2 = 0.0 if ss_total <= EPS else ss_between / ss_total
    sigma_total = math.sqrt(ss_total / float(n))
    sigma_within = math.sqrt(ss_within / float(n))
    source_eta2 = None
    if group_ids is not None:
        source_groups = defaultdict(list)
        for item in items:
            if item["group_id"] is None:
                continue
            source_groups[item["group_id"]].append(item["error_rate"])
        if len(source_groups) >= 2 and ss_total > EPS:
            source_within = 0.0
            for source_values in source_groups.values():
                source_mean = sum(source_values) / float(len(source_values))
                source_within += sum((value - source_mean) ** 2 for value in source_values)
            source_eta2 = 1.0 - source_within / ss_total
    return {
        "metric": "eta2_error_rate",
        "n_valid": n,
        "eta2": eta2,
        "ss_total": ss_total,
        "ss_within": ss_within,
        "ss_between": ss_between,
        "sigma_total": sigma_total,
        "sigma_within": sigma_within,
        "sigma_ratio": None if sigma_total <= EPS else sigma_within / sigma_total,
        "h_mad": None if not mad_global else 1.0 - weighted_mad / mad_global,
        "mad_global": mad_global,
        "mad_within_weighted": weighted_mad,
        "eta2_by_source": source_eta2,
        "used_for_assignment": False,
        "per_cluster": per_cluster,
    }


def mutual_info_proxy(labels, values, k, n_bins=6):
    paired = [
        (labels[index], values[index])
        for index in range(len(labels))
        if labels[index] >= 0 and values[index] is not None
    ]
    if len(paired) < max(20, 2 * k):
        return None
    finite = [value for _label, value in paired]
    edges = [quantile(finite, q / float(n_bins)) for q in range(n_bins + 1)]

    def bucket(value):
        for index in range(n_bins):
            if value <= edges[index + 1]:
                return index
        return n_bins - 1

    joint = Counter()
    label_counts = Counter()
    value_counts = Counter()
    for label, value in paired:
        b = bucket(value)
        joint[(label, b)] += 1
        label_counts[label] += 1
        value_counts[b] += 1
    total = float(len(paired))
    h_z = 0.0
    for count in label_counts.values():
        p = count / total
        h_z -= p * math.log(p + EPS)
    h_v = 0.0
    for count in value_counts.values():
        p = count / total
        h_v -= p * math.log(p + EPS)
    h_joint = 0.0
    for count in joint.values():
        p = count / total
        h_joint -= p * math.log(p + EPS)
    if h_z <= EPS:
        return None
    return max(0.0, (h_z + h_v - h_joint) / h_z)


def diagnose_model(samples, labels, k, model_id, n_error_min):
    u_values = []
    valid_mask = []
    raw = []
    metric_names = Counter()
    for sample in samples:
        score = (sample.get("scores") or {}).get(model_id) or {}
        if score.get("error_valid"):
            u_values.append(math.log1p(score["error_rate"]))
            valid_mask.append(True)
            raw.append(score["error_rate"])
            metric_names[score.get("metric_name")] += 1
        else:
            u_values.append(None)
            valid_mask.append(False)
            raw.append(None)
    d_err, b_cov, per_cluster = d_err_metric(labels, k, u_values, valid_mask, n_error_min)
    one_labels = [0 if label >= 0 else -1 for label in labels]
    d_one, _cov, _detail = d_err_metric(one_labels, 1, u_values, valid_mask, n_error_min)
    gain = None
    if d_err is not None and d_one not in (None, 0):
        gain = 1.0 - d_err / d_one
    z_err = []
    center, spread, constant = robust_scale([value for value in u_values if value is not None])
    for value, flag in zip(u_values, valid_mask):
        if not flag or constant or spread is None:
            z_err.append(None)
        else:
            z_err.append((value - center) / spread)
    nmi = mutual_info_proxy(labels, z_err, k)
    homogeneity = error_homogeneity_metrics(
        labels,
        raw,
        valid_mask,
        [sample.get("dataset") for sample in samples],
    )
    scoring_ids = sorted(
        {
            ((sample.get("scores") or {}).get(model_id) or {}).get("scoring_config_id")
            for sample in samples
            if ((sample.get("scores") or {}).get(model_id) or {}).get("scoring_config_id")
        }
    )
    scoring_protocols = sorted(
        {
            ((sample.get("scores") or {}).get(model_id) or {}).get("scoring_protocol")
            for sample in samples
            if ((sample.get("scores") or {}).get(model_id) or {}).get("scoring_protocol")
        }
    )
    return {
        "dispersion_evaluation_role": "posthoc_eval_dispersion",
        "eval_asr_system_id": model_id,
        "eval_scoring_protocol": scoring_protocols[0] if len(scoring_protocols) == 1 else None,
        "eval_scoring_protocols": scoring_protocols,
        "eval_scoring_config_ids": scoring_ids,
        "used_for_assignment": False,
        "n_valid": sum(valid_mask),
        "metric_name_counts": dict(metric_names),
        "d_err": d_err,
        "b_error_coverage": b_cov,
        "d_err_one_cluster": d_one,
        "gain": gain,
        "error_aligned_nmi": nmi,
        "error_homogeneity": homogeneity,
        "per_cluster": per_cluster,
        "raw_by_index": raw,
        "valid_mask": valid_mask,
        "scores": [(sample.get("scores") or {}).get(model_id) for sample in samples],
    }


def empty_score():
    return {
        "metric_name": None,
        "error_rate": None,
        "error_valid": False,
        "n_ref": None,
        "counts": None,
        "run_id": None,
        "scoring_config_id": None,
        "pipeline_id": None,
        "scoring_protocol": SCORING_PROTOCOL,
    }


def diagnostic_payload(score, missing=True, used_as_input=False):
    score = score or empty_score()
    return {
        "asr_system_id": None,
        "model_id": None,
        "run_id": score.get("run_id"),
        "scoring_config_id": score.get("scoring_config_id"),
        "scoring_protocol": score.get("scoring_protocol") or SCORING_PROTOCOL,
        "pipeline_id": score.get("pipeline_id"),
        "metric_name": score.get("metric_name"),
        "error_rate": score.get("error_rate"),
        "error_valid": bool(score.get("error_valid")),
        "n_ref": score.get("n_ref"),
        "counts": score.get("counts"),
        "used_as_input": used_as_input,
        "error_view_missing": missing or not bool(score.get("error_valid")),
        "assignment_role": "clustering_input" if used_as_input else "posthoc_diagnostic",
    }


def rules_version_for(args):
    all_labels = bool(getattr(args, "use_all_sure_tagger_labels", False))
    if all_labels and args.use_wer_input:
        return ALL_LABELS_WER_INPUT_RULES_VERSION
    if all_labels:
        return ALL_LABELS_RULES_VERSION
    if args.use_wer_input:
        return WER_INPUT_RULES_VERSION
    return CLUSTER_RULES_VERSION


def build_outputs(samples, labels, medoids, metrics, args, hashes, publishable, refine_history, diagnostics, primary_model):
    n = len(samples)
    k = len(medoids)
    use_wer_input = bool(args.use_wer_input)
    mode = "model_specific" if use_wer_input else MODE
    partition_scope = "model_specific" if use_wer_input else "shared"
    all_labels = bool(getattr(args, "use_all_sure_tagger_labels", False))
    assignment_basis = (
        ALL_LABELS_WER_ASSIGNMENT_BASIS
        if use_wer_input and all_labels
        else (
            "audio_tags_plus_wer_cer"
            if use_wer_input
            else (ALL_LABELS_ASSIGNMENT_BASIS if all_labels else ASSIGNMENT_BASIS)
        )
    )
    rules_version = rules_version_for(args)
    feature_specs = getattr(args, "feature_specs", default_feature_specs())
    used_features = [
        "%s:%s" % (kind, path)
        for kind in ("numeric", "binary", "categorical", "sets")
        for path, _axis in feature_specs[kind]
    ]
    if not all_labels:
        used_features = [
            "basic_acoustic.duration_sec",
            "basic_acoustic.nonspeech_ratio",
            "basic_acoustic.silence_segment_count",
            "basic_acoustic.silence_total_sec",
            "audio_quality.*",
            "room_acoustic.rt60_sec",
            "room_acoustic.c50_db",
            "sound_field_scene.speech_music_events",
            "sound_field_scene.external_noise_type",
            "sound_field_scene.noise_composition",
            "speaker.speaker_count",
            "speaker.speaker_overlap",
            "speaker.overlap_ratio",
            "speaker.speaker_change_count",
            "deterministic_derived.speaker_change_rate_per_min",
        ]
    excluded_features = [
        "reference_derived_text_except_metrics",
        "target_asr_transcript",
        "target_model_derived_tags",
        "room_acoustic.far_field",
        "language_content.*" if not all_labels else "",
    ]
    excluded_features = [item for item in excluded_features if item]
    if all_labels:
        excluded_features = [
            "wer_cer.error_rate",
            "reference_derived_metadata",
            "sample_manifest_metadata",
        ]
    if use_wer_input:
        used_features.append("wer_cer.error_rate[%s]" % args.wer_input_model)
    else:
        excluded_features.insert(0, "wer_cer.error_rate")
    summary_clusters = []
    cluster_rows = []
    primary = diagnostics.get(primary_model) if diagnostics else None
    heterogeneous_flags = [False] * k
    if primary:
        for label in range(k):
            detail = (primary.get("per_cluster") or {}).get(str(label), {})
            p05 = detail.get("p05")
            p95 = detail.get("p95")
            if p05 is not None and p95 is not None and (p95 - p05) >= 0.5:
                heterogeneous_flags[label] = True
    for label in range(k):
        members = [index for index, value in enumerate(labels) if value == label]
        member_samples = [samples[index] for index in members]
        majority = majority_secondary(member_samples)
        name = name_from_profile(
            metrics["anchors"].get(str(label), []),
            majority,
            heterogeneous_flags[label],
        )
        cluster_id = "C%02d" % label
        error_rate_stats = {}
        if primary:
            vals = [
                primary["raw_by_index"][index]
                for index in members
                if primary["valid_mask"][index]
            ]
            names = Counter()
            for index in members:
                score = primary["scores"][index] or {}
                if score.get("metric_name"):
                    names[score["metric_name"]] += 1
            error_rate_stats = {
                "n_valid": len(vals),
                "median": median(vals),
                "iqr": iqr(vals),
                "mad": mad(vals),
                "p05": quantile(vals, 0.05) if vals else None,
                "p95": quantile(vals, 0.95) if vals else None,
                "metric_name_counts": dict(names),
            }
        profile = {
            "cluster_id": cluster_id,
            "n": len(members),
            "share": len(members) / float(n),
            "medoid_sample_id": samples[medoids[label]]["sample_id"],
            "anchors": metrics["anchors"].get(str(label), []),
            "secondary_tags": majority,
            "source_share": metrics["source_shares"][label],
            "error_rate": error_rate_stats,
            "snr_db_median": median(
                [
                    sample["features"]["numeric"].get("audio_quality.snr_db")
                    for sample in member_samples
                    if sample["features"]["numeric"].get("audio_quality.snr_db") is not None
                ]
            ),
            "rt60_sec_median": median(
                [
                    sample["features"]["numeric"].get("room_acoustic.rt60_sec")
                    for sample in member_samples
                    if sample["features"]["numeric"].get("room_acoustic.rt60_sec") is not None
                ]
            ),
            "duration_sec_median": median(
                [
                    sample["features"]["numeric"].get("basic_acoustic.duration_sec")
                    for sample in member_samples
                    if sample["features"]["numeric"].get("basic_acoustic.duration_sec") is not None
                ]
            ),
            "nonspeech_ratio_median": median(
                [
                    sample["features"]["numeric"].get("basic_acoustic.nonspeech_ratio")
                    for sample in member_samples
                    if sample["features"]["numeric"].get("basic_acoustic.nonspeech_ratio") is not None
                ]
            ),
            "error_driven": use_wer_input,
            "error_heterogeneous": heterogeneous_flags[label],
            "cluster_name": name,
            "cluster_name_status": "template_unreviewed",
        }
        summary_clusters.append(profile)
        for index in members:
            sample = samples[index]
            secondary = secondary_tags(sample)
            unknown = []
            if sample["features"]["numeric"].get("speaker.speaker_count") is None:
                unknown.append("speaker.speaker_count")
            if nested_get(sample["tags"], "language_content.language") is None:
                unknown.append("language_content.language")
            needs_review = bool(sample["conflicts"]) or labels[index] < 0
            score = empty_score()
            if primary:
                score = primary["scores"][index] or empty_score()
            payload = diagnostic_payload(
                score,
                missing=sample.get("scores") is None,
                used_as_input=use_wer_input and primary_model == args.wer_input_model,
            )
            payload["asr_system_id"] = primary_model
            payload["model_id"] = primary_model
            row = {
                "sample_id": sample["sample_id"],
                "global_index": sample["global_index"],
                "dataset": sample["dataset"],
                "group_id": sample["group_id"],
                "mode": mode,
                "partition_scope": partition_scope,
                "cluster_id": cluster_id,
                "cluster_name": name,
                "cluster_name_status": profile["cluster_name_status"],
                "cluster_name_source": "template_plus_unreviewed",
                "cluster_name_version": CLUSTER_NAME_VERSION,
                "cluster_profile_ref": "cluster_summary.json#%s" % cluster_id,
                "secondary_tags": secondary,
                "assignment_confidence": assignment_confidence(sample, metrics, label),
                "assignment_approximate": not publishable,
                "assignment_gate": {
                    "tag_gate_passed": bool(metrics["anchors"].get(str(label))),
                    "tag_candidate_count": k,
                    "tag_radius_passed": labels[index] >= 0,
                    "top2_margin": None,
                },
                "needs_review": needs_review,
                "unknown_reason": unknown + sample["conflicts"],
                "error_driven": use_wer_input,
                "error_heterogeneous": heterogeneous_flags[label],
                "tag_anchor": {
                    "status": "passed" if metrics["anchors"].get(str(label)) else "failed",
                    "top_features": [
                        "%s=%s" % (item["feature"], item["value"])
                        for item in metrics["anchors"].get(str(label), [])[:3]
                    ],
                },
                "assignment_basis": assignment_basis,
                "tag_assignment_basis": (
                    "all_sure_tagger_labels_plus_error"
                    if use_wer_input and all_labels
                    else (
                        "audio_plus_error"
                        if use_wer_input
                        else ("all_sure_tagger_labels" if all_labels else "audio_only")
                    )
                ),
                "feature_provenance": {
                    "used": used_features,
                    "excluded": excluded_features,
                },
                "coverage": {
                    "cluster_evidence": "partial" if unknown else "complete",
                    "unknown_tags": unknown,
                },
                "evidence_tags": evidence_tags_for(sample),
                "cluster_rules_version": rules_version,
                "diagnostic_error": payload,
            }
            cluster_rows.append(row)
    for index, label in enumerate(labels):
        if label >= 0:
            continue
        sample = samples[index]
        score = empty_score()
        if primary:
            score = primary["scores"][index] or empty_score()
        payload = diagnostic_payload(
            score,
            missing=sample.get("scores") is None,
            used_as_input=use_wer_input and primary_model == args.wer_input_model,
        )
        payload["asr_system_id"] = primary_model
        payload["model_id"] = primary_model
        cluster_rows.append(
            {
                "sample_id": sample["sample_id"],
                "global_index": sample["global_index"],
                "dataset": sample["dataset"],
                "group_id": sample["group_id"],
                "mode": mode,
                "partition_scope": partition_scope,
                "cluster_id": "C_REVIEW_OUTLIER",
                "cluster_name": None,
                "needs_review": True,
                "unknown_reason": sample["conflicts"] or ["insufficient_tag_neighbors"],
                "error_driven": use_wer_input,
                "assignment_basis": "rejected",
                "cluster_rules_version": rules_version,
                "diagnostic_error": payload,
            }
        )
    cluster_rows.sort(key=lambda row: row["global_index"])
    dispersion = []
    for model_id, item in diagnostics.items():
        compact = dict(item)
        compact.pop("raw_by_index", None)
        compact.pop("valid_mask", None)
        compact.pop("scores", None)
        dispersion.append(compact)
    summary = {
        "cluster_rules_version": rules_version,
        "mode": mode,
        "partition_scope": partition_scope,
        "assignment_basis": assignment_basis,
        "n": n,
        "k": k,
        "production_publishable": publishable,
        "posthoc_partition": not use_wer_input,
        "tag_feature_mode": "all_sure_tagger_labels" if all_labels else "calibrated_audio_subset",
        "tag_feature_counts": {
            kind: len(feature_specs[kind])
            for kind in ("numeric", "binary", "categorical", "sets")
        },
        "tag_axes": list(feature_specs["axes"]),
        "metrics": {
            "j": metrics["j"],
            "c_tag": metrics["c_tag"],
            "q_tag": metrics["q_tag"],
            "a_unanchored": metrics["a_unanchored"],
            "cut_tag": metrics["cut_tag"],
            "b_non_main": metrics["b_non_main"],
            "source_driven": metrics["source_driven"],
            "gates": metrics["gates"],
            "recalls": metrics["recalls"],
        },
        "clusters": summary_clusters,
        "dispersion_evaluations": dispersion,
        "refine_history": refine_history,
        "inputs": hashes,
    }
    return cluster_rows, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-data scene clustering (calibrated tags by default)")
    parser.add_argument("--tags", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--index-map", type=Path, default=DEFAULT_INDEX_MAP)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument("--sampled-manifest", type=Path, default=DEFAULT_SAMPLED_MANIFEST)
    parser.add_argument("--wer-cer", type=Path, default=DEFAULT_WER_CER)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--use-wer-input",
        action="store_true",
        help="add the selected model's WER/CER error_rate as a fifth distance axis",
    )
    parser.add_argument(
        "--wer-input-model",
        default=DEFAULT_WER_INPUT_MODEL,
        help="ASR model supplying the optional WER/CER axis (default: %(default)s)",
    )
    parser.add_argument(
        "--use-all-sure-tagger-labels",
        "--use-all-sure-tagger-tags",
        "--use-all-labels",
        "--use-all-tags",
        dest="use_all_sure_tagger_labels",
        action="store_true",
        help="include every discovered sure-tagger label (default: calibrated audio subset)",
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--eval-models", default=",".join(BILINGUAL_MODELS))
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--n-error-min", type=int, default=10)
    parser.add_argument("--tau-num", type=float, default=3.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--q-min", type=float, default=0.12)
    parser.add_argument("--lift-min", type=float, default=1.4)
    parser.add_argument("--source-max-share", type=float, default=0.90)
    parser.add_argument("--b-non-main-max", type=float, default=0.20)
    parser.add_argument("--cut-tag-max", type=float, default=0.85)
    parser.add_argument("--lambda-t", type=float, default=1.0)
    parser.add_argument("--lambda-a", type=float, default=0.8)
    parser.add_argument("--lambda-k", type=float, default=0.15)
    parser.add_argument("--lambda-b", type=float, default=0.8)
    parser.add_argument("--lambda-r", type=float, default=0.8)
    parser.add_argument("--max-refine", type=int, default=6)
    parser.add_argument("--epsilon-refine", type=float, default=0.002)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument(
        "--cluster-methods",
        default="kmedoids",
        choices=cluster_methods.DEFAULT_METHODS,
        help="clustering method packaged in this release (default: %(default)s)",
    )
    parser.add_argument(
        "--wer-filter",
        action="store_true",
        help="use post-hoc mean SηK to drop weak tags-only candidates (never as a distance axis)",
    )
    parser.add_argument(
        "--no-wer-filter",
        action="store_false",
        dest="wer_filter",
        help="disable post-hoc WER/CER filtering of candidates",
    )
    parser.set_defaults(wer_filter=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.seed_values = [int(item) for item in args.seeds.split(",") if item.strip()]
    args.eval_model_ids = [item for item in args.eval_models.split(",") if item.strip()]
    args.cluster_method_list = [
        item.strip() for item in str(getattr(args, "cluster_methods", "")).split(",") if item.strip()
    ] or ["kmedoids"]
    for model_id in args.eval_model_ids:
        if model_id not in MODEL_IDS:
            raise SystemExit("unknown eval model %s" % model_id)
    if args.wer_input_model not in MODEL_IDS:
        raise SystemExit("unknown WER input model %s" % args.wer_input_model)
    print("loading tag snapshot (WER/CER not opened)")
    samples = load_tag_snapshot(args)
    hashes = {
        "tags_sha256": sha256_file(args.tags),
        "index_map_sha256": sha256_file(args.index_map),
        "sampled_manifest_sha256": sha256_file(args.sampled_manifest),
        "tagger_profile": None,
    }
    if args.run_metadata.is_file():
        metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
        hashes["tagger_profile"] = metadata.get("profile")
        hashes["tagger_status"] = metadata.get("status")
        hashes["speaker_claim_policy"] = metadata.get("speaker_claim_policy")
    print("loaded %s samples" % len(samples))
    k_max_support = int(math.floor(len(samples) / float(max(args.n_min, 1))))
    if k_max_support < args.k_min:
        raise SystemExit("sample support K_max=%s < k_min=%s" % (k_max_support, args.k_min))
    args.k_max = min(args.k_max, k_max_support)
    errors_attached = False
    error_spec = None
    if args.use_wer_input:
        print("opening WER/CER before clustering for input model %s" % args.wer_input_model)
        attach_error_diagnostics(samples, args)
        errors_attached = True
        hashes["sample_wer_cer_sha256"] = sha256_file(args.wer_cer)
        hashes["sidecar_sha256"] = sha256_file(args.sidecar) if args.sidecar.is_file() else None
        error_spec = fit_error_preprocess(samples, args.wer_input_model)
        print(
            "using WER/CER axis: n_valid=%s missing_rate=%s metric_counts=%s"
            % (
                error_spec["n_valid"],
                error_spec["missing_rate"],
                error_spec["metric_name_counts"],
            )
        )
    feature_specs = args.feature_specs
    scales = fit_numeric_preprocess(samples, feature_specs)
    print("computing %s" % ("d_tags_plus_wer_cer" if error_spec else "d_tag"))
    d_tag = build_tag_distance(samples, scales, args.tau_num, error_spec, feature_specs)
    _counts, review = neighbor_counts(d_tag, args.min_neighbors)
    active = [index for index in range(len(samples)) if index not in set(review)]
    print("review outliers before clustering: %s" % len(review))
    candidates = search_candidates(samples, d_tag, active, args)
    use_wer_filter = bool(getattr(args, "wer_filter", False)) and not args.use_wer_input
    if use_wer_filter and not args.skip_diagnostics:
        if not errors_attached:
            print("opening WER/CER after d_tag freeze for candidate filtering only")
            attach_error_diagnostics(samples, args)
            errors_attached = True
            hashes["sample_wer_cer_sha256"] = sha256_file(args.wer_cer)
            hashes["sidecar_sha256"] = sha256_file(args.sidecar) if args.sidecar.is_file() else None
        for item in candidates:
            eta2_values = []
            snk_values = []
            per_model = {}
            for model_id in args.eval_model_ids:
                diag = diagnose_model(
                    samples, item["labels"], item["k"], model_id, args.n_error_min
                )
                eta2 = (diag.get("error_homogeneity") or {}).get("eta2")
                snk = cluster_methods.snk_score(eta2, item["k"])
                per_model[model_id] = {"eta2": eta2, "snk": snk, "n_valid": diag.get("n_valid")}
                if eta2 is not None:
                    eta2_values.append(eta2)
                    snk_values.append(snk)
            item["posthoc"] = {
                "mean_eta2": (
                    sum(eta2_values) / float(len(eta2_values)) if eta2_values else None
                ),
                "mean_snk": (
                    sum(snk_values) / float(len(snk_values)) if snk_values else None
                ),
                "per_model": per_model,
                "used_for_assignment": False,
            }
    chosen, publishable = select_candidate(candidates, wer_filter=use_wer_filter)
    if chosen is None:
        raise SystemExit("no clusterable candidate was produced")
    print("refining split/merge on %s" % ("tags + WER/CER" if error_spec else "tags only"))
    labels, medoids, refine_history = refine_split_merge(
        samples, chosen["labels"], chosen["medoids"], d_tag, args
    )
    labels, medoids, mapping = canonicalize(samples, labels, d_tag)
    metrics = evaluate_candidate(samples, labels, medoids, d_tag, args)
    print("frozen K=%s publishable=%s J=%s Q_tag=%s" % (metrics["k"], publishable, metrics["j"], metrics["q_tag"]))

    diagnostics = {}
    diagnostic_model_ids = [] if args.skip_diagnostics else list(args.eval_model_ids)
    if args.use_wer_input and args.wer_input_model not in diagnostic_model_ids:
        diagnostic_model_ids.append(args.wer_input_model)
    if diagnostic_model_ids:
        print("summarizing WER/CER diagnostics and clustering-input provenance")
        if not errors_attached:
            attach_error_diagnostics(samples, args)
            errors_attached = True
        hashes["sample_wer_cer_sha256"] = sha256_file(args.wer_cer)
        hashes["sidecar_sha256"] = sha256_file(args.sidecar) if args.sidecar.is_file() else None
        for model_id in diagnostic_model_ids:
            diagnostics[model_id] = diagnose_model(
                samples, labels, metrics["k"], model_id, args.n_error_min
            )
            used_for_assignment = args.use_wer_input and model_id == args.wer_input_model
            diagnostics[model_id]["used_for_assignment"] = used_for_assignment
            diagnostics[model_id]["dispersion_evaluation_role"] = (
                "clustering_input_summary" if used_for_assignment else "posthoc_eval_dispersion"
            )
            homogeneity = diagnostics[model_id].get("error_homogeneity") or {}
            homogeneity["used_for_assignment"] = used_for_assignment
            print(
                "  %s D_err=%s gain=%s eta2=%s n_valid=%s"
                % (
                    model_id,
                    diagnostics[model_id]["d_err"],
                    diagnostics[model_id]["gain"],
                    (diagnostics[model_id].get("error_homogeneity") or {}).get("eta2"),
                    diagnostics[model_id]["n_valid"],
                )
            )
    primary_model = (
        args.wer_input_model
        if args.use_wer_input
        else (args.eval_model_ids[0] if args.eval_model_ids else "Qwen__Qwen3-ASR-1.7B")
    )
    rows, summary = build_outputs(
        samples,
        labels,
        medoids,
        metrics,
        args,
        hashes,
        publishable,
        refine_history,
        diagnostics,
        primary_model,
    )
    summary["search"] = {
        "n_candidates": len(candidates),
        "selected_method": chosen["method"],
        "selected_k_before_refine": chosen["k"],
        "selected_seed": chosen.get("seed"),
        "canonical_mapping": mapping,
        "candidate_metrics": [
            {
                "method": item["method"],
                "k": item["k"],
                "seed": item.get("seed"),
                "j": item["metrics"]["j"],
                "c_tag": item["metrics"]["c_tag"],
                "q_tag": item["metrics"]["q_tag"],
                "passed": item["metrics"]["passed"],
                "gates": item["metrics"]["gates"],
            }
            for item in candidates
        ],
    }
    summary["preprocess"] = {
        "numeric_scales": scales,
        "tau_num": args.tau_num,
        "n_review_precluster": len(review),
        "axes": list(feature_specs["axes"]) + ([ERROR_AXIS] if error_spec else []),
        "axis_aggregation": "equal_mean_over_available_axes",
        "tag_feature_mode": "all_sure_tagger_labels"
        if args.use_all_sure_tagger_labels
        else "calibrated_audio_subset",
        "feature_specs": {
            kind: [{"path": path, "axis": axis} for path, axis in feature_specs[kind]]
            for kind in ("numeric", "binary", "categorical", "sets")
        },
        "wer_input_enabled": bool(args.use_wer_input),
        "wer_input": error_spec,
    }
    output_dir = args.output_dir
    assignment_text = "".join("%s\n" % json_line(row) for row in rows)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    run_meta = {
        "cluster_rules_version": rules_version_for(args),
        "mode": "model_specific" if args.use_wer_input else MODE,
        "assignment_basis": (
            ALL_LABELS_WER_ASSIGNMENT_BASIS
            if args.use_wer_input and args.use_all_sure_tagger_labels
            else (
                "audio_tags_plus_wer_cer"
                if args.use_wer_input
                else (
                    ALL_LABELS_ASSIGNMENT_BASIS
                    if args.use_all_sure_tagger_labels
                    else ASSIGNMENT_BASIS
                )
            )
        ),
        "partition_scope": "model_specific" if args.use_wer_input else "shared",
        "started": datetime.now(timezone.utc).astimezone().isoformat(),
        "n": len(samples),
        "k": metrics["k"],
        "production_publishable": publishable,
        "diagnostics_attached": bool(diagnostics),
        "wer_input_enabled": bool(args.use_wer_input),
        "wer_input_model": args.wer_input_model if args.use_wer_input else None,
        "use_all_sure_tagger_labels": bool(args.use_all_sure_tagger_labels),
        "tag_feature_mode": "all_sure_tagger_labels"
        if args.use_all_sure_tagger_labels
        else "calibrated_audio_subset",
        "tag_feature_counts": {
            kind: len(feature_specs[kind])
            for kind in ("numeric", "binary", "categorical", "sets")
        },
        "tag_axes": list(feature_specs["axes"]),
        "outputs": {
            "cluster_assignments": str(output_dir / "cluster_assignments.jsonl"),
            "cluster_assignments_sha256": sha256_text(assignment_text),
            "cluster_summary": str(output_dir / "cluster_summary.json"),
            "cluster_summary_sha256": sha256_text(summary_text),
        },
        "inputs": hashes,
    }
    write_text_atomic(output_dir / "cluster_assignments.jsonl", assignment_text)
    write_text_atomic(output_dir / "cluster_summary.json", summary_text)
    write_text_atomic(
        output_dir / "run_metadata.json",
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        "wrote %s assignments and K=%s clusters to %s; publishable=%s Q_tag=%s"
        % (len(rows), metrics["k"], output_dir, publishable, metrics["q_tag"])
    )


if __name__ == "__main__":
    main()
