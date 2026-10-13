#!/usr/bin/env python3
"""Score masked structural predictions and render an AMI v0 report."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import unicodedata


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_ROOT = SCRIPT_DIR.parent
DEFAULT_UNIVERSE = SCRIPT_DIR / "field_universe.structural-v0.1.json"
DEFAULT_GT = COMPARE_ROOT / "evaluation" / "ami-v0" / "ground_truth.jsonl"
DEFAULT_OUTPUT_DIR = COMPARE_ROOT / "evaluation" / "ami-v0"

RESULT_STATES = (
    "result_known",
    "result_null",
    "result_missing",
    "schema_invalid",
    "runtime_failed",
)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError("%s:%d is invalid JSON: %s" % (path, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("%s:%d must contain an object" % (path, line_number))
            rows.append(value)
    return rows


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path, rows):
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_specs(universe):
    output = {}
    for field in universe.get("fields", []):
        path = field.get("path") if isinstance(field, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError("field universe contains an invalid path")
        if path in output:
            raise ValueError("duplicate field path: %s" % path)
        output[path] = field
    if not output:
        raise ValueError("field universe has no fields")
    return output


def valid_kind(value, kind):
    if value is None:
        return True
    if kind == "number":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
    if kind == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string_array":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if kind == "object":
        return isinstance(value, dict)
    return False


def keyed_rows(rows, label):
    result = {}
    for index, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("%s row %d has no sample_id" % (label, index))
        if sample_id in result:
            raise ValueError("%s has duplicate sample_id %s" % (label, sample_id))
        result[sample_id] = row
    return result


def prediction_state(prediction, path, spec):
    if prediction is None or prediction.get("status") == "runtime_failed":
        return "runtime_failed", None
    if prediction.get("status") == "schema_invalid" or not prediction.get(
        "schema_valid", False
    ):
        return "schema_invalid", None
    provided = prediction.get("provided_fields")
    if not isinstance(provided, list) or path not in provided:
        return "result_missing", None
    values = prediction.get("values")
    if not isinstance(values, dict) or path not in values:
        return "schema_invalid", None
    value = values[path]
    if value is None:
        return "result_null", None
    if not valid_kind(value, spec.get("kind")):
        return "schema_invalid", None
    return "result_known", value


def explicit_negative_or_empty(value):
    if value is False:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return True
    return value == []


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def harmonic_mean(left, right):
    if left is None or right is None:
        return None
    if left + right == 0:
        return 0.0
    return 2.0 * left * right / (left + right)


def mean_defined(values):
    present = [value for value in values if value is not None]
    return sum(present) / float(len(present)) if present else None


def base_counts(items, path, spec):
    states = {state: 0 for state in RESULT_STATES}
    explicit = 0
    known = []
    for ground_truth, prediction in items:
        value = ground_truth["ground_truth"]["values"][path]
        if explicit_negative_or_empty(value):
            explicit += 1
        state, predicted = prediction_state(prediction, path, spec)
        states[state] += 1
        if state == "result_known":
            known.append((value, predicted))
    output = {
        "eligible_gt": len(items),
        "explicit_negative_or_empty": explicit,
        "valid_prediction_rate": safe_divide(states["result_known"], len(items)),
        "interface_provided_count": states["result_known"] + states["result_null"],
    }
    output.update(states)
    output["interface_provided_rate"] = safe_divide(
        output["interface_provided_count"], len(items)
    )
    output["interface_unavailable"] = (
        len(items) > 0
        and output["interface_provided_count"] == 0
        and states["result_missing"] == len(items)
    )
    return output, known


def numeric_summary(items, path, spec, tolerance=None, exact_integer=False):
    output, known = base_counts(items, path, spec)
    errors = [float(predicted) - float(reference) for reference, predicted in known]
    absolute = [abs(value) for value in errors]
    output.update(
        {
            "metric_n": len(known),
            "mae": sum(absolute) / float(len(absolute)) if absolute else None,
            "median_absolute_error": statistics.median(absolute) if absolute else None,
            "bias": sum(errors) / float(len(errors)) if errors else None,
            "rmse": (
                math.sqrt(sum(value * value for value in errors) / float(len(errors)))
                if errors
                else None
            ),
        }
    )
    if tolerance is not None:
        correct = sum(abs(value) <= tolerance for value in errors)
        output["tolerance"] = tolerance
        output["within_tolerance_count"] = correct
        output["within_tolerance_on_known"] = safe_divide(correct, len(known))
        output["within_tolerance_e2e"] = safe_divide(correct, len(items))
    if exact_integer:
        correct = sum(reference == predicted for reference, predicted in known)
        output["exact_match_count"] = correct
        output["exact_accuracy_on_known"] = safe_divide(correct, len(known))
        output["exact_accuracy_e2e"] = safe_divide(correct, len(items))
    return output


def boolean_summary(items, path, spec):
    output, known = base_counts(items, path, spec)
    positive_gt = sum(
        ground_truth["ground_truth"]["values"][path] is True
        for ground_truth, _ in items
    )
    negative_gt = len(items) - positive_gt
    tp = sum(reference is True and predicted is True for reference, predicted in known)
    fp = sum(reference is False and predicted is True for reference, predicted in known)
    tn = sum(reference is False and predicted is False for reference, predicted in known)
    fn_predicted = sum(
        reference is True and predicted is False for reference, predicted in known
    )
    false_negative_e2e = positive_gt - tp
    false_positive_negative_class_e2e = negative_gt - tn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, positive_gt)
    negative_precision = safe_divide(tn, tn + fn_predicted)
    specificity = safe_divide(tn, negative_gt)
    positive_f1 = harmonic_mean(precision, recall)
    negative_f1 = harmonic_mean(negative_precision, specificity)
    correct = tp + tn
    output.update(
        {
            "metric_n": len(items),
            "known_metric_n": len(known),
            "positive_ground_truth": positive_gt,
            "negative_ground_truth": negative_gt,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative_known": fn_predicted,
            "false_negative_e2e": false_negative_e2e,
            "true_negative": tn,
            "negative_class_misses_e2e": false_positive_negative_class_e2e,
            "precision": precision,
            "recall": recall,
            "f1": positive_f1,
            "specificity": specificity,
            "balanced_accuracy": mean_defined([recall, specificity]),
            "macro_f1": mean_defined([positive_f1, negative_f1]),
            "accuracy_on_known": safe_divide(correct, len(known)),
            "accuracy_e2e": safe_divide(correct, len(items)),
        }
    )
    return output


def categorical_summary(items, path, spec):
    output, known = base_counts(items, path, spec)
    correct = sum(reference == predicted for reference, predicted in known)
    output.update(
        {
            "metric_n": len(items),
            "known_metric_n": len(known),
            "exact_match_count": correct,
            "accuracy_on_known": safe_divide(correct, len(known)),
            "accuracy_e2e": safe_divide(correct, len(items)),
        }
    )
    return output


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for offset in range(index, end):
            ranks[indexed[offset][0]] = rank
        index = end
    return ranks


def pearson(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def continuous_summary(items, path, spec):
    output = numeric_summary(items, path, spec)
    _, known = base_counts(items, path, spec)
    references = [float(reference) for reference, _ in known]
    predictions = [float(predicted) for _, predicted in known]
    output["spearman"] = pearson(average_ranks(references), average_ranks(predictions))
    return output


def normalize_asr(text):
    lowered = str(text or "").lower()
    characters = []
    for character in lowered:
        if unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def edit_counts(reference, hypothesis):
    """Return deterministic Levenshtein S/D/I counts for token sequences."""
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_token in enumerate(reference, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            if ref_token == hyp_token:
                diagonal = previous[hyp_index - 1]
            else:
                base = previous[hyp_index - 1]
                diagonal = (base[0] + 1, base[1] + 1, base[2], base[3])
            deletion_base = previous[hyp_index]
            deletion = (
                deletion_base[0] + 1,
                deletion_base[1],
                deletion_base[2] + 1,
                deletion_base[3],
            )
            insertion_base = current[hyp_index - 1]
            insertion = (
                insertion_base[0] + 1,
                insertion_base[1],
                insertion_base[2],
                insertion_base[3] + 1,
            )
            current.append(min((diagonal, deletion, insertion), key=lambda item: item))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {
        "distance": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def text_wer_summary(items, path, spec):
    output, known = base_counts(items, path, spec)
    totals = {
        "reference_words": 0,
        "hypothesis_words": 0,
        "word_errors": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    exact = 0
    known_ids = set()
    for ground_truth, prediction in items:
        sample_id = ground_truth["sample_id"]
        reference_text = ground_truth["ground_truth"]["values"][path]
        state, predicted_text = prediction_state(prediction, path, spec)
        if state == "result_known":
            known_ids.add(sample_id)
            hypothesis_text = predicted_text
        else:
            hypothesis_text = ""
        reference = normalize_asr(reference_text).split()
        hypothesis = normalize_asr(hypothesis_text).split()
        counts = edit_counts(reference, hypothesis)
        totals["reference_words"] += len(reference)
        totals["hypothesis_words"] += len(hypothesis)
        totals["word_errors"] += counts["distance"]
        totals["substitutions"] += counts["substitutions"]
        totals["deletions"] += counts["deletions"]
        totals["insertions"] += counts["insertions"]
        exact += reference == hypothesis
    output.update(totals)
    output.update(
        {
            "metric_n": len(items),
            "known_metric_n": len(known_ids),
            "wer": safe_divide(totals["word_errors"], totals["reference_words"]),
            "normalized_exact_match_count": exact,
            "normalized_exact_accuracy_e2e": safe_divide(exact, len(items)),
            "normalization": "unicode-punctuation-strip-lower-whitespace-v0.1",
        }
    )
    return output


def partial_multilabel_summary(items, path, spec):
    base, _ = base_counts(items, path, spec)
    labels = {}
    all_labels = set()
    for ground_truth, _ in items:
        masks = ground_truth["ground_truth"].get("label_mask_by_field", {}).get(
            path, {}
        )
        all_labels.update(label for label, enabled in masks.items() if enabled)
    for label in sorted(all_labels):
        eligible = []
        for ground_truth, prediction in items:
            mask = ground_truth["ground_truth"].get("label_mask_by_field", {}).get(
                path, {}
            )
            if not mask.get(label, False):
                continue
            reference_values = ground_truth["ground_truth"]["values"].get(path) or []
            reference = label in reference_values
            state, prediction_values = prediction_state(prediction, path, spec)
            predicted = (
                label in prediction_values if state == "result_known" else None
            )
            eligible.append((reference, predicted, state))
        positives = sum(reference for reference, _, _ in eligible)
        negatives = len(eligible) - positives
        tp = sum(reference and predicted is True for reference, predicted, _ in eligible)
        fp = sum(not reference and predicted is True for reference, predicted, _ in eligible)
        tn = sum(not reference and predicted is False for reference, predicted, _ in eligible)
        recall = safe_divide(tp, positives)
        precision = safe_divide(tp, tp + fp)
        labels[label] = {
            "eligible_gt": len(eligible),
            "positive_ground_truth": positives,
            "negative_ground_truth": negatives,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative_e2e": positives - tp,
            "true_negative": tn,
            "precision": precision,
            "positive_recall": recall,
            "f1": harmonic_mean(precision, recall),
            "known_prediction_count": sum(
                state == "result_known" for _, _, state in eligible
            ),
        }
    base.update({"metric_n": sum(value["eligible_gt"] for value in labels.values()), "labels": labels})
    return base


def score_field(items, path, spec):
    metric = spec.get("metric")
    if metric == "integer":
        return numeric_summary(items, path, spec, exact_integer=True)
    if metric == "tolerance":
        return numeric_summary(
            items, path, spec, tolerance=float(spec.get("tolerance", 0.0))
        )
    if metric == "continuous":
        return continuous_summary(items, path, spec)
    if metric == "boolean":
        return boolean_summary(items, path, spec)
    if metric == "categorical":
        return categorical_summary(items, path, spec)
    if metric == "text_wer":
        return text_wer_summary(items, path, spec)
    if metric == "positive_multilabel":
        return partial_multilabel_summary(items, path, spec)
    raise ValueError("unsupported score metric %s for %s" % (metric, path))


def expected_contract_value(rule_name, prediction, fields):
    values = prediction.get("values", {})
    provided = set(prediction.get("provided_fields", []))

    def dependency(path):
        if path not in provided:
            return "missing", None
        value = values.get(path)
        if value is None:
            return "null", None
        if not valid_kind(value, fields[path]["kind"]):
            return "invalid", None
        return "known", value

    if rule_name == "speaker_present_from_count":
        status, count = dependency("speaker.speaker_count")
        if status == "known":
            return "known", count > 0
        return status, None
    if rule_name == "music_present_from_events":
        status, events = dependency("sound_field_scene.speech_music_events")
        if status == "known":
            return "known", "music" in events
        return status, None
    if rule_name == "speaker_change_rate_per_min":
        count_status, count = dependency("speaker.speaker_change_count")
        duration_status, duration = dependency("basic_acoustic.duration_sec")
        statuses = (count_status, duration_status)
        if "invalid" in statuses:
            return "invalid", None
        if "missing" in statuses:
            return "missing", None
        if "null" in statuses or duration <= 0:
            return "null", None
        return "known", round(count * 60.0 / duration, 6)
    if rule_name == "filler_ratio":
        filler_status, filler = dependency("language_content.filler")
        words_status, words = dependency("language_content.word_count")
        statuses = (filler_status, words_status)
        if "invalid" in statuses:
            return "invalid", None
        if "missing" in statuses:
            return "missing", None
        if "null" in statuses or words <= 0:
            return "null", None
        return "known", round(float(filler) / float(words), 6)
    raise ValueError("unknown contract rule: %s" % rule_name)


def contract_outcome(rule, prediction, fields):
    if prediction is None or prediction.get("status") == "runtime_failed":
        return "runtime_failed"
    if prediction.get("status") == "schema_invalid":
        return "schema_invalid"
    target = rule["target"]
    provided = set(prediction.get("provided_fields", []))
    if target not in provided:
        return "missing_target"
    actual = prediction.get("values", {}).get(target)
    if actual is not None and not valid_kind(actual, fields[target]["kind"]):
        return "schema_invalid"
    dependency_status, expected = expected_contract_value(rule["name"], prediction, fields)
    if dependency_status == "missing":
        return "dependency_missing"
    if dependency_status == "invalid":
        return "dependency_invalid"
    if dependency_status == "null":
        return "dependency_null"
    if actual is None:
        return "value_mismatch"
    if isinstance(expected, float):
        return "pass" if abs(float(actual) - expected) <= 1e-6 else "value_mismatch"
    return "pass" if actual == expected else "value_mismatch"


def contract_summary(items, universe, fields):
    rules = {}
    for rule in universe.get("contract_rules", []):
        counts = {
            "pass": 0,
            "missing_target": 0,
            "value_mismatch": 0,
            "dependency_missing": 0,
            "dependency_null": 0,
            "dependency_invalid": 0,
            "schema_invalid": 0,
            "runtime_failed": 0,
        }
        for _, prediction in items:
            outcome = contract_outcome(rule, prediction, fields)
            counts[outcome] += 1
        counts["eligible"] = len(items)
        counts["checkable"] = counts["pass"] + counts["value_mismatch"]
        counts["pass_rate"] = safe_divide(counts["pass"], counts["checkable"])
        rules[rule["name"]] = counts
    all_pass = 0
    for _, prediction in items:
        outcomes = [
            contract_outcome(rule, prediction, fields)
            for rule in universe.get("contract_rules", [])
        ]
        if outcomes and all(
            outcome in ("pass", "dependency_missing", "dependency_null")
            for outcome in outcomes
        ):
            all_pass += 1
    return {
        "sample_count": len(items),
        "all_contracts_pass_count": all_pass,
        "all_contracts_pass_rate": safe_divide(all_pass, len(items)),
        "rules": rules,
    }


def adapter_summary(items, field_count):
    statuses = {}
    formats = {}
    provided_total = 0
    adapter_valid = 0
    native_valid = 0
    for _, prediction in items:
        if prediction is None:
            status = "missing_prediction_row"
            source_format = "missing"
            provided = []
        else:
            status = prediction.get("status", "unknown")
            source_format = prediction.get("source_format", "unknown")
            provided = prediction.get("provided_fields", [])
        statuses[status] = statuses.get(status, 0) + 1
        formats[source_format] = formats.get(source_format, 0) + 1
        provided_total += len(provided) if isinstance(provided, list) else 0
        if prediction is not None:
            adapter_valid += prediction.get("adapter_schema_valid") is True
            native_valid += prediction.get("native_current_schema_valid") is True
    return {
        "sample_count": len(items),
        "status_counts": statuses,
        "source_format_counts": formats,
        "mean_provided_field_fraction": safe_divide(
            provided_total, len(items) * field_count
        ),
        "adapter_schema_valid_count": adapter_valid,
        "adapter_schema_valid_rate": safe_divide(adapter_valid, len(items)),
        "native_current_schema_valid_count": native_valid,
        "native_current_schema_valid_rate": safe_divide(native_valid, len(items)),
    }


def split_items(gt_rows, prediction_map, split_name):
    rows = []
    for ground_truth in gt_rows:
        if split_name != "overall" and ground_truth.get("split") != split_name:
            continue
        rows.append((ground_truth, prediction_map.get(ground_truth["sample_id"])))
    return rows


def score_split(items, universe, fields):
    capabilities = {}
    partial = {}
    for path, spec in fields.items():
        metric = spec.get("metric")
        if metric == "unsupported":
            continue
        if metric == "positive_multilabel":
            if any(
                ground_truth["ground_truth"]
                .get("label_mask_by_field", {})
                .get(path, {})
                for ground_truth, _ in items
            ):
                partial[path] = score_field(items, path, spec)
            continue
        eligible_items = [
            (ground_truth, prediction)
            for ground_truth, prediction in items
            if ground_truth["ground_truth"]["gt_mask"].get(path, False)
        ]
        if eligible_items:
            capabilities[path] = score_field(eligible_items, path, spec)
    return {
        "sample_count": len(items),
        "capabilities": capabilities,
        "partial_multilabel": partial,
        "contracts": contract_summary(items, universe, fields),
        "adapter": adapter_summary(items, len(fields)),
    }


def field_sample_result(ground_truth, prediction, path, spec):
    reference = ground_truth["ground_truth"]["values"][path]
    state, predicted = prediction_state(prediction, path, spec)
    result = {"state": state}
    if state == "result_known":
        metric = spec.get("metric")
        if metric in ("integer", "boolean", "categorical"):
            result["correct"] = reference == predicted
        elif metric == "tolerance":
            result["absolute_error"] = abs(float(predicted) - float(reference))
            result["correct"] = result["absolute_error"] <= float(spec["tolerance"])
        elif metric == "continuous":
            result["absolute_error"] = abs(float(predicted) - float(reference))
        elif metric == "text_wer":
            ref_tokens = normalize_asr(reference).split()
            hyp_tokens = normalize_asr(predicted).split()
            counts = edit_counts(ref_tokens, hyp_tokens)
            result["word_errors"] = counts["distance"]
            result["reference_words"] = len(ref_tokens)
    return result


def per_sample_rows(gt_rows, systems, fields, universe):
    rows = []
    for system, prediction_map in systems.items():
        for ground_truth in gt_rows:
            prediction = prediction_map.get(ground_truth["sample_id"])
            field_results = {}
            for path, spec in fields.items():
                if ground_truth["ground_truth"]["gt_mask"].get(path, False):
                    field_results[path] = field_sample_result(
                        ground_truth, prediction, path, spec
                    )
            contracts = {
                rule["name"]: contract_outcome(rule, prediction, fields)
                for rule in universe.get("contract_rules", [])
            }
            rows.append(
                {
                    "schema_version": "structural-sample-score-v0.1",
                    "system": system,
                    "sample_id": ground_truth["sample_id"],
                    "split": ground_truth["split"],
                    "group_id": ground_truth["group_id"],
                    "prediction_status": (
                        prediction.get("status") if prediction else "runtime_failed"
                    ),
                    "fields": field_results,
                    "contracts": contracts,
                }
            )
    return rows


def format_percent(value):
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def format_number(value, digits=4):
    return "n/a" if value is None else ("%%.%df" % digits) % value


def metric_headline(spec, metric):
    if metric.get("interface_unavailable"):
        if spec.get("metric") == "text_wer":
            return "interface unavailable (e2e empty-hypothesis WER %s)" % format_percent(
                metric.get("wer")
            )
        return "interface unavailable (e2e abstention %s)" % format_percent(
            1.0 - (metric.get("accuracy_e2e") or 0.0)
        )
    name = spec.get("metric")
    if name == "integer":
        return "exact %s; MAE %s" % (
            format_percent(metric.get("exact_accuracy_e2e")),
            format_number(metric.get("mae")),
        )
    if name == "tolerance":
        return "within tolerance %s; MAE %s" % (
            format_percent(metric.get("within_tolerance_e2e")),
            format_number(metric.get("mae")),
        )
    if name == "continuous":
        return "MAE %s; Spearman %s" % (
            format_number(metric.get("mae")),
            format_number(metric.get("spearman")),
        )
    if name == "boolean":
        return "F1 %s; balanced acc %s; e2e acc %s" % (
            format_percent(metric.get("f1")),
            format_percent(metric.get("balanced_accuracy")),
            format_percent(metric.get("accuracy_e2e")),
        )
    if name == "categorical":
        return "e2e accuracy %s" % format_percent(metric.get("accuracy_e2e"))
    if name == "text_wer":
        return "WER %s; valid transcript %s" % (
            format_percent(metric.get("wer")),
            format_percent(metric.get("valid_prediction_rate")),
        )
    return ""


def render_report(metrics, universe, fields):
    lines = [
        "# AMI Structural Evaluation v0",
        "",
        "Status: legacy compatibility baseline. The stored outputs predate the current shared schema.",
        "",
        "Null, missing, schema-invalid, and runtime-failed predictions remain in `eligible_gt`. "
        "Continuous errors are computed on known numeric predictions and are always paired with valid prediction rate.",
        "",
        "## Dataset",
        "",
        "| Split | Samples | Groups |",
        "| --- | ---: | ---: |",
    ]
    for split_name in ("overall", "legacy_200", "diverse_100"):
        split = metrics["dataset"]["splits"][split_name]
        lines.append(
            "| %s | %d | %d |"
            % (split_name, split["sample_count"], split["group_count"])
        )
    for system, system_metrics in metrics["systems"].items():
        lines.extend(["", "## %s" % system])
        adapter = system_metrics["splits"]["overall"]["adapter"]
        format_text = ", ".join(
            "%s=%d" % (name, count)
            for name, count in sorted(adapter["source_format_counts"].items())
        )
        lines.extend(
            [
                "",
                "Source formats: %s." % format_text,
                "Adapter-normalized schema valid: %s; native current-structural-schema valid: %s; mean provided field fraction: %s."
                % (
                    format_percent(adapter["adapter_schema_valid_rate"]),
                    format_percent(adapter["native_current_schema_valid_rate"]),
                    format_percent(adapter["mean_provided_field_fraction"]),
                ),
            ]
        )
        for split_name in ("overall", "legacy_200", "diverse_100"):
            split = system_metrics["splits"][split_name]
            lines.extend(
                [
                    "",
                    "### %s" % split_name,
                    "",
                    "| Field | Eligible | Known | Null | Missing | Invalid/failed | Result |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for path, value in split["capabilities"].items():
                invalid_failed = value["schema_invalid"] + value["runtime_failed"]
                lines.append(
                    "| `%s` | %d | %d | %d | %d | %d | %s |"
                    % (
                        path,
                        value["eligible_gt"],
                        value["result_known"],
                        value["result_null"],
                        value["result_missing"],
                        invalid_failed,
                        metric_headline(fields[path], value),
                    )
                )
            for path, value in split["partial_multilabel"].items():
                for label, label_metric in value["labels"].items():
                    lines.append(
                        "| `%s[%s]` | %d | %d | %d | %d | %d | positive recall %s |"
                        % (
                            path,
                            label,
                            label_metric["eligible_gt"],
                            label_metric["known_prediction_count"],
                            value["result_null"],
                            value["result_missing"],
                            value["schema_invalid"] + value["runtime_failed"],
                            format_percent(label_metric["positive_recall"]),
                        )
                    )
        overall_contract = system_metrics["splits"]["overall"]["contracts"]
        lines.extend(
            [
                "",
                "### Contract Validation (overall)",
                "",
                "| Rule | Pass | Missing target | Mismatch | Dependency unavailable | Invalid/failed | Pass rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, value in overall_contract["rules"].items():
            unavailable = (
                value["dependency_missing"]
                + value["dependency_null"]
                + value["dependency_invalid"]
            )
            invalid_failed = value["schema_invalid"] + value["runtime_failed"]
            lines.append(
                "| `%s` | %d | %d | %d | %d | %d | %s |"
                % (
                    name,
                    value["pass"],
                    value["missing_target"],
                    value["value_mismatch"],
                    unavailable,
                    invalid_failed,
                    format_percent(value["pass_rate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- `legacy_200` contains EN2001a/EN2001b; `diverse_100` contains one sample from each of 100 other meetings.",
            "- AMI provides only positive speech examples here. Speech-event results are positive recall, not closed-world F1.",
            "- AMI has no usable GT in this manifest for music/noise, SNR/DNSMOS, room acoustics, far-field, or speaker profiles.",
            "- The legacy captioner output has no ASR, change count, overlap ratio, header, language, or lexical fields; these remain eligible misses where GT exists.",
            "- A field marked `interface unavailable` is an interface-coverage result, not a model-accuracy claim. Its e2e abstention metric is retained only to preserve the frozen denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_inputs(gt_rows, systems, universe, fields):
    expected_version = universe["universe_version"]
    gt_ids = set()
    for row in gt_rows:
        if row.get("field_universe_version") != expected_version:
            raise ValueError("ground truth field universe version mismatch")
        sample_id = row.get("sample_id")
        if sample_id in gt_ids:
            raise ValueError("duplicate ground-truth sample_id %s" % sample_id)
        gt_ids.add(sample_id)
        ground_truth = row.get("ground_truth", {})
        if set(ground_truth.get("values", {})) != set(fields):
            raise ValueError("%s has an incomplete ground-truth field universe" % sample_id)
        expected = sorted(
            path for path, enabled in ground_truth.get("gt_mask", {}).items() if enabled
        )
        if expected != ground_truth.get("evaluable_fields"):
            raise ValueError("%s gt_mask/evaluable_fields mismatch" % sample_id)
    for system, prediction_map in systems.items():
        unexpected = sorted(set(prediction_map) - gt_ids)
        if unexpected:
            raise ValueError("%s has unexpected prediction IDs: %s" % (system, unexpected[:10]))
        for sample_id, row in prediction_map.items():
            if row.get("system") != system:
                raise ValueError("%s prediction %s has a system mismatch" % (system, sample_id))
            if row.get("field_universe_version") != expected_version:
                raise ValueError("%s prediction %s has universe mismatch" % (system, sample_id))
            if set(row.get("values", {})) != set(fields):
                raise ValueError("%s prediction %s has incomplete values" % (system, sample_id))


def score(ground_truth_path, prediction_paths, universe_path, output_dir):
    universe = load_json(universe_path)
    fields = field_specs(universe)
    gt_rows = load_jsonl(ground_truth_path)
    systems = {}
    source_metadata = {}
    for system, path in prediction_paths.items():
        rows = load_jsonl(path)
        systems[system] = keyed_rows(rows, "%s predictions" % system)
        source_metadata[system] = {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "row_count": len(rows),
        }
    validate_inputs(gt_rows, systems, universe, fields)

    dataset_splits = {}
    for split_name in ("overall", "legacy_200", "diverse_100"):
        selected = [
            row
            for row in gt_rows
            if split_name == "overall" or row.get("split") == split_name
        ]
        dataset_splits[split_name] = {
            "sample_count": len(selected),
            "group_count": len(set(row["group_id"] for row in selected)),
        }
    metrics = {
        "schema_version": "structural-evaluation-metrics-v0.1",
        "field_universe_version": universe["universe_version"],
        "ground_truth": {
            "path": str(Path(ground_truth_path).resolve()),
            "sha256": sha256_file(ground_truth_path),
            "row_count": len(gt_rows),
        },
        "dataset": {"name": "AMI", "splits": dataset_splits},
        "prediction_sources": source_metadata,
        "systems": {},
        "denominator_policy": {
            "classification": "null/missing/invalid/failed remain errors in eligible_gt",
            "continuous": "errors on known values plus valid_prediction_rate",
            "asr": "null/missing/invalid/failed become empty hypotheses",
        },
    }
    for system, prediction_map in systems.items():
        split_metrics = {}
        for split_name in ("overall", "legacy_200", "diverse_100"):
            items = split_items(gt_rows, prediction_map, split_name)
            split_metrics[split_name] = score_split(items, universe, fields)
        metrics["systems"][system] = {"splits": split_metrics}

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "per_sample.jsonl",
        per_sample_rows(gt_rows, systems, fields, universe),
    )
    atomic_write_text(output_dir / "report.md", render_report(metrics, universe, fields))
    return metrics


def parse_prediction_arguments(values):
    predictions = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction must use SYSTEM=PATH")
        system, path = value.split("=", 1)
        system = system.strip()
        path = path.strip()
        if not system or not path:
            raise ValueError("--prediction must use SYSTEM=PATH")
        if system in predictions:
            raise ValueError("duplicate --prediction system %s" % system)
        predictions[system] = path
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--field-universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prediction", action="append", default=[])
    args = parser.parse_args()
    prediction_values = args.prediction or [
        "sure-tagger=%s"
        % (DEFAULT_OUTPUT_DIR / "sure-tagger.predictions.jsonl"),
        "captioner=%s" % (DEFAULT_OUTPUT_DIR / "captioner.predictions.jsonl"),
    ]
    prediction_paths = parse_prediction_arguments(prediction_values)
    metrics = score(
        args.ground_truth,
        prediction_paths,
        args.field_universe,
        args.output_dir,
    )
    summary = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "sample_count": metrics["dataset"]["splits"]["overall"]["sample_count"],
        "systems": sorted(metrics["systems"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
