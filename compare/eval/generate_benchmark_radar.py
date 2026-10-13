#!/usr/bin/env python3
"""Generate benchmark-v1 radar charts for structural and descriptive labels."""

import html
import json
import math
from pathlib import Path

import cairosvg


COMPARE_ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = COMPARE_ROOT / "evaluation" / "benchmark-v1"
OUTPUT_ROOT = COMPARE_ROOT / "docs" / "report" / "images"

SYSTEMS = (
    (
        "sure_tagger",
        COMPARE_ROOT / "docs" / "report" / "result" / "sure-tagger" / "score.json",
        "Sure-Tagger",
        "#77148b",
        "rgba(119, 20, 139, 0.12)",
    ),
    (
        "captioner",
        SCORE_ROOT / "captioner-deepseek-gender-20260911" / "score.json",
        "Qwen Captioner",
        "#4d68e8",
        "rgba(77, 104, 232, 0.10)",
    ),
    (
        "qwen3_omni",
        COMPARE_ROOT
        / "docs"
        / "report"
        / "result"
        / "qwen3-omni-v2-multi-api12"
        / "structural_score.json",
        "Qwen3-Omni Flash v2",
        "#d84c2f",
        "rgba(216, 76, 47, 0.10)",
    ),
)

DESCRIPTIVE_SOURCES = (
    (
        "sure_tagger",
        COMPARE_ROOT
        / "docs"
        / "report"
        / "result"
        / "descriptive"
        / "sure-tagger"
        / "summary.json",
    ),
    (
        "captioner",
        COMPARE_ROOT
        / "evaluation"
        / "descriptive-semantic-scoring-official-assisted-benchmark-qwen-captioner"
        / "summary.json",
    ),
    (
        "qwen3_omni",
        COMPARE_ROOT
        / "docs"
        / "report"
        / "result"
        / "descriptive"
        / "qwen3-omni-v2-multi-api12"
        / "score"
        / "summary.json",
    ),
)

LAYER_LABELS = {
    "layer_1": (
        ("speaker.asr_transcript", "ASR transcript"),
        ("language_content.language", "Language"),
        ("language_content.word_count", "Word count"),
        ("language_content.repetition.has_repetition", "Repetition flag"),
        ("language_content.repetition.repetition_count", "Repetition count"),
        ("deterministic_derived.filler_ratio", "Filler ratio"),
    ),
    "layer_2": (
        ("speaker.speaker_count", "Speaker count"),
        ("speaker.speaker_present", "Speaker present"),
        ("speaker.multi_speaker", "Multi-speaker"),
        ("speaker.speaker_overlap", "Speaker overlap"),
        ("speaker.gender", "Speaker gender"),
    ),
    "layer_3": (
        ("speaker.speaker_change_count", "Change count"),
        ("speaker.speaker_change", "Change flag"),
        ("speaker.overlap_ratio", "Overlap ratio"),
        ("deterministic_derived.speaker_change_rate_per_min", "Change rate/min"),
    ),
    "layer_5": (
        ("basic_acoustic.nonspeech_ratio", "Silence / non-speech ratio"),
        ("audio_quality.snr_db", "SNR (dB)"),
        ("room_acoustic.rt60_sec", "RT60 (s)"),
        ("room_acoustic.c50_db", "C50 (dB)"),
        ("sound_field_scene.speech_music_events", "Speech/music events"),
        ("sound_field_scene.music_present", "Music present"),
        ("sound_field_scene.external_noise_type", "External noise"),
    ),
}

# The current Sure-Tagger score calls this metric ``nonspeech_ratio`` while
# the legacy captioner score keeps the historical ``silence_ratio`` key.
FIELD_ALIASES = {
    "basic_acoustic.nonspeech_ratio": {
        "captioner": ("basic_acoustic.silence_ratio",),
    },
}

# Compatibility fallback for historical Qwen3-Omni scores that predate the
# speaker.gender axis. Current v1.0 scores supply the field directly.
NULL_BASELINE_FIELDS = {
    "speaker.gender": frozenset(("qwen3_omni",)),
}
NULL_BASELINE_SCORE = 0.5

LAYER_TITLES = {
    "layer_1": "Layer 1 - Lexical Content",
    "layer_2": "Layer 2 - Speaker",
    "layer_3": "Layer 3 - Temporal",
    "layer_5": "Layer 5 - Acoustic Scene & Events",
}

DESCRIPTIVE_LABELS = (
    ("background_condition", "Background condition"),
    ("background_sound_summary", "Background sound"),
    ("caption", "Caption"),
    ("clarity_intelligibility_impression", "Clarity / intelligibility"),
    ("descriptive_phrases", "Descriptive phrases"),
    ("dominant_content_impression", "Dominant content"),
    ("facets", "Facets"),
    ("scene_hypotheses", "Scene hypotheses"),
    ("speaker_descriptions", "Speaker descriptions"),
    ("topic", "Topic"),
)

ALL_LABEL_SHORT_NAMES = {
    "speaker.asr_transcript": "ASR",
    "language_content.language": "Language",
    "language_content.word_count": "Words",
    "language_content.repetition.has_repetition": "Repeat flag",
    "language_content.repetition.repetition_count": "Repeat count",
    "deterministic_derived.filler_ratio": "Filler ratio",
    "speaker.speaker_count": "Count",
    "speaker.speaker_present": "Present",
    "speaker.multi_speaker": "Multi",
    "speaker.speaker_overlap": "Overlap",
    "speaker.gender": "Gender",
    "speaker.speaker_change_count": "Change count",
    "speaker.speaker_change": "Change flag",
    "speaker.overlap_ratio": "Overlap ratio",
    "deterministic_derived.speaker_change_rate_per_min": "Change/min",
    "basic_acoustic.nonspeech_ratio": "Silence / non-speech",
    "sound_field_scene.speech_music_events": "Speech/music events",
    "sound_field_scene.music_present": "Music",
    "sound_field_scene.external_noise_type": "External noise",
}

DESCRIPTIVE_SHORT_NAMES = {
    "background_condition": "Background",
    "background_sound_summary": "Sound summary",
    "caption": "Caption",
    "clarity_intelligibility_impression": "Clarity",
    "descriptive_phrases": "Phrases",
    "dominant_content_impression": "Dominant content",
    "facets": "Facets",
    "scene_hypotheses": "Scene hypotheses",
    "speaker_descriptions": "Speaker descriptions",
    "topic": "Topic",
}


def load_scores():
    scores = {}
    for key, path, _, _, _ in SYSTEMS:
        with path.open("r", encoding="utf-8") as source:
            score = json.load(source)
        if score.get("descriptive_fields_scored") is not False:
            raise ValueError("radar input must exclude descriptive scores: %s" % path)
        scores[key] = score
    return scores


def load_descriptive_scores():
    scores = {}
    for key, path in DESCRIPTIVE_SOURCES:
        with path.open("r", encoding="utf-8") as source:
            score = json.load(source)
        if score.get("status") not in {
            "complete_with_missing_ground_truth",
            "waiting_for_ground_truth",
        }:
            raise ValueError("descriptive input is not complete: %s" % path)
        if score.get("sample_count") != 650 or score.get("scoring_success_count") != 648:
            raise ValueError("descriptive input must cover 650/648 samples: %s" % path)
        if not isinstance(score.get("field_mean_scores_on_successes"), dict):
            raise ValueError("descriptive input has no field means: %s" % path)
        scores[key] = score
    return scores


def field_score(scores, key, layer_id, field):
    labels = scores[key]["layer_summaries"].get(layer_id, {}).get("labels", {})
    candidates = (field,) + tuple(FIELD_ALIASES.get(field, {}).get(key, ()))
    for candidate in candidates:
        value = labels.get(candidate, {}).get("score_rate")
        if isinstance(value, (int, float)):
            return value
    if key in NULL_BASELINE_FIELDS.get(field, ()):
        return NULL_BASELINE_SCORE
    return None


def scored_labels(scores, layer_id):
    """Keep labels with a score or an explicit report-only null baseline."""
    return tuple(
        (field, label)
        for field, label in LAYER_LABELS[layer_id]
        if all(
            field_score(scores, key, layer_id, field) is not None
            for key, _, _, _, _ in SYSTEMS
        )
    )


def value_map(scores, layer_id, labels):
    return {
        key: [
            100.0 * field_score(scores, key, layer_id, field)
            for field, _ in labels
        ]
        for key, _, _, _, _ in SYSTEMS
    }


def descriptive_field_score(scores, key, field):
    value = scores[key]["field_mean_scores_on_successes"].get(field)
    if isinstance(value, (int, float)):
        # DeepSeek descriptive scoring is on a 0-2 scale; keep the radar on 0-100.
        return value / 2.0
    return None


def scored_descriptive_labels(scores):
    return tuple(
        (field, label)
        for field, label in DESCRIPTIVE_LABELS
        if all(descriptive_field_score(scores, key, field) is not None for key, _ in DESCRIPTIVE_SOURCES)
    )


def descriptive_value_map(scores, labels):
    return {
        key: [100.0 * descriptive_field_score(scores, key, field) for field, _ in labels]
        for key, _ in DESCRIPTIVE_SOURCES
    }


def esc(value):
    return html.escape(str(value), quote=True)


def points_for(cx, cy, radius, values, angles, scale=100.0):
    return [
        (
            cx + radius * value / scale * math.cos(angle),
            cy + radius * value / scale * math.sin(angle),
        )
        for value, angle in zip(values, angles)
    ]


def to_rps(values):
    """Convert absolute scores to relative performance scores in [0, 1].

    For each axis, RPS = current / best among the compared systems. All
    benchmark axes used here are higher-is-better, so no reciprocal is needed.
    """
    keys = list(values)
    if not keys:
        return values
    n = len(values[keys[0]])
    rps = {key: [] for key in keys}
    for index in range(n):
        best = max(values[key][index] for key in keys)
        for key in keys:
            current = values[key][index]
            rps[key].append(0.0 if best <= 0 else current / best)
    return rps


def point_string(points):
    return " ".join("%.2f,%.2f" % point for point in points)


def text_anchor(angle):
    cosine = math.cos(angle)
    if cosine > 0.18:
        return "start"
    if cosine < -0.18:
        return "end"
    return "middle"


def radar_svg(title, labels, values, output_width=1400, output_height=1100, mode="absolute"):
    width = output_width
    height = output_height
    cx = width / 2.0
    cy = 530.0
    radius = 350.0 if len(labels) <= 9 else 365.0
    angles = [(-math.pi / 2.0) + 2.0 * math.pi * i / len(labels) for i in range(len(labels))]
    if mode == "rps":
        scale = 1.0
        rings = (0.2, 0.4, 0.6, 0.8, 1.0)
        subtitle = "Relative Performance Score (current / best per axis)"
        footnote = "RPS = current / best among the three systems; 1.00 is the best system on that axis."
        value_fmt = "%.2f"
        ring_fmt = "%.1f"
    else:
        scale = 100.0
        rings = (20, 40, 60, 80, 100)
        subtitle = "Benchmark-v1 label scores (%)"
        footnote = "Higher is better; explicit null receives half credit (50%) under the benchmark rule."
        value_fmt = "%.1f"
        ring_fmt = "%d"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;letter-spacing:0px} .axis{font-size:15px;fill:#111} .value{font-size:12px;font-weight:600} .ring-label{font-size:12px;fill:#555}</style>',
        '<text x="%0.1f" y="52" text-anchor="middle" font-size="28" font-weight="600" fill="#111">%s</text>' % (cx, esc(title)),
        '<text x="%0.1f" y="80" text-anchor="middle" font-size="15" fill="#666">%s</text>' % (cx, esc(subtitle)),
    ]

    for level in rings:
        ring = [
            (cx + radius * level / scale * math.cos(angle), cy + radius * level / scale * math.sin(angle))
            for angle in angles
        ]
        parts.append('<polygon points="%s" fill="none" stroke="#a8a8a8" stroke-width="1.2"/>' % point_string(ring))
        parts.append('<text class="ring-label" x="%0.2f" y="%0.2f">%s</text>' % (cx + 7, cy - radius * level / scale + 4, ring_fmt % level))

    for angle in angles:
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        parts.append('<line x1="%0.2f" y1="%0.2f" x2="%0.2f" y2="%0.2f" stroke="#b0b0b0" stroke-width="1.1"/>' % (cx, cy, x, y))

    series_points = {}
    for (key, _, name, stroke, fill), series_values in zip(SYSTEMS, values.values()):
        points = points_for(cx, cy, radius, series_values, angles, scale=scale)
        series_points[key] = points
        parts.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="3" stroke-linejoin="round"/>' % (point_string(points), fill, stroke))
        for x, y in points:
            parts.append('<circle cx="%0.2f" cy="%0.2f" r="6" fill="%s" stroke="white" stroke-width="1.5"/>' % (x, y, stroke))

    label_radius = radius + (57 if len(labels) <= 9 else 48)
    for index, (_, label) in enumerate(labels):
        angle = angles[index]
        x = cx + label_radius * math.cos(angle)
        y = cy + label_radius * math.sin(angle)
        anchor = text_anchor(angle)
        parts.append('<text class="axis" x="%0.2f" y="%0.2f" text-anchor="%s" dominant-baseline="middle">%s</text>' % (x, y, anchor, esc(label)))

    # Point values are placed just outside each marker, with a small perpendicular offset
    # so the two systems remain distinguishable even when their scores are close.
    if len(labels) <= 9:
        offsets = (-13.0, 0.0, 13.0)
        for series_index, (key, _, _, _, _) in enumerate(SYSTEMS):
            series_values = values[key]
            for index, (point, value) in enumerate(zip(series_points[key], series_values)):
                angle = angles[index]
                offset = offsets[series_index]
                x = point[0] - offset * math.sin(angle)
                y = point[1] + offset * math.cos(angle)
                anchor = text_anchor(angle)
                parts.append('<text class="value" x="%0.2f" y="%0.2f" text-anchor="%s" dominant-baseline="middle" fill="#222" stroke="white" stroke-width="1.5" paint-order="stroke">%s</text>' % (x, y, anchor, value_fmt % value))

    legend_y = height - 92
    legend_spacing = 350
    legend_x = cx - legend_spacing * (len(SYSTEMS) - 1) / 2.0 - 115
    for index, (_, _, name, stroke, _) in enumerate(SYSTEMS):
        x = legend_x + index * legend_spacing
        parts.append('<line x1="%0.2f" y1="%0.2f" x2="%0.2f" y2="%0.2f" stroke="%s" stroke-width="4"/>' % (x, legend_y, x + 38, legend_y, stroke))
        parts.append('<circle cx="%0.2f" cy="%0.2f" r="6" fill="%s"/>' % (x + 19, legend_y, stroke))
        parts.append('<text x="%0.2f" y="%0.2f" font-size="17" fill="#111" dominant-baseline="middle">%s</text>' % (x + 51, legend_y, esc(name)))
    parts.append('<text x="%0.1f" y="%0.1f" text-anchor="middle" font-size="13" fill="#777">%s</text>' % (cx, height - 38, esc(footnote)))
    parts.append('</svg>')
    return "".join(parts)


def write_chart(filename, title, labels, values, mode="absolute"):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_ROOT / (filename + ".svg")
    png_path = OUTPUT_ROOT / (filename + ".png")
    chart_title = title if mode != "rps" else title + " · RPS"
    svg_path.write_text(radar_svg(chart_title, labels, values, mode=mode), encoding="utf-8")
    cairosvg.svg2png(bytestring=svg_path.read_bytes(), write_to=str(png_path), output_width=1400, output_height=1100)
    return png_path, svg_path


def write_chart_pair(filename, title, labels, values):
    generated = [write_chart(filename, title, labels, values, mode="absolute")]
    generated.append(write_chart(filename + "_rps", title, labels, to_rps(values), mode="rps"))
    return generated


def main():
    scores = load_scores()
    descriptive_scores = load_descriptive_scores()
    generated = []
    for layer_id in ("layer_1", "layer_2", "layer_3", "layer_5"):
        labels = scored_labels(scores, layer_id)
        if not labels:
            continue
        values = value_map(scores, layer_id, labels)
        generated.extend(write_chart_pair("benchmark_radar_%s" % layer_id, LAYER_TITLES[layer_id], labels, values))

    structural_total = []
    for layer_id in ("layer_1", "layer_2", "layer_3", "layer_5"):
        structural_total.extend(
            (field, "L%d: %s" % (
                int(layer_id.rsplit("_", 1)[1]),
                ALL_LABEL_SHORT_NAMES.get(field, label),
            ))
            for field, label in scored_labels(scores, layer_id)
        )
    structural_total_values = {
        key: [
            100.0 * field_score(scores, key, layer_id, field)
            for layer_id in ("layer_1", "layer_2", "layer_3", "layer_5")
            for field, _ in scored_labels(scores, layer_id)
        ]
        for key, _, _, _, _ in SYSTEMS
    }
    generated.extend(
        write_chart_pair(
            "benchmark_radar_all_labels",
            "All Scored Structural Labels",
            tuple(structural_total),
            structural_total_values,
        )
    )

    descriptive_labels = scored_descriptive_labels(descriptive_scores)
    descriptive_values = descriptive_value_map(descriptive_scores, descriptive_labels)
    generated.extend(
        write_chart_pair(
            "benchmark_radar_descriptive_labels",
            "All Scored Descriptive Labels",
            descriptive_labels,
            descriptive_values,
        )
    )

    total = tuple(structural_total) + tuple(
        (field, "D: %s" % DESCRIPTIVE_SHORT_NAMES.get(field, label))
        for field, label in descriptive_labels
    )
    total_values = {
        key: structural_total_values[key] + descriptive_values[key]
        for key, _, _, _, _ in SYSTEMS
    }
    generated.extend(
        write_chart_pair("benchmark_radar_total_labels", "All Structural + Descriptive Labels", total, total_values)
    )
    for png_path, svg_path in generated:
        print("%s\n%s" % (png_path, svg_path))


if __name__ == "__main__":
    main()
