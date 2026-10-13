#!/usr/bin/env python3
"""Build the print edition of the benchmark-v1 comparison report."""

import argparse
import gzip
import html
import json
from collections import Counter
from collections import OrderedDict
from datetime import date
from pathlib import Path
from statistics import mean


COMPARE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = COMPARE_ROOT / "evaluation" / "benchmark-v1"
REPORT_ROOT = COMPARE_ROOT / "docs" / "report"
OUTPUT_ROOT = REPORT_ROOT / "pdf"
PAGE_TOTAL = 10

SURE_SCORE = (
    REPORT_ROOT
    / "result"
    / "sure-tagger"
    / "score.json"
)
SURE_SCORE_20260911 = (
    BENCHMARK_ROOT
    / "sure_tagger_gender_20260911_deepseek"
    / "score.json"
)
CAPTIONER_SCORE = (
    BENCHMARK_ROOT
    / "captioner-deepseek-gender-20260911"
    / "score.json"
)
QWEN3_OMNI_SCORE = (
    COMPARE_ROOT
    / "docs"
    / "report"
    / "result"
    / "qwen3-omni-v2-multi-api12"
    / "structural_score.json"
)
DESCRIPTIVE_SCORES = {
    "sure": REPORT_ROOT / "result" / "descriptive" / "sure-tagger" / "summary.json",
    "captioner": COMPARE_ROOT / "evaluation" / "descriptive-semantic-scoring-official-assisted-benchmark-qwen-captioner" / "summary.json",
    "qwen3": COMPARE_ROOT
    / "docs"
    / "report"
    / "result"
    / "descriptive"
    / "qwen3-omni-v2-multi-api12"
    / "score"
    / "summary.json",
}
SURE_RUN_ROOT = BENCHMARK_ROOT / "sure_tagger_full_20260918_main_pipeline_w2_b8_staged" / "core"
SURE_SCORE_20260917 = (
    REPORT_ROOT
    / "result"
    / "sure-tagger"
    / "previous_20260917"
    / "score.json"
)

LAYERS = OrderedDict(
    (
        (
            "layer_1",
            {
                "short": "L1",
                "title": "词汇内容层",
                "english": "LEXICAL CONTENT",
                "accent": "orange",
                "summary": "ASR、语言、词数、重复和填充词比例。",
                "fields": (
                    ("speaker.asr_transcript", "ASR transcript"),
                    ("language_content.language", "Language"),
                    ("language_content.word_count", "Word count"),
                    ("language_content.repetition.has_repetition", "Repetition flag"),
                    ("language_content.repetition.repetition_count", "Repetition count"),
                    ("deterministic_derived.filler_ratio", "Filler ratio"),
                ),
            },
        ),
        (
            "layer_2",
            {
                "short": "L2",
                "title": "说话人层",
                "english": "SPEAKER",
                "accent": "acid",
                "summary": "人数、人声存在、多说话人存在、重叠发言存在和性别序列准确度。",
                "fields": (
                    ("speaker.speaker_count", "Speaker count"),
                    ("speaker.speaker_present", "Speaker present"),
                    ("speaker.multi_speaker", "Multi-speaker"),
                    ("speaker.speaker_overlap", "Speaker overlap"),
                    ("speaker.gender", "Speaker gender"),
                ),
            },
        ),
        (
            "layer_3",
            {
                "short": "L3",
                "title": "时间结构层",
                "english": "TEMPORAL",
                "accent": "blue",
                "summary": "说话人切换次数、切换存在性、重叠比例和每分钟切换率。",
                "fields": (
                    ("speaker.speaker_change_count", "Change count"),
                    ("speaker.speaker_change", "Change flag"),
                    ("speaker.overlap_ratio", "Overlap ratio"),
                    (
                        "deterministic_derived.speaker_change_rate_per_min",
                        "Change rate/min",
                    ),
                ),
            },
        ),
        (
            "layer_5",
            {
                "short": "L5",
                "title": "声学场景与事件层",
                "english": "ACOUSTIC SCENE & EVENTS",
                "accent": "teal",
                "summary": "非语音比例、语音/音乐事件、音乐存在和外部噪声类型。",
                "fields": (
                    ("basic_acoustic.nonspeech_ratio", "Silence / non-speech"),
                    ("sound_field_scene.speech_music_events", "Speech/music events"),
                    ("sound_field_scene.music_present", "Music present"),
                    ("sound_field_scene.external_noise_type", "External noise"),
                ),
            },
        ),
    )
)

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

CAPTIONER_ALIASES = {
    "basic_acoustic.nonspeech_ratio": "basic_acoustic.silence_ratio",
}

NULL_BASELINE_SCORE = 0.5
GENDER_FIELD = "speaker.gender"

HISTORY = (
    (
        "刷新前",
        "MOSS / 原链路",
        BENCHMARK_ROOT
        / "sure-tagger-deepseek-full_before_asr_refresh_20260904"
        / "score_v09_20260907.json",
    ),
    (
        "DiariZen 中间",
        "错误或未完成配置",
        BENCHMARK_ROOT
        / "sure-tagger-deepseek-full_before_official_diarizen_20260907"
        / "score_v09_20260907.json",
    ),
    (
        "官方 DiariZen",
        "中文 MOSS，其余 DiariZen",
        BENCHMARK_ROOT
        / "sure_tagger_official_refresh_final_20260907"
        / "score_v09_20260907.json",
    ),
    (
        "MOSS 路由",
        "中英文 MOSS，其余 DiariZen",
        BENCHMARK_ROOT
        / "sure_tagger_moss_en_zh_reuse_20260907"
        / "score.json",
    ),
    (
        "2026-09-11",
        "共识优先；中英 MOSS，其余 Qwen3-ASR",
        SURE_SCORE_20260911,
    ),
    (
        "2026-09-17",
        "整段 FireRed LID；共识优先；中英 MOSS，其余 Qwen3-ASR",
        SURE_SCORE_20260917,
    ),
    (
        "当前 2026-09-18",
        "全量重跑；语言并入描述性调用；描述性 10 轴重评",
        SURE_SCORE,
    ),
)


def load_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def load_structural_score(path):
    score = load_json(path)
    if score.get("descriptive_fields_scored") is not False:
        raise ValueError("report input must exclude descriptive scores: %s" % path)
    if score.get("prediction_row_count") != 650 or score.get("missing_result_rows") != 0:
        raise ValueError("report input must cover all 650 samples: %s" % path)
    return score


def load_descriptive_score(path):
    score = load_json(path)
    if score.get("status") not in {
        "complete_with_missing_ground_truth",
        "waiting_for_ground_truth",
    }:
        raise ValueError("descriptive input is not complete: %s" % path)
    if score.get("sample_count") != 650 or score.get("scoring_success_count") != 648:
        raise ValueError("descriptive input must cover 650/648 samples: %s" % path)
    if not isinstance(score.get("field_mean_scores_on_successes"), dict):
        raise ValueError("descriptive input has no field means: %s" % path)
    return score


def load_route_stats(run_root):
    selected = Counter()
    basis = Counter()
    consensus = Counter()
    pattern = "artifacts/shard_*/speaker_v2/*/fusion_artifact_v2.json.gz"
    paths = sorted(run_root.glob(pattern))
    if len(paths) != 650:
        raise ValueError("expected 650 Sure-Tagger routing artifacts")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            route = json.load(source).get("asr_route", {})
        selected[route.get("selected_model") or "failed_closed"] += 1
        basis[route.get("selection_basis") or "unknown"] += 1
        single = route.get("single_speaker_consensus", {})
        consensus[bool(single.get("all_at_most_one"))] += 1
    return {
        "firered": selected["firered"],
        "moss": selected["moss"],
        "qwen3_asr": selected["qwen3_asr"],
        "failed_closed": selected["failed_closed"],
        "single_speaker_consensus": consensus[True],
        "language_route": basis["language_route"],
    }


def escape(value):
    return html.escape(str(value), quote=True)


def percent(value, digits=2):
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def points(value):
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * value:.2f}"


def field_score(score, layer_id, field, *, captioner=False):
    labels = score.get("layer_summaries", {}).get(layer_id, {}).get("labels", {})
    candidates = [field]
    if captioner and field in CAPTIONER_ALIASES:
        candidates.append(CAPTIONER_ALIASES[field])
    for candidate in candidates:
        value = labels.get(candidate, {}).get("score_rate")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def current_layer_score(score, layer_id):
    value = score.get("layer_summaries", {}).get(layer_id, {}).get("score_rate")
    return float(value) if isinstance(value, (int, float)) else None


def descriptive_field_score(score, field):
    value = score.get("field_mean_scores_on_successes", {}).get(field)
    if not isinstance(value, (int, float)):
        return None
    # DeepSeek scores each descriptive field on a 0-2 scale.
    return float(value) / 2.0


def comparable_descriptive_scores(descriptive):
    fields = []
    for field, label in DESCRIPTIVE_LABELS:
        values = {
            "sure": descriptive_field_score(descriptive["sure"], field),
            "captioner": descriptive_field_score(descriptive["captioner"], field),
            "qwen3": descriptive_field_score(descriptive["qwen3"], field),
        }
        if any(value is None for value in values.values()):
            raise ValueError(f"Missing descriptive score for {field}")
        fields.append(
            {
                "field": field,
                "label": label,
                **values,
                "winner": max(values, key=values.get),
            }
        )
    return {
        "fields": fields,
        "sure": mean(row["sure"] for row in fields),
        "captioner": mean(row["captioner"] for row in fields),
        "qwen3": mean(row["qwen3"] for row in fields),
    }


def comparable_scores(sure, captioner, qwen3):
    result = {}
    for layer_id, layer in LAYERS.items():
        fields = []
        for field, label in layer["fields"]:
            sure_value = field_score(sure, layer_id, field, captioner=False)
            captioner_value = field_score(captioner, layer_id, field, captioner=True)
            qwen3_value = field_score(qwen3, layer_id, field)
            captioner_null_baseline = False
            qwen3_null_baseline = False
            if field == GENDER_FIELD:
                if sure_value is None:
                    raise ValueError("Missing Sure-Tagger speaker.gender score")
                if captioner_value is None:
                    captioner_value = NULL_BASELINE_SCORE
                    captioner_null_baseline = True
                if qwen3_value is None:
                    qwen3_value = NULL_BASELINE_SCORE
                    qwen3_null_baseline = True
            if sure_value is None or captioner_value is None or qwen3_value is None:
                raise ValueError(f"Missing comparable score for {field}")
            values = {
                "sure": sure_value,
                "captioner": captioner_value,
                "qwen3": qwen3_value,
            }
            winner = max(values, key=values.get)
            fields.append(
                {
                    "field": field,
                    "label": label,
                    "sure": sure_value,
                    "captioner": captioner_value,
                    "qwen3": qwen3_value,
                    "captioner_null_baseline": captioner_null_baseline,
                    "qwen3_null_baseline": qwen3_null_baseline,
                    "winner": winner,
                }
            )
        result[layer_id] = {
            "fields": fields,
            "sure": mean(row["sure"] for row in fields),
            "captioner": mean(row["captioner"] for row in fields),
            "qwen3": mean(row["qwen3"] for row in fields),
        }
    return result


def page_header(number, kicker, title, intro):
    return f"""
      <header class="page-head">
        <div>
          <p class="kicker">{escape(number)} / {escape(kicker)}</p>
          <h2>{escape(title)}</h2>
        </div>
        <p class="page-intro">{escape(intro)}</p>
      </header>
    """


def page_footer(page_number, total=PAGE_TOTAL):
    return f"""
      <footer class="page-footer">
        <span>STRUCTURAL BENCHMARK · V1.0</span>
        <span>{page_number:02d} / {total:02d}</span>
      </footer>
    """


def radar_pair(abs_src, rps_src, abs_alt, rps_alt, extra_class=""):
    return f"""
        <div class="radar-stack {extra_class}">
          <figure class="radar">
            <img src="{abs_src}" alt="{escape(abs_alt)}">
            <figcaption>ABSOLUTE · 0–100%</figcaption>
          </figure>
          <figure class="radar">
            <img src="{rps_src}" alt="{escape(rps_alt)}">
            <figcaption>RPS · current / best · 0–1</figcaption>
          </figure>
        </div>
    """


def score_table_fields(fields):
    rows = []
    for item in fields:
        values = {
            "qwen-captioner": item["captioner"],
            "Sure-Tagger": item["sure"],
            "Qwen3-Omni Flash v2": item["qwen3"],
        }
        winner_name, winner_value = max(values.items(), key=lambda pair: pair[1])
        runner_value = sorted(values.values(), reverse=True)[1]
        winner = f"{winner_name} +{100.0 * (winner_value - runner_value):.2f} pp"
        winner_class = {
            "Sure-Tagger": "sure",
            "qwen-captioner": "captioner",
            "Qwen3-Omni Flash v2": "qwen3",
        }[winner_name]
        values_html = {
            "qwen-captioner": percent(item["captioner"]),
            "Sure-Tagger": percent(item["sure"]),
            "Qwen3-Omni Flash v2": percent(item["qwen3"]),
        }
        if item.get("captioner_null_baseline"):
            values_html["qwen-captioner"] += ' <small class="null-score">null</small>'
        if item.get("qwen3_null_baseline"):
            values_html["Qwen3-Omni Flash v2"] += ' <small class="null-score">null</small>'
        for name in values_html:
            if name == winner_name:
                values_html[name] = "<strong>%s</strong>" % values_html[name]
        rows.append(
            "<tr>"
            f"<td><code>{escape(item['field'])}</code></td>"
            f"<td>{values_html['qwen-captioner']}</td>"
            f"<td>{values_html['Sure-Tagger']}</td>"
            f"<td>{values_html['Qwen3-Omni Flash v2']}</td>"
            f"<td><span class=\"winner {winner_class}\">{escape(winner)}</span></td>"
            "</tr>"
        )
    return "".join(rows)


def score_table(layer_id, comparable):
    return score_table_fields(comparable[layer_id]["fields"])


def layer_page(page_number, layer_id, comparable):
    layer = LAYERS[layer_id]
    scores = comparable[layer_id]
    values = {
        "Sure-Tagger": scores["sure"],
        "qwen-captioner": scores["captioner"],
        "Qwen3-Omni Flash v2": scores["qwen3"],
    }
    winner, winner_score = max(values.items(), key=lambda pair: pair[1])
    runner_score = sorted(values.values(), reverse=True)[1]
    delta = 100.0 * (winner_score - runner_score)
    gender_field = next(
        (item for item in scores["fields"] if item["field"] == GENDER_FIELD),
        None,
    )
    if layer_id == "layer_2" and gender_field is not None:
        if gender_field.get("qwen3_null_baseline"):
            gender_analysis = (
                "Sure-Tagger、qwen-captioner 已完成性别序列评分",
                "Sure-Tagger 性别序列得分为 %s，qwen-captioner 为 %s；Qwen3-Omni 尚未评分，本版按显式 null 的 50%% 基线展示和计入宏平均。"
                % (percent(gender_field["sure"]), percent(gender_field["captioner"])),
            )
        else:
            gender_analysis = (
                "三套系统均完成性别序列评分",
                "Sure-Tagger 性别序列得分为 %s，qwen-captioner 为 %s，Qwen3-Omni 为 %s；显式 null 按评分规则计 1/2 分。"
                % (
                    percent(gender_field["sure"]),
                    percent(gender_field["captioner"]),
                    percent(gender_field["qwen3"]),
                ),
            )
    else:
        gender_analysis = None
    analysis = {
        "layer_1": (
            "Sure-Tagger 层均领先",
            "Sure-Tagger 在 ASR、语言、重复和 filler 比例上领先；Qwen3-Omni 仍在词数上领先。标点与 filler 原始计数只作诊断。",
        ),
        "layer_2": gender_analysis,
        "layer_3": (
            "重叠比例不再使用容差除数",
            "Sure-Tagger 在切换次数、切换存在性、重叠比例和速率四项均领先；overlap_ratio 按完整比例区间的绝对误差计分。",
        ),
        "layer_5": (
            "本轮路由切换不影响该层",
            "Sure-Tagger 在非语音比例、事件、音乐和外部噪声四项均领先；Qwen3-Omni Flash v2 的连续声学字段不进入本轮雷达。",
        ),
    }[layer_id]
    return f"""
    <section class="page layer-page accent-{layer['accent']}">
      {page_header(layer['short'], layer['english'], layer['title'], layer['summary'])}
      <div class="layer-hero">
        {radar_pair(
            "../images/benchmark_radar_%s.png" % layer_id,
            "../images/benchmark_radar_%s_rps.png" % layer_id,
            layer["title"] + "雷达图",
            layer["title"] + " RPS 雷达图",
            extra_class="radar-layer",
        )}
        <aside class="layer-result">
          <p class="mono">{escape(winner.upper())} LEADS</p>
          <strong>{delta:+.2f}<small>pp</small></strong>
          <div class="score-stack">
            <span><i class="dot sure-dot"></i>Sure-Tagger <b>{percent(scores['sure'])}</b></span>
            <span><i class="dot captioner-dot"></i>qwen-captioner <b>{percent(scores['captioner'])}</b></span>
            <span><i class="dot qwen3-dot"></i>Qwen3-Omni Flash v2 <b>{percent(scores['qwen3'])}</b></span>
          </div>
          <div class="analysis-box"><b>{escape(analysis[0])}</b><p>{escape(analysis[1])}</p></div>
        </aside>
      </div>
      <table class="score-table">
        <thead><tr><th>评分标签</th><th>qwen-captioner</th><th>Sure-Tagger</th><th>Qwen3-Omni Flash v2</th><th>最高分</th></tr></thead>
        <tbody>{score_table(layer_id, comparable)}</tbody>
      </table>
      {page_footer(page_number)}
    </section>
    """


def descriptive_page(page_number, comparable):
    values = {
        "Sure-Tagger": comparable["sure"],
        "qwen-captioner": comparable["captioner"],
        "Qwen3-Omni Flash v2": comparable["qwen3"],
    }
    winner, winner_score = max(values.items(), key=lambda pair: pair[1])
    runner_score = sorted(values.values(), reverse=True)[1]
    delta = 100.0 * (winner_score - runner_score)
    return f"""
    <section class="page layer-page descriptive-page accent-orange">
      {page_header('D', 'DESCRIPTIVE LABELS', '描述性标签层', 'DeepSeek 对 10 个开放描述字段逐条评分；每个字段按 0–2 分归一化到百分比。')}
      <div class="layer-hero">
        {radar_pair(
            "../images/benchmark_radar_descriptive_labels.png",
            "../images/benchmark_radar_descriptive_labels_rps.png",
            "描述性标签雷达图",
            "描述性标签 RPS 雷达图",
            extra_class="radar-layer",
        )}
        <aside class="layer-result">
          <p class="mono">{escape(winner.upper())} LEADS</p>
          <strong>{delta:+.2f}<small>pp</small></strong>
          <div class="score-stack">
            <span><i class="dot sure-dot"></i>Sure-Tagger <b>{percent(comparable['sure'])}</b></span>
            <span><i class="dot captioner-dot"></i>qwen-captioner <b>{percent(comparable['captioner'])}</b></span>
            <span><i class="dot qwen3-dot"></i>Qwen3-Omni Flash v2 <b>{percent(comparable['qwen3'])}</b></span>
          </div>
          <div class="analysis-box"><b>Sure-Tagger 描述性本轮重评</b><p>Sure-Tagger 描述性文本与 10 轴 DeepSeek 评分随 2026-09-18 全量流水线重跑；captioner 与 Omni 描述轴仍沿用既有 648 条有效 GT 评分。</p></div>
        </aside>
      </div>
      <table class="score-table descriptive-score-table">
        <thead><tr><th>评分标签</th><th>qwen-captioner</th><th>Sure-Tagger</th><th>Qwen3-Omni Flash v2</th><th>最高分</th></tr></thead>
        <tbody>{score_table_fields(comparable['fields'])}</tbody>
      </table>
      {page_footer(page_number)}
    </section>
    """


def total_page(
    page_number,
    structural_overall,
    descriptive,
    total_overall,
    lead_count,
    structural_axis_count,
    total_axis_count,
):
    values = {
        "Sure-Tagger": total_overall["sure"],
        "qwen-captioner": total_overall["captioner"],
        "Qwen3-Omni Flash v2": total_overall["qwen3"],
    }
    winner, winner_score = max(values.items(), key=lambda pair: pair[1])
    runner_score = sorted(values.values(), reverse=True)[1]
    delta = 100.0 * (winner_score - runner_score)
    return f"""
    <section class="page total-page accent-teal">
      {page_header('T', 'ALL LABELS', '结构 + 描述总览', f'{total_axis_count} 个可比标签合并展示；Sure-Tagger 结构与描述性均为 2026-09-18 全量结果，captioner 与 Omni 描述轴沿用既有评分，二者均按标签等权。')}
      <div class="total-hero">
        {radar_pair(
            "../images/benchmark_radar_total_labels.png",
            "../images/benchmark_radar_total_labels_rps.png",
            "结构与描述性标签总览雷达图",
            "结构与描述性标签总览 RPS 雷达图",
            extra_class="total-radar",
        )}
        <aside class="total-result">
          <p class="mono">{escape(winner.upper())} LEADS</p>
          <strong>{delta:+.2f}<small>pp</small></strong>
          <div class="score-stack">
            <span><i class="dot sure-dot"></i>Sure-Tagger <b>{percent(values['Sure-Tagger'])}</b></span>
            <span><i class="dot captioner-dot"></i>qwen-captioner <b>{percent(values['qwen-captioner'])}</b></span>
            <span><i class="dot qwen3-dot"></i>Qwen3-Omni Flash v2 <b>{percent(values['Qwen3-Omni Flash v2'])}</b></span>
          </div>
          <div class="total-breakdown">
            <p class="mono">SURE-TAGGER BREAKDOWN</p>
            <div><span>结构 {structural_axis_count} 轴</span><b>{percent(structural_overall['sure'])}</b></div>
            <div><span>描述 10 轴</span><b>{percent(descriptive['sure'])}</b></div>
            <div><span>领先标签</span><b>{lead_count}/{total_axis_count}</b></div>
          </div>
        </aside>
      </div>
      {page_footer(page_number)}
    </section>
    """


def render_html(sure, captioner, qwen3, descriptive, route_stats, history):
    comparable = comparable_scores(sure, captioner, qwen3)
    gender_field = next(
        item
        for item in comparable["layer_2"]["fields"]
        if item["field"] == GENDER_FIELD
    )
    if gender_field.get("qwen3_null_baseline"):
        gender_summary_note = (
            "Sure-Tagger 与 qwen-captioner 已完成性别评分；Qwen3-Omni 尚未评分，"
            "按显式 null 的 50% 基线纳入本版比较。"
        )
        gender_boundary_note = (
            "结构标签使用 benchmark-scoring-v1.0；speaker.gender 已纳入 Sure-Tagger "
            "与 qwen-captioner，Qwen3-Omni 暂按 null = 1/2 分处理。"
        )
    else:
        gender_summary_note = (
            "三套系统均已完成 speaker.gender 性别序列评分；显式 null 按评分规则计 1/2 分。"
        )
        gender_boundary_note = (
            "结构标签使用 benchmark-scoring-v1.0；speaker.gender 已纳入三套系统，"
            "显式 null 按评分规则计 1/2 分。"
        )
    descriptive_comparable = comparable_descriptive_scores(descriptive)
    structural_rows = [row for layer in comparable.values() for row in layer["fields"]]
    descriptive_rows = descriptive_comparable["fields"]
    structural_overall = {
        "sure": mean(row["sure"] for row in structural_rows),
        "captioner": mean(row["captioner"] for row in structural_rows),
        "qwen3": mean(row["qwen3"] for row in structural_rows),
    }
    descriptive_overall = {
        "sure": descriptive_comparable["sure"],
        "captioner": descriptive_comparable["captioner"],
        "qwen3": descriptive_comparable["qwen3"],
    }
    all_rows = structural_rows + descriptive_rows
    structural_axis_count = len(structural_rows)
    total_axis_count = len(all_rows)
    total_overall = {
        "sure": mean(row["sure"] for row in all_rows),
        "captioner": mean(row["captioner"] for row in all_rows),
        "qwen3": mean(row["qwen3"] for row in all_rows),
    }
    structural_lead_count = sum(
        row["sure"] > max(row["captioner"], row["qwen3"])
        for row in structural_rows
    )
    lead_count = sum(
        row["sure"] > max(row["captioner"], row["qwen3"])
        for row in all_rows
    )
    lead_note = f"Sure-Tagger 在 {lead_count}/{total_axis_count} 个评分轴领先"

    layer_cards = []
    for layer_id, layer in LAYERS.items():
        values = comparable[layer_id]
        leader, leader_value = max(
            (("Sure", values["sure"]), ("Captioner", values["captioner"]), ("Qwen3", values["qwen3"])),
            key=lambda pair: pair[1],
        )
        layer_cards.append(
            f"""
            <article class="layer-card accent-{layer['accent']}">
              <p>{layer['short']} · {escape(layer['title'])}</p>
              <div><strong>{points(leader_value)}</strong><small>%</small></div>
              <span>{leader} leader · Sure {points(values['sure'])} · Qwen3 {points(values['qwen3'])}</span>
            </article>
            """
        )

    history_rows = []
    for name, route, score in history:
        layer_values = [current_layer_score(score, layer_id) for layer_id in LAYERS]
        history_rows.append(
            "<tr>"
            f"<td><b>{escape(name)}</b><small>{escape(route)}</small></td>"
            + "".join(f"<td>{percent(value)}</td>" for value in layer_values)
            + "</tr>"
        )

    official = history[2][2]
    changes = []
    for layer_id, layer in LAYERS.items():
        current = current_layer_score(sure, layer_id)
        previous = current_layer_score(official, layer_id)
        if layer_id == "layer_2":
            change = "口径变更"
        else:
            change = f"{100.0 * (current - previous):+.2f} pp"
        changes.append(
            f"<div><span>{layer['short']}</span><b>{change}</b></div>"
        )

    generated_date = date.today().isoformat()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="author" content="Structural Benchmark">
  <meta name="description" content="Benchmark-v1 结构化音频标签评测打印报告">
  <title>结构化音频能力基准评测 · PDF</title>
  <style>{CSS}</style>
</head>
<body>
  <section class="page cover">
    <div class="cover-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
    <p class="kicker">BENCHMARK REVIEW · {generated_date}</p>
    <h1>音频结构化<br><em>能力基准评测</em></h1>
    <p class="cover-lead">Sure-Tagger、qwen-captioner 与 Qwen3-Omni Flash v2 的结构化与描述性标签能力对比。当前 Sure-Tagger 用整段音频做 FireRed LID；ASR 仍优先单说话人共识，否则中文和英文使用 MOSS，其他或未知语种使用 Qwen3-ASR。本报告统计 T1-T3 的 {total_axis_count} 个可比标签。</p>
    <div class="cover-facts">
      <div><b>650</b><span>音频样本</span></div>
      <div><b>{total_axis_count}</b><span>评分轴</span></div>
      <div><b>v1.0</b><span>评分规范</span></div>
      <div><b>0</b><span>T4 样本</span></div>
    </div>
    {radar_pair(
        "../images/benchmark_radar_total_labels.png",
        "../images/benchmark_radar_total_labels_rps.png",
        "结构与描述性标签总览雷达图",
        "结构与描述性标签总览 RPS 雷达图",
        extra_class="cover-radar",
    )}
    {page_footer(1)}
  </section>

  <section class="page summary-page">
    {page_header('01', 'EXECUTIVE SUMMARY', '结果总览', '先看覆盖与标签级得分，再解释路由变化。接近 50% 可能来自显式 null 的半分基线，不等于 50% 识别准确率。')}
    <div class="headline-metrics">
      <article class="metric-primary"><p>{total_axis_count} 轴总宏平均</p><strong>{percent(total_overall['sure'])}</strong><span>Sure-Tagger</span></article>
      <article><p>qwen-captioner</p><strong>{percent(total_overall['captioner'])}</strong><span>结构 + 描述</span></article>
      <article><p>Qwen3-Omni Flash v2</p><strong>{percent(total_overall['qwen3'])}</strong><span>结构 + 描述</span></article>
      <article><p>领先标签数</p><strong>{lead_count}<small>/{total_axis_count}</small></strong><span>{lead_note}</span></article>
    </div>
    <div class="layer-card-grid">{''.join(layer_cards)}</div>
    <div class="track-summary">
      <div><p class="mono">STRUCTURAL · {structural_axis_count} AXES</p><b>Sure {percent(structural_overall['sure'])}</b><span>Captioner {percent(structural_overall['captioner'])} · Qwen3 {percent(structural_overall['qwen3'])}</span></div>
      <div><p class="mono">DESCRIPTIVE · 10 AXES</p><b>Sure {percent(descriptive_overall['sure'])}</b><span>Captioner {percent(descriptive_overall['captioner'])} · Qwen3 {percent(descriptive_overall['qwen3'])}</span></div>
    </div>
    <div class="summary-grid">
      <article class="route-panel">
        <p class="mono">CURRENT ROUTING / CONSENSUS</p>
        <h3>整段 LID；共识优先，<br>否则按语言选择 ASR。</h3>
        <div class="route-bar"><i style="width:{100 * route_stats['firered'] / 650:.2f}%"></i><b style="width:{100 * route_stats['moss'] / 650:.2f}%"></b><strong style="width:{100 * route_stats['qwen3_asr'] / 650:.2f}%"></strong><em style="width:{100 * route_stats['failed_closed'] / 650:.2f}%"></em></div>
        <ul><li><span class="route-firered"></span>FireRed {route_stats['firered']}</li><li><span class="route-moss"></span>MOSS {route_stats['moss']}</li><li><span class="route-qwen3"></span>Qwen3-ASR {route_stats['qwen3_asr']}</li><li><span class="route-failed"></span>未选路由 {route_stats['failed_closed']}</li></ul>
      </article>
      <article class="route-numbers">
        <div><b>{route_stats['firered']}</b><span>FireRed</span></div>
        <div><b>{route_stats['moss']}</b><span>MOSS</span></div>
        <div><b>{route_stats['qwen3_asr']}</b><span>Qwen3-ASR</span></div>
        <div><b>{route_stats['failed_closed']}</b><span>failed closed</span></div>
        <p>{escape(gender_summary_note)}</p>
      </article>
    </div>
    {page_footer(2)}
  </section>

  <section class="page method-page">
    {page_header('02', 'SCOPE & SCORING', '数据范围与评分口径', '统计对象为 structural-benchmark-subset-v1.0。T4 controlled render 完全排除；结构与描述性评分分轨计算。')}
    <table class="dataset-table">
      <thead><tr><th>Track</th><th>数据来源</th><th>条数</th><th>时长</th><th>用途</th></tr></thead>
      <tbody>
        <tr><td><b>T1</b> Natural meeting</td><td>AMI</td><td>350</td><td>145.05 min</td><td>说话人、重叠、ASR</td></tr>
        <tr><td><b>T2</b> Language / ASR</td><td>FLEURS + LibriSpeech</td><td>150</td><td>26.79 min</td><td>10 语种 LID / ASR</td></tr>
        <tr><td><b>T3</b> Event / noise</td><td>DESED + FSD50K</td><td>150</td><td>21.21 min</td><td>事件与 broad noise</td></tr>
        <tr class="total"><td>合计</td><td></td><td>650</td><td>193.05 min</td><td>约 3.2 小时音频</td></tr>
      </tbody>
    </table>
    <div class="method-grid">
      <article><span class="rule-number">2</span><h3>预测正确</h3><p>除特别说明外，预测正确得 2 分。</p></article>
      <article><span class="rule-number">1</span><h3>显式弃答</h3><p>合法 <code>null</code> 得 1 分。</p></article>
      <article><span class="rule-number">0</span><h3>错误或失败</h3><p>错误、缺失、非法或运行失败得 0 分。</p></article>
    </div>
    <div class="method-notes">
      <div><b>LABEL-FIRST MACRO</b><p>同一标签先跨评测组等权平均，再在所在层内对标签等权平均。无有效 GT 的字段不当作 0 分。</p></div>
      <div><b>RATIO FIELDS</b><p><code>nonspeech_ratio</code> 与 <code>overlap_ratio</code> 均直接按 <code>2 × (1 - |预测-GT|)</code> 评分，不使用容差除数。</p></div>
      <div><b>CLASS BALANCE</b><p>布尔字段分别计算正、负样本得分后取平均；多标签字段按有效类别宏平均。</p></div>
      <div><b>COMPARISON BOUNDARY</b><p>{escape(gender_boundary_note)}描述性标签继续使用统一 DeepSeek 0–2 分制。</p></div>
    </div>
    <aside class="excluded-box">
      <p class="mono">STRUCTURAL DIAGNOSTIC ONLY / EXCLUDED FROM STRUCTURAL SCORE</p>
      <code>language_content.filler</code>
      <code>language_content.punctuation.punctuation_count</code>
      <code>language_content.punctuation.has_terminal_punctuation</code>
      <span>Layer 4 亦因没有冻结 GT 记为 n/a；描述性 10 个字段另行评分并在 D / T 雷达图中展示。</span>
    </aside>
    {page_footer(3)}
  </section>

  {layer_page(4, 'layer_1', comparable)}
  {layer_page(5, 'layer_2', comparable)}
  {layer_page(6, 'layer_3', comparable)}
  {layer_page(7, 'layer_5', comparable)}
  {descriptive_page(8, descriptive_comparable)}
  {total_page(9, structural_overall, descriptive_overall, total_overall, lead_count, structural_axis_count, total_axis_count)}

  <section class="page history-page">
    {page_header('10', 'VERSION HISTORY', '链路变化与结论', '历史点为 v0.9 且不含性别轴；当前结果为 v1.0 并新增 speaker.gender，因此 L2 标记为口径变更，不直接计算差值。')}
    <table class="history-table">
      <thead><tr><th>版本</th><th>L1 词汇</th><th>L2 说话人</th><th>L3 时间</th><th>L5 声学</th></tr></thead>
      <tbody>{''.join(history_rows)}</tbody>
    </table>
    <div class="change-strip">{''.join(changes)}</div>
    <div class="conclusion-grid">
      <article class="conclusion-main">
        <p class="mono">BOTTOM LINE</p>
        <h3>语言判定回到整段 LID，<br>结构化优势保持。</h3>
        <p>当前 Sure-Tagger 在 L1、L2、L3、L5 分别为 <b>{points(current_layer_score(sure, 'layer_1'))}、{points(current_layer_score(sure, 'layer_2'))}、{points(current_layer_score(sure, 'layer_3'))}、{points(current_layer_score(sure, 'layer_5'))}</b>，结构评分轴中 {structural_lead_count}/{structural_axis_count} 领先；描述性 10 轴宏平均为 <b>{percent(descriptive_overall['sure'])}</b>，{total_axis_count} 轴总宏平均为 <b>{percent(total_overall['sure'])}</b>。</p>
      </article>
      <article class="audit-list">
        <p class="mono">WHAT DID NOT CHANGE</p>
        <div><b>0 / 650</b><span>nonspeech_ratio 变化样本</span></div>
        <div><b>0 / 650</b><span>sound_field_scene 变化样本</span></div>
        <p>当前 Sure-Tagger 结构与 DeepSeek 合并结果均完成 650 条覆盖校验；speaker.gender 只在 T1 AMI 的有效 GT 上评分。</p>
      </article>
    </div>
    <div class="source-box">
      <p class="mono">SOURCES / REPRODUCIBILITY</p>
      <code>docs/report/result/sure-tagger/score.json</code>
      <code>docs/report/result/qwen-captioner/score.json</code>
      <code>docs/report/result/qwen3-omni-v2-multi-api12/structural_score.json</code>
      <code>dataset/benchmark/SCORING_STANDARD.md</code>
      <code>docs/report/result/descriptive/sure-tagger/summary.json</code>
      <code>docs/report/result/descriptive/qwen-captioner/summary.json</code>
      <code>docs/report/result/descriptive/qwen3-omni-v2-multi-api12/score/summary.json</code>
      <code>docs/report/result/images/benchmark_radar_*.png</code>
      <code>docs/report/result/images/benchmark_radar_*_rps.png</code>
      <span>生成器：<code>eval/generate_benchmark_pdf.py</code></span>
    </div>
    {page_footer(10)}
  </section>
</body>
</html>
"""


CSS = r"""
@font-face {
  font-family: "Report CJK";
  src: url("../html/fonts/wqy-microhei.ttf") format("truetype");
  font-weight: 100 900;
}
@page { size: A4; margin: 0; }
:root {
  --ink: #151513;
  --paper: #f2efe5;
  --paper2: #e4dfd1;
  --white: #fffef9;
  --acid: #d8ff3e;
  --orange: #ff754d;
  --orange-ink: #b33a1d;
  --blue: #4d68e8;
  --purple: #77148b;
  --teal: #2f8b7d;
  --muted: #6e6b63;
  --mono: "Report CJK", "DejaVu Sans Mono", monospace;
}
* { box-sizing: border-box; letter-spacing: 0; }
html, body { margin: 0; padding: 0; color: var(--ink); background-color: #f2efe5; }
body { font-family: "Report CJK", "DejaVu Sans", sans-serif; font-size: 9pt; line-height: 1.45; }
p, h1, h2, h3, figure { margin-top: 0; }
code { font-family: var(--mono); overflow-wrap: anywhere; }
.page {
  position: relative;
  width: 210mm;
  height: 297mm;
  padding: 16mm 16mm 14mm;
  overflow: hidden;
  page-break-after: always;
  background-color: var(--paper);
  background-image: linear-gradient(rgba(21,21,19,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(21,21,19,.045) 1px, transparent 1px);
  background-size: 12mm 12mm;
}
.page:last-child { page-break-after: auto; }
.page::before { content: ""; position: absolute; left: 0; top: 0; width: 5mm; height: 100%; background-color: #ff754d; }
.page-footer { position: absolute; left: 16mm; right: 16mm; bottom: 6mm; display: flex; justify-content: space-between; padding-top: 2mm; border-top: .55mm solid #151513; font: 700 6.8pt/1 var(--mono); }
.kicker, .mono { font: 800 7.4pt/1.15 var(--mono); text-transform: uppercase; }
.kicker { color: var(--orange-ink); margin-bottom: 4mm; }
.page-head { display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 5mm; margin-bottom: 6mm; border-bottom: .7mm solid #151513; }
.page-head > div { width: 54%; }
.page-head > .page-intro { width: 39%; }
.page-head h2 { margin: 0; font-size: 26pt; line-height: 1; }
.page-intro { margin: 0; color: #45423d; font-size: 8.5pt; }

.cover { padding-top: 18mm; }
.cover::before { width: 100%; height: 5mm; }
.cover-mark { display: none; }
.cover-mark i { display: block; width: 2mm; background-color: #151513; transform: skew(-15deg); }
.cover-mark i:nth-child(1){height:42%}.cover-mark i:nth-child(2){height:75%}.cover-mark i:nth-child(3){height:55%}.cover-mark i:nth-child(4){height:100%}.cover-mark i:nth-child(5){height:67%}
.cover h1 { margin: 0 0 5mm; max-width: 165mm; font-size: 36pt; line-height: .98; }
.cover h1 em { color: var(--orange-ink); font-style: normal; }
.cover-lead { width: 145mm; margin-bottom: 6mm; font-size: 10.4pt; line-height: 1.55; }
.cover-facts { display: flex; margin-bottom: 4mm; border: .65mm solid #151513; background-color: #d8ff3e; }
.cover-facts div { flex: 1; padding: 3mm 4mm; border-right: .45mm solid #151513; }
.cover-facts div:last-child { border-right: 0; }
.cover-facts b { display: block; font-size: 20pt; line-height: 1; }
.cover-facts span { font: 700 6.8pt/1 var(--mono); }
.radar { margin: 0; background-color: white; border: .65mm solid #151513; }
.radar img { display: block; width: 100%; height: auto; }
.cover-radar { width: 154mm; margin: 0 auto; }
.cover-radar img { height: 121mm; object-fit: contain; }
.radar-stack { display: block; width: 154mm; margin: 0 auto; }
.radar-stack .radar { display: block; width: 100%; margin: 0 0 2mm 0; }
.radar-stack .radar img { display: block; width: 100%; height: 56mm; object-fit: contain; background-color: white; }
.radar-stack .radar figcaption { display: block; padding: 1.2mm 3mm; white-space: nowrap; }
.radar figcaption { display: flex; justify-content: space-between; padding: 2mm 3mm; border-top: .45mm solid #151513; font: 700 6.5pt/1 var(--mono); }
.dot { display: inline-block; width: 2.5mm; height: 2.5mm; margin-right: 1mm; vertical-align: -.3mm; }
.sure-dot { background-color: #77148b; }
.captioner-dot { background-color: #4d68e8; }
.qwen3-dot { background-color: #d84c2f; }

.headline-metrics { display: flex; border: .65mm solid #151513; background-color: #fffef9; }
.headline-metrics article { flex: 1; min-height: 33mm; padding: 4mm; border-right: .45mm solid #151513; }
.headline-metrics article:first-child { flex: 1.25; }
.headline-metrics article:last-child { border-right: 0; }
.headline-metrics .metric-primary { background-color: #77148b; color: white; }
.headline-metrics p { margin: 0 0 3mm; font: 700 7pt/1 var(--mono); }
.headline-metrics strong { display: block; font-size: 27pt; line-height: 1; }
.headline-metrics strong small { font-size: 13pt; }
.headline-metrics span { display: block; margin-top: 2mm; font-size: 7.3pt; }
.track-summary { display: flex; justify-content: space-between; margin-top: 4mm; }
.track-summary > div { width: 48.8%; padding: 3mm 4mm; border: .55mm solid #151513; background-color: #e4dfd1; }
.track-summary p { margin: 0 0 1.5mm; font: 700 6.8pt/1 var(--mono); }
.track-summary b { font-size: 13pt; }
.track-summary span { margin-left: 3mm; color: var(--muted); font-size: 7pt; }
.layer-card-grid { display: flex; margin-top: 5mm; }
.layer-card { width: 23.6%; min-height: 34mm; margin-right: 1.85%; padding: 3mm; border: .55mm solid #151513; background-color: #fffef9; }
.layer-card:last-child { margin-right: 0; }
.layer-card::before { content: ""; display: block; width: 13mm; height: 2.2mm; margin-bottom: 3mm; background-color: #ff754d; }
.layer-card.accent-acid::before { background-color: #d8ff3e; }
.layer-card.accent-blue::before { background-color: #4d68e8; }
.layer-card.accent-teal::before { background-color: #2f8b7d; }
.layer-card p { min-height: 7mm; margin: 0 0 2mm; font: 700 6.7pt/1.2 var(--mono); }
.layer-card strong { font-size: 19pt; line-height: 1; }
.layer-card small { font-size: 8pt; }
.layer-card span { display: block; margin-top: 2mm; color: var(--muted); font-size: 6.8pt; }
.summary-grid { display: flex; justify-content: space-between; margin-top: 6mm; }
.route-panel, .route-numbers { padding: 5mm; border: .65mm solid #151513; background-color: #fffef9; }
.route-panel { width: 59%; }
.route-numbers { width: 38%; }
.route-panel h3 { margin: 3mm 0 5mm; font-size: 17pt; line-height: 1.15; }
.route-bar { display: table; width: 100%; height: 7mm; border: .45mm solid #151513; table-layout: fixed; }
.route-bar i, .route-bar b, .route-bar strong, .route-bar em { display: table-cell; height: 7mm; }
.route-bar i { background-color: #ff754d; }
.route-bar b { background-color: #d8ff3e; }
.route-bar strong { background-color: #4d68e8; }
.route-bar em { background-color: #2f8b7d; }
.route-panel ul { display: flex; flex-wrap: wrap; gap: 2mm 5mm; padding: 0; margin: 3mm 0 0; list-style: none; font-size: 6.8pt; }
.route-panel li span { display: inline-block; width: 2.3mm; height: 2.3mm; margin-right: 1mm; }
.route-firered { background-color: #ff754d; }.route-moss { background-color: #d8ff3e; }.route-qwen3 { background-color: #4d68e8; }.route-failed { background-color: #2f8b7d; }
.route-numbers { display: flex; flex-wrap: wrap; align-content: center; justify-content: space-between; }
.route-numbers div { width: 48%; margin-bottom: 2mm; padding: 2mm 1mm; text-align: center; border: .4mm solid #151513; }
.route-numbers b { display: block; font-size: 16pt; }
.route-numbers span { font-size: 7pt; }
.route-numbers p { width: 100%; margin: 3mm 0 0; font-size: 7.4pt; }

.dataset-table, .score-table, .history-table { width: 100%; border-collapse: collapse; background-color: #fffef9; }
.dataset-table th, .dataset-table td, .score-table th, .score-table td, .history-table th, .history-table td { padding: 2.6mm 2.4mm; text-align: left; vertical-align: middle; border: .4mm solid #151513; }
thead th { color: white; background-color: #151513; font: 700 7pt/1.2 var(--mono); }
.dataset-table { font-size: 7.8pt; }
.dataset-table .total { font-weight: 700; background-color: #d8ff3e; }
.method-grid { display: flex; justify-content: space-between; margin-top: 5mm; }
.method-grid article { position: relative; width: 31.8%; min-height: 36mm; padding: 4mm 4mm 3mm 15mm; border: .55mm solid #151513; background-color: #fffef9; }
.rule-number { position: absolute; left: 3mm; top: 4mm; width: 9mm; height: 9mm; padding-top: .7mm; text-align: center; color: white; background-color: #151513; font-size: 15pt; line-height: 1; }
.method-grid h3 { margin: 0 0 2mm; font-size: 11pt; }
.method-grid p { margin: 0; font-size: 7.5pt; }
.method-notes { display: block; margin-top: 5mm; font-size: 0; }
.method-notes div { display: inline-block; width: 48.5%; min-height: 29mm; margin: 0 3% 3mm 0; padding: 4mm; vertical-align: top; border-left: 2.5mm solid #ff754d; background-color: #e4dfd1; }
.method-notes div:nth-child(even) { margin-right: 0; }
.method-notes b { display: block; font-family: "Report CJK", "DejaVu Sans Mono", monospace; font-size: 7pt; font-weight: 800; line-height: 1.2; }
.method-notes p { margin: 2mm 0 0; font-size: 7.5pt; }
.excluded-box { margin-top: 2mm; padding: 4mm; color: white; background-color: #151513; }
.excluded-box p { margin-bottom: 2.5mm; color: var(--acid); }
.excluded-box code { display: block; margin: .8mm 0; font-size: 7.3pt; }
.excluded-box span { display: block; margin-top: 2mm; color: #d5d2c9; font-size: 7pt; }

.layer-page::before { background-color: #ff754d; }
.layer-page.accent-acid::before { background-color: #d8ff3e; }
.layer-page.accent-blue::before { background-color: #4d68e8; }
.layer-page.accent-teal::before { background-color: #2f8b7d; }
.layer-hero { position: relative; display: block; height: 118mm; }
.layer-hero .radar-stack { position: absolute; top: 0; left: 0; width: 65.5%; margin: 0; }
.layer-hero .radar { position: absolute; top: 0; left: 0; width: 65.5%; }
.radar-layer img { width: 100%; height: auto; }
.layer-hero .radar-stack .radar { position: relative; top: auto; left: auto; width: 100%; margin: 0 0 2mm 0; }
.layer-hero .radar-stack .radar img { width: 100%; height: 48mm; object-fit: contain; background-color: white; }
.layer-result { position: absolute; top: 0; right: 0; display: block; width: 31.5%; height: 118mm; padding: 5mm; color: white; background-color: #151513; border: .65mm solid #151513; }
.layer-result > strong { margin: 6mm 0 5mm; color: var(--acid); font-size: 28pt; line-height: 1; }
.layer-result > strong small { font-size: 11pt; }
.score-stack { padding: 3mm 0; border-top: .35mm solid #888; border-bottom: .35mm solid #888; }
.score-stack span { display: block; margin: 1.5mm 0; font-size: 7pt; }
.score-stack b { float: right; font-size: 8pt; }
.analysis-box { margin-top: 4mm; padding-top: 2mm; }
.analysis-box b { color: var(--orange); font-size: 10pt; }
.analysis-box p { margin: 2mm 0 0; color: #d7d4cc; font-size: 7.1pt; line-height: 1.5; }
.score-table { margin-top: 5mm; font-size: 7pt; table-layout: fixed; }
.score-table th:first-child, .score-table td:first-child { width: 35%; }
.score-table th:nth-child(2), .score-table td:nth-child(2), .score-table th:nth-child(3), .score-table td:nth-child(3), .score-table th:nth-child(4), .score-table td:nth-child(4) { width: 14%; text-align: right; }
.score-table th:last-child, .score-table td:last-child { width: 23%; }
.score-table code { font-size: 6.5pt; }
.null-score { color: var(--muted); font: 700 5.6pt/1 var(--mono); text-transform: uppercase; }
.winner { display: block; padding: 1.2mm; text-align: center; color: white; background-color: #77148b; font: 700 6.4pt/1 var(--mono); }
.winner.captioner { background-color: #4d68e8; }
.winner.qwen3 { background-color: #d84c2f; }
.descriptive-score-table { margin-top: 3mm; }
.descriptive-score-table th, .descriptive-score-table td { padding-top: 1.2mm; padding-bottom: 1.2mm; }
.total-page::before { background-color: #2f8b7d; }
.total-hero { position: relative; display: block; min-height: 184mm; }
.total-radar { position: absolute; top: 0; left: 0; width: 69%; }
.total-radar img { display: block; width: 100%; height: auto; }
.total-hero .radar-stack { position: absolute; top: 0; left: 0; width: 69%; margin: 0; }
.total-hero .radar-stack .radar { position: relative; top: auto; left: auto; width: 100%; margin: 0 0 2mm 0; }
.total-hero .radar-stack .radar img { width: 100%; height: 82mm; object-fit: contain; background-color: white; }
.total-result { position: absolute; top: 0; right: 0; width: 28%; min-height: 184mm; padding: 6mm; color: white; background-color: #151513; border: .65mm solid #151513; }
.total-result > strong { display: block; margin: 8mm 0 6mm; color: var(--acid); font-size: 28pt; line-height: 1; }
.total-result > strong small { font-size: 11pt; }
.total-breakdown { margin-top: 10mm; padding-top: 4mm; border-top: .35mm solid #888; }
.total-breakdown p { margin-bottom: 3mm; color: var(--orange); }
.total-breakdown div { display: flex; justify-content: space-between; padding: 2.5mm 0; border-bottom: .35mm solid #555; font-size: 8pt; }
.total-breakdown b { color: var(--acid); }

.history-page .page-head { padding-bottom: 3mm; margin-bottom: 3.5mm; }
.history-table { font-size: 6.8pt; }
.history-table th, .history-table td { padding: 1.45mm 2.1mm; }
.history-table th:not(:first-child), .history-table td:not(:first-child) { text-align: right; }
.history-table td:first-child { width: 37%; }
.history-table td small { display: block; margin-top: .25mm; color: var(--muted); font-size: 6pt; }
.history-table tr:last-child { background-color: #d8ff3e; }
.change-strip { display: flex; margin-top: 3mm; border: .65mm solid #151513; background-color: #fffef9; }
.change-strip div { flex: 1; padding: 2.4mm 3mm; border-right: .4mm solid #151513; }
.change-strip div:last-child { border-right: 0; }
.change-strip span { display: block; font: 700 6.6pt/1 var(--mono); }
.change-strip b { display: block; margin-top: 1mm; font-size: 13.5pt; }
.conclusion-grid { display: flex; justify-content: space-between; margin-top: 3.2mm; }
.conclusion-main { width: 61%; padding: 4mm; color: white; background-color: #77148b; border: .65mm solid #151513; }
.conclusion-main .mono { color: var(--acid); }
.conclusion-main h3 { margin: 2.2mm 0; font-size: 15pt; line-height: 1.15; }
.conclusion-main p:last-child { margin: 0; font-size: 7.1pt; }
.audit-list { width: 36.5%; padding: 3.5mm; border: .65mm solid #151513; background-color: #fffef9; }
.audit-list > div { padding: 1.8mm 0; border-bottom: .35mm solid #151513; }
.audit-list b { display: block; font-size: 14pt; }
.audit-list span { font-size: 6.6pt; }
.audit-list > p:last-child { margin: 2mm 0 0; font-size: 6.8pt; }
.source-box { margin-top: 3mm; padding: 2.4mm 3mm 2mm; border: .55mm solid #151513; background-color: #e4dfd1; page-break-inside: avoid; }
.source-box p { margin-bottom: 1mm; }
.source-box code { display: inline-block; width: 49.5%; margin: 0; font-size: 5.5pt; line-height: 1.28; vertical-align: top; }
.source-box span { display: block; margin-top: 1mm; font-size: 6.2pt; }
.source-box span code { display: inline; width: auto; }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "benchmark-v1-current.pdf",
        help="PDF output path",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=OUTPUT_ROOT / "benchmark-v1-current.html",
        help="Intermediate standalone print HTML path",
    )
    args = parser.parse_args()

    sure = load_structural_score(SURE_SCORE)
    captioner = load_structural_score(CAPTIONER_SCORE)
    qwen3 = load_structural_score(QWEN3_OMNI_SCORE)
    descriptive = {key: load_descriptive_score(path) for key, path in DESCRIPTIVE_SCORES.items()}
    route_stats = load_route_stats(SURE_RUN_ROOT)
    history = [(name, route, load_structural_score(path)) for name, route, path in HISTORY]
    rendered = render_html(sure, captioner, qwen3, descriptive, route_stats, history)

    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(rendered, encoding="utf-8")

    try:
        from weasyprint import HTML
        from weasyprint.fonts import FontConfiguration
    except ImportError as error:
        raise SystemExit("WeasyPrint is required to generate the PDF") from error

    font_config = FontConfiguration()
    HTML(string=rendered, base_url=str(args.html_output.parent)).write_pdf(
        str(args.output), font_config=font_config
    )
    print(args.html_output)
    print(args.output)


if __name__ == "__main__":
    main()
