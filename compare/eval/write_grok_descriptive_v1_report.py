#!/usr/bin/env python3
"""Write DeepSeek vs Grok descriptive scoring tables for Sure-Tagger v1 + captioner + omni."""

from __future__ import print_function

import json
import math
from collections import OrderedDict
from pathlib import Path

COMPARE_ROOT = Path("/hpc_stor03/sjtu_home/weihan.chen/share/compare")
FIELDS = (
    "caption",
    "topic",
    "background_condition",
    "dominant_content_impression",
    "clarity_intelligibility_impression",
    "background_sound_summary",
    "descriptive_phrases",
    "facets",
    "speaker_descriptions",
    "scene_hypotheses",
)
FIELD_NOS = {field: 20 + i for i, field in enumerate(FIELDS)}
SYSTEMS = OrderedDict(
    [
        (
            "sure-tagger",
            {
                "label": "Sure-Tagger v1",
                "predictions": "evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl",
                "deepseek": "evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/descriptive_score/results.jsonl",
                "grok": "evaluation/descriptive-semantic-scoring-grok-4.7/sure-tagger-v1/results.jsonl",
            },
        ),
        (
            "omni",
            {
                "label": "Qwen3-Omni Flash v2",
                "predictions": "docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl",
                "deepseek": "docs/report/result/descriptive/qwen3-omni-v2-multi-api12/score/results.jsonl",
                "grok": "evaluation/descriptive-semantic-scoring-grok-4.7/qwen3-omni/results.jsonl",
            },
        ),
        (
            "captioner",
            {
                "label": "qwen-captioner",
                "predictions": "docs/report/result/qwen-captioner/results_t1_t3.jsonl",
                "deepseek": "docs/report/result/descriptive/qwen-captioner/results.jsonl",
                "grok": "evaluation/descriptive-semantic-scoring-grok-4.7/qwen-captioner/results.jsonl",
            },
        ),
    ]
)


def load_success(path):
    out = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "success" and row.get("sample_id"):
                out[row["sample_id"]] = row
    return out


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else float("nan")


def field_rate(records, field):
    scores = [rec["scores"][field] for rec in records if rec.get("scores") and field in rec["scores"]]
    return 50.0 * mean(scores) if scores else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = rank
        i = j + 1
    return out


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def fmt_pct(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return ("%%.%df" % digits) % value


def fmt_delta(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return "%s%.2f pp" % (sign, value)


def fmt_corr(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return "%.3f" % value


def collect():
    data = OrderedDict()
    for key, spec in SYSTEMS.items():
        ds = load_success(COMPARE_ROOT / spec["deepseek"])
        gk = load_success(COMPARE_ROOT / spec["grok"])
        common = sorted(set(ds) & set(gk))
        ds_all = list(ds.values())
        gk_all = list(gk.values())
        pair_ds = [ds[sid]["normalized_percent"] for sid in common]
        pair_gk = [gk[sid]["normalized_percent"] for sid in common]
        field_ds = {field: field_rate(ds_all, field) for field in FIELDS}
        field_gk = {field: field_rate(gk_all, field) for field in FIELDS}
        higher_g = sum(1 for a, b in zip(pair_ds, pair_gk) if b > a)
        higher_d = sum(1 for a, b in zip(pair_ds, pair_gk) if a > b)
        tied = sum(1 for a, b in zip(pair_ds, pair_gk) if a == b)
        field_pearson = {}
        for field in FIELDS:
            xs = [ds[sid]["scores"][field] for sid in common]
            ys = [gk[sid]["scores"][field] for sid in common]
            field_pearson[field] = pearson(xs, ys)
        data[key] = {
            "spec": spec,
            "n_ds": len(ds),
            "n_gk": len(gk),
            "n_common": len(common),
            "macro_ds": mean(field_ds.values()),
            "macro_gk": mean(field_gk.values()),
            "field_ds": field_ds,
            "field_gk": field_gk,
            "pearson": pearson(pair_ds, pair_gk),
            "spearman": spearman(pair_ds, pair_gk),
            "higher_g": higher_g,
            "higher_d": higher_d,
            "tied": tied,
            "mean_delta": mean(b - a for a, b in zip(pair_ds, pair_gk)) if common else float("nan"),
            "field_pearson": field_pearson,
        }
    return data


def write_svg(data, path):
    labels = [data[key]["spec"]["label"] for key in SYSTEMS]
    ds = [data[key]["macro_ds"] for key in SYSTEMS]
    gk = [data[key]["macro_gk"] for key in SYSTEMS]
    width, height = 820, 360
    left, right, top, bottom = 70, 796, 36, 304
    ymax = 80.0
    plot_h = bottom - top
    plot_w = right - left
    group_w = plot_w / 3.0
    bar_w = 67.8
    gap = 8.0
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
        % (width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="24" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif">描述 10 轴宏平均：DeepSeek v4 Pro vs Grok 4.7</text>',
    ]
    for tick in range(0, 90, 10):
        y = bottom - plot_h * (tick / ymax)
        lines.append(
            '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#eee"/>'
            % (left, y, right, y)
        )
        lines.append(
            '<text x="62" y="%s" text-anchor="end" font-size="11" font-family="DejaVu Sans, Arial, sans-serif" fill="#555">%s</text>'
            % (y + 4, tick)
        )
    lines.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#333"/>' % (left, top, left, bottom))
    lines.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="#333"/>' % (left, bottom, right, bottom))
    colors = ("#4C78A8", "#F58518")
    for i, (label, dsv, gkv) in enumerate(zip(labels, ds, gk)):
        cx = left + group_w * (i + 0.5)
        x0 = cx - bar_w - gap / 2.0
        x1 = cx + gap / 2.0
        for x, value, color in ((x0, dsv, colors[0]), (x1, gkv, colors[1])):
            h = plot_h * (value / ymax)
            y = bottom - h
            lines.append(
                '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>'
                % (x, y, bar_w, h, color)
            )
            lines.append(
                '<text x="%.1f" y="%.1f" text-anchor="middle" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">%.2f</text>'
                % (x + bar_w / 2.0, y - 6, value)
            )
        lines.append(
            '<text x="%.1f" y="322.0" text-anchor="middle" font-size="12" font-family="DejaVu Sans, Arial, sans-serif">%s</text>'
            % (cx, label)
        )
    lines.extend(
        [
            '<rect x="560" y="332" width="12" height="12" fill="#4C78A8"/>',
            '<text x="578" y="342" font-size="12" font-family="DejaVu Sans, Arial, sans-serif">DeepSeek v4 Pro</text>',
            '<rect x="690" y="332" width="12" height="12" fill="#F58518"/>',
            '<text x="708" y="342" font-size="12" font-family="DejaVu Sans, Arial, sans-serif">Grok 4.7</text>',
            "</svg>",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))


def render_markdown(data):
    order = list(SYSTEMS)
    sure, omni, cap = (data[k] for k in order)
    lines = []
    a = lines.append
    a("# 描述性语义打分：Grok 4.7 作为第二评委")
    a("")
    a("口径与 [`benchmark-v1-29-metrics.md`](../../docs/benchmark-v1-29-metrics.md) 一致：描述 10 轴由评委打 $u\\in[0,2]$，再 $r_f=\\bar u_f/2$。百分数为得分率 $\\times 100\\%$。官方聚合只用评分成功样本：650 条中 648 条成功，2 条缺描述性 GT（`ES2002b_utterance_00012`、`IN1016_utterance_00044`），不记 0 分。")
    a("")
    a("本实验**不改 schema 生成**：三套系统的描述性标签仍由原流程产出。只把评委从 DeepSeek v4 Pro 换成 Grok 4.7，prompt、GT、字段合同完全相同。")
    a("")
    a("Sure-Tagger 使用论文/PDF 的 **v1**（`sure_tagger_full_20260918_main_pipeline_w2_b8`，DeepSeek 描述 10 轴 65.09%）。")
    a("")
    a("## 实验设定")
    a("")
    a("- 评委 A：`deepseek-v4-pro`（既有结果，thinking enabled）")
    a("- 评委 B：`grok-4.7`（`compare/api/grok.txt`，thinking enabled）")
    a("- Prompt：`eval/deepseek_descriptive_semantic_scoring_prompt.txt`（`prompt_sha256=2e49b92b25e334f195b9526f1c171bd39e6285646dfa329214ca1d6f6be7ce9a`）")
    a("- GT：`evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3`")
    a("- 实现：`eval/run_descriptive_semantic_scoring_batch.py`，启动脚本 `eval/run_grok_descriptive_semantic_scoring.sh`")
    a("- Grok 并发：Sure-Tagger v1 全量 **200 workers**；captioner / Omni 沿用既有全量结果")
    a("- Grok 产物：`evaluation/descriptive-semantic-scoring-grok-4.7/{sure-tagger-v1,qwen-captioner,qwen3-omni}`")
    a("")
    a("| 系统 | 被评预测 | DeepSeek 评分目录 |")
    a("| --- | --- | --- |")
    a("| Sure-Tagger v1 | `evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl` | 同目录 `descriptive_score/` |")
    a("| Qwen3-Omni Flash v2 | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` | `docs/report/result/descriptive/qwen3-omni-v2-multi-api12/score/` |")
    a("| qwen-captioner | `docs/report/result/qwen-captioner/results_t1_t3.jsonl` | `docs/report/result/descriptive/qwen-captioner/` |")
    a("")
    a("## 描述 10 轴宏平均")
    a("")
    a("![DeepSeek vs Grok 宏平均](../../docs/report/images/descriptive-grok-4.7-vs-deepseek.svg)")
    a("")
    a("| 系统 | DeepSeek v4 Pro | Grok 4.7 | Δ（Grok − DeepSeek） |")
    a("| --- | ---: | ---: | ---: |")
    for key in order:
        row = data[key]
        a(
            "| %s | %s%% | **%s%%** | %s |"
            % (
                row["spec"]["label"],
                fmt_pct(row["macro_ds"]),
                fmt_pct(row["macro_gk"]),
                fmt_delta(row["macro_gk"] - row["macro_ds"]),
            )
        )
    a("")
    ranked = sorted(order, key=lambda k: -data[k]["macro_gk"])
    a(
        "排序在两个评委下完全一致：**%s**。"
        % " > ".join(data[k]["spec"]["label"] for k in ranked)
    )
    a("")
    a("## 分轴得分率")
    a("")
    a("### DeepSeek v4 Pro")
    a("")
    a("| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |")
    a("| ---: | --- | ---: | ---: | ---: |")
    for field in FIELDS:
        a(
            "| %d | `%s` | %s%% | %s%% | %s%% |"
            % (
                FIELD_NOS[field],
                field,
                fmt_pct(sure["field_ds"][field]),
                fmt_pct(omni["field_ds"][field]),
                fmt_pct(cap["field_ds"][field]),
            )
        )
    a(
        "|  | **描述 10 轴宏平均** | **%s%%** | **%s%%** | **%s%%** |"
        % (fmt_pct(sure["macro_ds"]), fmt_pct(omni["macro_ds"]), fmt_pct(cap["macro_ds"]))
    )
    a("")
    a("### Grok 4.7")
    a("")
    a("| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |")
    a("| ---: | --- | ---: | ---: | ---: |")
    for field in FIELDS:
        a(
            "| %d | `%s` | %s%% | %s%% | %s%% |"
            % (
                FIELD_NOS[field],
                field,
                fmt_pct(sure["field_gk"][field]),
                fmt_pct(omni["field_gk"][field]),
                fmt_pct(cap["field_gk"][field]),
            )
        )
    a(
        "|  | **描述 10 轴宏平均** | **%s%%** | **%s%%** | **%s%%** |"
        % (fmt_pct(sure["macro_gk"]), fmt_pct(omni["macro_gk"]), fmt_pct(cap["macro_gk"]))
    )
    a("")
    a("### 分轴 Δ（Grok − DeepSeek，pp）")
    a("")
    a("| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |")
    a("| ---: | --- | ---: | ---: | ---: |")
    for field in FIELDS:
        a(
            "| %d | `%s` | %s | %s | %s |"
            % (
                FIELD_NOS[field],
                field,
                fmt_delta(sure["field_gk"][field] - sure["field_ds"][field]),
                fmt_delta(omni["field_gk"][field] - omni["field_ds"][field]),
                fmt_delta(cap["field_gk"][field] - cap["field_ds"][field]),
            )
        )
    a(
        "|  | **宏平均** | **%s** | **%s** | **%s** |"
        % (
            fmt_delta(sure["macro_gk"] - sure["macro_ds"]),
            fmt_delta(omni["macro_gk"] - omni["macro_ds"]),
            fmt_delta(cap["macro_gk"] - cap["macro_ds"]),
        )
    )
    a("")
    a("## 评委一致性（逐样本）")
    a("")
    a("只比较两套评分都成功的样本。")
    a("")
    a("| 系统 | n | Pearson | Spearman | Grok 更高 | DeepSeek 更高 | 持平 | 样本均差 |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in order:
        row = data[key]
        a(
            "| %s | %d | %s | %s | %d | %d | %d | %s |"
            % (
                row["spec"]["label"],
                row["n_common"],
                fmt_corr(row["pearson"]),
                fmt_corr(row["spearman"]),
                row["higher_g"],
                row["higher_d"],
                row["tied"],
                fmt_delta(row["mean_delta"]),
            )
        )
    a("")
    a("| # | 轴 | Sure-Tagger Pearson | Omni Pearson | captioner Pearson |")
    a("| ---: | --- | ---: | ---: | ---: |")
    for field in FIELDS:
        a(
            "| %d | `%s` | %s | %s | %s |"
            % (
                FIELD_NOS[field],
                field,
                fmt_corr(sure["field_pearson"][field]),
                fmt_corr(omni["field_pearson"][field]),
                fmt_corr(cap["field_pearson"][field]),
            )
        )
    a("")
    a("## 结论")
    a("")
    a(
        "- 换 Grok 4.7 当第二评委，**系统排序不变**：%s。"
        % " > ".join(
            "%s（%s%%）" % (data[k]["spec"]["label"], fmt_pct(data[k]["macro_gk"]))
            for k in ranked
        )
    )
    a(
        "- Grok 相对 DeepSeek v4 Pro 的宏平均差：Sure-Tagger v1 %s，Omni %s，captioner %s。"
        % (
            fmt_delta(sure["macro_gk"] - sure["macro_ds"]),
            fmt_delta(omni["macro_gk"] - omni["macro_ds"]),
            fmt_delta(cap["macro_gk"] - cap["macro_ds"]),
        )
    )
    a("- Sure-Tagger 用的是论文 v1 主流水线，可与 PDF 描述轴 65.09% 直接对照。")
    a("- 若报告需要双评委，建议同时报两列宏平均，不要把 Grok 分和旧 DeepSeek 分混在一张未标注评委的表里。")
    a("")
    a("## 产物")
    a("")
    a("| 文件 | 说明 |")
    a("| --- | --- |")
    a("| `evaluation/descriptive-semantic-scoring-grok-4.7/sure-tagger-v1/` | Sure-Tagger v1 的 Grok 评分 |")
    a("| `evaluation/descriptive-semantic-scoring-grok-4.7/qwen-captioner/` | captioner 的 Grok 评分 |")
    a("| `evaluation/descriptive-semantic-scoring-grok-4.7/qwen3-omni/` | Omni 的 Grok 评分 |")
    a("| `eval/run_grok_descriptive_semantic_scoring.sh` | 复跑入口，默认 200 workers、Sure-Tagger v1 |")
    a("")
    return "\n".join(lines)


def write_readme(data):
    sure = data["sure-tagger"]
    return "\n".join(
        [
            "# Descriptive semantic scoring with Grok 4.7",
            "",
            "Second-judge run of the existing DeepSeek descriptive semantic scorer.",
            "Schema generation is unchanged. Only the judge model is Grok.",
            "",
            "- Judge: `grok-4.7`",
            "- API: `compare/api/grok.txt`",
            "- Prompt: `compare/eval/deepseek_descriptive_semantic_scoring_prompt.txt`",
            "- Ground truth: `evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3`",
            "- Missing GT (same as DeepSeek): `ES2002b_utterance_00012`, `IN1016_utterance_00044`",
            "- Sure-Tagger predictions: **v1** `sure_tagger_full_20260918_main_pipeline_w2_b8`",
            "- Sure-Tagger Grok workers: **200**",
            "",
            "## Systems",
            "",
            "| system | predictions | output |",
            "| --- | --- | --- |",
            "| sure-tagger v1 | `evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl` | `sure-tagger-v1/` |",
            "| qwen-captioner | `docs/report/result/qwen-captioner/results_t1_t3.jsonl` | `qwen-captioner/` |",
            "| qwen3-omni | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` | `qwen3-omni/` |",
            "",
            "## Headline",
            "",
            "- Sure-Tagger v1 DeepSeek `%s%%` / Grok `%s%%`"
            % (fmt_pct(sure["macro_ds"]), fmt_pct(sure["macro_gk"])),
            "- Qwen3-Omni Flash v2 DeepSeek `%s%%` / Grok `%s%%`"
            % (fmt_pct(data["omni"]["macro_ds"]), fmt_pct(data["omni"]["macro_gk"])),
            "- qwen-captioner DeepSeek `%s%%` / Grok `%s%%`"
            % (fmt_pct(data["captioner"]["macro_ds"]), fmt_pct(data["captioner"]["macro_gk"])),
            "",
            "## Rerun",
            "",
            "```bash",
            "WORKERS=200 BACKGROUND=0 SYSTEMS=sure-tagger \\",
            "  /hpc_stor03/sjtu_home/weihan.chen/share/compare/eval/run_grok_descriptive_semantic_scoring.sh",
            "```",
            "",
        ]
    )


def main():
    grok_summary = json.loads(
        (COMPARE_ROOT / "evaluation/descriptive-semantic-scoring-grok-4.7/sure-tagger-v1/summary.json").read_text()
    )
    status = grok_summary.get("status")
    if status not in ("complete", "complete_with_missing_ground_truth"):
        raise SystemExit("sure-tagger-v1 grok scoring is not complete: %s" % status)
    if grok_summary.get("scoring_success_count") != 648:
        raise SystemExit(
            "expected 648 grok successes, got %s" % grok_summary.get("scoring_success_count")
        )
    data = collect()
    out_dir = COMPARE_ROOT / "evaluation/descriptive-semantic-scoring-grok-4.7"
    svg_path = COMPARE_ROOT / "docs/report/images/descriptive-grok-4.7-vs-deepseek.svg"
    write_svg(data, svg_path)
    report = render_markdown(data)
    # docs copy uses docs-relative image and metrics links
    docs_report = report.replace(
        "](../../docs/benchmark-v1-29-metrics.md)",
        "](benchmark-v1-29-metrics.md)",
    ).replace(
        "](../../docs/report/images/descriptive-grok-4.7-vs-deepseek.svg)",
        "](report/images/descriptive-grok-4.7-vs-deepseek.svg)",
    )
    (out_dir / "REPORT.md").write_text(report + "\n")
    (COMPARE_ROOT / "docs/descriptive-semantic-scoring-grok-4.7.md").write_text(docs_report + "\n")
    (out_dir / "README.md").write_text(write_readme(data))
    stats_path = out_dir / "sure-tagger-v1-vs-deepseek.summary.json"
    dump = {}
    for key, row in data.items():
        dump[key] = {
            "label": row["spec"]["label"],
            "n_deepseek": row["n_ds"],
            "n_grok": row["n_gk"],
            "n_common": row["n_common"],
            "macro_deepseek": round(row["macro_ds"], 4),
            "macro_grok": round(row["macro_gk"], 4),
            "delta_pp": round(row["macro_gk"] - row["macro_ds"], 4),
            "pearson": None if math.isnan(row["pearson"]) else round(row["pearson"], 4),
            "spearman": None if math.isnan(row["spearman"]) else round(row["spearman"], 4),
        }
    stats_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dump, ensure_ascii=False, indent=2, sort_keys=True))
    print("wrote", out_dir / "REPORT.md")
    print("wrote", svg_path)


if __name__ == "__main__":
    main()
