# 描述性语义打分：Grok 4.7 作为第二评委

口径与 [`benchmark-v1-29-metrics.md`](benchmark-v1-29-metrics.md) 一致：描述 10 轴由评委打 $u\in[0,2]$，再 $r_f=\bar u_f/2$。百分数为得分率 $\times 100\%$。官方聚合只用评分成功样本：650 条中 648 条成功，2 条缺描述性 GT（`ES2002b_utterance_00012`、`IN1016_utterance_00044`），不记 0 分。

本实验**不改 schema 生成**：三套系统的描述性标签仍由原流程产出。只把评委从 DeepSeek v4 Pro 换成 Grok 4.7，prompt、GT、字段合同完全相同。

Sure-Tagger 使用论文/PDF 的 **v1**（`sure_tagger_full_20260918_main_pipeline_w2_b8`，DeepSeek 描述 10 轴 65.09%）。

## 实验设定

- 评委 A：`deepseek-v4-pro`（既有结果，thinking enabled）
- 评委 B：`grok-4.7`（`compare/api/grok.txt`，thinking enabled）
- Prompt：`eval/deepseek_descriptive_semantic_scoring_prompt.txt`（`prompt_sha256=2e49b92b25e334f195b9526f1c171bd39e6285646dfa329214ca1d6f6be7ce9a`）
- GT：`evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3`
- 实现：`eval/run_descriptive_semantic_scoring_batch.py`，启动脚本 `eval/run_grok_descriptive_semantic_scoring.sh`
- Grok 并发：Sure-Tagger v1 全量 **200 workers**；captioner / Omni 沿用既有全量结果
- Grok 产物：`evaluation/descriptive-semantic-scoring-grok-4.7/{sure-tagger-v1,qwen-captioner,qwen3-omni}`

| 系统 | 被评预测 | DeepSeek 评分目录 |
| --- | --- | --- |
| Sure-Tagger v1 | `evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl` | 同目录 `descriptive_score/` |
| Qwen3-Omni Flash v2 | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` | `docs/report/result/descriptive/qwen3-omni-v2-multi-api12/score/` |
| qwen-captioner | `docs/report/result/qwen-captioner/results_t1_t3.jsonl` | `docs/report/result/descriptive/qwen-captioner/` |

## 描述 10 轴宏平均

![DeepSeek vs Grok 宏平均](report/images/descriptive-grok-4.7-vs-deepseek.svg)

| 系统 | DeepSeek v4 Pro | Grok 4.7 | Δ（Grok − DeepSeek） |
| --- | ---: | ---: | ---: |
| Sure-Tagger v1 | 65.09% | **66.45%** | +1.35 pp |
| Qwen3-Omni Flash v2 | 62.89% | **64.90%** | +2.01 pp |
| qwen-captioner | 48.44% | **50.72%** | +2.29 pp |

排序在两个评委下完全一致：**Sure-Tagger v1 > Qwen3-Omni Flash v2 > qwen-captioner**。

## 分轴得分率

### DeepSeek v4 Pro

| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |
| ---: | --- | ---: | ---: | ---: |
| 20 | `caption` | 69.26% | 58.69% | 45.47% |
| 21 | `topic` | 79.58% | 76.51% | 65.39% |
| 22 | `background_condition` | 70.08% | 78.46% | 48.50% |
| 23 | `dominant_content_impression` | 76.25% | 72.22% | 52.76% |
| 24 | `clarity_intelligibility_impression` | 69.36% | 80.21% | 49.88% |
| 25 | `background_sound_summary` | 68.48% | 64.78% | 48.73% |
| 26 | `descriptive_phrases` | 55.59% | 40.05% | 34.25% |
| 27 | `facets` | 58.53% | 52.11% | 42.61% |
| 28 | `speaker_descriptions` | 41.94% | 52.74% | 49.23% |
| 29 | `scene_hypotheses` | 61.88% | 53.09% | 47.55% |
|  | **描述 10 轴宏平均** | **65.09%** | **62.89%** | **48.44%** |

### Grok 4.7

| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |
| ---: | --- | ---: | ---: | ---: |
| 20 | `caption` | 68.49% | 63.50% | 50.41% |
| 21 | `topic` | 79.36% | 75.99% | 67.29% |
| 22 | `background_condition` | 69.96% | 78.95% | 48.64% |
| 23 | `dominant_content_impression` | 75.34% | 72.29% | 56.20% |
| 24 | `clarity_intelligibility_impression` | 70.99% | 78.90% | 49.38% |
| 25 | `background_sound_summary` | 67.44% | 65.95% | 48.46% |
| 26 | `descriptive_phrases` | 57.37% | 45.45% | 42.87% |
| 27 | `facets` | 61.54% | 56.56% | 48.30% |
| 28 | `speaker_descriptions` | 43.51% | 54.12% | 45.66% |
| 29 | `scene_hypotheses` | 70.48% | 57.25% | 50.02% |
|  | **描述 10 轴宏平均** | **66.45%** | **64.90%** | **50.72%** |

### 分轴 Δ（Grok − DeepSeek，pp）

| # | 轴 | Sure-Tagger v1 | Qwen3-Omni Flash v2 | qwen-captioner |
| ---: | --- | ---: | ---: | ---: |
| 20 | `caption` | -0.78 pp | +4.81 pp | +4.94 pp |
| 21 | `topic` | -0.22 pp | -0.53 pp | +1.90 pp |
| 22 | `background_condition` | -0.12 pp | +0.49 pp | +0.15 pp |
| 23 | `dominant_content_impression` | -0.91 pp | +0.07 pp | +3.44 pp |
| 24 | `clarity_intelligibility_impression` | +1.63 pp | -1.30 pp | -0.50 pp |
| 25 | `background_sound_summary` | -1.04 pp | +1.17 pp | -0.28 pp |
| 26 | `descriptive_phrases` | +1.78 pp | +5.40 pp | +8.61 pp |
| 27 | `facets` | +3.01 pp | +4.46 pp | +5.70 pp |
| 28 | `speaker_descriptions` | +1.56 pp | +1.38 pp | -3.57 pp |
| 29 | `scene_hypotheses` | +8.60 pp | +4.15 pp | +2.48 pp |
|  | **宏平均** | **+1.35 pp** | **+2.01 pp** | **+2.29 pp** |

## 评委一致性（逐样本）

只比较两套评分都成功的样本。

| 系统 | n | Pearson | Spearman | Grok 更高 | DeepSeek 更高 | 持平 | 样本均差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sure-Tagger v1 | 648 | 0.929 | 0.924 | 396 | 250 | 2 | +1.35 pp |
| Qwen3-Omni Flash v2 | 648 | 0.881 | 0.890 | 388 | 219 | 41 | +2.01 pp |
| qwen-captioner | 648 | 0.764 | 0.753 | 418 | 219 | 11 | +2.29 pp |

| # | 轴 | Sure-Tagger Pearson | Omni Pearson | captioner Pearson |
| ---: | --- | ---: | ---: | ---: |
| 20 | `caption` | 0.903 | 0.797 | 0.789 |
| 21 | `topic` | 0.956 | 0.833 | 0.815 |
| 22 | `background_condition` | 0.895 | 0.857 | 0.630 |
| 23 | `dominant_content_impression` | 0.915 | 0.815 | 0.561 |
| 24 | `clarity_intelligibility_impression` | 0.835 | 0.806 | 0.118 |
| 25 | `background_sound_summary` | 0.921 | 0.647 | 0.601 |
| 26 | `descriptive_phrases` | 0.863 | 0.796 | 0.654 |
| 27 | `facets` | 0.858 | 0.836 | 0.089 |
| 28 | `speaker_descriptions` | 0.711 | 0.894 | 0.093 |
| 29 | `scene_hypotheses` | 0.875 | 0.724 | 0.384 |

## 结论

- 换 Grok 4.7 当第二评委，**系统排序不变**：Sure-Tagger v1（66.45%） > Qwen3-Omni Flash v2（64.90%） > qwen-captioner（50.72%）。
- Grok 相对 DeepSeek v4 Pro 的宏平均差：Sure-Tagger v1 +1.35 pp，Omni +2.01 pp，captioner +2.29 pp。
- Sure-Tagger 用的是论文 v1 主流水线，可与 PDF 描述轴 65.09% 直接对照。
- 若报告需要双评委，建议同时报两列宏平均，不要把 Grok 分和旧 DeepSeek 分混在一张未标注评委的表里。

## 产物

| 文件 | 说明 |
| --- | --- |
| `evaluation/descriptive-semantic-scoring-grok-4.7/sure-tagger-v1/` | Sure-Tagger v1 的 Grok 评分 |
| `evaluation/descriptive-semantic-scoring-grok-4.7/qwen-captioner/` | captioner 的 Grok 评分 |
| `evaluation/descriptive-semantic-scoring-grok-4.7/qwen3-omni/` | Omni 的 Grok 评分 |
| `eval/run_grok_descriptive_semantic_scoring.sh` | 复跑入口，默认 200 workers、Sure-Tagger v1 |
