# 人工分层偏好：评者一致性与系统自动分对齐

数据：`dataset/benchmark/demo/answer`。四套题共 100 道，每题两人独立作答。选项为 A Sure-Tagger、B Captioner、C Omni、D 无法判断。

本文同时报告两件事：

1. **人与人**：100 题双编码的观察一致率、Cohen $\kappa$、Gwet AC1。
2. **人与系统**：把三套系统的自动得分变成 A/B/C 第一名，再和人类选择比。

- **L1 / L2 / L3 / L5**：用 Benchmark v1 结构 19 轴，在该样本、该层上对三套系统打分，取 argmax。计分与 [`benchmark-v1-29-metrics.md`](benchmark-v1-29-metrics.md) 相同：已知且对得 2 分，显式 `null` 得 1 分，缺失/错误得 0 分；层内对有 GT 的轴做微平均得到得分率。
- **D**：仍用描述 10 轴的 DeepSeek v4 Pro / Grok 4.7 归一化得分率 argmax。这两个评委**没有**给 L1–L5 打分。
- T2 的 L2/L3/L5、T3 的 L1/L2/L3 官方没有该层 GT，自动裁判记为 NA，**不计入**人机一致率的分母。

Sure-Tagger 为论文 **v1**（`result_with_id_descriptive.jsonl`，语种已经核验）。Captioner / Omni 结构预测分别来自 `docs/report/result/qwen-captioner/results_t1_t3.jsonl` 与 `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl`。

逐题表：[`human-preference-item-table.csv`](../dataset/benchmark/demo/docs/human-preference-item-table.csv)；结构分层分：[`structural-layer-scores.json`](../dataset/benchmark/demo/docs/structural-layer-scores.json)。

## 评者配对

| 套 | 评者 1 | 评者 2 | 题数 |
| --- | --- | --- | ---: |
| 第 1 套 | `gaoshuo` | `whf` | 25 |
| 第 2 套 | `bicheng` | `whf` | 25 |
| 第 3 套 | `djh` | `wenwen` | 25 |
| 第 4 套 | `chaolei` | `djh` | 25 |

`whf` 做了第 1、2 套，`djh` 做了第 3、4 套。总体 kappa 是四对评者的双编码题池，不是同一对人标完全部 100 题。

## 人类选择分布

| 选项 | 次数 | 占比 |
| --- | ---: | ---: |
| A Sure-Tagger | 159 | 79.5% |
| B Captioner | 1 | 0.5% |
| C Omni | 23 | 11.5% |
| D 无法判断 | 17 | 8.5% |
| 合计 | 200 | 100.0% |

两人一致的 72 题里，共识为 A 67、B 0、C 4、D 1。

## 评者一致性

A 占 79.5%，Cohen $\kappa$ 会被 kappa 悖论压低；Gwet AC1 更稳。Landis & Koch：κ 0.21–0.40 一般，0.41–0.60 中等，0.61–0.80 较充分。

### 总体

| 指标 | 数值 |
| --- | --- |
| 题数 | 100 |
| 两人相同 | 72 / 100 (72.0%) |
| Cohen $\kappa$ | 0.198（轻微） |
| Gwet AC1 | 0.683 |
| 双方都选 A/B/C 的题 | 84 / 100 (84.0%)，其中相同 71 / 84 (84.5%) |

### 分套

| 套 | 评者 | 相同 | $P_o$ | Cohen $\kappa$ | Gwet AC1 |
| --- | --- | --- | ---: | ---: | ---: |
| 第 1 套 | `gaoshuo` vs `whf` | 18 / 25 (72.0%) | 72.0% | 0.371 | 0.672 |
| 第 2 套 | `bicheng` vs `whf` | 18 / 25 (72.0%) | 72.0% | 0.186 | 0.684 |
| 第 3 套 | `djh` vs `wenwen` | 17 / 25 (68.0%) | 68.0% | 0.156 | 0.639 |
| 第 4 套 | `chaolei` vs `djh` | 19 / 25 (76.0%) | 76.0% | 0.080 | 0.738 |

### 分层

| 层 | 题数 | 相同 | $P_o$ | Cohen $\kappa$ | Gwet AC1 |
| --- | ---: | --- | ---: | ---: | ---: |
| L1 词汇内容 | 20 | 13 / 20 (65.0%) | 65.0% | 0.343 | 0.575 |
| L2 说话人 | 20 | 18 / 20 (90.0%) | 90.0% | -0.053 | 0.897 |
| L3 时间结构 | 20 | 16 / 20 (80.0%) | 80.0% | -0.081 | 0.787 |
| L5 声学场景与事件 | 20 | 13 / 20 (65.0%) | 65.0% | 0.136 | 0.595 |
| D 描述性 | 20 | 12 / 20 (60.0%) | 60.0% | -0.000 | 0.542 |

### 分轨道

| 轨道 | 题数 | 相同 | $P_o$ | Cohen $\kappa$ | Gwet AC1 |
| --- | ---: | --- | ---: | ---: | ---: |
| `T1_natural_meeting_utterance` | 55 | 44 / 55 (80.0%) | 80.0% | 0.268 | 0.780 |
| `T2_clean_librispeech` | 10 | 6 / 10 (60.0%) | 60.0% | 0.245 | 0.527 |
| `T2_multilingual_fleurs` | 15 | 10 / 15 (66.7%) | 66.7% | 0.128 | 0.618 |
| `T3_events_noise` | 20 | 12 / 20 (60.0%) | 60.0% | 0.096 | 0.533 |

### 混淆（评者1 × 评者2）

| 评者1 \ 评者2 | A | B | C | D | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A** | 67 | 1 | 8 | 7 | 83 |
| **B** | 0 | 0 | 0 | 0 | 0 |
| **C** | 4 | 0 | 4 | 2 | 10 |
| **D** | 5 | 0 | 1 | 1 | 7 |
| **合计** | 76 | 1 | 13 | 10 | 100 |

不一致主要是 A↔C 和 A↔D。B 几乎无人选。

## 系统自动分 vs 人类

### 结构层可评范围

80 道结构题（20 条 × L1/L2/L3/L5）里：

| 自动裁判 | 题数 | 说明 |
| --- | ---: | --- |
| 唯一第一名 | 51 | 计入人机一致率 |
| 并列第一 | 2 | 不计入严格一致率 |
| 无该层 GT（NA） | 27 | T2 的 L2/L3/L5、T3 的 L1/L2/L3 |

Captioner 结构字段大量显式 `null`，官方规则记 50% 半分。Sure-Tagger 只要高于 50% 就会在该层胜出。这是规则使然，不是 Captioner 得了「真的 50 分」。

### 主表：人类 vs 自动第一名

「评分」= 该层可裁判题 × 2 人。D 用描述评委；L1–L5 用结构 argmax。

| 范围 | 自动裁判 | 可评题 | 两人评分对齐 | 去掉人类 D | 两人共识且非 D |
| --- | --- | ---: | --- | --- | --- |
| L1–L5 合计 | 结构 19 轴 | 51 | 83 / 102 (81.4%) | 83 / 97 (85.6%) | 36 / 39 (92.3%) |
| L1 词汇内容 | 结构（NA 4 / 并列 1） | 15 | 20 / 30 (66.7%) | 20 / 30 (66.7%) | 8 / 10 (80.0%) |
| L2 说话人 | 结构（NA 9 / 并列 1） | 10 | 19 / 20 (95.0%) | 19 / 20 (95.0%) | 9 / 9 (100.0%) |
| L3 时间结构 | 结构（NA 9 / 并列 0） | 11 | 21 / 22 (95.5%) | 21 / 22 (95.5%) | 10 / 10 (100.0%) |
| L5 声学场景与事件 | 结构（NA 5 / 并列 0） | 15 | 23 / 30 (76.7%) | 23 / 25 (92.0%) | 9 / 10 (90.0%) |
| D 描述性 | DeepSeek 描述分 | 20 | 21 / 40 (52.5%) | 21 / 35 (60.0%) | 8 / 12 (66.7%) |
| D 描述性 | Grok 描述分 | 20 | 19 / 40 (47.5%) | 19 / 35 (54.3%) | 7 / 12 (58.3%) |
| 100 题混合 | 结构 + DeepSeek(D) | 71 | 104 / 142 (73.2%) | 104 / 132 (78.8%) | 44 / 51 (86.3%) |
| 100 题混合 | 结构 + Grok(D) | 71 | 102 / 142 (71.8%) | 102 / 132 (77.3%) | 43 / 51 (84.3%) |

结构层人机一致率明显高于 D 层：人类本来就大量选 A，结构自动第一名也大量是 A。描述评委则经常把 Omni 排第一，所以 D 层人对不齐。

### 人类选择 vs 结构第一名（可评题，两人评分）

| 人类 \ 结构第一 | A | B | C | 合计 |
| --- | ---: | ---: | ---: | ---: |
| **A** | 77 | 1 | 3 | 81 |
| **B** | 0 | 0 | 0 | 0 |
| **C** | 9 | 1 | 6 | 16 |
| **D** | 4 | 0 | 1 | 5 |
| **合计** | 90 | 2 | 10 | 102 |

### 20 条样本的结构层得分率

| 样本 | L1 | L2 | L3 | L5 |
| --- | --- | --- | --- | --- |
| `EN2001a_utterance_00162` | A 50/50/44 | A 90/50/30 | A 64/50/34 | A 97/50/50 |
| `EN2001e_utterance_00071` | A 71/50/53 | A 100/50/70 | A 69/50/61 | A 90/50/50 |
| `ES2009d_utterance_00020` | A 80/50/64 | A 90/60/30 | A 100/50/35 | A 99/50/50 |
| `ES2012d_utterance_00018` | A 92/50/57 | A 90/60/50 | A 80/50/54 | A 83/50/50 |
| `ES2015a_utterance_00006` | A 97/65/60 | A 70/30/30 | A 100/37/37 | A 93/50/50 |
| `ES2015b_utterance_00031` | A 89/50/66 | A 70/50/30 | A 74/50/29 | A 75/50/50 |
| `ES2016c_utterance_00009` | C 78/50/94 | A 100/50/90 | A 100/50/88 | A 64/50/50 |
| `ES2016d_utterance_00015` | C 90/50/91 | A 70/50/50 | A 81/50/57 | A 88/50/50 |
| `IS1006b_utterance_00045` | B 42/50/47 | A 90/50/30 | A 99/50/34 | A 98/50/50 |
| `TS3004c_utterance_00089` | C 73/50/91 | A 73/50/30 | A 73/50/34 | A 90/50/50 |
| `TS3006a_utterance_00015` | A 78/50/48 | TIE 70/70/30 | A 55/50/20 | C 60/50/69 |
| `fleurs_t2_0055` | A 98/50/91 | NA | NA | NA |
| `fleurs_t2_0059` | A 99/50/90 | NA | NA | NA |
| `fleurs_t2_0347` | C 79/50/99 | NA | NA | NA |
| `librispeech_t2_0048` | A 100/50/99 | NA | NA | NA |
| `librispeech_t2_0096` | TIE 97/50/97 | NA | NA | NA |
| `t3_desed_0112` | NA | NA | NA | A 94/50/83 |
| `t3_desed_0156` | NA | NA | NA | A 94/50/50 |
| `t3_fsd50k_0045` | NA | NA | NA | A 56/50/50 |
| `t3_fsd50k_0101` | NA | NA | NA | A 100/50/50 |
单元格为「第一名 A/B/C 得分率%」。NA 表示该层无官方 GT。

### D 层 vs DeepSeek / Grok

20 条音频上 DeepSeek 与 Grok 没有并列第一；第一名相同 17 / 20 (85.0%)。

| 评委第一名 | DeepSeek | Grok |
| --- | ---: | ---: |
| A Sure-Tagger | 10 | 9 |
| B Captioner | 1 | 1 |
| C Omni | 9 | 10 |

第一名不同的样本：

| sample_id | DeepSeek（A/B/C %） | Grok（A/B/C %） |
| --- | --- | --- |
| `ES2012d_utterance_00018` | A 75.5 / 57.2 / 60.2 | C 74.4 / 60.1 / 75.0 |
| `ES2015a_utterance_00006` | C 73.3 / 36.2 / 77.2 | A 79.9 / 31.2 / 77.3 |
| `TS3004c_utterance_00089` | A 70.5 / 58.2 / 64.5 | C 69.7 / 58.1 / 70.4 |

人类 D 层 40 个评分：A 31、B 1、C 3、D 5。

| 套 | 题 | 样本 | 人1 | 人2 | 结构第一 | DeepSeek | Grok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 套 | Q04 | `fleurs_t2_0055` | D | A | — | A | A |
| 第 1 套 | Q08 | `IS1006b_utterance_00045` | A | A | — | A | A |
| 第 1 套 | Q11 | `ES2016c_utterance_00009` | C | D | — | C | C |
| 第 1 套 | Q16 | `EN2001e_utterance_00071` | A | C | — | C | C |
| 第 1 套 | Q25 | `t3_fsd50k_0101` | A | A | — | B | B |
| 第 2 套 | Q03 | `ES2015b_utterance_00031` | A | A | — | A | A |
| 第 2 套 | Q09 | `t3_fsd50k_0045` | A | B | — | C | C |
| 第 2 套 | Q15 | `ES2016d_utterance_00015` | A | D | — | A | A |
| 第 2 套 | Q17 | `fleurs_t2_0347` | A | C | — | C | C |
| 第 2 套 | Q22 | `ES2009d_utterance_00020` | A | A | — | A | A |
| 第 3 套 | Q04 | `librispeech_t2_0096` | A | A | — | C | C |
| 第 3 套 | Q10 | `EN2001a_utterance_00162` | A | A | — | A | A |
| 第 3 套 | Q12 | `ES2015a_utterance_00006` | A | A | — | C | A |
| 第 3 套 | Q18 | `librispeech_t2_0048` | A | D | — | C | C |
| 第 3 套 | Q21 | `fleurs_t2_0059` | A | D | — | C | C |
| 第 4 套 | Q03 | `TS3006a_utterance_00015` | A | A | — | A | A |
| 第 4 套 | Q09 | `t3_desed_0156` | A | A | — | A | A |
| 第 4 套 | Q15 | `ES2012d_utterance_00018` | A | A | — | A | C |
| 第 4 套 | Q17 | `t3_desed_0112` | A | A | — | C | C |
| 第 4 套 | Q21 | `TS3004c_utterance_00089` | A | A | — | A | C |

## 两人不一致的题

共 28 / 100 题两人答案不同。

| 套 | 题 | 层 | 样本 | 评者1 | 评者2 | 结构第一 | DeepSeek | Grok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 套 | Q03 | L1 | `IS1006b_utterance_00045` | gaoshuo `A` | whf `C` | B | A | A |
| 第 1 套 | Q04 | D | `fleurs_t2_0055` | gaoshuo `D` | whf `A` | — | A | A |
| 第 1 套 | Q05 | L3 | `t3_fsd50k_0101` | gaoshuo `D` | whf `A` | NA | B | B |
| 第 1 套 | Q11 | D | `ES2016c_utterance_00009` | gaoshuo `C` | whf `D` | — | C | C |
| 第 1 套 | Q15 | L5 | `t3_fsd50k_0101` | gaoshuo `A` | whf `D` | A | B | B |
| 第 1 套 | Q16 | D | `EN2001e_utterance_00071` | gaoshuo `A` | whf `C` | — | C | C |
| 第 1 套 | Q19 | L1 | `fleurs_t2_0055` | gaoshuo `A` | whf `C` | A | A | A |
| 第 2 套 | Q08 | L1 | `ES2015b_utterance_00031` | bicheng `C` | whf `A` | A | A | A |
| 第 2 套 | Q09 | D | `t3_fsd50k_0045` | bicheng `A` | whf `B` | — | C | C |
| 第 2 套 | Q11 | L1 | `ES2009d_utterance_00020` | bicheng `C` | whf `A` | A | A | A |
| 第 2 套 | Q12 | L5 | `fleurs_t2_0347` | bicheng `C` | whf `A` | NA | C | C |
| 第 2 套 | Q15 | D | `ES2016d_utterance_00015` | bicheng `A` | whf `D` | — | A | A |
| 第 2 套 | Q17 | D | `fleurs_t2_0347` | bicheng `A` | whf `C` | — | C | C |
| 第 2 套 | Q19 | L1 | `t3_fsd50k_0045` | bicheng `C` | whf `D` | NA | C | C |
| 第 3 套 | Q09 | L5 | `librispeech_t2_0096` | djh `A` | wenwen `D` | NA | C | C |
| 第 3 套 | Q17 | L5 | `ES2015a_utterance_00006` | djh `A` | wenwen `D` | A | C | A |
| 第 3 套 | Q18 | D | `librispeech_t2_0048` | djh `A` | wenwen `D` | — | C | C |
| 第 3 套 | Q19 | L3 | `librispeech_t2_0096` | djh `A` | wenwen `D` | NA | C | C |
| 第 3 套 | Q20 | L2 | `EN2001a_utterance_00162` | djh `A` | wenwen `C` | A | A | A |
| 第 3 套 | Q21 | D | `fleurs_t2_0059` | djh `A` | wenwen `D` | — | C | C |
| 第 3 套 | Q24 | L1 | `librispeech_t2_0096` | djh `A` | wenwen `C` | TIE | C | C |
| 第 3 套 | Q25 | L3 | `EN2001a_utterance_00162` | djh `A` | wenwen `C` | A | A | A |
| 第 4 套 | Q04 | L5 | `t3_desed_0156` | chaolei `D` | djh `A` | A | A | A |
| 第 4 套 | Q06 | L1 | `TS3004c_utterance_00089` | chaolei `A` | djh `C` | C | A | C |
| 第 4 套 | Q07 | L3 | `t3_desed_0112` | chaolei `D` | djh `A` | NA | C | C |
| 第 4 套 | Q12 | L5 | `t3_desed_0112` | chaolei `D` | djh `A` | A | C | C |
| 第 4 套 | Q14 | L2 | `t3_desed_0156` | chaolei `C` | djh `A` | NA | A | A |
| 第 4 套 | Q23 | L5 | `TS3006a_utterance_00015` | chaolei `D` | djh `C` | C | A | A |

## 结论

- 人类高度偏向 Sure-Tagger：200 个评分里 A 占 79.5%，B 只有 1 次。
- 两人相同 72 / 100 (72.0%)。Cohen κ=0.198（轻微）；Gwet AC1=0.683。
- **结构层 ↔ 人类**：有唯一结构第一名的题上，人类评分对齐 83 / 102 (81.4%)；去掉无法判断后 83 / 97 (85.6%)；两人共识且非 D 时 36 / 39 (92.3%)。自动第一名也以 A 为主，和人类同向。
- **描述层 ↔ 人类**：D 对 DeepSeek 21 / 40 (52.5%)，对 Grok 19 / 40 (47.5%)。人类偏 A，评委常把 Omni 排第一，两边不是同一排序。
- **混合人机**（L1–L5 用结构分，D 用 DeepSeek）104 / 142 (73.2%)；若 D 改用 Grok 则 102 / 142 (71.8%)。
- 80 道结构题中 27 道无该层 GT（NA），不能做人机对比。Captioner 结构分接近 50% 是 `null` 半分，不是识别准确率。

## 数据与口径

| 项 | 路径 / 说明 |
| --- | --- |
| 人类答案 | `dataset/benchmark/demo/answer/` |
| 结构计分 | `eval/score_benchmark_v1.py`，单样本层微平均 |
| Sure-Tagger v1 | `evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl` |
| Captioner | `docs/report/result/qwen-captioner/results_t1_t3.jsonl` |
| Omni | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` |
| 描述分 | 各样本 `comparison.json`（DeepSeek v4 Pro / Grok 4.7） |
| 逐题表 | `dataset/benchmark/demo/docs/human-preference-item-table.csv` |
| 结构分层分 | `dataset/benchmark/demo/docs/structural-layer-scores.json` |

