# Benchmark v1.0 各系统标签 null 比例

更新时间：2026-09-20  
数据版本：`structural-benchmark-subset-v1.0`（650 条，T1–T3）  
对应报告：[`docs/report/pdf/benchmark-v1-current.pdf`](report/pdf/benchmark-v1-current.pdf)  
评分规范：`benchmark-scoring-v1.0`；描述性 10 轴按 JSON 显式 `null` 统计

本文统计 PDF 中三套系统——**Sure-Tagger**、**qwen-captioner**、**Qwen3-Omni Flash v2**——在每个标签上的显式 `null` 比例。结构字段以 scorer 的 `result_null` 为准；描述性字段直接扫描预测 JSONL。

---

## 1. 统计口径

| 状态 | 含义 | 是否计入 null |
| --- | --- | --- |
| `result_known` | 字段存在且通过 schema 规范化 | 否 |
| `result_null` | 合法显式弃答：JSON `null`，或父对象为 `null` 导致叶子一并弃答 | **是** |
| `schema_invalid` | 有值但类型/枚举/范围非法 | 否（单独列出） |
| `[]` / `0` / `false` | 已确认空集、零或条件不成立 | 否 |
| `result_missing` / `runtime_failed` | 字段缺失或该样本无预测 | 本轮三套均为 0 |

两种分母：

1. **全量 650**：系统实际写出了多少 `null`，含 T3 无语音样本上的说话人/ASR 弃答。
2. **有效评分样本**：该字段 `gt_mask = true` 且进入 PDF 得分的样本。这才是「半分基线」真正作用的范围。

显式 `null` 在结构轴与描述轴都记 $s=1$（满分 2 的 50%）。因此接近 50% 的轴分**可能只是弃答基线**，不是识别准确率。

预测来源：

| 系统 | 结构 / 描述预测 | 结构评分 |
| --- | --- | --- |
| Sure-Tagger | `docs/report/result/sure-tagger/result_with_id_descriptive.jsonl` | `score.json` |
| qwen-captioner | `docs/report/result/qwen-captioner/results.jsonl` | `score.json` |
| Qwen3-Omni Flash v2 | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` | `structural_score.json` |

---

## 2. 结论摘要

1. **qwen-captioner 以弃答为主。** 29 轴全量宏平均 null 率 **78.3%**，结构 19 轴 **89.7%**。有效评分样本上结构 19 轴仍有 **87.0%** 为 `null`。PDF 中 captioner 结构 52.70%、总分 51.23%，大部分是 50% 半分，而不是「约一半标签预测正确」。
2. **Qwen3-Omni 词汇轴能写、声学轴常弃。** 全量 29 轴宏平均 null 率 **29.8%**。L1 有效样本仅 **5.1%** null，但 `nonspeech_ratio` 有效样本 **94.9%** null、`speaker_change_rate_per_min` **86.9%** null、`external_noise_type` **68.7%** null。L5 有效样本宏平均 null 率 **85.2%**，与 PDF 中 Omni L5 仅 55.20% 一致。
3. **Sure-Tagger 几乎总给值。** 全量 29 轴宏平均 null 率 **5.6%**；有效评分样本上结构 19 轴降至 **2.2%**。全量里 L1 约 15% 的 ASR/词法 null 主要来自 T3 无语音，不进入 L1 得分。真正有影响的是 T1 `speaker.gender`：**133/350 = 38.0%** 显式弃答。
4. **描述性轴三套差异最大的是结构化子字段。** caption 本身很少为 null（Sure-Tagger 0%、captioner 1.7%、Omni 22.2%），但 captioner 的 `facets` 96.3%、`scene_hypotheses` 93.7%、`descriptive_phrases` 82.0% 为 null。Omni 在 T3 的 `caption`/`topic` 大量弃答（约 80–99%），拉高描述轴全量 null 率到 14.6%。
5. **`language_content.language` 对 Sure-Tagger 不是 null 问题，是非法枚举。** 全量 0 null，但 **116/650 = 17.8%** `schema_invalid`（如 `nn`、`war`、`zh-mandarin`）。有效 L1 样本上非法值降为 7/500。captioner 该轴 1 条非法、其余几乎全是 null。

分层全量 650 宏平均 null 率：

| 层 | 轴数 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: | ---: |
| L1 词汇内容 | 6 | 13.4% | 92.0% | 25.6% |
| L2 说话人 | 5 | 5.5% | 83.3% | 33.2% |
| L3 时间结构 | 4 | 3.7% | 95.3% | 41.5% |
| L5 声学场景与事件 | 4 | 0.0% | 88.7% | 58.0% |
| D 描述性 | 10 | 3.9% | 56.7% | 14.6% |
| **29 轴宏平均** | **29** | **5.6%** | **78.3%** | **29.8%** |

有效评分样本上结构 19 轴宏平均：Sure-Tagger **2.2%**，captioner **87.0%**，Omni **30.5%**。

---

## 3. 二十九轴：全量 650 样本

分母一律 650。单元格为 `null数 (比例)`。Sure-Tagger 的 `language` 另标非法值。

### 3.1 结构 19 轴

| 层 | 标签 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | --- | ---: | ---: | ---: |
| L1 | `speaker.asr_transcript` | 98 (15.1%) | 591 (90.9%) | 162 (24.9%) |
| L1 | `language_content.language` | 0 (0.0%)；非法 116 (17.8%) | 577 (88.8%)；非法 1 | 163 (25.1%) |
| L1 | `language_content.word_count` | 98 (15.1%) | 603 (92.8%) | 162 (24.9%) |
| L1 | `language_content.repetition.has_repetition` | 98 (15.1%) | 600 (92.3%) | 175 (26.9%) |
| L1 | `language_content.repetition.repetition_count` | 98 (15.1%) | 610 (93.8%) | 175 (26.9%) |
| L1 | `deterministic_derived.filler_ratio` | 131 (20.2%) | 606 (93.2%) | 162 (24.9%) |
| L2 | `speaker.speaker_count` | 6 (0.9%) | 576 (88.6%) | 160 (24.6%) |
| L2 | `speaker.speaker_present` | 6 (0.9%) | 406 (62.5%) | 133 (20.5%) |
| L2 | `speaker.multi_speaker` | 6 (0.9%) | 528 (81.2%) | 160 (24.6%) |
| L2 | `speaker.speaker_overlap` | 6 (0.9%) | 583 (89.7%) | 160 (24.6%) |
| L2 | `speaker.gender` | 154 (23.7%) | 614 (94.5%) | 465 (71.5%) |
| L3 | `speaker.speaker_change_count` | 6 (0.9%) | 628 (96.6%) | 160 (24.6%) |
| L3 | `speaker.speaker_change` | 6 (0.9%) | 565 (86.9%) | 160 (24.6%) |
| L3 | `speaker.overlap_ratio` | 77 (11.8%) | 635 (97.7%) | 160 (24.6%) |
| L3 | `deterministic_derived.speaker_change_rate_per_min` | 6 (0.9%) | 650 (100.0%) | 599 (92.2%) |
| L5 | `basic_acoustic.nonspeech_ratio` | 0 (0.0%) | 650 (100.0%) | 630 (96.9%) |
| L5 | `sound_field_scene.speech_music_events` | 0 (0.0%) | 512 (78.8%) | 159 (24.5%) |
| L5 | `sound_field_scene.music_present` | 0 (0.0%) | 532 (81.8%) | 151 (23.2%) |
| L5 | `sound_field_scene.external_noise_type` | 0 (0.0%) | 613 (94.3%) | 567 (87.2%) |
|  | **结构 19 轴宏平均** | **6.4%** | **89.7%** | **37.8%** |

Sure-Tagger 的 L1 词法/ASR 叶子 98 条 null，与父对象 `language_content.repetition` / 无转写样本对齐：T3 无语音时整组弃答，外加 T1 极少数失败。`filler_ratio` 额外多 33 条，因为非英语且无冻结 filler 词表时也写 `null`。

captioner 在 `speaker.speaker_present` 上 null 最低（62.5%），说明它更愿意判断「有没有人说话」，但人数、重叠、时间结构、声学几乎全弃。`speaker_change_rate_per_min` 与 `nonspeech_ratio` 为 **650/650**。

Omni 说话人核心字段（count/change/overlap）稳定在约 25% null，与约 150 条 T3 + 少量 T1/T2 弃答相符；性别、派生语速、非语音比例、外部噪声则是另一档（71–97%）。

### 3.2 描述性 10 轴

空数组 `[]` **不计** null。`speaker_descriptions` 对 Sure-Tagger 另有 39 条 `[]`（确认无人可描述），未计入下表。

| 标签 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: |
| `descriptive.caption` | 0 (0.0%) | 11 (1.7%) | 144 (22.2%) |
| `descriptive.topic` | 157 (24.2%) | 155 (23.8%) | 196 (30.2%) |
| `descriptive.background_condition` | 0 (0.0%) | 351 (54.0%) | 118 (18.2%) |
| `descriptive.dominant_content_impression` | 0 (0.0%) | 133 (20.5%) | 111 (17.1%) |
| `descriptive.clarity_intelligibility_impression` | 1 (0.2%) | 356 (54.8%) | 120 (18.5%) |
| `descriptive.background_sound_summary` | 0 (0.0%) | 376 (57.8%) | 253 (38.9%) |
| `descriptive.descriptive_phrases` | 1 (0.2%) | 533 (82.0%) | 2 (0.3%) |
| `descriptive.facets` | 1 (0.2%) | 626 (96.3%) | 2 (0.3%) |
| `descriptive.speaker_descriptions` | 77 (11.8%) | 532 (81.8%) | 3 (0.5%) |
| `descriptive.scene_hypotheses` | 14 (2.2%) | 609 (93.7%) | 2 (0.3%) |
| **描述 10 轴宏平均** | **3.9%** | **56.7%** | **14.6%** |

captioner 能写一句 caption / topic，但很少展开 facets、scene、phrases。Omni 相反：结构化描述字段几乎总写，T3 事件音频上 caption/topic 大量 `null`。Sure-Tagger 的 topic null（24.2%）集中在 T3（DESED 69.3%，FSD50K 98.7%），符合「无足够语音内容不猜主题」。

---

## 4. 二十九轴：有效评分样本

只统计进入 PDF 得分的样本。L1 词汇轴有效 500 条（T1 350 + T2 150；`filler_ratio` 仅英语 410 条）。L2/L3 仅 T1 350 条。L5 事件轴仅 T3 150 条；`nonspeech_ratio` 仅 T1 350 条。描述轴 648 条有 GT（2 条缺 GT，不影响下表结构字段）。

| 层 | 标签 | 有效 N | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | --- | ---: | ---: | ---: | ---: |
| L1 | `speaker.asr_transcript` | 500 | 3 (0.6%) | 448 (89.6%) | 22 (4.4%) |
| L1 | `language_content.language` | 500 | 0 (0.0%)；非法 7 | 445 (89.0%) | 23 (4.6%) |
| L1 | `language_content.word_count` | 500 | 3 (0.6%) | 459 (91.8%) | 22 (4.4%) |
| L1 | `language_content.repetition.has_repetition` | 500 | 3 (0.6%) | 457 (91.4%) | 33 (6.6%) |
| L1 | `language_content.repetition.repetition_count` | 500 | 3 (0.6%) | 466 (93.2%) | 33 (6.6%) |
| L1 | `deterministic_derived.filler_ratio` | 410 | 3 (0.7%) | 372 (90.7%) | 16 (3.9%) |
| L2 | `speaker.speaker_count` | 350 | 0 (0.0%) | 312 (89.1%) | 13 (3.7%) |
| L2 | `speaker.speaker_present` | 350 | 0 (0.0%) | 180 (51.4%) | 12 (3.4%) |
| L2 | `speaker.multi_speaker` | 350 | 0 (0.0%) | 272 (77.7%) | 13 (3.7%) |
| L2 | `speaker.speaker_overlap` | 350 | 0 (0.0%) | 318 (90.9%) | 13 (3.7%) |
| L2 | `speaker.gender` | 350 | 133 (38.0%) | 347 (99.1%) | 333 (95.1%) |
| L3 | `speaker.speaker_change_count` | 350 | 0 (0.0%) | 333 (95.1%) | 13 (3.7%) |
| L3 | `speaker.speaker_change` | 350 | 0 (0.0%) | 303 (86.6%) | 13 (3.7%) |
| L3 | `speaker.overlap_ratio` | 350 | 3 (0.9%) | 340 (97.1%) | 13 (3.7%) |
| L3 | `deterministic_derived.speaker_change_rate_per_min` | 350 | 0 (0.0%) | 350 (100.0%) | 304 (86.9%) |
| L5 | `basic_acoustic.nonspeech_ratio` | 350 | 0 (0.0%) | 350 (100.0%) | 332 (94.9%) |
| L5 | `sound_field_scene.speech_music_events` | 150 | 0 (0.0%) | 106 (70.7%) | 137 (91.3%) |
| L5 | `sound_field_scene.music_present` | 150 | 0 (0.0%) | 107 (71.3%) | 129 (86.0%) |
| L5 | `sound_field_scene.external_noise_type` | 150 | 0 (0.0%) | 116 (77.3%) | 103 (68.7%) |

读 PDF 分数时应用本表，而不是全量 650：

- Sure-Tagger 在已评分的词汇/说话人/时间轴上几乎不弃答；L2 被性别 38.0% null 拉低（null 得 50%，已知样本再按 gender 编辑距离计分）。
- captioner 除 `speaker_present`（51.4%）外，有效样本 null 率普遍 70–100%。`nonspeech_ratio` 与 `speaker_change_rate_per_min` 有效样本得分恰好 50.0%，与 100% null 完全吻合。
- Omni 在 T1/T2 词汇和说话人核心字段上只有约 4% null，但性别 95.1%、派生语速 86.9%、非语音比例 94.9%、T3 事件 69–91% 仍大量弃答。这解释了「L1 接近 Sure-Tagger、L5 接近 captioner」的雷达图形状。

---

## 5. 分 Track

### 5.1 T1 AMI（350，自然语言会议）

| 标签 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: |
| `speaker.asr_transcript` | 3 (0.9%) | 304 (86.9%) | 12 (3.4%) |
| `language_content.language` | 0；非法 4 | 305 (87.1%) | 13 (3.7%) |
| `language_content.word_count` | 3 (0.9%) | 312 (89.1%) | 12 (3.4%) |
| `language_content.repetition.has_repetition` | 3 (0.9%) | 311 (88.9%) | 19 (5.4%) |
| `language_content.repetition.repetition_count` | 3 (0.9%) | 320 (91.4%) | 19 (5.4%) |
| `deterministic_derived.filler_ratio` | 3 (0.9%) | 313 (89.4%) | 12 (3.4%) |
| `speaker.speaker_count` | 0 (0.0%) | 312 (89.1%) | 13 (3.7%) |
| `speaker.speaker_present` | 0 (0.0%) | 180 (51.4%) | 12 (3.4%) |
| `speaker.multi_speaker` | 0 (0.0%) | 272 (77.7%) | 13 (3.7%) |
| `speaker.speaker_overlap` | 0 (0.0%) | 318 (90.9%) | 13 (3.7%) |
| `speaker.gender` | 133 (38.0%) | 347 (99.1%) | 333 (95.1%) |
| `speaker.speaker_change_count` | 0 (0.0%) | 333 (95.1%) | 13 (3.7%) |
| `speaker.speaker_change` | 0 (0.0%) | 303 (86.6%) | 13 (3.7%) |
| `speaker.overlap_ratio` | 3 (0.9%) | 340 (97.1%) | 13 (3.7%) |
| `deterministic_derived.speaker_change_rate_per_min` | 0 (0.0%) | 350 (100.0%) | 304 (86.9%) |
| `basic_acoustic.nonspeech_ratio` | 0 (0.0%) | 350 (100.0%) | 332 (94.9%) |

T1 是说话人/时间/性别/非语音比例的唯一评分组。captioner 与 Omni 的性别轴几乎全 null（99.1%、95.1%），得分会贴近 50%。Sure-Tagger 性别仍有 38.0% 弃答，是其结构轴里唯一高 null 项。

### 5.2 T2 LibriSpeech（50）与 FLEURS（100）

仅 L1 词汇轴计分。FLEURS 的 `filler_ratio` 只评英语 10 条。

| 标签 | 组 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | --- | ---: | ---: | ---: |
| `speaker.asr_transcript` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 2 (4.0%) |
| `speaker.asr_transcript` | FLEURS 100 | 0 (0.0%) | 95 (95.0%) | 8 (8.0%) |
| `language_content.language` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 2 (4.0%) |
| `language_content.language` | FLEURS 100 | 0；非法 3 | 91 (91.0%) | 8 (8.0%) |
| `language_content.word_count` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 2 (4.0%) |
| `language_content.word_count` | FLEURS 100 | 0 (0.0%) | 98 (98.0%) | 8 (8.0%) |
| `language_content.repetition.has_repetition` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 4 (8.0%) |
| `language_content.repetition.has_repetition` | FLEURS 100 | 0 (0.0%) | 97 (97.0%) | 10 (10.0%) |
| `language_content.repetition.repetition_count` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 4 (8.0%) |
| `language_content.repetition.repetition_count` | FLEURS 100 | 0 (0.0%) | 97 (97.0%) | 10 (10.0%) |
| `deterministic_derived.filler_ratio` | Libri 50 | 0 (0.0%) | 49 (98.0%) | 2 (4.0%) |
| `deterministic_derived.filler_ratio` | FLEURS 英 10 | 0 (0.0%) | 10 (100.0%) | 2 (20.0%) |

Sure-Tagger 在 T2 有效样本上 **0 null**。captioner 在干净朗读上几乎全部弃答（95–98%）。Omni 在 Libri 约 4%、FLEURS 约 8–10%。

### 5.3 T3 DESED / FSD50K（各 75）

仅 L5 事件/噪声轴计分。`music_present` 在 DESED 无独立得分（`score_rate = null`），但仍有预测状态。

| 标签 | 组 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | --- | ---: | ---: | ---: |
| `sound_field_scene.speech_music_events` | DESED 75 | 0 (0.0%) | 49 (65.3%) | 67 (89.3%) |
| `sound_field_scene.speech_music_events` | FSD50K 75 | 0 (0.0%) | 57 (76.0%) | 70 (93.3%) |
| `sound_field_scene.music_present` | DESED 75 | 0 (0.0%) | 53 (70.7%) | 67 (89.3%) |
| `sound_field_scene.music_present` | FSD50K 75 | 0 (0.0%) | 54 (72.0%) | 62 (82.7%) |
| `sound_field_scene.external_noise_type` | DESED 75 | 0 (0.0%) | 44 (58.7%) | 47 (62.7%) |
| `sound_field_scene.external_noise_type` | FSD50K 75 | 0 (0.0%) | 72 (96.0%) | 56 (74.7%) |

Sure-Tagger 在 T3 有效事件轴 **0 null**（空事件写 `[]` 而不是 `null`）。captioner 在 DESED 噪声类型上相对最敢写（58.7% null），FSD50K 则 96.0% 弃答。Omni 事件轴弃答高于 captioner。

### 5.4 描述性轴分 Track 宏平均

10 个描述字段在该 track 上的 JSON null 率再取平均。

| Track | n | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: | ---: |
| T1 AMI | 350 | 0.8% | 56.0% | 6.3% |
| T2 LibriSpeech | 50 | 1.6% | 65.0% | 4.0% |
| T2 FLEURS | 100 | 1.1% | 66.2% | 7.0% |
| T3 DESED | 75 | 10.8% | 39.9% | 41.1% |
| T3 FSD50K | 75 | 16.5% | 58.4% | 44.5% |

Sure-Tagger / Omni 的描述 null 主要出现在 T3（无可靠主题/caption）。captioner 在所有 track 都高，T3 DESED 略低是因为 caption 仍常写、部分背景字段有值。

---

## 6. 未进入 29 轴的结构字段

这些字段留在原始输出里，不进 PDF 雷达图。分母 650。父对象为 `null` 的叶子按显式 null 计。

| 标签 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: |
| `basic_acoustic.channels` | 0 (0.0%) | 650 (100.0%) | 628 (96.6%) |
| `basic_acoustic.duration_sec` | 0 (0.0%) | 650 (100.0%) | 596 (91.7%) |
| `basic_acoustic.sample_rate_hz` | 0 (0.0%) | 650 (100.0%) | 628 (96.6%) |
| `basic_acoustic.silence_segments` | 0 (0.0%) | 650 (100.0%) | 629 (96.8%) |
| `audio_quality.snr_db` | 0 (0.0%) | 650 (100.0%) | 650 (100.0%) |
| `audio_quality.dnsmos_sig` | 6 (0.9%) | 650 (100.0%) | 650 (100.0%) |
| `audio_quality.dnsmos_bak` | 3 (0.5%) | 650 (100.0%) | 650 (100.0%) |
| `audio_quality.dnsmos_ovrl` | 3 (0.5%) | 650 (100.0%) | 650 (100.0%) |
| `audio_quality.dnsmos_p808` | 0 (0.0%) | 650 (100.0%) | 650 (100.0%) |
| `room_acoustic.rt60_sec` | 0 (0.0%) | 650 (100.0%) | 650 (100.0%) |
| `room_acoustic.c50_db` | 0 (0.0%) | 650 (100.0%) | 650 (100.0%) |
| `room_acoustic.far_field` | 650 (100.0%) | 650 (100.0%) | 650 (100.0%) |
| `sound_field_scene.noise_composition` | 0 (0.0%) | 650 (100.0%) | 114 (17.5%) |
| `speaker.profiles` | 6 (0.9%) | 527 (81.1%) | 163 (25.1%) |
| `language_content.filler` | 98 (15.1%) | 597 (91.8%) | 162 (24.9%) |
| `language_content.punctuation.punctuation_count` | 98 (15.1%) | 600 (92.3%) | 173 (26.6%) |
| `language_content.punctuation.has_terminal_punctuation` | 98 (15.1%) | 591 (90.9%) | 173 (26.6%) |

`room_acoustic.far_field` 三套系统全部为 `null`，属于 schema 允许的「证据不足不写死」字段，不是系统特有缺陷。captioner 对全部声学测量、DNSMOS、房间声学均为 100% null；`speaker.profiles` 为 81.1% null。Omni 同样不输出 SNR/DNSMOS/RT60/C50。

未评分描述字段：

| 标签 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: |
| `descriptive.uncertainties` | 0 (0.0%) | 15 (2.3%) | 2 (0.3%) |
| `descriptive.usability_notes` | 0 (0.0%) | 233 (35.8%) | 2 (0.3%) |

---

## 7. 和 PDF 分数的对照

结构单样本：known 且正确得 2，显式 null 得 1，错误/缺失/非法得 0，再除以 2 得到得分率。若某轴有效样本 100% null，得分率必为 50%。

| PDF 汇总 | Sure-Tagger | qwen-captioner | Qwen3-Omni Flash v2 |
| --- | ---: | ---: | ---: |
| 29 轴总宏平均 | 76.28% | 51.23% | 62.79% |
| 结构 19 轴 | 82.16% | 52.70% | 62.74% |
| 描述 10 轴 | 65.09% | 48.44% | 62.89% |
| 结构 19 轴有效样本宏平均 null | 2.2% | 87.0% | 30.5% |
| 描述 10 轴全量宏平均 null | 3.9% | 56.7% | 14.6% |

典型对齐：

- captioner `basic_acoustic.nonspeech_ratio`：有效 350/350 null → 得分 50.00%。
- captioner `deterministic_derived.speaker_change_rate_per_min`：350/350 null → 得分 50.00%。
- Omni `basic_acoustic.nonspeech_ratio`：332/350 = 94.9% null → 得分 51.5%，只略高于半分。
- Sure-Tagger T1 `speaker.gender`：133/350 = 38.0% null，已知 217 条再按序列编辑距离计分，轴分为 67.8%。

因此比较三套系统时，应同时看 **known 率** 和 **known 条件下的准确率**。captioner 的 ~51% 不宜解读为「略优于随机」；它主要是合法弃答。Omni 的总分高于 captioner，主要来自 L1 低 null 且准确，而不是声学事件识别。

---

## 8. 可复现命令要点

结构字段直接读 scorer 汇总：

- 全量：`score.json` / `structural_score.json` 的 `all_input_field_states[field].result_null / 650`
- 有效样本：各 `evaluation_groups[*].fields[field].result_null / eligible_samples`（仅 `metric_role = capability`）

描述性字段扫描 `descriptive.<name> is null`（父对象 `descriptive` 为 null 则 10 轴全计 null）。空列表不计入。

相关文档：[`benchmark-v1-29-metrics.md`](benchmark-v1-29-metrics.md)、[`tag-schema-concise.md`](tag-schema-concise.md)、[`docs/report/result/README.md`](report/result/README.md)。
