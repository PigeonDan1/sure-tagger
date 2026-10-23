# 500 条音频 Ground Truth（无推理版）

本目录含 500 条音频的 ground truth，按 `tag-schema(2).md` 组织。构建过程不运行任何模型，也不根据场景名称推断标签。
场景元数据仅发布 `generation.acoustic_scene_id` 和
`generation.acoustic_environment`；P01-P28 语用场景分类已弃用。

## 文件

- `ground_truth.jsonl`：带样本 ID、来源、时间戳和字段状态的完整记录。
- `schema_ground_truth.jsonl`：每行只含 schema 的 8 个顶层分组。
- `sure_tagger_gt_tags_only.jsonl`：每行只含 sure-tagger 结构化输出使用的 7 个非 descriptive 分组。
- `ground_truth_index.csv`：便于筛选的扁平索引。
- `ground_truth_summary.json`：数量、来源和 null 字段统计。

完整记录额外包含独立的 `frame_snr_ground_truth`，它不属于
`tag-schema(2).md`，因此不会写入 `schema_ground_truth.jsonl` 或
`sure_tagger_gt_tags_only.jsonl`。其 `mean_db` 使用 Brouhaha-maker
detailed-SNR 训练标签的 2 秒窗 / 10 ms hop 规则，并对所有完整窗口的 dB
值做算术平均；详细定义见 `SNR_GROUND_TRUTH_METHOD.md`。

## 填值原则

| 数据来源 | 填入字段 |
| --- | --- |
| 最终 WAV header | duration、sample rate、channels |
| 最终 session manifest | achieved SNR、说话人数、重叠率 |
| 最终 speech/noise stems | `frame_snr_ground_truth.mean_db`（Brouhaha detailed-SNR 训练标签口径） |
| 渲染轮次序列 | 说话人切换，以及由其确定性计算的字段 |
| scene script 的显式配置/文本统计 | 音乐与噪声配置、reference transcript、语言内容统计 |
| `qc/rir_metrics.jsonl` | measured RT60、measured C50 |

`speaker.profiles`、四项 DNSMOS、`room_acoustic.far_field` 和全部 `descriptive` 子字段均为 `null`。`nonspeech_ratio` 与 `silence_segments` 使用最终 session manifest 的 utterance 时间戳补集；这表示未安排人类语音的区间，不等同于物理静音。多语种和 code-switch 样本无法用 schema 的单一语言值表达，`language_content.language` 为 `null`；已知语言列表保留在完整记录的 `generation.languages_present` 中。

本开源包只保留当前版本的 GT；早期推断版与已废弃版本不在发布范围内。
