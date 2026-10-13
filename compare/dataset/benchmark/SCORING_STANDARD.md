# Benchmark 结构化标签评分表

更新时间：2026-09-07  
适用数据：`structural-benchmark-subset-v1.0`（650 条，T1--T3）  
评分规范版本：`benchmark-scoring-v0.9`

本文只列 benchmark-v1 的 35 个结构化标签。`descriptive.*` 是开放式描述字段，不属于本 benchmark 的结构化评分标签。

## 统一计分口径

- 只对 `gt_mask=true`、GT 合法且没有输入泄漏的样本计分；无有效样本时 `score_rate=null`，不记为 0 分。
- 除表中特别说明外：预测正确得 2 分，显式输出 `null` 得 1 分，预测错误、字段缺失、格式非法或运行失败得 0 分。
- 普通字段：`score_rate = score_sum / (2N)`。二值字段按正、负样本分别算得分率后取平均，避免类别不平衡；只有负样本的类别只统计误报，不产生能力分。
- 连续字段：已知预测的单样本分数为 `2 * max(0, 1 - abs(预测-GT) / T)`；显式 `null` 仍得 1 分。
- `basic_acoustic.nonspeech_ratio` 和 `speaker.overlap_ratio` 不使用容差除数，已知预测的单样本分数为 `2 * (1 - abs(预测-GT))`；由于预测和 GT 均限定在 `[0,1]`，分数范围仍为 `[0,2]`。
- `word_count`、`speaker_change_count` 和 `speaker_change_rate_per_min` 采用对称归一化绝对误差：`2 * (1 - abs(预测-GT) / max(预测, GT, 1))`。该公式对多算和少算一视同仁，也能安全处理零值。
- 多标签字段逐类别评分，再对有 GT 正例的类别做宏平均；不同 GT 语义、来源或规则版本的 track 分开报告。
- 层级汇总采用两级宏平均：先对同一个标签在所有可评分评测组中的得分等权平均，再对该层所有可评分标签的标签分等权平均。不得直接平均“评测组 × 标签”轴。
- 提交文件整体无法解析、sample ID 冲突或控制字段合同失败时，本次提交的官方分数全部为 `null`，修复后重跑。

## 标签评分表

| 标签名 | 标签含义 | 评分方式 | 说明 |
| --- | --- | --- | --- |
| `basic_acoustic.duration_sec` | 音频时长，单位为秒。 | 与输入 WAV header 的 GT 绝对误差不超过 `0.001 s` 得 2 分；`null` 得 1 分；其余得 0 分。 | **合同标签**。覆盖全部可读取 WAV header 的 T1--T3 样本；不进入能力雷达。 |
| `basic_acoustic.sample_rate_hz` | 音频采样率，单位为 Hz。 | 正整数与输入 WAV header 完全相等得 2 分；`null` 得 1 分；其余得 0 分。 | **合同标签**。覆盖全部可读取 WAV header 的样本；不进入能力雷达。 |
| `basic_acoustic.channels` | 音频声道数。 | 正整数与输入 WAV header 完全相等得 2 分；`null` 得 1 分；其余得 0 分。 | **合同标签**。覆盖全部可读取 WAV header 的样本；不进入能力雷达。 |
| `basic_acoustic.nonspeech_ratio` | 官方 GT 名称为 `silence_ratio`：AMI 合并 utterance turn 的补集时长占整段音频时长的比例，范围 `[0,1]`。 | 绝对比例误差评分：`2 * (1 - abs(预测-GT))`；误差 0 得 2 分，误差 `0.5` 得 1 分，误差 1 得 0 分。 | **能力标签**。评 T1 350 条，GT 规则版本为 `ami-turn-complement-v1`。预测端兼容同义字段 `basic_acoustic.silence_ratio`；若两个字段同时存在，以 `nonspeech_ratio` 为准。 |
| `basic_acoustic.silence_segments` | 非语音区间数组，每项包含开始和结束时间。 | 暂不评分。 | **诊断标签**。非语音区间、端点和 collar 规则尚未冻结；只检查 schema 或覆盖率。 |
| `audio_quality.snr_db` | 信噪比，单位为 dB。 | 连续值评分规则保留；当前无有效 GT。 | **诊断标签**。T4 已禁用；本版不进入能力分。 |
| `audio_quality.dnsmos_sig` | DNSMOS P.835 SIG，表示语音信号质量，范围 `[1,5]`。 | 暂不评分。 | **诊断标签**。当前没有冻结的等价能力 GT，只检查 schema 或记录原始测量。 |
| `audio_quality.dnsmos_bak` | DNSMOS P.835 BAK，表示背景噪声干扰程度，范围 `[1,5]`。 | 暂不评分。 | **诊断标签**。当前没有冻结的等价能力 GT，只检查 schema 或记录原始测量。 |
| `audio_quality.dnsmos_ovrl` | DNSMOS P.835 OVRL，表示整体语音质量，范围 `[1,5]`。 | 暂不评分。 | **诊断标签**。当前没有冻结的等价能力 GT，只检查 schema 或记录原始测量。 |
| `audio_quality.dnsmos_p808` | DNSMOS P.808 主观整体质量分，范围 `[1,5]`。 | 暂不评分。 | **诊断标签**。当前没有冻结的等价能力 GT，只检查 schema 或记录原始测量。 |
| `room_acoustic.far_field` | 是否属于远场拾音。 | 暂不评分。 | **诊断标签**。near/far 定义尚未冻结，当前通常为 `null`。 |
| `room_acoustic.rt60_sec` | 混响衰减 60 dB 所需时间，单位为秒。 | 连续值评分规则保留；当前无有效 GT。 | **诊断标签**。T4 已禁用；本版不进入能力分。 |
| `room_acoustic.c50_db` | C50 清晰度指标，单位为 dB。 | 连续值评分规则保留；当前无有效 GT。 | **诊断标签**。T4 已禁用；本版不进入能力分。 |
| `sound_field_scene.speech_music_events` | 音频中是否出现 `speech`、`music`、`singing`，输出为多标签数组。 | 固定三类逐类按 2/1/0 评分；每类做正负平衡，再对有正例的类别宏平均。 | **能力标签**。评 T3；T3 的 DESED/FSD50K 分开报告。当前 `singing` 无正例，只统计误报，不进入宏平均。 |
| `sound_field_scene.music_present` | 音频中是否存在音乐。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。评 T3，并与事件数组独立计分；两者都已知时另查一致性。 |
| `sound_field_scene.external_noise_type` | 外部噪声大类数组：`music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment`。 | 固定六类逐类按 2/1/0 评分；每类做正负平衡，再对有正例的类别宏平均。 | **能力标签**。T3 评 150 条；无正例类别只统计误报。 |
| `sound_field_scene.noise_composition` | 六类背景声下更细粒度的具体声音标签对象。 | 暂不评分。 | **诊断标签**。细类 crosswalk 和等价 GT 尚未冻结，只检查 schema 或覆盖率。 |
| `speaker.speaker_count` | 音频片段内的不同说话人数。 | 非负整数与 GT 完全相等得 2 分；`null` 得 1 分；其余得 0 分。 | **能力标签**。评有完整 speaker GT 的 T1；T1 builder 未完成的样本保持 mask=false。 |
| `speaker.speaker_present` | 音频中是否存在说话人。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。与 `speaker_count` 独立计分；两者都已知时检查 `speaker_count > 0`。 |
| `speaker.asr_transcript` | 整段音频的语音转写文本。 | 先按冻结规则归一化并计算逐样本 WER/CER；`e=min(编辑距离/max(参考长度,1),1)`，得分为 `2*(1-e)`；`null` 得 1 分。 | **能力标签**。T1 与 T2 有 reference 的样本可评；非 CJK 用 WER，中文、日文、韩文用 CER；按 track、语言和 tokenizer 兼容组宏平均。 |
| `speaker.multi_speaker` | 是否至少有两个不同说话人。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。与 `speaker_count` 独立计分；两者都已知时检查 `speaker_count > 1`。 |
| `speaker.speaker_change_count` | 按冻结时间线规则统计的说话人切换次数。 | 已知预测为 `p`、GT 为 `g` 时，得分为 `2 * (1 - abs(p-g) / max(p,g,1))`；完全相等得 2 分，`null` 得 1 分，缺失或非法得 0 分。 | **能力标签**。评 T1；另报 exact accuracy、MAE 和归一化绝对误差。 |
| `speaker.speaker_change` | 音频中是否发生过说话人切换。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。独立计分；与 count 都已知时检查 `speaker_change_count > 0`。 |
| `speaker.overlap_ratio` | 多人重叠发言时长占有效语音并集时长的比例，范围 `[0,1]`。 | 绝对比例误差评分：`2 * (1 - abs(预测-GT))`；误差 0 得 2 分，误差 `0.5` 得 1 分，误差 1 得 0 分。 | **能力标签**。评 T1；T1 使用冻结 overlap/collar 规则。 |
| `speaker.speaker_overlap` | 是否存在多人同时发言。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。评 T1；必须使用与 `overlap_ratio` 相同版本的重叠区间规则。 |
| `speaker.profiles` | 逐说话人的结构画像，包括 ID、语速、相对音高和相对音量。 | 暂不评分。 | **诊断标签**。当前缺少可靠的逐说话人对齐和属性 GT；内部 profile 属性随整个对象只做 schema、覆盖率或审计，不单独计分。 |
| `language_content.language` | 音频的语言类别，规范为 `ar`、`zh`、`en`、`es`、`fr`、`hi`、`ja`、`ko`、`ru`、`sw`。 | canonical language 完全相等得 2 分；`null` 得 1 分；其余得 0 分；先按语言计算，再做语言宏平均。 | **能力标签**。T1 为 corpus-level English，T2 为 clip/locale GT；不同 GT 语义组不直接合并。 |
| `language_content.word_count` | 按统一指定的多语言 tokenizer 统计的非标点词或字符数。 | 已知预测为 `p`、GT 为 `g` 时，得分为 `2 * (1 - abs(p-g) / max(p,g,1))`；完全相等得 2 分，`null` 得 1 分，缺失或非法得 0 分。 | **能力标签**。仅评 T1/T2 中 tokenizer 规则已统一且 reference 可用的样本；另报 exact accuracy、MAE 和归一化绝对误差。 |
| `language_content.punctuation.punctuation_count` | canonical 原始转写中的 Unicode 标点总数。 | 暂不评分。 | **诊断标签**。规则和 reference 保留用于审计，但不进入能力分数或雷达图。 |
| `language_content.punctuation.has_terminal_punctuation` | 去除末尾空白后，转写是否以冻结集合中的句末标点结束。 | 暂不评分。 | **诊断标签**。规则和 reference 保留用于审计，但不进入能力分数或雷达图。 |
| `language_content.repetition.has_repetition` | 是否存在冻结规则定义的相邻重复 token n-gram。 | 布尔值正确得 2 分，`null` 得 1 分，其余得 0 分；最终采用正负类平衡得分。 | **能力标签**。当前规则最大 n-gram 为 3；与 `repetition_count` 独立计分。 |
| `language_content.repetition.repetition_count` | 冻结 n-gram 规则统计出的相邻重复次数。 | 非负整数与 GT 完全相等得 2 分；`null` 得 1 分；其余得 0 分。 | **能力标签**。只评 T1/T2 中 repetition 规则和 reference 可用的样本。 |
| `language_content.filler` | 按对应语言 filler 词表统计的填充词次数。 | 暂不评分。 | **诊断标签**。词表规则和原始预测保留用于审计；派生的 `filler_ratio` 仍按独立规则评分。 |
| `deterministic_derived.speaker_change_rate_per_min` | 每分钟说话人切换次数。 | 已知预测为 `p`、GT 为 `g` 时，得分为 `2 * (1 - abs(p-g) / max(p,g,1))`；完全相等得 2 分，`null` 得 1 分，缺失或非法得 0 分。 | **能力标签**。GT 为 `round_half_even_6(60 * speaker_change_count / duration_sec)`；评依赖 GT 可用且时长大于 0 的 T1 样本；另报 exact accuracy、MAE、归一化绝对误差和公式一致性。 |
| `deterministic_derived.filler_ratio` | 填充词数占词数的比例，范围 `[0,1]`。 | 连续值评分，`T=0.05`；误差 0 得 2 分，误差 `0.025` 得 1 分，误差大于等于 `0.05` 得 0 分。 | **能力标签**。GT 为 `round_half_even_6(filler / word_count)`；只评两项 GT 可用且 `word_count > 0` 的样本，并另报公式一致性。 |
