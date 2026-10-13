# 全量音频标签与逐说话人转写 Planner 设计

日期：2026-10-08  
状态：设计草案

## 1. 目标与边界

输入是音频及其样本 ID。输出包括：

1. 符合 [tag-schema.md](tag-schema.md) 的每样本一行 JSONL：6 个原始分组中的 31 个字段、`deterministic_derived` 的 2 个字段、`descriptive` 的 12 个字段，以及位于 `descriptive.speaker_descriptions[].gender` 的结构化性别值。
2. 独立、版本化的 `timed_transcript` JSONL：保留每个说话人的文本、说话时段和有条件提供的字词时间戳。
3. 可审计的证据记录：每个候选值的工具、模型版本、输入裁剪范围、原始输出、运行状态和裁决结果。

现有 `speaker.asr_transcript` 是整段文本，不包含说话人或时间戳；它还有 schema 指定的来源合同。Planner 不把多模型融合文本悄悄写回这个同名字段。若要改变其来源语义，应另立 schema 版本；`timed_transcript` 可独立采用最佳可验证候选。本文只依据 schema 的字段定义、生成约束，以及 `models` 目录中的模型卡和接口设计工具组合。

## 2. 职责划分

| 部件 | 职责 |
| --- | --- |
| 大模型 Planner | 根据尚缺字段、冲突位置和资源预算选择工具；解释需要复核的音频片段；在结构字段冻结后生成描述性文本。 |
| 工具适配器 | 将各模型输入输出归一化为带原音频时间坐标的类型化证据；保存原始结果，不直接写最终标签。 |
| 证据存储 | 保存候选、模型版本、参数、裁剪偏移、语言、时间粒度、失败原因及相关性来源。 |
| 确定性求值器 | 计算区间、比率、计数、文本统计及固定映射；执行类型与跨字段约束。 |
| 冲突裁决器 | 将硬约束失败和模型意见分歧分开处理，只对受影响的片段或字段增量重跑。 |
| 输出器 | 分别生成 schema JSONL、`timed_transcript` JSONL 和证据索引。 |

每个工具在注册表中声明 `input`、`output`、支持语言、最长音频、时间粒度、权重与运行器可用性、显存/延迟、失败语义和已校准的适用范围。调度器先做健康检查，再按预期减少的不确定性与调用成本安排并发；不能把不同模型的原始置信分数直接相加或多数投票。`false`、`0`、`[]` 是已知值，只有 `null` 表示未知。

## 3. 字段覆盖与工具分工

下表列出全部顶层目标。表中的字段名均相对于该行分组；复合对象的内部格式仍以 schema 为准。

| 分组 | 必须产出的字段 | 主证据与互证 |
| --- | --- | --- |
| `basic_acoustic`（5） | `duration_sec`, `sample_rate_hz`, `channels`, `nonspeech_ratio`, `silence_segments` | 原始媒体元数据给前三项；[FireRedVAD](../../sure-tagger-main/models/FireRedVAD/README.md) 的人声区间给后两项，[Brouhaha](../../sure-tagger-main/models/brouhaha/brouhaha-vad/README.md) 的 VAD 作边界复核。这里的“非语音”包括音乐和噪声，不等于物理静音。 |
| `audio_quality`（5） | `snr_db`, `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl`, `dnsmos_p808` | Brouhaha 估计 SNR；[DNSMOS](../../sure-tagger-main/models/DNS-Challenge/DNSMOS/README.md) 产生四个质量指标。按适用的语音片段计算并记录聚合口径，避免把纯非语音片段解释成语音质量。 |
| `room_acoustic`（3） | `far_field`, `rt60_sec`, `c50_db` | [Rec-RIR](../../sure-tagger-main/models/Rec-RIR/README.md) 先输出估计的 RIR 波形，再用固定 DSP 公式求 RT60/C50；Brouhaha 的 C50 只作独立复核。没有经过标定的远场检测器时，`far_field` 保持 `null`。 |
| `sound_field_scene`（4） | `speech_music_events`, `music_present`, `external_noise_type`, `noise_composition` | FireRed AED 给 `speech`、`singing`、`music` 时间段；[DASS](../../sure-tagger-main/models/DASS/saurabhati__DASS_medium_AudioSet_48.9/README.md) 的 AudioSet 类别映射到 schema 的六个背景声大类；[PANNs Cnn14](../../sure-tagger-main/models/audioset_tagging_cnn/README.md) 作事件复核。Cnn14 当前是片段级分类，需要滑窗才可估计发生时段。 |
| `speaker`（9） | `speaker_count`, `speaker_present`, `asr_transcript`, `multi_speaker`, `speaker_change_count`, `speaker_change`, `overlap_ratio`, `speaker_overlap`, `profiles` | [MOSS](../../sure-tagger-main/models/MOSS-Transcribe-Diarize/README.md) 给联合转写与分段候选；[DiariZen](../../sure-tagger-main/models/diarizen-official/model/README.md)、[pyannote Community-1](../../sure-tagger-main/models/speaker/pyannote_community_1/README.md)、[Nemotron 3](../../sure-tagger-main/models/speaker/nvidia/Nemotron-3-Diarization/README.md) 给独立说话人时间线；[CAM++](../../sure-tagger-main/models/speech_campplus_sv_zh-cn_16k-common-v1.0.0/README.md) 或 ECAPA 声纹核验跨段身份。逐人语速、相对音高和音量来自已确认的非重叠语音。`asr_transcript` 遵守自身来源合同。 |
| `language_content`（5） | `language`, `word_count`, `punctuation`, `repetition`, `filler` | FireRed LID 提供语言候选，Qwen3-ASR 的语言结果作复核；文本统计由版本化的 tokenizer、标点与 filler 规则计算。ASR 无可靠文本时不把缺失误写成 `0`。 |
| `deterministic_derived`（2） | `speaker_change_rate_per_min`, `filler_ratio` | 只由确定性程序按 schema 公式生成；依赖缺失或分母为零时为 `null`。 |
| `descriptive`（12） | `caption`, `topic`, `background_condition`, `dominant_content_impression`, `clarity_intelligibility_impression`, `background_sound_summary`, `descriptive_phrases`, `facets`, `speaker_descriptions`, `scene_hypotheses`, `usability_notes`, `uncertainties` | 结构性值冻结后由语言模型生成；`topic` 必须有 `speaker.asr_transcript` 的语义证据；场景只写成假设。`facets`、逐人描述、场景假设均保留有效 `evidence_tags`。 |

`speaker.gender` 是独立评测轴，JSON 值放在 `descriptive.speaker_descriptions[].gender`。应对同一说话人的干净语音运行专门的声音性别分类器并在证据不足或分类器分歧时写 `null`；不得由 `pitch`、语言或描述文本推断。三个允许值仅为 `male`、`female`、`null`。

复合字段也要按 schema 完整落地：`silence_segments[]` 含 `start_sec`/`end_sec`；`noise_composition` 固定含 `music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment` 六类及各类具体标签数组；`profiles[]` 含 `speaker_id`、`speech_rate.{band,value,unit}`、`pitch`、`speaker_volume`；`punctuation` 含计数和末尾标点布尔值；`repetition` 含存在性和重复计数。`facets[]`、`speaker_descriptions[]`、`scene_hypotheses[]` 则分别保留各自的描述、证据路径和置信字段。不要用空对象或缺键伪装为有效结果。

## 4. 执行计划

### 4.1 预处理与初筛

读取原始时长、采样率和声道；为要求单声道 16 kHz 的模型产生派生波形，但保留原始文件和原声道。所有裁剪输出都记录相对原音频的偏移，所有最终时间戳统一换算为原音频秒数。声道差异明显时，保留逐声道诊断结果，避免直接下混掩盖重叠。

首轮可并行调用 FireRed VAD/AED、语言识别、Brouhaha、DNSMOS、DASS/PANNs 的适用片段，以及初步说话人分段。确认没有人声时跳过 ASR、声纹与逐人画像；“未检出”与“工具失败”必须区分。

### 4.2 转写与说话人时间线

有人声时，运行 MOSS 的联合候选，同时运行一套独立 diarization 和一套独立 ASR。语言支持范围内优先利用 FireRedASR2-AED 的字词时间戳；更广语言可用 [Qwen3-ASR](../../sure-tagger-main/models/Qwen3-ASR-0.6B/README.md) 复核文本。重叠、人数较多或边界冲突时追加 DiariZen、pyannote、Nemotron/Sortformer 中适用的工具。DiariZen 的“至多四人”指同时重叠能力；Nemotron 3 的人数上限为八人，超过适用范围时不能把模型输出当完整人数结论。

对长音频分块时，块间保留上下文和重叠，统一回写绝对时间，再消除重复片段。MOSS 的 `[S01]` 等编号仅在该次输入内部有效；跨块或跨模型通过活动区间匹配、声纹验证与全局约束关联。最终 `speaker_1`、`speaker_2` 按各人首次开口时间排序，`speaker.profiles`、逐人转写、描述和性别序列使用同一映射。

对于单人、非重叠片段，可以用 diarization 时间窗口约束 ASR；对于重叠片段，裁出时间窗口仍包含其他人的声音，不能仅靠区间相交把整句归给某一人。保留 MOSS 与独立模型的原始候选，并对冲突处定向复核。仍无法判断时，片段可有文本但 `speaker_id: null`，或有说话人时段但 `text: null`。

### 4.3 时间对齐、结构化与描述

先选定文本，再使用 Qwen ForcedAligner 为受支持语言的短片段定位字词。ForcedAligner 只对输入的文本定位，不是第二个 ASR；其模型说明限制为一次最多五分钟、十一种语言，而且本地 `.oasr` 资产还需要运行器健康检查。无法可靠对齐时只交付段级时间，不伪造词级时间。

确定说话人时间线后，再计算人数、切换、重叠比例及逐人画像；确定文本后再计算词数、标点、重复、filler。最后生成两个派生比率，冻结全部结构字段，再调用语言模型生成 `descriptive`。一个字段的失败只影响其依赖闭包；其他有独立证据的字段仍可输出。

切换计数需要冻结明确的 turn 规则：按时间扫描说话人活动集合，忽略纯静音间隔；一个新的可归属发言 turn 的主说话人与上一 turn 不同时计一次。重叠期间出现第二人声不自动算切换，只有主发言权实际转移时才计数。主说话人无法可靠判定的转场不强行计入，并保留原始多说话人活动区间供复核。规则版本与重叠边界容差一起记录，避免同一段音频因后处理差异得到不同计数。

## 5. 逐人转写的数据合同

`timed_transcript` 与 schema JSONL 通过 `sample_id` 关联，一行对应一个样本。示例只展示必要字段，`source_refs` 指向证据记录而不是模型自由编造的说明。

```json
{
  "version": "timed_transcript.v1",
  "sample_id": "example-001",
  "timebase": "original_audio_seconds",
  "segments": [
    {
      "start_sec": 0.48,
      "end_sec": 1.66,
      "speaker_id": "speaker_1",
      "text": "我们先看这个结果。",
      "words": null,
      "source_refs": ["moss:segment:1", "diarization:turn:1"],
      "status": "accepted"
    },
    {
      "start_sec": 1.52,
      "end_sec": 2.24,
      "speaker_id": "speaker_2",
      "text": "好的。",
      "words": null,
      "source_refs": ["moss:segment:2", "diarization:turn:2"],
      "status": "accepted"
    }
  ],
  "speaker_transcripts": [
    {"speaker_id": "speaker_1", "text": "我们先看这个结果。"},
    {"speaker_id": "speaker_2", "text": "好的。"}
  ]
}
```

不同说话人的片段允许时间重叠。每个片段必须满足 `0 <= start_sec < end_sec <= duration_sec`；同一说话人的稳定文本片段应按开始时间排序。词级时间若非空，须落在所在片段内，并在约定的标点归一化规则下重建片段文本。`speaker_transcripts` 仅拼接该说话人已接受的文本，不吸收 `speaker_id: null` 的片段。原始模型置信分数和人工修改记录保留在证据记录，不伪装为统一概率。

## 6. 互证与冲突处理

### 6.1 硬约束：程序校验

| 关系 | 检查方式 |
| --- | --- |
| 时间基准 | 所有时段在原音频范围内；`silence_segments` 排序、不重叠；跨块偏移只加一次。 |
| 非语音比例 | 若区间与比例来自同一 VAD 结果，`nonspeech_ratio = 非语音区间并集时长 / duration_sec`；不同 VAD 结果之间仅作软比较。 |
| 人数 | 两端已知时，`speaker_present == (speaker_count > 0)`，`multi_speaker == (speaker_count >= 2)`；0 或 1 人不应有多人重叠或切换。 |
| 切换与重叠 | 从版本化的决策时间线计算 `speaker_change_count` 和 `overlap_ratio`；后者分母为有效语音并集时长；布尔值与非零结果一致。重叠边界的帧量化容差需预先固定。 |
| 音乐 | 两端已知时，`music_present` 与 `speech_music_events` 是否包含 `music` 一致；事件数组按 `speech`、`singing`、`music` 固定顺序去重。 |
| 派生率 | 分母大于零且依赖齐全时，`round(speaker_change_count * 60 / duration_sec, 6)` 与 `round(filler / word_count, 6)`；否则按合同留 `null`。 |
| 身份与描述 | profile ID 唯一且按首次开口排序；人数和 profile 同时可靠时数量相符；逐人描述只能引用已有 ID；每条 `evidence_tags` 指向本样本非 `null` 的真实结构字段。 |
| 复合字段 | `noise_composition` 六键齐全；`punctuation` 和 `repetition` 的必需子键齐全且内部计数、布尔值一致；词级时间在所属片段内。 |

### 6.2 软互证：触发局部复核

| 证据组合 | 分歧时检查什么 |
| --- | --- |
| VAD、MOSS、独立 ASR | ASR 是否在非人声区间出现幻觉；弱语音是否被 VAD 漏检。 |
| MOSS、独立 diarization、声纹 | 说话人数、转场、跨块同一人和重叠区间；不能按模型标签字符串直接匹配身份。 |
| MOSS、FireRed、Qwen 文本 | 同一片段的内容差异、语言、漏词与重复；ForcedAligner 不参与文本投票。 |
| 语言、ASR 与文本统计 | code-switching、tokenizer 口径、词数、标点、重复和 filler；全局语言不能代替片段级语言证据。 |
| AED、DASS、PANNs | 音乐、歌声和背景声类别；DASS 与 PANNs 同属 AudioSet 类任务，不能当完全独立的两票。 |
| Brouhaha、DNSMOS、Rec-RIR | SNR、背景质量与混响是否共同支持文字描述；它们测量的不是同一个量，不能互相覆盖。 |

冲突裁决按以下顺序执行：验证工具是否成功及输出是否越界；定位受影响的原音频片段；按适用范围追加独立工具或重跑该片段；重算受影响的确定性字段；仍无法消除的歧义保留候选与失败原因，相关标签填 `null`，并在 `descriptive.uncertainties` 记录可解释的缺口。不得因一个细字段为 `null` 就无条件抹掉具有独立证据的粗粒度字段。描述模型不得反向改写已冻结的结构字段。

## 7. 验收方式与已知边界

离线验收至少覆盖无语音、单人、多人与重叠、长音频、噪声/音乐、混合语言六类样本。分别评估 31+2+12 字段的有效覆盖率和准确率、硬约束违规率、逐人 WER/CER 或 cpWER、diarization 错误率、时间边界误差、无法归属片段比例、人工复核量与 GPU 成本。阈值、文本归一化、类别映射、模型版本及工具预算都应冻结为可复现配置。

当前模型集合不能自动补齐 schema 之外的所有 rich transcription 维度；笑声、情绪、具名身份和具体录音地点等不应被当作已有结构标签。PANNs 本地 Cnn14 权重只提供片段级类别；Qwen 强制对齐并不覆盖所有语言；性别分类器只输出 schema 所需的二元候选，不能据此确定个人身份。证据不足时保留 `null`，明确确认没有项目时才使用 `[]`。
