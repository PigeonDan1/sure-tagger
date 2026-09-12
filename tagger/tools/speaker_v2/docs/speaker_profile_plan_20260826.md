# Speaker v2 说话人画像与语速计划

> 2026-09-11 更新：本文的 v0.1/phase 0~1 内容保留为历史设计记录；当前 v0.2
> 已接入独立性别模型，现行合同见 `development.md` 和 `docs/tags-and-methods.md`。

> 历史快照：本文记录 2026-09-04 的 OpenASR 组合链路，不描述当前运行实现。
> 2026-09-10 起，其他、混合、未知或 LID 失败的 ASR 使用 Qwen3-ASR-0.6B，
> speaker timeline 使用官方 DiariZen；FireRedASR2-AED 仅用于中英文的
> 单说话人共识。当前路由以
> [direct_public_output_20260820.md](direct_public_output_20260820.md) 为准。

> 日期：2026-09-04（路由修订）  
> 状态：Phase 0~1 已实现（deterministic MVP）；进入数据集校准与 shadow 评测
> 范围：`tagger/tools/speaker_v2` 的样本级公开 speaker 输出、内部 evidence、评测和部署合同。

## 1. 决策摘要

可以在现有 `speaker` 输出下增加 `profiles`（说话人画像）。但不建议把
`tag_qwen-omni-captioner` 的画像字段原样搬过来：captioner 只读 caption 文本，
而 tagger 需要从音频、时间轴、VAD 和转写证据推导；两者的证据边界不同。

建议分两层推进：

1. **MVP 不引入新的画像模型**：复用语言路由后的 MOSS/OpenASR 时间轴、MOSS
   或 OpenASR stdout FireRedASR2-AED 转写、Whisper lexical timeline、
   Brouhaha/FireRedVAD speech coverage，加一个
   确定性声学统计适配器，先发布每个说话人的语速、音高档位和相对音量。
2. **语义画像按需引入模型**：年龄、性别、情绪、口音、`voice_traits`、
   `delivery_style` 等不应由现有 diarizer、ECAPA/CAM++ 或 DASS 越权推断。只有
   在标注集、偏差/隐私评估和 license 评审通过后，才单独引入属性模型；模型失败
   或证据不足继续输出 `null`，不影响已有六个 speaker claim。

本计划的首个公开新增字段是 `profiles[].speech_rate`。字段名采用
`speech_rate`，避免把播放速度或采样率误解为说话速度；若公司最终要求与
captioner 对齐，可在导出层将 `speech_rate.band` 映射为 captioner 的 `speed`。

## 2. 现有能力盘点

| 现有来源 | 当前能力 | 可复用到画像 | 不能直接承担 |
| --- | --- | --- | --- |
| MOSS Transcribe-Diarize | 联合 ASR、说话人、片段起止时间；`segments` 带 `speaker_id` 和文本 | 说话人归属、文本单位计数、片段级语速 | 没有稳定的逐词时间戳；ASR 文本和说话人共享一个模型依赖组 |
| OpenASR 0.1.40 | 非中文分支唯一 runtime；一次运行以下四个 pack 和 bundled Stream-VAD | stdout text 与 version-locked trace raw turns 的统一运行边界 | OpenASR stdout diarization JSON 不是本项目的 timeline 合同；必须解析锁定版本的 trace turns |
| FireRedASR2-AED q4_k pack | OpenASR stdout 的整段 JSON text；当前 evidence 标记 `timestamp_status=none` | 非中文、混合或未知语言分支的公开 transcript | 不单独执行，不提供 lexical timeline 或 C/M/O/X timeline，也不为中文 MOSS 分支补位 |
| DiariZen fp16 + ReDimNet2-B6 fp16 + bundled FireRed Stream-VAD | OpenASR diarization 与 debug trace raw turns（非中文） | 非中文 `speaker_count`、change、overlap 和画像时间轴 | 与下列 forced aligner 同属完整调用依赖；单独执行 DiariZen `.oasr` 不是受支持的 runtime |
| Qwen3-ForcedAligner 0.6B q4_k pack | FireRed speaker attribution 的必需 OpenASR capability pack | 使 FireRed text 能进入 diarized attribution 链路 | 缺失时 offline `transcribe --diarize` 失败，不产生部分 diarization 结果 |
| Sortformer / Pyannote Community-1 | 历史/可选诊断 timeline | 仅在显式诊断配置中交叉检查 | 不进入当前语言路由的 C/M/O/X 公开 claim |
| Whisper Base lexical | 可选 lexical units 和粗/实验时间戳；已有 projection/assignment 工具 | 作为 MOSS 文本的独立时间证据和语速质量 guard | 默认不是逐词生产时钟；不能作为 speaker event vote |
| Brouhaha / FireRedVAD | speech segments、speech coverage | 计算说话有效时长，排除静音和非语音 | 只提供 speech coverage，不提供 speaker identity 或画像 |
| ECAPA / CAM++ | clean region 的 same/different speaker verification | 画像的依赖审计和区域选择参考 | 是身份比较模型，不是年龄、性别、情绪或音色描述模型 |
| DASS | AudioSet 背景声音分类 | 作为噪声质量上下文 | 人声/背景声分类不能转成某个说话人的画像 |
| 现有 audio probe / 多通道处理 | 时长、采样率、声道等；多数 speaker 路由最终为 16 kHz 单声道 | 输入质量门控、记录降混限制 | 单声道不能可靠支持 `stereo_position`，元数据不能代替听感属性 |

实现依据：speaker v2 当前 resolver 只发布 `speaker_count`、`multi_speaker`、
`speaker_change_count`、`speaker_change`、`overlap_ratio`、`speaker_overlap`；
时间轴摘要已经包含 `activity_segments`、`speaker_ids`、重叠区间和切换候选点。
MOSS 证据还保留 `asr_transcript`、带文本的 segments 以及 speaker-text track；OpenASR
ASR evidence 只保留整段 stdout `text` / `asr_transcript`，并明确
`timestamp_status=none`；timeline 证据保留
`timeline_summary`、`raw_turns` 和 `openasr_model_chain`。FireRed
LID 严格返回已识别的 `zh`/`zh-*` alias 时选择 MOSS；其它、mixed、unknown 或 LID
失败时选择 OpenASR。选中分支失败时不跨分支补位；非中文 stdout text 与 trace
turns 各自只决定依赖它的字段。

OpenASR binary/FireRed pack 为 Apache-2.0，DiariZen 为 CC BY-NC 4.0，ReDimNet 为
MIT，Qwen3-ForcedAligner 为 Apache-2.0。默认 runtime 为
`.runtime/openasr/0.1.40/openasr`；四个 pack 分别位于
`models/firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr`、
`models/diarizen-large-s80-v2/diarizen-large-s80-v2-fp16.oasr`、
`models/redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr` 和
`models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr`。aligner
pack 固定大小为 765,395,744 bytes、SHA-256
`5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b`，OpenASR
catalog revision 为 `3a82628cf7036c2ad0128f2d35bd1cced09b3100`。

该历史版本的 C/M/O/X policy 为 `speaker_v2.claim_policy.20260904.2`，rule 使用
`selection="language_routed"`、无 fallback source，resolver 必须收到样本
`language_route`。standalone CLI 只用 `--openasr-enable` /
`--openasr-disable` 控制完整组合调用。OpenASR executable（包括 CLI/env 路径覆盖）
在首次使用对应文件身份时
按 `openasr 0.1.40`、42,268,384 bytes 与 SHA-256
`98d973d0d72b47198ecf17e4c00b2710d84c351b77d3c8024a7256bcea59a1a5` 强校验，
通过后的 version/size/digest verification 写入 runtime provenance。

能力边界：所有非 `zh` 结果都会按策略进入 OpenASR，但固定 FireRed/ForcedAligner
catalog 只声明 en/zh 语系。英文端到端 smoke 通过；真实法语样本无 hint 时返回了
结构合法却语义错误的英文 text，同时 DiariZen timeline 正常，显式
`--language fr` 则 exit 6。非 en/zh 的 ASR 与依赖文本的语速画像尚未获生产验证；
结构校验不能识别 silent wrong transcript，timeline 画像与文本画像必须保留独立
缺失/质量边界。

## 3. Captioner schema 参考与边界

`tag_qwen-omni-captioner/schema.full.json` 的 `speakerProfile` 当前包含：

- `speaker_id`、`accent`、`age`、`gender`、`emotion`、`distance`；
- `speaker_volume`、`speed`、`pitch`、`stereo_position`、`amplification`；
- `voice_traits`、`articulation`、`prosody`、`delivery_style`。

其约束是“caption 中有明确语义才填写，未知为 `null`”，不是从音频直接测量。
当前完整抽取快照（`coverage-full.json`）有 100 个样本、165 个 profile entity；
profile 字段的样本级已知率大多约为 0.97–0.98，`accent`/`age` 约为 0.72。
这些数字只说明文本抽取覆盖情况，不能当作音频模型准确率，也不能作为 tagger
发布值的先验。

对齐原则：

| Captioner 字段 | Speaker v2 计划处理 |
| --- | --- |
| `speed` | 映射为 `speech_rate.band`，先做；数值速率保留单位和证据基础 |
| `pitch` | 先用 F0 统计映射 `low/mid/high/variable`，不用于推断性别 |
| `speaker_volume` | 只做同片段内相对音量；增益未知时为 `null` |
| `distance` | 暂不做 speaker-specific 公共值；已有全局 near/far 不能冒充每个说话人距离 |
| `stereo_position` | 默认不做；多数链路降为单声道，只有通道布局和校准满足条件才可扩展 |
| `accent`、`age`、`gender`、`emotion` | 不由现有模型推断，作为后续独立属性模型候选 |
| `voice_traits`、`articulation`、`prosody`、`delivery_style` | 先保留内部特征/证据，不直接发布开放字符串 |
| `amplification` | 由录音链路/扩声检测另行评估，不能从直播或会议主题推断 |

captioner 的 caption、生成结果和人工先验不得进入 tagger inference 或 resolver；
只能用于离线 benchmark、字段词汇讨论和误差分析。

## 4. 建议的公开合同

### 4.1 v0.1 结构

在现有 `speaker` 对象中增加：

```json
{
  "speaker_count": 2,
  "multi_speaker": true,
  "speaker_change_count": 1,
  "speaker_change": true,
  "overlap_ratio": 0.125,
  "speaker_overlap": true,
  "profiles": [
    {
      "speaker_id": "speaker_1",
      "speech_rate": {
        "band": "normal",
        "value": 4.2,
        "unit": "zh_char_per_sec"
      },
      "pitch": "mid",
      "speaker_volume": "normal"
    },
    {
      "speaker_id": "speaker_2",
      "speech_rate": {
        "band": "fast",
        "value": 165.0,
        "unit": "word_per_min"
      },
      "pitch": "high",
      "speaker_volume": "low"
    }
  ]
}
```

字段语义：

- `profiles` 是当前片段内、由决策时间轴区分出的 speaker-local 列表，不是跨文件
  的声纹身份。无法得到可靠时间轴时为 `null`；确认没有语音时为 `[]`。
- `speaker_id` 使用 `speaker_1`、`speaker_2` 等片段内稳定索引。画像必须来自与
  `speaker_count` 相同的 decision timeline，不能让 profile 数量与另一路 timeline
  静默拼接。
- `speech_rate.band` 取 `slow`、`normal`、`fast`、`variable` 或 `null`。不在
  首版承诺 `ultrafast/free`，避免把极短片段、朗读、唱歌或 ASR 失败误报为速度档位。
- `speech_rate.value` 只有在文本单位、说话有效时长和语言单位都可靠时才填写；
  中文使用 `zh_char_per_sec`，拉丁语系使用 `word_per_min`，不能跨语言直接比较数值。
  `unit` 未知时 value 必须为 `null`。
- `pitch` 是相对 F0 档位（`low/mid/high/variable/null`），不等于音高感知的绝对
  Hz，也不能直接映射为 `male/female`。
- `speaker_volume` 是同一片段内的相对响度档位（`low/normal/loud/variable/null`），
  不表示跨录音可比较的 dB。
- `confidence`、模型分数、原始 F0、VAD 区间、文本证据、校准 profile 和 evidence
  id 继续只写内部 artifact，不进入 tags-only 公共对象。

### 4.2 空值和一致性规则

1. profile 计算失败不能覆盖六个已有 speaker claim；新增字段独立为 `null`。
2. profile 数量、speaker id 和选中 timeline 的可观测 speaker 集合必须可审计；不能
   为了补齐 schema 虚构 profile。
3. 重叠区间、低于最小时长的片段、无 speech coverage 的区间不参与语速/F0 汇总。
4. 若只有整段文本而没有 speaker 对齐，允许保留内部 clip-level rate，但公共
   `profiles[].speech_rate` 必须为 `null`。
5. `variable` 只在同一 speaker 的有效片段之间存在稳定差异并达到校准阈值时使用；
   “模型不确定”用 `null`，不能滥用 `variable`。

## 5. 语速实现方案（首个交付重点）

### 5.1 证据和路由

1. 按语言路由选出 `speaker_count` 的 decision timeline：中文使用 MOSS，其他或
   unknown 使用 OpenASR 0.1.40 trace raw turns；profile 不再另选一套 timeline。
2. 中文只使用 MOSS joint segments 的 `speaker_id`、文本和起止时间；非中文使用
   OpenASR trace timeline。当前同一次调用的 FireRedASR2-AED 只有整段 stdout
   text，没有受信 lexical timestamp，不能直接分配给某个 speaker；没有其它受信
   lexical evidence 时，非中文公开 `speech_rate` 保持 `null`。需要独立校验时可打开
   Whisper lexical timeline，并通过现有 `project_asr_track` 投影到 speaker timeline；
   Whisper 不成为 speaker event 票。
3. 用 Brouhaha speech segments；不可用时使用 FireRedVAD。将 VAD 区间与 speaker
   activity segments 求交，去除重叠和静音，得到每个 speaker 的 speech-active seconds。
4. 在内部 evidence 中记录上游 evidence id、文本来源、VAD 来源、语言、有效时长、
   文本单位数、片段数、丢弃原因和 `rate_basis`。

### 5.2 语言感知的数值

- 中文/日文等 CJK：去除标点和明显填充符后计有效汉字/音节代理，输出
  `zh_char_per_sec`（名称可扩展为 `cjk_char_per_sec`）。
- 英文及主要拉丁语系：按已有 tokenizer 计词，输出 `word_per_min`；缩写、数字和
  混合语言规则在 benchmark 中冻结。
- 公式为“文本单位数 / speech-active seconds”。不能用整条 clip duration 直接除，
  否则会把停顿、换人和背景静音当作慢语速。
- 首版至少要求一名 speaker 累计有效语音约 3 秒且达到最小文本单位数；不满足即
  `null`。阈值应在 dev 集冻结，不能在 test 集调参。
- 使用片段级 robust median/trimmed mean；片段之间变异超过冻结阈值且证据充分时标为
  `variable`。`band` 的切分点按语言、场景和录音域在 dev 集校准，不硬编码全球统一
  WPM 或字符/秒阈值。

### 5.3 没有新模型的原因和限制

语速的第一版不需要训练模型：时间轴、文本单位和 VAD 已在当前链路中存在，新增的
只是一个可复现的统计适配器。需要补充的主要是 `librosa/scipy` 等声学计算依赖和
schema/evidence，不是新的神经网络权重。

但 MOSS 目前主要提供片段级而非稳定逐词时间戳，因此首版应把 `value` 标为
`estimated`/内部质量等级；若 benchmark 证明片段级数值不稳定，再启用现有 Whisper
的 word-timestamp 模式，仍不必先引入新模型。

## 6. 基础声学画像（同一 MVP）

### 6.1 音高

- 对每个 speaker 的 clean、非重叠、speech-active 区间做 F0 估计，汇总 median、
  低/高分位数和 voiced-frame ratio。
- 先公开相对 `pitch` 档位；原始 Hz、算法、voiced ratio、失败原因只进 evidence。
- 8 kHz、强噪声、混响、极短片段或 voiced ratio 不足时输出 `null`。
- 不能用 pitch 档位推导年龄或性别，也不能把不同录音增益/麦克风差异解释为说话人
  音色变化。

### 6.2 相对音量

- 对同一片段内各 speaker 的 RMS/响度做 robust 标准化，优先报告相对 `low/normal/loud`。
- 记录是否降混、是否存在 AGC/压缩、参考 speaker 数量和可用 speech duration。
- 只有单人或录音增益不可追踪时，优先 `null` 或 `normal` 的低风险结果；不输出跨样本
  可比较的绝对 dB。

### 6.3 暂不公开的画像

`distance`、`stereo_position`、开放式 `voice_traits`、`articulation`、`prosody`、
`delivery_style` 在首版只可作为内部诊断特征。它们需要更长的干净语音、明确的听感
标注和跨域评测，不能由简单 F0/RMS 规则伪造。

## 7. 是否引入新模型

### 7.1 当前结论

**Phase 0/1 不引入新模型。** 现有模型足以支撑“谁在什么时间说话、说了多少可计数
文本、有效说话多久”，这正是语速和基础声学画像的最小闭环。

以下现有资产不应被误用为画像模型：

- ECAPA/CAM++ 只做 speaker verification/identity comparison；
- MOSS（中文）与 OpenASR trace raw turns（非中文）负责当前语言路由时间轴；
  后者的完整调用依赖 DiariZen、ReDimNet、Qwen3-ForcedAligner 和 bundled Stream-VAD；
- MOSS 负责中文联合 ASR，OpenASR stdout FireRed 文本负责非中文 ASR；
- Brouhaha/独立 FireRedVAD 只负责额外 speech coverage，不等于 OpenASR bundled Stream-VAD；
- DASS 是背景声音分类器；
- FireRed LID 的语言识别不等于口音识别。

### 7.2 何时需要新模型

只有以下需求明确进入产品范围，才启动独立模型评估：

| 属性 | 是否需要新模型 | 原因与门槛 |
| --- | --- | --- |
| 年龄段 | 需要 | 现有链路无 age head；敏感属性，必须有授权标注、分群偏差评估和 opt-in 决策 |
| 性别 | 需要或明确不做 | 需要性别/声音表现定义、隐私和误用边界；pitch 不能替代性别分类 |
| 情绪 | 需要 | 需要 SER 模型和与公司枚举的标签映射；会议语音中的 neutral/混合情绪要有 abstain |
| 口音/方言 | 需要 | FireRed LID 只能作为语言线索，不能从语言、地点或姓名推断 accent |
| 音色/韵律开放标签 | 未必 | 可先做 openSMILE/eGeMAPS 等 DSP 特征的内部实验，但不能直接把规则结果发布为自然语言标签 |

若业务确认需要上述属性，建议先 benchmark 一个隔离的
`speaker_attribute` adapter（候选可包括 WavLM/Wav2Vec2 类 SSL encoder 加轻量
任务头或 SpeechBrain SER 方案），而不是修改 Sortformer/MOSS 的主模型。选型必须同时
记录 checkpoint、SHA256、runtime、license、语言/域、显存、RTF、失败率和 null/abstain
策略；未完成这些记录前不下载或接入生产模型。

### 7.3 新模型准入条件

- 有 speaker-level、meeting-separated 的 dev/test 标注，且 test 不用于阈值调节；
- 报告 macro-F1/MAE、校准、abstain 覆盖率、短语音/重叠/噪声/语言切片，而不是只报总体
  accuracy；
- 与 deterministic MVP 对比有可重复增益，不得降低六个既有 speaker claim；
- 对年龄/性别/情绪等属性完成隐私、偏差和产品使用边界评审；
- 模型失败写 missing evidence，不能静默回填 caption、native metadata 或未校准猜测。

## 8. 分阶段实施计划

当前实现记录：Phase 0~1 已在 v2 接入。新增
`speaker_profile_deterministic` evidence 和 nullable `speaker.profiles`；不引入新
神经模型，不改变六个既有 speaker claim。后续需要用分层 dev/test 集校准阈值、报告
覆盖率和 macro-F1，达到发布门槛后再考虑 Phase 2。

### Phase 0：合同和数据冻结

1. 冻结 `speaker_profile` v0.1 schema、`profiles` 空值规则和 `speech_rate` 单位。
2. 在不改动六个现有 claim 的前提下，增加 profile 版本、policy hash、模型/依赖清单
   到 run manifest 和 fusion artifact。
3. 从现有 AMI 1k 和 `caption_pairs_3000` 中分层抽取中文/英文、1/2/3+ speaker、
   overlap、短/长片段，建立人工语速和 pitch/音量小规模 gold。
4. 明确 captioner 结果只作为离线对照，不作为推理输入。

**完成条件**：旧 profile 输出逐字段 parity；新字段默认 `null`/`[]` 合同通过 schema
和单元测试；六个旧 claim 的数值和路由不变。

### Phase 1：MVP 语速 + 基础声学画像（无新模型）

1. 新增 `tagger/tools/speaker_v2/speaker_profile.py`（或等价模块），实现时间轴/VAD
   对齐、语言单位计数、speech-active duration、rate band、F0 和相对 RMS。
2. 在 `speaker_evidence.py` 中新增 `speaker_profile_acoustic` evidence；父依赖只能
   是实际 decision timeline、VAD 和 lexical evidence，不把派生字段当独立票。
3. 在 resolver/public adapter 中加入 `profiles`，只映射通过质量门控的字段。
4. 复用现有 subprocess 隔离模式；优先利用 MOSS runtime 已有的 `numpy/scipy/librosa`
   进行计算，必要时建立轻量 profile-DSP runtime，不把大型属性模型带入默认 profile。
5. 为每个 speaker 保存 internal metrics、coverage、弃用区间和 rate basis，公共输出
   只保留 schema 允许的字段。

**完成条件**：profile-to-timeline 映射覆盖率达到 95% 以上；不足时正确 abstain；语速
band 在 dev 上达到预设 macro-F1 目标（建议 ≥0.75），数值误差、语言切片和失败率均有
报告；profile 计算失败不影响旧 claim。

### Phase 2：质量校准和可选扩展

1. 评估启用 Whisper word timestamps 对 rate value 的增益，比较 MOSS segment-only、
   MOSS+VAD 和 MOSS+Whisper projection 三种方案。
2. 对 pitch/音量做采样率、AGC、混响和短片段 slice；冻结跨语言 band threshold。
3. 只有有足够标注时，才考虑 `distance`、更细的 prosody 或内部 voice-quality 特征；
   先存 artifact，默认不公开。
4. 在非 AMI meeting-separated 数据上复核，不因 AMI 内结果好就切换默认 profile。

### Phase 3：属性模型闸门（按需）

1. 业务确认需要 age/gender/emotion/accent 后，冻结每个属性的标签定义、允许的
   `null/abstain`、隐私边界和标注方案。
2. 对候选模型做独立 adapter benchmark、license/runtime 审计和分群评测。
3. 以 shadow evidence 接入；属性模型只写自己的 `speaker_attribute` 能力，不能参与
   count/multi/overlap/change 的决策。
4. 达到准入条件后再选择是否加入 `quality-shadow`；保留显式开关和 `legacy-shadow`
   回滚，不物理删除现有模型。

### Phase 4：发布和持续观测

1. 先发布 `profiles` 的 nullable 字段和 artifact，观察真实数据 coverage、null 率、
   语言/时长切片和额外 RTF。
2. 公开 schema 版本、profile schema hash、rate unit 和 evidence lineage。
3. 每次新模型或阈值变更重新跑 AMI 1k、非 AMI shadow 和回归测试；不通过时只回滚新增
   profile，不阻塞旧 speaker 六字段。

## 9. 代码、测试和产物清单

| 位置 | 计划变更 |
| --- | --- |
| `tagger/tools/speaker_v2/speaker_profile.py` | 新增确定性 profile metrics、语言单位、F0/RMS、空值与质量门控 |
| `tagger/pipelines/speaker_evidence.py` | 采集 profile evidence、依赖闭包、失败/缺失记录 |
| `tagger/tools/speaker_v2/contracts.py` | 增加 profile evidence/schema 合同和有限数校验 |
| `tagger/tools/speaker_v2/resolver.py` | 从 decision timeline 构造 `profiles`，保持旧 claim 路由不变 |
| `tagger/tools/speaker_v2/artifacts.py` | 写入 profile schema/version、metrics lineage 和运行清单 |
| `scripts/run_speaker_evidence_v2.py` | 增加 profile-DSP 开关、阈值和 runtime 参数；新模型只在后续显式打开 |
| `tests/test_speaker_v2_profile.py` | rate 单位/阈值、timeline 对齐、overlap 排除、短片段 abstain、F0/RMS 门控 |
| `tests/test_speaker_v2_profiles_pipeline.py` | profile 与 profile model overrides、失败隔离、artifact parity |
| `docs/`（本目录） | 维护 benchmark、阈值、模型清单、license 和发布记录 |

最小集成测试包括：单说话人、双说话人、换人、重叠、无语音、无文本、中文 MOSS
与非中文 OpenASR 组合链路、stdout 空文本、合法 0-turn trace、非法 trace、模型 pack
缺失、选定 runtime 失败、8 kHz、
双声道降混、短片段和 profile
计算异常。所有测试都应确认 native metadata、captioner 输出和 reference transcript
没有进入 inference/resolver。

## 10. 风险和明确不做的事

- **ASR 误差**：语速依赖文本单位；ASR 错误或漏字时降低质量等级并可输出 `null`，不
  用固定关键词补值。
- **speaker 对齐错误**：独立 Whisper 单元必须经过已有 projection；未唯一归属时不
  给某个 speaker 分配速率。
- **重叠和背景语音**：重叠区间不参与单 speaker 的 rate/F0 汇总；背景 speech 不
  虚构新的 profile。
- **录音条件**：RMS、F0 会受麦克风、AGC、采样率、混响和降混影响；公共值只做相对
  档位，不承诺跨录音绝对可比。
- **敏感属性**：年龄、性别、情绪和口音不在 MVP 自动推断，不从姓名、语言、地域、
  话题或 caption 反推。
- **模型膨胀**：不为了补齐 captioner 的全部字段而引入一组未经评估的模型；先证明
  语速和基础声学画像的产品价值。
- **发布门禁**：当前 speaker v2 已直接发布可决策六字段；画像新增字段采用独立
  nullable/abstain，不恢复旧 certification gate。

## 11. 最终建议

**Phase 0 + Phase 1 已完成接入**：`profiles[].speech_rate`、`pitch`、
`speaker_volume` 和内部 evidence 已实现，未下载新模型。下一步是分层 benchmark 和
阈值校准，再决定是否做 Phase 2；只有产品明确需要年龄/性别/情绪/口音，且有标注、
隐私、license 和跨域指标时，才启动 Phase 3 属性模型评估。
