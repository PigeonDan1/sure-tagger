# Benchmark 数据构成与标签覆盖

更新时间：2026-09-19  
数据版本：`structural-benchmark-subset-v1.0`  
面向写作的构成报告见 [`../../docs/benchmark-650-dataset-composition.md`](../../docs/benchmark-650-dataset-composition.md)。
统计对象：本目录当前 `benchmark.manifest.jsonl`、各 track metadata 和 GT sidecar

本文描述实际落盘的 650 条数据，不以原始数据集总规模或上游候选池规模代替本子集统计。T4 controlled
已从当前 benchmark 禁用；其历史源数据不属于本目录的有效评测输入。
字段角色和得分率以 [`SCORING_STANDARD.md`](SCORING_STANDARD.md) 的
`benchmark-scoring-v0.8` 为准；默认能力雷达包含 18 个轴。
统一 `benchmark.manifest.jsonl` 是用于审计的封装清单，不是模型请求体；它可能保留路径、track
和 source-like ID。正式 runner 必须从中生成只含解码音频和 opaque sample ID 的 model-facing
runtime view。本文提到的 transcript、标签、stems、source annotation 和 recipe 都属于
GT/provenance，不能传给被测系统；若实现暴露了封装元数据，必须按评分规范设置 leakage mask。

## 1. 总体构成

| Track | 数据集 | 条数 | 输入音频时长 | 单条时长范围 | 主要用途 |
| --- | --- | ---: | ---: | ---: | --- |
| T1 natural meeting | AMI | 350 | 145.05 min | 15.900--94.523 s | speaker、overlap、ASR、英语会议语音 |
| T2 language/ASR | FLEURS + LibriSpeech | 150 | 26.79 min | 2.175--23.670 s | 10 种语言 LID/ASR、英文 clean anchor |
| T3 event/noise | DESED + FSD50K | 150 | 21.21 min | 0.522--28.922 s | 事件和 broad noise 闭世界多标签 |
| **合计** |  | **650** | **193.05 min** |  | 约 3.2 小时系统输入音频 |

T1/T2/T3 配额分别为 350/150/150。T2 由 FLEURS 100 + LibriSpeech 50 组成；T3 由
DESED 75 + FSD50K 75 组成。

## 2. 结构字段覆盖总览

下表逐项列出冻结 field universe 的全部 35 个结构字段。每个字段使用自己的 `gt_mask` 和 eligible
分母独立计分；系统的细粒度字段为 `null` 不会屏蔽同组粗粒度或派生字段。系统内部依赖和目标都
已知时另做合同一致性检查，该检查不覆盖预测值。

| # | 结构字段 | 有效来源/状态 | 评分政策 |
| ---: | --- | --- | --- |
| 1 | `basic_acoustic.duration_sec` | 全部 650 条 WAV header | tolerance/MAE |
| 2 | `basic_acoustic.sample_rate_hz` | 全部 650 条 WAV header | integer exact |
| 3 | `basic_acoustic.channels` | 全部 650 条 WAV header | integer exact |
| 4 | `basic_acoustic.nonspeech_ratio` | T1 350 条；官方 `silence_ratio` GT 由合并 AMI utterance turn 的补集物化 | absolute ratio，`2*(1-abs(预测-GT))`；兼容预测字段 `silence_ratio` |
| 5 | `basic_acoustic.silence_segments` | 当前无冻结 interval GT | interval 指标；mask=false |
| 6 | `audio_quality.snr_db` | 当前无冻结 GT | mask=false |
| 7 | `audio_quality.dnsmos_sig` | 无独立 GT | mask=false |
| 8 | `audio_quality.dnsmos_bak` | 无独立 GT | mask=false |
| 9 | `audio_quality.dnsmos_ovrl` | 无独立 GT | mask=false |
| 10 | `audio_quality.dnsmos_p808` | 无独立 GT | mask=false |
| 11 | `room_acoustic.far_field` | near/far 定义未冻结 | mask=false |
| 12 | `room_acoustic.rt60_sec` | 当前无冻结 GT | mask=false |
| 13 | `room_acoustic.c50_db` | 当前无冻结 GT | mask=false |
| 14 | `sound_field_scene.speech_music_events` | T3 150 条 closed-world | multilabel；本版无 `singing` 正例 |
| 15 | `sound_field_scene.music_present` | T3 从 GT event 物化 | **独立 boolean 计分**；系统 event set 可为 `null` |
| 16 | `sound_field_scene.external_noise_type` | T3 150 | multilabel；本版无 `channel_environment` 正例 |
| 17 | `sound_field_scene.noise_composition` | 无等价 DASS fine-label GT | mask=false |
| 18 | `speaker.speaker_count` | T1 350 条 turn | integer exact/MAE |
| 19 | `speaker.speaker_present` | T1 从 GT speaker activity 物化 | **独立 boolean 计分**；系统 count 可为 `null` |
| 20 | `speaker.asr_transcript` | T1 350 + T2 150 | WER/CER；共 500 条 reference |
| 21 | `speaker.multi_speaker` | T1 | 独立 boolean F1/accuracy |
| 22 | `speaker.speaker_change_count` | T1 turn 可派生 | 对称归一化绝对误差；另报 exact/MAE |
| 23 | `speaker.speaker_change` | T1 | 独立 boolean F1/accuracy；count 可为 `null` |
| 24 | `speaker.overlap_ratio` | T1 turn 可派生 | continuous MAE/Spearman |
| 25 | `speaker.speaker_overlap` | T1 | 独立 boolean F1/accuracy；ratio 可为 `null` |
| 26 | `speaker.profiles` | 缺少可稳定对齐的完整 profile GT | 容器合同/coverage；能力 mask=false |
| 27 | `language_content.language` | AMI corpus-level English；T2 150 条 | categorical accuracy/macro-F1 |
| 28 | `language_content.word_count` | T1/T2 reference transcript 可物化 | Headline/capability；对称归一化绝对误差；另报 exact/MAE；需冻结各语言 tokenizer |
| 29 | `language_content.punctuation.punctuation_count` | T1/T2 reference transcript 可物化 | Headline/capability；对称归一化绝对误差；另报 exact/MAE |
| 30 | `language_content.punctuation.has_terminal_punctuation` | T1/T2 reference transcript 可物化 | Headline/capability；独立 boolean；punctuation count 可为 `null` |
| 31 | `language_content.repetition.has_repetition` | T1/T2 reference transcript 可物化 | Headline/capability；独立 boolean；repetition count 可为 `null` |
| 32 | `language_content.repetition.repetition_count` | T1/T2 reference transcript 可物化 | Headline/capability；独立 integer exact/MAE |
| 33 | `language_content.filler` | T1 英语规则可物化；T2 需语言词表 | Headline/capability；独立 integer exact/MAE |
| 34 | `deterministic_derived.speaker_change_rate_per_min` | T1 从 GT count+duration 物化 | **对称归一化绝对误差**；系统 count/duration 可为 `null` |
| 35 | `deterministic_derived.filler_ratio` | 有 reference word/filler GT 的样本 | **独立 continuous 计分**；系统 word/filler 可为 `null` |

这里的“从 GT 物化”只描述参考答案的生成方式，不表示 captioner 可以从 ASR 或其它预测字段
直接推导这些值。所有能力字段（含 `nonspeech_ratio`、6 个词面叶字段和既有派生目标）都要求系统
独立输出、独立进入自己的 eligible 分母并提供 `score_rate`；某个依赖字段为 `null` 不会屏蔽
目标字段。例如 captioner 可以正确给出 `speaker_present=true`，同时令无法确定的
`speaker_count=null`；前者正常获得能力分，后者在自己的字段分母中记 abstention。
`descriptive.*` 不属于这 35 个结构字段，继续不计分。

## 3. T1 AMI：350 条

### 3.1 Meeting 与抽样

- 覆盖完整本地 utterance 树的 170 个 meeting。
- 160 个 meeting 各取 2 条，10 个 meeting 各取 3 条。
- 每个 meeting 内按时间排序后取 midpoint quantiles，不使用随机种子。
- meeting 类别样本数：EN 33、ES 123、IB 15、IN 20、IS 79、TS 80。
- 每条在 `metadata.jsonl` 中保存源 manifest 路径、原行号、源 manifest SHA256、原始记录 SHA256
  和输入音频 SHA256，可回溯到完整 AMI 树。

### 3.2 实际说话人结构

以下计数直接按选中 manifest 内出现的唯一 speaker ID 和原始 turn 区间统计，用于说明数据构成；
它们不是替代正式 GT builder 的新口径。

| Clip 内原始 speaker 数 | 条数 |
| ---: | ---: |
| 1 | 45 |
| 2 | 84 |
| 3 | 122 |
| 4 | 99 |

305/350 条包含至少两名说话人。按不同 speaker 原始 turn 区间是否相交作描述性扫描，288 条存在
overlap、62 条未观察到 overlap。每条包含 1--26 个原始 utterance，350/350 均有 reference
transcript。AMI reference manifest 含 transcript、speaker 和逐词时间戳，只能用于 GT；统一
benchmark manifest 已将 `text.transcript` 和 `native_metadata` 清空。

`t1-ami/speaker_gender_gt/` 另提供 350 条样本、975 个活跃说话人对的性别 GT：官方 AMI
`participants.xml` 覆盖 750 对，其余 225 对为 Qwen3.5-Omni-Plus 回退；说话人实例为男性 666、
女性 309。完整分布与使用边界见构成报告第 3.5 节。

## 4. T2 多语言与 ASR：150 条

FLEURS 从十个 locale 的 400 条上游面板中各取 10 条；LibriSpeech 从 100 条 clean panel 中取
50 条。每个分层仍使用 midpoint quantiles。

| 来源/语言 | locale | 条数 |
| --- | --- | ---: |
| FLEURS 阿拉伯语（埃及） | `ar_eg` | 10 |
| FLEURS 普通话简体中文 | `cmn_hans_cn` | 10 |
| FLEURS 英语（美国） | `en_us` | 10 |
| FLEURS 西班牙语（拉丁美洲） | `es_419` | 10 |
| FLEURS 法语（法国） | `fr_fr` | 10 |
| FLEURS 印地语（印度） | `hi_in` | 10 |
| FLEURS 日语（日本） | `ja_jp` | 10 |
| FLEURS 韩语（韩国） | `ko_kr` | 10 |
| FLEURS 俄语（俄罗斯） | `ru_ru` | 10 |
| FLEURS 斯瓦希里语（肯尼亚） | `sw_ke` | 10 |
| LibriSpeech 英语 clean anchor | `en` | 50 |

FLEURS 100/100 和 LibriSpeech 50/50 均有 reference transcript 与语言标签。FLEURS 时长为
3.960--20.880 s，LibriSpeech 为 2.175--23.670 s。两者在当前 metadata 中均记录为
`CC-BY-4.0`；它们不提供自然会议 overlap、环境事件、SNR 或房间声学 GT。

## 5. T3 事件与背景声：150 条

DESED 75 条来自 synthetic21 strong annotation，`gt_strength=D0`；FSD50K 75 条为 clip-level
verified/weak 多标签，`gt_strength=W`，其中 dev 38、eval 37。两者只在本 track 内按
`t3-native-label-closed-world-v1.0` 解释为闭世界：原生标签映射为正例，没有正映射的目标类记为
负例。该负例不是人工确认负例，可能含漏标假负例。

### 5.1 冻结目标类及实际正负例

| Schema 字段 | 目标标签 | 正例 | 闭世界负例 | 主要正例来源 |
| --- | --- | ---: | ---: | --- |
| `speech_music_events` | `speech` | 69 | 81 | DESED Speech；FSD50K Speech/Conversation/Speech synthesizer |
| `speech_music_events` | `singing` | **0** | 150 | 本子集无原生 singing 正标签 |
| `speech_music_events` | `music` | 38 | 112 | FSD50K Music/Musical instrument |
| `external_noise_type` | `music` | 38 | 112 | FSD50K Music/Musical instrument |
| `external_noise_type` | `animal` | 20 | 130 | Cat/Dog/Bird/Animal 等 |
| `external_noise_type` | `mechanical` | 68 | 82 | appliance、vehicle、mechanism、alarm/door 等 |
| `external_noise_type` | `nature` | 13 | 137 | water/rain/thunder/ocean 等 |
| `external_noise_type` | `formless` | 20 | 130 | frying/explosion/fireworks/crackle/whoosh 等 |
| `external_noise_type` | `channel_environment` | **0** | 150 | 本子集无原生 channel/environment 正标签 |

多标签可同时出现，因此各类正例数不能相加当作样本数。150 条中观察到 10 个不同 DESED 原生标签
和 87 个不同 FSD50K 原生标签；每个实际出现的原生标签到目标类的精确映射见
[`crosswalk.json`](t3-event-noise/crosswalk.json)，逐样本正负 class mask 见
[`closed_world_gt.jsonl`](t3-event-noise/closed_world_gt.jsonl)。

`singing` 和 `channel_environment` 在本子集中只有闭世界负例，可用于观察误报，但没有正例召回、
F1 或 AP 的有效正例分母。报告 macro 指标时必须披露或排除这两个 zero-positive class，不能把
“150 个负例”表述为覆盖了这两类正声音。

### 5.2 License 构成

- DESED 75 条保留 dataset-specific DCASE/DESED 条款引用。
- FSD50K 75 条中：CC BY 3.0 为 41 条，CC BY-NC 3.0 为 13 条，Sampling+ 1.0 为 13 条，
  CC0 为 8 条。

这些条款是源 metadata 的记录，不构成新的统一再分发许可。

## 6. 历史 T4 受控声学（已禁用）

以下内容保留用于审计此前的 350 条 T4 子集；T4 已从当前 benchmark 禁用，不再由构建器复制或由
runtime/scorer 消费。

### 6.1 组合因素

| 因素 | 档位与条数 |
| --- | --- |
| speaker count | 0: 35；1: 105；2: 105；3: 70；4: 35 |
| speaker change count | 0: 140；1: 76；2: 52；3: 23；5: 59 |
| requested overlap ratio | 0: 192；0.05: 74；0.20: 54；0.40: 30 |
| noise role | none: 82；music: 174；noise: 94 |
| room type | small: 117；medium: 118；large: 115 |
| requested SNR (dB) | none: 82；20: 72；10: 52；5: 52；0: 42；-5: 50 |

speaker event 正例为 `speech` 315 条、`music` 174 条、`singing` 0 条。94 条有效 noise-role 样本的
`external_noise_type` 为 `formless`；本来源池没有加入 animal/mechanical/nature/
channel_environment broad-noise role。82 条 `noise_role=null` 的样本虽然当前 renderer 仍可能
写出全零 `stems/noise/*.wav`，但不计为有效 noise role。music role 通过 `event_presence=music`
表达，不在 T4 `external_noise_type` 中重复写 `music`。

### 6.2 连续 GT 与 mask

| 字段 | `gt_mask=true` | 实测范围 | 中位数 |
| --- | ---: | ---: | ---: |
| `snr_db` | 268 | -5.000--20.000 dB | 5.000 dB |
| `rt60_sec` | 350 | 1.347019--5.374905 s | 2.264024 s |
| `c50_db` | 350 | 5.851622--23.798909 dB | 16.660920 dB |
| `external_noise_type` | 268 | clip-level set | -- |
| `noise_composition` | 0 | 不可评 | -- |

82 条 `noise_role=null` 的记录没有有限 SNR，保持 `null` 且 `mask_policy.snr_db=false`，不
用 Infinity、0 dB 或任意上限代填；全零 noise 文件不改变这个 mask。
RT60/C50 来自保存的 RIR 和冻结测量算法。requested 值只用于分层，连续指标读取 `measured`。
其中 35 条 `speaker_count=0`/`noise_role=null` 的输入是全零静音，但 room GT 仍由独立保存的
RIR 测量提供，因此当前 room mask 继续保留 350 条；这部分应在结果中标注为不可从混音内容
直接观测的压力测试层。可另报 `room_observable_mask=(active_speech_sec>0)` 的
observable-only 诊断值（当前 315 条），但不能替换官方分母。

## 7. 明确缺口与使用边界

1. `singing` 与 `channel_environment` 没有正例；不能报告这两类的正例召回或把它们纳入未说明的
   macro-F1。
2. T4 broad noise 正例仅有 `formless`；T3 才提供 animal/mechanical/nature/music 的正例覆盖。
3. AMI 本版覆盖英语会议，T2 覆盖十种语言，但没有片段级 code-switching GT。
4. T1 speaker/change/overlap 来源标注齐全，但除原始 manifest 外尚需版本化派生器生成 scorer 直接
   消费的 dense GT；历史 T4 GT 不属于当前 benchmark。
5. T3 闭世界负例是预注册评测假设，不能称作人工核听确认的负例。
6. 数据未覆盖 DNSMOS、far-field、DASS fine noise composition 和开放描述文本。
7. FSD50K、DESED、MUSAN、RIRS_NOISES 等源许可异质；本地 bundle 完整不等于可以统一公开再分发。

机器计数与路径/hash 验收结果见 [`benchmark.summary.json`](benchmark.summary.json)。重建或复验命令：

```bash
python3 compare/dataset/benchmark/build_benchmark_subset.py
python3 compare/dataset/benchmark/build_benchmark_subset.py --validate-only
```
