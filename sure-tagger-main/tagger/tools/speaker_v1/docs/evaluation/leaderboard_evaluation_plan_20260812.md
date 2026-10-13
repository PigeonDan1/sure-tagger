# Sure-Tagger 多说话人模型能力评测与榜单协议

> 版本：`evaluation-protocol-v0.2`
>
> 日期：2026-08-12
>
> 状态：评测设计，尚未生成正式榜单
>
> 适用输入：现有 utterance-level `sample`，不修改输入 schema

> 文档边界：本文用于“模型能力发现、模型选型与组合消融”，在任何生产链路职责分配之前执行。[`evidence_fusion_design.md`](../evidence_fusion_design.md) 和 [`speaker_evidence_pipeline_brief.md`](../speaker_evidence_pipeline_brief.md) 描述待评测的 fusion baseline/目标架构；其中的“默认模型角色”不是选型结论，也不应限制本协议的候选池和单模型评测范围。

候选资产与真实运行状态见 [`candidate_status_20260812.md`](candidate_status_20260812.md)。其中 `diagnostic_measured`、`blocked`、`N/A` 和 `pending` 必须分开记录，不能用空白或 0 混淆。

## 0. 先说结论

这轮工作要回答的不是“如何给当前 pipeline 的几个 claim 校准置信度”，而是：

1. 哪个模型擅长估计 speaker count；
2. 哪个模型擅长判断是否存在多个 speaker；
3. 哪个模型擅长定位 overlap；
4. 哪个模型擅长定位真正的 floor transfer/change；
5. 哪个模型适合做 speaker identity/continuity 验证；
6. 哪些 ASR、VAD 或 separation 模型能提供有用的辅助证据；
7. 哪些模型的错误互补，组合后能带来真实增益。

因此采用“先测模型能力，再设计 Sure-Tagger 链路”的顺序：

```text
候选模型池
    |
    v
同一 utterance、同一预处理、盲推理
    |
    v
统一 gold scorer（推理完成后才读取 native metadata）
    |
    +--> count leaderboard
    +--> multi-speaker leaderboard
    +--> overlap leaderboard
    +--> change leaderboard
    +--> identity / VAD / ASR 专项榜
    |
    v
错误分析、分层结果、置信度和成本
    |
    v
模型适配矩阵与组合消融
    |
    v
按 claim 和数据场景选择模型组合
```

本协议不预设 MOSS、Sortformer、CAM++ 或任何现有模型的职责。现有架构只作为待比较的 baseline；如果新模型在某个能力上更好，组合方案应当替换 baseline。

## 1. 评测对象与三个层次

需要分开回答三个问题，不能用一个总分混在一起。

| 层次 | 问题 | 产物 |
| --- | --- | --- |
| 单模型能力 | 模型独立完成某项任务的准确性、覆盖率和稳定性如何 | 模型能力榜 |
| 证据互补性 | 模型是否能发现另一个模型漏掉的正确事件，错误是否相关 | 增量收益/互补性矩阵 |
| 系统组合 | 多模型在固定规则或学习型 resolver 下是否改善最终标签 | Fusion/portfolio 榜 |

只有第一层测完之后，才决定第二层和第三层。不能因为某个模型已经在当前 pipeline 中被指定为 `identity_guard`，就只评 identity；也不能因为某个模型输出 timeline，就默认它适合所有 speaker claim。

## 2. 输入边界与公平运行规则

### 2.1 主评测输入

- 每条输入仍是现有的 utterance-level `sample`。
- 模型主评测只读取当前 `sample.audio.path` 指向的音频文件，整份文件作为一个 sample。
- 不增加 `start_sec`、`end_sec`、邻接音频或 recording 级输入字段。
- 不读取同一 recording 的其它 utterance，不跨 sample 借用 speaker ID、embedding、transcript 或上下文。
- `sample.native_metadata` 和 reference words 只能在所有模型推理完成后进入 gold scorer。
- `sample.text.transcript` 不进入主评测推理。它是数据集携带的 reference/输入字段，不能在 audio-only 榜中给模型使用。

### 2.2 输入模式分榜

为了避免把“音频能力”和“文本辅助能力”混为一谈，设置两个输入模式：

| 模式 | 模型可以读取 | 用途 |
| --- | --- | --- |
| `audio_only`（主榜） | 当前 sample 音频 | 比较真实的声学 speaker 能力 |
| `audio_plus_hypothesis_text`（辅助榜） | 当前 sample 音频 + 独立 ASR 生成的 hypothesis text/timestamps | 测量 ASR 文本对边界、assignment 和冲突判断的增益 |

辅助榜中的文本必须由独立 ASR 在推理阶段生成，不能使用 native reference transcript。给模型喂 gold transcript 的结果只能叫 `oracle_text`，单独列为上限实验，不能进入主榜。

### 2.3 其它公平规则

- 需要 `num_speakers` 的模型默认运行 `blind_count`，不能传入 gold speaker count。
- `oracle_num_speakers` 只作为诊断榜，用来区分“模型声学能力”和“聚类/人数先验依赖”。
- streaming 和 offline/full-context 分开排名；不能把因果延迟模型与离线模型直接混排。
- 统一采样率、通道处理、sample-relative timebase 和输出裁剪范围。
- 模型原生后处理参数固定为 upstream default，另在 dev split 上运行一次预注册的 `dev_tuned` profile；test 不调参。
- 每个模型保留 raw timeline、frame score、segment score、embedding、ASR units、失败原因、运行时间和显存峰值。
- 模型不支持某个任务时输出 `N/A/unsupported`，不能把 `N/A` 当成错误或负例。

## 3. 要测的能力和 gold 定义

公开 metadata 目前有三个 bool，但评测必须把 metadata 中的所有相关字段拆成可测任务。gold 由 native annotation 在推理后生成，模型选择阶段不能接触这些 gold。

| 能力 ID | 对应 metadata/内部任务 | 评测对象 |
| --- | --- | --- |
| `C` | 内部 `speaker_count`、`active_speaker_count` | 当前 utterance 内有效 speaker 的人数 |
| `M` | `speaker.multi_speaker` | 是否至少有两个有效 speaker |
| `O` | `speaker.speaker_overlap`、`is_overlapped`、`overlap_ratio` | 是否有 material simultaneous speech，以及 overlap 时间区域 |
| `X` | `speaker_change`、`speaker_change_count`、change points | 现行 any-change 与候选 floor-transfer 分轨评测 |
| `I` | speaker identity/continuity | 两个声学片段是否来自同一/不同 speaker |
| `V` | speech coverage | speech/silence frame 和边界 |
| `A` | transcript/lexical clock | 文本、词时间和 speech coverage |
| `D` | full diarization | speaker-attributed timeline 的整体质量 |

### 3.1 统一 canonical time grid

1. 从 native utterance intervals 构造 sample-relative reference RTTM。
2. 所有 timeline 映射到统一的 20 ms frame grid；需要更细边界的模型另保存原始时间。
3. 每个 frame 保存 active speaker set，而不是只保存一个 primary speaker。
4. speaker ID 通过 Hungarian/最优 permutation mapping 对齐，不能直接比较模型输出的匿名 ID 字符串。

### 3.2 Count 和 multi-speaker

Count 必须分成两条明确的 gold track。当前公开 schema 只有三个 bool；count 榜评估的是它们的内部源字段，不是已公开的第四个 tag。

- `metadata_v0_active_count`（主榜）：严格复现 `speaker_metrics_v0.1.0`/`speaker_diarization_v0.1` 的默认预处理：先丢弃小于 `0.10 s` 的单个 segment，再按 speaker 合并间隔不超过 `0.30 s` 的 segment，最后统计 `utterances[].active_speaker_count`；`metadata_v0_multi = (metadata_v0_active_count >= 2)`。
- `material_100ms_count`（敏感性榜）：每个 speaker 在当前 sample 内的 union activity 至少达到 `0.10 s` 才计数；`material_100ms_multi = (material_100ms_count >= 2)`。
- 同时保存 `raw_any_activity_count`、每个 speaker 的 union activity 和被阈值排除的 speaker 列表。
- 主汇报默认使用 `metadata_v0_*`；`material_100ms_*` 只用来测量极短 backchannel/标注碎片对排名的影响。若两条 track 结论不同，必须并列报告，不能挑选更有利的一条。
- 1/2/3/4/5 人分别报告，不只报告总平均。
- 模型的 slot 上限、overflow 和 unsupported 单独记录；例如最多 4 speaker 的模型遇到 5 人 sample 时不能静默截断。

### 3.3 Overlap

现行 public v0 主轨严格复现 `speaker_metrics_v0.1.0`：先执行上述 segment 过滤和同 speaker 合并，只保留单个连续 overlap 区间至少 `0.10 s` 的区间，然后计算：

```text
overlap_ratio = overlap_duration / speech_union_duration
is_overlapped = overlap_ratio >= 0.05
```

同时报告以下 track：

- `metadata_v0_overlap`（主榜）：上述 v0 全部预处理 + `overlap_ratio >= 0.05`；4%-6% 样本仍计入，因为它们是现行 contract 的真实边界。
- `raw_frame_overlap`（诊断）：在 20 ms grid 上的未阈值 simultaneous activity mask。
- `gray_zone_excluded`（稳健性辅助榜）：暂不计 ratio 在 `[0.04, 0.06]` 的 sample，只用来观察排名对 5% 边界的敏感性，不取代主榜。

gold 既包括 frame-level overlap mask，也包括 one-to-one overlap events、持续时间和 ratio。

### 3.4 Change

`speaker_change` 的现行公开语义是 `speaker_change_count > 0`，而 v2 fusion 设计希望收窄为 floor transfer。两者必须分 track，不能用新语义对旧 tag 静默评分：

1. `public_v0_change`（现行 tag 主榜）：完全按冻结的 `speaker_diarization_v0.1` change-point 生成规则，评估 bool、count 和 point matching；
2. `floor_transfer_v2`（新语义候选榜）：主导 floor 从 speaker A 转移到 speaker B，使用确定的 gap/overlap 规则和 0.25 s/0.5 s collar；
3. `different_speaker_onset`（事件诊断榜）：短 backchannel、插话或 overlap 中第二 speaker 的进入点。

对当前 Sure-Tagger 字段的榜单使用 `public_v0_change`。`floor_transfer_v2` 必须标明为新 semantics 试验，只有字段定义/版本完成评审后才能取代现行主榜。无法从 native interval 唯一确定 floor 的样本标记 `ambiguous`，不强行作为 floor-transfer 二元真值。

### 3.5 Identity、VAD 和 ASR gold

- `I`：从 native non-overlap intervals 生成同 speaker/不同 speaker pairs，按片段时长、语言和噪声分层；主评测只做 sample-local pairs，避免跨 sample 身份泄漏。
- `V`：native speaker interval 的 speech union 作为 frame-level speech reference，另报 onset/offset boundary error。
- `A`：native words 作为 reference，报告 WER/CER、word coverage、timestamp error；有可靠 speaker timeline 时再报告 SA-WER/cpWER。
- 没有独立 transcript reference 的数据集只能报告 lexical agreement/assignment ambiguity，不能命名为 ASR accuracy。

### 3.6 Gold 状态

每条 claim 的 gold 都有状态：

| gold 状态 | 含义 | 计入主指标 |
| --- | --- | --- |
| `clear` | annotation 能唯一确定标签/事件 | 是 |
| `ambiguous` | 例如 floor 语义无法从 annotation 唯一确定 | 单独报告，不进入对应的 floor-transfer 主指标；已有 public v0 contract 的边界样本不因“难”而排除 |
| `insufficient` | 没有足够 speech/reference | 从对应 claim 的 `n_evaluable` 排除，并报告数量 |

### 3.7 Gold provenance 和 scorer registry

`native_metadata` 不天然等于 gold。每个 dataset/claim 先登记 annotation 来源（human、forced alignment、model-generated 或混合）、可评能力、已知误差和人工抽检结果。model-generated annotation 不得在未审计时作为同类模型的权威 gold。

每个榜单行关联一条冻结 contract：

```text
claim_id / public_tag_or_internal_field
semantics_version
gold_source + annotation_provenance
preprocessing (min segment, merge gap, frame grid)
event_definition (min duration, merge gap, matching IoU/collar)
aggregation (frame micro, sample macro, session macro)
scorer_version + scorer_config_hash
```

P0 预注册的初始 scorer config 至少固定：20 ms grid；overlap 事件 min duration `0.10 s`、相邻事件 merge gap `0 s`、one-to-one matching 的最小 IoU；public v0 change 的 segment filter `0.10 s`、same-speaker merge gap `0.30 s`、different-speaker max gap `1.00 s`；floor-transfer v2 的 dominance window、允许 gap、overlap/backchannel 规则和 collar。未冻结的参数不得进正式榜。

## 4. 候选模型池

候选模型不限定为当前已部署的五路。先按能力类别建立池，再根据许可证、checkpoint 可得性和 smoke 结果决定是否进入全量运行。

### 4.1 Full timeline / diarization 候选

| 候选 | 主要输出 | 进入计划 | 注意事项 |
| --- | --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | joint ASR + anonymous speaker timeline | Round 1 | overlap 可能受单流生成限制；训练数据 provenance 待核 |
| NVIDIA Streaming Sortformer 4spk-v2 | overlap-aware frame activity、timeline、count candidate | Round 1 | 最多 4 slots；AMI 明确存在训练污染 |
| pyannote Community-1 | segmentation、OSD、embedding、count、overlap、assignment | Round 1（token 后） | gated 权重；固定 revision 和 telemetry 设置 |
| NeMo clustering diarizer（MarbleNet + TitaNet + clustering） | VAD + embedding + non-overlap timeline/count | Round 1 baseline | 默认不建模 overlap；不能把 overlap N/A 当 false |
| SpeechBrain ECAPA + 固定 VAD + AHC/spectral clustering | modular count/turn/continuity timeline | Round 1 baseline | 透明异构 baseline；checkpoint license 逐 revision 核查 |
| WeSpeaker / 3D-Speaker（ECAPA、CAM++、ERes2Net/TitaNet） | embedding + clustering/count | Round 2 | 同一 checkpoint 的不同 wrapper 不算独立 fusion 证据 |
| VBx / x-vector + Bayesian HMM | classical non-overlap timeline | Round 2 | 用于判断复杂模型是否真正带来增益 |
| DiariZen | end-to-end diarization | Round 2 audit | 先核实可复现 checkpoint、语言和 license |
| EEND-EDA / EEND-VC / TS-VAD | overlap-capable diarization | Round 2 audit | 部分配置需要 speaker count/enrollment；blind 与 oracle 分开 |

### 4.2 Overlap/OSD 专项候选

- Sortformer v2 的 frame activity；
- pyannote OSD/Community-1 内部 overlap 输出；
- EEND powerset/EDA；
- 可复现的 WavLM-OSD、CAT-Net 或其它公开 OSD checkpoint（待 license、权重和推理脚本审计）；
- 物理多通道 activity 只作为 layout-specific baseline，不作为通用 mono 模型。

OSD 模型只参加 `O` 榜；如果不能输出 speaker attribution，不能直接参加 `C` 或 `X` 榜。

### 4.3 Identity 专项候选

- 当前 CAM++ 中文 checkpoint；
- SpeechBrain ECAPA-TDNN；
- WeSpeaker/3D-Speaker 的 ECAPA、ERes2Net、TitaNet 或英文 checkpoint；
- x-vector/ResNet speaker verification baseline。

身份模型用同一套 native clean pairs 评测。CAM++ 在 AMI English 上是否适合，必须由该榜的英文分层结果决定，不能从中文 checkpoint 名称或 upstream threshold 推断。

### 4.4 VAD、ASR 和音频变换候选

| 类别 | 候选 |
| --- | --- |
| VAD | FireRedVAD、Brouhaha VAD、Silero VAD、WebRTC VAD、MarbleNet VAD |
| ASR/lexical clock | Whisper Base/Small/Large-v3、Paraformer-zh、SenseVoiceSmall、FunASR Conformer/Zipformer |
| separation transformation | SepFormer、Conv-TasNet、MossFormer2、TF-GridNet |

separation 不是 speaker evidence source。它只能作为变换链参加增量实验：`原音频 -> separation -> VAD/embedding/diarizer`，并与原音频 baseline 比较是否真实改善 `O`、`I` 或 `D`。

### 4.5 候选纳入门槛

模型进入全量榜必须满足：

1. 权重和代码可离线固定，能记录 revision/hash；
2. 许可证允许当前评测和后续部署，或明确标记 `audit_only`；
3. 能在 10 条分层 smoke 上完成至少 90% invocation；
4. 输出可转换到统一协议，或明确标记具体能力为 `N/A`；
5. 不能使用 native metadata、gold count 或邻接音频。

未满足条件的模型保留在 `candidate_pool`，不进入正式 rank，但报告阻塞原因。

### 4.6 候选扩展顺序

“不限于现有模型”不等于一次性安装所有项目。按对当前模型池的新信息量和可比性分批进入：

| 优先级 | 候选 | 必要性 | 首要对比 |
| --- | --- | --- | --- |
| P0 | MOSS、Sortformer v2、CAM++、Whisper Base、FireRedVAD | 已部署，先建立端到端评测基线 | 所有各自适用的 `C/M/O/X/D/I/V/A` |
| P0 diagnostic | Brouhaha VAD | 本地权重和环境已可离线运行；先作为 `V/SNR/C50` 辅助诊断，不进入 speaker claim 主榜 | `V` standalone、FireRedVAD 互补性、声学难度分层 |
| P1 | pyannote Community-1、NeMo clustering diarizer、ECAPA+clustering | 同时引入 powerset、modular neural clustering 和异构 embedding baseline | `C/M/X/D`，pyannote 另测 `O` |
| P1 | SpeechBrain ECAPA、ERes2Net/TitaNet 中至少一个 | 避免 identity 榜只有中文 CAM++ | `I`，重点看 AMI English、短 crop 和 overlap-adjacent slice |
| P1 | Silero/WebRTC/MarbleNet 中至少两个 | VAD 对成本和 domain shift 敏感，单一 FireRedVAD 无法形成榜单 | `V`，以及更换 VAD 对 clustering diarizer 的增量影响 |
| P2 | VBx、EEND-EDA/EEND-VC、DiariZen、独立 OSD | 在基线结果暴露 count/overlap/change 缺口后定向扩展 | 相应 claim 的 conditional recall 和 cost per additional correct sample |
| P2 | Whisper Small/Large-v3、Paraformer、SenseVoice/Zipformer | 区分 ASR 模型规模、语言和 timestamp 机制的影响 | `A` 与 text-assisted 增量，不参加纯声学 speaker 事件投票 |
| P3 | separation 组合 | 成本高且是 transformation，只在已知困难 slice 上有必要 | paired ablation，不独立排名为 speaker model |

每批结束后根据错误分布决定下一批：如果新模型与当前最佳模型错误几乎完全重合，即使 standalone 指标接近，也不优先纳入生产 portfolio。

## 5. 模型能力摸底实验

### 5.1 每个模型都跑所有适用任务

榜单的最小比较单位是 `checkpoint + frozen adapter/pipeline + profile`，不是一个模型品牌名。在查看 test gold 前，为每个候选登记“原子输出如何确定性映射为某个 task prediction”；有冻结 adapter 就参评，无法形成该任务输出才记 `N/A`。

不是先给模型分配角色，而是先收集它能产生的全部证据。例如：

- MOSS 同时评 `C/M/O/X/D/A`；
- Sortformer 同时评 `C/M/O/X/D`，不因没有 ASR 就给 `A` 分数；
- ECAPA+clustering 评 `C/M/X/D`，如果该冻结 pipeline 不产生并发 activity，`O` 标 `N/A`；
- CAM++ atomic profile 评 `I`；`CAM++ + 固定 VAD + 固定 clustering` 是另一个 composed candidate，可评 `C/M/X/D`，但 pair score 不能直接参加 `O`；
- pyannote full diarization profile 可评 `C/M/O/X/D`；若能从冻结的 cluster assignment 产生 sample-local same/different pair prediction，该 composed profile 也参加 `I`；
- Whisper 评 `A` 和 lexical/boundary diagnostics，不能凭单流文本判断没有 overlap；
- VAD 评 `V`，不能推断 speaker count。

每个模型的结果行应至少包含：

```text
model_id
revision / checkpoint_sha256
adapter_id / adapter_version / config_hash
input_mode
inference_profile (blind / oracle_count / streaming / offline)
capabilities_declared
capabilities_observed
evidence_family_id / upstream_dependency_ids / derived_from
raw_outputs_path
n_total / n_evaluable / n_unsupported / n_failed / n_abstained
primary_metrics
confidence_metrics
runtime / RTF / peak_vram
license / training_provenance / contamination_flag
```

### 5.2 运行 profile

同一模型至少产生以下可区分 profile：

| profile | 是否进主榜 | 目的 |
| --- | --- | --- |
| `blind_audio_only` | 是 | 测量真实端到端能力 |
| `dev_tuned_audio_only` | 是，单独一行 | 仅用 dev 调整阈值/聚类参数后的能力 |
| `oracle_num_speakers` | 否 | 判断错误来自人数估计还是声学分离 |
| `audio_plus_hypothesis_text` | 辅助榜 | 测量 ASR 文本增益 |
| `oracle_text` | 否 | 上限分析，不能作为部署结果 |

### 5.3 不预先固定模型角色

在第一轮报告之前，不写“某模型是 event vote、某模型是 identity guard”。这些是评测后的结论。评测报告中同时保存：

- 模型宣称的能力；
- 实际测出的能力；
- 在哪些 slice 上能力成立；
- 哪些错误是独有的，哪些与其它模型重合；
- 是否值得作为独立证据或只适合作诊断。

## 6. 指标体系

所有指标按 dataset、language、layout、duration、reference count、overlap ratio、SNR 和样本难度分层。正式数据同时报告 frame-micro、sample-macro 和 session-macro 聚合，主排序使用 session-macro，防止长 sample 或长 meeting 主导结果。只有存在足够独立 session 时才给出 session-cluster bootstrap 95% CI；单 session 数据只报 descriptive point estimate 和错误清单。

### 6.1 Count (`C`)

- exact accuracy；
- MAE、RMSE、bias；
- within ±1 accuracy；
- under-count rate、over-count rate；
- 1/2/3/4/5 speaker confusion matrix；
- count distribution NLL/Brier（模型有完整 count probability 时）；
- unsupported/overflow rate；
- blind 与 oracle-count 差值。

Count 榜的主排序为 exact accuracy 和 MAE；不能让一个只输出 lower bound 的模型与 exact-count 模型混成同一分数。lower-bound recall、upper-bound violation 另列。

### 6.2 Multi-speaker (`M`)

- precision、recall、F1；
- balanced accuracy、MCC、specificity；
- AUROC/AUPRC（有连续 sample score 时）；
- 按真实人数分层的 recall；
- abstention/coverage、certified-only precision；
- negative specificity，避免 AMI 正例比例过高导致 accuracy 虚高。

### 6.3 Overlap (`O`)

- canonical frame precision/recall/F1；
- `metadata_v0_overlap` sample-level precision、recall、F1、MCC 和 specificity；
- overlap event one-to-one F1，event IoU；
- onset/offset error；
- overlap duration MAE/bias；
- overlap ratio MAE 和 reliability；
- `metadata_v0_overlap`、`raw_frame_overlap` 与 `gray_zone_excluded` 分轨结果。

没有 overlap 输出的模型在此处写 `N/A`，不能写 0，也不能用“没有检测到 overlap”充当负例。

### 6.4 Change (`X`)

- 现行 `public_v0_change` bool/count/point precision、recall、F1；
- floor-transfer point/event precision、recall、F1；
- collar 0.25 s 和 0.5 s 两套结果；
- boundary MAE；
- change event count MAE；
- floor-transfer 与 `different_speaker_onset/backchannel` 分开报告；
- 静音边界、短 backchannel、overlap 边界的 false positive rate。

`public_v0_change` 和 `floor_transfer_v2` 必须分别排名。汇报表的 track/semantics version 不得省略。

### 6.5 Full diarization (`D`)

- DER（collar 0、0.25 s）；
- JER；
- MISS、FA、CONFUSION 分解；
- overlap include/exclude 两套；
- oracle VAD 与 pipeline VAD 两套；
- frame speaker attribution F1/coverage；
- RTTM speaker permutation 使用 Hungarian mapping。

### 6.6 Identity (`I`)

- ROC-AUC、AUPRC；
- EER、minDCF；
- TPR@FAR=1%/5%；
- Brier、ECE、NLL；
- 按 crop 时长、语言、同/不同 channel、overlap 邻近程度分层；
- oracle clean crops 与 predicted clean crops 分开。

### 6.7 VAD (`V`) 与 ASR (`A`)

VAD：frame F1、miss/FA、speech coverage、onset/offset error、短 speech recall。Brouhaha 与 FireRedVAD 的 native raw frame score、阈值和后处理必须分别登记；当前 AMI 诊断只报告 `raw_native_speech_union_20ms`，不能把 Brouhaha 的 `confidence=1.0` 当成概率。Brouhaha 的 `SNR/C50` 只用于 acoustic-difficulty slice 和辅助 metadata 诊断，若没有独立 gold 则不参加质量数值排名。详见 [`brouhaha_vad_diagnostic_report_20260812.md`](brouhaha_vad_diagnostic_report_20260812.md)。

ASR：WER/CER、word coverage、word timestamp MAE、segment boundary error；有 gold speaker timeline 时增加 SA-WER/cpWER、word-speaker assignment accuracy 和 ambiguous/unassigned rate。

ASR 的文本一致性只能作为 lexical/ambiguity diagnostic，不能因为某个 ASR 没转出第二路文本就判定 `speaker_overlap=false`。

## 7. 置信度的公平测量

置信度测量不是先给当前 resolver 的 claim 配参数，而是对候选模型的原始输出做统一审计。

### 7.1 先测区分能力，再测校准

顺序固定为：

1. 用 audio-only blind profile 测 standalone quality；
2. 比较 raw score 的排序能力（AUROC/AUPRC、PR curve、EER 等）；
3. 再在 session-separated calibration split 上，用同一套预注册方法（Platt、isotonic 或 temperature，按输出类型选择）把每个模型映射到概率；
4. 在冻结的 test split 报 Brier、NLL、ECE、reliability diagram 和 risk-coverage。

这里校准的是“模型对 gold 任务的预测可靠性”，不是校准当前架构已经决定的 claim ownership。每一个候选模型都接受相同的流程；如果新模型在 overlap 上比现有模型更好，榜单应直接显示这一结果。

### 7.2 不同原始分数不直接比较

- sigmoid activity probability、cosine similarity、word probability、log probability 和固定 adapter confidence 不在同一量纲；
- 不做跨模型原始分数加权平均；
- MOSS 没有可用 native score 时，不能伪造 0-1 概率，confidence 栏写 `N/A`，只比较 hard prediction quality；
- 没有连续 score 的模型可另报 test-time perturbation consistency/abstention，但不能把一致性冒充概率校准。

### 7.3 报告模型可靠性而不是“谁的数字更大”

榜单中的置信度列至少包括：

```text
native_score_type
score_range
score_available_rate
AUROC / AUPRC（如适用）
Brier / NLL / ECE（校准后）
coverage at target risk
abstention rate
```

## 8. 从单模型结果决定模型职责

### 8.1 单模型能力矩阵

输出一张 `model_capability_matrix`，每个单元格不是主观描述，而是实测结果：

| model | C count | M multi | O overlap | X change | D DER/JER | I identity | V VAD | A ASR | best slices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOSS joint profile | pending | pending | pending | pending | pending | pending/N/A 由 adapter registry 冻结 | pending/N/A | pending | pending |
| Sortformer v2 timeline profile | pending | pending | pending | pending | pending | pending/N/A | pending/N/A | pending/N/A | pending |
| pyannote Community-1 full profile | pending | pending | pending | pending | pending | pending/N/A | pending/N/A | pending/N/A | pending |
| ECAPA + fixed VAD + clustering | pending | pending | pending/N/A | pending | pending | pending | 依赖冻结 VAD | pending/N/A | pending |
| CAM++ atomic identity profile | N/A | N/A | N/A | N/A | N/A | pending | N/A | N/A | pending |
| CAM++ + fixed VAD + clustering | pending | pending | N/A | pending | pending | pending | 依赖冻结 VAD | N/A | pending |
| Brouhaha VAD v0.9.0 atomic | N/A | N/A | N/A | N/A | N/A | N/A | **diagnostic measured; formal rank pending** | N/A | pending |

### 8.2 “适合”如何判定

不定义一个隐藏总分。对每个 claim 单独按照以下顺序判断：

1. 主指标（例如 overlap frame/event F1，count MAE，change F1）；
2. `n_evaluable` 和 unsupported/failed/abstain coverage；
3. 最差关键 slice，而不是只看总体平均；
4. 置信度和目标风险下 coverage；
5. RTF、峰值显存、许可证和训练污染。

两个候选的比较基于 session-paired metric difference CI，而不是“各自 95% CI 是否重叠”。只有差值落在预注册的 equivalence margin 内时才标记 tie；若样本不足以判断，标记“未决”。模型职责由结果产生：

- **claim specialist**：在一个 claim 及关键 slices 稳定领先；
- **generalist**：多个 claim 均位于 Pareto 前沿；
- **complementary witness**：单模型不一定第一，但在另一模型失败的样本上有高 conditional recall；
- **diagnostic-only**：能暴露风险，但不能独立支持该 claim；
- **not suitable**：能力缺失、覆盖不足或错误率不可接受。

### 8.3 互补性和增量收益

对每个模型 `m` 和 claim `c`，除 standalone 指标外计算：

```text
conditional_recall(m | baseline fails, c)
unique_correct_rate(m, c)
error_overlap(m, baseline, c)
joint_false_negative_rate(m + baseline, c)
incremental_gain(m added to baseline, c)
cost_per_additional_correct_sample
```

例如 overlap 模型的选择不只看其自身 F1，还要看：当当前最佳模型漏掉短 overlap 时，它能否补回；如果两者总在同一批样本上同时失败，则不构成有价值的独立证据。

互补性统计必须使用 out-of-fold 预测：先在其它 session 上选阈值/训练 resolver，再计算当前 held-out session 的 `unique_correct` 和 incremental gain。不得在同一批 test 样本上穷举模型对、选出最佳组合后仍报原 test 分数；这会产生 portfolio selection bias。对两组模型差异使用 session-cluster paired bootstrap/permutation test，并对同一 claim 中的多个候选比较做 FDR 控制。

## 9. Fusion 和组合榜

Fusion 必须在单模型榜之后单独评测，不能用组合结果反推某个单模型“本来就擅长”。

### 9.1 必跑消融

对每个 claim 至少运行：

1. 每个单模型；
2. 两两组合；
3. 逐一加入/移除模型的 leave-one-model-out；
4. `audio_only` 与 `audio_plus_hypothesis_text` 对比；
5. 有/无 separation transformation 对比；
6. blind count 与 oracle count 对比。

组合方法按复杂度分三档：

| 组合档位 | 方法 | 作用 |
| --- | --- | --- |
| `operator` | intersection/high-precision、union/high-recall、简单 event alignment | 不依赖学习器，显示证据上下界 |
| `generic_resolver` | 用 dev session 训练小型 logistic/isotonic/GBDT | 测量候选证据的真实增益，test 冻结 |
| `hypothesis_test` | 假设-预测-检验，按信息增益选择下一模型 | 测量冲突场景下的补证价值 |

generic resolver 的输入来自所有候选模型的原始/校准后结构化输出，不能写死“CAM++ 只做 identity”或“Sortformer 只在冲突时调用”。模型职责固定在评测结束后，再把最佳组合压缩成生产链路。

同一 root evidence 的派生 claim 不构成交叉验证。例如同一 timeline 的 `count -> multi`、`overlap mask -> duration -> ratio -> bool`、`segments -> change points` 都只是一个 evidence family。normalized prediction 必须保存 `evidence_family_id`、`upstream_dependency_ids`、`derived_from`；互补性、fusion 和 hypothesis-test 仅按独立 root family 计算证据数。

#### Hypothesis-test 的专项评测

1. 先在其它 session 预注册 conflict trigger，对 held-out sample 冻结 `H1/H2/H_other`、每个分支的 prediction/falsifier、允许的独立后续 evidence 和调用预算。
2. 分支被淘汰只表示解释不成立，不直接 certify 其它分支；最终 claim 仍需要通过正常 event/identity/coverage/calibration 门槛。
3. 同时在 all-sample 和 frozen-conflict subset 报告：`correct_branch_retention`、`wrong_branch_elimination`、`H_other_recall`、`unresolved_rate`、固定 false-certification risk 下的 final claim gain、附加 RTF/显存和每解决一个 conflict 的成本。
4. 下一个工具的选择 policy 只在 train/dev session 学习，test 冻结；不得在 test 上看到分支结果后更换 falsifier。

### 9.2 组合选择门槛

一个新模型只在以下条件同时成立时进入推荐 portfolio：

1. 在 held-out session 上相对当前组合有正向 incremental gain，且 95% CI 不支持明显负增益；
2. 增益在至少一个预注册关键 slice 上成立，不是少数异常 sample 造成；
3. 没有让关键最差 slice、failure rate 或 abstention 恶化到门槛之外；
4. 增加的 RTF、显存、许可和维护成本在预设预算内；
5. dependency/provenance 审计证明它不只是已有证据的重复包装。

### 9.3 组合榜字段

```text
system_id
component_models
claim
dataset / split
standalone_best_metric
fusion_metric
incremental_gain
coverage
joint_false_negative_rate
conflict_rate
abstention_rate
RTF / peak_vram
license_bundle
```

## 10. 榜单形式

不要发布一个把 DER、count MAE、WER 和 cosine 混成单一分数的总榜。建议发布以下榜单：

1. **Count 榜**：exact accuracy、MAE、±1、under/over-count；
2. **Multi-speaker 榜**：F1、MCC、specificity、AUPRC、coverage；
3. **Overlap 榜**：frame/event F1、IoU、duration/ratio MAE；
4. **Change 榜**：必须拆成三条独立 track，不能把不同语义混成一个分数：
   - `public_v0_change` 主榜：严格复现当前公开 `speaker_change` 的 any-change contract；
   - `floor_transfer_v2` 候选榜：只评主导 floor 从 A 转移到 B 的新语义；
   - `different_speaker_onset` 诊断榜：评第二 speaker 的进入点、backchannel 和插话，不能当作 floor transfer；
   三条 track 分别报告 event/point F1、boundary error 和 backchannel false-positive rate。
5. **Full diarization 榜**：DER、JER、MISS/FA/CONFUSION；
6. **Identity 榜**：EER、minDCF、AUC、Brier/ECE；
7. **VAD 榜**：frame F1、boundary error、coverage；
8. **ASR/speaker-text 榜**：WER/CER、timestamp、assignment ambiguity；
9. **Fusion/portfolio 榜**：组合增益、风险覆盖率、成本。

每张正式榜的排名顺序为：主质量指标 -> coverage -> 最差关键 slice -> calibration/risk -> 成本。多 session 正式数据的数值附 session-cluster bootstrap 95% CI；单 session 诊断表显式写 `CI=N/A (one session)` 且不生成 rank。

每一行至少包含：

```text
rank
model/system + revision
track (audio_only / text_assisted / streaming / offline)
dataset / split
n_total / n_evaluable / n_unsupported / n_failed / n_abstained
primary metric + 95% CI
confidence metrics
RTF / peak VRAM
max speakers / overlap capability
license
training provenance / contamination flag
semantics_version / scorer_version / scorer_config_hash
notes
```

如果汇报必须展示“推荐模型”，在每个 claim 榜末增加：

```text
recommended_for_claim = model or tied models
recommended_slices = [...]
reason = metric + coverage + robustness + cost
```

推荐结果必须引用榜单数字，不由当前 pipeline 的既有结构决定。

## 11. 数据集、切分与 AMI 处理

### 11.1 当前 AMI demo

`ami_en2001a_utterances` 有 195 条 utterance-level sample，全部来自同一个 `EN2001a` meeting：

- speaker count 分布：1/2/3/4/5 人分别为 14/92/64/24/1；
- 当前规则下有大量 overlap，按绝对时长和 ratio 阈值统计的正例数量不同；
- 适合检查 adapter、时间对齐、人数/overlap 压力和冲突处理；
- 不适合当作最终泛化榜，也不能按 utterance 随机拆 train/dev/test；
- Sortformer model card 明确包含 AMI 训练数据，AMI 结果必须标记 `known_contaminated`；
- MOSS 的训练数据 provenance 当前标记 `unknown`，不宣称 clean。

污染状态必须按 `(model revision, dataset, split)` 登记，不能只给 dataset 贴一个标签。AMI 其它 meeting 与 EN2001a 分开能防止本次评测内部的 session leakage，但不能消除预训练已使用 AMI 的污染。因此分为：

| 轨道 | 用途 | 能否宣称泛化 |
| --- | --- | --- |
| `AMI_single_meeting_diagnostic` | 工程 smoke、scorer 校验、非排名的描述性错误对照 | 不能 |
| `heldout_session_known_contaminated` | 测量同语料内的 session transfer，仍保留污染标签 | 不能宣称 clean 泛化 |
| `heldout_session_unknown_provenance` | 描述性/辅助选型，不与 clean 模型混排 | 仅能宣称该冻结 split 上的实测结果 |
| `heldout_session_clean` | 正式模型选择候选 | 在满足下述数据量门槛时可以 |

### 11.2 后续数据集

按 license 和训练污染审计，优先补充：

- AMI 其它 meetings、ICSI；
- AliMeeting、AISHELL-4（中文会议/重叠）；
- VoxConverse（自然场景）；
- DIHARD III（多领域）；
- CALLHOME（电话/短 turn）；
- 其它明确标注 speaker intervals、overlap 和 words 的会议语料。

切分单位必须是 meeting/session/recording。校准、dev、test 之间不能共享 recording；CI 也按 recording/session cluster bootstrap，而不是把 utterance 当 IID 样本。

正式排名前必须在 P0 预注册 minimum data gate：每个 test domain 的最少独立 session 数、每个 bool claim 的最少正/负 sample 数、每种 speaker-count bucket 的最少样本和关键 slice 覆盖。未达门槛时只报 point estimate/错误列表，不排名、不做显著性结论。

## 12. 首轮执行计划

### P0：冻结评测协议

- 固定 canonical grid、collar、material overlap、floor-transfer 语义；
- 固定 manifest hash 和 dataset provenance；
- 写出 gold scorer，并用人工抽检验证 count/overlap/change；
- 建立 candidate registry、license registry 和 model revision/hash 表。

### P1：分层 smoke

从 AMI 选 10-20 条，不按 manifest 前 N 条截取，覆盖：

- 1、2、3、4、5 speaker；
- 无 overlap、短 overlap、长 overlap、4%-6% gray zone；
- 无 change、floor transfer、backchannel/插话；
- 短/长 sample、低/高 speech coverage。

先跑当前五路，再加入已通过 smoke 的 Brouhaha VAD；随后加入 pyannote、Silero/WebRTC、ECAPA+clustering 和 NeMo clustering。Brouhaha 只参加 `V`/声学诊断，不参加 `C/M/O/X/I/D`。目的只是验证模型可运行、输出可解析和 capability coverage，不生成正式名次。

### P2：AMI 全量诊断

跑完整 195 条，生成每个模型的 raw artifact、统一 prediction JSONL、分层 point estimate 和第一版 descriptive capability matrix。当前已完成 MOSS/Sortformer/FireRedVAD/CAM++/Whisper 的 speaker evidence 诊断，以及 Brouhaha 的 VAD 对照诊断。该阶段不拆 calibration/dev/test，不运行 `dev_tuned`、不做概率校准、不给 cluster CI/显著性、不生成 rank；所有结果标记污染不对称/单 meeting 限制。

### P3：跨 meeting 正式评测

取得其它会议数据后，按 session 拆 calibration/dev/test，扩展 DiariZen、EEND/TS-VAD、VBx、更多 identity/ASR/VAD 模型。此阶段才生成正式 leaderboard rank。

### P4：能力选型与组合

- 根据单模型榜选每个 claim 的候选 specialist/generalist；
- 用 pairwise gain、conditional recall 和 joint-negative risk 选互补模型；
- 运行组合消融和 hypothesis-test policy；
- 只把经过真实数据验证的职责写回生产部署计划。

### P5：校准与上线门禁

模型职责确定后，再冻结各模型/claim 的阈值和概率校准 profile；在 clean held-out test 上验证 risk-coverage、abstention 和 false-certification rate。校准失败时降低 coverage 或 abstain，不调低门槛。

## 13. 交付物

每轮评测必须生成：

1. `candidate_registry.json`：模型、revision、hash、许可证、训练污染、能力声明；
2. `raw_outputs/<model>/<sample>.json`：模型原始输出；
3. `normalized_predictions.jsonl`：统一时间和字段后的预测；
4. `gold_scoring.jsonl`：推理后生成的 reference 和 gold 状态；
5. `model_capability_matrix.csv`：模型 × 能力 × 数据 slice；
6. `pairwise_complementarity.csv`：错误重合、条件召回和增量收益；
7. `leaderboard_<track>.md/html/pdf`：满足多 session 门槛时的分榜和 95% CI；单 session 输出命名为 `diagnostic_table_*`，不生成 rank；
8. `fusion_ablation.jsonl`：组合消融、成本和风险覆盖率；
9. `run_manifest.json`：命令、环境、GPU、代码快照和所有 artifact hash。

当前文档是评测协议，不代表已经跑完上述模型或已经产生名次。正式汇报时，榜单中的空白应明确写 `pending`，模型未支持的能力写 `N/A`，不能用 0 填充。

当前 AMI 全量诊断已完成，但仍不构成正式榜单：speaker evidence 结果见 [`ami_single_meeting_diagnostic_report_20260812.md`](ami_single_meeting_diagnostic_report_20260812.md)，Brouhaha VAD 对照见 [`brouhaha_vad_diagnostic_report_20260812.md`](brouhaha_vad_diagnostic_report_20260812.md)。对应 artifact 分别位于 `ami_en2001a_utterances/outputs/ami_single_meeting_diagnostic_20260812/` 和 `ami_en2001a_utterances/outputs/brouhaha_vad_full_diagnostic_20260812/`。两个报告都并列 scorer 口径和单 meeting 限制，不能混排成一个总榜。

## 14. 当前 demo 的说明性结果

在 `EN2001a_utterance_00000` 上，native reference 为 3 个 speaker、`multi_speaker=true`、存在 overlap。已有 shadow 结果显示：

| 模型 | count (`metadata_v0_active_count`) | multi (`metadata_v0_multi`) | overlap (`metadata_v0_overlap`) | change (`public_v0_change`) |
| --- | ---: | --- | --- | --- |
| MOSS | 2 | true | false | true（shadow candidate；scorer=`public_v0`） |
| Sortformer v2 | 3 | true | true | true（shadow candidate；scorer=`public_v0`） |

其中 count 严格对应 `metadata_v0_active_count`，multi 是同一 scorer 的 `metadata_v0_multi` 派生值，overlap 对应 `metadata_v0_overlap`；change 仅表示当前 shadow scorer 的 `public_v0_change` 候选，尚未经过正式 scorer registry、校准或跨 session 评测。`floor_transfer_v2` 和 `different_speaker_onset` 不在这张 demo 表中，不能从 `candidate true` 推断它们为真。

这只能说明两条模型输出存在可分析的差异，不能据一条 sample 宣布 Sortformer 是“人数模型”或 MOSS 是“换人模型”。只有在完整分层矩阵和跨 meeting test 上重复观察到优势，才能把某项能力写成模型职责。

### 3.8 SURE 链路覆盖映射（核对结果）

对照 3 节 8 项能力检查 `sure-harness` 打分链路的覆盖情况。核对基准：`sure/external/sure-evaluation`（基础引擎，任务 = `asr, classification, kws, s2tt, sa_asr, sd, se, slu, tse, tts, vc`）与 `sure/external/sure-evaluation-vad`（VAD 引擎，commit `87b6bc4`，额外任务 `vad`），以及 harness 登记白名单（`asr, s2tt, sd, ser, tts, vc, kws, slu, gr, speech_understanding, sa-asr, sa_asr`）。

覆盖分两层看：**官方指标**（SURE route 直接输出的分数）与**链路 artifact 可补算**（SD 评测保留的 `corpus_ref.rttm` / `corpus_hyp.rttm` 及报告 details 能否支撑伴随 scorer 重算）。

| 能力 | 本计划要求（第 3 节口径） | SURE 官方指标 | 链路 artifact 可补算 |
| --- | --- | --- | --- |
| `C` speaker count | 1/2/3/4/5 人分层 count | ✅ count_mae / count_accuracy（`sd.any.*.sd_structure_v1`，v0 预处理，details 含 1–5 人分层） | ✅ |
| `M` multi-speaker | bool（≥2 有效 speaker） | ✅ multi_accuracy（`sd.any.multi_accuracy.sd_structure_v1`） | ✅ |
| `O` overlap | bool + overlap_ratio + 区间/事件 | ✅ overlap_accuracy / overlap_ratio_mae / overlap_frame_f1 / overlap_event_f1（`sd.any.*.sd_structure_v1`） | ✅ |
| `X` speaker change | bool/count/points，`public_v0_change` 与 `floor_transfer_v2` 分轨 | ✅ public_v0：change_bool_accuracy / change_count_mae / change_point_f1_025 / change_point_f1_05（`sd.any.*.sd_structure_v1`，冻结 v0 规则 + collar 匹配） | ⚠️ floor_transfer_v2 语义未冻结，暂缓 |
| `I` identity | 同/不同 speaker pair 判别 | 无 SV 任务 | ❌ RTTM 匿名标签没有声纹 similarity，pair 判别必须走 embedding 模型 |
| `V` speech coverage | 20 ms 帧 speech/silence + onset/offset 边界误差 | ✅ `vad`（f1/p_fa/p_miss/dcf_nist/auc_roc）+ onset_mae / offset_mae / boundary_f1（`vad.any.*.vad_boundary_v1`） | ✅ |
| `A` lexical clock | WER/CER、word coverage、timestamp error、SA-WER/cpWER | ✅ `asr`（zh CER / en WER / cs MER）+ word_coverage（`asr.{zh,en}.word_coverage.*.token_coverage_v1`）+ `sa_asr` cpWER | ⚠️ 缺 timestamp error（需词级时间戳数据集） |
| `D` full diarization | speaker-attributed timeline 整体质量 | `sd`（DER，meeteval，collar 0.25） | ✅ 报告 details 有 per-session missed/falarm/speaker_error 时间分解 |

为什么 DER 标量本身不能替代 `C/M/O`：

- DER 是时间加权连续标量 `(missed + false alarm + speaker error) / reference speech`，没有 count 误差项；它和 sample-level 的离散指标不在同一聚合层面。反例：模型多预测一个只活跃 1 s 的假 speaker，DER 几乎不动，但该 sample 的 `C/M` 判错；反之模型把 5 人 merge 成 2 人会让 speaker error 时间变大，DER 只能说明"错了很多时间"，无法报告 count 判对率，更给不了 1/2/3/4/5 分层。
- overlap 错误会**间接**反映在 DER 里（参考 overlap 帧漏掉某 speaker 计入 miss/speaker error），但 DER 不输出 overlap ratio、is_overlapped 或事件匹配；且从 DER 数值无法区分边界错与 overlap 漏检。计划书要求逐事件报告，DER 给不了。

`C/M/O` 已按 SURE 的新指标规范实现并出分：`sure-harness/sure/external/sure-evaluation` 新增节点 `scoring/sd_structure` v1，`sd` 任务增加 7 条 route（DER 默认路由不变），配套 `docs/tasks/sd.md`、`tests/test_sd_structure_scoring.py` 与重新生成的 `docs/pipeline_catalog.jsonl`。实现复现 3.2/3.3 的 v0 预处理（0.10 s 段过滤、0.30 s 同 speaker 合并、0.10 s overlap 事件、ratio 阈值 0.05、20 ms 帧网格、事件 one-to-one IoU 0.5），报告 details 含 per-session 明细与 1–5 人分层。已在 `runs/sd/*/score/` 的 `corpus_ref.rttm`/`corpus_hyp.rttm` 上出分（见 `model_evaluate/evaluation_matrix.md`）。`C/M/O` 只依赖时间轴与 distinct label，不需要 Hungarian speaker mapping。限制是：

1. SURE 的 ref 是 AMI 原始参考 RTTM，不等于计划书的 native metadata；3.2/3.3 的 v0 预处理（0.10 s 段过滤、0.30 s 同 speaker 合并、0.10 s overlap 事件、ratio ≥ 0.05）必须在伴随 scorer 中复现后才能对口径。
2. 非 overlap 模型（聚类式 diarizer 等）输出恒为非重叠时间轴，`O` 得 0 分是正确结果而不是 N/A，需与"模型不支持该任务"区分。

结论：SURE 官方指标现在直接覆盖 `C/M/O`（sd_structure 新 route）、`X` 的 public_v0 轨道（change 四指标）、`A`（CER/WER/MER/cpWER + word_coverage）、`V`（独立 VAD 引擎五指标 + 边界三指标）、`D`（整体 DER + 时间分解）；`X` 的 floor_transfer_v2 语义冻结前暂缓，`I` 必须走 embedding/声纹链，`A` 的 timestamp error 需词级时间戳数据集。`C/M/O/X` 分数当前基于 AMI 参考 RTTM，对齐 native metadata 口径仍需按 3.2–3.4 在伴随 scorer 复核，不能直接把 SURE 的 AMI 原始参考当 native metadata。

补全 SURE 侧的建议（非本计划阻塞项）：向 `sure-evaluation` 提交任务提案新增 `sv`（identity pair）；`A` 的 timestamp error 需先建词级时间戳参考集；正式接入 `vad` 需要 harness 升级到含 `vad` 的引擎版本并把 `vad` 加入任务白名单。

## 4. 候选模型池

候选模型不限定为当前已部署的五路。先按能力类别建立池，再根据许可证、checkpoint 可得性和 smoke 结果决定是否进入全量运行。

### 4.1 Full timeline / diarization 候选

| 候选 | 主要输出 | 进入计划 | 注意事项 |
| --- | --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | joint ASR + anonymous speaker timeline | Round 1 | overlap 可能受单流生成限制；训练数据 provenance 待核 |
| NVIDIA Streaming Sortformer 4spk-v2 | overlap-aware frame activity、timeline、count candidate | Round 1 | 最多 4 slots；AMI 明确存在训练污染 |
| pyannote Community-1 | segmentation、OSD、embedding、count、overlap、assignment | Round 1（token 后） | gated 权重；固定 revision 和 telemetry 设置 |
| NeMo clustering diarizer（MarbleNet + TitaNet + clustering） | VAD + embedding + non-overlap timeline/count | Round 1 baseline | 默认不建模 overlap；不能把 overlap N/A 当 false |
| SpeechBrain ECAPA + 固定 VAD + AHC/spectral clustering | modular count/turn/continuity timeline | Round 1 baseline | 透明异构 baseline；checkpoint license 逐 revision 核查 |
| WeSpeaker / 3D-Speaker（ECAPA、CAM++、ERes2Net/TitaNet） | embedding + clustering/count | Round 2 | 同一 checkpoint 的不同 wrapper 不算独立 fusion 证据 |
| VBx / x-vector + Bayesian HMM | classical non-overlap timeline | Round 2 | 用于判断复杂模型是否真正带来增益 |
| DiariZen | end-to-end diarization | Round 2 audit | 先核实可复现 checkpoint、语言和 license |
| EEND-EDA / EEND-VC / TS-VAD | overlap-capable diarization | Round 2 audit | 部分配置需要 speaker count/enrollment；blind 与 oracle 分开 |

### 4.2 Overlap/OSD 专项候选

- Sortformer v2 的 frame activity；
- pyannote OSD/Community-1 内部 overlap 输出；
- EEND powerset/EDA；
- 可复现的 WavLM-OSD、CAT-Net 或其它公开 OSD checkpoint（待 license、权重和推理脚本审计）；
- 物理多通道 activity 只作为 layout-specific baseline，不作为通用 mono 模型。

OSD 模型只参加 `O` 榜；如果不能输出 speaker attribution，不能直接参加 `C` 或 `X` 榜。

### 4.3 Identity 专项候选

- 当前 CAM++ 中文 checkpoint；
- SpeechBrain ECAPA-TDNN；
- WeSpeaker/3D-Speaker 的 ECAPA、ERes2Net、TitaNet 或英文 checkpoint；
- x-vector/ResNet speaker verification baseline。

身份模型用同一套 native clean pairs 评测。CAM++ 在 AMI English 上是否适合，必须由该榜的英文分层结果决定，不能从中文 checkpoint 名称或 upstream threshold 推断。

### 4.4 VAD、ASR 和音频变换候选

| 类别 | 候选 |
| --- | --- |
| VAD | FireRedVAD、Brouhaha VAD、Silero VAD、WebRTC VAD、MarbleNet VAD |
| ASR/lexical clock | Whisper Base/Small/Large-v3、Paraformer-zh、SenseVoiceSmall、FunASR Conformer/Zipformer |
| separation transformation | SepFormer、Conv-TasNet、MossFormer2、TF-GridNet |

separation 不是 speaker evidence source。它只能作为变换链参加增量实验：`原音频 -> separation -> VAD/embedding/diarizer`，并与原音频 baseline 比较是否真实改善 `O`、`I` 或 `D`。

### 4.5 候选纳入门槛

模型进入全量榜必须满足：

1. 权重和代码可离线固定，能记录 revision/hash；
2. 许可证允许当前评测和后续部署，或明确标记 `audit_only`；
3. 能在 10 条分层 smoke 上完成至少 90% invocation；
4. 输出可转换到统一协议，或明确标记具体能力为 `N/A`；
5. 不能使用 native metadata、gold count 或邻接音频。

未满足条件的模型保留在 `candidate_pool`，不进入正式 rank，但报告阻塞原因。

### 4.6 候选扩展顺序

“不限于现有模型”不等于一次性安装所有项目。按对当前模型池的新信息量和可比性分批进入：

| 优先级 | 候选 | 必要性 | 首要对比 |
| --- | --- | --- | --- |
| P0 | MOSS、Sortformer v2、CAM++、Whisper Base、FireRedVAD | 已部署，先建立端到端评测基线 | 所有各自适用的 `C/M/O/X/D/I/V/A` |
| P0 diagnostic | Brouhaha VAD | 本地权重和环境已可离线运行；先作为 `V/SNR/C50` 辅助诊断，不进入 speaker claim 主榜 | `V` standalone、FireRedVAD 互补性、声学难度分层 |
| P1 | pyannote Community-1、NeMo clustering diarizer、ECAPA+clustering | 同时引入 powerset、modular neural clustering 和异构 embedding baseline | `C/M/X/D`，pyannote 另测 `O` |
| P1 | SpeechBrain ECAPA、ERes2Net/TitaNet 中至少一个 | 避免 identity 榜只有中文 CAM++ | `I`，重点看 AMI English、短 crop 和 overlap-adjacent slice |
| P1 | Silero/WebRTC/MarbleNet 中至少两个 | VAD 对成本和 domain shift 敏感，单一 FireRedVAD 无法形成榜单 | `V`，以及更换 VAD 对 clustering diarizer 的增量影响 |
| P2 | VBx、EEND-EDA/EEND-VC、DiariZen、独立 OSD | 在基线结果暴露 count/overlap/change 缺口后定向扩展 | 相应 claim 的 conditional recall 和 cost per additional correct sample |
| P2 | Whisper Small/Large-v3、Paraformer、SenseVoice/Zipformer | 区分 ASR 模型规模、语言和 timestamp 机制的影响 | `A` 与 text-assisted 增量，不参加纯声学 speaker 事件投票 |
| P3 | separation 组合 | 成本高且是 transformation，只在已知困难 slice 上有必要 | paired ablation，不独立排名为 speaker model |

每批结束后根据错误分布决定下一批：如果新模型与当前最佳模型错误几乎完全重合，即使 standalone 指标接近，也不优先纳入生产 portfolio。

## 5. 模型能力摸底实验

### 5.1 每个模型都跑所有适用任务

榜单的最小比较单位是 `checkpoint + frozen adapter/pipeline + profile`，不是一个模型品牌名。在查看 test gold 前，为每个候选登记“原子输出如何确定性映射为某个 task prediction”；有冻结 adapter 就参评，无法形成该任务输出才记 `N/A`。

不是先给模型分配角色，而是先收集它能产生的全部证据。例如：

- MOSS 同时评 `C/M/O/X/D/A`；
- Sortformer 同时评 `C/M/O/X/D`，不因没有 ASR 就给 `A` 分数；
- ECAPA+clustering 评 `C/M/X/D`，如果该冻结 pipeline 不产生并发 activity，`O` 标 `N/A`；
- CAM++ atomic profile 评 `I`；`CAM++ + 固定 VAD + 固定 clustering` 是另一个 composed candidate，可评 `C/M/X/D`，但 pair score 不能直接参加 `O`；
- pyannote full diarization profile 可评 `C/M/O/X/D`；若能从冻结的 cluster assignment 产生 sample-local same/different pair prediction，该 composed profile 也参加 `I`；
- Whisper 评 `A` 和 lexical/boundary diagnostics，不能凭单流文本判断没有 overlap；
- VAD 评 `V`，不能推断 speaker count。

每个模型的结果行应至少包含：

```text
model_id
revision / checkpoint_sha256
adapter_id / adapter_version / config_hash
input_mode
inference_profile (blind / oracle_count / streaming / offline)
capabilities_declared
capabilities_observed
evidence_family_id / upstream_dependency_ids / derived_from
raw_outputs_path
n_total / n_evaluable / n_unsupported / n_failed / n_abstained
primary_metrics
confidence_metrics
runtime / RTF / peak_vram
license / training_provenance / contamination_flag
```

### 5.2 运行 profile

同一模型至少产生以下可区分 profile：

| profile | 是否进主榜 | 目的 |
| --- | --- | --- |
| `blind_audio_only` | 是 | 测量真实端到端能力 |
| `dev_tuned_audio_only` | 是，单独一行 | 仅用 dev 调整阈值/聚类参数后的能力 |
| `oracle_num_speakers` | 否 | 判断错误来自人数估计还是声学分离 |
| `audio_plus_hypothesis_text` | 辅助榜 | 测量 ASR 文本增益 |
| `oracle_text` | 否 | 上限分析，不能作为部署结果 |

### 5.3 不预先固定模型角色

在第一轮报告之前，不写“某模型是 event vote、某模型是 identity guard”。这些是评测后的结论。评测报告中同时保存：

- 模型宣称的能力；
- 实际测出的能力；
- 在哪些 slice 上能力成立；
- 哪些错误是独有的，哪些与其它模型重合；
- 是否值得作为独立证据或只适合作诊断。

## 6. 指标体系

所有指标按 dataset、language、layout、duration、reference count、overlap ratio、SNR 和样本难度分层。正式数据同时报告 frame-micro、sample-macro 和 session-macro 聚合，主排序使用 session-macro，防止长 sample 或长 meeting 主导结果。只有存在足够独立 session 时才给出 session-cluster bootstrap 95% CI；单 session 数据只报 descriptive point estimate 和错误清单。

### 6.1 Count (`C`)

- exact accuracy；
- MAE、RMSE、bias；
- within ±1 accuracy；
- under-count rate、over-count rate；
- 1/2/3/4/5 speaker confusion matrix；
- count distribution NLL/Brier（模型有完整 count probability 时）；
- unsupported/overflow rate；
- blind 与 oracle-count 差值。

Count 榜的主排序为 exact accuracy 和 MAE；不能让一个只输出 lower bound 的模型与 exact-count 模型混成同一分数。lower-bound recall、upper-bound violation 另列。

### 6.2 Multi-speaker (`M`)

- precision、recall、F1；
- balanced accuracy、MCC、specificity；
- AUROC/AUPRC（有连续 sample score 时）；
- 按真实人数分层的 recall；
- abstention/coverage、certified-only precision；
- negative specificity，避免 AMI 正例比例过高导致 accuracy 虚高。

### 6.3 Overlap (`O`)

- canonical frame precision/recall/F1；
- `metadata_v0_overlap` sample-level precision、recall、F1、MCC 和 specificity；
- overlap event one-to-one F1，event IoU；
- onset/offset error；
- overlap duration MAE/bias；
- overlap ratio MAE 和 reliability；
- `metadata_v0_overlap`、`raw_frame_overlap` 与 `gray_zone_excluded` 分轨结果。

没有 overlap 输出的模型在此处写 `N/A`，不能写 0，也不能用“没有检测到 overlap”充当负例。

### 6.4 Change (`X`)

- 现行 `public_v0_change` bool/count/point precision、recall、F1；
- floor-transfer point/event precision、recall、F1；
- collar 0.25 s 和 0.5 s 两套结果；
- boundary MAE；
- change event count MAE；
- floor-transfer 与 `different_speaker_onset/backchannel` 分开报告；
- 静音边界、短 backchannel、overlap 边界的 false positive rate。

`public_v0_change` 和 `floor_transfer_v2` 必须分别排名。汇报表的 track/semantics version 不得省略。

### 6.5 Full diarization (`D`)

- DER（collar 0、0.25 s）；
- JER；
- MISS、FA、CONFUSION 分解；
- overlap include/exclude 两套；
- oracle VAD 与 pipeline VAD 两套；
- frame speaker attribution F1/coverage；
- RTTM speaker permutation 使用 Hungarian mapping。

### 6.6 Identity (`I`)

- ROC-AUC、AUPRC；
- EER、minDCF；
- TPR@FAR=1%/5%；
- Brier、ECE、NLL；
- 按 crop 时长、语言、同/不同 channel、overlap 邻近程度分层；
- oracle clean crops 与 predicted clean crops 分开。

### 6.7 VAD (`V`) 与 ASR (`A`)

VAD：frame F1、miss/FA、speech coverage、onset/offset error、短 speech recall。Brouhaha 与 FireRedVAD 的 native raw frame score、阈值和后处理必须分别登记；当前 AMI 诊断只报告 `raw_native_speech_union_20ms`，不能把 Brouhaha 的 `confidence=1.0` 当成概率。Brouhaha 的 `SNR/C50` 只用于 acoustic-difficulty slice 和辅助 metadata 诊断，若没有独立 gold 则不参加质量数值排名。详见 [`brouhaha_vad_diagnostic_report_20260812.md`](brouhaha_vad_diagnostic_report_20260812.md)。

ASR：WER/CER、word coverage、word timestamp MAE、segment boundary error；有 gold speaker timeline 时增加 SA-WER/cpWER、word-speaker assignment accuracy 和 ambiguous/unassigned rate。

ASR 的文本一致性只能作为 lexical/ambiguity diagnostic，不能因为某个 ASR 没转出第二路文本就判定 `speaker_overlap=false`。

## 7. 置信度的公平测量

置信度测量不是先给当前 resolver 的 claim 配参数，而是对候选模型的原始输出做统一审计。

### 7.1 先测区分能力，再测校准

顺序固定为：

1. 用 audio-only blind profile 测 standalone quality；
2. 比较 raw score 的排序能力（AUROC/AUPRC、PR curve、EER 等）；
3. 再在 session-separated calibration split 上，用同一套预注册方法（Platt、isotonic 或 temperature，按输出类型选择）把每个模型映射到概率；
4. 在冻结的 test split 报 Brier、NLL、ECE、reliability diagram 和 risk-coverage。

这里校准的是“模型对 gold 任务的预测可靠性”，不是校准当前架构已经决定的 claim ownership。每一个候选模型都接受相同的流程；如果新模型在 overlap 上比现有模型更好，榜单应直接显示这一结果。

### 7.2 不同原始分数不直接比较

- sigmoid activity probability、cosine similarity、word probability、log probability 和固定 adapter confidence 不在同一量纲；
- 不做跨模型原始分数加权平均；
- MOSS 没有可用 native score 时，不能伪造 0-1 概率，confidence 栏写 `N/A`，只比较 hard prediction quality；
- 没有连续 score 的模型可另报 test-time perturbation consistency/abstention，但不能把一致性冒充概率校准。

### 7.3 报告模型可靠性而不是“谁的数字更大”

榜单中的置信度列至少包括：

```text
native_score_type
score_range
score_available_rate
AUROC / AUPRC（如适用）
Brier / NLL / ECE（校准后）
coverage at target risk
abstention rate
```

## 8. 从单模型结果决定模型职责

### 8.1 单模型能力矩阵

输出一张 `model_capability_matrix`，每个单元格不是主观描述，而是实测结果：

| model | C count | M multi | O overlap | X change | D DER/JER | I identity | V VAD | A ASR | best slices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOSS joint profile | pending | pending | pending | pending | pending | pending/N/A 由 adapter registry 冻结 | pending/N/A | pending | pending |
| Sortformer v2 timeline profile | pending | pending | pending | pending | pending | pending/N/A | pending/N/A | pending/N/A | pending |
| pyannote Community-1 full profile | pending | pending | pending | pending | pending | pending/N/A | pending/N/A | pending/N/A | pending |
| ECAPA + fixed VAD + clustering | pending | pending | pending/N/A | pending | pending | pending | 依赖冻结 VAD | pending/N/A | pending |
| CAM++ atomic identity profile | N/A | N/A | N/A | N/A | N/A | pending | N/A | N/A | pending |
| CAM++ + fixed VAD + clustering | pending | pending | N/A | pending | pending | pending | 依赖冻结 VAD | N/A | pending |
| Brouhaha VAD v0.9.0 atomic | N/A | N/A | N/A | N/A | N/A | N/A | **diagnostic measured; formal rank pending** | N/A | pending |

### 8.2 “适合”如何判定

不定义一个隐藏总分。对每个 claim 单独按照以下顺序判断：

1. 主指标（例如 overlap frame/event F1，count MAE，change F1）；
2. `n_evaluable` 和 unsupported/failed/abstain coverage；
3. 最差关键 slice，而不是只看总体平均；
4. 置信度和目标风险下 coverage；
5. RTF、峰值显存、许可证和训练污染。

两个候选的比较基于 session-paired metric difference CI，而不是“各自 95% CI 是否重叠”。只有差值落在预注册的 equivalence margin 内时才标记 tie；若样本不足以判断，标记“未决”。模型职责由结果产生：

- **claim specialist**：在一个 claim 及关键 slices 稳定领先；
- **generalist**：多个 claim 均位于 Pareto 前沿；
- **complementary witness**：单模型不一定第一，但在另一模型失败的样本上有高 conditional recall；
- **diagnostic-only**：能暴露风险，但不能独立支持该 claim；
- **not suitable**：能力缺失、覆盖不足或错误率不可接受。

### 8.3 互补性和增量收益

对每个模型 `m` 和 claim `c`，除 standalone 指标外计算：

```text
conditional_recall(m | baseline fails, c)
unique_correct_rate(m, c)
error_overlap(m, baseline, c)
joint_false_negative_rate(m + baseline, c)
incremental_gain(m added to baseline, c)
cost_per_additional_correct_sample
```

例如 overlap 模型的选择不只看其自身 F1，还要看：当当前最佳模型漏掉短 overlap 时，它能否补回；如果两者总在同一批样本上同时失败，则不构成有价值的独立证据。

互补性统计必须使用 out-of-fold 预测：先在其它 session 上选阈值/训练 resolver，再计算当前 held-out session 的 `unique_correct` 和 incremental gain。不得在同一批 test 样本上穷举模型对、选出最佳组合后仍报原 test 分数；这会产生 portfolio selection bias。对两组模型差异使用 session-cluster paired bootstrap/permutation test，并对同一 claim 中的多个候选比较做 FDR 控制。

## 9. Fusion 和组合榜

Fusion 必须在单模型榜之后单独评测，不能用组合结果反推某个单模型“本来就擅长”。

### 9.1 必跑消融

对每个 claim 至少运行：

1. 每个单模型；
2. 两两组合；
3. 逐一加入/移除模型的 leave-one-model-out；
4. `audio_only` 与 `audio_plus_hypothesis_text` 对比；
5. 有/无 separation transformation 对比；
6. blind count 与 oracle count 对比。

组合方法按复杂度分三档：

| 组合档位 | 方法 | 作用 |
| --- | --- | --- |
| `operator` | intersection/high-precision、union/high-recall、简单 event alignment | 不依赖学习器，显示证据上下界 |
| `generic_resolver` | 用 dev session 训练小型 logistic/isotonic/GBDT | 测量候选证据的真实增益，test 冻结 |
| `hypothesis_test` | 假设-预测-检验，按信息增益选择下一模型 | 测量冲突场景下的补证价值 |

generic resolver 的输入来自所有候选模型的原始/校准后结构化输出，不能写死“CAM++ 只做 identity”或“Sortformer 只在冲突时调用”。模型职责固定在评测结束后，再把最佳组合压缩成生产链路。

同一 root evidence 的派生 claim 不构成交叉验证。例如同一 timeline 的 `count -> multi`、`overlap mask -> duration -> ratio -> bool`、`segments -> change points` 都只是一个 evidence family。normalized prediction 必须保存 `evidence_family_id`、`upstream_dependency_ids`、`derived_from`；互补性、fusion 和 hypothesis-test 仅按独立 root family 计算证据数。

#### Hypothesis-test 的专项评测

1. 先在其它 session 预注册 conflict trigger，对 held-out sample 冻结 `H1/H2/H_other`、每个分支的 prediction/falsifier、允许的独立后续 evidence 和调用预算。
2. 分支被淘汰只表示解释不成立，不直接 certify 其它分支；最终 claim 仍需要通过正常 event/identity/coverage/calibration 门槛。
3. 同时在 all-sample 和 frozen-conflict subset 报告：`correct_branch_retention`、`wrong_branch_elimination`、`H_other_recall`、`unresolved_rate`、固定 false-certification risk 下的 final claim gain、附加 RTF/显存和每解决一个 conflict 的成本。
4. 下一个工具的选择 policy 只在 train/dev session 学习，test 冻结；不得在 test 上看到分支结果后更换 falsifier。

### 9.2 组合选择门槛

一个新模型只在以下条件同时成立时进入推荐 portfolio：

1. 在 held-out session 上相对当前组合有正向 incremental gain，且 95% CI 不支持明显负增益；
2. 增益在至少一个预注册关键 slice 上成立，不是少数异常 sample 造成；
3. 没有让关键最差 slice、failure rate 或 abstention 恶化到门槛之外；
4. 增加的 RTF、显存、许可和维护成本在预设预算内；
5. dependency/provenance 审计证明它不只是已有证据的重复包装。

### 9.3 组合榜字段

```text
system_id
component_models
claim
dataset / split
standalone_best_metric
fusion_metric
incremental_gain
coverage
joint_false_negative_rate
conflict_rate
abstention_rate
RTF / peak_vram
license_bundle
```

## 10. 榜单形式

不要发布一个把 DER、count MAE、WER 和 cosine 混成单一分数的总榜。建议发布以下榜单：

1. **Count 榜**：exact accuracy、MAE、±1、under/over-count；
2. **Multi-speaker 榜**：F1、MCC、specificity、AUPRC、coverage；
3. **Overlap 榜**：frame/event F1、IoU、duration/ratio MAE；
4. **Change 榜**：必须拆成三条独立 track，不能把不同语义混成一个分数：
   - `public_v0_change` 主榜：严格复现当前公开 `speaker_change` 的 any-change contract；
   - `floor_transfer_v2` 候选榜：只评主导 floor 从 A 转移到 B 的新语义；
   - `different_speaker_onset` 诊断榜：评第二 speaker 的进入点、backchannel 和插话，不能当作 floor transfer；
   三条 track 分别报告 event/point F1、boundary error 和 backchannel false-positive rate。
5. **Full diarization 榜**：DER、JER、MISS/FA/CONFUSION；
6. **Identity 榜**：EER、minDCF、AUC、Brier/ECE；
7. **VAD 榜**：frame F1、boundary error、coverage；
8. **ASR/speaker-text 榜**：WER/CER、timestamp、assignment ambiguity；
9. **Fusion/portfolio 榜**：组合增益、风险覆盖率、成本。

每张正式榜的排名顺序为：主质量指标 -> coverage -> 最差关键 slice -> calibration/risk -> 成本。多 session 正式数据的数值附 session-cluster bootstrap 95% CI；单 session 诊断表显式写 `CI=N/A (one session)` 且不生成 rank。

每一行至少包含：

```text
rank
model/system + revision
track (audio_only / text_assisted / streaming / offline)
dataset / split
n_total / n_evaluable / n_unsupported / n_failed / n_abstained
primary metric + 95% CI
confidence metrics
RTF / peak VRAM
max speakers / overlap capability
license
training provenance / contamination flag
semantics_version / scorer_version / scorer_config_hash
notes
```

如果汇报必须展示“推荐模型”，在每个 claim 榜末增加：

```text
recommended_for_claim = model or tied models
recommended_slices = [...]
reason = metric + coverage + robustness + cost
```

推荐结果必须引用榜单数字，不由当前 pipeline 的既有结构决定。

## 11. 数据集、切分与 AMI 处理

### 11.1 当前 AMI demo

`ami_en2001a_utterances` 有 195 条 utterance-level sample，全部来自同一个 `EN2001a` meeting：

- speaker count 分布：1/2/3/4/5 人分别为 14/92/64/24/1；
- 当前规则下有大量 overlap，按绝对时长和 ratio 阈值统计的正例数量不同；
- 适合检查 adapter、时间对齐、人数/overlap 压力和冲突处理；
- 不适合当作最终泛化榜，也不能按 utterance 随机拆 train/dev/test；
- Sortformer model card 明确包含 AMI 训练数据，AMI 结果必须标记 `known_contaminated`；
- MOSS 的训练数据 provenance 当前标记 `unknown`，不宣称 clean。

污染状态必须按 `(model revision, dataset, split)` 登记，不能只给 dataset 贴一个标签。AMI 其它 meeting 与 EN2001a 分开能防止本次评测内部的 session leakage，但不能消除预训练已使用 AMI 的污染。因此分为：

| 轨道 | 用途 | 能否宣称泛化 |
| --- | --- | --- |
| `AMI_single_meeting_diagnostic` | 工程 smoke、scorer 校验、非排名的描述性错误对照 | 不能 |
| `heldout_session_known_contaminated` | 测量同语料内的 session transfer，仍保留污染标签 | 不能宣称 clean 泛化 |
| `heldout_session_unknown_provenance` | 描述性/辅助选型，不与 clean 模型混排 | 仅能宣称该冻结 split 上的实测结果 |
| `heldout_session_clean` | 正式模型选择候选 | 在满足下述数据量门槛时可以 |

### 11.2 后续数据集

按 license 和训练污染审计，优先补充：

- AMI 其它 meetings、ICSI；
- AliMeeting、AISHELL-4（中文会议/重叠）；
- VoxConverse（自然场景）；
- DIHARD III（多领域）；
- CALLHOME（电话/短 turn）；
- 其它明确标注 speaker intervals、overlap 和 words 的会议语料。

切分单位必须是 meeting/session/recording。校准、dev、test 之间不能共享 recording；CI 也按 recording/session cluster bootstrap，而不是把 utterance 当 IID 样本。

正式排名前必须在 P0 预注册 minimum data gate：每个 test domain 的最少独立 session 数、每个 bool claim 的最少正/负 sample 数、每种 speaker-count bucket 的最少样本和关键 slice 覆盖。未达门槛时只报 point estimate/错误列表，不排名、不做显著性结论。

## 12. 首轮执行计划

### P0：冻结评测协议

- 固定 canonical grid、collar、material overlap、floor-transfer 语义；
- 固定 manifest hash 和 dataset provenance；
- 写出 gold scorer，并用人工抽检验证 count/overlap/change；
- 建立 candidate registry、license registry 和 model revision/hash 表。

### P1：分层 smoke

从 AMI 选 10-20 条，不按 manifest 前 N 条截取，覆盖：

- 1、2、3、4、5 speaker；
- 无 overlap、短 overlap、长 overlap、4%-6% gray zone；
- 无 change、floor transfer、backchannel/插话；
- 短/长 sample、低/高 speech coverage。

先跑当前五路，再加入已通过 smoke 的 Brouhaha VAD；随后加入 pyannote、Silero/WebRTC、ECAPA+clustering 和 NeMo clustering。Brouhaha 只参加 `V`/声学诊断，不参加 `C/M/O/X/I/D`。目的只是验证模型可运行、输出可解析和 capability coverage，不生成正式名次。

### P2：AMI 全量诊断

跑完整 195 条，生成每个模型的 raw artifact、统一 prediction JSONL、分层 point estimate 和第一版 descriptive capability matrix。当前已完成 MOSS/Sortformer/FireRedVAD/CAM++/Whisper 的 speaker evidence 诊断，以及 Brouhaha 的 VAD 对照诊断。该阶段不拆 calibration/dev/test，不运行 `dev_tuned`、不做概率校准、不给 cluster CI/显著性、不生成 rank；所有结果标记污染不对称/单 meeting 限制。

### P3：跨 meeting 正式评测

取得其它会议数据后，按 session 拆 calibration/dev/test，扩展 DiariZen、EEND/TS-VAD、VBx、更多 identity/ASR/VAD 模型。此阶段才生成正式 leaderboard rank。

### P4：能力选型与组合

- 根据单模型榜选每个 claim 的候选 specialist/generalist；
- 用 pairwise gain、conditional recall 和 joint-negative risk 选互补模型；
- 运行组合消融和 hypothesis-test policy；
- 只把经过真实数据验证的职责写回生产部署计划。

### P5：校准与上线门禁

模型职责确定后，再冻结各模型/claim 的阈值和概率校准 profile；在 clean held-out test 上验证 risk-coverage、abstention 和 false-certification rate。校准失败时降低 coverage 或 abstain，不调低门槛。

## 13. 交付物

每轮评测必须生成：

1. `candidate_registry.json`：模型、revision、hash、许可证、训练污染、能力声明；
2. `raw_outputs/<model>/<sample>.json`：模型原始输出；
3. `normalized_predictions.jsonl`：统一时间和字段后的预测；
4. `gold_scoring.jsonl`：推理后生成的 reference 和 gold 状态；
5. `model_capability_matrix.csv`：模型 × 能力 × 数据 slice；
6. `pairwise_complementarity.csv`：错误重合、条件召回和增量收益；
7. `leaderboard_<track>.md/html/pdf`：满足多 session 门槛时的分榜和 95% CI；单 session 输出命名为 `diagnostic_table_*`，不生成 rank；
8. `fusion_ablation.jsonl`：组合消融、成本和风险覆盖率；
9. `run_manifest.json`：命令、环境、GPU、代码快照和所有 artifact hash。

当前文档是评测协议，不代表已经跑完上述模型或已经产生名次。正式汇报时，榜单中的空白应明确写 `pending`，模型未支持的能力写 `N/A`，不能用 0 填充。

当前 AMI 全量诊断已完成，但仍不构成正式榜单：speaker evidence 结果见 [`ami_single_meeting_diagnostic_report_20260812.md`](ami_single_meeting_diagnostic_report_20260812.md)，Brouhaha VAD 对照见 [`brouhaha_vad_diagnostic_report_20260812.md`](brouhaha_vad_diagnostic_report_20260812.md)。对应 artifact 分别位于 `ami_en2001a_utterances/outputs/ami_single_meeting_diagnostic_20260812/` 和 `ami_en2001a_utterances/outputs/brouhaha_vad_full_diagnostic_20260812/`。两个报告都并列 scorer 口径和单 meeting 限制，不能混排成一个总榜。

## 14. 当前 demo 的说明性结果

在 `EN2001a_utterance_00000` 上，native reference 为 3 个 speaker、`multi_speaker=true`、存在 overlap。已有 shadow 结果显示：

| 模型 | count (`metadata_v0_active_count`) | multi (`metadata_v0_multi`) | overlap (`metadata_v0_overlap`) | change (`public_v0_change`) |
| --- | ---: | --- | --- | --- |
| MOSS | 2 | true | false | true（shadow candidate；scorer=`public_v0`） |
| Sortformer v2 | 3 | true | true | true（shadow candidate；scorer=`public_v0`） |

其中 count 严格对应 `metadata_v0_active_count`，multi 是同一 scorer 的 `metadata_v0_multi` 派生值，overlap 对应 `metadata_v0_overlap`；change 仅表示当前 shadow scorer 的 `public_v0_change` 候选，尚未经过正式 scorer registry、校准或跨 session 评测。`floor_transfer_v2` 和 `different_speaker_onset` 不在这张 demo 表中，不能从 `candidate true` 推断它们为真。

这只能说明两条模型输出存在可分析的差异，不能据一条 sample 宣布 Sortformer 是“人数模型”或 MOSS 是“换人模型”。只有在完整分层矩阵和跨 meeting test 上重复观察到优势，才能把某项能力写成模型职责。

### 3.8 SURE 链路覆盖映射（核对结果）

对照 3 节 8 项能力检查 `sure-harness` 打分链路的覆盖情况。核对基准：`sure/external/sure-evaluation`（基础引擎，任务 = `asr, classification, kws, s2tt, sa_asr, sd, se, slu, tse, tts, vc`）与 `sure/external/sure-evaluation-vad`（VAD 引擎，commit `87b6bc4`，额外任务 `vad`），以及 harness 登记白名单（`asr, s2tt, sd, ser, tts, vc, kws, slu, gr, speech_understanding, sa-asr, sa_asr`）。

| 能力 | 本计划要求（第 3 节口径） | SURE 最接近的任务/指标 | 覆盖状态 |
| --- | --- | --- | --- |
| `C` speaker count | 1/2/3/4/5 人分层 count | 无对应任务 | ❌ 未覆盖 |
| `M` multi-speaker | bool（≥2 有效 speaker） | 无（classification 仅单标签 accuracy，无内置 count spec） | ❌ 未覆盖 |
| `O` overlap | bool + overlap_ratio + 区间/事件 | 无（SD 仅整体 DER，不分解 overlap 子事件） | ❌ 未覆盖 |
| `X` speaker change | bool/count/points，`public_v0_change` 与 `floor_transfer_v2` 分轨 | 无 change 事件指标 | ❌ 未覆盖 |
| `I` identity | 同/不同 speaker pair 判别 | 无 SV 任务；`ecapa_tdnn_sim` 只是 SE 内部打分节点，不是独立 SV 链 | ❌ 未覆盖 |
| `V` speech coverage | 20 ms 帧 speech/silence + onset/offset 边界误差 | `vad`（f1/p_fa/p_miss/dcf_nist/auc_roc，10 ms 网格、collar 0） | ⚠️ 部分覆盖 |
| `A` lexical clock | WER/CER、word coverage、timestamp error、SA-WER/cpWER | `asr`（zh CER / en WER / cs MER）+ `sa_asr`（仅 cpWER） | ⚠️ 部分覆盖 |
| `D` full diarization | speaker-attributed timeline 整体质量 | `sd`（仅 DER，meeteval，collar 0.25） | ⚠️ 部分覆盖 |

具体缺口：

- `A`：SURE 只给 CER/WER/MER 和 cpWER，没有 word coverage、timestamp error；SA-WER 无对应 route。
- `V`：VAD 指标只在 `sure-evaluation-vad` 引擎存在，harness 任务白名单没有 `vad`；此前登记只能用 `speech_understanding` 顶替（见 `model_evaluate/sure_evaluation_plan.md` 10.2/10.3）。且其 10 ms 网格 + collar 0 与本节 20 ms 网格口径不同，不能直接复现本计划的 V gold。
- `D`：只有整体 DER，不提供 miss/false alarm/confusion 分解，也不承载 `C/M/O/X` 子事件。
- `classification` 只能做 `key<TAB>label` 单标签 accuracy，理论上可勉强承载 `M`/`O`/`X` 的 bool claim，但分层 count、frame mask、event matching 无法表达；且用自定义 label spec 即偏离 SURE 标准 benchmark，需自行维护 label 语义。

结论：SURE 标准链路只能对 `A`（ASR 子集）、`V`（独立 VAD 引擎）、`D`（整体 DER）给出标准 benchmark 分数；`C/M/O/X/I` 五类能力当前没有对应任务。本计划要求的 gold 口径（20 ms grid、v0 segment 过滤/合并、Hungarian speaker mapping、0.10 s 事件、one-to-one IoU 匹配等）SURE 无法复现，这些能力必须继续走 tagger 伴随评测 scorer，SURE 分数只作为公开 benchmark 参照列。

补全 SURE 侧的建议（非本计划阻塞项）：向 `sure-evaluation` 提交任务提案新增 `sv`（identity pair）；`A` 的 timestamp error 需先建词级时间戳参考集；正式接入 `vad` 需要 harness 升级到含 `vad` 的引擎版本并把 `vad` 加入任务白名单。
