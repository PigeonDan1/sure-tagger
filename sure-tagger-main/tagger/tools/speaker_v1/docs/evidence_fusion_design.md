# Sure-Tagger 多说话人证据协作链路设计

> 状态：设计草案，不修改当前生产 pipeline
>
> 初稿日期：2026-08-10；text evidence 更新：2026-08-11
>
> 目标输出：utterance-level `speaker.multi_speaker`、`speaker.speaker_change`、`speaker.speaker_overlap`

> 选型边界：本文中的默认模型/角色是待评测的 fusion baseline，不是已证实的模型专长。单模型必须先按 [`evaluation/leaderboard_evaluation_plan_20260812.md`](evaluation/leaderboard_evaluation_plan_20260812.md) 在所有适用能力上独立评测，再用实测榜单决定 specialist、guard 和组合关系。

## 1. 结论先行

新的 speaker 链路不应继续采用“选择一条 route，生成一条 timeline，再把派生值当作高置信标签”的结构。输入严格保持现有 utterance-level `sample` contract，所有证据工具只处理当前 `sample.audio`，在 sample 内形成证据协作链路：

1. 用 MOSS-Transcribe-Diarize 0.9B 产生问题无关的初始 speaker timeline 和 transcript。
2. 用不同建模范式的 pyannote Community-1 产生第二条完整 timeline。
3. 按语言选择独立 lexical clock：中文/中英用 Paraformer-zh 的 CIF token interval，多语/未知语言用 Whisper base 的 word timestamp；再与 MOSS segment text 对齐，并投影到 pyannote timeline，形成可比较的 speaker-text track。
4. 用 Silero VAD 验证 speech coverage，用 CAM++ 验证候选片段是同一人还是不同人。
5. 对冲突建立 claim-local 的 `H1/H2/H3` 假设集：分别表示来源 A 的局部事件解释成立、来源 B 的局部事件解释成立、双方都不完整或都错；为每个分支生成可证伪预测和 dependency closure。
6. 只调用最能区分存活分支的独立证人，例如 Streaming Sortformer、当前 sample 音频内的物理通道、CAM++ 或 sample-local transformation；共享上游的派生结果不能让假设循环自证。
7. 保留每个模型的原始输出、能力边界、输入 lineage、假设、反证和未解决冲突，不做无来源的多数投票。
8. 先认证底层时间事件，再确定性派生公开标签；分支排除不能代替正常认证，无法认证时输出 `null`。

作为待评测 baseline 的第一版模型组合是：

| 角色 | 默认选择 | 原因 |
| --- | --- | --- |
| 初始宽能力观察 | MOSS-Transcribe-Diarize 0.9B | 一次得到 transcript、speaker、timestamp；Apache-2.0；50+ 语言 |
| 独立完整 timeline | pyannote Community-1 | 约 33 MB 模型组件；与 MOSS 架构和目标函数不同；包含 speaker counting、assignment 和 overlap 能力 |
| 独立 lexical clock | 中文/中英：Paraformer-zh；多语：Whisper base | Paraformer 有原生 CIF token interval；Whisper 有实验性 attention-DTW word interval；二者都不输出 speaker ID |
| 异构 lexical 第三观察 | SenseVoiceSmall | 非自回归、多语种中文友好；无稳定 token/word timestamp，只暴露 agreement pattern，不能多数投票裁决真值，且需许可 gate |
| 流式粗时间证人 | Sherpa-ONNX Zipformer zh-en | RNN-T token emission point 与前述时间机制不同；只做顺序/粗边界诊断，不冒充 word interval |
| speech coverage | Silero VAD | 约 2 MB；MIT；CPU 快；只承担 speech/non-speech 证据 |
| speaker identity 复核 | ModelScope CAM++ | 7.2M 参数、约 28 MB；Apache-2.0；适合在干净片段上做 same/different speaker |
| overlap 冲突证人 | NVIDIA Streaming Sortformer 4spk v2 | 117M；CC-BY-4.0；输出独立的 frame-level speaker activity；只在适用且有冲突时调用 |
| 冲突变换 | SpeechBrain SepFormer | 只处理候选双人 overlap 区域；变换结果本身不算证据 |
| speaker-text 对照 | MOSS joint text vs selected lexical clock x pyannote | 检查 lexical 覆盖、边界 ambiguity 和 backchannel；这是复合 diagnostic/guard，不是额外 speaker 票 |

Sortformer 需要区分版本：旧的 offline v1 是 `CC-BY-NC-4.0`，不进入生产候选；streaming v2 是 `CC-BY-4.0`；streaming v2.1 使用 NVIDIA Open Model License。v2.1 对新增 meeting 场景更稳健，但不是所有 benchmark 都全面优于 v2，且必须先通过法务审查。若项目的合规白名单只接受已预审的标准许可证，默认选择 v2；若 v2.1 条款获批，再做同条件替换实验。

## 2. 目标与非目标

### 2.1 目标

- 让多个小模型各自只回答其擅长的窄问题。
- 区分“独立证据一致”与“同一 timeline 的多个派生字段一致”。
- 对 true、false 和 abstain 使用不同的证据要求。
- 保持 raw-only input 和 tags-only public output；完整证据只写内部 artifact。
- 支持 mixed mono、separated headset 和 microphone array 三类输入。
- 保持 utterance-level raw sample 输入不变；所有时间、speaker ID、embedding 和 text track 都只在当前 sample 内有效。
- 可以逐工具校准、做消融、追踪成本和复现每个标签。
- 为每条匿名 speaker timeline 保存可追溯的 text/lexical-unit 对齐，但不把内部 transcript 暴露为新的 public tag。
- 对 material conflict 保存可复现的假设、预测、falsifier、检验选择和分支状态，使定向补证与人工复核可审计。

### 2.2 非目标

- 不识别真实姓名或跨数据集真实身份。
- 不从声音推断 gender、age、ethnicity、accent 等敏感属性。
- 不把 transcript、VAD 或 participant roster 单独当作 speaker 数证据。
- 不用词汇习惯、姓名、语言或语义内容推断真实 speaker identity。
- 不把“ASR 没转出第二路文字”当作无 overlap 或单说话人的负证据。
- 不把 source separation 生成的 stem 直接当作“两个人”的证明。
- 不按 `recording_id` 获取或拼接其它 utterance，也不跨 sample 借用 speaker ID、embedding、transcript 或上下文。
- 不用生成式 LLM 覆盖底层声学证据或强制裁决所有冲突。
- 不把模型 transcript 放进 public tags；内部 text artifact 必须遵守单独的 PII、访问控制和 retention policy。
- 不预设两个冲突来源中必有一个正确，也不把“分支暂时无矛盾”当作该分支已获支持或已认证。

## 3. 从 AUDIO-MIND 迁移什么

AUDIO-MIND 的价值不在“多调用几个工具”，而在证据权责划分。对 sure-tagger，建议迁移以下原则：

| AUDIO-MIND 原则 | Speaker 链路中的落法 |
| --- | --- |
| 初始音频感知 | MOSS 先给出完整 timeline、transcript 和候选事件；不确定性由后续一致性、覆盖度和能力检查产生 |
| 按证据缺口调用 | 缺 speaker identity 时调用 CAM++，缺 overlap 证据时调用适用的独立 overlap witness，缺词/时间对照时调用独立 ASR |
| 工具有能力边界 | 每个工具声明可支持和不可支持的 claim |
| 变换不是证据 | crop、denoise、separate 只产生新的音频输入 |
| 定向重听 | 只在 change/overlap 候选点附近，使用当前 sample 内可用的两侧上下文重新运行模型 |
| 保留冲突的中性摘要 | resolver 同时保存 supporting 和 conflicting evidence |
| 假设驱动取证 | 对冲突建立包含“双方都不完整”的分支，为各分支推导可证伪预测，再选择能最大程度区分分支的独立工具 |
| 最终回到原音频 | 所有派生结论必须回链到当前 `sample.audio` 的时间范围；人工复核播放该 sample 原音频而非只听 stem |
| 有限循环 | 为额外模型、变换和重听设置预算，证据仍不足就 abstain |

这里不照搬 AUDIO-MIND 的文本规划器。Phase 1 使用确定性 evidence-gap state machine，更容易复现和校准；小型 LLM 最多用于生成内部审计摘要，不能修改标签。

## 4. 标签语义先固定

当前标准见 [speaker_metadata_standard.md](../speaker_metadata_standard.md)。v2 应在运行前固定 `speaker_semantics_version`，否则模型一致也可能是在认证不同概念。

### 4.1 公开标签

| 标签 | v2 推荐语义 | 明确不包含 |
| --- | --- | --- |
| `multi_speaker` | target utterance 内至少两个不同匿名 speaker 有 material speech activity | participant roster、通道数、同一人的串音 |
| `speaker_overlap` | target utterance 内至少两个不同 speaker 同时 active，且绝对时长和 speech-relative ratio 均达到阈值 | 极短泄漏、混响尾音、同一人的多通道复制 |
| `speaker_change` | target utterance 内发生 primary floor holder 的 A -> B 转移 | A 持续讲话时 B 的纯 backchannel onset；仅 active set 改变 |

建议另外保留两个内部事件，避免把语义挤进一个 bool：

- `different_speaker_onset_points`：新的不同 speaker 开始 active，包括 overlap onset。
- `floor_transfer_points`：主要发言权从 A 转到 B。

如果产品决定 `speaker_change` 应表示“不同 speaker onset”，可以改变映射，但必须更换 semantics version，不能静默改变定义。

### 4.2 Material event

公开标签不应被几毫秒边界尾音触发。建议把以下参数写入 rule version，而不是硬编码：

```text
frame_hop_sec
min_speaker_activity_sec
min_overlap_duration_sec
overlap_ratio_threshold
speaker_change_context_sec
boundary_collar_sec
max_floor_transfer_gap_sec
```

当前 `overlap_ratio_threshold=0.05` 可以保留。认证层应另外设置阈值灰区，例如开发集初始实验可使用 `[0.04, 0.06]` 作为不自动认证区，而不是改变公开标签的 5% 定义。

仅用于开发集起跑、不得直接固化为生产标准的一组初值：

| 参数 | 初值 | 用途 |
| --- | ---: | --- |
| `frame_hop_sec` | 0.02 s | canonical occupancy 比较网格 |
| `min_speaker_activity_sec` | 0.10 s | 排除切片边缘的毫秒级尾音 |
| `min_overlap_duration_sec` | 0.10 s | 与 5% ratio 同时满足才是 material overlap |
| overlap certification gray zone | 0.04-0.06 | `<=0.04` 候选 false，`>=0.06` 候选 true，中间 abstain |
| `overlap_event_iou_min` | 0.50 | 两个来源的候选 overlap event 区域匹配 |
| `overlap_frame_f1_min` | 0.80 | event 被切碎时的 occupancy 对照门槛 |
| `scope_coverage_ratio_min` | 0.98 | 低于该覆盖率不认证 negative；缺口仍须单独审计 |
| `max_uncovered_gap_sec` | 0.08 s | 防止高总体 coverage 掩盖一段足以容纳 material event 的连续缺口 |
| `speaker_change_context_sec` | 0.30 s/side | 提取边界两侧 clean speech；不足则 abstain |
| `boundary_collar_sec` | 0.25 s | 不同 timeline 的 change point 一对一匹配 |
| `max_floor_transfer_gap_sec` | 1.00 s | 超过该静音 gap 不自动视作连续换人 |

所有初值都必须在 held-out dev set 上按语言、设备和场景校准。若公开 metadata 标准最终给出不同定义值，认证 gray zone 只能围绕标准值设置，不能反过来改变标准。

### 4.3 Canonical floor-state 算法

每条来源都用同一确定性算法从 raw speaker activity 产生 floor state，避免各 adapter 私自解释“主要说话人”：

1. 把未填 gap 的 speaker activity 投影到 20 ms grid。
2. 一帧只有一个 speaker active 时，该 speaker 是 floor candidate；无人 active 时为 `SILENCE`。
3. 多人 active 时，若上一 floor holder 仍 active，则保持该 holder；若上一 holder 已不 active 且有多个新 speaker，则标为 `UNKNOWN`，直到出现唯一且达到 material 时长的 speaker。
4. candidate 连续达到 `min_speaker_activity_sec` 才提交为 floor holder，提交后状态时间回溯到该 candidate run 的首帧；短暂第二人 onset 只记入 `different_speaker_onset_points`。
5. silence 不被填回 activity。仅当 silence gap `<= max_floor_transfer_gap_sec` 时，前后的 A/B floor state 才可组成连续 transfer 候选。
6. A -> B 的 canonical point 是已通过最小时长检查的 B candidate run 首帧；同时保存 `A_end_sec`、`B_floor_start_sec` 和中间 gap，不能只保存一个浮点时间。

每条 timeline 先独立运行该算法，再对 A -> B 候选做一对一 collar matching。不能先融合 timeline 再从融合结果“发现”两个模型共同支持的边界。A/B 长时间并发且旧 holder 不再可判、双方同时结束、边界落在 clip 边缘或 clean context 不足时，floor state 保持 `UNKNOWN`。

### 4.4 语义与接口版本

本设计的三项语义与当前 `speaker_diarization_v0.1` 不完全相同：v2 给 `multi_speaker` 增加 material activity 门槛，给 overlap 增加绝对时长门槛，并把 change 收窄为 floor transfer。因此 v2 不能直接覆盖相同 tag path 后静默上线。

实施前必须同步修订 [`speaker_metadata_standard.md`](../speaker_metadata_standard.md)，并在 dataset/run manifest 暴露：

```text
metadata_version
speaker_semantics_version
certification_rule_version
calibration_profile_id
```

tags-only sample 仍只包含 bool/null，但所属 dataset/run manifest 必须能唯一确定上述版本。旧、新语义不得混在同一数据发布版本中；标准未升级前，v2 只能 side-by-side 产出实验 artifact。

## 5. 证据不是标签

### 5.1 Claim contract

| Evidence claim | 可支持 | 不能单独支持 |
| --- | --- | --- |
| `speech_activity` | 某时段存在 speech、speech coverage、候选 speech boundary | speaker 数、speaker 身份、overlap |
| `speaker_activity` | 匿名 speaker 在某时段 active、候选 speaker count/change/overlap | 真实身份、名字 |
| `overlap_activity` | 某时段存在至少两路语音活动 | 哪两个人、是否故意打断 |
| `same_speaker_score` | 两个足够干净片段是同一人或不同人的相似度 | speech boundary、overlap 本身 |
| `word_activity` | 某局部存在可识别 lexical speech、词/短响应和时间戳；发现 VAD/diarizer 空洞候选 | 完整 speech coverage、canonical speech union、speaker 身份；普通单流 ASR 不能证明或否定 overlap |
| `speaker_attributed_lexical_unit` | 某 ASR lexical unit 与某 diarizer speaker activity 的时间关联 | 独立 speaker 票；它同时依赖 ASR 和被挂载的 diarizer |
| `transcript_agreement` | 两个 ASR 对内容、lexical-unit 覆盖和时间范围的稳定性 | 任一 transcript 正确、完整 speech coverage、speaker 数或身份 |
| `lexical_turn_cue` | backchannel、话语未完成/继续等候选语义线索 | floor transfer；只能触发 guard 或复核 |
| `channel_activity` | 某物理通道在某时段有 speech | 通道等于独立 speaker，除非唯一映射已验证 |
| `spatial_activity` | 不同方向存在声源活动的辅助线索 | 确切 speaker 身份；混响环境中的最终真值 |
| `transformed_audio` | 提供给下一工具的新输入 | 任何 speaker 标签；crop/separation 本身不是证据 |

### 5.2 正证据和负证据不对称

- 一个局部、清晰的第二 speaker 事件可以支持 `multi_speaker=true`。
- `multi_speaker=false` 要求在当前数据域通过负例召回门槛的来源完整覆盖整个 target scope；“没检测到”不能自动变成 false。
- 一个局部、时间匹配的双人活动可以支持 `speaker_overlap=true`。
- `speaker_overlap=false` 只表示没有达到操作阈值的 material overlap，不表示物理上绝对零重叠。
- 被切在音频边缘的 change、缺少两侧上下文的 change、过短的 speaker 片段应返回 `insufficient`。

正式认证 false 还必须校准“模型组合”的联合漏检风险。两个模型分别达到 recall 门槛，不代表它们在共享训练语料、设备或声学条件下不会一起漏检。negative calibration profile 至少绑定模型 revision 对、domain、layout、language、duration 和 speaker-count bucket；缺少对应组合校准时，最多输出 `supported`。

### 5.3 Text evidence 的能力边界

speaker text 最有价值的地方是把“声学事件发生在哪里”映射到“哪些词落在事件两侧”，不是用语言内容替代声学判断：

- MOSS 的 speaker、timestamp 和 text 是同一次 joint generation，只属于 `G1_moss_joint`。
- Paraformer-zh 或 Whisper 产生第二条 lexical timeline，但本身都没有 speaker ID。中文 profile 优先使用与 MOSS 建模差异更大的 Paraformer；多语 profile 使用 Whisper，并显式登记其与 MOSS 的共享声学前端 lineage。
- `selected ASR units x pyannote timeline` 是依赖 `{lexical ASR group, G2_pyannote_powerset}` 的复合 track，不会生成新的 speaker evidence group。
- 同一批 ASR lexical units 分别投影到 MOSS 和 pyannote，只能暴露两条 diarization 的 assignment 差异；不能冒充两次独立 ASR。
- MOSS 与任一独立 ASR 的 transcript 一致表示两条识别路线稳定，不表示内容一定正确；只有独立 reference 才能计算真实 WER/CER。
- ASR 时间输出不是同一种量：CIF token interval、attention-DTW word interval、RNN-T emission point、VAD sentence interval 和 forced-alignment interval 必须分类型保存，不能直接互换或等权投票。
- 当前 raw input 的 `sample.text.transcript` 可能来自人工标注或上游系统，默认只作 `G_oracle_native` 评测参考，不喂 production resolver；只有 provenance 明确且不存在 label leakage 时才能登记为普通 evidence。
- 文本可以承担正向 `lexical_presence_cue`、`boundary_ambiguity_guard`、`backchannel_review_trigger` 和 `crosstalk_guard`，默认不能承担完整 `speech_coverage_guard`、正向 `change_boundary_vote`、`speaker_identity_vote` 或 `overlap_event_vote`。

## 6. 开源模型调研

以下规模来自官方 README、model card 或模型仓文件，调研日期为 2026-08-10。不同项目的 benchmark、collar、overlap 处理和数据集不同，数值不能横向排名。

| 模型/工具 | 规模与许可 | 能提供的证据 | 主要限制 | v2 定位 |
| --- | --- | --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | 0.9B；权重约 1.82 GB；Apache-2.0 | joint transcript、speaker segments、timestamps、声学事件 | transcript 与 diarization 来自同一模型，只算一个 evidence group；短 overlap 召回必须单独按数据域评测 | 默认初始观察 |
| pyannote Community-1 | segmentation 约 5.9 MB + embedding 约 26.6 MB；CC-BY-4.0；gated；代码 MIT | 第二条完整 diarization timeline、speaker count、assignment、overlap | 需接受 Hugging Face 条件；应关闭或明确配置 telemetry；与单独 `segmentation-3.0` 共享 lineage | 默认独立 timeline |
| pyannote segmentation-3.0 | 约 5.9 MB；MIT；gated | speech、speaker segmentation、OSD | 如果已经使用 Community-1，不能再算独立 overlap 证据 | fast profile 的专用 OSD 或 Community-1 内部组件 |
| Silero VAD | JIT 约 2 MB；MIT；8/16 kHz | speech mask、coverage、边界辅助 | 不知道 speaker，也不知道 overlap；召回率、音乐和强噪声表现必须按域校准 | 所有 profile 必跑 |
| CAM++ (`iic/speech_campplus_sv_zh-cn_16k-common`) | 7.2M；约 28 MB；ModelScope 标注 Apache-2.0 | same/different speaker score、cluster continuity | overlap、短片段和域外语言会污染 embedding；只能使用当前 sample 内的干净片段，不足则 abstain | 默认 identity verifier |
| SpeechBrain ECAPA-TDNN | embedding checkpoint 约 83 MB；Apache-2.0；VoxCeleb | CAM++ 的替代 speaker embedding | 与 VoxCeleb 域相关；不能与相同训练域 embedding 简单算两票 | 英文或 CAM++ 对照实验 |
| NVIDIA Streaming Sortformer 4spk v2/v2.1 | 117M；checkpoint 约 471 MB；v2 为 CC-BY-4.0，v2.1 为 NVIDIA Open Model License | 80 ms frame、`T x 4` speaker activity probability，直接给出 overlap-aware timeline | 每次 invocation 固定 4 个 speaker slot；5+ 人时性能明显退化；以英文数据为主但训练含 AISHELL-4、AliMeeting | 冲突时的独立 overlap/timeline 证人；离线采用长 buffer 配置 |
| WeSpeaker diarization | 工具 Apache-2.0；checkpoint 另行登记 | VAD + speaker embedding + clustering 的 speaker count、continuity 和非重叠 timeline | 默认一段只分配一个 speaker，不支持 overlap；短 utterance sample 的 embedding 数不足时会退化 | speaker count/continuity 的低成本备选 |
| NeMo clustering diarizer | MarbleNet VAD + TitaNet + spectral clustering；各 checkpoint 许可分别登记 | speech、speaker count、非重叠 timeline 和 embedding | 不建模同时说话；与 Sortformer 有 vendor/部分训练数据相关性 | count/cluster 审计，不参与 overlap 认证 |
| NVIDIA Multilingual MarbleNet VAD v2.0 | 91.5K；NVIDIA Open Model License；16 kHz | 20 ms speech probability、speech coverage 和边界 | 不区分 speaker，也不检测 overlap；自定义许可需法务确认 | Silero 的多语 VAD 对照候选 |
| 3D-Speaker modular diarization | 代码 Apache-2.0；CAM++ 7.2M | VAD + segmentation + embedding + clustering；可选 overlap | README 明示小于 30 秒或 speaker 很多时较弱；开启 overlap 时使用 pyannote segmentation，不能算独立 | 无 Community-1 时的模块化备选 |
| Whisper tiny/base | 39M/74M；代码和官方权重 MIT | 第二套多语 text；`word_timestamps=True` 可输出 attention-DTW word interval | 不提供 speaker identity；官方 CLI 将 word timestamp 标为 experimental；tiny/base 及不同 runtime 只算一个家族；中文需使用非 `.en` checkpoint | 多语 lexical clock；base 为默认，tiny 只作低成本 scout |
| Paraformer-zh | 220M；checkpoint 约 881 MB；代码 MIT，所列 Hugging Face 权重标注 Apache-2.0 | 普通话为主的中英 ASR；`pred_timestamp=True` 输出原生 CIF character/token interval | 不提供 native speaker attribution；英文 BPE 需合并后才能称 word；必须固定具体 artifact 与 revision | 中文/中英 lexical clock 首选，不参与 core speaker 投票 |
| SenseVoiceSmall | 约 234M；原始 `model.pt` 936,291,369 bytes；代码 MIT；权重为 FunASR Model Open Source License v1.1 | 普通话、粤语、英语、日语、韩语 ASR；非自回归 lexical content | 官方链路没有稳定 token/word timestamp；`sentence_timestamp` 主要是外接 VAD region；不提供 native diarization；自定义许可需法务确认 | 异构 lexical 第三观察；无 reference 时不能裁决真值，不能生成 word-level speaker-text track |
| Sherpa-ONNX streaming Zipformer zh-en | 官方未列参数数；FP32 三组件合计约 357 MB；模型与代码 Apache-2.0 | 中英流式 RNN-T text 和逐 token emission timestamp | emission point 不是 token/word start-end；受 streaming right-context 延迟影响；不提供 speaker ID | audit 的相对独立粗时间证人 |
| FunASR `fa-zh` forced aligner | 官方约 38M；不是 ASR；代码 MIT；权重 artifact 许可标注存在差异，需按更严格条款复核 | 给定已知中文 transcript 后输出 character/token 强制对齐 interval | transcript/audio 不匹配会退化；依赖输入文本，不能回算成独立 text 票；与 Paraformer 同属 FunASR/SANM-CIF 生态 | 候选 transcript 通过 lexical-stability gate 后的边界复核，不进默认栈 |
| SpeechBrain SepFormer WHAMR/Libri2Mix | 主要 masknet 约 113 MB；model card Apache-2.0；固定 2-source、8 kHz | 从冲突片段生成两个候选 stem | 训练域与真实会议差异大；可能产生残留、置换和伪 stem | 只做 transformation，之后仍需 VAD/embedding 验证 |
| pyroomacoustics DOA/BSS | 代码 MIT；确定性 DSP | microphone array 上的 DOA、beamforming、BSS 辅助证据 | 需要阵列几何；混响和近共线源会退化 | array 输入的可选物理证据 |
| CAT-Net OSD | 代码 MIT；轻量 OSD 架构 | 专用 frame-level overlap probability | 官方仓当前提供训练代码，未核实可直接部署的通用预训练 checkpoint | Phase 3 自训练候选，不进 Phase 1 |

### 6.1 哪些模型实际具备 ASR

| 模型 | ASR/text 能力 | 原生时间输出 | speaker-attributed text | 在 v2 中的用法 |
| --- | --- | --- | --- | --- |
| MOSS-Transcribe-Diarize | 有，联合生成 text、speaker 和 timestamp | speaker segment interval；当前 adapter contract 不保证词级时间 | 有，来自同一次 joint generation | `G1` 主 track；text、speaker、time 不能拆成多票 |
| Whisper tiny/base | 有，约 99 语；中文使用 multilingual checkpoint | 可选的 attention-DTW word interval；官方仍标为 experimental | 无 | `G9` 多语 lexical clock；base/tiny/不同 runtime 同族只算一票 |
| Paraformer-zh | 有，普通话为主的中英混合 ASR | 原生 CIF character/token start-end interval；需显式 `pred_timestamp=True` | 无；示例中的 speaker label 来自另接 CAM++ | `G12` 中文/中英 lexical clock 首选 |
| SenseVoiceSmall | 有，覆盖中、粤、英、日、韩 | 没有稳定 token/word timestamp；`sentence_timestamp` 主要是外接 VAD segment | 无 | `G11` 独立文本内容观察；无 reference 时不能裁决哪份 transcript 正确，也不能直接投影成词级 speaker track |
| Sherpa-ONNX Zipformer zh-en | 有，streaming RNN-T 中英 ASR | token emission point，不是 start-end interval | 无 | `G13` 流式顺序/粗边界 diagnostic |
| FunASR `fa-zh` | 无；输入是 audio + 已知 transcript | forced-alignment character/token interval | 无 | `G14` 候选 transcript 通过 stability gate 后的时间复核；不是新的 ASR/text 票 |
| pyannote Community-1 / Streaming Sortformer | 无 | speaker activity timeline | 无 | 接收外部 lexical units 后产生复合 speaker-text track |
| CAM++ / ECAPA | 无 | 无 lexical time | 无 | 只做 same/different speaker identity guard |
| Silero / MarbleNet VAD | 无 | speech interval/probability | 无 | 只做 speech coverage/boundary guard |
| SepFormer | 无 | 无 | 无 | 只生成 transformed stems；另跑 ASR 后仍继承 derived lineage |

因此，只有 MOSS 原生输出 speaker-attributed text；其它 ASR 都必须把 lexical units 投影到某条 diarizer timeline，所得结果同时依赖 ASR 和 diarizer。不能让“每个模型都转一份文字”，更不能因 transcript 数量增加而升级 speaker event 票数。

默认按语言选一条 lexical clock，不在同一批数据中无版本切换：

- `zh_zh-en`：Paraformer-zh 提供独立文本和 CIF token interval；SenseVoiceSmall 只在 lexical content 冲突时提供第三份观察。若仍无 reference，冲突保持 unresolved，不能用 2/3 多数票命名真值。
- `multilingual_or_unknown`：Whisper base 提供文本和实验性 word interval；tiny 只作低成本 scout，与 base 不重复计票。
- `streaming_zh-en_audit`：Zipformer 提供相对独立的 emission-point 顺序检查，但不参与 interval overlap fraction。
- 争议边界只有在候选 transcript 通过 versioned lexical-stability gate 后才可送 `fa-zh`；这不把候选提升为真值，forced aligner 输出仍继承所用 transcript 的依赖，不能成为新 text 票。

每条 lexical evidence 必须登记 `timestamp_method`：`joint_segment_interval`、`native_cif_token_interval`、`attention_dtw_word_interval`、`rnnt_token_emission_point`、`vad_segment_interval` 或 `forced_alignment_token_interval`。只有经过 capability probe 和本域校准的 interval 类型才能计算 lexical-unit 与 speaker occupancy 的交叠；point 和 VAD segment 不能伪装成 word interval。

### 6.2 推荐的许可策略

代码仓 license 与具体 checkpoint license 必须分开登记：

- MOSS checkpoint model card 为 Apache-2.0，可进入默认实验和生产候选。
- pyannote Community-1 checkpoint 为 CC-BY-4.0 且 gated，需要接受条件、保留 attribution，并由项目确认发布方式。
- ModelScope CAM++ API 标注 Apache-2.0，下载时固定 model revision 和 license snapshot。
- Whisper 代码和官方权重为 MIT；仍需固定具体 checkpoint revision、解码参数和 word-timestamp 设置。tiny/base、faster-whisper、whisper.cpp 和 Transformers 转换版若源自同一 checkpoint，仍属于同一模型家族。
- 中文/中英 profile 若回退到 Whisper，必须使用 multilingual `tiny`/`base`，不能误用 `tiny.en`/`base.en`；语言提示、temperature/fallback、`condition_on_previous_text` 和任何外部 VAD/hallucination filtering 也进入 lineage。
- 所列 `funasr/paraformer-zh` Hugging Face 权重标注 Apache-2.0、FunASR 代码为 MIT；仍需固定下载来源和 revision，不能把其它同名 ModelScope artifact 的许可自动继承过来。
- SenseVoice 代码为 MIT，官方权重使用 FunASR Model Open Source License v1.1；官方说明允许在遵守条款时商用，但 attribution/model-name 等义务和具体转换权重仍需法务逐项确认。默认只在隔离评测环境启用。
- 所列 Sherpa-ONNX Zipformer zh-en artifact 和 sherpa-onnx 代码标注 Apache-2.0；必须固定 encoder/decoder/joiner 三个组件及 token 文件的 revision/hash。
- `fa-zh` 在不同发布页的权重许可标注不一致；生产按更严格条款处理，在法务确认前只做隔离评测。它不是 ASR，许可通过也不会成为独立文本票。
- Sortformer 必须按版本登记：offline v1 为 CC-BY-NC-4.0；streaming v2 为 CC-BY-4.0；streaming v2.1 为 NVIDIA Open Model License。默认生产候选是 v2，v2.1 在法务批准前只能做隔离评测。
- WeSpeaker 文档称 pretrained model 继承训练数据许可，即使某个 Hub card 标注 Apache-2.0，也应采用更严格解释并做法律复核。
- Source separation checkpoint 必须逐个检查训练数据和模型卡；不能因为 SpeechBrain/Asteroid 代码开源就推断所有权重可商用。

### 6.3 不建议绑定的旧路线

| 候选 | 原因 |
| --- | --- |
| NeMo MSDD / `diar_msdd_telephonic` | 2026-03-23 已从 NeMo 主分支删除，删除说明将其归为旧且不再维护的模型；新链路不应固定在旧 NeMo 版本上 |
| `pyannote/overlapped-speech-detection` | 旧 pipeline 依赖旧 segmentation；专用类已从当前 pyannote 源码移除，而且与 Community-1 同源，不能补足独立证据 |
| Offline Sortformer 4spk v1 | 123M、最多 4 人、CC-BY-NC-4.0；许可和长音频适用性都不如 streaming v2/v2.1 |

### 6.4 Sortformer applicability gate

4-speaker 限制作用于当前 utterance sample 的一次实际 model invocation。不得为了满足限制而加载 recording 或相邻 utterance；模型只能看到 `sample.audio` 或它的 sample-local crop。

- 只有当前 sample 自带的可信 participant upper bound、物理 channel mapping 或人工确认约束能证明 invocation `<=4`；不能拿待裁决的 MOSS/pyannote count 反过来证明冲突证人适用。
- 当前 sample 可能超过 4 人或没有可信上界时，Sortformer 不得为 exhaustive speaker count、`overlap=false` 或完整 sample timeline 提供认证票。
- 可以在当前 sample 范围内对候选区域 crop 并重置模型状态，经专项校准后作为 local positive witness；它不能作为当前 sample 的完整 negative witness。
- 官方仍报告 5-9 speaker 结果，因此这不是“模型无法运行”，而是 certification capability gate。

每次调用必须记录 `sample_id`、`invocation_audio_id`、sample-relative 输入范围、`max_speaker_slots=4`、可信 speaker 上界及其 evidence ID，以及用途是 `sample_full_timeline` 还是 `sample_local_positive_only`。

## 7. Evidence independence graph

每条 evidence 必须有 `independence_group` 和 `lineage`。同组输出只能为同一 claim 提供一票。

| Group | 来源 | 相关性说明 |
| --- | --- | --- |
| `G1_moss_joint` | MOSS transcript、timestamp、speaker timeline、per-channel MOSS | 同一 checkpoint 和生成过程 |
| `G2_pyannote_powerset` | Community-1 timeline、其 segmentation/OSD 输出 | 同一 pipeline 或同一 segmentation checkpoint |
| `G3_sortformer_e2e` | Streaming Sortformer frame probabilities 和派生 segments | 独立 end-to-end 架构；v2/v2.1 只能选一个版本算一票 |
| `G4_campplus_identity` | ModelScope/3D-Speaker/WeSpeaker 中同一 CAM++ checkpoint 的 embedding 和 clustering | 同一 embedding family；重复 wrapper 不增加独立性 |
| `G5_ecapa_identity` | SpeechBrain ECAPA embedding | 可作为 CAM++ 对照，但同为 VoxCeleb 类训练域时仍有相关误差 |
| `G6_silero_vad` | Silero speech timestamps 及其派生 nonspeech ratio | 只对 speech claim 独立 |
| `G7_physical_channel` | 已验证唯一 participant 映射的 close-talk channel activity | 物理视角最强，但要防串音和同人多通道 |
| `G8_array_spatial` | DOA/beamforming | 只在阵列几何已知时适用 |
| `G9_whisper_asr` | Whisper text/attention-DTW timestamps；tiny/base 及转换 runtime | 同一 Whisper 家族只算一票；与 MOSS 共享 Whisper-style feature/frontend lineage，按相关证据校准 |
| `G10_nemo_cluster` | MarbleNet、TitaNet、spectral clustering 及其派生 timeline | 不支持 overlap；与 Sortformer 共享 vendor/部分训练数据，相关性要显式降权 |
| `G11_sensevoice_asr` | SenseVoiceSmall text；外接 VAD sentence interval 不拆组 | 非-Whisper lexical family；只对文本内容 claim 独立，无词级时间能力 |
| `G12_paraformer_asr` | Paraformer-zh text/CIF token interval | 中文 lexical clock；与 MOSS/Whisper 建模差异较大，但与 FunASR/`fa-zh` 的 vendor/recipe 相关性要登记 |
| `G13_zipformer_asr` | Sherpa-ONNX Zipformer text/token emission point | RNN-T 流式家族；只对 text 和粗时间顺序 claim 独立 |
| `G14_funasr_forced_alignment` | `fa-zh` 给定 transcript 后的 token interval | 不是 ASR；继承输入 transcript 依赖，与 Paraformer 的 SANM/CIF 生态相关，不能提供 text vote |
| `G_oracle_native` | 独立制作的人工 annotation | 只用于 calibration/audit，不得在 production evaluation 中泄漏 |

`independence_group` 不是“模型名称不同就独立”。registry 还应保存 `shared_training_corpus`、`shared_frontend`、`shared_checkpoint` 和 `shared_vendor_or_recipe`。例如 NeMo clustering 与 Sortformer 可分组，但若共享训练语料，融合器不能假设二者误差完全独立。

因此 group 用于防止重复计票，不是统计独立性的证明。特别是 `certified=false` 必须引用适用的 `joint_negative_calibration_profile_id` 或经批准的保守风险上界。

复合 text track 必须保存完整依赖集合：

```text
MOSS speaker-text track
  dependency_groups = {G1_moss_joint}

Paraformer tokens x pyannote timeline       # zh / zh-en profile
  dependency_groups = {G12_paraformer_asr, G2_pyannote_powerset}

Whisper words x pyannote timeline           # multilingual profile
  dependency_groups = {G9_whisper_asr, G2_pyannote_powerset}

Whisper words x Sortformer timeline
  dependency_groups = {G9_whisper_asr, G3_sortformer_e2e}

Whisper per-channel text x physical channel activity
  dependency_groups = {G9_whisper_asr, G7_physical_channel}

SenseVoice transcript content
  dependency_groups = {G11_sensevoice_asr}  # no word-level projection

fa-zh alignment of a Paraformer candidate transcript
  dependency_artifact_ids = {paraformer_transcript_artifact, fa_zh_evidence}
  dependency_groups = union(closure(paraformer_transcript_artifact), {G14_funasr_forced_alignment})
```

若 forced aligner 输入的是依赖 MOSS、Paraformer 和 SenseVoice comparison 构造的候选 transcript artifact，闭包必须包含 `{G1, G11, G12, G14}`；不能因为最终字符串采用 Paraformer 格式就只登记 `G12 + G14`。无 reference 时，这个候选及其 alignment 仍是 diagnostic，不得命名为 transcript 真值。

两个复合 track 只要共享任一 group，就不能为同一个 claim 贡献两张独立票。MOSS 与 Whisper checkpoint/decoder 不同，但共享 Whisper-style acoustic feature/frontend lineage；二者应标记 `shared_frontend_lineage=true` 并通过联合校准确定 lexical guard 权重，而不是假设完全独立。

下列组合属于伪交叉认证：

- MOSS diarization + MOSS transcript。
- 同一 Whisper words 分别挂到 pyannote 的 raw/exclusive timeline。
- Whisper tiny + base、同一 checkpoint 的不同 runtime，或 beam/temperature/语言提示不同。
- SenseVoice `sentence_timestamp` + 生成该区间的同一 FSMN-VAD。
- Paraformer transcript + 用该 transcript 驱动的 `fa-zh` forced alignment。
- Community-1 + 单独调用同一 `segmentation-3.0`。
- 3D-Speaker overlap mode + pyannote OSD。
- 3D-Speaker CAM++ + 同 checkpoint 的 WeSpeaker/CAM++ wrapper。
- 同一模型不同 seed、重复运行、不同量化版本。
- `segments -> speakers -> recording_summary -> public flags`。
- source separation 后再次运行原模型，但没有记录 parent lineage。

## 8. v2 总体架构

```text
utterance-level sample + sample.audio.path
                 |
                 v
        [0. sample contract / audio probe]
                 |
       +---------+----------+----------+
       |                    |          |
       v                    v          v
 [MOSS joint]       [Silero mask] [selected lexical clock]
       |                    |          |
       +---------+----------+----------+
                 |
                 v
       [evidence-gap planner]
                 |
       +---------+--------------------+
       |              |               |
       v              v               v
 [Community-1]   [CAM++ verify]  [channel / DOA]
       |              |               |
       +---------+----+---------------+
                 |
                 v
 [timebase + speaker alignment + lexical alignment]
                 |
                 v
        [per-claim evidence resolver]
          |                    |
          | certified          | conflict / gap
          v                    v
 certified claim events   [H1 / H2 / H3 hypothesis set]
          |                    |
          v                    v
 deterministic metrics   [predictions + falsifiers
          |                + dependency closure]
          v                    |
 public bool/null              v
                       [select independent
                        discriminating test]
                               |
                               v
                       [sample-local crop / separation /
                        Sortformer / identity / channel /
                        lexical third observation]
                               |
                    new evidence -----> alignment + resolver
                               |
                         unresolved ----> abstain / human review
```

### 8.1 处理粒度

输入固定为现有 utterance-level raw sample，不新增或重解释任何 input 字段：

- 每个工具读取同一份 `sample.audio.path` 指向的完整音频文件；整份文件就是当前 utterance，不存在 raw input `start_sec/end_sec`。
- `duration_sec`、采样率和通道数由 audio probe 从该文件派生；label scope 是当前 sample 的 `[0, duration_sec]`，所有标准化 timestamp 都相对 sample 音频起点。
- `recording_id` 不是新 raw input 字段，只能从已有 `sample.native_metadata` 复制为可选 opaque provenance；缺失时内部兼容 metadata 使用 `sample_id`。它不能作为 artifact join key，也不能用于加载或拼接同一录音中的其它 utterance。
- speaker ID、embedding cluster、text track 和 alignment 都是 sample-local，不跨 sample 复用。
- targeted crop、denoise 或 separation 只能在当前 sample 边界内产生 derived audio，并完整记录 parent lineage。
- channel/DOA 证据只能使用 `sample.audio.path` 文件内实际存在的通道；native metadata 只能辅助判断适用性，不能据此额外获取其它音频。

utterance-level 输入的能力边界必须显式反映到状态：

- sample 太短、clean non-overlap speech 不足时，CAM++/clustering 不强制判断，相关 claim 为 `insufficient`。
- change 落在 sample 起止边缘且 sample 内缺少单侧上下文时，不获取相邻 utterance，直接 `insufficient`。
- 短 backchannel 只能使用当前 sample 内可见的 activity/text 判断，不能借用其它 utterance 的 embedding 或 speaker roster。

### 8.2 三种运行 profile

| Profile | 工具 | 输出定位 |
| --- | --- | --- |
| `fast` | audio probe + Silero + MOSS | 高覆盖 `supported/insufficient` 标签；原则上不宣称 certified |
| `certify` | fast + Community-1 + 按语言选择 Paraformer-zh 或 Whisper base + 按需 CAM++ | 默认离线打标；生成两条 speaker-text track 并对三项公开标签做正式认证 |
| `audit` | certify + SenseVoice third observation / Zipformer coarse clock / forced alignment + physical/native audit + Streaming Sortformer v2/v2.1、ECAPA、separation、human | 阈值校准、错误分析和困难样本复核；自定义许可模型只在获批环境使用 |

## 9. Evidence-gap planner 与假设分支检验

Phase 1 使用确定性状态机：

```text
RUN_BASELINE
  -> CHECK_SCOPE_AND_COVERAGE
  -> RUN_SECOND_TIMELINE
  -> ALIGN_SPEAKERS
  -> ALIGN_WORDS_AND_BUILD_TEXT_TRACKS
  -> RESOLVE_MULTI_SPEAKER
  -> RESOLVE_OVERLAP
  -> RESOLVE_CHANGE
  -> FINISH | BUILD_CONFLICT_HYPOTHESES | ABSTAIN

BUILD_CONFLICT_HYPOTHESES
  -> DERIVE_PREDICTIONS_AND_FALSIFIERS
  -> SELECT_DISCRIMINATING_TEST
  -> ACQUIRE_TARGETED_EVIDENCE
  -> ELIMINATE_BRANCHES
  -> REENTER_ALIGNMENT_AND_RESOLVERS | ABSTAIN
```

### 9.1 假设分支与判定边界

假设分支是 claim-local、event-local 的冲突诊断和 acquisition planning artifact，不是新的 evidence group，也不能直接输出 bool。它应写成 `H_overlap(2.20-2.40s)` 或“候选 A/B 是否为不同 speaker”，不能写成“MOSS 整条链路正确”：同一来源可能只在一个局部事件上正确。每个 material conflict 至少包含：

```text
H1 = source A 对该 claim/局部事件的解释成立
H2 = source B 对该 claim/局部事件的解释成立
H3 / H_other = 双方都不完整、都错，或遗漏了其它故障机制
```

不得只建立 H1/H2 并假设二者必有一个正确；必要时把 `H_other` 进一步拆成“两个候选事件都真”和“两个候选事件都假”。每个分支必须保存 assumed event/value、适用时间范围、继承的 `dependency_artifact_ids/groups`、可观测 predictions、typed falsifiers、允许调用的独立工具和当前 `untested/viable/falsified/untestable` 状态。假设状态与 claim 的 `certified/supported/conflicted/insufficient` 状态严格分离。

prediction、falsifier、目标 region、所需 capability 和允许的 dependency groups 必须在定向取证前，按 versioned template 冻结。禁止看到新工具结果后改写预测或补造反证条件；模板无法表达的新 failure mode 进入 `H_other`，由人工复核后再升级模板版本。

预测只有在能被不继承该假设冲突来源的证据检验时才有区分价值。validator 必须与被检验来源的完整 dependency closure 隔离；若共享任一关键 group，它最多是 expected derived consistency，不能支持或反驳该分支。例如“假设 pyannote 正确 -> 用 pyannote occupancy 给 ASR token 归人 -> speaker-text track 与 pyannote 一致”是循环自证；同 checkpoint 重跑、参数变体和 separation 后再次运行原模型也不构成新的独立 validator。

falsifier 分三类：

| 类型 | 作用 | 示例 |
| --- | --- | --- |
| hard contradiction | 直接淘汰分支 | timestamp 越过 sample scope；违反 `overlap=true => multi_speaker=true`；依赖闭包出现循环 |
| calibrated independent falsifier | 达到专项校准门槛时淘汰分支 | 干净片段 CAM++ 明确 same speaker，反驳“两段属于不同人”；适用的独立 timeline 在候选区给出相反 activity |
| diagnostic mismatch | 只降低优先级或触发复核 | transcript 内容不一致、单流 ASR 未转出第二路、separation stem 不稳定 |

“预期事件没有出现”只有在 validator 对当前 claim、domain、region 和 scope 具备校准后的 negative capability 时才是 falsifier。局部 crop、单流 ASR 或未经负例校准的 timeline absence 只能记为 `untestable/diagnostic mismatch`，不能淘汰正向分支，更不能认证全 sample 的 `false`。

常见冲突使用固定模板，Phase 1 不依赖生成式 LLM：

| 冲突 | H1 | H2 | H3 | 优先区分证据 |
| --- | --- | --- | --- | --- |
| MOSS 两人、pyannote 一人 | pyannote merge 了两人 | MOSS split 了同一人 | 两者都漏掉噪声/第三活动机制 | A/B clean CAM++；适用的第三 timeline；当前 sample 物理通道 |
| 两个 overlap bool 相同但位置不同 | 仅区域 A 真实 | 仅区域 B 真实 | 拆成“两区都真”与“两区都假/其它”子分支 | 分别对两个窗口运行独立 overlap witness、identity 与 coverage 检查 |
| change 与 backchannel 解释冲突 | 发生 A -> B floor transfer | A 持续持有 floor，B 仅短响应 | 边界上下文不足或两 timeline 都错 | sample 内边界两侧 activity、CAM++、VAD gap；timed lexical cue 只作 guard |
| 双物理通道看似双人、mixed diarizer 只见一人 | 两个真实 participant 同时活动 | 同一 speaker 串音/复制到两通道 | 映射无效、噪声或其它 channel failure | 独立 activity、跨通道 embedding、相对能量与传播延迟；通道数本身不计票 |

分支结果按以下规则回到正常 resolver：

| 结果 | 允许动作 |
| --- | --- |
| 一个分支被支持，竞争分支被反驳 | 保留该解释并重新运行正常认证；不能跳过 event/identity/coverage/calibration 条件 |
| 一个分支被淘汰，另一个只“未被反驳” | 最多 `supported -> null`，不能用排除法自动 certified |
| 多个分支仍存活 | `conflicted -> null`，继续有限取证或人工复核 |
| H1/H2 都被淘汰 | 转入 H3，记录未知故障；不得强制二选一 |
| `false` 分支存活 | 仍须完整 scope coverage 和对应模型组合的 joint-negative calibration |

`hypothesis_case` artifact 必须至少保存：

- `case_id`、`sample_id`、`claim_id/value/interval`、`template_id` 和 `semantic_version`；scope 必须是当前 `sample_id`。
- conflict/assumed evidence IDs、每个分支的 dependency closure 和 `prohibited_validator_groups`。
- 冻结的 `predictions[{predicate, region, falsifier, required_capability, allowed_groups}]`。
- 检验选择理由、实际 tests/evidence IDs、分支状态、淘汰原因和 budget termination reason。
- `certification_effect`，且只能是 `none`、`remove_conflict` 或 `reenter_resolver`，不能是 `certify`。

### 9.2 按 claim 选择检验动作

假设模板确定“需要区分什么”，下表再把证据缺口映射为具体工具动作：

| 证据缺口 | 动作 |
| --- | --- |
| 两条 timeline speaker count 不同 | 在双方干净非重叠片段上跑 CAM++；必要时重新 clustering |
| 都检测多人但 speaker mapping 不稳定 | 用 CAM++ 相似度 + 非重叠时间覆盖做 Hungarian matching |
| overlap bool 相同但时间区域不同 | 当前 sample invocation 有可信 `<=4` 上限时运行 Streaming Sortformer v2；否则优先使用 sample 自带的物理通道/独立训练 OSD，或把 sample-local fresh-state crop 限为 local positive witness；没有适用证人就标记 conflicted |
| overlap 冲突集中在短区域 | 在当前 sample 内取候选区及可用两侧上下文，定向重跑适用的独立证人；可做一次 2-source separation，再在 stem 上运行 VAD/embedding，但 separation 不增加票数 |
| change point 不一致 | 只在当前 sample 内取候选点两侧干净窗口，比较 embedding、VAD boundary 和两条 speaker-text track；text 只作 ambiguity/review guard；任一侧上下文不足则 `insufficient` |
| MOSS 与 selected lexical clock 的内容或 interval 大幅不一致 | 记录 transcript/timing disagreement；先局部 re-decode。文本仍冲突时用 SenseVoiceSmall 暴露第三种 agreement pattern，但无 reference 时不裁决真值；时间仍冲突时可用 Zipformer 作粗顺序检查，候选 transcript 通过 stability gate 后再条件调用 forced aligner；三者都不能升级 speaker event 票 |
| lexical unit 有 interval 但 diarizer/VAD 标为 silence | 触发 speech-coverage conflict，回看原音频并重跑局部 VAD/diarizer |
| 同一 lexical unit 被 timeline 分配给多个 speaker | 若同帧多人 active，保存 `ambiguous` candidates；禁止强制挑一个 speaker |
| 多个 close-talk channel 在同一时间转出高度相同文字 | 联合相对能量、延迟和 embedding 检查串音/复制，不能按通道数直接判多人 |
| separated headset 疑似串音 | 比较通道相对能量、跨通道 embedding 和 mixed-audio diarizer |
| 仍冲突或质量门禁失败 | `conflicted`/`insufficient`，公开值为 `null` |

初始预算建议：

```text
max_acquisition_rounds = 3
max_hypotheses_per_conflict = 4
max_predictions_per_hypothesis = 4
max_baseline_utterance_diarizers = 2
max_baseline_utterance_asr_tracks = 2
max_lexical_third_observer_models = 1
max_targeted_asr_redecodes_per_claim = 1
max_targeted_transformations_per_claim = 1
max_conflict_witness_models_per_claim = 1
```

预算是防止分支、工具噪声和计算成本无限累积，不是准确率定律；应在消融实验中调整。达到预算后仍有多个可行分支时必须 abstain，不能选择“矛盾最少”的分支强制输出。

## 10. Timebase、speaker alignment 与不可变 activity

### 10.1 Timebase

- 保存每个模型原始 timestamps，并统一映射为相对当前 sample 音频起点的时间；所有区间必须裁剪到 `[0, duration_sec]`。
- 建议同时投影到 20 ms canonical occupancy grid，便于比较 mask；原始 endpoint 不丢弃。
- 记录 resample、mixdown、sample-local crop、denoise、separation 的完整 parent chain；derived audio 的局部时间必须映射回 sample-relative time。
- endpoint 比较使用 collar，不要求浮点时间完全相等。初始实验可从 250 ms collar 开始。

overlap ratio 的分母必须统一。每条来源同时保存自己的 native overlap numerator、native speech-union denominator 和 native ratio；resolver 另用同一个 canonical speech mask 重算：

```text
canonical_overlap_ratio(source)
  = duration(source_overlap_mask AND canonical_speech_mask)
  / duration(canonical_speech_mask)
```

不同来源只能比较 canonical ratio，不能直接比较分母不同的 native ratio。canonical speech mask 由 speech-coverage resolver 产生并有独立状态；它未通过 coverage 门禁时，依赖 5% ratio 的 overlap 标签不能 certified。

### 10.2 Speaker ID alignment

不同模型的 `spk_001` 没有可比性。对齐顺序建议：

1. 只在高质量非重叠区域提取 embedding。
2. 构造跨模型 speaker 的 embedding similarity 和时间交集矩阵。
3. Hungarian matching 最大化联合分数。
4. 无法一一映射的 speaker 保留 unmatched，不强行合并。
5. public label 使用对齐后的匿名 consensus ID，真实 ID 永不推断。

### 10.3 Activity 不可变原则

原始 speech/speaker activity interval 是证据，turn grouping 是派生视图：

- 不得为了合并同一 speaker 的 turn 而填满中间 gap。
- overlap 必须从原始 occupancy 计算，不得从 gap-filled turn 计算。
- speech duration 和 coverage 也从原始 activity union 计算。
- `merge_gap` 只能影响 turn count 和展示，不得改变 activity、overlap 或 speaker coverage。

这条规则直接避免当前实现中 `A:0-1, B:1-1.2, A:1.2-2` 被合并成伪 overlap。

### 10.4 Lexical alignment 与 speaker-text track

保留 MOSS 的 joint speaker segment text/timestamps，以及按语言 profile 选择的独立 lexical clock 原始输出。当前 MOSS contract 不保证词级时间，因此绝不通过均匀切分 segment 来伪造 MOSS word timestamps；也不能把 SenseVoice 的 VAD sentence region 或 Zipformer 的 emission point 伪造成 token start-end。alignment 采用以下确定性步骤：

1. raw text、raw token/word 和原始 timestamp value 在 artifact 有效生命周期内不可变保存，同时受访问、retention 和 delete policy 管理；另保存 `timestamp_method`、`value_type=interval|point|segment`、frame/subsampling、外接 VAD evidence ID 和 capability-probe 结果。
2. 拉丁文字可做 Unicode normalization、case/punctuation/whitespace 归一化；中文按校准后的 character/word tokenizer 比较。Paraformer 的中文 character 可直接作为 lexical unit，英文 BPE 必须按模型 tokenizer 合并；不得用一个仅适合英文空格分词的规则覆盖所有语言。
3. 用 sequence alignment 联合 lexical edit cost 与适用于该时间类型的 temporal constraint，把 MOSS segment 内的 token span 匹配到 lexical units；只比整段字符串会掩盖局部漏词和时间漂移。
4. 只有 `native_cif_token_interval`、通过 capability gate 的 `attention_dtw_word_interval` 或合格的 forced-alignment interval，才能与 diarizer raw speaker occupancy 求交。把 timestamp uncertainty collar 加到 lexical interval 后，只有恰好一个 material speaker active 且覆盖达到门槛时才为 `assigned`；只要存在第二个 material competitor，无论谁占优都保存全部 candidates 并标为 `ambiguous`；没有 speaker 覆盖时为 `unassigned`。普通单流 ASR 的响度偏好不能被解释为 source attribution。
5. `rnnt_token_emission_point` 只能做 token 顺序和 collar 内粗边界检查；`vad_segment_interval` 只能做 segment coverage。二者都不能计算 lexical-unit occupancy fraction，也不能强制归人。
6. lexical unit 横跨 change collar 时不切造新 unit，保存 `boundary_ambiguous=true`；不得为了提高 agreement 移动 ASR 或 diarizer endpoint。
7. 先对匿名 speaker ID 做 Hungarian alignment，再比较 matched lexical unit 的 speaker assignment、coverage 和边界位置。

默认生成：

```text
track_moss_joint
  = MOSS native segment text/time + MOSS native speaker assignment

track_paraformer_pyannote          # zh / zh-en certify profile
  = Paraformer CIF token intervals projected onto Community-1 raw speaker activity

track_whisper_pyannote             # multilingual certify profile
  = Whisper attention-DTW word intervals projected onto Community-1 raw speaker activity

track_whisper_sortformer            # conditional audit only
  = Whisper words projected onto Sortformer activity

transcript_sensevoice              # conditional third lexical observation
  = SenseVoice text only; VAD sentence region is not a word-level track

track_zipformer_emissions          # conditional streaming audit
  = Zipformer token emission points; ordering/coarse-boundary diagnostic only

track_fazh_consensus               # conditional post-consensus audit
  = versioned candidate transcript forced-aligned by fa-zh; inherits full transcript dependencies and remains diagnostic
```

每个复合 track 必须通过 `source_artifacts` 记录 ASR/diarizer evidence ID，并保存 `dependency_artifact_ids`、`dependency_groups`、`timestamp_method`、normalizer/tokenizer version、assignment thresholds 和 ambiguity 状态。它不替换 raw input transcript，也不因 speaker-text agreement 自动变成新的 diarization source。

## 11. Per-tag resolver

所有 resolver 输出 `certified | supported | conflicted | insufficient`，不直接输出裸 bool。状态定义如下：

| 状态 | 可执行定义 | 默认 public 映射 |
| --- | --- | --- |
| `certified` | 当前 bool 的所有必需 claim role、applicability、coverage、联合校准和冲突检查均通过，或由另一个 certified claim 按版本化逻辑规则严格蕴含 | 发布 bool |
| `supported` | 至少一条适用证据支持某个值且没有 material conflict，但缺少独立 event 票、guard、coverage 或组合校准中的至少一项 | `null` |
| `conflicted` | 适用且质量合格的来源在对齐后仍对值或底层事件发生 material disagreement | `null`，可进入 targeted acquisition/human review |
| `insufficient` | 没有 material conflict，但因输入、上下文、能力、适用性或质量不足，连单边支持条件也未达到 | `null` |

`abstain` 是把 `conflicted/insufficient` 映射为不发布标签的动作，不是第五种 resolver 状态。额外取证后可以从 `supported/conflicted` 重新进入 resolver，但不能通过简单多数票直接转为 certified。

hypothesis engine 只能移除已经被独立证据反驳的 conflict edge，随后以 `remove_conflict/reenter_resolver` 重新执行正常门禁；它自身永远不能创建 certification edge。只剩一个 `viable` 分支或“没有观察到矛盾”都不等于该分支得到支持。

状态作用于当前 sample 内的具体 claim，而不是整个 sample 的一个总状态：`overlap_exists`、`overlap_timeline`、`speaker_count_exact` 和 `multi_speaker_bool` 可以同时处于不同状态。例如 exact count 可 conflicted，而 `multi_speaker=true` 仍 certified；系统不得输出一个含糊的全局 `timeline_certified=true`。

### 11.1 Text evidence 交叉认证速查

| text 信号 | 实际对照 | 可承担的 role | 能帮助的 claim | 不能推出 |
| --- | --- | --- | --- | --- |
| transcript 内容/lexical 覆盖一致 | MOSS vs selected lexical clock；冲突时 SenseVoice 只提供第三种 agreement pattern；按 normalized lexical unit 对齐 | `lexical_stability_diagnostic`、ASR QA | 漏词/幻觉候选区域、定向重跑位置 | transcript 真值、完整 speech coverage、speaker 数/identity |
| speaker-attributed lexical unit 一致 | MOSS joint track vs Paraformer/Whisper interval x pyannote；复合 comparison 保存传递依赖闭包 | `assignment_consistency_diagnostic` | 暴露 merge/split 或 assignment 不稳定，触发复核 | 正向 change boundary guard；第三张 speaker/change event 票 |
| lexical interval 与 activity 一致 | Paraformer CIF 或通过 gate 的 Whisper interval vs Silero/diarizer speech mask | `lexical_presence_cue`、`coverage_gap_detector` | 某局部存在可识别语音；发现 VAD/diarizer 空洞候选 | 完整 coverage、canonical speech union、negative gate、overlap 或多人 |
| lexical unit 横跨候选边界 | interval 与候选 change/silence collar 冲突 | `boundary_ambiguity_guard` | 阻止自动认证并触发重跑/复核 | 自己提出或正向认证 change point |
| 不同时间机制边界一致 | CIF interval、attention-DTW interval、RNN-T emission point、VAD 能量谷 | `coarse_boundary_diagnostic` | 提高边界复核优先级；point 只做 collar 匹配 | 任意一个方法的时间就是真值 |
| 短响应语义 | “嗯/对/yeah/uh-huh”等 lexical unit + 短时第二 speaker onset | `backchannel_review_trigger` | 标记“可能是 backchannel”并要求重查 canonical floor state | 自己否定 floor transfer 或认证 `speaker_change=false` |
| 跨物理通道重复文字 | 同词序列、近同步 word time + channel 能量/延迟/embedding | `crosstalk_guard` | 同一 speaker 串音或复制通道的候选 | 仅凭文字证明同一人 |
| 两个 stem 的不同 transcript | separation 后分别 ASR | diagnostic | 提高 overlap 复核优先级 | overlap event 票；stem 可能是伪分离 |
| ASR 未输出第二路文字 | 任一单流 ASR 的 absence | 无 | 仅作为 ASR limitation warning | `multi_speaker=false`、`speaker_overlap=false` |

### 11.2 Metadata 交叉认证速查

这里的“交叉认证”是比较不同 evidence group 的底层事件，不是拿同一 timeline 派生出的 metadata 字段互相证明。为兼容现有 metadata，表中保留 `recording_summary` 和 `utterances[]` 字段名，但它们只描述当前 sample：`recording_summary` 不跨 sample 聚合，`utterances[]` 对本输入只有当前 sample 一个元素。

| metadata / 公开标签 | 主证据 | 交叉证据 | 认证时真正比较什么 | 不能作为第二票的字段 |
| --- | --- | --- | --- | --- |
| `recording_summary.speaker_count`、`utterances[].active_speaker_count`、`speaker.multi_speaker` | MOSS + Community-1 的 speaker activity | CAM++/WeSpeaker clean-speech cluster；适用时 Sortformer；已验证物理通道 | 对齐后的 material speaker 集合、第二人有效时长、unmatched cluster | `speakers[]` 长度、通道数、participant roster 单独使用 |
| `segments[]`、`speakers[]` | 两条完整 diarization timeline | clean-speech embedding 和物理通道 | occupancy、speaker permutation alignment、merge/split 冲突 | 从同一 `segments[]` 汇总出的 `speakers[]` |
| `overlap_segments[]`、`utterances[].overlap_duration_sec`、`utterances[].overlap_ratio`、`utterances[].is_overlapped`、`recording_summary.overlap_ratio_speech`、`recording_summary.overlap_ratio_audio`、`speaker.speaker_overlap` | 两个具备 overlap 能力的独立 frame/timeline 来源 | Sortformer 或物理通道作冲突证人；CAM++ 只验证两人不同 | 同一时间区域的 occupancy mask、event IoU、绝对时长和统一分母 ratio | `overlap_segments -> overlap_ratio -> bool` 的内部一致性；VAD；separation stem |
| `recording_summary.speaker_change_points`、`recording_summary.speaker_change_count`、`utterances[].speaker_change_count`、`utterances[].speaker_change`、`speaker.speaker_change` | 两条 timeline 的 canonical floor-transfer 边界 | CAM++ 验证边界两侧 different speaker；VAD 检查 speech/gap；speaker-text 只检查 ambiguity | 一对一 acoustic collar matching、边界两侧 clean speech、A/B identity；text 不贡献正向 boundary 票 | ASR 标点、assignment switch 或 lexical cue 单独使用、静音边界、仅有 different-speaker onset |
| `utterances[].primary_speaker_id`、`utterances[].primary_speaker_coverage` | 对齐后的 consensus activity | CAM++ continuity、close-talk mapping | primary speaker 原始 activity / utterance 总时长 | 单一路线生成的 duration 与 coverage 互证 |
| `recording_summary.dominant_speaker_ratio` | 当前 sample 的 consensus activity | CAM++ continuity、第二 diarizer | dominant speaker 原始 activity / 当前 sample speech union | `speaker_balance` 或 `turn_count` |
| `utterances[].speech_union_duration_sec`、实现扩展 `recording_summary.union_speech_duration_sec` | Silero 或 MarbleNet speech mask | diarizer speech union；ASR lexical activity 只作正向空洞检测 | frame occupancy、coverage、边界 collar；`recording_summary` 扩展字段需在 v2 标准登记，数值仍只覆盖当前 sample | ASR absence 不能缩短 speech union；speech 时长不能证明 speaker 数或 overlap |
| `speakers/segments[].source_channel_id`、`recording_summary.crosstalk_level`、内部 channel-purity evidence | 物理 channel activity | mixed-audio diarizer + 跨通道 embedding/能量差 | channel 到 participant 的唯一性、串音延迟和相对能量 | “一个通道一个 speaker”的先验；由 overlap ratio 派生的 crosstalk 等级 |
| `segments[].text`、内部 lexical-unit/speaker-text tracks | MOSS joint text + 按语言选择的 Paraformer/Whisper | interval lexical units x pyannote/Sortformer compound track；SenseVoice 只提供第三份内容观察 | token/time stability、speaker assignment disagreement、change 边界 ambiguity guard | raw transcript 真值；正向 change boundary 或独立 speaker 身份票 |

### 11.3 `multi_speaker`

认证 true：

- 至少两个适用于 speaker identity 的独立 evidence group 支持 scope 内存在两个不同 speaker；并且
- 第二 speaker 达到 material speech 门槛；并且
- speaker mapping 不只是通道数或 cluster 数，CAM++/physical mapping 至少支持其一。

认证 false：

- 两个在当前数据域达到 negative-certification recall 门槛、且完整覆盖 scope 的独立来源都只发现一个稳定 speaker；并且
- 该 recall 门槛同时在顺序换人和“第二人只出现在 overlap 中”的开发集切片上通过；并且
- 该模型组合通过对应 domain/layout 的联合第二人漏检风险门槛；并且
- speech coverage 足够、没有 max-speaker/model applicability violation；并且
- embedding 不显示稳定的第二 cluster。

如果一个系统判断 2 人、另一个判断 3 人，但二者都可靠地支持至少 2 人，可以认证 public `multi_speaker=true`，同时把 exact `speaker_count` 标为 conflicted。

text 只能验证候选第二 speaker 区域是否包含可识别 speech、或通过跨通道重复词触发 crosstalk guard。连续文字被切到 A/B 不证明两人，不同文字也不证明不同人；ASR silence 更不能认证 `multi_speaker=false`。

### 11.4 `speaker_overlap`

认证 true：

- 至少两个独立 group 在同一局部区域支持 >=2 speaker active；并且
- overlap mask 的 event IoU 或 frame precision/recall 达到校准阈值；并且
- 绝对 overlap 时长和 `overlap_duration / speech_union_duration` 都达到门槛；并且
- speaker identity 证据排除同一 speaker 串音或重复通道。

认证 false：

- 两个在当前数据域达到 overlap negative-certification recall 门槛的独立来源完整覆盖 scope；并且
- 二者均低于 false-side threshold；并且
- 适用的模型组合通过对应 domain/layout 的联合漏检风险门槛；并且
- 同一候选时间区域内没有物理通道并发或其它未解释的 concurrent activity。顺序出现的第二 speaker 不反驳 `overlap=false`。

两条 timeline 只要都输出 `true` 还不够。如果它们标出的 overlap 区域完全不同，只能是 conflicted。

在 MOSS 的短 overlap 召回完成本域校准之前，不能用 `MOSS + Community-1` 认证 `speaker_overlap=false`。此时需要 `Community-1 + Streaming Sortformer`、Community-1 + 当前 sample 音频内已验证的物理通道，或其它两条通过负例门槛的 overlap-capable 路线；若没有，输出 `supported/insufficient -> null`。Sortformer 只有在覆盖完整当前 sample 的 invocation 中，且 `<=4` 上限由当前 sample 的可信证据确定时，才能提供 negative 票。

这里使用的 MOSS、Whisper、Paraformer、SenseVoice 和 Zipformer 都是单流识别：overlap 中只转出一路或漏掉短词是预期 failure mode，不能作为 negative。只有已验证 participant mapping 的独立物理通道各自出现并发 lexical activity 时，text 才能加强 `G7_physical_channel` 的同一张物理证据票；它仍不是新的 lexical overlap 票。

### 11.5 `speaker_change`

按推荐的 floor-transfer 语义，认证 true：

- 两个独立 timeline 在 collar 内给出相同 A -> B transition；并且
- A、B 都在当前 sample 内达到 `min_speaker_activity_sec`，禁止借用 sample 外 activity 或 context；并且
- change 两侧都在当前 sample 内有最小 clean-speech context，不足时返回 `insufficient`；并且
- CAM++ 或其它 identity evidence 支持 A、B 不同；并且
- A 持续讲话、B 只做短 backchannel 的情况没有被误判成 floor transfer。

认证 false：

- 两条 timeline 都在当前数据域达到 floor-transfer negative-certification recall 门槛；并且
- 该模型组合通过对应 domain/layout 的联合 floor-transfer 漏检风险门槛；并且
- 当前 sample scope 覆盖完整，且 sample 内两侧 context 充足；并且
- 两条 timeline 的 primary floor holder 不变；并且
- 没有未解释的 unmatched change point。

边界被 clip、纯 overlap、长静音隔开或 identity 不可判时，应返回 insufficient，而不是 false。

两条 acoustic timeline 已给出同一 A -> B 候选后，`track_moss_joint` 与当前 profile 的 Paraformer/Whisper x pyannote track 在相近位置改变 assignment，只能记作派生一致性 diagnostic：前者复述 MOSS 边界，后者复述 pyannote occupancy，不能反过来充当正向 boundary guard。真正有用的 text guard 是发现 lexical unit 横跨候选静音/transfer collar、时间类型不适用或 assignment ambiguous，从而阻止自动认证并触发复核。短 backchannel 词也只能触发重新检查 floor-state，不能单独否定 transfer。

### 11.6 硬一致性检查

这些约束不增加独立 evidence group：

```text
speaker_overlap=true  => multi_speaker=true
speaker_change=true   => multi_speaker=true
multi_speaker=false   => speaker_overlap=false and speaker_change=false
overlap_duration_sec <= speech_union_duration_sec <= scope_duration_sec
speaker_count == number of material consensus speakers
```

resolver 可以从 certified 前件做单向逻辑派生，例如 certified `multi_speaker=false` 可派生 overlap/change false，certified overlap/change true 可派生 `multi_speaker=true`。派生 record 必须写 `derivation_type="logical_implication"`、父 certification ID 和 rule version；它不能反向支持父 claim，也不能被再次当作模型票，从而避免循环认证。没有 certified 前件时，这些约束只负责拒绝不一致结果。

## 12. Evidence 与认证 artifact schema

以下 artifact 一律以 `sample_id` 作为 scope 和存储主键。示例中的 `recording_id` 只是可选 pass-through provenance：只能来自已有 `sample.native_metadata`，缺失时取 `sample_id`；不得把它加入 raw input、用于跨 sample join 或音频查找。

### 12.1 Evidence record

```json
{
  "evidence_id": "ev_01J...",
  "sample_id": "AMI:ES2005a:utt_00010",
  "recording_id": "ES2005a",
  "claim_type": "speaker_activity",
  "scope": {
    "audio_id": "sample_AMI_ES2005a_utt_00010",
    "start_sec": 0.0,
    "end_sec": 15.0,
    "timebase": "sample",
    "coverage_ratio": 1.0
  },
  "source": {
    "tool": "pyannote_community_1",
    "tool_version": "adapter_v0.1.0",
    "model_id": "pyannote/speaker-diarization-community-1",
    "model_revision": "pinned-revision",
    "checkpoint_sha256": "...",
    "model_family": "pyannote_powerset",
    "independence_group": "G2_pyannote_powerset",
    "license_id": "CC-BY-4.0",
    "license_review_status": "approved"
  },
  "lineage": {
    "parent_audio_id": "sample_AMI_ES2005a_utt_00010",
    "parent_evidence_ids": [],
    "preprocessing": ["decode", "resample_16k_mono"]
  },
  "capabilities": ["speaker_activity", "speaker_count", "overlap_activity"],
  "limitations": ["anonymous_ids", "domain_calibration_required"],
  "applicability": {
    "applicable": true,
    "input_constraints_checked": true,
    "calibration_slice": "zh_en_meeting_nearfield_v1"
  },
  "value": {
    "segments": []
  },
  "quality": {
    "status": "ok",
    "raw_score_available": false,
    "warnings": []
  },
  "artifact_ref": "artifacts/speaker_v2/AMI_ES2005a_utt_00010/evidence/ev_01J.json.gz"
}
```

### 12.2 Speaker-text track record

```json
{
  "track_id": "st_paraformer_pyannote_01J...",
  "sample_id": "AMI:ES2005a:utt_00010",
  "recording_id": "ES2005a",
  "track_type": "asr_lexical_units_x_diarizer",
  "scope": {
    "audio_id": "sample_AMI_ES2005a_utt_00010",
    "start_sec": 0.0,
    "end_sec": 15.0,
    "timebase": "sample"
  },
  "source_artifacts": {
    "asr_evidence_id": "ev_paraformer_units",
    "diarizer_evidence_id": "ev_pyannote_activity",
    "external_timing_evidence_ids": []
  },
  "dependency_artifact_ids": ["ev_paraformer_units", "ev_pyannote_activity"],
  "dependency_groups": ["G12_paraformer_asr", "G2_pyannote_powerset"],
  "timing": {
    "timestamp_method": "native_cif_token_interval",
    "value_type": "interval",
    "unit_type": "character_or_bpe_token",
    "capability_probe_status": "passed"
  },
  "normalization": {
    "normalizer_version": "multilingual_text_norm_v0.1",
    "tokenizer_profile_id": "zh_en_word_align_v1",
    "raw_text_preserved": true
  },
  "assignment": {
    "method": "lexical_unit_interval_x_raw_speaker_occupancy",
    "profile_id": "lexical_unit_speaker_assignment_v1",
    "timestamp_uncertainty_collar_sec": 0.08,
    "unique_speaker_coverage_ratio_min": 0.85,
    "material_competitor_activity_sec": 0.04,
    "source_attributed_asr": false
  },
  "lexical_units": [
    {
      "unit_id": "u_000142",
      "raw_text": "嗯",
      "normalized_text": "嗯",
      "start_sec": 2.14,
      "end_sec": 2.31,
      "asr_confidence": null,
      "speaker_candidates": [
        {"speaker_id": "spk_001", "activity_overlap_ratio": 0.88},
        {"speaker_id": "spk_000", "activity_overlap_ratio": 0.74}
      ],
      "assignment_state": "ambiguous",
      "assigned_speaker_id": null,
      "boundary_ambiguous": false
    }
  ],
  "coverage": {
    "scope_duration_sec": 15.0,
    "asr_timed_unit_duration_sec": 10.8,
    "diarizer_scope_coverage_ratio": 0.995,
    "lexical_unit_count": 142,
    "assigned_count": 121,
    "ambiguous_count": 12,
    "unassigned_count": 9
  },
  "applicability": {
    "applicable": true,
    "language_profile": "zh_zh-en",
    "timing_capability_checked": true
  },
  "quality": {
    "status": "ok",
    "warnings": []
  }
}
```

上述 assignment threshold 只是开发集初值，必须按 lexical-unit duration、语言和 timestamp method 分别校准。对普通 ASR，`assigned` 必须满足 uncertainty collar 内只有一个 material speaker；`activity_overlap_ratio` 最大但存在第二个 material speaker 时仍是 `ambiguous`。

schema 必须用 `track_type + timestamp_method + value_type` 做判别式校验：

| `track_type` | 允许的时间类型 | 允许的 attribution 字段 | 禁止项 |
| --- | --- | --- | --- |
| `asr_lexical_units_x_diarizer` | 通过 gate 的 lexical `interval` | `speaker_candidates`、`assignment_state`、条件式 `assigned_speaker_id` | point/segment timing；存在 material competitor 时仍标 assigned |
| `moss_joint_segments` | `joint_segment_interval` | 仅模型原生 `native_speaker_id`，并标记 `source_attributed_asr=true` | 伪 word interval；把 native attribution 写成独立投影票 |
| `transcript_only` | 无 unit time，或仅保留 source-level region | 无 | 所有 speaker candidates/assignment 字段 |
| `token_emission_diagnostic` | `rnnt_token_emission_point` | 无 | `speaker_candidates`、`activity_overlap_ratio`、`assignment_state`、`assigned_speaker_id` |
| `vad_segment_coverage` | `vad_segment_interval` | 无 | 任意 lexical-unit speaker attribution |
| `forced_alignment_units` | `forced_alignment_token_interval` | 只有显式引用 diarizer 且闭包/quality gate 通过后，才可按第一行规则投影 | 从输入 transcript 或 aligner 本身生成 speaker identity |

除 `moss_joint_segments.native_speaker_id` 这一原生 variant 外，`value_type=point|segment` 时 schema 必须拒绝全部归人字段，而不只是拒绝 occupancy ratio。

`source_artifacts` 中任何非空 `external_timing_evidence_ids` 都必须直接出现在 `dependency_artifact_ids`，并把其递归 groups 并入 `dependency_groups`。例如 SenseVoice `sentence_timestamp` 若来自一个 FSMN-VAD artifact，不能只写 `timestamp_method="vad_segment_interval"` 而漏掉该 VAD edge。

MOSS joint track 使用同一 schema 的另一 variant：`track_type="moss_joint_segments"`，ASR 和 diarizer source 都指向同一个 MOSS evidence，`dependency_groups=["G1_moss_joint"]`，`timestamp_method="joint_segment_interval"`，`source_attributed_asr=true`。它只保存模型原生 segment-level attribution；禁止从 segment 文本均匀切出伪 word interval。

forced-alignment variant 必须使用 `track_type="forced_alignment_units"`，并在 `source_artifacts` 明确保存 `forced_aligner_evidence_id`、`input_transcript_artifact_id` 和可选 `diarizer_evidence_id`。三者及外接 VAD 都进入 `dependency_artifact_ids`；`dependency_groups` 是这些 artifact 的递归并集。缺少 typed transcript edge 时 schema validation 必须失败。

### 12.3 Speaker-text comparison record

track 自身不得内嵌与其它 track 的 agreement，否则会漏掉 comparison 的传递依赖。跨 track 对照单独保存：

```json
{
  "comparison_id": "stc_moss_paraformer_pyannote_01J...",
  "sample_id": "AMI:ES2005a:utt_00010",
  "recording_id": "ES2005a",
  "scope": {
    "audio_id": "sample_AMI_ES2005a_utt_00010",
    "start_sec": 0.0,
    "end_sec": 15.0,
    "timebase": "sample"
  },
  "left_track_id": "st_moss_joint_01J...",
  "right_track_id": "st_paraformer_pyannote_01J...",
  "speaker_alignment_id": "sa_moss_pyannote_01J...",
  "lexical_alignment_id": "la_moss_paraformer_01J...",
  "dependency_artifact_ids": [
    "st_moss_joint_01J...",
    "st_paraformer_pyannote_01J...",
    "sa_moss_pyannote_01J...",
    "la_moss_paraformer_01J..."
  ],
  "dependency_groups": [
    "G1_moss_joint",
    "G2_pyannote_powerset",
    "G4_campplus_identity",
    "G12_paraformer_asr"
  ],
  "dependency_closure_verified": true,
  "calibration_profile_id": "speaker_text_compare_zh_en_v1",
  "inclusion_rule": {
    "scope_intersection_only": true,
    "exclude_assignment_states": ["ambiguous", "unassigned"],
    "exclude_boundary_ambiguous": true,
    "timestamp_methods_allowed": ["joint_segment_interval", "native_cif_token_interval"]
  },
  "measurements": {
    "token_agreement": {"numerator": 126, "denominator": 139, "ratio": 0.906475},
    "unit_in_segment_collar": {"numerator": 121, "denominator": 139, "ratio": 0.870504},
    "speaker_assignment_agreement": {"numerator": 103, "denominator": 121, "ratio": 0.85124}
  },
  "roles_allowed": ["diagnostic", "ambiguity_guard", "review_trigger"],
  "roles_forbidden": ["event_vote", "identity_vote", "positive_change_boundary_guard"],
  "quality": {"status": "ok", "warnings": []}
}
```

`dependency_groups` 是两个 track、speaker alignment 和 lexical alignment 的传递闭包快照；resolver 必须从被引用 artifact 重新计算并核对，不能相信调用方手填。`speaker_assignment_agreement` 只是派生一致率，不是 attribution accuracy，也不能隐式增加 event vote。

### 12.4 Certification record

```json
{
  "tag_path": "speaker.speaker_overlap",
  "unit_id": "AMI:ES2005a:utt_00010",
  "value": true,
  "status": "certified",
  "rule_version": "speaker_overlap_cert_v0.1",
  "semantics_version": "speaker_semantics_v2.0",
  "calibration_profile_id": "zh_en_meeting_nearfield_v1",
  "postprocessing_profile_ids": ["moss_activity_v1", "pyannote_activity_v1"],
  "evidence_edges": [
    {
      "artifact_type": "evidence",
      "artifact_id": "ev_moss_overlap",
      "artifact_claim": "overlap_activity",
      "role": "event_vote",
      "dependency_groups": ["G1_moss_joint"]
    },
    {
      "artifact_type": "evidence",
      "artifact_id": "ev_pyannote_overlap",
      "artifact_claim": "overlap_activity",
      "role": "event_vote",
      "dependency_groups": ["G2_pyannote_powerset"]
    },
    {
      "artifact_type": "evidence",
      "artifact_id": "ev_campplus_identity",
      "artifact_claim": "different_speaker",
      "role": "identity_guard",
      "dependency_groups": ["G4_campplus_identity"]
    },
    {
      "artifact_type": "evidence",
      "artifact_id": "ev_silero_speech",
      "artifact_claim": "speech_coverage",
      "role": "coverage_guard",
      "dependency_groups": ["G6_silero_vad"]
    }
  ],
  "vote_groups_by_claim": {
    "overlap_activity": ["G1_moss_joint", "G2_pyannote_powerset"],
    "different_speaker": ["G4_campplus_identity"],
    "speech_coverage": ["G6_silero_vad"]
  },
  "conflicting_artifact_edges": [],
  "measurements": {
    "overlap_event_iou": 0.833,
    "overlap_frame_f1": 0.91,
    "canonical_speech_mask_evidence_id": "ev_silero_speech",
    "canonical_speech_union_duration_sec": 3.02,
    "overlap_by_source": [
      {
        "evidence_id": "ev_moss_overlap",
        "overlap_duration_sec": 0.2,
        "native_speech_union_duration_sec": 3.0,
        "native_overlap_ratio": 0.066667,
        "canonical_overlap_ratio": 0.066225,
        "scope_coverage_ratio": 1.0
      },
      {
        "evidence_id": "ev_pyannote_overlap",
        "overlap_duration_sec": 0.24,
        "native_speech_union_duration_sec": 3.0,
        "native_overlap_ratio": 0.08,
        "canonical_overlap_ratio": 0.07947,
        "scope_coverage_ratio": 1.0
      }
    ],
    "material_speaker_activity_sec": [2.4, 0.8],
    "speaker_mapping_coverage": 0.94
  },
  "resolved_thresholds": {
    "min_speaker_activity_sec": 0.1,
    "min_overlap_duration_sec": 0.1,
    "public_overlap_ratio": 0.05,
    "certify_false_ratio_max": 0.04,
    "certify_true_ratio_min": 0.06,
    "overlap_event_iou_min": 0.5,
    "overlap_frame_f1_min": 0.8,
    "scope_coverage_ratio_min": 0.98,
    "max_uncovered_gap_sec": 0.08,
    "identity_threshold_profile_id": "campplus_zh_en_meeting_v1"
  },
  "threshold_bundle_sha256": "...",
  "derivation_exact": true,
  "calibrated_probability": null,
  "warnings": []
}
```

certification edge 可以引用原子 evidence、speaker-text track 或 comparison artifact，但必须携带其传递 `dependency_groups`，并由 resolver 对照 artifact store 重算闭包。只有 claim capability 与 role 都允许的 group 才进入 `vote_groups_by_claim`；compound text comparison 只能承担其 `roles_allowed` 中的 diagnostic/ambiguity guard，不能提供 event vote。

这里的 `event_vote`、`identity_guard` 和 `coverage_guard` 必须分开计数。CAM++ 在示例中只支持 `different_speaker`，绝不能被计为第三张 overlap event 票。所有 resolved threshold 和模型后处理配置都必须通过 profile ID + hash 可复现；示例数值只是前述开发集初值。

`confidence=1.0` 不能再表示“派生代码执行成功”。建议拆成：

- `derivation_exact`：在给定输入 timeline 下公式是否确定性完成。
- `calibrated_probability`：在对应 domain calibration slice 上的经验概率，可空。
- `status`：最终认证状态，公开输出主要依据。

### 12.5 与当前 metadata 的兼容边界

v2 evidence 与当前 metadata 都使用 sample-relative time。v2 从入口开始只接受当前 utterance 音频，规范时间范围固定为 `[0, duration_sec]`，不存在 recording-to-sample 的时间转换。真正的兼容差异是：v2 保存多来源 evidence 和逐 claim 状态，而当前 metadata 要求单一 `primary_route` 及完整的 `segments[]/speakers[]`；二者不能靠改字段名直接等价。

建议分成两个 artifact：

1. `fusion_artifact_v2` 保存所有 sample-relative source timeline、对齐、claim event 和 certification，是 v2 的权威记录。
2. 可选 `compat_metadata` 只在满足导出条件时映射到修订后的 speaker metadata standard；它不是 public tag 的权威来源。

兼容 adapter 必须：

- 直接验证 segment、overlap、change 和 utterance 时间都属于 sample-relative `[0, duration_sec]`；禁止引入 `sample_origin_recording_sec` 或任何 recording offset。
- 保留 `metadata_version`、`sample_id`、`input_kind`、`duration_sec` 等必需字段；兼容 schema 需要 `recording_id` 时，只能传递上述 opaque provenance，缺失时取 `sample_id`。新版本暂定 `speaker_diarization_v2.0`，最终值由标准评审确定。
- 在现有必填字段仍保留的情况下令 `primary_route="evidence_fusion_v2"`；具体模型 route 只存在 evidence artifact 中。
- 给完整 sample timeline、speaker count、overlap timeline 和 change timeline 分别保存 claim status。若当前 schema 不能表达 partial/conflicted timeline，则禁止导出看似 `quality.status=ok` 的兼容 metadata。
- 只有完整 sample timeline 已认证时，才把 consensus `segments[]/speakers[]` 当作 canonical metadata；否则可单独发布已认证的三个 public bool，但不能用任意“最佳模型”伪装成 consensus timeline。
- `selected_text_export_track_id` 必须由 versioned language/profile policy 在推理前确定，再经过 applicability/quality gate；禁止因它更贴合某个待认证 speaker label 而事后选择。普通 ASR track 只导出 uncertainty collar 内恰好一个 material speaker active 的 `assigned` lexical units；`ambiguous/unassigned` 留在受控 artifact，不能为凑完整句强塞给某个 speaker。MOSS native segment text 只能按其 joint source 原样标注 provenance，不能伪装成独立归因真值。
- raw input `sample.text.transcript` 永不被模型 transcript 覆盖；若需要保留 MOSS、Paraformer、Whisper 或 SenseVoice 全文，分别通过 evidence ID 引用并服从 retention/delete policy。

v2 public tags 应直接由 certification record 映射，不再经过当前 `public_results_from_metadata` 对单条 timeline 的二次派生。上线前必须先升级 metadata standard、output/run manifest 和 consumer；raw `corpus/sample` input schema 保持不变。本任务不修改当前生产标准或 pipeline。

## 13. 证据协作示例

### 13.1 一致的 overlap

假设一个 3.3 秒 target：

| 来源 | 结果 |
| --- | --- |
| MOSS | A: 0.00-2.40，B: 2.20-3.00，native overlap ratio 0.067 |
| Community-1 | X: 0.02-2.42，Y: 2.18-3.02，native overlap ratio 0.080 |
| CAM++ | A/X 相似，B/Y 相似；A 与 B 为 different speaker |
| Silero | 0.00-3.02 为 speech union，coverage 正常 |

匿名 ID 对齐后，两条 overlap 区域 IoU 高；按同一个 Silero canonical speech union 重算后 ratio 分别约 0.066/0.079，均越过 true-side 6%；identity guard 支持两人，因此：

```text
multi_speaker = true / certified
speaker_overlap = true / certified
speaker_change = 取决于是否发生 floor transfer，不能从 overlap 自动推出
```

### 13.2 Bool 相同但事件位置冲突

假设 MOSS 和 Community-1 都输出 `speaker_overlap=true`，但 MOSS 的区域是 `2.20-2.40 s`，Community-1 的区域是 `0.60-0.82 s`，两者 event IoU 为 0。此时不能把两个 true 当作两票一致：

1. 在覆盖完整当前 sample 的 invocation 适用条件满足时，让 Streaming Sortformer 输出第三条 frame mask。
2. 若 Sortformer 与 MOSS 在 `2.20-2.40 s` 对齐，可以由 `G1 + G3` 认证 public overlap true，但仍保留 Community-1 的额外区域冲突。
3. 若 Sortformer 与二者都不对齐，或覆盖完整当前 sample 的 invocation 未通过 4-slot applicability gate 且无当前 sample 音频内的物理通道证据，则 `conflicted -> null`。

### 13.3 MOSS 和 pyannote 的 speaker count 冲突

MOSS 把一个 speaker 分成 A/B，pyannote 只发现 X：

1. 在 MOSS A/B 的非重叠干净区域提取 CAM++ embedding。
2. 如果 A/B 高度相似，且适用的 Streaming Sortformer/物理通道/独立训练 OSD 没有双人活动，记录 `moss_cluster_split`，`multi_speaker` 不能由 MOSS 单独认证。
3. 如果 A/B embedding 明显不同，且上述独立证人在同一位置发现第二人，pyannote 的单人结果成为冲突证据。
4. 仍无法解释时输出 `conflicted -> null`，而不是三模型简单投票。

### 13.4 Negative overlap 的认证

假设 Community-1 与 Streaming Sortformer 都完整覆盖 scope，在本域 overlap-recall 校准中分别达到负例门槛，二者的组合也通过 joint miss-risk 门槛，并且 canonical overlap ratio 都低于 4%；Silero 显示 speech coverage 完整，物理通道也没有未解释的同时活动。此时可以发布：

```text
speaker_overlap = false / certified
```

如果只有 Community-1 未检出，而 MOSS 尚未通过短 overlap 负例校准，则最多是 `supported`，对外仍为 `null`。

### 13.5 Separation 只生成新输入

候选 overlap 片段经 SepFormer 得到 stem 1/2：

- 两个 stem 都有 VAD speech 且 CAM++ 不同，默认也只产生 diagnostic evidence，用于调整复核优先级，不能增加 overlap event 票。
- 一个 stem 无 speech、两个 stem embedding 相同或出现严重残留，不能反向证明无 overlap。
- certification record 必须引用 raw overlap 候选和 separation parent lineage。
- 只有在完整的 `separation -> VAD/embedding` 变换链完成专项校准后，它才能承担有限 support/guard role；整条链仍是一个继承 parent 的 derived group，不是独立声学证人。

### 13.6 用 speaker text 检查 change ambiguity 与 backchannel

真 floor transfer 示例：

```text
MOSS joint:
  A: [3.10, 4.08] "这个方案的成本是..."
  B: [4.24, 5.30] "我补充一个风险"

Paraformer tokens x pyannote:
  units before 4.12 -> projected A
  units after 4.22  -> projected B

CAM++:
  clean context A vs B -> different speaker
```

两条 acoustic timeline 的 A -> B event 已在 collar 内匹配，CAM++ 只承担 identity guard。复合 text track 把 matched lexical units 的 assignment switch 定位在同一边界，只是预期的派生一致性，因为 assignment 分别来自 MOSS/pyannote 自己的 occupancy；它不能升级认证状态。若某个高质量 lexical interval 横跨声学系统声称的静音 gap 或 transfer collar，则记为 `boundary_ambiguity_guard` 失败并进入复核。

backchannel 示例：A 在 `[8.00, 10.00]` 持续讲话，B 在 `[8.70, 8.90]` 说“嗯”，selected ASR 也只在该短区间匹配到 backchannel lexical unit。此时记录 B 的 `different_speaker_onset`，floor-state 继续保持 A。lexical cue 只触发“检查是否误把 B onset 当 change”的规则；`speaker_change=false` 仍必须由 acoustic floor-state 和 negative calibration 认证，不能只靠“嗯”这个词。

类似地，两个 close-talk channel 在相同时间转出几乎相同的长词序列，只是 crosstalk 候选；只有再结合 channel 能量差、延迟和 same-speaker embedding，才可阻止“两个通道 = 两个人”的假阳性。

## 14. Calibration 与小型融合模型

Phase 1 先使用显式规则，因为各模型 raw score 未统一校准。收集足够开发集后，可以训练很小的 claim-specific resolver：

- logistic regression、isotonic calibration 或受约束的小型 GBDT；
- 输入仅为来源分数、带完整 dependency closure 的 acoustic/text diagnostics、lexical-unit assignment ambiguity、duration、SNR、layout、language、coverage 等结构化特征；由 diarizer occupancy 派生的 assignment switch 不得作为同一 diarizer change boundary 的正向重复特征；
- 分别训练 `multi_speaker`、`overlap`、`change`，不使用一个通用黑箱；
- 模型输出只帮助选择 `certified/supported/conflicted/insufficient`，不能绕过硬逻辑约束；
- negative profile 必须按具体模型 revision 组合评估 joint false-negative risk，不能用两个单模型 recall 的乘积替代；
- text feature 只能影响已声明的 guard/复核决策，不能学习词汇内容到 speaker identity 的捷径；训练和报告必须按语言分层；
- 必须报告 risk-coverage curve，而不是只报告总体 accuracy。

这个 resolver 是证据融合器，不是新的独立 evidence group。

## 15. 评测设计

### 15.1 数据集

| 数据 | 用途 | 注意事项 |
| --- | --- | --- |
| AMI IHM | close-talk channel、人工 speaker/time、串音压力测试 | meeting-level split；人工 annotation 只作 gold，不喂 resolver |
| AMI SDM/Mix-Headset | far-field/mixed diarization 和 overlap | 与 IHM 同会议时避免跨 split 泄漏 |
| AliMeeting | 中文会议、多说话人和 overlap | 校准中文/远场域 |
| AISHELL-4 | 中文远场会议 | 验证跨 corpus 泛化 |
| VoxConverse | 多域 speaker diarization | overlap 分布与会议数据不同 |
| LibriMix/WHAMR | 可控 overlap、噪声和混响 stress test | synthetic 只能补充，不能替代真实会议；检查数据许可 |

### 15.2 指标

底层 timeline：

- DER/JER，分别报告 include/exclude overlap 和 collar 设置。
- speech detection error、speaker confusion、speaker count MAE/accuracy。
- overlap frame precision/recall/F1、event IoU、duration MAE。
- change point precision/recall/F1，采用一对一 collar matching。

text/speaker attribution：

- 有独立 reference 时报告 WER/CER、speaker-attributed WER 或 word-speaker attribution accuracy。
- 无 reference 时只报告 MOSS/selected-ASR token agreement、matched-unit coverage、unit-in-MOSS-segment-collar ratio 和 assignment disagreement；这些是稳定性指标，不得命名为 accuracy。SenseVoice 第三观察只能报告 pairwise agreement pattern，不能报告“多数正确率”。
- 按 overlap/non-overlap、change collar、backchannel、language、lexical-unit duration 和 `timestamp_method` 分层报告 `assigned/ambiguous/unassigned` 比例。
- 单独评估 text guard 对 change false positive、crosstalk false positive 和 VAD/diarizer 空洞候选检出的增益与误导率；ASR absence 不进入 negative coverage 指标。

公开标签：

- 每个 bool 的 precision、recall、F1、confusion matrix。
- `certified` 子集 precision 和 coverage。
- risk-coverage curve、abstention rate、conflict catch rate。
- 每个模型组合的 joint false-negative rate、error coincidence，以及 negative certification violation rate。
- 按 language、layout、duration、SNR、speaker count、overlap ratio 分层。

工程指标：

- 每小时音频 wall time、GPU/CPU memory、模型下载体积。
- 每个 profile 的平均工具调用数、额外 acquisition 次数和 artifact 大小。
- 每增加一个工具带来的增益、成本和误导率。

### 15.3 必做消融

1. MOSS only。
2. MOSS + Silero。
3. MOSS + Community-1。
4. MOSS + Community-1 + CAM++。
5. MOSS + Community-1 + CAM++ + language-selected Paraformer/Whisper speaker-text track。
6. lexical 冲突样本上有/无 SenseVoice 第三观察；禁止用 2/3 majority 代替 reference。
7. 有/无 lexical alignment、boundary-ambiguity/backchannel/crosstalk guards。
8. always-on 所有工具 vs evidence-gap conditional acquisition。
9. claim-specific resolver vs 简单 majority vote。
10. 有/无 targeted separation。
11. 完整 utterance 音频取证 vs 仅在 sample 内 targeted crop 取证。
12. 有/无原始 activity immutable 约束。
13. 有/无 native oracle，后者只作为 audit 上界，不能作为生产成绩。
14. 有/无 claim-local hypothesis branching；比较冲突解决率、错误认证率和额外调用成本。
15. 按 frozen prediction 的区分能力选工具 vs 固定工具顺序；两者使用相同模型与调用预算。

## 16. Failure injection 测试

除真实数据外，需要固定以下合成单测和回归样本：

- `A:0-1, B:1-1.2, A:1.2-2`，确保 turn merge 不制造 overlap。
- change point 在 utterance 边界外，确保不会出现 `multi_speaker=false, change=true`。
- 第二 speaker 只有数毫秒边界尾音。
- 5% overlap threshold 两侧和灰区。
- 同一 speaker 串到两个 close-talk channel。
- 两个不同 speaker 同方向、同一 speaker 强混响多方向。
- 音乐、人群噪声、电视背景声被误判为 speaker。
- 两个 diarizer bool 相同但事件位置完全不同。
- 只有一条 overlap event 票 + CAM++ identity guard，确保 CAM++ 不被误计为第二张 overlap 票。
- 两个单模型 negative recall 合格但缺少 joint profile，确保 false 不能 certified。
- 当前 sample 可能超过 4 人或没有可信的 sample-level `<=4` 上界，确保 Sortformer 不提供完整 sample 的 negative 票。
- 同一片段以 fresh-state crop 调用 Sortformer 时，只允许形成 local positive witness。
- 只有 0.5 秒短 utterance，embedding context 不足。
- separation 产生一个空 stem 或两个同 speaker stem。
- exact speaker count conflicted 但 `multi_speaker=true` certified，确保没有全局 timeline status 泄漏。
- v0.1 与 v2 semantics 混入同一 manifest，确保发布校验直接失败。
- 模型输出 timestamp 越过 `[0, duration_sec]`，确保 schema validation 拒绝，或按已登记策略裁剪并产生 warning；derived crop 的局部时间必须正确回映到 sample-relative time。
- 同一 speaker 连续说出不同主题/语言，确保 text 不制造 speaker split。
- 两个 speaker 顺序说出相同短句，确保 text 不制造 same-speaker merge。
- selected ASR 在 VAD silence 中 hallucinate lexical unit，确保只触发 gap conflict 而不扩张 canonical speech mask。
- overlap 中 ASR 只输出一路文字，确保不能用于认证 overlap false。
- lexical interval 的主要覆盖者是 A、但 collar 内存在 material B activity，确保仍为 `ambiguous` 而不是按占优者强制归人。
- lexical unit 横跨 change/overlap collar，确保保存 `boundary_ambiguous` 而不是强制归人。
- 同一 Whisper words 投影到两条 timeline，确保 dependency graph 只保留一个 `G9_whisper_asr`。
- SenseVoice 只有 VAD sentence interval，确保不能生成 word-level speaker track 或任何 assignment 字段；Zipformer emission point 也不能出现 speaker candidates、assigned speaker 或 occupancy fraction。
- comparison artifact 漏掉 MOSS、speaker alignment 或 lexical alignment 的任一 dependency group，确保 schema validation 失败。
- track 使用 external VAD 但未把其 evidence ID 加入 dependency closure，确保 schema validation 失败。
- forced-alignment track 缺少 `input_transcript_artifact_id`，或只登记 aligner group 而漏掉 transcript closure，确保 schema validation 失败。
- diarizer-derived assignment switch 与该 diarizer change point 同时进入 resolver，确保不会重复贡献正向 boundary 权重。
- backchannel lexical cue 与 acoustic floor-state 冲突，确保 text 触发复核而不覆盖声学 event。
- H1/H2 都被独立证据反驳，确保进入 `H_other/conflicted`，不能选择“更接近”的一个分支。
- H1 被淘汰而 H2 仅未被反驳、且缺少 event/identity/coverage 任一认证角色，确保仍为 `supported/null`。
- validator 与被检验假设共享 dependency group，或使用 ASR x pyannote track 验证 pyannote，确保检验被拒绝且不改变分支状态。
- 同 checkpoint rerun、参数变体或 separation 后运行原模型，确保继承 parent lineage，不能伪装成独立 validator。
- 先获得 targeted evidence、再尝试改写 prediction/falsifier，确保 versioned template freeze 校验失败。
- local crop 未检测到第二人/overlap，确保不能认证 full-sample false；单流 ASR 未输出第二路同样不能淘汰正向分支。
- 一个分支没有发现矛盾但没有正向支持，确保 hypothesis 状态仍为 `viable`，claim 不升级。
- 构造 `overlap -> multi_speaker -> overlap` 或 `change -> multi_speaker -> change` 的派生环，确保 resolver 拒绝循环认证。
- change 位于 sample 边缘且当前 sample 内上下文不足，确保直接 `insufficient`；尝试读取相邻 utterance、跨 sample speaker ID/embedding/text 必须被 input contract 拒绝。
- 假设取证耗尽 budget 且多个分支仍 viable，确保输出 `null` 并保存 termination reason。

## 17. 实现边界与目录建议

不直接重构当前 `metrics.py`，先并行实现 side-by-side v2：

```text
tagger/tools/speaker_v2/
  contracts.py              # claim/evidence/certification schema
  model_registry.py         # model, revision, license, lineage, capability
  evidence_store.py
  timebase.py
  speaker_alignment.py
  lexical_alignment.py      # multilingual token/time alignment
  speaker_text_tracks.py    # ASR x diarizer compound evidence
  speaker_text_compare.py   # dependency-closed cross-track diagnostics
  floor_state.py            # canonical floor-holder state machine
  applicability.py          # invocation-level capability gates
  calibration_registry.py   # single-model and joint-negative profiles
  planner.py
  conflict_hypotheses.py    # claim-local templates, frozen predictions, branch state
  tools/
    moss.py
    pyannote.py
    sortformer.py
    silero_vad.py
    campplus.py
    channel_activity.py
    separation.py
    whisper_asr.py
    paraformer_asr.py       # zh/zh-en native CIF token intervals
    sensevoice_asr.py       # conditional content observation; license gated
    zipformer_asr.py        # conditional RNN-T emission-point audit
    funasr_forced_align.py  # conditional post-consensus alignment
  resolvers/
    speech_coverage.py
    sample_timeline.py
    multi_speaker.py
    overlap.py
    speaker_change.py
    text_guards.py
  export_compat_metadata.py
  audit.py

tagger/pipelines/speaker_evidence.py
scripts/run_speaker_evidence_v2.py
```

内部 artifact 建议：

```text
artifacts/speaker_v2/<sample_id>/
  evidence/<evidence_id>.json.gz
  audio/<derived_audio_id>.wav
  alignments/<alignment_id>.json.gz
  speaker_text/<track_id>.json.gz
  speaker_text_comparisons/<comparison_id>.json.gz
  hypotheses/<case_id>.json.gz
  certifications/<sample_id>.json.gz
  fusion_artifact_v2.json.gz
  compat_metadata.json.gz        # optional; only when export contract passes
```

public output 继续只有三个 bool/null，不暴露工具、证据、confidence 或路径。

## 18. 分阶段落地

### Phase 0：语义与合规

- 决定 `speaker_change` 是 floor transfer 还是 different-speaker onset。
- 固定 material activity、overlap threshold、gray zone 和 collar 的初始值。
- 升级 [`speaker_metadata_standard.md`](../speaker_metadata_standard.md) 与 output/run manifest，禁止新旧语义共用无版本 tag contract；raw input schema 保持不变。
- 确定模型 transcript/lexical-unit artifact 的 PII、访问、加密和 retention/delete policy。
- 确认 pyannote、Streaming Sortformer、Paraformer-zh、条件 SenseVoiceSmall/Zipformer/`fa-zh` 具体 artifact 的许可证是否可接受。
- 建立 model registry，记录 checkpoint revision、SHA256、license 和 telemetry 设置。

### Phase 1：离线最小闭环

- 接入 MOSS、Silero、Community-1、CAM++，以及按语言选择的 Paraformer-zh/Whisper lexical clock。
- 所有模型独立处理同一个 utterance-level sample，保留各自的 sample-relative raw timeline。
- 实现 multilingual lexical/time alignment、MOSS joint track、selected-ASR x pyannote track、dependency-closed comparison，以及 lexical presence/boundary ambiguity/backchannel/crosstalk guards；默认不让 text 增加 speaker event 票。
- 实现 timebase、Hungarian speaker alignment、三个 rule-based resolver。
- 实现 speaker count、overlap location、change/backchannel 三类确定性冲突模板；预测和 falsifier 在取证前冻结，分支结果只能回到原 resolver。
- 实现 `certified/supported/conflicted/insufficient` 和 tags-only 映射。
- 实现 sample-relative `compat_metadata` adapter；完整 sample timeline 未认证时验证其拒绝导出行为。
- 在 AMI 小集合上完成 failure injection 与第一轮阈值校准。

### Phase 2：跨域与条件调用

- 加入 AliMeeting/AISHELL-4/VoxConverse。
- 分 layout/language/quality 做工具可靠性表。
- 分语言、timestamp method 和 overlap/change 场景校准 lexical-unit assignment、ASR agreement 与 text guard 的误导率。
- 在中文 lexical 冲突集评估 SenseVoiceSmall 第三观察；只要求其 text capability 与 license gate，不再假设 token timestamp。另行评估 Zipformer 粗时间，以及候选 transcript 通过 stability gate 后的 `fa-zh` forced alignment。
- 实现 evidence-gap planner、成本预算和 targeted re-run；在 frozen predictions 上按预期区分能力选择下一项独立工具，而不是按固定顺序全跑。
- 接入 Streaming Sortformer v2；按 invocation 执行 4-slot gate，并区分覆盖完整 sample 的证据与 sample-local fresh-state positive witness；若 v2.1 许可获批，做同条件替换实验。
- 比较 always-on 与 conditional acquisition。

### Phase 3：困难样本

- 加入 SepFormer、当前 sample 音频内的 array DOA/物理通道规则和人工复核升级路径。
- 评估 CAT-Net 自训练 OSD 是否能替代 gated OSD。
- 建立人工复核 UI，直接播放原音频和冲突时间范围。

### Phase 4：生产决策

- 固定可商用模型集合和版本。
- 用 held-out corpus 确定 certification precision/coverage 门槛。
- 只有通过 calibration、license、资源和隐私审查的 profile 才进入批量数据打标。

## 19. 需要产品确认的四个决定

1. 是否整体采用 v2 语义：material speaker activity、overlap 的绝对+相对双门槛，以及 floor-transfer change。
2. public output 是否严格只发布 `certified`，还是允许 `supported` 也输出 bool。
3. pyannote gated checkpoint、Sortformer 目标版本，以及 Paraformer/SenseVoice/Zipformer/`fa-zh` 具体权重的使用、attribution 与许可证条件是否可接受。
4. 内部是否保存完整模型 transcript/lexical units，还是只保存对齐指标和受限片段；对应 retention/PII/delete 策略是什么。

输入粒度不是待确认项：严格按现有 utterance-level sample 处理，只读取 `sample.audio.path`，不增加字段或获取相邻音频。其余建议的默认答案是：整体采用并显式版本化 v2 语义；只发布 certified；接受并登记 pyannote 条件、Sortformer 先用 CC-BY-4.0 的 streaming v2；中文 lexical clock 优先 Paraformer-zh，多语用 Whisper base，SenseVoice/Zipformer/`fa-zh` 先隔离评测；speaker text 只存受控内部 artifact，并设置短于声学 evidence 的 retention。

## 20. 参考资料

访问日期为 2026-08-10 至 2026-08-11。

- [Audio-Mind: An Auditable Agentic Framework for Audio Understanding](https://arxiv.org/abs/2605.28480v1)
- [Audio-Mind official repository](https://github.com/DELTA-DoubleWise/Audio-Mind)
- [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
- [MOSS-Transcribe-Diarize model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote legacy overlapped-speech-detection model card](https://huggingface.co/pyannote/overlapped-speech-detection)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [CAM++ ModelScope model](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [NVIDIA NeMo speaker diarization documentation](https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/speaker_diarization/intro.rst)
- [NVIDIA Streaming Sortformer 4spk v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
- [NVIDIA Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [NeMo MSDD removal commit](https://github.com/NVIDIA-NeMo/Speech/commit/d9e94a163233bdb74caffac461970723fedea67a)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [NVIDIA Multilingual MarbleNet VAD v2.0](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0)
- [NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Whisper word timestamp implementation](https://github.com/openai/whisper/blob/main/whisper/timing.py)
- [SenseVoice official repository](https://github.com/QwenAudio/SenseVoice)
- [SenseVoiceSmall model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [FunASR SenseVoice timestamp capability PR](https://github.com/modelscope/FunASR/pull/3414)
- [Paraformer-zh model card](https://huggingface.co/funasr/paraformer-zh)
- [FunASR Paraformer implementation](https://github.com/modelscope/FunASR/blob/main/funasr/models/paraformer/model.py)
- [Sherpa-ONNX streaming Zipformer zh-en model](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20)
- [Sherpa-ONNX online recognizer result contract](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-recognizer.h)
- [FunASR `fa-zh` model card](https://huggingface.co/funasr/fa-zh)
- [ModelScope speech timestamp prediction model](https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline)
- [FunASR Model Open Source License Agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)
- [SenseVoiceSmall license clarification](https://github.com/QwenAudio/SenseVoice/issues/334#issuecomment-5083546605)
- [SpeechBrain SepFormer WHAMR](https://huggingface.co/speechbrain/sepformer-whamr)
- [Asteroid source separation toolkit](https://github.com/asteroid-team/asteroid)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [CAT-Net overlap detection](https://github.com/MANARA-Lab-UM6P/CAT-Net)
- [pyannote legacy OSD removal commit](https://github.com/pyannote/pyannote-audio/commit/93ad8b90ca7c7c7fbf82e39e4a5a005543bda66e)
