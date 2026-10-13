# Sure-Tagger 多说话人证据协作 Pipeline（汇报版）

> 日期：2026-08-11
>
> 目标：为 utterance 生成 `speaker.multi_speaker`、`speaker.speaker_overlap`、`speaker.speaker_change`，证据不足时输出 `null`。
>
> 输入约束：保持现有 utterance-level `sample` 不变。`sample.audio` 只使用现有的 `path` 字段，整份文件就是当前 utterance。所有工具只处理这份音频，不拼接其它 sample，不向 recording 级扩展。

> 选型边界：本页是候选 fusion baseline 的汇报版，表中“默认来源”尚不代表模型专长已获得实验证明。模型职责以 [`evaluation/leaderboard_evaluation_plan_20260812.md`](evaluation/leaderboard_evaluation_plan_20260812.md) 的独立能力榜和组合消融结果为准。

## 1. 核心思路

旧方案是“选择一个模型生成 timeline，再由同一 timeline 派生全部标签”。新方案参考 AUDIO-MIND，改成“多个小模型分工取证，按标签融合证据”：

- MOSS 和 pyannote 独立观察底层 speaker event。
- CAM++ 判断两个干净片段是否为不同 speaker。
- Silero VAD 检查语音覆盖和边界质量。
- Paraformer/Whisper 提供独立文本及时间线，检查漏检、边界歧义、backchannel 和串音。
- 发生冲突时先构造 claim-local 的 `H1/H2/H3` 假设，推导可证伪预测，再按区分能力调用 Sortformer、当前 sample 音频内的物理通道或 separation 等额外工具。
- 不做简单多数投票；先认证底层事件，再确定性生成公开标签。

## 2. Pipeline

<figure class="pipeline-figure">
  <img src="speaker_evidence_pipeline_brief_assets/pipeline.png" alt="Sure-Tagger utterance-level 多说话人证据协作流程图">
  <figcaption><strong>图 1｜多说话人证据协作 Pipeline。</strong>所有模型观察同一个 utterance sample；蓝色为底层事件票，绿色为身份与 claim 检查，黄色为覆盖校验，红色为文本歧义与复核校验。发生冲突后先进入紫色的假设－预测－检验节点，再按区分能力补充独立证据；无法认证时输出 <code>null</code>。</figcaption>
</figure>

实际执行分六步：

1. 直接读取现有 utterance-level `sample.audio.path`；不增加 `start_sec/end_sec` 等 input 字段，也不获取相邻 utterance。
2. 对同一份 `sample.audio` 并行生成 MOSS、pyannote、Silero 和独立 ASR 证据。
3. 把所有时间统一为相对当前 sample 音频起点的 `[0, duration_sec]`，只在 sample 内对齐匿名 speaker ID 和 lexical units。
4. 分别解析 `multi_speaker`、`speaker_overlap`、`speaker_change`，不共享一个总置信度。
5. 对具体 claim 和局部事件构造“来源 A 的局部解释正确、来源 B 的局部解释正确、双方都不完整”分支；先冻结可证伪预测，再选择不继承冲突来源的独立工具检验。
6. 分支排除后重新进入正常 resolver；仍无法区分，或剩余分支未达到原有认证门槛，则 abstain，公开值为 `null`。

## 3. 哪些证据形成交叉验证

交叉验证不是“模型数量”，而是不同证据角色共同满足一个 claim。

| 证据角色 | 默认来源 | 回答的问题 | 是否算底层事件票 |
| --- | --- | --- | --- |
| `event_vote` | MOSS、pyannote；冲突时可用 Sortformer/当前 sample 音频内的物理通道 | 同一时间是否发生第二 speaker、overlap 或 floor transfer | 是 |
| `identity_guard` | CAM++；可选 ECAPA/可信物理映射 | 候选 A、B 是否确实为不同 speaker | 否，只验证身份条件 |
| `coverage_guard` | Silero VAD | scope 是否被完整检查，speech union 和边界是否可信 | 否，只验证覆盖条件 |
| `ambiguity/review_guard` | Paraformer/Whisper speaker-text track | 是否存在 lexical 空洞、跨边界 token、backchannel 或串音风险 | 否，只能阻止误认证或触发复核 |

三类公开标签的认证关系如下：

| 标签 | 主要交叉验证 | 认证重点 |
| --- | --- | --- |
| `multi_speaker=true` | MOSS speaker activity + pyannote speaker activity + CAM++ | 两条 timeline 都发现 material 第二 speaker，CAM++ 支持两段来自不同人 |
| `speaker_overlap=true` | 两条 overlap-capable timeline + CAM++ + Silero | 两个模型必须在同一时间区域发现双人 activity；比较 event IoU、绝对时长和统一分母 ratio |
| `speaker_change=true` | 两条 timeline 的 canonical A -> B floor transfer + CAM++ | change point 在 collar 内匹配，边界两侧均有干净语音且 A/B 不同；不能把短 backchannel 当换人 |
| 三个标签的 `false` | 两条通过本域负例校准的完整来源 + coverage guard | “没有检测到”不等于 false；必须校准模型组合的联合漏检风险 |

以下情况看似一致，但不算独立交叉验证：

- 同一 timeline 派生出的 `segments -> speakers -> summary -> bool`。
- MOSS diarization 与 MOSS transcript，因为二者来自同一次 joint generation。
- 同一 Whisper checkpoint 的不同 runtime/参数，或同一份 words 投影到多个 timeline。
- separation 生成两个 stem；它只是输入变换，必须回到原音频并由其它模型验证。

## 4. Text 如何参与交叉验证

只有 MOSS 原生输出 `speaker + segment text + timestamp`。Paraformer、Whisper、SenseVoice 本身没有 speaker ID，需要与 diarizer timeline 组合。

默认构造两条 speaker-text track：

```text
MOSS joint track
  = MOSS native speaker segment + text + time

independent ASR x pyannote track
  = Paraformer CIF token interval 或 Whisper word interval
    x pyannote speaker activity
```

二者可以检查：

- `lexical_presence`：ASR 有词但 VAD/diarizer 标为 silence，提示可能漏 speech。
- `boundary_ambiguity`：高质量 token 横跨候选 change/gap，阻止自动认证并触发重跑。
- `backchannel`：短“嗯/对/yeah”与短时第二 speaker onset 同时出现时，要求重查 floor state。
- `crosstalk`：当前 `sample.audio.path` 文件内的多个物理通道出现近同步相同文本时，结合能量、延迟和 embedding 排查串音。
- `assignment disagreement`：MOSS 与 ASR x pyannote 的 speaker-text 分配差异，可暴露 merge/split 问题。

Text 的边界必须明确：

- ASR 没有转出第二路文字，不能认证 `multi_speaker=false` 或 `speaker_overlap=false`。
- ASR x pyannote 的 speaker assignment 来自 pyannote，不能反过来作为 pyannote change 的新票。
- SenseVoiceSmall 有文本能力，但没有稳定词级时间，只作为第三份 lexical observation。
- 同一份 Whisper words 投影到多个 timeline，仍只有一个 Whisper evidence group。

## 5. 一个交叉验证示例

假设候选区间 `12.20-12.45 s`：

| 来源 | 观察 |
| --- | --- |
| MOSS | A 与 B 在 `12.20-12.42 s` 同时 active |
| pyannote | X 与 Y 在 `12.18-12.45 s` 同时 active |
| CAM++ | A/X 对齐、B/Y 对齐，A 与 B 为不同 speaker |
| Silero | scope speech coverage 完整，可计算统一 speech-union 分母 |

两条独立 timeline 的 overlap 区域高度重合，CAM++ 排除了同一人串音，Silero 提供可靠分母，因此可以认证 `speaker_overlap=true`。

如果 MOSS 和 pyannote 都输出 `true`，但标出的 overlap 区域完全不同，则不能认证；应调用 Sortformer 或当前 sample 音频内已有的物理通道检查，仍冲突则输出 `null`。ASR 只转出一路文本不影响这一判断。

## 6. 用假设－预测－检验消解冲突

冲突不直接做二选一。系统对具体 claim 和局部时间区域保存至少三个分支，并让每个分支对独立可观测量作出预测；假设不是“MOSS 整条链路正确”这类全局判断：

```text
H1：来源 A 对该局部事件的解释成立
H2：来源 B 对该局部事件的解释成立
H3 / H_other：双方都不完整、都错，或遗漏了第三种故障
        |
        v
生成并冻结 prediction + falsifier + dependency groups
        |
        v
调用最能区分分支的独立工具
```

例如 MOSS 判断 A/B 两人、pyannote 只判断 X 一人：

| 假设 | 如果正确，应观察到什么 | 优先检验 |
| --- | --- | --- |
| `H1: two-speaker event` | A/B 的干净片段 embedding 不同，第二人候选区有独立 activity | CAM++；适用的 Sortformer/物理通道 |
| `H2: one-speaker event` | MOSS A/B embedding 高度相似，没有独立双人事件 | CAM++；第二 timeline/物理证据 |
| `H3: other/incomplete` | H1/H2 的预测都失败，或出现噪声、混响、背景声等未解释证据 | 新故障分类或人工复核 |

判定边界必须保守：

- 由 pyannote 派生的 speaker-text assignment 不能反过来证明 pyannote 正确；共享依赖的结果只能作 diagnostic。
- prediction 和 falsifier 必须在调用额外工具前冻结；不能看到结果后修改假设。
- 一个分支被反驳，不代表另一个分支自动正确；剩余分支仍需满足 event、identity、coverage 和校准门槛。
- “暂时没有发现矛盾”只表示分支仍可行，不等于得到正向支持。
- `false` 分支即使存活，也必须通过完整覆盖和模型组合的联合漏检校准。

该机制主要用于排除不可能解释、选择下一项高信息量证据和生成可审计的冲突说明，不能单独产出 `certified`。

## 7. 冲突处理与输出

每个 claim 使用四种状态：

| 状态 | 含义 | 公开输出 |
| --- | --- | --- |
| `certified` | event、identity、coverage、校准和冲突检查全部通过 | bool |
| `supported` | 有单边支持，但缺少独立证据或校准 | `null` |
| `conflicted` | 合格来源对值或事件位置仍有冲突 | `null`，定向取证/人工复核 |
| `insufficient` | 输入、上下文、适用性或质量不足 | `null` |

所有 evidence 保存模型版本、能力、时间范围、输入 lineage 和依赖组。相同 checkpoint、同一 pipeline 的派生字段或共享上游的复合结果不得重复计票。

`recording_id` 不是 raw input 新字段：只能从已有 `sample.native_metadata` 复制为可选 opaque provenance，缺失时内部兼容 metadata 使用 `sample_id`。它不能作为 artifact join key，也不能用来读取、拼接或借用其它 utterance 的音频、speaker ID、embedding 或 transcript。短 sample 内没有足够 clean speech/context 时，对应 claim 返回 `insufficient/null`。

## 8. 推荐落地顺序

Phase 1 最小闭环：

- MOSS + pyannote Community-1 + Silero + CAM++。
- 中文/中英使用 Paraformer-zh，多语使用 Whisper base。
- 实现 timebase、speaker alignment、speaker-text track、三个 rule-based resolver，以及 speaker count/overlap/change 三类确定性 `H1/H2/H3` 冲突模板。
- public 只发布 `certified`，其它状态统一映射为 `null`。

Phase 2 冲突取证：

- 接入 Streaming Sortformer、当前 sample 音频内已有的物理通道、SenseVoice 第三观察和 targeted rerun；按预测区分能力选择下一项工具，而不是固定顺序全跑。
- 按语言、layout、speaker count、overlap ratio 校准模型组合。
- 用 risk-coverage curve 评估“高精度认证覆盖率”，而不是只看总体 accuracy。

上线前还需确认：标签语义版本、负例联合校准、模型许可，以及内部 transcript 的 PII/retention 策略。

详细规则、artifact schema、failure injection 和模型调研见 [evidence_fusion_design.md](evidence_fusion_design.md)。当前 metadata 定义见 [speaker_metadata_standard.md](../speaker_metadata_standard.md)。
