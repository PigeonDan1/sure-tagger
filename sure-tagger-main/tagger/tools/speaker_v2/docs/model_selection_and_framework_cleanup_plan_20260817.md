# Speaker v2 模型选择与框架精简计划

> 历史计划：本文的 C/M/O/X Sortformer/Pyannote/MOSS primary/fallback 设计已由
> 2026-09-04 语言路由取代。2026-09-10 起，其他、混合或未知语言的 ASR 使用
> Qwen3-ASR-0.6B，speaker timeline 使用官方 DiariZen；FireRedASR2-AED 仅用于
> 中英文单说话人共识，且禁止跨 ASR 候选回退。
> 本文其余内容仅供追溯当时的模型选择过程。

> 状态：Phase 0–3 工程接入完成；2026-08-20 已移除运行时发布门禁并启用六字段直接输出
>
> 日期：2026-08-17
>
> 实施记录：[model_selection_and_framework_cleanup_implementation_20260817.md](model_selection_and_framework_cleanup_implementation_20260817.md)
>
> 依据：1k 模型测评汇总矩阵与部署 manifest（历史产物，当前 checkout 未保留）、
> [证据融合设计](../../speaker_v1/docs/evidence_fusion_design.md)、
> [当时部署计划](../../speaker_v1/docs/deployment_plan.md)
>
> 适用范围：utterance-level speaker v2 的模型路由、adapter 接入、运行 profile 和候选清单治理。

## 1. 目标与结论

本轮不继续堆叠候选模型。目标是把当前“所有 speaker timeline 对所有 claim 基本等权”的 resolver 改为按 claim 分工，并在 shadow 验证后完成两项模型替换：

1. `I`：SpeechBrain ECAPA 替换 CAM++，CAM++ 降为中文域 fallback。
2. `V`：Brouhaha 替换 FireRedVAD 作为 coverage primary，FireRedVAD 降为低误报/边界 guard。
3. `C/M/O/X`：不再让 MOSS、Sortformer、pyannote 对所有 claim 等权；每个 claim 使用独立的 primary、guard、fallback 和 excluded source。
4. Whisper 默认退出纯 speaker 主链，只在 lexical/audit profile 启用。
5. 未实施候选只从当前计划和默认 profile 中移出或归档，不删除仍被其它 tagger 链路使用的公共 worker、runtime 或模型资产。

当前 v2 可执行入口中的六路模型均已真实部署：MOSS、Sortformer、pyannote Community-1、FireRedVAD、CAM++、Whisper Base。不存在可以直接按“未部署占位模型”从运行代码删除的一批模型。ECAPA 和 Brouhaha 虽未接入 `speaker_evidence.py`，但已有本地资产和正式测评，属于待接入优选项，不属于清理对象。

本计划不调整未进入本次 1k 测评的 `D`，不新增公开 schema，也不授权物理删除任何模型权重或 runtime。`A` 只处理是否进入 speaker 默认 profile，不在本计划内重新选择 ASR 模型。

## 2. 决策依据

冻结数据为 `ami_utterance_1k_v1`：1,000 条 utterance、167 场原始会议、7.7514 小时。身份链路使用同一批数据构建的 `ami_identity_trials_v1`，共 23,815 个 trial，dev/test 的 meeting 和 global speaker 隔离。

| Claim | 当前最佳结果 | 选择结论 |
| --- | --- | --- |
| `C` count | Sortformer：MAE `0.2920`，accuracy `0.7350` | Sortformer primary；MOSS fallback；pyannote 不提供 count 决策票 |
| `M` multi-speaker | Sortformer accuracy `0.9650`；MOSS accuracy `0.9580` 且误报更少 | Sortformer 负责 positive candidate；MOSS 负责 negative/false-positive guard |
| `O` overlap | pyannote：frame F1 `0.7087`，event F1 `0.5218` | pyannote primary；Sortformer secondary；MOSS 只能做正例 corroboration，不能用 negative 否决 |
| `X` speaker change | MOSS：count MAE `1.4730`、F1@0.25s `0.5716`；Sortformer bool accuracy `0.9550`、F1@0.5s `0.6275` | MOSS 负责次数和精定位；Sortformer 负责存在性和补召回；pyannote 不参与等权决策 |
| `V` speech coverage | Brouhaha：F1 `0.9535`、DCF `0.0876`；FireRed：PFA `0.0675` | Brouhaha primary；FireRed 作为低 FA guard，不做 speaker event 票 |
| `I` identity | ECAPA：EER `0.1079`、minDCF `0.5074`；CAM++：`0.1740/0.7601` | AMI/英文 profile 使用 ECAPA primary；CAM++ 仅保留中文域 fallback |
| `A` lexical | MOSS 的中英文 CER/WER 均优于 Whisper Base | Whisper 不进入 speaker resolver，默认只在 audit profile 启用 |

以上数字是 standalone 结果，不等于融合后的最终质量。新 resolver 必须重新生成 prediction 并走同一 SURE scorer，不能把各单项最优数字直接拼成组合模型结果。

## 3. 前置约束

实施时必须同时遵守以下约束：

- Profile 名称继续保留 `shadow` 后缀以兼容既有运行参数；可决策的公开 `speaker` 字段直接输出，冲突或缺失时为 `null`。
- Sortformer 的 AMI 结果存在已知训练数据污染风险，并且模型最多支持 4 speaker。它可以作为本轮方向性选择依据，同时仍需补 5+ speaker applicability 测试并持续记录风险。
- pyannote Community-1 的 license review 当前为 `pending`。许可批准前，它只能在 shadow/support 中作为 `O` primary；生产 profile 的 `O` primary 仍由 Sortformer 兜底。
- ECAPA 当前正式结果验证的是 atomic speaker verification。现有评测使用冻结 clean regions；生产 adapter 必须从预测 timeline 盲选 region，因此替换 CAM++ 前还要补 end-to-end predicted-region 评测。
- Brouhaha 当前有独立 VAD 评测和公共 worker，但旧 runtime 存在兼容性风险。完成 speaker v2 adapter smoke 前不能直接删除 FireRedVAD。
- native speaker annotation 和输入 transcript 不得进入 inference 或 resolver；只能在推理完成后进入 scorer。
- 同一模型的派生字段、不同 wrapper 或共享上游结果不能被当作独立证据票。

### 3.1 清理分级

| 级别 | 对象 | 动作 |
| --- | --- | --- |
| 默认关闭 | Whisper | 从纯 speaker 默认 profile 关闭；保留 lexical/audit 能力和资产 |
| 替换后降级 | CAM++、FireRedVAD | 根据 ECAPA/Brouhaha 持续评测结果分别降为语言 fallback、coverage guard；暂不物理删除 |
| 保留并改职责 | MOSS、Sortformer、pyannote | 从全 claim 等权改为第 4 节的 per-claim 角色 |
| 接入而非清理 | ECAPA、Brouhaha | 补 v2 adapter、manifest、smoke 和 shadow 后成为 `I/V` primary |
| 只归档文档 | Silero、WebRTC、MarbleNet、TitaNet、ERes2Net、EEND、TS-VAD、VBx、Paraformer 等 | 从 active candidate 清单移入 backlog；它们本来就未进入 v2 runtime |
| 禁止随 v2 清理删除 | 公共 subprocess worker、Brouhaha/PANNs/DNSMOS/Rec-RIR handler、共享模型资产 | 由其它 tagger pipeline 使用，必须单独做全仓引用审计和删除审批 |

[2026-08-12 候选状态登记](../../speaker_v1/docs/evaluation/candidate_status_20260812.md)只能作为历史基线，其中“ECAPA checkpoint 不存在”和“pyannote blocked”等描述已经被 2026-08-17 的资产与评测事实取代，不能作为删除依据。

## 4. 目标模型职责

### 4.1 Claim policy

| 模型 | Primary | Guard/secondary | 默认不参与 |
| --- | --- | --- | --- |
| Sortformer | `C` | `M` positive、`O` secondary、`X` recall | `I/V` |
| MOSS | `X` | `C` fallback、`M` negative guard、`O` positive corroboration、ASR audit | `O` negative veto、`I/V` |
| pyannote Community-1 | `O`（shadow；生产需 license approved） | overlap conflict witness | `C/M/X` 等权决策、`I/V` |
| ECAPA | `I`（AMI/英文） | identity witness | `C/M/O/X/V` 直接投票 |
| Brouhaha | `V` coverage | coverage witness | `C/M/O/X/I` |
| FireRedVAD | 无 | `V` 低 FA/边界 guard | speaker event 投票 |
| CAM++ | 中文域 `I` fallback | ECAPA 故障时的 identity diagnostic | 英文默认 primary、`C/M/O/X/V` |
| Whisper Base | 无 | lexical/audit clock | 所有 speaker claim 投票 |

resolver 必须显式接收版本化 `claim_policy`，不能再依赖 `timelines` 的列表顺序决定 primary，也不能把所有可用 timeline 自动加入每个 claim。

### 4.2 运行 profile

计划保留三个 profile：

| Profile | 默认启用 | 用途 |
| --- | --- | --- |
| `legacy-shadow` | 当前六路和当前 resolver 行为 | 回滚与结果复现，不再新增能力 |
| `quality-shadow` | Sortformer、MOSS、pyannote、ECAPA、Brouhaha；FireRed guard；Whisper 关闭；CAM++ 按语言配置 | 默认质量验证 profile |
| `lean-shadow` | Sortformer、pyannote、ECAPA、Brouhaha | 成本优先 profile；`X` 退化为 Sortformer 单路，必须单独报告质量损失 |

`quality-shadow` 是否替换默认 profile 由持续质量与成本观测决定。CLI 的显式 enable/disable 参数优先于 profile，最终展开后的模型集合和 `claim_policy` 必须写入 run manifest。

## 5. 实施阶段

### Phase 0：冻结基线与配置合同

目标：先建立可比较、可回滚的配置边界，不改变现有输出。

实施项：

1. 为 resolver 增加版本化 `claim_policy` 合同，字段至少包括 `primary_sources`、`guard_sources`、`fallback_sources`、`excluded_sources` 和 policy version。
2. 在 `run_speaker_evidence_v2.py` 增加 `--profile`，首先只提供 `legacy-shadow`，并保证它与当前默认行为一致。
3. run manifest 和 fusion artifact 写入展开后的 profile、policy version、policy hash、启用模型和禁用原因。
4. 冻结当前 1k standalone 报告、当前完整 v2 smoke 和现有部署 manifest，作为回归基线。

完成条件：

- `legacy-shadow` 的 evidence 数量、claim 状态和 artifact schema 与当前基线一致。
- policy 缺失、未知 source、同一 source 同时出现在 primary/excluded 时必须 fail closed。
- 所有新增合同和 profile 展开均有单元测试。

### Phase 1：按 claim 路由当前 timeline

目标：先用已经接入的 MOSS、Sortformer、pyannote 改造 `C/M/O/X`，不等待 ECAPA/Brouhaha。

实施项：

1. `C`：Sortformer 可用时只以它产生 count candidate；Sortformer missing/failed 时回退到 MOSS。pyannote count 只保留 diagnostic observation。
2. `M`：Sortformer 生成 positive candidate；MOSS 作为低误报 guard。正负认证规则在 dev 上冻结，禁止用 test 调阈值。
3. `O`：license 未批准时 Sortformer 为 production fallback；shadow 中 pyannote 为 primary、Sortformer 为 secondary。MOSS positive 可 corroborate，negative 不得 veto。
4. `X`：MOSS 负责 change count/精定位，Sortformer 负责存在性和 recall witness；pyannote 只保留 diagnostic。
5. 把 Whisper 从 `quality-shadow` 默认关闭，保留 `--whisper-enable` 或 audit profile 的显式开启能力。
6. 每个 claim artifact 记录实际使用的 primary、fallback 原因、被排除证据及排除原因。

完成条件：

- 单个模型失败时，只触发该 claim 定义的 fallback，不允许重新退化为全局等权。
- MOSS 的 overlap negative 不会覆盖 pyannote/Sortformer positive。
- pyannote 的 count/change 不进入 candidate 决策，但原始 evidence 仍可审计。
- 对 1k 数据生成 per-claim standalone、fusion 和 ablation 报告。

### Phase 2：接入 ECAPA 并替换英文 I primary

目标：把已经完成正式 SV 评测的 ECAPA 接入 v2 evidence pipeline。

实施项：

1. 以 `tagger/tools/speaker_v2/model_evaluate/runs/scripts/identity_predict_ecapa.py` 的冻结模型加载/embedding 实现为基准，并复用 `scripts/run_ecapa_identity_dataset.py` 的 sample-local 调度经验，抽取无 gold 依赖的 `tagger/tools/speaker_v2/ecapa_identity.py`。
2. 复用 CAM++ 的 predicted-timeline non-overlap region 选择合同；adapter 只能接收 sample-local audio 和上游预测 region，不得读取 `native_metadata`。
3. 在 subprocess worker 注册独立 `ecapa_identity_estimate`，冻结模型 revision、SHA256、runtime package lock 和 license snapshot。
4. 在 `SpeakerEvidenceConfig`、CLI、worker slots、deployment manifest 和 artifact lineage 中增加 ECAPA。
5. 将 dev 上冻结的 threshold 和 Platt calibration 作为具名 calibration profile；raw cosine 不能直接作为 probability。
6. 先跑 atomic trial parity，再跑 predicted-region end-to-end shadow；记录 region coverage、abstain、失败率和依赖闭包。
7. 根据 predicted-region 评测结果决定 AMI/英文 `quality-shadow` 是否默认使用 ECAPA；CAM++ 保留在中文 profile 或显式 fallback 中。

完成条件：

- 冻结 atomic benchmark 上 EER `<=0.1129`、minDCF `<=0.5274`，且测试 trial 数保持 `17,259`。
- inference scope 审计确认 native metadata、gold speaker ID 和 reference transcript 均未进入 adapter。
- predicted-region end-to-end 结果不劣于同口径 CAM++，并报告统计不确定性；未达到时不替换默认 primary。
- ECAPA worker 故障时产生显式 missing evidence，不得静默使用未校准 CAM++ 分数替代。

### Phase 3：接入 Brouhaha 并替换 V primary

目标：把 Brouhaha 从独立诊断工具接入 v2 的 `speech_coverage` evidence。

实施项：

1. 基于现有 Brouhaha client 新增 v2 coverage collector；保留其 `V` 能力边界，不增加 `C/M/O/X` 票。
2. evidence 保存 raw/processed speech segments、阈值、后处理、模型 hash、runtime 和 lineage。
3. 在 CLI、`SpeakerEvidenceConfig`、worker slots、run manifest 和 deployment manifest 中增加 Brouhaha。
4. 保留 FireRedVAD 并行 shadow，比较 coverage、PFA、Pmiss、边界、失败率、RTF 和内存。
5. `quality-shadow` 使用 Brouhaha primary、FireRed low-FA guard；`lean-shadow` 只使用 Brouhaha。
6. 完成 runtime 兼容性回归后才能把 FireRed 从默认 primary 降级。

完成条件：

- 冻结 1k 上 F1 `>=0.9485`、DCF `<=0.0926`、PFA `<=0.1500`，且合法空预测继续保留在分母内。
- 10 条分层 smoke 和 1k 全量均无崩溃，失败率不高于 `1%`。
- Brouhaha/FireRed 的 coverage 不被 resolver 解释为 overlap、change 或 multi-speaker 正例。
- 记录相对 FireRed 的额外 CPU/GPU、RTF 和常驻内存成本。

### Phase 4：直接输出、默认 profile 与框架精简

目标：直接发布 resolver 可决策结果，并按后续质量观测决定是否切换默认 profile 和精简路径。

实施项：

1. Public adapter 直接映射六字段结果；默认 profile 暂时保持 `legacy-shadow`，保留显式选择和一键回滚能力。
2. Whisper 默认关闭，只在 lexical/audit profile 启用。
3. ECAPA 通过后，CAM++ 从英文默认 profile 移除；中文 profile 和 rollback profile 保留。
4. Brouhaha 通过后，FireRed 从 V primary 降为 guard；在完成跨域对照前不物理删除。
5. 更新候选登记：ECAPA、pyannote、Brouhaha 移入“已测/已接入”状态；Silero、WebRTC、MarbleNet、TitaNet、WeSpeaker/ERes2Net、EEND/TS-VAD、DiariZen、VBx、Paraformer 等移入 archived backlog。
6. 删除或归档文档中过期的“ECAPA checkpoint 不存在”“pyannote blocked”等描述。

物理删除代码、runtime 或模型资产必须同时满足：

- 当前代码、测试、CLI、所有 active profile 和 deployment manifest 均无引用。
- 共享 tagger 其它 pipeline 无引用；必须对整个 `tagger/` 做引用审计，而不是只查 `speaker_v2/`。
- 新 profile 已完成至少一次 1k 全量和一次非 AMI shadow，且 rollback snapshot 可用。
- 删除目标有明确清单、大小、所有者和恢复位置，并经过单独审批。

不得在本阶段删除：

- `tagger/tools/subprocess_worker.py` 中的 Brouhaha、PANNs、DNSMOS、Rec-RIR 等公共入口。
- ECAPA/Brouhaha 权重和 runtime。
- pyannote、Sortformer 或 MOSS 的 adapter 与资产。
- 尚承担 rollback、中文 fallback 或跨域对照职责的 CAM++/FireRed 资产。

### Phase 5：直接发布与持续观测

运行时不再设置 certification 或 production 阻断条件。以下项目作为持续观测和 profile 选择依据：

1. 在第 6.1 节定义的非 AMI、meeting-separated 数据上复核 `C/M/O/X/V`，排除 Sortformer 的 AMI 污染影响。
2. `I` 按语言/域分别冻结 threshold 和 calibration profile；禁止把 AMI/英文阈值直接用于中文。
3. 持续记录 pyannote license review；部署方需要无 pyannote 的组合时显式选择或覆盖相应模型配置。
4. 新 resolver 完成 out-of-fold fusion/ablation，不只比较 standalone 最优值。
5. 报告每路模型的 RTF、显存、内存、失败率和每条样本增量收益。
6. public adapter 映射可决策的 `supported/certified` claim；`conflicted/insufficient` 继续输出 `null`。

## 6. 质量观测基线

1k AMI 复测使用下表作为告警基线。超过容差时记录回归、评估 profile 调整或回滚，但不由代码阻断发布。

| Claim | 1k 观测基线 |
| --- | --- |
| `C` | count MAE `<=0.3020`；count accuracy `>=0.7250` |
| `M` | multi accuracy 不低于 `0.9550`；同时报告 recall、specificity、FP/FN，不能只看高正例占比下的 accuracy |
| `O` | frame F1 `>=0.6987`；event F1 `>=0.5118` |
| `X` | change count MAE `<=1.5230`；F1@0.5s `>=0.6147` |
| `V` | F1 `>=0.9485`；DCF `<=0.0926`；PFA `<=0.1500` |
| `I` atomic | EER `<=0.1129`；minDCF `<=0.5274`；TPR@FAR=1% 不低于 `0.7064` |

除总体指标外，持续报告模型失败/空预测、1/2/3/4 人、短音频、overlap 正负、change 正负、语言、时长和模型 fallback slice。通过丢弃失败样本获得的提升不计入有效评测结果。

### 6.1 非 AMI 数据合同与观测

生产复核数据冻结为 `non_ami_speaker_shadow_v1`。首选来源是 SURE 配置中已有登记但当前未落盘的 `AliMeeting eval`；数据获取不属于本计划的代码改动。冻结要求如下：

- 使用全部可合法取得的 eval meeting，并从中确定性抽取至少 1,000 条 sample-local cut；少于 1,000 条时明确记录数据不足，不用重复采样补足。
- 保存原始音频、RTTM、cut manifest、提取脚本、许可、来源版本和 SHA256；按 meeting 分 dev/test，test 不参与阈值和 policy 调整。
- 从 RTTM 统一派生 `C/M/O/X/V` gold，并使用与 AMI 1k 相同的预处理和 SURE scorer 口径。
- 若其中 5+ speaker cut 少于 100 条，必须另建 `speaker_slot_overflow_v1` 非 AMI 集合补足；在该 slice 上不允许把 Sortformer 的 4-slot 输出认证为 exact count。
- 同一冻结 test 上比较 `quality-shadow` 与 `legacy-shadow`，按 meeting bootstrap 报告 95% CI。accuracy/F1 类指标的差值 CI 下界必须 `>=-0.02`；MAE/DCF 类指标的差值 CI 上界必须 `<=0.05`；失败率必须 `<=1%`。
- 若 AliMeeting 无法取得、许可不允许或有效规模不足，必须另选带完整 RTTM 的非 AMI meeting corpus并满足同一合同；不得用 AMI 的另一批 cut 代替。

`I` 另行冻结语言/域匹配的 non-AMI verification trial；在该 trial、threshold 和通过阈值登记完成前，ECAPA/CAM++ 的切换仍只属于 shadow 决策。

## 7. 测试计划

### 7.1 单元测试

- profile 展开和 CLI override 优先级。
- 每个 claim 的 primary/guard/fallback/excluded source 过滤。
- primary missing/failed/unsupported 时的确定性 fallback。
- MOSS overlap negative 不得 veto；pyannote count/change 不得进入 candidate。
- ECAPA/Brouhaha 输入 scope、输出 contract、模型 hash 和 missing evidence。
- license pending 时 pyannote 不能进入 production certification。
- run manifest/fusion artifact 的 policy version、hash 和实际 source 记录。

### 7.2 集成测试

- 10 条分层 utterance 跑 `legacy-shadow`、`quality-shadow`、`lean-shadow`。
- 对每个 worker 做 failure injection，确认其它 claim 不被错误重路由。
- 禁用模型时确认对应 subprocess 不启动、模型不加载、worker slot 不占用。
- ECAPA 和 Brouhaha 分别完成 CPU/GPU/runtime healthcheck、超时和 resume 测试。
- 比较 profile 的 wall time、RTF、峰值显存和常驻内存。

### 7.3 数据集评测

- 先复跑 `ami_utterance_1k_v1` standalone，确认 adapter 迁移 parity。
- 再生成 per-claim fusion prediction，使用现有 SURE 路由完成 1k 评分。
- 对每个 source 做 leave-one-model-out ablation，证明 guard/fallback 带来真实增量。
- 最后在非 AMI 数据完成 meeting-separated 评测，作为跨域风险观测。

## 8. 代码变更清单

| 文件/目录 | 计划变更 |
| --- | --- |
| `scripts/run_speaker_evidence_v2.py` | 增加 profile、ECAPA、Brouhaha 参数；保留显式 override 和 legacy 回滚 |
| `tagger/pipelines/speaker_evidence.py` | 增加两个 collector/config/worker slot；把展开后的 policy 传给 resolver |
| `tagger/tools/speaker_v2/resolver.py` | 从全 timeline 通用处理改为 per-claim source routing；记录角色和 fallback |
| `tagger/tools/speaker_v2/contracts.py` | 增加 claim policy/profile 合同校验 |
| `tagger/tools/speaker_v2/ecapa_identity.py` | 新增无 gold 依赖的 production-style identity adapter |
| `tagger/tools/speaker_v2/brouhaha_coverage.py` | 新增仅提供 `speech_coverage` 的 v2 adapter |
| `tagger/tools/subprocess_worker.py` | 注册 ECAPA worker；复用现有 Brouhaha worker，不删除公共 handler |
| `tests/test_speaker_v2.py` | 增加 profile、路由、fallback、scope 和两个 adapter 测试 |
| `.deploy/speaker_v2/` | 新增 model/runtime manifest、healthcheck、smoke 和新 deployment manifest |
| `tagger/tools/speaker_v2/model_evaluate/` | 增加 fusion/ablation 预测与 SURE 报告，不覆盖现有 standalone 结果 |
| `docs/evaluation/` | 更新候选状态，归档未实施 backlog 和过期 blocked 描述 |

## 9. 发布与回滚

发布顺序固定为：

1. `legacy-shadow` parity。
2. 当前模型的 claim-aware resolver shadow。
3. ECAPA shadow。
4. Brouhaha shadow。
5. 1k fusion/ablation。
6. 非 AMI shadow。
7. 根据持续观测结果决定是否将 `quality-shadow` 设为默认。
8. 经过单独审批后再做物理清理。

任一阶段触发以下条件时回滚到 `legacy-shadow`：

- 任一质量指标超过第 6 节容差。
- 模型失败率超过 `1%`，或发生 silent fallback/样本漏记。
- artifact 无法重现实际启用模型、policy 或 calibration profile。
- native metadata、reference transcript 或 gold speaker ID 进入推理路径。
- pyannote 许可、模型 revision/hash 或 runtime 健康状态不满足部署要求。

回滚只切换 profile 和 deployment manifest，不删除新 adapter、评测结果或诊断 artifact，以便定位差异。

## 10. 完成定义

本计划完成需同时达到：

- v2 resolver 已按 claim 路由，不再进行无差别 timeline 等权融合。
- ECAPA、Brouhaha 已进入 speaker v2，并持续生成 `I`、`V` 评测结果。
- Whisper 不再默认启动；CAM++/FireRed 已按计划降为 fallback/guard。
- 1k standalone、fusion、ablation 和非 AMI 报告持续补齐。
- active profile、deployment manifest、候选登记和运行手册状态一致。
- 未实施候选已归档；共享工具和仍有回滚价值的资产没有被误删。
