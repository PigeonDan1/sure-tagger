# Speaker v2 模型选择与框架精简实施记录

> 日期：2026-08-17  
> 2026-08-20 更新：certification/production 发布门禁已从运行时移除，六字段可决策结果直接进入 public adapter；下列未完成评测改为非阻断跟踪项。  
> 2026-09-04 路由更正：第 1–11 节冻结的是 2026-08-17 至 2026-09-01 的历史实现，
> 其中 public claim source、profile 模型集合和第 11 节双路行为均已废止；只有第 12 节
> 描述当前 speaker 路由。当前先运行
> standalone FireRed LID；`zh`/`zh-*` 和 `en`/`en-*` 进入 MOSS，其他、混合或未知语言进入
> 官方 DiariZen + Qwen3-ASR-0.6B 链路。中英文的单说话人共识仍可选择
> FireRedASR2-AED；失败时不跨 ASR 候选回退。  
> 对应计划：[model_selection_and_framework_cleanup_plan_20260817.md](model_selection_and_framework_cleanup_plan_20260817.md)  
> 历史范围：Phase 0–3 的工程接入状态、2026-08-20 直接发布行为，以及 2026-08-31 双路 ASR 接入；当时回归更新至 2026-09-01，后续数据只作历史质量观测。

## 1. 2026-09-01 时点结论（历史）

Phase 0–3 的代码、配置、adapter、worker、artifact 和单元测试接入已经完成。新实现已放在 `tagger/tools/speaker_v2/`，运行入口仍为 `scripts/run_speaker_evidence_v2.py` 和 `tagger/pipelines/speaker_evidence.py`。

本节记录当时“可选择、可回滚、可审计、可运行”的工程版本。2026-08-20 起不再以 certification 或 production gate 阻断输出：

- 独立 `run_speaker_evidence_v2.py` 默认 profile 仍是 `legacy-shadow`；主 tagging
  pipeline 默认使用 `quality-shadow`。两者都可通过 CLI 显式切换。
- 对独立 speaker CLI，`quality-shadow` 和 `lean-shadow` 需要通过 `--profile` 显式
  选择；主 tagging pipeline 已默认选择 `quality-shadow`。
- ECAPA 已完成 predicted-timeline adapter 接入，但 predicted-region 端到端身份评测尚未完成。
- Brouhaha 已完成 coverage-only adapter 接入；1 条多说话人真实集成 smoke 已通过，但其余分层 smoke、1k 稳定性和成本评测尚未完成。
- 1k fusion/ablation、非 AMI 复核、pyannote license 和运行成本仍需持续跟踪，但不再阻断 public adapter 输出。
- 没有物理删除任何模型、runtime、adapter 或公共 subprocess worker。
- 已新增并核验 `.deploy/speaker_v2/deployment_manifest_20260817.json`；旧清单与旧部署快照继续保留用于回滚。

## 2. 阶段状态

| 阶段 | 工程状态 | 后续观测项 |
| --- | --- | --- |
| Phase 0：配置合同 | 已完成 | 1k 数据级 legacy parity 仍需随正式 fusion run 固化 |
| Phase 1：per-claim 路由 | 已完成 | 1k fusion、per-claim ablation 和 guard calibration 尚未完成 |
| Phase 2：ECAPA 接入 | 已完成 | predicted-region I 端到端评测和跨域 calibration 尚未完成 |
| Phase 3：Brouhaha 接入 | 已完成 | 已完成 1/10 条真实分层 smoke；其余 smoke、1k 全量稳定性、RTF/资源评测尚未完成 |
| Phase 4：默认切换与精简 | 部分完成 | public adapter 已直接输出；主 tagging 默认已切到 quality，standalone 默认和物理精简未切换 |
| Phase 5：直接发布与持续观测 | 已完成发布改造 | 非 AMI、license、统计置信区间和运行成本作为非阻断跟踪项 |

“工程状态已完成”表示接口和回归测试已落地；数据评测状态仍需与代码发布状态分开描述。

## 3. Phase 0：版本化 profile、policy 与 artifact

### 3.1 Profile 合同

新增 `tagger/tools/speaker_v2/profiles.py`，提供：

- `available_profiles()`：当前返回 `legacy-shadow`、`quality-shadow`、`lean-shadow`。
- `expand_profile(profile_id)`：展开模型开关和完整 `claim_policy`。
- 版本化 policy schema、当时 policy `speaker_v2.claim_policy.20260831.1` 和稳定
  SHA256 hash。
- source registry、字段类型、未知 source、重复 source、角色冲突和 policy hash 的 fail-closed 校验。

当时模型集合为（不描述 2026-09-04 当前路由）：

| Profile | 默认启用 | 默认关闭 |
| --- | --- | --- |
| `legacy-shadow` | MOSS、FireRedASR2-AED、FireRedVAD、CAM++、Whisper Base、Sortformer、pyannote | ECAPA、Brouhaha |
| `quality-shadow` | MOSS、FireRedASR2-AED、FireRedVAD、Sortformer、pyannote、ECAPA、Brouhaha | CAM++、Whisper Base |
| `lean-shadow` | FireRedASR2-AED、Sortformer、pyannote、ECAPA、Brouhaha | MOSS、FireRedVAD、CAM++、Whisper Base |

### 3.2 CLI 和运行配置

`scripts/run_speaker_evidence_v2.py` 已增加：

- 独立 speaker CLI 的 `--profile` choices 直接来自 `available_profiles()`，默认值为
  `legacy-shadow`；主 tagging CLI 的 `--speaker-profile` 默认值为 `quality-shadow`。
- 九路模型的 tri-state `--*-enable` / `--*-disable`；原有 disable 参数继续兼容。
- ECAPA 和 Brouhaha 的模型、runtime、设备及 worker 参数。
- FireRed ASR 的模型、源码、runtime、设备、beam/precision、timeout，以及
  `--firered-asr-disable-lid` / `--firered-asr-lid-model` 路由控制参数。
- profile 展开后再应用 CLI override，并把实际模型集合、override、policy 和 profile 传入 `SpeakerEvidenceConfig`。

`SpeakerEvidenceConfig` 会再次校验 profile/policy 一致性，记录实际开关和禁用原因。单模型显式 override 不会改变其它模型的 profile 默认值。

### 3.3 Artifact 可复现性

sample fusion、certification artifact 和 run manifest 现已共同记录：

- `run_profile`；
- 展开后的 `claim_policy`；
- `policy_version` 和 `policy_hash`；
- 每个 claim 的路由选择、fallback 原因、guard observation 和 excluded evidence。

其中 run manifest 还会额外记录九路模型的 profile 默认状态、实际启用状态、配置和禁用原因；sample artifact 不复制整份模型 inventory。

旧调用会按 `legacy-shadow` 补齐 policy 元数据，从而保留兼容性。profile 名称中的
`shadow` 只表示模型组合和回滚身份；2026-08-20 起 public adapter 会直接发布
`supported` 且值合法的 speaker 字段。

## 4. Phase 1：C/M/O/X 的 per-claim 路由

resolver 已从“所有可用 timeline 自动等权参与每个 claim”改为按版本化 policy 路由。当前 `quality-shadow` 路由如下：

| Claim | Primary | Guard / secondary | Fallback | Excluded |
| --- | --- | --- | --- | --- |
| C `speaker_count` | Sortformer | 无 | MOSS | pyannote |
| M `multi_speaker` | Sortformer | MOSS negative false-positive guard | 无 | pyannote |
| O `speaker_overlap` | pyannote | Sortformer secondary、MOSS positive-only corroboration | Sortformer | 无 |
| X `speaker_change` | MOSS | Sortformer recall witness | Sortformer | pyannote |

已经实现的关键语义：

- 只有 primary 不可用时才选择有序 fallback，不会重新退化为全局等权。
- pyannote 的 count/change evidence 仍保留用于审计，但不会进入对应 candidate 决策。
- MOSS 的 overlap negative 只作为诊断，不能否决 pyannote/Sortformer 的 overlap positive。
- route artifact 会记录实际 primary/fallback、配置角色和被排除证据。
- identity candidate timeline 按 count policy 选择，不再依赖 evidence 列表顺序。

当前 guard observation 明确标记 `guards_affect_candidate: false`。也就是说，guard 角色和规则已经进入合同与审计，但在 dev calibration 和 1k ablation 完成前不会被误当成额外等权票或认证捷径。

## 5. Phase 2：ECAPA 身份链路工程接入

新增 `tagger/tools/speaker_v2/ecapa_identity.py`，并完成以下接线：

- source name：`speechbrain_ecapa_voxceleb`；
- capability：仅 `speaker_identity_comparison`，不产生 timeline/count 票；
- 复用 CAM++ 的 predicted-timeline non-overlap region 选择合同；
- adapter 只接受 sample-local audio 和预测 region，显式拒绝 native metadata、gold label、reference transcript 等字段；
- 接入独立 `ecapa_identity_estimate` subprocess worker 和常驻 worker slot；
- 接入 pipeline、CLI、run profile、missing evidence、artifact lineage 和模型 hash；
- comparison 输出保持与现有 identity resolver 可兼容，并支持冻结的 threshold/Platt 参数。

当前默认 calibration profile 来自 AMI 1k atomic clean-region dev。artifact 已明确记录：

- `calibration_scope: atomic_clean_region_dev`；
- `atomic_calibration_not_production_region_calibration: true`；
- `predicted_region_gate_passed: false`。

因此，ECAPA 的工程接入完成不等于身份效果验证完成。仍需补 predicted-region 端到端评测，报告 region coverage、abstain、失败率、EER/minDCF 和相对 CAM++ 的同口径结果，作为英文/AMI profile 选型依据。

## 6. Phase 3：Brouhaha coverage 链路工程接入

新增 `tagger/tools/speaker_v2/brouhaha_coverage.py`，并复用现有 Brouhaha client 和公共 `brouhaha_estimate` worker：

- source name：`brouhaha_vad`；
- evidence type/capability：仅 `speech_coverage`；
- 保存 raw speech segments、规范化 speech/silence segments、coverage ratio、binarization 和 boundary postprocess；
- 固定 checkpoint hash，并记录 model/runtime/lineage；
- 空预测是合法的零 coverage，不会被静默丢弃；
- evidence 明确标记 `not_a_speaker_timeline` 和 `not_a_speaker_event_vote`；
- 接入 pipeline、CLI、profile、worker slot、missing evidence 和九模型 run manifest。

`quality-shadow` 同时保留 Brouhaha 和 FireRedVAD，`lean-shadow` 只启用 Brouhaha；
独立 speaker CLI 默认的 `legacy-shadow` 仍只启用 FireRedVAD，主 tagging pipeline
默认使用 `quality-shadow`。当前没有删除或覆盖 FireRedVAD，Brouhaha 也不会直接产生
C/M/O/X/I 投票。

## 7. 真实 `quality-shadow` 集成结果

已在无可用 GPU 的节点上使用 CPU 跑通一条多说话人/重叠样本：

- sample：`EN2001a_utterance_00054`，时长 `24.019s`，AMI native reference 为 3 人、存在多说话人、重叠和 speaker change；
- profile：`quality-shadow`，开启 post-inference `--score-native`；
- run：`success_count=1`、`failure_count=0`；
- MOSS、Sortformer、pyannote、ECAPA、Brouhaha、FireRedVAD 均产生真实 `estimated` evidence，Whisper 按 profile 明确记录为 disabled/missing，CAM++ 未启动；
- Sortformer 和 MOSS 均预测 3 人，pyannote 预测 2 人；三路对 M/O/X 的布尔判断均与该样本 native reference 一致；
- ECAPA 从 Sortformer 预测 timeline 盲选出 4 个 non-overlap region，生成 6 个 comparison；没有读取 native metadata；历史 evidence 中的 `counts_for_certification=false` 只保留为评测标记，不再阻断公开输出；
- Brouhaha 只声明 `speech_coverage`，coverage ratio 为 `0.786535`；FireRedVAD 对照值为 `0.721096`；
- 实际 C/M/O/X decision source 分别为 Sortformer、Sortformer、pyannote、MOSS，均符合冻结 policy；
- fusion、certification 和 run manifest 的 profile/policy version/hash 一致；这条
  2026-08-17 历史 smoke 运行时公开 speaker 字段仍全部为 `null`，2026-08-20 起
  `supported` 且值合法的字段已改为直接发布。

真实产物保存在 `.deploy/speaker_v2/smoke_quality_shadow_20260817_en2001a_00054/`。CPU smoke 出现 pyannote TorchCodec、Brouhaha 旧训练 runtime 和 Matplotlib cache 警告，但没有导致模型 evidence 缺失；这些兼容性警告仍需纳入后续 10 条 smoke 和成本稳定性观测。

## 8. 单元测试结果

Phase 0–3 工程接入完成时的验收记录为 **68/68 单元测试通过**，覆盖原有 speaker-v2 回归以及 profile/policy、artifact、ECAPA、Brouhaha 和 pipeline 接线。

随后又补充了两项 CLI/profile override 与 artifact 禁用原因回归。本文复核时使用 Python 3.11 执行：

```bash
python3.11 -m unittest discover -s tests -p 'test_speaker_v2*.py'
```

当前扩展套件结果为 **70/70 通过**。因此 68/68 的阶段验收记录保持成立，新增的两项兼容回归也已通过。

测试覆盖重点包括：

- legacy policy 输出兼容和 quality policy per-claim 路由；
- policy 缺失/未知 source/hash 冲突 fail closed；
- primary 可用时不触发 fallback；
- MOSS overlap negative 不构成 veto；
- profile 默认与显式 CLI override 的优先级；
- run manifest 的九模型清单、policy 和禁用原因；
- ECAPA predicted-region 输入净化、gold/native 拒绝、worker 路由和 lineage；
- Brouhaha coverage-only 能力边界、空预测、segment 规范化和 checkpoint 校验；
- 原有 scope、dependency closure、certification、timeline、lexical 和 native-data 隔离回归。

## 9. 后续验证与观测项

以下项目仍需补齐，但不再作为运行时发布门禁：

1. **1k fusion/ablation**：需要在同一冻结 1k manifest 上生成 `legacy-shadow`、`quality-shadow`、`lean-shadow` 的完整 prediction、fusion、per-claim 报告和 leave-one-model-out ablation。
2. **Predicted-region I 评测**：ECAPA 仍需完成预测 region 端到端身份评测，不能把 atomic clean-region 指标直接当成该链路的实际效果。
3. **其余分层 smoke**：当前仅完成 `quality-shadow` 的 1 条多人/overlap/change 样本；仍需覆盖单人、短片段、长片段、低 coverage 和模型 missing/failure，并运行三种 profile。
4. **Brouhaha 1k 稳定性和成本**：需验证失败率、RTF、CPU/GPU、峰值内存、常驻内存及相对 FireRed 的代价。
5. **非 AMI 复核**：必须使用 meeting-separated、非 AMI 数据复核 C/M/O/X/V，排除 Sortformer 的 AMI 数据重合风险。
6. **License 状态**：pyannote Community-1 license review 仍需记录；部署方应按实际授权选择 profile。
7. **运行观测**：继续冻结 calibration、统计置信区间、失败率和回滚 snapshot，并维护 deployment manifest。

建议后续顺序为：补齐分层 smoke → predicted-region I 评测 → 1k 三 profile fusion/ablation → 非 AMI 评测 → license/成本/统计观测 → 按结果决定是否切换 standalone 默认 profile。

## 10. 删除与回滚边界

本次没有执行任何物理清理。以下对象均保留：

- MOSS、FireRedASR2-AED、FireRedLID、Sortformer、pyannote、Whisper、CAM++、
  FireRedVAD、ECAPA、Brouhaha 的代码和模型资产；
- `tagger/tools/subprocess_worker.py` 及其中所有共享 handler；
- 历史 runtime、rollback profile 和原有评测产物；
- 现有 deployment manifest 和 2026-08-17 新增的 claim-aware shadow 清单。

独立 speaker CLI 默认仍保持 `legacy-shadow`，主 tagging pipeline 默认使用
`quality-shadow`；这些名称只是模型组合，不再是发布门禁。若显式运行
`quality-shadow` 或 `lean-shadow` 出现质量、许可或运行时问题，可切回
`--profile legacy-shadow`（主 tagging CLI 对应 `--speaker-profile legacy-shadow`），
不需要删除新 adapter 或评测 artifact。

## 11. 2026-08-31 双路 ASR 接入（历史，2026-09-04 已废止）

本节只保留当时实现和回归结果，不描述当前生产链路。当前合同以文首 2026-09-04
更正及 [`direct_public_output_20260820.md`](direct_public_output_20260820.md) 为准；
以下反向语言选择、双路并发和跨分支回退均不得用于当前配置。

speaker-v2 现在在同一个 sample inference scope 内并行调用 MOSS-Transcribe-Diarize
和 FireRedASR2-AED。MOSS evidence 继续承担 speaker timeline、speaker claim 和
speaker-text assignment；FireRed evidence (`fireredasr2_aed`) 只承担 ASR/lexical
candidate，不进入 C/M/O/X claim policy。FireRed 默认使用
`models/FireRedASR2-AED`、`models/FireRedASR2S` 和
`.runtime/fireredasr2_aed_py311_torch280_cu128_v1/bin/python`，并通过独立的
`firered_asr_estimate` 常驻 worker 隔离依赖。

当时实现采用的反向语言选择、双模型并发和 availability 补位规则已全部删除；为避免
被复制到新部署，本记录不再复述其逐条件算法。2026-08-31 的 24/24、112/112、
207/207 回归数字只验证当时实现，不能作为当前语言路由的回归结论。FireRed 与 MOSS
的 7 集历史对照结果（68,996 条，FireRed failed/missing 均为 0）见
[`asr双路评测报告_20260831.md`](../evalue/asr双路评测报告_20260831.md)。

## 12. 2026-09-07 语言路由链路

当前先独立运行 FireRed LID。已识别的 `zh`/`zh-*` 和 `en`/`en-*` 进入 MOSS joint
ASR/diarization；其他、mixed、unknown 和 LID failure 进入
`non_chinese_or_unknown` 官方 DiariZen + Qwen3 ASR 分支。ASR 策略名为
`single_speaker_consensus_firered_for_zh_en_else_qwen3_asr`。选中 ASR 不向另一候选回退，
Sortformer/Pyannote 也不产生 C/M/O/X 公开 claim。

当前 claim policy 为 `speaker_v2.claim_policy.20260910.1`。四个 C/M/O/X rule
均使用 `selection="language_routed"`，`fallback_sources` 为空。resolver 强制要求
样本 `language_route` 对象；缺失或不受支持的 route 直接失败，不能按 evidence
availability 选择分支。每条 resolved claim route 记录
`language_selected_source`；该 source 可用时 `selection="language_route"`，不可用
时 `selection="none"`、记录失败原因并产生 `null` claim，不尝试另一 source。

> 以下 OpenASR 0.1.40 pack、CLI 和 runtime 细节是历史归档，仅用于复现旧产物；当前实现
> 已改为独立 FireRedASR2-AED + 官方 DiariZen，不能按下述命令启动当前链路。

非中文分支固定使用 OpenASR 0.1.40 一次加载：

- FireRedASR2-AED q4_k：stdout JSON text，产生 ASR evidence；
- DiariZen large s80 v2 fp16、ReDimNet2-B6 fp16、Qwen3-ForcedAligner 0.6B q4_k、
  bundled FireRed Stream-VAD：完成 diarization 与 FireRed speaker attribution，
  产生版本锁定 debug trace raw turns 和 timeline evidence。

```bash
OPENASR_DIARIZEN_PACK=models/diarizen-large-s80-v2/diarizen-large-s80-v2-fp16.oasr \
OPENASR_REDIMNET_PACK=models/redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr \
OPENASR_FORCED_ALIGNER_PACK=models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr \
OPENASR_DIARIZE_DEBUG=1 \
.runtime/openasr/0.1.40/openasr transcribe AUDIO \
  --model-pack models/firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr \
  --diarize --offline --no-punctuate --format json
```

实现配置为
`DiariZenTimelineConfig(runtime_executable, diarizen_pack, redimnet_pack, firered_asr_pack, forced_aligner_pack, timeout_sec, openasr_version, runtime_executable_sha256)`。
独立 speaker CLI 与主 tagging CLI 均提供 `--openasr-executable`、
`--openasr-firered-asr-model-pack`、`--diarizen-model-pack`、
`--redimnet-model-pack`、`--qwen3-forced-aligner-model-pack` 和
`--openasr-timeout-sec`。五个默认路径可分别用 `TAGGER_OPENASR_RUNTIME`、
`TAGGER_DIARIZEN_MODEL_PACK`、`TAGGER_OPENASR_REDIMNET_PACK`、
`TAGGER_OPENASR_FIRERED_ASR_PACK`、`TAGGER_OPENASR_FORCED_ALIGNER_PACK` 覆盖；
主 tagging 默认 aligner 路径的 local-config 键为
`OPENASR_FORCED_ALIGNER_MODEL_PACK`。底层子进程用 `OPENASR_FORCED_ALIGNER_PACK`
传递实际路径。

独立 speaker CLI 只用组合开关 `--openasr-enable` / `--openasr-disable` 同时覆盖
profile 中的 `firered_asr` 与 `diarizen` 状态；旧的两套独立 enable/disable 参数已
移除。OpenASR runtime 默认部署路径为 `.runtime/openasr/0.1.40/openasr`；在进程内
首次使用对应文件身份时，对 resolved executable（包括 CLI/env 路径覆盖）执行强
校验：`--version` stdout 为 `openasr 0.1.40`，大小
42,268,384 bytes，SHA-256
`98d973d0d72b47198ecf17e4c00b2710d84c351b77d3c8024a7256bcea59a1a5`。校验失败
发生在推理前；成功调用的 runtime provenance 保存 `engine_version`、
`executable_verified`、`executable_sha256`、`executable_size_bytes` 和
`executable_version_output`。

adapter 结果为 `text`、`segments` 和 `runtime`；内部 timeline evidence 保存
`timeline_summary`、`raw_turns`、`openasr_model_chain`，ASR evidence 保存
`text`/`asr_transcript`。合法的 0-turn/0-speaker trace 是可用时间线，公开 count 为
0；空 stdout text 只使 ASR 为 missing。模型/runtime/输出合同失败按相关组件
fail closed，不使用 MOSS、输入 transcript 或诊断模型补值。

许可证边界：OpenASR binary、FireRed pack 与 Qwen3-ForcedAligner pack 为
Apache-2.0，DiariZen pack 为 CC BY-NC 4.0，ReDimNet pack 为 MIT。所有 `.oasr`
都是 OpenASR model pack；DiariZen `.oasr` 不是 standalone runtime。aligner pack
位于 `models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr`，固定
大小为 765,395,744 bytes、SHA-256
`5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b`、OpenASR
catalog revision `3a82628cf7036c2ad0128f2d35bd1cced09b3100`。

能力边界：所有非 `zh` 结果进入该分支是路由合同，不代表全语种 ASR 已支持。固定
FireRed/ForcedAligner catalog 只声明 en/zh 语系。英文端到端 smoke 通过；真实
法语样本无 hint 时 exit 0 但返回语义错误的英文 text，同时 DiariZen turn 正常；
显式 `--language fr` 时 exit 6。非 en/zh ASR 尚未获生产验证，且结构校验无法自动
识别 silent wrong transcript。
