# Speaker v2 标签与链路清单

> 更新日期：2026-09-07  
> 范围：当前 `speaker_v2` 的公开标签、内部 claim、内部 evidence capability 及规划能力。本文不包含 tagger 通用 pipeline 的其它公开标签。

## 1. 结论

- 当前正式 claim 为 `C/M/O/X`；standalone resolver 的公开 speaker 对象包含
  `speaker_count`、`multi_speaker`、`speaker_change_count`、`speaker_change`、
  `overlap_ratio`、`speaker_overlap` 和 `profiles`。其中 change count 和 overlap
  ratio 从 X/O 实际选中的 decision timeline 派生。
- 主 tagging adapter 另行派生 `speaker_present`，并发布语言路由选中的
  `speaker.asr_transcript`，所以 tags-only speaker 合同共九个字段。
- `I/V` 和分支内原始 `A` 输出是内部 evidence capability；`A` 的路由结果通过上一条
  tagging 字段公开。`D` 尚未接入正式 claim/output。
- Speaker v2 推理阶段不读取 native metadata。当前有效链路均属于“通过可评估模型”。native metadata 只允许在推理完成后用于评测。
- Certification gate 已移除；可决策的六字段结果会直接写入 `evaluation_output.speaker` 和 `public_adapter.speaker`。
- 下表按当前语言路由展开；旧版 Sortformer/Pyannote 评测数字仅作为历史基线，不能解释当前公开值。
- 表中置信度是冻结数据集上的全局 SURE 指标估计，不是单条样本的校准概率。
- FireRed LID 先决定原语言路由，每条样本并发运行 MOSS 和官方 DiariZen。C/M/O/X
  仍使用语言路由选中的 timeline；中英文且两边有效人数都 `<=1` 时 ASR 使用
  FireRed，中英文其它情况使用 MOSS，其他语言使用 Qwen3-ASR-0.6B。forced aligner 不再属于当前
  speaker claim 链路。
- 其他语言进入 DiariZen + Qwen3 ASR；FireRed 的 en/zh 能力边界不再覆盖其他语言。
  Qwen3 ASR 声明支持 30 种语言和 22 种中文方言，DiariZen timeline 仍独立判断。

## 2. 标签链路总表

| 标签名 | 链路 | 链路类型 | 预估置信度（SURE） |
| --- | --- | --- | --- |
| C `speaker_count`（公开 schema） | 中文/英文：音频 → MOSS timeline → count；其他语言：音频 → 官方 DiariZen timeline → count | 通过语言路由模型 | 新路由评测待补；旧 Sortformer/Pyannote 数字不适用 |
| M `multi_speaker`（公开 schema） | 语言路由时间线 → `speaker_count >= 2` | 通过语言路由模型 | 新路由评测待补 |
| O `speaker_overlap` / `overlap_ratio`（公开 schema） | 语言路由时间线（中文/英文 MOSS，其他语言官方 DiariZen）→ overlap bool/regions/ratio | 通过语言路由模型 | 新路由评测待补 |
| X `speaker_change` / `speaker_change_count`（公开 schema） | 语言路由时间线（中文/英文 MOSS，其他语言官方 DiariZen）→ change bool/count/points | 通过语言路由模型 | 新路由评测待补 |
| I `speaker_identity_comparison`（内部 evidence） | 按语言路由选择 MOSS/DiariZen timeline → non-overlap regions → ECAPA embedding/cosine → same/different | 通过可评估模型 | 新路由评测待补 |
| I `speaker_identity_comparison`（内部 evidence） | 预测 timeline → non-overlap regions → CAM++ embedding/score → same/different（legacy；仅显式 identity 诊断） | 通过可评估模型 | 约 **82.60%**（`1-EER`）；test accuracy `94.37%`；当前运行阈值仍需校准 |
| V `speech_coverage`（内部 evidence） | Brouhaha → speech segments/coverage（`quality-shadow` coverage source） | 通过可评估模型 | **95.35%** VAD F1；boundary F1 `52.95%`；DCF `0.0876` |
| V `speech_coverage`（内部 evidence） | FireRedVAD → speech segments/coverage（低误报/边界 guard；`legacy-shadow` coverage source） | 通过可评估模型 | **92.24%** VAD F1；PFA `6.75%`；boundary F1 `19.09%` |
| A `routed_asr_transcript`（由 tagging adapter 公开） | 中英文且 MOSS + DiariZen 有效人数都 `<=1`：FireRedASR2-AED；中英文其它情况：MOSS；其他语言：Qwen3-ASR-0.6B | 通过人数共识与语言路由模型 | 新路由评测待补；FireRed/Qwen3 text 不构成 lexical timeline；旧 ASR 对照数字仅作历史参考 |
| A `lexical_timeline`（内部 audit capability） | Whisper Base → transcript/lexical clock | 通过可评估模型 | 中文约 **64.52%**（`1-CER`）；英文约 **94.84%**（`1-WER`）；word coverage `65.88%/95.53%`；时间戳未单独评测 |
| D `full_diarization`（规划能力） | 当前没有正式 claim/output 链路 | — | — |

## 3. 公开与内部状态

| 能力 | 当前状态 |
| --- | --- |
| `C/M/O/X` | 对应六个 standalone 公开字段。resolver 直接发布可决策 candidate；冲突、缺失或非法派生仍输出 `null`。 |
| `profiles` | standalone resolver 的确定性公开画像数组；与 decision timeline 对齐，不引入新模型。 |
| `speaker_present` | 仅由主 tagging adapter 根据已校验的 `speaker_count > 0` 确定性派生。 |
| `A` / `speaker.asr_transcript` | 选中分支的原始输出和路由诊断属于内部 evidence；standalone 通过 `speaker_asr_transcript` 返回选中文本，主 tagging adapter 将其写入公开 `speaker.asr_transcript`。 |
| `I/V` | 内部 evidence，用于假设验证和 coverage 诊断，不直接写公开 speaker 标签。 |
| `D` | 当前没有正式 claim、resolver 输出或本轮 1k 链路。 |

## 4. 路由行为说明

1. standalone FireRed LID 先确定原结果路由，随后每条样本并发运行 MOSS 与 DiariZen。
   C/M/O/X 继续使用原语言路由选中的 timeline；两边人数有效且都 `<=1` 时 ASR 选择
   FireRed，否则按原语言路由。任一人数缺失不触发共识，选中 ASR 失败时禁止跨候选补位。
2. 当前 policy 为 `speaker_v2.claim_policy.20260910.1`；C/M/O/X rule 的
   `selection="language_routed"` 且没有 fallback source。resolver 要求样本
   `language_route`；缺失或非法 route 不按 evidence availability 猜测。实际路由
   始终记录 `language_selected_source`：source 可用时为
   `selection="language_route"`，不可用时为 `selection="none"`、记录原因并发布
   `null`，不尝试另一 source。
3. 当前 guard 只生成 observation，`guards_affect_candidate=false`，不会修改 candidate。
4. `excluded/audit` evidence 会保留用于审计，但不参与该 claim 的候选决策。
5. 当前没有完成 1k fusion/ablation，因此没有多模型组合后的联合 SURE 分数；表中分数均为单模型分数。
6. C/M/O/X 的 claim source 由语言路由固定：中文和英文为 MOSS，其他或未知语言为
   官方 DiariZen timeline；不再把 Sortformer/Pyannote 作为这些公开 claim 的
   主判或联合投票。
7. ASR route 在原 FireRed LID 结果上增加 MOSS/DiariZen 单说话人共识：中英文
   共识成立时公开文本取 FireRed，否则取 MOSS；其他语言始终取 Qwen3 ASR。输入 transcript
   不会被当作模型输出。

独立 speaker CLI 使用 `--firered-asr-enable/disable`、
`--qwen3-asr-enable/disable` 与 `--diarizen-enable/disable` 分别控制；旧的
`--openasr-enable/disable` 仅作为兼容组别名。MOSS 分支由
`--moss-enable/disable` 控制，路由选择不会在分支之间回退。

## 5. 置信度口径

- `C`：使用 exact `count_accuracy`，同时报告 `count_mae`。
- `M`：使用 `multi_accuracy`。
- `O`：使用样本级 `overlap_accuracy` 作为主估计，同时报告更严格的 frame/event F1。
- `X`：使用 `change_bool_accuracy` 作为布尔标签估计，同时报告 change-point F1 和 count MAE。
- `I`：由于 benchmark 正负样本约为 `1:10`，主估计使用更保守的 `1-EER`；test accuracy 仅作补充。
- `V`：使用时长级 VAD F1，同时报告边界 F1、PFA 或 DCF。
- `A`：使用 `1-CER`/`1-WER` 作为文本正确度代理；该分数不能代表 lexical timestamp 的置信度。

评测数据为冻结的 `ami_utterance_1k_v1`：C/M/O/X/V 共 1,000 条 utterance；I 从同一批数据构建 23,815 个 verification trials。AMI 上的分数仍需通过非 AMI、meeting-separated 数据复核，尤其是可能存在 AMI 预训练污染的模型。

## 6. 当前配置注意事项

独立 speaker CLI 无参数时仍默认：

```text
--profile legacy-shadow
```

profile 只选择附加 evidence/诊断模型，不改变公开语言路由。因此，不显式指定 profile 时：

- C/M/O/X 严格使用语言路由选中的单一 timeline；Sortformer/Pyannote 即使运行也只作诊断；
- 每条样本运行 MOSS 与官方 DiariZen，C/M/O/X 仍按 FireRed LID 选择单一 timeline；
- 两边有效人数都 `<=1` 时 ASR 使用 FireRed，否则按原语言路由，失败时不跨候选回退；
- I 使用 CAM++，ECAPA 默认关闭；
- V 使用 FireRedVAD，Brouhaha 默认关闭。

`quality-shadow` 只改变辅助诊断模型，不是启用语言路由的前提；如需该组合可显式使用：

```text
--profile quality-shadow
```

主 tagging CLI 的 `--speaker-profile` 已默认 `quality-shadow`，无需额外参数；如需
回滚可显式传 `--speaker-profile legacy-shadow`。

## 7. 依据

- 路由/profile：`tagger/tools/speaker_v2/profiles.py`
- resolver/public adapter：`tagger/tools/speaker_v2/resolver.py`
- 直接公开输出合同：`tagger/tools/speaker_v2/docs/direct_public_output_20260820.md`
- evidence 采集：`tagger/pipelines/speaker_evidence.py`
- 历史 ASR 对照评测：`tagger/tools/speaker_v2/evalue/asr双路评测报告_20260831.md`
- SURE 汇总：`tagger/tools/speaker_v1/model_evaluate/evaluation_matrix.md`
- 模型选择实施记录：`tagger/tools/speaker_v2/docs/model_selection_and_framework_cleanup_implementation_20260817.md`
