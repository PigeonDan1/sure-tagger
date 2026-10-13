# AMI EN2001a 多说话人能力诊断报告

> 运行日期：2026-08-12
>
> 评测轨道：`AMI_single_meeting_diagnostic`
>
> 状态：诊断性结果，不是正式 leaderboard，不用于生产模型选型或概率校准

## 1. 目的和边界

本报告对当前可运行的多说话人证据链做一次全量 AMI 诊断，目的是回答“每个模型在什么能力上表现较好、错误是否互补”，而不是给现有 pipeline 的固定角色做校准。所有模型都读取同一条现有 utterance-level `sample.audio.path`，没有增加输入字段、没有拼接相邻 sample，也没有把 native metadata 送入推理。

数据位于 `/hpc_stor03/sjtu_home/weihan.chen/share/tagger/ami_en2001a_utterances`，共 195 条 sample，全部来自一个 meeting `EN2001a`。每条生成 WAV 可能包含多个原始 `utt_id` annotation；本报告的 gold 是在当前 sample 音频范围内对这些 native intervals 做的 sample-scope scorer 投影。该数据适合工程链路和错误形态诊断，不提供可识别的 session-cluster 置信区间。

已知数据限制：Sortformer 的训练资料包含 AMI，标记为 `known_contaminated`；MOSS 训练 provenance 为 `unknown`。因此本报告不能宣称 clean-domain 泛化，也不对模型产生正式名次。

## 2. 运行和产物

运行命令使用以下冻结 profile：

| 证据源 | profile / device | 输出能力 |
| --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | `cuda:1` | timeline、count、overlap candidate、change candidate、joint text |
| NVIDIA Streaming Sortformer 4spk-v2 | `cuda:0` | timeline、count、overlap candidate、change candidate |
| FireRedVAD | CPU | speech coverage |
| Brouhaha VAD | `.runtime/fireredvad_rebuild_py310` / CPU | independent speech coverage diagnostic；不提供 speaker attribution |
| CAM++ | CPU | sample-local identity comparison；未校准 |
| Whisper Base | `cuda:3` | lexical clock、boundary/lexical diagnostic；时间为 experimental attention-DTW |

运行摘要：`processed=195`、`success=195`、`failure=0`。原始 evidence 和 fusion artifact 位于：

```text
/hpc_stor03/sjtu_home/weihan.chen/share/tagger/ami_en2001a_utterances/outputs/ami_single_meeting_diagnostic_20260812/
```

每条 sample 都有五类 speaker evidence artifact；CAM++ 有 193 条 `estimated`、2 条 `missing`（没有足够长的候选 clean region），其余四类均为 195/195 `estimated`。另有 Brouhaha VAD 对照 artifact，见 [`brouhaha_vad_diagnostic_report_20260812.md`](brouhaha_vad_diagnostic_report_20260812.md)。所有证据的 `calibration_profile_id` 为空；CAM++ 的 threshold status 是 `upstream_default_uncalibrated_for_ami`。Brouhaha 成功推理时的 adapter `confidence=1.0` 也只是状态占位，不是概率。

## 3. Gold 语义

主对照轨使用 `speaker_metrics_v0.1.0` 的 sample-scope 配置：最短 segment `0.10 s`、同 speaker merge gap `0.30 s`、overlap event 最短 `0.10 s`、change max gap `1.00 s`。count 和 change 与当前 shadow scorer 的定义一致；overlap 必须特别区分两种口径：

| 轨道 | 定义 | gold 正例 |
| --- | --- | ---: |
| `metadata_v0_overlap`（本报告主对照） | overlap duration / speech union `>= 0.05` | 122/195 |
| `shadow_any_overlap_0.1s` | 只要存在至少一个连续 `0.10 s` overlap event | 166/195 |

旧的 `evaluation_only` 字段仍保留第二条 shadow 口径，不能与 `metadata_v0_overlap` 的数字直接混用。后续正式榜必须固定 `semantics_version`、`scorer_version` 和 `scorer_config_hash` 后再发布。

Gold count 分布为：1/2/3/4/5 speaker = `14/92/64/24/1`。`multi_speaker` 正例为 181/195，`public_v0_change` 正例为 175/195。

## 4. Standalone 诊断结果

以下是 point estimate；括号中的 CI 不报告，因为只有一个 meeting。`O` 使用 `metadata_v0_overlap`，不是旧 shadow overlap。

| 模型 | Count exact | Count MAE | Multi F1 / recall | Overlap F1 / recall | Change F1 / recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOSS | 72.31% | 0.318 | 0.977 / 0.956 | 0.667 / 0.500 | 0.968 / 0.937 |
| Sortformer v2 | **82.05%** | **0.190** | **0.994 / 0.989** | **0.780 / 0.639** | **0.988 / 0.977** |

两者在这条单会议诊断上都没有 false positive（specificity=1.000），但 AMI 的正例比例很高，不能只看 accuracy。该现象也可能来自模型后处理偏保守，必须在有足够负例的跨 meeting 数据上复核。

### 4.1 Count 分层

| gold speaker 数 | sample 数 | MOSS exact / MAE | Sortformer exact / MAE |
| ---: | ---: | ---: | ---: |
| 1 | 14 | 14/14 / 0.000 | 14/14 / 0.000 |
| 2 | 92 | 86/92 / 0.065 | 89/92 / 0.033 |
| 3 | 64 | 37/64 / 0.438 | 46/64 / 0.281 |
| 4 | 24 | 4/24 / 1.125 | 11/24 / 0.583 |
| 5 | 1 | 0/1 / 1.000（预测 4） | 0/1 / 2.000（预测 3） |

当前结果只支持一个诊断性观察：随着真实 speaker 数增加，两个模型都更容易 under-count，Sortformer 的误差较小。5-speaker 只有一个 sample，且 Sortformer 配置最多 4 slots，不能从该点推断其通用上限。

### 4.2 Overlap 语义敏感性

若改用旧 `shadow_any_overlap_0.1s` gold，MOSS/Sortformer 的 overlap F1 分别为 `0.863/0.939`；改用正式 `metadata_v0_overlap` 后变为 `0.667/0.780`。这不是模型突然变差，而是 gold contract 不同。正式评测必须只在同一 semantics track 内比较。

## 5. 错误互补性

互补性按 sample-paired 正确/错误统计，不把两个模型的原始分数直接相加。以下数字仍是单 meeting descriptive counts：

| Claim | 预测一致 | 两者都正确 | 仅 MOSS 正确 | 仅 Sortformer 正确 | 两者都错 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Count exact | 156 | 134 | 7 | 26 | 28 |
| Multi | 187 | 186 | 1 | 7 | 1 |
| Overlap (`metadata_v0`) | 168 | 129 | 5 | 22 | 39 |
| Change | 186 | 183 | 1 | 8 | 3 |

几个有用的条件统计：

- overlap 中 MOSS 失败的 61 条里，Sortformer 单独正确 22 条，conditional recall `22/61=36.1%`；Sortformer 失败的 44 条里，MOSS 单独正确 5 条，反向 conditional recall `5/44=11.4%`。
- count exact 中 Sortformer 单独正确 26 条，MOSS 单独正确 7 条；两者同时错误 28 条，说明简单 union 不能直接变成 exact-count resolver。
- multi 和 change 的错误高度重合且正例占比高；这两项更像“第二模型用于发现漏检”的场景，而不是独立多数票。

因此，Sortformer 在本诊断中同时显示出较好的 count、multi、overlap、change point 候选表现，并且对 MOSS 有一定补漏价值；这只是 `AMI_single_meeting_diagnostic` 上的候选结论，不是已验证的生产 specialist。后续应加入 pyannote/独立 OSD/异构 ECAPA 或 EEND，检验增益是否来自真正独立的 evidence family。

## 6. Shadow fusion 和交叉证据

当前 shadow resolver 没有启用 public adapter，因此 195/195 条的公开值都是 `null`。状态分布如下：

| Claim | `supported` | `conflicted` | 主要冲突/阻塞 |
| --- | ---: | ---: | --- |
| Count | 156 | 39 | 两条 timeline count 不一致；无 calibrated upper bound |
| Multi | 187 | 8 | 两条 timeline bool 不一致 |
| Overlap | 112 | 83 | bool 不一致或 event alignment 不足 |
| Change | 115 | 80 | point alignment 不足或 bool 不一致 |

每条 sample 均保留：MOSS joint text track、Whisper lexical clock x timeline track、两条 speaker-text comparison。Whisper comparison 的 symmetric lexical error 只作 diagnostic（195 条 mean 0.255），不能当作 ASR gold accuracy；CAM++ 也只提供未校准的 identity comparison，且本轮候选 clean region 是从 Sortformer timeline 选择的，因此它不是独立的人数/事件票，不能把 cosine/threshold 当作人数概率。

Brouhaha 与 FireRedVAD 的 195 条 VAD 对照均成功。按 native speaker interval union 的 20 ms 诊断口径，Brouhaha frame precision/recall/F1 为 `0.9827/0.9408/0.9613`，FireRedVAD 为 `0.9930/0.8741/0.9298`；Brouhaha 的 speech recall 更高但 false alarm 略多。两路 silence-ratio delta（Brouhaha minus FireRedVAD）均值 `-0.0641`、平均绝对差 `0.0697`，说明它们不是同一 evidence 的复制。该对照只属于 `V` coverage 诊断，不能给 `C/M/O/X` 增加事件票，也不能把 Brouhaha 判定的 speech 当成 overlap 或 change 的正例。

这说明“交叉验证”在系统中确实产生了可观测的冲突信息：例如首条 sample 的 native count=3、MOSS=2、Sortformer=3，正式 overlap gold=true，而两个模型 overlap prediction 分别为 false/true，resolver 因此保留冲突或支持状态，不输出未经校准的 bool。它不是简单投票，也没有把一个模型的输出伪装成概率。

## 7. 当前结论与下一步

1. 本轮没有足够数据建立正式榜单；不要把 `AMI_single_meeting_diagnostic` 的点估计写成泛化排名。
2. 若只看当前诊断，Sortformer 是 count/multi/overlap/change 的候选领先者；MOSS 在部分 sample 上提供独有正确结果，仍有互补价值。该结论必须在 session-separated、污染可审计的数据上复现。
3. 正式评测先冻结 `metadata_v0_active_count`、`metadata_v0_overlap`、`public_v0_change` 三个主 scorer，再在独立 dev session 做阈值/校准；`floor_transfer_v2` 和 `different_speaker_onset` 单独列榜。
4. 下一批优先加入 pyannote Community-1、独立 OSD、ECAPA/3D-Speaker 和至少一种 EEND/TS-VAD；VAD 方向同时保留 Brouhaha/FireRedVAD 对照，用 out-of-fold prediction 计算 unique correct、conditional recall、joint false-negative rate 和增量成本。
5. 只有在多个 meeting/session、明确 clean/contaminated provenance、足够正负例和 calibration split 均满足后，才生成 `leaderboard_<track>` 和推荐模型职责。

相关协议见 [`leaderboard_evaluation_plan_20260812.md`](leaderboard_evaluation_plan_20260812.md)，候选状态见 [`candidate_status_20260812.md`](candidate_status_20260812.md)。本报告只修改文档，运行产生的 JSONL/artifact 是诊断性输出，不改变生产代码或公开 metadata contract。
