# 多说话人候选模型状态登记

> 版本：`candidate-registry-descriptive-20260812`
>
> 目的：记录哪些候选已经真实跑过、哪些只能评特定 claim、哪些因资产/许可/环境原因尚未进入评测。
>
> 这不是 leaderboard。AMI `EN2001a` 只有一个 meeting，所有已测数字都是诊断 point estimate，不用于泛化排名。

## 1. 已真实测量的候选

| 模型/冻结 profile | 已测数据 | 可评 claim | 当前证据 | 当前状态 | 不能据此宣称 |
| --- | --- | --- | --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | AMI EN2001a，195/195 | `C/M/O/X/D/A`（按 adapter registry） | Count exact `72.31%`，MAE `0.318`；`M` F1 `0.977`；`O` F1 `0.667`；`X` F1 `0.968` | `diagnostic_measured` | clean-domain specialist、已校准概率 |
| NVIDIA Streaming Sortformer 4spk-v2 | AMI EN2001a，195/195 | `C/M/O/X/D` | Count exact `82.05%`，MAE `0.190`；`M` F1 `0.994`；`O` F1 `0.780`；`X` F1 `0.988` | `diagnostic_measured_known_contaminated` | 通用 5+ speaker 上限、clean 泛化 |
| CAM++ 中文 checkpoint | AMI EN2001a，193/195 identity candidate | `I` atomic；不能由 atomic profile 评 `C/M/O/X/D` | cosine/same-different 诊断，AMI English 未校准；clean crop coverage 不足时 abstain | `diagnostic_measured_uncalibrated` | 英文 identity specialist、人数概率 |
| Whisper Base multilingual | AMI EN2001a，195/195 | `A`、lexical/boundary diagnostic | 词级/segment lexical clock；与 speaker timeline 的 assignment disagreement | `diagnostic_measured` | overlap=false 或 speaker event 正例票 |
| FireRedVAD | AMI EN2001a，195/195（VAD 对照） | `V` | 对 `raw_native_speech_union_20ms`：frame F1 `0.9298` | `diagnostic_measured` | speaker count/identity/overlap/change |
| Brouhaha VAD v0.9.0 | AMI EN2001a，10/10 smoke、195/195 全量 | `V`、`SNR/C50` acoustic diagnostics | frame F1 `0.9613`；相对 FireRedVAD 有不同 coverage 错误形态；SNR/C50 无独立 gold | `diagnostic_pass_compatibility_risk` | speaker count/identity/overlap/change；production VAD |

## 2. 候选但尚未可评

| 候选 | 预期 claim | 阻塞原因 | 进入条件 |
| --- | --- | --- | --- |
| SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`) | `I`；与固定 VAD 组合后可审计 `C/M/X/D` | 共享 `models/` 没有 checkpoint；现有环境只有代码；本轮在线仓库拉取未稳定完成 | 将 checkpoint、license/revision、SHA256 放入 `models/`，完成独立环境/10 条 pair smoke |
| pyannote Community-1 | `C/M/O/X/D`，可派生 `I` | gated 权重，无授权 token | 完成授权、固定 revision、离线加载和 scope smoke |
| Silero VAD | `V` | 没有共享本地权重和冻结 runtime | 权重/版本/hash 固定后与 FireRed/Brouhaha 同 gold 评测 |
| WebRTC VAD | `V` | 当前环境没有 package；输出采样率/帧长 contract 尚未登记 | 固定 package/version，完成 10 条 smoke；作为 deterministic baseline 单独列 family |
| NVIDIA MarbleNet VAD | `V` | 没有本地 checkpoint；NeMo profile 未建立 | 权重、许可证和 NeMo 环境固定后评测 |
| NeMo clustering diarizer / TitaNet | `C/M/X/D`，通常 `O=N/A` | 没有完整本地模型 bundle 和 clustering config | 固定 VAD、embedding、cluster 参数，blind count smoke 通过 |
| WeSpeaker / 3D-Speaker ECAPA/ERes2Net/TitaNet | `I` 或 modular `C/M/X/D` | 没有对应英文 checkpoint/统一 adapter | 先做 checkpoint/license audit，再进入 identity pair smoke |
| EEND/TS-VAD/EEND-EDA、DiariZen、VBx | 重点 `O/D/X` 或 classical baseline | 当前没有可离线固定的本地权重/配置 | 逐模型完成 reproducibility、license、slot/count applicability audit |

`blocked` 与 `N/A` 的含义不同：`blocked` 是本应评测但资产或环境尚未满足；`N/A` 是模型原生能力不适用于该 claim。两者都不能填成 0，也不能进入正式 rank。

## 3. 当前 AMI 的 claim 观察（仅诊断）

| claim | 当前较强 standalone 观察 | 互补证据 | 仍缺什么 |
| --- | --- | --- | --- |
| `C` count | Sortformer 在该 meeting exact/MAE 更好 | MOSS 有 7 条独有正确，Sortformer 有 26 条独有正确 | 多 meeting、5+ speaker、slot overflow 评测；ECAPA/pyannote clustering baseline |
| `M` multi | 两条 timeline 都很高，但正例比例高 | 第二 timeline 可发现漏检 | 足够 negative session、joint-negative calibration |
| `O` overlap | Sortformer F1/recall 更高 | MOSS 在 5 条上独有正确，Sortformer 在 22 条上独有正确 | 独立 OSD/EEND、干净负例、event-level held-out 增益 |
| `X` change | Sortformer point estimate 更高 | MOSS 有少量独有正确 | 冻结 public-v0 与 floor-transfer-v2，边界/ backchannel 分层 |
| `I` identity | CAM++ 仅提供未校准诊断 | ECAPA/其它英文 embedding 尚未加入 | 英文 clean pairs、session-separated threshold/calibration |
| `V` coverage | Brouhaha recall/F1 高于 FireRedVAD，但 FA 略高 | 两路 mask 差异可解释 coverage 风险 | 多域 gold、raw score calibration、resolver 增量实验 |
| `A` lexical | Whisper 提供独立 lexical clock | MOSS joint text 与 Whisper assignment 可互查 | 多 ASR、WER/词时间 gold、text-assisted OOF ablation |

## 4. 评测规则

1. 每个模型先在所有适用 claim 上跑 standalone；不能因为当前架构给它分配了某个角色就跳过其它能力。
2. 只有独立 root evidence family 才能计互补性；同一 timeline 的派生字段、同一 ASR words 的多次投影不算新增票。
3. `audio_only` 主榜与 `audio_plus_hypothesis_text` 辅助榜分开；native reference transcript 只用于事后 gold/scorer。
4. 正式组合必须使用 session-separated、out-of-fold prediction；单 meeting 结果只作 diagnostic。
5. 模型“擅长某 claim”的结论必须同时看 standalone 指标、关键 slice、coverage、错误互补、校准和成本；不能只看一个总 accuracy。
