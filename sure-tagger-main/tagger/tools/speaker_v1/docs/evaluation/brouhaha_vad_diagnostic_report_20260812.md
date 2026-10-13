# Brouhaha VAD / 声学辅助证据诊断报告

> 运行日期：2026-08-12
>
> 评测轨道：`AMI_single_meeting_diagnostic / raw_native_speech_union_20ms`
>
> 状态：已通过离线 smoke 和 AMI 全量诊断；不是正式 VAD leaderboard，也不是多说话人模型排名

## 1. 结论

Brouhaha 已在共享目录中具备可复现的离线推理条件，可以作为 `V`（speech coverage/VAD）以及 `SNR/C50` 声学质量候选。它不能直接产生 speaker count、`multi_speaker`、overlap、speaker change、speaker identity 或完整 diarization 证据，因此不进入 `C/M/O/X/I/D` 的 speaker 榜。

在当前 AMI 单会议诊断中，Brouhaha 与 FireRedVAD 对同一 utterance 的 speech mask 产生了明显但不固定方向的差异。按未阈值 native speech union 的 20 ms 诊断口径，Brouhaha 的 frame F1 高于 FireRedVAD，主要来自更高的 speech recall；这只能说明它是有信息量的 coverage witness，不能说明它是 speaker 标签的独立正例票。

## 2. 资产和运行环境

| 项目 | 固定值 |
| --- | --- |
| checkpoint | `models/brouhaha/brouhaha-vad/models/best/checkpoints/best.ckpt` |
| checkpoint 大小 | `47,224,097` bytes |
| checkpoint SHA256 | `9c237e4a7b1de8b456dbee25db853342bf374b19d8732b72b61356519e390ae1` |
| 代码目录 | `models/brouhaha/brouhaha-vad` |
| 环境 | `.runtime/fireredvad_rebuild_py310` |
| 运行时 | Python 3.10.20、Torch 2.2.2 CPU、pyannote.audio 3.3.0、Brouhaha 0.9.0 |
| 模型原生设置 | 16 kHz、mono downmix、6 s chunk、frame outputs `vad/snr/c50` |

checkpoint 是真实 Zip 权重，不是 Git LFS pointer。直接 API 和 subprocess adapter 都能离线返回 `annotation`、frame-level `snr`、`c50`；单条约 29.6 s 的 AMI 音频 CPU 推理约 7-8 s。

当前运行会提示 checkpoint 训练于 pyannote.audio 0.0.1、Torch 1.12.1+cu102，而环境为 pyannote.audio 3.3.0、Torch 2.2.2+cpu；Lightning 还会在内存中升级 checkpoint，并提示 torchvision 不可用。因此本结果是 `diagnostic_pass / compatibility_risk`，在兼容性回归完成前不得作为 production approval。

## 3. 评测边界和 gold 口径

- 输入仍是现有 utterance-level `sample`，只读取当前 `sample.audio.path`，不拼接相邻 sample。
- native speaker intervals 只在推理完成后用于离线 scorer，未进入 Brouhaha 或 FireRedVAD 推理。
- 本报告的 VAD gold 是 `raw_native_speech_union_20ms`：把当前 sample 内 native speaker intervals 的 union 投影到 20 ms frame grid。
- 该口径没有套用 `metadata_v0_active_count` 的 segment filter，也没有人工修订短片段；因此不能与未来正式 VAD contract 的数字混排。
- AMI `EN2001a` 只有一个 meeting，所有数字均为 descriptive point estimate，不生成 rank 或 session CI。

## 4. Smoke 和全量结果

分层 smoke 选取 10 条样本，覆盖 1-5 speaker、无 overlap、低/高 overlap、短/长 utterance。Brouhaha 和 FireRedVAD 均为 `10/10` 成功。

全量诊断命令为：

```bash
cd /hpc_stor03/sjtu_home/weihan.chen/share/tagger
.runtime/fireredvad_rebuild_py310/bin/python \
  scripts/run_brouhaha_silence_comparison.py \
  --manifest ami_en2001a_utterances/manifest.jsonl \
  --output ami_en2001a_utterances/outputs/brouhaha_vad_full_diagnostic_20260812/comparison.jsonl
```

结果为 `195/195` 两路成功、`0` invocation failure。按 `raw_native_speech_union_20ms` 汇总：

| 模型 | frame precision | frame recall | frame F1 | sample-macro F1 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| FireRedVAD | 0.9930 | 0.8741 | 0.9298 | 0.9284 | 更保守，miss 较多 |
| Brouhaha VAD | 0.9827 | 0.9408 | 0.9613 | 0.9545 | recall 较高，FA 略多 |

两路预测的 speech-mask disagreement 不是固定偏差：Brouhaha minus FireRedVAD 的 sample silence-ratio delta 均值为 `-0.0641`、中位数 `-0.0660`、平均绝对差 `0.0697`，范围 `[-0.1604, 0.0774]`。负值表示 Brouhaha 判为 speech 的覆盖通常更大。按 native speaker 数分层的 delta 均值为：1 人 `-0.1021`、2 人 `-0.0613`、3 人 `-0.0641`、4 人 `-0.0549`、5 人 `-0.0041`；这些分层很稀疏，不能解释为人数因果关系。

## 5. 如何作为交叉证据

Brouhaha 的合理用法是 `coverage_guard` 或 VAD 专项候选：

1. 用它和 FireRedVAD 的 frame mask、speech duration、onset/offset 差异定位 coverage 风险。
2. 在某条 overlap/change 事件上，如果任一路把关键区间判为 silence，标记 `coverage_insufficient`，而不是把该路的 silence 当作 overlap=false 或 change=false。
3. 在 dev session 上校准二者的 raw frame score，再测加入 Brouhaha 后对 `O/X/D` 的增量收益；只有 out-of-fold 的 conditional recall 和 joint false-negative 风险改善，才允许作为生产辅助证据。
4. Brouhaha 的 `SNR/C50` 只能用于声学难度分层、失败解释和 slice 分析。AMI 没有独立 SNR/C50 gold，本轮不对这两个数做准确率排名。

举例：`EN2001a_utterance_00000` 上 FireRedVAD nonspeech ratio 为 `0.503688`，Brouhaha 为 `0.556369`；native scorer 显示该 sample 有 3 个 speaker 且存在 overlap。两路 coverage 不一致可以触发重查 overlap 的时间分母，但不能由 Brouhaha 的更长 speech mask 推导出 3 人，也不能投票认证 overlap=true。

## 6. 置信度和上线状态

当前 Brouhaha adapter 成功时写入的 `confidence=1.0` 仅表示 inference status，不是概率，也没有 AMI/session calibration。正式评测必须保存 native frame score，按 session-separated calibration split 计算 Brier、NLL、ECE 和 risk-coverage；FireRedVAD 也使用同一流程。不能直接比较两者的 adapter confidence。

推荐登记：

```text
model_id = brouhaha_vad_v0.9.0
evidence_family_id = brouhaha_activity
capabilities = [V, SNR, C50]
speaker_capabilities = [C:N/A, M:N/A, O:N/A, X:N/A, I:N/A, D:N/A]
status = diagnostic_pass_compatibility_risk
calibration_profile_id = null
```

原始对照产物：

```text
ami_en2001a_utterances/outputs/brouhaha_vad_smoke_20260812/
ami_en2001a_utterances/outputs/brouhaha_vad_full_diagnostic_20260812/comparison.jsonl
```

## 7. 下一步

- 在多 meeting/session 数据上建立正式 `V` 榜，并补测 Silero、WebRTC 或 MarbleNet；保持 `N/A` 与失败分开。
- 以同一 session-separated split 测 `FireRedVAD -> resolver`、`Brouhaha -> resolver` 和 `FireRedVAD+Brouhaha`，报告 coverage、事件召回、joint false-negative、RTF 和成本。
- 完成旧 checkpoint/runtime 的兼容性回归，确认升级后的 pyannote/Torch 不改变边界和 SNR/C50 输出；在此之前保留 `audit_only` 标记。
- 不把 Brouhaha 加入人数、overlap、change 的独立事件投票，除非未来获得明确的 speaker-attributed 或 overlap-capable 模型输出并重新登记 lineage。
