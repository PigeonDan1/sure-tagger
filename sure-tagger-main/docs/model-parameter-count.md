# 统计链路模型参数量

本文统计 `sure-tagger-main` 当前默认完整 tagging 链路中实际涉及的模型参数量。
统计对象是模型权重中的参数个数，不是 checkpoint 文件大小，也不是运行时激活张量的元素个数。

## 1. 统计范围

统计入口为 [`scripts/run_tagger.py`](../scripts/run_tagger.py)，默认配置为：

- 完整 pipeline（`FULL_STAGES`）；
- `quality-shadow` speaker profile；
- 音频存在且有效；
- 各模型推理成功；
- 按真实代码路由统计，不把不可达分支强行放入单条样本的激活量。

完整 stage 列表定义在 [`tagger/pipelines/tagging.py`](../tagger/pipelines/tagging.py) 的 `FULL_STAGES` 中，包含：

```text
language_deterministic
audio_probe
silence
speaker
brouhaha
dnsmos
firered_aed
dass
recrir
firered_lid
deterministic_derived
```

speaker-v2 的主链路位于 [`tagger/pipelines/speaker_evidence.py`](../tagger/pipelines/speaker_evidence.py)。每条有效样本的 speaker 链路都会：

1. 运行 FireRed LID；
2. 并发运行 MOSS-Transcribe-Diarize 和官方 DiariZen；
3. 根据语言和两套 diarization 的人数共识选择 ASR；
4. 根据语言选择 gender 模型；
5. 运行 `quality-shadow` 启用的 ECAPA、Brouhaha、FireRed VAD 等辅助模型。

## 2. 结论

主口径包含：

- DiariZen 的 VBx/PLDA 与 x-vector transform 参数；
- Griko gender 路由使用的 SVM 参数；
- 共享模型只计一次；
- Qwen3 的 tied `lm_head` 不重复计数。

| 指标 | 参数量 | 约等于 |
|---|---:|---:|
| 所有真实可达分支的模型总参数量（并集） | **3,906,361,448** | **3.906B** |
| 最大激活参数量 | **3,114,600,249** | **3.115B** |
| 最小激活参数量 | **1,950,985,214** | **1.951B** |

这里的“激活参数量”指一条样本按真实路由会涉及的模型参数之和，而不是只统计某一个模型的前向计算量。

## 3. 固定链路参数

以下组件在 `quality-shadow` 完整链路中属于固定部分，合计：

```text
1,935,492,618
```

| 模型/组件 | 参数量 | 统计说明 |
|---|---:|---|
| FireRed VAD | 588,417 | `model_state_dict` 参数 |
| FireRed AED | 588,931 | `model_state_dict` 参数 |
| FireRed LID | 868,270,688 | 排除 positional encoding buffer |
| MOSS-Transcribe-Diarize | 908,513,280 | safetensors 张量总和 |
| DiariZen WavLM segmenter | 69,454,355 | `pytorch_model.bin` 参数 |
| DiariZen WeSpeaker ResNet34 | 6,634,336 | 排除 BatchNorm running statistics |
| DiariZen VBx/PLDA + x-vector transform | 49,792 | `plda.npz` 和 `xvec_transform.npz` 数组元素 |
| SpeechBrain ECAPA embedding + classifier | 22,150,912 | embedding + classifier，排除运行统计量 |
| Brouhaha | 3,930,599 | checkpoint 参数，排除运行统计量 |
| DNSMOS primary | 287,292 | ONNX initializer 参数 |
| DNSMOS P.808 | 54,945 | ONNX initializer 参数 |
| DASS | 48,648,239 | safetensors 参数 |
| Rec-RIR | 6,320,832 | 仅模型 state，不含 optimizer state |

固定部分的计算式为：

```text
588,417
+ 588,931
+ 868,270,688
+ 908,513,280
+ 69,454,355
+ 6,634,336
+ 49,792
+ 22,150,912
+ 3,930,599
+ 287,292
+ 54,945
+ 48,648,239
+ 6,320,832
= 1,935,492,618
```

## 4. 真实路由与激活量

路由代码见 [`tagger/pipelines/speaker_evidence.py`](../tagger/pipelines/speaker_evidence.py) 中的 `run_record`、`_select_speaker_asr` 和 `_single_speaker_count_consensus`。

### 4.1 ASR 路由

- FireRed LID 为中文或英文时，基础 speaker timeline 使用 MOSS。
- 中文/英文且 MOSS、DiariZen 都报告至多一位说话人时，ASR 切换到 FireRedASR2-AED。
- 中文/英文但多人、人数共识失败或人数不可用时，使用 MOSS 的 transcript。
- 非中文、mixed、unknown 或 LID 失败时，使用 Qwen3-ASR-0.6B。
- MOSS 和官方 DiariZen 每条样本都会先运行，因此它们属于固定链路，即使最终 transcript 采用 FireRed 或 Qwen3 也不会从固定部分扣除。

### 4.2 Gender 路由

语言映射见 [`tagger/tools/speaker_v2/speaker_gender.py`](../tagger/tools/speaker_v2/speaker_gender.py) 的 `gender_model_for_language`：

- 中文/英文：Alice；
- 配置的非洲语言：AfriSpeech；
- 其它语言、mixed、unknown、LID 失败：Griko。

### 4.3 可达组合

| 真实条件 | 额外涉及模型 | 激活参数量 |
|---|---|---:|
| 中文/英文，MOSS 与 DiariZen 均至多一人 | FireRedASR2-AED + Alice | **3,114,600,249** |
| 中文/英文，多人或人数共识不可用 | Alice（MOSS 已在固定链路中） | **1,950,985,214** |
| 非洲语言 | Qwen3-ASR-0.6B + AfriSpeech | 2,725,188,716 |
| 其它语言、mixed、unknown、LID 失败 | Qwen3-ASR-0.6B + Griko SVM | 2,719,983,831 |

最大值计算：

```text
1,935,492,618 + 1,163,615,035 + 15,492,596
= 3,114,600,249
```

最小值计算：

```text
1,935,492,618 + 15,492,596
= 1,950,985,214
```

注意：最小值不是 `Qwen3 + Griko`。原因是 Qwen3 路由还需要额外加载 782,426,112 个参数，而中英文多人路径已经由固定链路中的 MOSS 提供 transcript，只需追加 Alice。

## 5. 总参数并集

所有真实可达分支的模型并集为：

```text
固定链路                         1,935,492,618
+ FireRedASR2-AED               1,163,615,035
+ Qwen3-ASR-0.6B                  782,426,112
+ Alice gender                     15,492,596
+ AfriSpeech gender                7,269,986
+ Griko SVM                        2,065,101
------------------------------------------------
=                                  3,906,361,448
```

其中各分支模型参数为：

| 模型 | 参数量 |
|---|---:|
| FireRedASR2-AED | 1,163,615,035 |
| Qwen3-ASR-0.6B | 782,426,112 |
| Alice gender | 15,492,596 |
| AfriSpeech embedding + head | 7,269,986 |
| Griko SVM | 2,065,101 |

## 6. 统计口径

### 6.1 计入规则

- 以实际加载的 checkpoint / ONNX initializer / safetensors tensor shape 为准。
- 对同一模型的共享或绑定权重去重。
- 将模型后端中实际用于推理的可学习或统计参数计入；DiariZen 的 PLDA 与 x-vector transform 因为参与 VBx 推理，单独计入。
- Griko 的 SVM 支持向量、dual coefficient 和 intercept 计入，共 `2,065,101` 个数值参数。

### 6.2 排除规则

- positional encoding buffer 不计入 FireRed ASR/LID。
- BatchNorm running mean、running variance、`num_batches_tracked` 不计入。
- CMVN、标签表、词表、配置项不计入。
- Rec-RIR 的 optimizer state 不计入。
- Griko StandardScaler 的均值、方差等预处理常数不计入主口径。
- 不按文件字节数估算参数量。

### 6.3 神经网络-only 对照值

如果只统计神经网络权重，不计 DiariZen PLDA/x-vector transform 和 Griko SVM，则：

| 指标 | 参数量 |
|---|---:|
| 模型总参数量 | 3,904,246,555 |
| 最大激活参数量 | 3,114,550,457 |
| 最小激活参数量 | 1,950,935,422 |

## 7. 当前链路中不计入的模型

以下组件在默认 `quality-shadow` 完整链路中未运行，不能计入本统计：

- PANNs；
- topic 模块；
- Whisper lexical clock；
- CampPlus identity；
- Sortformer；
- Pyannote Community-1；
- 旧 OpenASR pack / forced aligner 路径。

此外，subprocess worker 会缓存并复用模型。本文报告的是逻辑参数量；如果统计多个独立 worker 的物理驻留副本，需要按 worker 数量和进程隔离情况另行计算，不能与单条样本的激活参数量混用。
