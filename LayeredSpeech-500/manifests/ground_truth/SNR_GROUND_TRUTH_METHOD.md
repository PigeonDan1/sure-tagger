# SNR Ground Truth 口径

当前数据同时保留两个不同用途的 SNR 参考，不能互相混用。

| 字段 | 口径 | 用途 |
| --- | --- | --- |
| `tags.audio_quality.snr_db` | 渲染 manifest 中的整段 achieved SNR；在 dry speech active samples 上计算 post-RIR speech RMS / noise RMS | 原始 schema GT |
| `frame_snr_ground_truth.mean_db` | 使用已知 speech/noise stems，按 Brouhaha-maker detailed-SNR 训练标签规则计算，再对所有完整窗口的 dB 值取算术平均 | 与 Brouhaha 帧式 SNR 更接近的独立 GT |

## `frame_snr_ground_truth` 算法

1. 将所有 post-RIR speaker stems 相加为 speech，并读取最终 noise stem。音乐若在 noise stem 中，也属于背景项。
2. 用 dry speaker stems 的 `abs(sample) > 1e-6` 并集作为已知 VAD；在 speech 上应用该 mask。
3. 用 VAD-active 的 post-RIR speech RMS 对 speech 和 noise 施加同一个归一化增益。该步骤不改变已渲染的 speech/noise 比例。
4. 按 Brouhaha-maker 默认参数计算 detailed SNR：2 秒窗、10 ms hop、noise power 分母加 `1e-3`、signal/noise ratio 下限 `1e-10`。
5. 保留所有完整窗口，不在窗口级再次用 VAD 筛选；对窗口 SNR dB 做算术平均。

版本为 `known_stems_brouhaha_detailed_snr.v2`。该值完全由已知 stems 确定，不运行 Brouhaha 模型。

## 与 Sure-Tagger 的关系

Sure-Tagger 的 `audio_quality.snr_db` 是 Brouhaha 模型输出，不是物理能量比。其处理方式是：

1. Brouhaha 返回整段逐帧 SNR 预测。
2. Sure-Tagger 递归展开全部数值；只要出现 NaN/Inf 就返回 `null`。
3. 对全部有限预测直接取算术平均，不使用 VAD mask，也不丢弃静音区预测。

Brouhaha 模型训练时，SNR target 来自上述 2 秒 detailed-SNR 标签，并在 loss 中使用 target VAD 作为权重。因此“训练 target 的生成”和“Sure-Tagger 对模型预测的发布聚合”是两层规则：前者在 speech samples 上做 VAD masking，后者对返回的预测帧不做 VAD filtering。

旧版 `known_stems_brouhaha_grid.v1` 错把模型约 61.9375 ms / 16.875 ms 的输出网格当成训练 SNR 能量窗，并额外过滤了窗口，已经废弃。
