# Ground Truth 字段策略

| Schema 字段 | 新版取值 | 直接来源 |
| --- | --- | --- |
| `basic_acoustic.duration_sec/sample_rate_hz/channels` | 已知值 | 最终 WAV header |
| `basic_acoustic.nonspeech_ratio/silence_segments` | 已知值 | 最终 session utterance 时间戳的补集；表示非人类语音区间 |
| `audio_quality.snr_db` | 已知值 | session `achieved_snr_db` |
| `frame_snr_ground_truth.mean_db` | 已知值（独立于 schema） | post-RIR speech stems 与 noise stem；仅平均与 dry speech 相交的完整帧 |
| `audio_quality.dnsmos_*` | `null` | 未运行 DNSMOS |
| `room_acoustic.rt60_sec/c50_db` | 已知值 | RIR 实测文件 |
| `room_acoustic.far_field` | `null` | 无显式标签 |
| `sound_field_scene.*` | 已知值 | L5 渲染配置 |
| `speaker.speaker_count/overlap_ratio` | 已知值 | session manifest |
| `speaker.speaker_change*` | 已知值 | 渲染轮次序列的确定性计算 |
| `speaker.asr_transcript` | 已知值 | 合成脚本 reference transcript |
| `speaker.profiles` | `null` | 不使用目标 profile 充当音频真值 |
| `language_content.*` | 已知值或 `null` | 冻结脚本文本统计；混合语言的单值 language 为 `null` |
| `deterministic_derived.*` | 已知值 | schema 公式 |
| `descriptive.*` | `null` | 禁止描述性推断 |
