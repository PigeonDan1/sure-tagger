名称（sure-tagger，建议命名）	和公司对齐的新名称	所用或备选模型	现状（2026-08-23 核对代码）
basic_acoustic.duration_sec	attribute.duration	ffprobe / audio_probe 确定性脚本	✅ 已实现（audio_probe stage）
basic_acoustic.sample_rate_hz	attribute.sample_rate	ffprobe / audio_probe	✅ 已实现
basic_acoustic.channels	attribute.channels	ffprobe / audio_probe	✅ 已实现
basic_acoustic.nonspeech_ratio	annotation[].vad.nonspeech_ratio	FireRedVAD	✅ 已实现；优先 native_metadata 确定性生成，缺失时 FireRedVAD 兜底
basic_acoustic.silence_segments	annotation[].vad.silence_segments	FireRedVAD	✅ 已实现（同上）
音频质量		
audio_quality.snr_db	annotation[].audio_quality.snr	Brouhaha	✅ 已实现（brouhaha stage → audio_quality.snr_db）
audio_quality.dnsmos_sig	annotation[].audio_quality.dnsmos_sig	DNSMOS	✅ 已实现
audio_quality.dnsmos_bak	annotation[].audio_quality.dnsmos_bak	DNSMOS	✅ 已实现
audio_quality.dnsmos_ovrl	annotation[].audio_quality.dnsmos	DNSMOS	✅ 已实现
audio_quality.dnsmos_p808	annotation[].audio_quality.dnsmos_p808	DNSMOS P.808	✅ 已实现（personalized 模型，--dnsmos-personalized）
空间声学信息		
room_acoustic.far_field	annotation[].room_acoustic.far_field	暂无可靠直接模型；Rec-RIR + Brouhaha 规则融合 / Qwen3-Omni 辅助	❌ 未实现。字段仅占位恒为 null，无 stage 产出；Qwen3-8B 与 vLLM runtime 已备好但未接线
room_acoustic.rt60_sec	annotation[].room_acoustic.rt60	Rec-RIR	✅ 已实现（recrir stage → room_acoustic.rt60_sec）
room_acoustic.c50_db	annotation[].room_acoustic.c50	Rec-RIR；Brouhaha C50 可交叉验证	⚠️ Rec-RIR 产出 room_acoustic.c50_db；Brouhaha C50 已降级为内部 evidence（internal.brouhaha_c50_db），两者自动交叉验证未实现
声学环境		
sound_field_scene.speech_music_events（三个）	annotation[].task_extension.sound_event	FireRedVAD-AED	⚠️ AED 已实现（speech_music_events/music_present；2026-08-25 起 singing/music 需事件占比 ≥ --firered-aed-min-singing-ratio / --firered-aed-min-music-ratio 默认 0.10 才公开，caption_pairs_3000 校准消除语音帧误报）；DASS 辅助验证未实现
sound_field_scene.music_present是否有背景音乐	annotation[].others.music_state	FireRedVAD-AED；DASS 可辅助验证	⚠️ 同上（music 占比门控 2026-08-25）
sound_field_scene.external_noise_type背景噪音	annotation[].sweeper_scene.external_noise_info.type	DASS-Small 50.1（主） / CED-Small（备） / PANNs（baseline）	✅ 已实现（2026-08-23，dass stage）：使用 DASS medium AudioSet-2M 48.9（saurabhati/DASS_medium_AudioSet_48.9），checkpoint 复制到 models/DASS；项目内 CUDA runtime 为 `.runtime/dass_runtime_py311/bin/python`，可由 `bash scripts/setup_dass_gpu.sh` 幂等部署和验证。2026-08-23 起 DASS 为默认链路背景噪音主模型（panns 于 2026-08-25 废弃删除，PANNs 工具模块保留仅作交叉验证 evidence）；默认排除主语音、Silence、声学场景、混响、回声，--no-exclusion 关闭全部排除以便观察原始类别分布。2026-08-24 起同时产出 sound_field_scene.noise_composition（全量 527 类按 docs/DASS.md 归组，每类 top-3 ≥0.25，音乐类别以 FireRed AED music_present 门控；人类/未归类与各类别分数只进内部 evidence category_events，⑦ 类别留作 far_field/混响补充证据）；同日 external_noise_type 默认阈值 0.5→0.25（phase2 校准：真实噪声类分数偏软 0.1–0.45，干净语音 <0.15），并把 external_noise_type 改为输出 DASS.md 类别键（未被排除且 ≥0.25 的标签所归类别，按类内最高分降序，人类/未归类永不公开），具体标签由 noise_composition 展开。CED 未接入。注：计划写的是 DASS-Small 50.1，实际部署的是 medium 48.9，同为 DASS 系列
sound_field_scene.acoustic_scene 声学环境	annotation[].sweeper_scene.acoustic_scene 	DCASE ASC：WestAI DCASE24（主） / CNN14 DCASE-2020（备）	❌ 未实现，无 stage、无输出字段
多说话人		
speaker.multi_speaker	annotation[].speaker.multi_speaker	speaker-first language route selects MOSS/DiariZen timeline	✅ 已实现（2026-09-17）：speaker-first segment LID or full-audio fallback selects the route, then按选中 timeline 计算 `speaker_count >= 2`
speaker.speaker_count	annotation[].speaker.speaker_count	speaker-first language route selects MOSS/DiariZen timeline	✅ 已实现（2026-09-17）：按选中时间线的 observed speaker count；非中文/英文仍调用官方 DiariZen 完成 diarization
speaker.speaker_change	annotation[].speaker.speaker_change	语言路由选中的 MOSS timeline 或官方 DiariZen timeline	✅ 已实现（2026-09-04）：从选中时间线检测 change points
speaker.speaker_change_count	annotation[].speaker.speaker_change_count	语言路由选中的 MOSS timeline 或官方 DiariZen timeline	✅ 已实现（2026-09-04）：选中时间线的 change point 计数
speaker.speaker_overlap	annotation[].speaker.speaker_overlap	语言路由选中的 MOSS timeline 或官方 DiariZen timeline	✅ 已实现（2026-09-04）：从选中时间线检测 overlap
speaker.overlap_ratio	annotation[].speaker.overlap_ratio	语言路由选中的 MOSS timeline 或官方 DiariZen timeline	✅ 已实现（2026-09-04）：由选中时间线的 overlap/speech union 派生
speaker.profiles	（公司 schema 暂无对应；对齐 captioner speakerProfile 的 speed/pitch/speaker_volume）	确定性统计适配器（复用语言路由 decision timeline / 带时间范围的文本或可选 lexical evidence / coverage evidence，无新模型）	✅ 已实现（2026-09-04 更新路由）：语速（zh_char_per_sec / word_per_min）、相对音高档位、片内相对音量；OpenASR stdout 只有整段 text、不能直接归属 speaker，无其它受信 lexical evidence 时非中文语速为 null；不推断年龄/性别/情绪/口音
speaker.asr_transcript	annotation[].transcription.text	MOSS + DiariZen 人数共识；FireRedASR2-AED / MOSS / Qwen3-ASR-0.6B	✅ 已实现（2026-09-10）：每条样本并发运行 MOSS 和 DiariZen；中英文且两边有效人数都 <=1 时取 FireRed，中英文其它情况取 MOSS，其他/混合/未知取 Qwen3 ASR；不含时间戳/speaker ID，不读取输入 transcript
语言		
language_content.language	annotation[].transcription.language	FireRedLID	✅ 已实现（2026-09-17 更新）：先并行运行 MOSS/官方 DiariZen，优先用语言无关的官方 DiariZen timeline（不可用时用 MOSS）做 speaker-first LID；选中 timeline 先扣除 overlap，每个 speaker 只保留最长连续 speech 片段。至少两个说话人的最长片段 >=2 秒且合计 >=3 秒时，各运行一次 FireRed LID，并按实际输入片段时长投票，结果直接驱动语言路由；不拼接多个片段，平票或任一路径失败回退整段 LID。空 transcript 的 language-only 请求默认启动内部 speaker-v2，但不公开填充 speaker 字段；不从 ASR 文本猜测
language_content.word_count	annotation[].transcription.word_count	确定性脚本	✅ 已实现
language_content.punctuation	annotation[].transcription.punctuation	确定性脚本	✅ 已实现
language_content.repetition	annotation[].transcription.repeat_times	确定性脚本	✅ 已实现
language_content.filler	annotation[].transcription.filler_count	确定性脚本	✅ 已实现
deterministic_derived.speaker_change_rate_per_min	（结构性 contract 字段）	确定性公式	✅ 已实现：`round(speaker_change_count * 60 / duration_sec, 6)`
deterministic_derived.filler_ratio	（结构性 contract 字段）	确定性公式	✅ 已实现：`round(filler / word_count, 6)`

## 当前链路

入口 `scripts/run_tagger.py` → `tagger/pipelines/tagging.py`，当前 stage：

```
language_deterministic / audio_probe / silence / speaker /
brouhaha / dnsmos / firered_aed / dass / recrir / firered_lid /
deterministic_derived
```

`panns` stage 与 `sound_field_scene.sound` 字段已于 2026-08-25 废弃删除
（DASS 为背景噪音主模型，具体标签由 `noise_composition` 展开）；PANNs
工具模块保留，仅作后续交叉验证 evidence 用，不注册、不进入公开输出。

输出为内部 schema（`basic_acoustic.*`、`audio_quality.*`、`room_acoustic.*`、
`sound_field_scene.*`、`speaker.*`、`language_content.*`、`deterministic_derived.*`）的 tags-only JSONL，
详细说明见 `docs/pipeline-architecture.md`。命名已于 2026-08-23 按本表第一列
统一。

（历史 2026-09-04 pack 记录）非中文链路默认使用 `.runtime/openasr/0.1.40/openasr` 一次加载 FireRed q4_k、
DiariZen fp16、ReDimNet fp16、Qwen3-ForcedAligner q4_k 和 bundled FireRed Stream-VAD。
四个显式 pack 分别为
`models/firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr`、
`models/diarizen-large-s80-v2/diarizen-large-s80-v2-fp16.oasr`、
`models/redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr`、
`models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr`。DiariZen pack 的 SHA-256 为
`1d4b2c2e9fe3a3f6eebfc8f2280bca04dcbc6f9b14e4d9cbccb0fdca40e88311`，上游 source
metadata 为 `BUT-FIT/diarizen-wavlm-large-s80-md-v2`；它是模型 pack，不是 standalone
runtime。许可证：OpenASR/FireRed 为 Apache-2.0，DiariZen 为 CC BY-NC 4.0，
ReDimNet 为 MIT，Qwen3-ForcedAligner 为 Apache-2.0。aligner pack 的 SHA-256 为
`5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b`。stdout 文本与 version-locked trace raw turns 分别产出 ASR 与
C/M/O/X，并按组件独立置空；LID 失败按 unknown 进入该分支，不回退 MOSS。
当前 policy 为 `speaker_v2.claim_policy.20260910.1`，C/M/O/X rule 固定使用
`selection="language_routed"` 且没有 fallback source；resolver 必须收到样本
`language_route`。实际 route 始终记录 `language_selected_source`；source 可用时
记录 `selection="language_route"`，不可用时记录 `selection="none"`、原因与 null
claim。独立 speaker CLI 通过 `--openasr-enable` /
`--openasr-disable` 整体控制组合分支。OpenASR binary（包括路径覆盖后的 resolved
executable）首次使用对应文件身份时强制
校验 `openasr 0.1.40`、42,268,384 bytes 和 SHA-256
`98d973d0d72b47198ecf17e4c00b2710d84c351b77d3c8024a7256bcea59a1a5`，结果写入
runtime provenance。
该路由规则不等于全语种能力：catalog 只声明 FireRed/ForcedAligner 的 en/zh 支持。
英文端到端 smoke 通过；真实法语样本无 hint 时出现 exit 0 但 ASR 语义错误、timeline
正常，显式 `--language fr` 则 exit 6。因此非 en/zh ASR 尚未获生产验证，当前结构
校验不能自动识别 silent wrong transcript。

## 结构性 gap

1. **公司 schema 导出层缺失（最大 gap）**：pipeline 输出的是内部 schema，目前
   没有任何 stage 或脚本把内部 tags 转成公司格式（`attribute.*` +
   `annotation[].*`）。根目录 `schema.json` 和 `company-label-alignment.md`
   已定义目标格式与映射规则，但无对应代码。
2. `bridge/sure-tagger` 是包含旧 topic 实验的历史语言层原型，输出的是
   sure_tagger 自有 `{value, confidence, ...}` 格式，也不是当前公开 schema。
   当前 sure-tagger 不产出 topic；开放短语 topic 由下游语言模型生成。

## 下一步清单

1. 未实现标签：
   - `far_field`（暂时不实现）
   - `acoustic_scene`（暂时不实现）
2. 实现内部 schema → 公司 `annotation[]` 格式的导出/桥接层（可复用
   `company-label-alignment.md` 的映射规则）。
3. 可选增强：Brouhaha C50 与 Rec-RIR C50 的自动交叉验证；DASS 对 AED
   music_present/speech_music_events 的验证。
