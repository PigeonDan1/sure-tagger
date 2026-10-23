# SA-ASR 数据合成计划

版本：v0.2（能力边界与当前实现）  
更新日期：2026-09-06

## 1. 目标与原则

本项目用于构造可控、可追溯的 SA-ASR（speaker-attributed ASR）检测数据。第一阶段不追求大规模，而是先建立约十几条高质量、覆盖面清楚的基准语音，再围绕每条基准做单变量和场景变量调整。

核心原则：

1. 每条变体只改变一个主要因素，保留 `source_id`、参数和测量差异。
2. 将 base TTS、edit checkpoint 重生成和房间/噪声仿真分开标记。
3. 不把“请求成功”当作“控制生效”；所有控制都要有参数记录和可检查的验收条件。
4. 保留干净语音、单人 stem、混合音频和标注，支持从单句检测扩展到多说话人场景。

## 2. 当前系统能力范围

当前系统由 `saasr-sim-mini` 渲染管线和 `dots.tts-base` wrapper 组成。主要入口和实现位置如下：

- `src/saasr_mini/pipeline.py`：加载对白、调用语音后端、排程、混音、导出标注。
- `src/saasr_mini/speech.py`：离线 fixture 后端和 dots.tts JSON-RPC 后端。
- `src/saasr_mini/scheduling.py`：按 utterance 时长和 overlap hint 计算时间戳。
- `scripts/gen_academic_meeting_suite.py`：生成当前学术研讨会代表性多说话人评测集。
- `rednote-hilab__dots.tts-base/model.py`：dots.tts 的本地 JSON-RPC wrapper。

### 2.1 语音生成与音色

当前可以：

- 使用 `rednote-hilab/dots.tts-base` 本地 checkpoint 生成 48 kHz 单声道 WAV。
- 使用参考音频加对应转写进行零样本音色克隆；每个 speaker 在 `voices*.yaml` 中配置一组 `prompt_audio`、`prompt_text` 和 `language`。
- 原有会议 prompt 库仍由 `scripts/prepare_meeting_refs.py` 维护：中文 6 条来自 AliMeeting near-field participant/headset，英文 9 条来自 AMI headset；来源区间、通道、校准增益和哈希记录在 `configs/refs/meeting_selection.json` 与 `configs/refs/manifest.tsv`。
- 独立说话人扩展库另行维护：`configs/refs/alimeeting_headset_100` 保留 60 个 participant-level AliMeeting near-field prompt；`configs/refs/aishell3_speaker_pool` 保留 Hugging Face 镜像实际可下载的 107 个 AISHELL-3 独立说话人，每人一条 prompt；`configs/refs/aishell4_speaker_90` 提供 90 个 AISHELL-4 recording/speaker-tier 独立 prompt。三者均保留 raw/enhanced、转写和逐条 manifest，不覆盖原有 6 条中文参考。
- 当前正式参考库只保留中文和英文；跨语言实验使用中英文 speaker 混合或单句 code-switch。
- 通过固定 `seed`、`num_steps`、`guidance_scale`、`speaker_scale`、`ode_method` 和 `normalize_text` 复现实验条件。
- 通过持久化 JSON-RPC server 调用，避免每个 utterance 重复加载 2B 模型。
- 可选使用独立 `DotsTtsEditBackend` 调用 `dots.tts.edit` checkpoint；基础 TTS 和 edit 使用各自虚拟环境，并在同一场景内顺序驻留，避免同时占用显存。
- clean 基线可通过 `audio.speaker_rms_dbfs` 开启说话人级活动段 RMS 校准；每个说话人只计算一个固定增益并应用到其全部 utterance，校准前后的源文件和增益会写入 manifest。

默认链路是：dots.tts 生成 48 kHz 音频，saasr-sim-mini 根据配置需要重采样到目标采样率（常见为 16 kHz），然后再进行 RMS 校准、排程和混音。启用 edit 时，管线优先把 base 的原生 48 kHz 文件交给 edit runtime，edit 结果再重采样、校准和混音。

### 2.2 文本和模板

当前可以：

- 传入普通 `text` 或兼容 SGLang Omni 的 `input` 字段。
- 传入参考文本、语言和 `normalize_text`，支持数字、符号等文本归一化实验。
- base 数据生成只选择普通 `tts` 路径；当前数据生成器不再使用 `instruction_tts`。
- 风格和情绪控制统一记录为 edit checkpoint 原生的 `edit_instruction` 字段。
- 在对白 utterance 上设置 `input_audio_path` 可跳过 TTS 并复用已验收的基准 WAV；成组场景可使用 `rir_seed_key`、`noise_seed_key` 固定随机 realization，避免把随机差异混入单变量比较。

重要边界：

- 当前 `dots.tts-base` runtime 的 `generate()` 没有独立的 `emotion`、`pitch`、`rate` 或 `pause` 参数。
- 旧的 base `instruction_tts` 路径不作为当前数据生产能力；不能把自然语言 instruction 当作已经生效的情绪标签。

### 2.3 已实现的变量调整

#### A. dots.tts.edit 模型变量

当前已通过独立后端接入 dots.tts.edit 的原生 markup 路径：

- YAML 中的 `speech_edit_backend.default_rate` 可生成整句 `<rate, factor=...>` 指令。
- 对白中的 `edit_instruction` 可逐句覆盖默认语速，并传入 `<rate>`、`<pitch>`、`<emo>`、`<pause/>` 等 checkpoint 原生标签。
- base TTS 完成后先释放 base 模型，再加载 edit；edit 后才执行每说话人 RMS 校准和多说话人排程。
- 输出保留 `source_utterances`（edit 前）、`raw_utterances`（edit 后、校准前）和 `utterances`（校准后）三层。
- manifest 同时记录文件总时长比和活动语音跨度比。edit runtime 会把首尾静音规范化到约 250 ms，不能把文件总时长变化直接解释为语速变化。

这条路径属于模型重生成，不是确定性 DSP。已验证 checkpoint、RPC 和 48 kHz 输入链路可以运行，但单条 `factor=0.90` 冒烟样本的活动跨度只从 2.182 秒变为 2.214 秒，因此当前仍需逐条验收控制强度，不能按 factor 直接推导实际时长，也不能自动宣称情绪语义生效。

当前封装固定 `target_text=source_text`，只接入不改变标注转写的韵律编辑。字词插入、删除和替换虽然是 checkpoint 能力，但尚未接入 SA-ASR 的源文本/目标文本标注协议。

#### B. 场景和混音变量

当前渲染器可以：

- 支持 1 至 12 个 speaker，配置每个 speaker 的位置和参考音色。
- 通过 `overlap_ratio`、`max_simultaneous_speakers` 以及 `none`、`backchannel`、`interrupt` 三类 overlap hint 构造轮次和重叠。
- 生成每个 speaker 的 dry stem、经过 RIR 的 wet stem 以及最终 mixture。
- 配置房间尺寸、麦克风位置、speaker 位置和 `rt60_seconds`。
- 可在场景中用 `rir_path` 接入 `rir/Audio` 下的真实/测量脉冲响应；未提供时使用确定性近似 RIR。一个场景内的 `rir_paths` 必须全部解析到同一个 RIR 文件；当前没有经过验证的同房间 RIR 家族元数据，因此混用不同测量场景会被配置校验拒绝。
- 每个带噪声或外部 RIR 的 scene 必须声明 `environment`；系统会校验噪声和 RIR 与 `meeting_room`、`open_space`、`restaurant`、`public_transit`、`vehicle` 等环境一致。相同 dialogue 可以被多个 scene 复用，从而比较同一段发言在不同环境中的表现。
- 在配置 SNR 时叠加确定性噪声，并记录 target/achieved SNR；现已支持 `office`、`airport`、
  `station`、`train`、`cafe`、`street` 六类程序化环境 profile，也可以读取外部 WAV 资产。
- 导出 `segments.rttm`、`manifest.json` 和 `exports/sa_asr.jsonl`。

未指定 `rir_path` 时，RIR 是基于几何反射和衰减的确定性近似；指定后可使用
`rir/Audio` 下的外部 WAV。环境 profile 是 license-free 的程序化 fixture；场景可用
`noise_profile` 选择 profile，或用 `noise_path` 接入已授权的真实背景录音。

### 2.4 当前不具备或暂不承诺的能力

- 基于自然语言 instruction 的稳定、可重复、可判定情绪控制。
- `emotion="happy"` 这类官方离散情绪参数。
- dots.tts-base 模型内部的原生 pitch/rate/pause 控制；这些能力当前来自独立 edit checkpoint。
- dots.tts.edit 对 rate、pitch、emotion、pause 请求强度的确定性保证。
- 语义情绪标签与声学表现的一一对应保证。
- 真实多人录音中的串扰、设备频响、非平稳背景噪声和实测 RIR。
- 自动判断“听起来是否开心/悲伤”的可靠情绪分类器。
- 仅凭客观 F0、RMS、时长指标证明情绪语义已经生效。

## 3. 第一阶段：代表性基准语音

### 3.1 规模

先制作 12 至 16 条基准 utterance，建议首批固定为 15 条：5 类内容场景 × 3 个长度档位。每条只生成一个干净、无混响、无噪声的原始版本，之后所有变量变体都从这条原始版本派生。

长度档位建议：

- 短句：约 1--2 秒，适合检测端点、停顿和短语音误差。
- 中句：约 3--5 秒，作为主要 A/B 基准。
- 长句：约 7--10 秒，覆盖长上下文、连续语音和排程边界。

内容场景建议：

1. 陈述/确认：例如测试结果、项目状态、执行确认。
2. 提问/应答：包含问号、短回答和轮次切换。
3. 数字/时间/专名：用于文本归一化和识别易错项。
4. 中英混码或技术术语：用于代码切换和词边界。
5. 带标点的自然段：用于停顿、换气和长句切分。

第一阶段的基准文本应尽量情绪中性，不在正文中加入“开心、悲伤”等情绪词，避免把文本内容变化误当作情绪控制。

### 3.2 基准记录

每条基准必须记录：

```json
{
  "id": "zh_base_001",
  "text": "...",
  "language": "zh",
  "speaker_id": "zh_alimeeting_00",
  "reference_audio": "configs/refs/audio_enhanced/zh_alimeeting_00.wav",
  "reference_text": "...",
  "seed": 42,
  "model": "rednote-hilab/dots.tts-base",
  "sample_rate": 48000,
  "generator": "dots_tts_backend"
}
```

同时保存音频哈希、时长、峰值、RMS、F0 统计和生成参数。基准音频通过验收后才允许派生变体。

## 4. 第二阶段：单变量扩展

每条基准先做 isolated pairs，再做少量三档或五档梯度。建议矩阵如下：

| 维度 | 首批建议档位 | 备注 |
|---|---|---|
| pitch | `<pitch, ...>` 请求 / 不编辑 | 记录原始 markup，并以 F0 指标和人工听感验收 |
| speed | `<rate, factor=...>` 请求 / 不编辑 | 记录实际活动跨度比，不按 factor 推导时长 |
| pause | `<pause/>` 请求 / 不编辑 | 记录请求位置并检查实际输出，不承诺精确毫秒数 |
| emotion | `<emo, emotion=...>` 请求 / 不编辑 | 只标记 requested emotion，必须单独做语义验收 |
| gain | 场景级或说话人校准增益 | 与 edit 请求分开记录，避免混淆响度和风格 |
| text normalization | off / on | 只选数字、时间、符号样本 |
| seed | 至少 3 个 seed | 用于估计模型自然韵律波动，不作为控制旋钮 |

不同 edit 类型分别建立数据族；组合 edit 只有在单变量样本通过验收后才加入。每个样本保留完整 markup、实际声学测量，并从 base 版本建立单向 source 链。

## 5. 第三阶段：场景组合

单变量通过验收后，再将基准或其变体放入 SA-ASR 场景：

1. 单 speaker、无噪声、无重叠：检查文本和时间戳基础质量。
2. 2 speaker、低重叠：检查普通轮次切换和 speaker attribution。
3. 2--4 speaker、中等重叠：覆盖 backchannel 和 interrupt。
4. 4--12 speaker、不同距离和 RT60：覆盖拥挤会议及并发上限。
5. SNR 梯度：建议先做 `30 / 20 / 10 dB`，再决定是否加入更低 SNR。

每种场景至少保留一份 clean mixture、若干含噪/含混响 mixture、所有 speaker stems、RIR、RTTM 和完整 manifest。

组合实验必须引用已经存在的基准或变体 ID，不直接在场景脚本中重新生成一份不可追溯的语音。

## 6. 质检与验收

### 6.1 文件级检查

- WAV 可读、非空、单声道、采样率正确。
- 无 NaN/Inf，无明显削波；记录 peak 和 clipping fraction。
- 时长在预期范围内；edit 变体记录文件时长比和活动跨度比。
- 变体与 source 的 sample rate、speaker、文本和 reference 保持一致。

### 6.2 控制级检查

- pitch：F0 中位数方向正确，时长变化不应成为主要差异。
- speed：活动跨度方向正确，并记录 F0 是否发生系统性漂移。
- pause：人工和声学检测均确认请求位置附近出现预期停顿变化。
- gain：RMS/峰值方向正确，但不得引入削波。
- emotion：只记录请求标签；只有通过主观或分类器验收后才能标记为控制生效。
- seed：同一输入、同一参数、同一 seed 的结果可复现；不同 seed 作为自然波动样本单独标记。

### 6.3 语音与场景检查

- 对基准和每类变量至少人工抽听一组 A/B。
- 对含文本的样本使用可用 ASR 做转写核对；转写失败时记录为环境阻塞，不默认为音频正确。
- 对多人场景检查 RTTM、stem 长度、重叠比例和 achieved SNR 与 manifest 一致。
- 对低能量参考音频、明显噪声或异常短/长输出建立 rejected 列表，不进入主数据集。

## 7. 数据组织与版本

建议目录结构：

```text
outputs_academic_meeting_suite_v1/
├── sources/
│   ├── prepared/
│   └── edited/
├── sessions/<session_id>/
│   ├── mixture.wav
│   ├── dry_stems/
│   ├── stems/
│   ├── rirs/
│   └── segments.rttm
├── manifests/sessions.json
├── index.jsonl
├── qc_report.json
├── summary.json
└── README.md
```

命名建议：`<base_id>__<family>__<label>__v<version>.wav`。manifest 至少包含：`id`、`source_id`、`file`、`text`、`language`、`speaker_id`、`edit_family`、`label`、`parameters`、`features`、`seed`、`generator`、`model_revision` 和 `sha256`。

版本号只在数据协议、基准集合或变量定义变化时递增；单纯重跑同一参数应保留运行时间和 hash，不覆盖历史结果。

## 8. 里程碑

1. **M0：能力冻结**：确认 15 条基准文本、参考音色、采样率、seed 和 manifest 字段。
2. **M1：基准生成**：生成并验收 clean base utterance。
3. **M2：单变量集**：完成已验收 dots.tts.edit 变体的成对/梯度样本。
4. **M3：场景集**：将通过验收的语音组合为 1/2/4/12 speaker、重叠和 SNR 场景。
5. **M4：评测包**：导出统一 JSONL/RTTM、质量报告和 rejected 清单，供检测评测使用。

## 9. 当前决策与后续问题

- `dots.tts.edit` 已作为独立后端接入；只有通过声学指标和主观验收的样本才能进入对应变量集，失败样本进入 rejected 清单。
- 第一阶段优先保证 source/variant 可追溯和单变量隔离，再扩大语言、speaker 数量和声学环境。
- 如果需要把输出标成真正的语义情绪，仍需单独评估 `dots.tts.edit` 的情绪可判定性，或改用经过情绪对齐的 checkpoint；不能只因 wrapper 接受 `<emo>` 标签就认定控制成功。
