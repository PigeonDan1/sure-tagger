# Tagging Pipeline 标签与打标方法

本文档说明 `scripts/run_tagger.py` 当前能够生成的公开标签、每个标签的
计算方法、音频预处理方式，以及输入和输出 JSON 示例。

Pipeline 主入口、stage 调度、fallback、无 transcript guard 和补标模式见
`docs/pipeline-architecture.md`。

## 1. 运行方式

默认运行（DASS 使用 GPU）：

```bash
python3 scripts/run_tagger.py \
  --manifest phase1_asr_samples/manifest.jsonl \
  --output phase1_asr_samples/outputs/full_pipeline_tags.jsonl
```

在当前部署环境中让 Rec-RIR 使用 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_tagger.py \
  --manifest phase1_asr_samples/manifest.jsonl \
  --output phase1_asr_samples/outputs/full_pipeline_tags.jsonl \
  --recrir-use-gpu
```

模型和独立 runtime 的路径配置在 `tagger/local_config.py`。DASS 默认使用项目内
`.runtime/dass_runtime_py311` 的 CUDA runtime；需要 CPU 调试时传
`--dass-use-cpu`。FireRed VAD/AED/LID 与 Brouhaha 的通用工具配置默认走 CPU。
speaker-v2 的其他语言分支使用两个独立
runtime：Qwen3-ASR-0.6B 负责 ASR 文本；官方 DiariZen 负责 speaker timeline。
任一路失败不会覆盖另一路的有效结果。

官方 DiariZen 部署在 `models/diarizen-official/`，使用：

- `BUT-FIT/diarizen-wavlm-large-s80-md-v2` segmenter；
- `pyannote/wespeaker-voxceleb-resnet34-LM` speaker embedder；
- 模型发布的 `config.toml`、VBx clustering、PLDA 与 x-vector transform；
- `.runtime/diarizen_official_py310/bin/python` 中的 pinned DiariZen source 和
  vendored `pyannote.audio`。

segmenter checkpoint SHA-256 为
`689a10f0cbde6345b4bf9b1670cf3196488634b34e5b07bea0fe40d508e1196b`，
WeSpeaker checkpoint SHA-256 为
`366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56`。
完整版本、文件和验证记录见 `models/diarizen-official/DEPLOYMENT.json`。pipeline
使用输入第 0 声道；短于 16 秒时静音补齐，输出裁回原时长。中英文共识使用的 FireRed ASR 由
`.runtime/fireredasr2_aed_py311_torch280_cu128_v1/bin/python` 调用本地
`models/FireRedASR2-AED`，不参与 DiariZen clustering。
其他语言 ASR 由 `.runtime/qwen3_asr_0_6b_py311_torch280_cu128_v1/bin/python`
调用本地 `models/Qwen3-ASR-0.6B`，同样不参与 DiariZen clustering。

只跑某一个样本的完整链路：

```bash
python3 scripts/run_tagger.py \
  --manifest phase2_asr_sample/manifest.jsonl \
  --output outputs/one_sample_tags.jsonl \
  --sample-id EN2001a_utterance_00000 \
  --speaker-profile quality-shadow
```

基于已有 tags-only 输出补某个样本的指定标签：

```bash
python3 scripts/run_tagger.py \
  --manifest phase2_asr_sample/manifest.jsonl \
  --input-tags outputs/phase2_full_pipeline_tags.jsonl \
  --output outputs/phase2_asr_patch.jsonl \
  --sample-id EN2001a_utterance_00000 \
  --only-tags speaker.asr_transcript \
  --missing-only
```

`--only-tags` 支持公开 tag path、分组名或 stage 名，例如 `speaker`、
`basic_acoustic.nonspeech_ratio`、`speaker.asr_transcript`、`recrir`。配合
`--missing-only` 时，已有非空字段不会被覆盖。

## 2. 输入 JSONL

manifest 每行是一个封闭的 raw-only JSON 对象。例如：

```json
{
  "corpus": {
    "dataset_name": "phase1_asr_samples",
    "source_urls": {
      "article": [],
      "github": [],
      "huggingface": [],
      "dataset_card": []
    },
    "native_metadata": {}
  },
  "sample": {
    "sample_id": "example_001",
    "audio": {
      "path": "audio/example.wav"
    },
    "text": {
      "transcript": ""
    },
    "native_metadata": {}
  }
}
```

音频路径可以是绝对路径，也可以是相对当前工作目录或 manifest 所在目录的
路径。完整输入约束见 `development.md`。

## 3. 当前可打标签

当前 tagging pipeline 定义了 33 个公开对象字段（对应 structural schema 的 35 个叶子路径）。`room_acoustic.far_field`
仍是预留字段，暂时输出 `null`。sure-tagger 不再公开 topic；下游语言模型
可基于确定性标签和 `speaker.asr_transcript` 推断一个不限定值域的描述性
`topic` 短语。

### 3.1 基础声学标签

| Tag | 类型 / 单位 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `basic_acoustic.duration_sec` | number / 秒 | 使用 Python WAV header；不支持时使用 `ffprobe` | 原始音频持续时间，保留 6 位小数。 |
| `basic_acoustic.sample_rate_hz` | integer / Hz | 读取原始音频流元数据 | 源文件采样率，不是模型重采样后的采样率。 |
| `basic_acoustic.channels` | integer | 读取原始音频流元数据 | 源文件声道数，不是模型降混后的声道数。 |
| `basic_acoustic.silence_segments` | array | 优先使用 native metadata 中的 speech/silence/utterance/word segments；没有可用 metadata 时用 FireRed VAD | 未检测到人类语音的时间段，每项为 `{"start_sec": number, "end_sec": number}`。 |
| `basic_acoustic.nonspeech_ratio` | number / `[0, 1]` | 未检测到人类语音的区间总时长除以 `duration_sec` | 非语音占整段音频的比例；不是物理静音比例。 |

### 3.2 音频质量标签

| Tag | 类型 / 单位 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `audio_quality.snr_db` | number / dB | Brouhaha 输出逐帧 SNR，再对所有有效预测取算术均值 | 语音相对背景噪声的强度；通常越大越好。 |
| `audio_quality.dnsmos_sig` | number / MOS `[1, 5]` | DNSMOS P.835 SIG | 语音信号本身的质量和失真程度；越高越好。 |
| `audio_quality.dnsmos_bak` | number / MOS `[1, 5]` | DNSMOS P.835 BAK | 背景噪声的不干扰程度；越高越好。 |
| `audio_quality.dnsmos_ovrl` | number / MOS `[1, 5]` | DNSMOS P.835 OVRL | 综合语音和背景影响的整体质量；越高越好。 |
| `audio_quality.dnsmos_p808` | number / MOS `[1, 5]` | DNSMOS P.808 | P.808 模型预测的主观整体质量；越高越好。 |

DNSMOS 将短于 9.01 秒的音频循环补足，使用 9.01 秒窗口和 1 秒 hop，
最后对所有完整窗口的分数取均值。

Brouhaha 同时预测逐帧 C50，但该值只作为内部 evidence
（`internal.brouhaha_c50_db`）用于与 `room_acoustic.c50_db` 交叉验证，
不进入公开输出。

### 3.3 空间声学标签

| Tag | 类型 / 单位 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `room_acoustic.rt60_sec` | number / 秒 | Rec-RIR 估计 RIR；对 Schroeder 能量衰减曲线的 `-5 dB` 到 `-25 dB` 区间做 T20 线性拟合，并外推到 `-60 dB` | 混响衰减时间；通常越大表示混响尾部越长。 |
| `room_acoustic.c50_db` | number / dB | 在 Rec-RIR 估计结果中定位最大绝对幅值的直达声，计算其后 50 ms 早期能量与剩余晚期能量之比 `10*log10(E_early/E_late)` | 基于估计 RIR 的清晰度；越大通常越清晰。 |

`room_acoustic.c50_db` 是 Rec-RIR 的物理指标；Brouhaha 的直接 C50 模型预测
只在内部 evidence 中保留，两者来源不同，不能互相覆盖。

### 3.4 声场和声音事件标签

| Tag | 类型 / 单位 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `sound_field_scene.speech_music_events` | string array | FireRed AED | 检出的 `speech`、`singing`、`music` 类别，始终按这个固定顺序排列。`singing`、`music` 仅在事件占比达到 `--firered-aed-min-singing-ratio` / `--firered-aed-min-music-ratio`（均默认 0.10，2026-08-25 起：caption_pairs_3000 上语音帧被帧级误判为歌声/音乐产生的短段占比均低于 0.10，真实歌声/音乐占比远高于此）时进入数组。事件时间段和帧比例仅保留为内部 evidence。 |
| `sound_field_scene.music_present` | boolean | FireRed AED | `speech_music_events` 是否包含 `music`（受上述占比门控）。保留该字段用于直接进行音乐样本筛选。 |
| `sound_field_scene.external_noise_type` | string array | DASS AudioSet-2M | 检出的 docs/DASS.md 噪音类别键数组：`music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment`（人类声音与未归类标签永不公开）。类别由全量 527 类向量中未被排除且达到默认阈值 `0.25`（2026-08-24 在 phase2 上校准：DASS-medium 真实噪声类分数偏软 0.1–0.45，干净语音低于 0.15）的标签归组而来，按各类别最高分降序排列；具体标签见 `noise_composition`。 |
| `sound_field_scene.noise_composition` | object | DASS AudioSet-2M + FireRed AED 门控 | 按 docs/DASS.md 类别归组的背景声组成，展开 `external_noise_type` 每个类别的具体标签。固定含 `music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment` 六个键，每键为按分数降序的标签名数组；每类最多 `--dass-composition-top-k`（默认 3）个，入组阈值 `--dass-composition-threshold`（默认 `0.25`，2026-08-25 起与类别阈值对齐——此前 0.30 会在 0.25–0.30 分数段产生「有类别、无组成」的空档）。音乐类别以 FireRed AED 为准：`music_present` 为 `false` 时为空数组，为 `true` 或 AED 未运行（`null`）时输出 DASS 音乐类标签；人类声音与未归类标签只进内部 evidence。成功无检出时各键为空数组，工具失败时整个字段为 `null`。 |

`sound_field_scene.sound` 与 panns stage 已于 2026-08-25 废弃删除
（`noise_composition` 取代其功能）。PANNs 工具模块保留供后续交叉验证
evidence 使用，但不注册、不可选择、不进入公开输出。

DASS 的默认阈值是 `0.25`，按互不重叠分块（上游 extractor 固化的 10.24
秒窗口）取每类最大概率。`external_noise_type` 输出的是 docs/DASS.md 的
类别键：全量 527 类向量中任一**未被排除**的标签达到阈值，其所属类别
即出现在结果里，按各类别最高分降序排列；排除政策（主语音、静音、声学
场景、混响、回声）同样作用于类别推导——Silence 不会把干净语音样本标
成 `formless`，`--no-exclusion` 时才放开；人类声音与未归类标签永不公开。阈值从 AudioSet 惯例 `0.50` 下调是 phase2 校准的结果：
DASS-medium 对真实噪声类输出的 sigmoid 分数普遍偏软（0.1–0.45），而
干净语音样本的可输出类最高分不超过 0.15，`0.25` 能在不误报干净语音的
前提下找回真实噪声标签。排除策略只影响内部 evidence 的 top 事件
（默认排除主语音、静音、声学场景、混响和回声；全有/全无，传
`--no-exclusion` 后全部关闭）。

`noise_composition` 把 `external_noise_type` 的每个类别展开为具体标签：
它对全量 527 维 sigmoid 向量按 docs/DASS.md 的 7 类能力划分归组（不受
排除策略影响），每类取分数最高的前 3 个且不低于 0.25 的标签（与
`external_noise_type` 的类别阈值对齐，保证有类别的行组成非空）。音乐类以
FireRed AED 的 `music_present` 门控——AED 判定无音乐时音乐桶为空数组，
即使 DASS 有音乐类高分标签；AED 未运行时不做门控。人类声音类别和
未归类标签只保留在内部 evidence 的 `category_events` 中，不进入公开
输出；声道/环境/背景类别（inside/outside、reverberation、echo 等）的
分数也保留在 evidence 中，作为后续 far_field 和混响标签的补充证据
来源。

### 3.5 说话人标签

| Tag | 类型 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `speaker.speaker_count` | non-negative integer | speaker v2 的 count claim；中文和英文使用 MOSS timeline，其他语言使用官方 DiariZen timeline | 样本内解析出的说话人数；其他语言由 WavLM large-s80-v2、pyannote WeSpeaker 和 VBx/PLDA 完成 diarization。 |
| `speaker.speaker_present` | boolean | 从已校验的 `speaker_count` 确定性派生 | 是否存在说话人；人数大于 0 为 `true`，等于 0 为 `false`，人数未知时为 `null`。 |
| `speaker.multi_speaker` | boolean | 对语言路由选中的 MOSS 或官方 DiariZen timeline 执行 `speaker_count >= 2` | 是否包含两个或更多不同说话人。 |
| `speaker.speaker_change_count` | non-negative integer | 从 speaker v2 为 change claim 选中的 timeline 派生 | 样本内说话人切换次数。 |
| `speaker.speaker_change` | boolean | 对语言路由选中的 MOSS 或官方 DiariZen timeline 检测切换 | 是否发生说话人切换。 |
| `speaker.overlap_ratio` | number / `[0, 1]` | 从 speaker v2 为 overlap claim 选中的 timeline 派生，分母为 speech union duration | 重叠发言时长占有效语音时长的比例。 |
| `speaker.speaker_overlap` | boolean | 对语言路由选中的 MOSS 或官方 DiariZen timeline 检测重叠 | 是否存在多人同时发言。 |
| `speaker.profiles` | array of object / nullable | speaker v2 画像：复用语言路由选中的 decision timeline；语速、音高和音量为确定性计算，性别使用独立语言路由模型 | 按各说话人第一次发言的时间排列为 `speaker_1`、`speaker_2`…。每项含 `gender`（`male`/`female`/`null`）、`speech_rate`、`pitch` 和 `speaker_volume`。`zh`/`en` 性别使用 Alice，`sw` 及已配置非洲语言使用 AfriSpeech，其余使用 Griko；单说话人输入整段音频，多说话人仅拼接无 overlap 片段，净时长小于 1 秒或模型失败时性别为 `null`。`speech_rate.band` 取 `slow`/`normal`/`fast`/`variable`，`unit` 取 `zh_char_per_sec`/`word_per_min`；累计有效语音不足 3 秒或文本单位不足 8 时为 `null`。`pitch` 是相对 F0 档位，不用于派生性别；`speaker_volume` 是片段内相对响度。无法得到可靠时间轴时 profiles 为 `null`，确认没有语音时为 `[]`。模型分数和原始 F0/RMS/区间只进内部 artifact。 |
| `speaker.asr_transcript` | string / nullable | 每条样本一起运行 MOSS 与 DiariZen；中英文且两边有效人数都 `<=1` 时取 FireRedASR2-AED，中英文其它情况取 MOSS，其它/混合/未知语言取 Qwen3-ASR-0.6B | 整段音频的上游 ASR 文本；去除首尾空白，不含时间戳和 speaker ID。绝不以 `sample.text.transcript` 补值；任一人数缺失时中英文使用 MOSS，选定路径的文本缺失或非法时为 `null`，且不跨候选补位。路由 source、两边人数和语言判断保存在 speaker artifact。 |

总线直接调用 `tagger/pipelines/speaker_evidence.py`，默认 profile 是
`quality-shadow`，不再读取 native metadata 生成 speaker 公开值，也不存在旧的
MOSS enable 门禁或 channel-activity 分流。每条样本并发运行 MOSS 与官方 DiariZen；
FireRed LID 仍选择公开 speaker timeline；中英文且双边有效人数都 `<=1` 时 ASR
改选 FireRed，中英文其它情况使用 MOSS，其他语言使用 Qwen3 ASR。各结果分别验证，缺失人数不触发共识，选中 ASR 失败时
也不跨候选补值。可用 `--speaker-profile lean-shadow` 选择较低成本的辅助诊断组合。证据、
时间线、对齐结果和 claim fusion 只保存为内部
artifact，不进入公开 tags-only 输出。

当前 policy 为 `speaker_v2.claim_policy.20260910.1`。C/M/O/X rule 使用
`selection="language_routed"`，resolver 必须接收样本的 `language_route` 对象，
route 缺失或非法时不从 evidence availability 推断，也没有另一分支 fallback。
实际 route 始终记录 `language_selected_source`；该 source 可用时记录
`selection="language_route"`，不可用时记录 `selection="none"`、失败原因和 `null`
claim。独立 speaker CLI 使用 `--firered-asr-enable/disable`、
`--qwen3-asr-enable/disable` 和 `--diarizen-enable/disable` 分别控制；旧
`--openasr-enable/disable` 仅为兼容组别名。

### 3.6 语言内容标签

这些标签优先读取 `sample.text.transcript`。当输入 transcript 为空且需要文本
标签时，pipeline 会使用 `speaker.asr_transcript` 作为替代文本。这里不包含
描述性 topic；topic 由 sure-tagger 下游语言模型生成开放短语。
其中“文本标签”只指 `word_count`、`punctuation`、`repetition` 和 `filler`。
`language` 始终来自 FireRed LID：只请求该字段时直接运行 LID；speaker 已因其它字段
运行时复用其路由使用的同一 observation/evidence，不从 ASR 文本推断。

| Tag | 类型 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `language_content.language` | string | standalone FireRed LID 音频模型 | 输出语言/方言码，并驱动 speaker-v2 的中文/英文 MOSS/FireRed、其他语言 Qwen3 ASR + 官方 DiariZen 路由；失败时该字段为 `null`，speaker route 按 unknown 进入其他语言分支。 |
| `language_content.word_count` | integer | simple multilingual tokenizer | transcript 中的词数。 |
| `language_content.punctuation` | object | Unicode punctuation counter | `punctuation_count` 和 `has_terminal_punctuation`。 |
| `language_content.repetition` | object | consecutive token ngram rule | `has_repetition` 和 `repetition_count`。 |
| `language_content.filler` | integer | filler lexicon rule | filler token 数量。 |

### 3.7 确定性派生标签

| Tag | 类型 | 打标方法 | 含义 |
| --- | --- | --- | --- |
| `deterministic_derived.speaker_change_rate_per_min` | number | `round(speaker_change_count * 60 / duration_sec, 6)` | 每分钟说话人切换次数；依赖缺失或时长不为正时为 `null`。 |
| `deterministic_derived.filler_ratio` | number | `round(filler / word_count, 6)` | filler 占词数比例；依赖缺失或词数不为正时为 `null`。 |

## 4. 模型输入预处理

Pipeline 保留原音频文件，不会先生成一个供所有模型共享的转换版本。每个
模型适配器根据自身输入要求单独降混和重采样。语言内容标签读取原始
transcript；为空时读取 speaker-v2 ASR，不额外处理音频。

| 模块 | 实际送入模型的格式 | 处理方式 |
| --- | --- | --- |
| 音频元数据探测 | 原始格式 | 只读取源文件，不重采样、不降混。 |
| Native metadata VAD | 原始 `sample.native_metadata` | 如果存在 `silence_segments`、`speech_segments`、`vad_segments`、`segments`、`utterances` 或 `words`，直接用确定性脚本转换为非语音 tags。 |
| FireRed VAD / AED | 16 kHz、单通道、16-bit PCM WAV | VAD 只在没有可用 metadata speech segments 时运行；AED 不满足要求时使用 FFmpeg 生成临时 WAV，推理后删除。 |
| Brouhaha | 16 kHz、单通道 | pyannote 在内存中对多通道取均值并重采样。 |
| DNSMOS | 16 kHz、单通道 | 多通道取均值，使用 librosa 重采样。 |
| Rec-RIR | 16 kHz、单通道 | torchaudio 对多通道取均值并重采样，临时 WAV 推理后删除。 |
| Speaker v2 | 各模型适配器要求的单通道采样率 | 每条样本并发运行 MOSS joint ASR/diarization 与官方 DiariZen；speaker C/M/O/X 仍按语言选择原 timeline。中英文且双边人数都 `<=1` 时 ASR 使用 FireRedASR2-AED，中英文其它情况使用 MOSS，其他语言使用 Qwen3-ASR-0.6B。DiariZen 取第 0 声道，短于 16 秒时内部补零并裁回原时长；ECAPA、独立 FireRed VAD 和 Brouhaha 仅作辅助 evidence。 |
| 确定性语言内容 | 原始 transcript，或 transcript 为空时的 speaker-v2 ASR | 不做音频预处理；只对 `word_count`、`punctuation`、`repetition`、`filler` 使用该文本。`language` 始终来自 FireRed LID；空 transcript 时复用 speaker 路由已经产生的同一 observation/evidence。 |

因此 8 kHz 或双通道音频可以正常进入模型。需要注意，8 kHz 上采样只能
满足模型输入格式，无法恢复原音频 4 kHz 以上已经不存在的频率信息；多通道
取平均也可能丢失通道间差异。

## 5. 输出 JSON 示例

公开输出也是 JSONL，每个输入样本对应一行，只包含 tag 值。输出行和输入
manifest 行顺序一致，但不会包含 `sample_id`、模型名、分数、置信度、
warning 或推理证据。

下面是当前 Pipeline 实际输出结构的完整示例：

```json
{
  "basic_acoustic": {
    "channels": 2,
    "duration_sec": 15.963062,
    "sample_rate_hz": 16000,
    "nonspeech_ratio": 0.674248,
    "silence_segments": [
      {"start_sec": 0.0, "end_sec": 3.29},
      {"start_sec": 3.83, "end_sec": 4.15},
      {"start_sec": 5.21, "end_sec": 5.33},
      {"start_sec": 8.41, "end_sec": 13.22},
      {"start_sec": 13.74, "end_sec": 15.963062}
    ]
  },
  "audio_quality": {
    "dnsmos_bak": 1.066441,
    "dnsmos_ovrl": 1.045056,
    "dnsmos_p808": 2.162502,
    "dnsmos_sig": 1.174226,
    "snr_db": -1.159983
  },
  "room_acoustic": {
    "far_field": null,
    "rt60_sec": 0.801379,
    "c50_db": 14.032025
  },
  "sound_field_scene": {
    "speech_music_events": ["singing", "music"],
    "music_present": true,
    "external_noise_type": ["music"],
    "noise_composition": {
      "music": ["Background music"],
      "animal": [],
      "mechanical": [],
      "nature": [],
      "formless": [],
      "channel_environment": []
    }
  },
  "speaker": {
    "speaker_count": null,
    "speaker_present": null,
    "multi_speaker": null,
    "speaker_change_count": null,
    "speaker_change": null,
    "overlap_ratio": null,
    "speaker_overlap": null,
    "profiles": null,
    "asr_transcript": "Good morning. Let us begin the meeting."
  },
  "language_content": {
    "language": "en",
    "word_count": 12,
    "punctuation": {
      "punctuation_count": 2,
      "has_terminal_punctuation": true
    },
    "repetition": {
      "has_repetition": false,
      "repetition_count": 0
    },
    "filler": 1
  },
  "deterministic_derived": {
    "speaker_change_rate_per_min": null,
    "filler_ratio": 0.083333
  }
}
```

列表字段成功执行但没有检出时输出空数组，例如：

```json
{
  "speech_music_events": ["speech"],
  "music_present": false,
  "external_noise_type": []
}
```

工具未部署、音频无法读取、推理失败或结果校验失败时，对应字段输出
`null`。不同模型相互隔离，例如 DASS 失败只会使 `external_noise_type` 和
`noise_composition` 为 `null`，不会清空 FireRed 的 `speech_music_events`
和 `music_present`。

## 6. 当前预留字段

以下字段属于公开 schema，但尚未接入当前 tagging pipeline 的已注册工具，
因此目前输出 `null`：

| Tag | 计划含义 |
| --- | --- |
| `room_acoustic.far_field` | 是否为远场拾音或远距离声源。 |

## 7. 内部产物

Rec-RIR 估计的完整 RIR 波形不进入公开 JSON。默认保存位置为输出文件同级的
`artifacts/rir/` 目录，例如：

```text
phase1_asr_samples/outputs/artifacts/rir/
```

公开输出只保留从该 RIR 计算得到的 `room_acoustic.rt60_sec` 和
`room_acoustic.c50_db`。

Speaker v2 的内部产物默认保存在：

```text
<output-directory>/artifacts/speaker_v2/<row-key>/
```

其中包含逐模型 evidence、timeline alignment 和 fusion artifact；重复的
`sample_id` 通过 manifest 行号组成的 row key 隔离。
