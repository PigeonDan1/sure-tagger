# saasr-sim-mini 配置说明

本目录中的 YAML 文件用途不同。用于实际运行的主配置是“场景配置”；说话人音色和环境噪声示例则是被主配置引用的辅助配置，不能都直接作为一次运行的入口。

## 配置文件之间的关系

以 `configs/mini_dotstts.yaml` 为例，运行时的关系如下：

```text
configs/mini_dotstts.yaml
├── scenes[].dialogue
│   └── configs/dialogues/demo_2spk.json
└── speech_backend.voices
    └── configs/voices.yaml
```

三类文件的职责是：

| 文件类型 | 示例 | 作用 |
|---|---|---|
| 场景配置 YAML | `configs/mini_dotstts.yaml` | 定义 TTS 后端、说话人数、轮次/重叠、噪声、房间和输出场景，是 `saasr-mini run` 的入口。 |
| 说话人配置 YAML | `configs/voices.yaml` | 将 speaker 编号映射到参考音频、参考文本和语言，用于 dots.tts 零样本音色克隆。 |
| 对白配置 JSON | `configs/dialogues/demo_2spk.json` | 定义每句文本、说话人编号、角色和 `overlap_hint`。它不是 YAML，但由场景 YAML 引用。 |

此外，`configs/environment_noise_examples.yaml` 只是环境噪声 profile 的示例集合，说明应如何给场景添加噪声，不是一个完整可运行场景。

## 示例：`mini_dotstts.yaml`

下面是该配置的核心结构：

```yaml
seed: 20260805
audio:
  sample_rate: 16000
  peak_dbfs: -1.0
speech_backend:
  type: dots_tts
  model_dir: ../../rednote-hilab__dots.tts-base
  device: cuda:0
  precision: float16
  num_steps: 10
  guidance_scale: 1.2
  voices: voices.yaml
scenes:
  - id: demo_2spk
    dialogue: dialogues/demo_2spk.json
    speakers: 2
    overlap_ratio: 0.0
    max_simultaneous_speakers: 1
    snr_db: null
    room:
      dimensions_m: [5.2, 4.1, 2.8]
      microphone_position_m: [2.6, 2.0, 1.2]
      speaker_positions_m:
        - [1.3, 1.5, 1.3]
        - [3.9, 2.7, 1.3]
      rt60_seconds: 0.24
```

### 顶层与音频参数

- `seed`：仿真中需要随机数的部分使用的种子，主要影响 RIR、噪声等。它不会自动成为 dots.tts 的生成种子；如果需要固定 TTS 结果，应在 `speech_backend` 下单独配置 `seed`。
- `audio.sample_rate`：最终输出采样率。dots.tts 通常生成 48 kHz 音频，管线会在排程和混音前重采样到这里的值。
- `audio.peak_dbfs`：最终混合音频的峰值上限。超过该值时整体衰减；这是上限而不是目标响度，因此不会把过小的音频自动放大。
- `audio.speaker_rms_dbfs`：可选的每说话人活动段 RMS 校准目标。本例没有配置，所以不会执行说话人级统一增益。

### `speech_backend`

- `type: dots_tts`：使用本地 dots.tts JSON-RPC 后端，而不是内置的 fixture 语音生成器。
- `model_dir`：dots.tts 模型目录。相对路径相对于当前 YAML 文件所在目录解析。
- `device`：推理设备，例如 `cuda:0`。
- `precision`：推理精度，例如 `float16`。
- `num_steps`：TTS 推理步数。步数越高通常计算量越大，质量和稳定性需要结合实际模型验证。
- `guidance_scale`：文本条件引导强度，控制模型对输入条件的遵循程度。
- `voices`：引用说话人配置 YAML。该文件中每个键（如 `'0'`、`'1'`）对应一个 speaker 编号，并包含 `prompt_audio`、`prompt_text` 和 `language`。

本例的两个英文说话人由 `voices.yaml` 中的 0、1 号 AMI headset 音色提供。对白文件只使用编号，不重复存储音色信息。

### `speech_edit_backend`

`speech_edit_backend` 是可选的 TTS 后处理阶段，和生成音色的 `speech_backend` 相互独立。使用本地 dots.tts.edit checkpoint 调整整句语速时可配置：

```yaml
speech_edit_backend:
  type: dots_tts_edit
  model_dir: ../../dots-studio__dots.tts.edit
  device: cuda:0
  precision: bfloat16
  default_rate: 0.88
  use_xvector: auto
  speaker_scale: 1.5
  num_steps: 10
  guidance_scale: 1.2
  seed: 20260414
```

- `default_rate`：为每条语音生成 `<rate, factor=...>` 原生编辑指令；小于 1 表示减慢，大于 1 表示加快。checkpoint 接受 `0.55..0.99` 或 `1.01..1.95`，步进为 `0.01`。
- `device`：`cuda:N` 会通过独立子进程的 `CUDA_VISIBLE_DEVICES` 绑定设备；`cpu` 会隐藏 CUDA，但实际速度和精度是否可用仍取决于 checkpoint。
- `use_xvector: auto`：交给 edit runtime 按指令类型决定是否启用源说话人引导。语速编辑会自动启用。
- `num_steps`、`guidance_scale`、`speaker_scale`：直接传给 edit runtime；省略前两项时使用 checkpoint 的默认采样设置。
- `checkpoint_path`、`python`、`work_dir`：均可按需覆盖；未配置时分别使用 `<model_dir>/checkpoints`、`<model_dir>/.venv/bin/python` 和 `<model_dir>/artifacts/saasr_edit_outputs`。

需要逐句控制时，在对白 utterance 中使用 `edit_instruction`，它优先于 `default_rate`：

```json
{
  "utterance_id": "u000",
  "speaker_index": 0,
  "text": "今天的测试顺利完成了。",
  "edit_instruction": "<pitch, semitones=-5>今天的测试顺利完成了。</pitch>"
}
```

该字段是 dots.tts.edit 的原生 markup，不是自然语言 instruction。当前封装固定 `target_text=source_text`，适用于语速、音高、情绪、停顿等不改转写文本的编辑；插入、删除、替换字词暂未接入 SA-ASR 标注协议。

### `scenes`

`scenes` 是场景列表，因此一个主 YAML 可以一次定义多个场景。每个场景的字段含义如下：

- `id`：场景 ID，也用于组织输出目录和 manifest 记录。
- `dialogue`：相对于当前配置文件的对白 JSON 路径。
- `speakers`：说话人数。本例为 2，必须和对白中的 speaker 编号、房间中的位置数量匹配。
- `overlap_ratio`：目标同时说话比例。本例为 0；调度器会结合实际 TTS 时长和对白中的 overlap hint 排程。
- `max_simultaneous_speakers`：并发说话人数上限。本例最多一人说话。
- `snr_db`：混合语音相对于背景噪声的目标信噪比。本例为 `null`，即不加背景噪声。
- `environment`：场景的声学环境标签。可选 `dry`、`meeting_room`、`open_space`、`restaurant`、`public_transit`、`vehicle`、`residential`。
- `noise_environment` / `rir_environment`：只有当资产文件名无法自动识别环境时才需要显式填写；填写后必须与 `environment` 一致。

对白 JSON 中的 `overlap_hint` 常用值包括：

- `none`：普通轮次，倾向于不重叠；
- `backchannel`：较强的附和/插入重叠倾向；
- `interrupt`：中等强度的打断倾向。

这些 hint 是调度倾向，不是单句的绝对重叠保证；最终结果还受实际语音时长和并发上限影响。

### `room`

- `dimensions_m`：房间长、宽、高，单位为米。
- `microphone_position_m`：麦克风三维坐标 `[x, y, z]`。
- `speaker_positions_m`：每个说话人的三维坐标，数量必须等于 `speakers`。
- `rt60_seconds`：目标混响时间。本例为 0.24 秒。

默认情况下，管线依据说话人与麦克风的位置，为每个说话人生成确定性近似 RIR。若场景提供 `rir_path`，则从该 WAV 读取真实/测量 RIR（自动重采样），并为每个说话人写出对应 wet stem；manifest 会记录 `rir_backend=audio_asset` 和文件名。

使用外部测量 RIR 时，一个场景必须保持同一声学条件：

- `rir_path`：所有说话人共享同一条 RIR，适合单场景的统一房间条件；
- `rir_paths`：当前只接受每个条目解析后都指向同一个文件；不同文件会在配置加载阶段直接报错。

当前 RIR 目录没有可靠的“同一房间/同一麦克风阵列”家族元数据，因此不能仅凭文件名允许跨文件混用。将来补充经过验证的 RIR 家族注册表后，才可以根据房间尺寸、说话人与麦克风位置为各说话人选择同一家族中的不同 RIR。

噪声和 RIR 也必须属于同一声学环境。例如，会议室场景可以使用 `h060_Office_ConferenceRoom_3txts.wav` 与 `OMEETING`/`OOFFICE`，但不能搭配 `TCAR` 或户外 RIR。相同的 dialogue JSON 可以被多个 scene 重复引用，以构造同一段发言在会议室、开阔空间、餐厅或车辆内的对照样本；环境标签和对应资产由每个 scene 单独声明。

```yaml
environment: restaurant
noise_path: ../../background_noise_library/demand_16k/PCAFETER/PCAFETER_ch01_clip00.wav
rir_path: ../../rir/Audio/h109_CoffeeShop_1txts.wav
```

## 实际生成流程

```text
参考音频/参考文本 + 对白文本
        ↓
逐句、逐说话人调用 dots.tts
        ↓
可选：释放 base 模型，逐句调用 dots.tts.edit
        ↓
可选的每说话人 RMS 校准
        ↓
按照对白和 overlap_ratio 安排时间戳
        ↓
生成各说话人的 dry stem
        ↓
按房间位置施加各自 RIR，或按 `rir_path` 施加外部 RIR，生成 wet stem
        ↓
混合说话人并按 snr_db 添加一个场景级噪声轨
        ↓
施加最终峰值上限并导出 mixture、stems、RIR、RTTM 和 manifest
```

启用编辑时会保留三个源语音层级：`source_utterances` 是 edit 前音频，`raw_utterances` 是 edit 后、RMS 校准前音频，`utterances` 是 RMS 校准后、排程所用音频。base 和 edit 不会同时驻留，适合在同一张显卡上顺序运行；多场景任务会在场景边界按需重新加载模型。

运行示例：

```bash
saasr-mini run \
  --config configs/mini_dotstts.yaml \
  --output demo_dotstts_output
```

如果没有设置 `noise_profile` 或 `noise_path`，场景会使用默认的合成噪声；若要指定办公室或机场等环境，应在场景中增加例如：

```yaml
snr_db: 12.0
noise_profile: office
```

`noise_profile` 和 `noise_path` 不能同时设置。完整的环境噪声说明见 [`BACKGROUND_NOISE.md`](BACKGROUND_NOISE.md)。
