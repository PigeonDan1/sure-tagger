# 公司音频标签对齐命名方案

本文档定义 sure-tagger 打标结果与公司音频样本标签格式的字段命名对齐方式。

参考规范：

- `bridge/公司音频标签规范/测试集-数据集标签.md`
- `bridge/公司音频标签规范/测试集-样本标签.md`
- `bridge/公司音频标签规范/训练集-样本标签.md`

## 1. 原则

1. sure-tagger 的输出是样本级标签，不回写公司数据集标签。
2. 公司样本标签对象分为两层：顶层 `attribute` 保存整条音频文件的属性；
   `annotation[]` 保存每个切分片段或 utterance 的标注。
3. `channels`、`sample_rate`、`duration`、`path`、`size`、`file_type` 这类
   文件级属性写入 `attribute`，不写入 `annotation[]`。
4. topic、speaker、VAD、transcription、声音事件、声学质量等和具体切分片段
   相关的结果写入对应的 `annotation[]` 元素。
5. 如果输入已经有数据集标签，仍保留在 `corpus.native_metadata` 或原始数据集
   metadata 中，只作为先验、fallback 或校验值使用。
6. 不新增 `custom.sure_tagger`。公司当前没有的字段，按语义加入对应样本标签
   分组，作为建议扩展字段。
7. 数值标签不要硬塞进枚举字段。枚举字段只写派生结果，原始数值写到对应数值字段。

## 2. 目标样本标签结构

公司侧实际样本标签结构示例：

```json
{
  "sample_id": "amicorpus_AMI_IS1007c_H03",
  "parent_sample_id": "amicorpus_AMI_IS1007c_H03",
  "attribute": {
    "duration": 2113109,
    "path": "/mnt/lustre/.../IS1007c.Headset-3.wav",
    "size": 67619542,
    "channels": 1,
    "sample_rate": 16000,
    "file_type": "audio"
  },
  "annotation": [
    {
      "seg_id": "000000000",
      "timestamp": {
        "begin_time": 812.81,
        "end_time": 813.13
      },
      "transcription": {
        "language": "en",
        "text": ["YEAH"]
      }
    }
  ]
}
```

对齐规则：

- `attribute.duration` 沿用公司样本格式，建议写毫秒整数：
  `round(basic_acoustic.duration_sec * 1000)`。
- `annotation[].timestamp.begin_time` 和 `annotation[].timestamp.end_time`
  表示切分片段在父音频中的时间，单位秒，应优先保留输入已有值。
- 如果 sure-tagger 输入本身就是一个 utterance 音频切片，则输出一个
  `annotation[]` 元素；若输入是带多个切分片段的父音频，则把打标结果写回
  对应 `seg_id` 或时间区间匹配的 `annotation[]` 元素。
- 如果某个结果只对整条父音频有效，不能可靠对应到某个 `annotation[]`，
  不应复制到每个片段；需要重新按片段运行，或与公司确认是否接受
  `attribute` 下的文件级扩展字段。

## 3. 数据集 metadata 输入

数据集标签作为输入先验时，使用公司数据集标签原结构放入 manifest：

```json
{
  "corpus": {
    "native_metadata": {
      "application_domain": "会议办公",
      "audio": {
        "acoustic": {
          "sample_rate": 16000,
          "channels": 1,
          "snr_estimation": ">20db高信噪比",
          "background": "quiet"
        },
        "speech": {
          "language": "en",
          "dialect": "unknown"
        },
        "tag": ["speech"],
        "voiceprint": {
          "speaker_count": "多人"
        }
      }
    }
  }
}
```

这些输入只作为先验或 fallback。若样本级模型/确定性脚本实际检测结果与先验冲突，
样本级结果应独立保留在样本标签对象中。

## 4. 可从数据集 metadata 读取的先验

| 数据集 metadata 字段 | 对应 sure-tagger 标签 | 样本标签落点 |
| --- | --- | --- |
| `audio.acoustic.sample_rate` | `basic_acoustic.sample_rate_hz` | `attribute.sample_rate` |
| `audio.acoustic.channels` | `basic_acoustic.channels` | `attribute.channels` |
| `audio.acoustic.snr_estimation` | `audio_quality.snr_db` | 只能作为粗档位 fallback；精确值写 `annotation[].audio_quality.snr`。 |
| `audio.acoustic.background` | `sound_field_scene.sound` / `music` | 作为片段背景先验，输出写 `annotation[].audio.acoustic.background`。 |
| `audio.acoustic.distance` | `room_acoustic.far_field` | 输出写 `annotation[].speaker.distance`。 |
| `audio.speech.language` | `language_content.language` | 输出写 `annotation[].transcription.language`。 |
| `audio.speech.dialect` | 无当前 sure-tagger 标签 | 原样保留；当前不自动打方言。 |
| `audio.tag` | `sound_field_scene.speech_music_events` / `music` / `sound` | 作为粗类先验；派生输出写 `annotation[].audio.tag`。 |
| `audio.voiceprint.speaker_count` | `speaker.speaker_count` / `speaker.multi_speaker` | 精确可解析时写 `annotation[].speaker.speaker_count`；`单人`/`双人` 可映射为 `1`/`2`，`多人`/`嘈杂人群` 只能作为 `multi_speaker=true` 的先验，不能伪造精确人数。 |

## 5. 全量标签一对一映射

下面是 sure-tagger 当前全部公开标签的主映射。每个 sure-tagger tag 只指定一个
公司样本标签主落点；少数字段仍可在导出时派生额外枚举，但不改变这里的主映射。

状态说明：

- `已有-实际样本`：你提供的公司实际样本 JSON 中已经出现。
- `已有-测试集样本`：公司测试集样本规范中已经存在。
- `已有-训练集样本`：公司训练集样本规范中已经存在，建议测试集也复用。
- `待添加`：公司当前样本规范中没有，建议新增到样本标签内。

| sure-tagger 标签 | 公司样本标签主落点 | 状态 | 写入规则 |
| --- | --- | --- | --- |
| `basic_acoustic.duration_sec` | `attribute.duration` | 已有-实际样本 | 秒转毫秒整数，例如 `2113.109 -> 2113109`。 |
| `basic_acoustic.sample_rate_hz` | `attribute.sample_rate` | 已有-实际样本 | 直接写 Hz 数值，例如 `16000`。 |
| `basic_acoustic.channels` | `attribute.channels` | 已有-实际样本 | 直接写声道数，例如 `1`。 |
| `basic_acoustic.nonspeech_ratio` | `annotation[].vad.nonspeech_ratio` | 待添加 | 写 `[0, 1]` 数值，按当前 segment/utterance 计算。 |
| `basic_acoustic.silence_segments` | `annotation[].vad.silence_segments` | 待添加 | 数组元素使用 `start_sec`、`end_sec`，时间相对当前 segment。 |
| `audio_quality.snr_db` | `annotation[].audio_quality.snr` | 已有-训练集样本 | 写精确 dB 数值。 |
| `audio_quality.dnsmos_sig` | `annotation[].audio_quality.dnsmos_sig` | 待添加 | DNSMOS SIG 分数。 |
| `audio_quality.dnsmos_bak` | `annotation[].audio_quality.dnsmos_bak` | 待添加 | DNSMOS BAK 分数。 |
| `audio_quality.dnsmos_ovrl` | `annotation[].audio_quality.dnsmos` | 已有-训练集样本 | 作为主 DNSMOS 分数写入已有 `dnsmos` 字段。 |
| `audio_quality.dnsmos_p808` | `annotation[].audio_quality.dnsmos_p808` | 待添加 | DNSMOS P.808 分数。 |
| `room_acoustic.far_field` | `annotation[].speaker.distance` | 已有-测试集样本 | 映射为 `near` / `far`；无法判断时不写。 |
| `room_acoustic.rt60_sec` | `annotation[].room_acoustic.rt60` | 待添加 | 单位秒。 |
| `room_acoustic.c50_db` | `annotation[].room_acoustic.c50` | 待添加 | Rec-RIR 派生 C50；Brouhaha C50 已降级为内部 evidence（`internal.brouhaha_c50_db`）。 |
| `sound_field_scene.speech_music_events` | `annotation[].task_extension.sound_event` | 已有-训练集样本 | 写 FireRed AED 事件，如 `speech,singing,music`。 |
| `sound_field_scene.music_present` | `annotation[].others.music_state` | 已有-测试集样本 | true 写 `是`，false 写 `否`。 |
| `sound_field_scene.external_noise_type` | `annotation[].sweeper_scene.external_noise_info.type` | 已有-测试集样本 | 写 DASS 检出类别（docs/DASS.md 类别键，多个可逗号连接）；具体标签由 `noise_composition` 展开。 |
| `speaker.speaker_count` | `annotation[].speaker.speaker_count` | 待添加 | 非负整数，表示当前片段内有效说话人数。 |
| `speaker.multi_speaker` | `annotation[].speaker.multi_speaker` | 待添加 | boolean；与 `speaker_count >= 2` 保持一致。 |
| `speaker.speaker_change_count` | `annotation[].speaker.speaker_change_count` | 待添加 | 非负整数，表示当前片段内说话人切换次数。 |
| `speaker.speaker_change` | `annotation[].speaker.speaker_change` | 待添加 | boolean；表示是否存在符合当前语义版本的说话人切换。 |
| `speaker.overlap_ratio` | `annotation[].speaker.overlap_ratio` | 待添加 | `[0, 1]` 数值，固定为 overlap 时长除以 speech union 时长。 |
| `speaker.speaker_overlap` | `annotation[].speaker.speaker_overlap` | 待添加 | boolean；表示是否达到当前 overlap 判定条件。 |
| `language_content.topic` | `annotation[].topic` | 已有-训练集样本 | 写 `major_topic/minor_topic`。 |
| `language_content.language` | `annotation[].transcription.language` | 已有-测试集样本 | 直接写语言码，如 `en`、`zh`。 |
| `language_content.word_count` | `annotation[].transcription.word_count` | 待添加 | integer。 |
| `language_content.punctuation` | `annotation[].transcription.punctuation` | 待添加 | 保留 `punctuation_count` 和 `has_terminal_punctuation`。 |
| `language_content.repetition` | `annotation[].transcription.repeat_times` | 已有-测试集样本 | 写 `repetition_count`；`has_repetition` 可由 `repeat_times > 0` 派生。 |
| `language_content.filler` | `annotation[].transcription.filler_count` | 待添加 | integer。 |

需要额外注意的合并关系：

- `sound_field_scene.speech_music_events` 和 `sound_field_scene.external_noise_type`
  都可以在展示层汇总为 sound event，但主落点保持上表不变，避免 FireRed AED
  事件和 DASS 背景噪音混在一个不可追踪字段里。
- `audio_quality.snr_db` 可以额外派生
  `annotation[].audio_quality.snr_estimation`，但主数值字段仍是
  `annotation[].audio_quality.snr`。
- `basic_acoustic.nonspeech_ratio` 可以额外派生
  `annotation[].transcription.speech_status`，但主数值字段仍是
  `annotation[].vad.nonspeech_ratio`。

## 6. 样本级输出字段

### 6.1 顶层 `attribute`

这些字段描述整条音频文件本身，优先写入 `attribute`。

| sure-tagger/输入字段 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `sample.sample_id` | `sample_id` | 保留输入样本 ID。 |
| 父音频 ID | `parent_sample_id` | 有父音频时保留父 ID；没有时可等于 `sample_id`。 |
| `sample.audio.path` | `attribute.path` | 保留公司可访问的音频路径。 |
| 文件大小 | `attribute.size` | 从文件系统或输入 metadata 写入字节数。 |
| 音频类型 | `attribute.file_type` | 音频固定写 `audio`。 |
| `basic_acoustic.duration_sec` | `attribute.duration` | 写毫秒整数，例如 `2113.109s -> 2113109`。 |
| `basic_acoustic.sample_rate_hz` | `attribute.sample_rate` | 直接写采样率，例如 `16000`。 |
| `basic_acoustic.channels` | `attribute.channels` | 直接写声道数，例如 `1`。 |

公司测试集样本规范中也有 `annotation[].sample_rate`，但在你给出的实际 JSON 中
采样率位于 `attribute.sample_rate`。因此 sure-tagger 的新输出以
`attribute.sample_rate` 为准；只有公司下游明确要求每个片段重复采样率时，才镜像
到 `annotation[].sample_rate`。

### 6.2 基础音频与声学质量

这些结果通常应按 utterance/segment 写入对应的 `annotation[]`。如果当前运行只对
整条父音频做了检测，不要默认复制到所有片段。

| sure-tagger 标签 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `audio_quality.snr_db` | `annotation[].audio_quality.snr` | 复用训练集样本字段，写精确 dB 数值。 |
| `audio_quality.dnsmos_ovrl` | `annotation[].audio_quality.dnsmos` | 复用训练集样本字段，作为主 DNSMOS 分数。 |
| `audio_quality.dnsmos_sig` | `annotation[].audio_quality.dnsmos_sig` | 建议扩展字段。 |
| `audio_quality.dnsmos_bak` | `annotation[].audio_quality.dnsmos_bak` | 建议扩展字段。 |
| `audio_quality.dnsmos_p808` | `annotation[].audio_quality.dnsmos_p808` | 建议扩展字段。 |

### 6.3 VAD 与非语音

| sure-tagger 标签 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `basic_acoustic.nonspeech_ratio` | `annotation[].vad.nonspeech_ratio` | 建议扩展字段，写 `[0, 1]` 数值。 |
| `basic_acoustic.silence_segments` | `annotation[].vad.silence_segments` | 建议扩展字段，元素使用 `start_sec`、`end_sec`，相对当前 segment。 |
| `basic_acoustic.nonspeech_ratio` | `annotation[].transcription.speech_status` | 派生枚举：有语音写 `speech`，整段无可闻内容写 `sil`，音频不可用写 `invalid`。 |

### 6.4 声场、混响与声音事件

| sure-tagger 标签 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `room_acoustic.far_field` | `annotation[].speaker.distance` | 有可靠值时映射 `near`/`far`；当前为 null 时不写。 |
| `room_acoustic.rt60_sec` | `annotation[].room_acoustic.rt60` | 建议扩展字段，单位秒。 |
| `room_acoustic.c50_db` | `annotation[].room_acoustic.c50` | 建议扩展字段；注明为 Rec-RIR C50。 |
| `sound_field_scene.speech_music_events` | `annotation[].task_extension.sound_event` | 复用训练集样本字段。多个事件建议用逗号分隔字符串，或与公司确认是否可扩为数组。 |
| `sound_field_scene.music_present` | `annotation[].others.music_state` | true 写 `是`，false 写 `否`。 |
| `sound_field_scene.external_noise_type` | `annotation[].sweeper_scene.external_noise_info.type` | 写 DASS 检出类别（docs/DASS.md 类别键，多个可逗号连接）；具体标签由 `noise_composition` 展开。 |

### 6.5 说话人

| sure-tagger 标签 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `speaker.speaker_count` | `annotation[].speaker.speaker_count` | 建议扩展字段，写非负整数。 |
| `speaker.multi_speaker` | `annotation[].speaker.multi_speaker` | 建议扩展字段，写 boolean。 |
| `speaker.speaker_change_count` | `annotation[].speaker.speaker_change_count` | 建议扩展字段，写非负整数。 |
| `speaker.speaker_change` | `annotation[].speaker.speaker_change` | 建议扩展字段，写 boolean。 |
| `speaker.overlap_ratio` | `annotation[].speaker.overlap_ratio` | 建议扩展字段，写 `[0, 1]` 数值；分母为当前片段的 speech union 时长。 |
| `speaker.speaker_overlap` | `annotation[].speaker.speaker_overlap` | 建议扩展字段，写 boolean。 |

数值与 boolean 字段必须保持同一条模型/时间轴链路的口径。`multi_speaker` 应与
`speaker_count >= 2` 一致；`speaker_change_count == 0` 时 `speaker_change` 必须为
false。`speaker_overlap` 仍按当前语义版本的事件与阈值规则生成，不建议下游仅根据
四舍五入后的 `overlap_ratio` 自行重算。

### 6.6 语言内容

| sure-tagger 标签 | 公司样本字段 | 命名/写入规则 |
| --- | --- | --- |
| `language_content.topic` | `annotation[].topic` | 复用训练集样本字段，写 `major_topic/minor_topic`。 |
| `language_content.language` | `annotation[].transcription.language` | 直接写语言码。 |
| transcript 输入 | `annotation[].transcription.text` | 保留数组格式，例如 `[transcript]`。 |
| `language_content.word_count` | `annotation[].transcription.word_count` | 建议扩展字段，写 integer。 |
| `language_content.punctuation` | `annotation[].transcription.punctuation` | 建议扩展字段，保留 `punctuation_count` 和 `has_terminal_punctuation`。 |
| `language_content.repetition` | `annotation[].transcription.repeat_times` | 可写 `repetition_count`。 |
| `language_content.filler` | `annotation[].transcription.filler_count` | 建议扩展字段，写 integer。 |

## 7. 样本级派生枚举

这些字段虽然可以从 sure-tagger 结果派生，但仍写入样本标签对象，而不是回写数据集
标签。如果数据集标签中原本已有同名枚举，保持原值不改。

### 7.1 SNR 档位

| `audio_quality.snr_db` | 样本字段 `annotation[].audio_quality.snr_estimation` |
| --- | --- |
| `snr_db > 20` | `>20db高信噪比` |
| `10 < snr_db <= 20` | `10~20db中信噪比` |
| `0 <= snr_db <= 10` | `0~10db低信噪比` |
| `snr_db < 0` | `<0db极低信噪比` |

### 7.2 音频类型

| 条件 | 样本字段 `annotation[].audio.tag` 加入值 |
| --- | --- |
| transcript 非空，或 `speech_music_events` 包含 `speech` | `speech` |
| `music_present == true`，或 `speech_music_events` 包含 `music` | `music` |
| `sound` 非空，或 SNR 明显较低 | `noise` |
| `speech_music_events` 包含非 `speech` / `music` 事件 | `audio_event` |
| 没有任何可判定信息 | `other` |

### 7.3 背景噪声

| 条件 | 样本字段 `annotation[].audio.acoustic.background` |
| --- | --- |
| `sound` 为空，`music_present == false`，且 SNR 较高 | `quiet` |
| 有 speech 且有明显 music/noise | `mix` |
| 有明显背景声、音乐或低 SNR | `noisy` |

### 7.4 语音状态

| 条件 | 样本字段 `annotation[].transcription.speech_status` |
| --- | --- |
| transcript 非空，或检测到 speech | `speech` |
| `nonspeech_ratio` 接近 1，且无 speech/transcript | `sil` |
| 音频文件不可读、严重乱码或无法打标 | `invalid` |

## 8. 建议新增样本字段清单

公司实际样本 JSON 已经使用或建议直接复用的字段：

```text
sample_id
parent_sample_id
attribute.path
attribute.size
attribute.duration
attribute.channels
attribute.sample_rate
attribute.file_type
annotation[].seg_id
annotation[].timestamp.begin_time
annotation[].timestamp.end_time
annotation[].transcription.language
annotation[].transcription.text
annotation[].transcription.speech_status
annotation[].transcription.repeat_times
annotation[].speaker.distance
annotation[].sweeper_scene.distance
annotation[].sweeper_scene.external_noise_info.type
annotation[].others.music_state
```

训练集样本规范中已存在、建议测试集复用的字段：

```text
annotation[].audio_quality.snr
annotation[].audio_quality.dnsmos
annotation[].topic
annotation[].task_extension.sound_event
```

公司当前没有但与现有分组语义一致、建议新增到样本标签中的字段：

```text
annotation[].audio.tag
annotation[].audio.acoustic.background
annotation[].audio_quality.dnsmos_sig
annotation[].audio_quality.dnsmos_bak
annotation[].audio_quality.dnsmos_p808
annotation[].audio_quality.snr_estimation
annotation[].vad.nonspeech_ratio
annotation[].vad.silence_segments
annotation[].room_acoustic.rt60
annotation[].room_acoustic.c50
annotation[].speaker.speaker_count
annotation[].speaker.multi_speaker
annotation[].speaker.speaker_change_count
annotation[].speaker.speaker_change
annotation[].speaker.overlap_ratio
annotation[].speaker.speaker_overlap
annotation[].transcription.word_count
annotation[].transcription.punctuation
annotation[].transcription.filler_count
```
