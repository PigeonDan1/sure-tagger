# 音频标签 Schema 简明标注说明

本文是 [`tag-schema.md`](tag-schema.md) 的标注版摘要，供批量数据标注使用。每个音频样本输出一个
JSON 对象（批量文件通常为 JSONL，每行一个样本）。字段名和枚举值必须使用英文；自然
语言描述可以使用中文或项目指定的统一语言。

## 1. 通用规则

### 1.1 `null`、`[]` 和数值

| 写法 | 含义 | 示例 |
| --- | --- | --- |
| `null` | 不知道、证据不足、无法评估或不适用 | 无法可靠判断说话人数时 `speaker_count: null` |
| `[]` | 已确认没有可报告项目 | 确认没有音乐事件时 `speech_music_events: []` |
| `0` / `false` | 已确认数值为零或条件不成立 | 确认没有说话人时 `speaker_count: 0`、`speaker_present: false` |

不要把“没有听到”直接当成“确认不存在”。不能测量的 SNR、DNSMOS、RT60、C50 等字段填
`null`，不要凭听感编造数值；不能确认数组是否为空时填 `null`，不要填 `[]`。

### 1.2 说话人与背景人声的边界

`speaker.*` 只统计当前音频中作为主要语音、能够区分并参与当前交互的说话人。电视、
广播、手机外放、远处人群、另一段录音中的人声等，若只是背景声，不计入
`speaker_count`、`profiles`、`speaker_change_count` 或 `overlap_ratio`。例如：前景一人说话，
电视里还有人讲话，仍标为 `speaker_count: 1`；电视人声可以作为背景声记录，但不能新建
`speaker_2`。如果背景中的人声与前景同等重要、明确属于本段要标注的对话，则按可区分的
实际说话人计入。

`sound_field_scene.speech_music_events` 的 `speech` 表示音频中检测到人类语音事件，范围
可以比 `speaker.*` 更宽；因此背景中的电视人声可以使 `speech` 出现，但不自动增加
`speaker_count`。

### 1.3 说话人编号

`speaker.profiles[].speaker_id` 使用 `speaker_1`、`speaker_2` 等格式，按每位说话人第一
次开始发言的时间排序，而不是按模型顺序或音色排序。`speaker_descriptions` 必须引用已有
的 profile，不能创建新的身份。性别标签位于
`descriptive.speaker_descriptions[].gender`，评测时作为 `speaker.gender` 序列使用；
它不属于 `speaker.profiles`，也不能根据音高、音色或语速推断。

整段转写字段 `speaker.asr_transcript`（有时简称 speaker transcript）由系统生成，人工
标注时不填写。人工需要填写的是每个 profile 内的 `transcript`：按该说话人切分的时间段，
并记录每段对应的 ASR 文本。

最小示例：

```json
{
  "speaker_id": "speaker_1",
  "transcript": [
    {"start_sec": 0.42, "end_sec": 2.80, "text": "我们先确认一下数据。"}
  ]
}
```

### 1.4 证据边界

只标注音频或提供的结构化证据能够支持的内容。地点、房型、设备型号、职业、年龄、
身份、情绪、口音、民族、麦克风距离等不能写成确定事实；必要的推测只能写在
`scene_hypotheses`，并使用“可能”“更像”等措辞。字段之间应保持一致：已知
`speaker_count` 时，`speaker_present` 和 `multi_speaker` 按规则投影，不要随意写冲突值。

## 2. 输出骨架

```json
{
  "basic_acoustic": {},
  "audio_quality": {},
  "room_acoustic": {},
  "sound_field_scene": {},
  "speaker": {},
  "language_content": {},
  "deterministic_derived": {},
  "descriptive": {}
}
```

## 3. 结构化字段

### 3.1 基本声学 `basic_acoustic`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `basic_acoustic.duration_sec` | 音频总时长，单位秒。 | 必须是非负数；应来自音频元数据或可靠测量，不能从文件名猜测。 |
| `basic_acoustic.sample_rate_hz` | 音频采样率，单位 Hz。 | 正整数；不知道时填 `null`。 |
| `basic_acoustic.channels` | 音频声道数。 | 正整数；单声道为 `1`，立体声为 `2`。 |
| `basic_acoustic.nonspeech_ratio` | 整段中未检测到人类语音的时间比例，范围 `[0,1]`。 | 它不是“物理静音比例”：音乐、环境声和噪声也可能属于 non-speech。该字段描述语音活动，不直接决定 `speaker_count`；背景人声也不能因此自动新建 speaker profile。 |
| `basic_acoustic.silence_segments` | 非语音区间数组，每项为 `{"start_sec": number, "end_sec": number}`。 | 按起始时间排序、不重叠且不超出音频时长；不能可靠确定时间戳时为 `null`。 |

### 3.2 录音质量 `audio_quality`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `audio_quality.snr_db` | 平均信噪比，单位 dB。 | 数值越大通常表示目标语音相对背景噪声越强；需要测量值，不能用“听起来很清楚”替代。 |
| `audio_quality.dnsmos_sig` | DNSMOS P.835 的语音信号质量分数，范围通常 `[1,5]`。 | 只反映语音本身，不等同于背景噪声分数。 |
| `audio_quality.dnsmos_bak` | DNSMOS P.835 的背景噪声干扰分数，范围通常 `[1,5]`。 | 描述背景噪声是否影响听感。 |
| `audio_quality.dnsmos_ovrl` | DNSMOS P.835 的整体语音质量分数，范围通常 `[1,5]`。 | 综合语音和背景的整体质量。 |
| `audio_quality.dnsmos_p808` | DNSMOS P.808 主观整体质量分数，范围通常 `[1,5]`。 | 没有对应测量时填 `null`。 |

### 3.3 房间声学 `room_acoustic`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `room_acoustic.far_field` | 是否为远场拾音。 | 当前 schema 预留字段，通常为 `null`；不要仅凭“声音小”或“像在房间里”填写。 |
| `room_acoustic.rt60_sec` | RT60 混响时间，单位秒。 | 需要声学测量；不能把它直接写成“大房间”或“混响明显”。 |
| `room_acoustic.c50_db` | C50 清晰度指标，单位 dB。 | 是数值声学指标，不是自然语言的“清晰/不清晰”标签。 |

### 3.4 声场与声音事件 `sound_field_scene`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `sound_field_scene.speech_music_events` | 音频中出现的目标事件数组，只允许 `speech`、`singing`、`music`。 | 固定顺序为 `speech`、`singing`、`music`，不得重复。`singing` 是歌唱，`music` 是伴奏、器乐或音乐播放；两者可以同时存在。 |
| `sound_field_scene.music_present` | 是否存在音乐，布尔值。 | 只要明确听到音乐即可为 `true`；若事件数组也已知，应与其中是否包含 `music` 一致。 |
| `sound_field_scene.external_noise_type` | 背景外部声音的大类数组。 | 只能使用 `music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment`；不包含当前要统计的前景说话人。 |
| `sound_field_scene.noise_composition` | 背景声细类组成对象。 | 非 `null` 时必须同时包含上述六个键，每个键的值都是具体声音标签字符串数组；无法确认完整六类组成时整体填 `null`，不要用空对象代替。 |

### 3.5 说话人 `speaker`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `speaker.speaker_count` | 当前标注范围内可区分的主要说话人数，非负整数。 | 不计背景电视、广播、远处人群等背景人声；无法可靠区分人数时填 `null`。 |
| `speaker.speaker_present` | 是否存在主要说话人。 | 已知人数时等于 `speaker_count > 0`；它不是“是否出现任何人声”的宽泛判断。 |
| `speaker.multi_speaker` | 是否至少有两名主要说话人。 | 已知 `speaker_count` 时等于 `speaker_count >= 2`。 |
| `speaker.speaker_change_count` | 主要说话人之间发生交替的次数。 | 只统计说话人切换，不统计停顿、句号或同一人的重新起句。 |
| `speaker.speaker_change` | 是否发生过主要说话人切换。 | 通常等于 `speaker_change_count > 0`；人数为 0 或 1 时应为 `false`。 |
| `speaker.overlap_ratio` | 主要说话人重叠发言时长 / 主要语音并集时长，范围 `[0,1]`。 | 背景人声不计入分子或分母；只有一名主要说话人时通常为 `0`。 |
| `speaker.speaker_overlap` | 是否存在两名或以上主要说话人同时发言。 | 不是“有背景声”的意思；音乐、噪声和回声都不算 speaker overlap。 |
| `speaker.profiles` | 逐主要说话人的结构化画像数组。 | `[]` 表示确认没有主要说话人；无法可靠获得逐人信息时为 `null`。数组数量与已知 `speaker_count` 应一致。 |

#### `speaker.profiles[]` 内部字段

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `speaker.profiles[].speaker_id` | 样本内临时身份编号。 | 只能是 `speaker_1`、`speaker_2` 等，并按首次发言时间排序；不是现实身份。 |
| `speaker.profiles[].transcript` | 该说话人的带时间轴 ASR 片段数组。 |  每项必须包含 `start_sec`、`end_sec`、`text`；只放该 `speaker_id` 的语音，按开始时间排序。多人重叠时，不同 profile 可以有时间重叠。 |
| `speaker.profiles[].transcript[].start_sec` | 该 ASR 片段的开始时间，单位秒。 | 非负数，且必须小于 `end_sec`；相对于整段音频起点。 |
| `speaker.profiles[].transcript[].end_sec` | 该 ASR 片段的结束时间，单位秒。 | 大于 `start_sec`，不应超过音频时长；不要把下一段的静音时间算进来。 |
| `speaker.profiles[].transcript[].text` | 该时间段内的 ASR 转写结果。 | 写实际听到的语音内容；听不清的词不要臆测，可按项目约定使用不可辨识标记。不要把背景电视、广播或其他非该说话人的语音写入此处。 |
| `speaker.profiles[].speech_rate` | 该说话人的语速对象。 | 必须包含 `band`、`value`、`unit`；不能可靠计算精确值时只填可支持的档位，`value` 和 `unit` 一起为 `null`。 |
| `speaker.profiles[].speech_rate.band` | 相对语速档位。 | 只能是 `slow`、`normal`、`fast`、`variable` 或 `null`。 |
| `speaker.profiles[].speech_rate.value` | 语速数值。 | 中文使用 `zh_char_per_sec`，拉丁文字使用 `word_per_min`；没有可靠计数和时长时为 `null`。 |
| `speaker.profiles[].speech_rate.unit` | 语速单位。 | 只能是 `zh_char_per_sec`、`word_per_min` 或 `null`。 |
| `speaker.profiles[].pitch` | 片段内相对音高。 | 只能是 `low`、`mid`、`high`、`variable` 或 `null`；不得据此推断性别、年龄或情绪。 |
| `speaker.profiles[].speaker_volume` | 片段内相对音量。 | 只能是 `low`、`normal`、`loud`、`variable` 或 `null`；是相对描述，不是录音电平。 |

### 3.6 语言与文本 `language_content`

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `language_content.language` | 主要语种或语言码。 | 使用项目约定的 ISO 语言码（如 `zh`、`en`、`ja`）；混合、未知或无法判断时填 `null`，不要用“中文+英文”等自由字符串替代。 |
| `language_content.word_count` | 所有 profile transcript 的 `text` 合并后得到的词数/文本单位数。 | 按时间顺序合并各说话人文本后统计；无法可靠转写或统计时为 `null`，不要求人工补写 `speaker.asr_transcript`。 |
| `language_content.punctuation` | 合并后的 transcript 文本的标点统计对象。 | 非 `null` 时必须同时包含 `punctuation_count`（非负整数）和 `has_terminal_punctuation`（布尔值）。 |
| `language_content.repetition` | 合并后的 transcript 文本中的连续重复统计对象。 | 非 `null` 时必须同时包含 `has_repetition`（布尔值）和 `repetition_count`（非负整数）；只统计连续 token/n-gram 重复。 |
| `language_content.filler` | 合并后的 transcript 文本中的填充词数量。 | 按项目固定词表统计，如“嗯”“呃”等；词表不适用或无法确认时填 `null`，不要把普通停顿都算成 filler。 |

### 3.7 规则派生字段 `deterministic_derived`

| 字段名 | 字段定义 | 计算规则与补充说明 |
| --- | --- | --- |
| `deterministic_derived.speaker_change_rate_per_min` | 每分钟说话人切换次数。 | `round(speaker_change_count * 60 / duration_sec, 6)`；依赖缺失或时长不大于 0 时为 `null`。 |
| `deterministic_derived.filler_ratio` | filler 数量占词数的比例。 | `round(filler / word_count, 6)`；`word_count <= 0` 或依赖缺失时为 `null`。不要用派生值反推原始字段。 |

## 4. 描述性字段 `descriptive`

描述性字段用于自然语言总结，不能替代结构化事实。能被固定规则唯一确定的事实应放在
结构化字段中；描述文本只能引用已知事实，不能凭空增加地点、身份、年龄、情绪等信息。

### 4.1 顶层描述字段

| 字段名 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `descriptive.caption` | 对整段音频的一至三句综合描述。 | 概括主导内容、背景声、录音听感和说话交互；不要逐项机械复述所有数值。 |
| `descriptive.topic` | 主要语义主题短语。 | 只能根据 profile transcript 汇总的 ASR 内容或清楚听到的内容填写；`speaker.asr_transcript` 不需要人工补填。没有足够语音内容时为 `null`，不能仅凭语言、人数或声学条件猜主题。 |
| `descriptive.background_condition` | 背景整体状态的短语或短句。 | 例如“背景整体较安静”“持续存在明显噪声”；不要把没有冻结阈值的主观档位当成结构标签。 |
| `descriptive.dominant_content_impression` | 主导可听内容。 | 只能描述语音、歌唱、音乐、噪声、静音或混合内容；不写话题、情绪或说话风格。 |
| `descriptive.clarity_intelligibility_impression` | 语音清晰度和可懂度的听感描述。 | “识别出语言”或“有转写”不等于“清晰易懂”；应结合实际听感和可用质量证据。 |
| `descriptive.background_sound_summary` | 背景声的自然语言总结。 | 只总结已标注的 `external_noise_type`/`noise_composition` 或直接听到的声音，不补充具体地点、声源方位或未提供的细节。 |
| `descriptive.descriptive_phrases` | 去重的简短描述短语数组。 | 例如“多人交替发言”“背景较安静”“轻微混响”；不要写完整 transcript 句子。 |
| `descriptive.facets` | 可扩展的描述维度数组。 | 只创建有证据的维度，例如 `soundscape`、`recording_impression`、`room_acoustics`、`interaction_pattern`。 |
| `descriptive.speaker_descriptions` | 逐主要说话人的描述数组。 | 每位可区分说话人最多一项；自然语言只描述 profile 支持的语速、相对音高和相对音量。 |
| `descriptive.scene_hypotheses` | 场景或交互形态假设数组。 | 例如“可能接近双人访谈”；必须明确是推测，不能把假设写入 `speaker` 或声场结构字段。 |
| `descriptive.usability_notes` | 面向下游使用的数据建议。 | 例如“适合常规 ASR”“重叠语音可能增加 diarization 难度”“建议人工复核”。 |
| `descriptive.uncertainties` | 当前证据的缺失、冲突或判断边界。 | 记录本样本实际存在的不确定性，不要套用与样本无关的通用免责声明。 |

### 4.2 描述性数组的内部字段

| 对象字段 | 字段定义 | 补充说明 |
| --- | --- | --- |
| `descriptive.facets[].name` | 描述维度名称。 | 建议使用稳定的英文 `snake_case`，但不强制固定枚举。 |
| `descriptive.facets[].description` | 该维度的自然语言描述。 | 只写能由证据直接支持或保守综合得到的内容。 |
| `descriptive.facets[].evidence_tags` | 支撑该描述的结构化字段路径数组。 | 只能引用本样本中值非 `null` 的真实路径；没有可引用路径时使用 `[]`，不得虚构 `audio` 等路径。 |
| `descriptive.facets[].confidence` | 对该描述的自评置信度。 | 数值范围 `[0,1]`；不是经过校准的概率。 |
| `descriptive.speaker_descriptions[].speaker_id` | 被描述的 profile ID。 | 必须存在于 `speaker.profiles`，且不能新建身份。 |
| `descriptive.speaker_descriptions[].gender` | 该说话人的结构化性别标签。 | 只能是 `male`、`female` 或 `null`；按首次发言顺序组成评测用 `speaker.gender` 序列。不得根据 pitch、音色、语言或年龄刻板印象推断。 |
| `descriptive.speaker_descriptions[].description` | 该说话人的自然语言描述。 | 只写语速、相对音高、相对音量及其稳定性等可由 profile 支持的内容。 |
| `descriptive.speaker_descriptions[].descriptive_phrases` | 该说话人的简短短语数组。 | 例如“语速偏快”“音量变化明显”；不写性别、年龄、身份或情绪。 |
| `descriptive.speaker_descriptions[].evidence_tags` | 支撑该说话人描述的结构字段路径。 | 通常引用 `speaker.profiles`；路径必须真实存在且非 `null`。 |
| `descriptive.speaker_descriptions[].confidence` | 对该说话人描述的自评置信度。 | 范围 `[0,1]`。 |
| `descriptive.scene_hypotheses[].description` | 场景假设文本。 | 使用“可能”“更像”“疑似”等保守措辞。 |
| `descriptive.scene_hypotheses[].evidence_tags` | 支撑该假设的结构字段路径。 | 只能引用真实、非 `null` 的结构字段。 |
| `descriptive.scene_hypotheses[].confidence` | 假设置信度。 | 范围 `[0,1]`；证据弱时不得给高置信度。 |

## 5. 跨字段一致性检查

| 已知条件 | 应满足的关系 |
| --- | --- |
| `speaker_count = 0` | `speaker_present=false`、`multi_speaker=false`、`speaker_change_count=0`、`speaker_change=false`、`speaker_overlap=false`、`profiles=[]`（确认无语音时）。 |
| `speaker_count = 1` | `speaker_present=true`、`multi_speaker=false`、`speaker_change=false`、`speaker_overlap=false`、`speaker_change_count=0`。 |
| `speaker_count >= 2` | `speaker_present=true`、`multi_speaker=true`；是否切换和重叠仍按实际音频填写。 |
| `overlap_ratio = 0` | `speaker_overlap=false`；`overlap_ratio > 0` 时应为 `true`。 |
| `speech_music_events` 已知 | 只能是 `[speech, singing, music]` 的子集，且 `music_present` 等于是否包含 `music`。 |
| `profiles` 非 `null` | `speaker_id` 唯一、按首次发言排序，数量与已知 `speaker_count` 一致；每个 profile 的 `transcript` 只包含该说话人的片段。 |
| `profiles[].transcript` 非 `null` | 每项都有合法的 `start_sec < end_sec` 和 `text`，按时间排序；不同说话人的片段允许重叠。 |
| 派生字段 | 只按第 3.7 节公式计算，不用派生值反推 `speaker_change_count`、`filler` 或 `word_count`。 |

## 6. 标注顺序建议

1. 完整听一遍，先区分主要说话人与背景中的人声、音乐和噪声。
2. 先填写可验证的结构化字段，再检查人数、切换、重叠和事件之间的一致性。
3. 将每位说话人的语音切成带 `start_sec`、`end_sec`、`text` 的 `profiles[].transcript`；不要填写整段 `speaker.asr_transcript`。
4. 根据 profile transcript 和其它结构字段填写语言、统计字段及派生字段。
5. 最后填写描述性字段；不确定的内容宁可用 `null` 或写入 `uncertainties`，不要猜测。
