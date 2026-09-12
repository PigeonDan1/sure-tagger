# Development

本文件记录项目开发期的全局约束。它不是运行期 agent 指导文件；整体开发完成后如需面向 agent 的说明，另建 `AGENTS.md`。

根目录文档只放项目级接口和边界。具体工具的调度顺序、依赖、模型配置、缓存、失败联动和内部实现说明，放到对应 `tagger/tools/...` 目录下。

## 目标

本项目构建 sample-level 的 ASR 数据集打标工具。公开输出只包含本次打标得到的 tag 值，不包含 `sample_id`、原始记录、工具名、证据、置信度、来源、状态、warning、审计信息或推理过程。

当前 tag 分组：

| 分组 | 字段 |
| --- | --- |
| `basic_acoustic` | `duration_sec`、`sample_rate_hz`、`channels`、`silence_segments`、`nonspeech_ratio` |
| `audio_quality` | `snr_db`、`dnsmos_sig`、`dnsmos_bak`、`dnsmos_ovrl`、`dnsmos_p808` |
| `room_acoustic` | `far_field`、`rt60_sec`、`c50_db` |
| `sound_field_scene` | `speech_music_events`、`music_present`、`external_noise_type`、`noise_composition` |
| `speaker` | `speaker_count`、`speaker_present`、`multi_speaker`、`speaker_change_count`、`speaker_change`、`overlap_ratio`、`speaker_overlap`、`profiles`、`asr_transcript` |
| `language_content` | `language`、`word_count`、`punctuation`、`repetition`、`filler` |
| `deterministic_derived` | `speaker_change_rate_per_min`、`filler_ratio` |

其中 `language_content.language` 由 FireRed LID 音频模型产出（`firered_lid`
stage），其余 `language_content` 字段由确定性文本工具产出。FireRed LID 的
结果同时驱动 speaker 路由：中文（`zh` 或 `zh-*`）和英文（`en` 或 `en-*`）使用
MOSS joint ASR/diarization；其他、混合或未知语言分别调用 Qwen3-ASR-0.6B 和官方
DiariZen。FireRed 只在中英文单说话人共识成立时产出整段转写；Qwen3 ASR 为其他
语言产出整段转写。DiariZen 使用发布的 WavLM large-s80-v2 segmenter、
pyannote WeSpeaker、VBx 和 PLDA 产出 speaker timeline。两路分别校验、分别失败：
任一路失败不会覆盖另一路的有效结果。每条样本只选择一个语言分支；LID 失败按
unknown 进入其他语言分支，且不会跨分支补位。
`speaker.asr_transcript` 是该语言路由选中的 ASR 文本按时间顺序拼接得到的确定性
上游字段；不包含时间戳或 speaker ID，也不读取输入 transcript。
当前 claim policy 为 `speaker_v2.claim_policy.20260910.1`，C/M/O/X rule 的
`selection` 固定为 `language_routed`。resolver 必须收到样本的 `language_route`
对象；缺失或非法 route 直接报错，不能从可用 evidence 猜测或选择另一分支。实际
claim route 始终记录 `language_selected_source`；选中 source 可用时为
`selection=language_route`，不可用时为 `selection=none`、相关 claim 为 `null` 并
记录原因，不会尝试另一分支。policy 中的 fallback 列表为空。独立 speaker CLI
分别提供 `--firered-asr-enable/disable`、`--qwen3-asr-enable/disable` 与
`--diarizen-enable/disable`；旧 `--openasr-enable/disable` 只作为兼容组别名。
批量 ASR/diarization 刷新可使用 `refresh-shadow` profile；该 profile 保留
MOSS、FireRedASR2-AED、Qwen3-ASR-0.6B、官方 DiariZen 和 LID，关闭不影响本次结构标签的 ECAPA、
Brouhaha 及其它辅助模型，以避免多模型同时驻留造成显存竞争。

官方 DiariZen 部署在 `models/diarizen-official/`，runtime 为
`.runtime/diarizen_official_py310/bin/python`。segmenter checkpoint SHA-256 为
`689a10f0cbde6345b4bf9b1670cf3196488634b34e5b07bea0fe40d508e1196b`，
WeSpeaker checkpoint SHA-256 为
`366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56`。
模型、代码、config、PLDA 与 x-vector transform 的固定版本和校验值见
`models/diarizen-official/DEPLOYMENT.json`。短于 16 秒的输入只在 DiariZen 内部
静音补齐，timeline 在公开前裁回原时长。上游全部 cluster 都为 inactive sentinel
时触发的 `negative dimensions` 边界错误被规范化为空 timeline，其他异常仍按失败处理。
开放描述性 `topic` 由 sure-tagger 下游语言模型推断，不属于本项目公开 schema。
`sound_field_scene.external_noise_type` 由 DASS 音频模型产出（`dass`
stage），值为 docs/DASS.md 类别键数组（见下方约束 bullet）。DASS 是
默认链路的背景噪音主模型，默认请求项目内 CUDA runtime；`--dass-use-cpu`
仅用于显式 CPU 调试。排除策略是全有/全无的，传 `--no-exclusion`
时全部关闭。PANNs stage 与 `sound_field_scene.sound` 字段已于
2026-08-25 废弃删除（`noise_composition` 取代其功能）；工具模块
`panns_background_detector` 保留可导入，仅作后续交叉验证 evidence 用，
不注册、不进入公开输出。
Brouhaha 的 C50 预测只作为内部 evidence（`internal.brouhaha_c50_db`）用于
与 `room_acoustic.c50_db` 交叉验证，不进入公开输出。

## 全局原则

- 先结构化 closed raw-only 输入，再运行登记的 Python 标签程序。
- `corpus` 和 `sample` 在 pipeline 中视为不可变原始输入。
- 输入对象只能包含数据集原生数据，不得混入工具输出、推理、归一化结果、解析文档正文、运行时 provenance 或默认值。
- 最终 tag 只能来自 registry 登记的 Python 标签程序输出；登记工具可以在显式启用时调用模型或外部 API，但不得由人工推断、未登记脚本、corpus-level 默认值、native metadata 或绕过 pipeline 的 LLM 输出直接生成。
- 工具内部可以使用模型、规则、阈值、缓存、证据和审计信息，但这些内容不得进入公开 tags-only 输出。
- 程序未实现、不可用、失败、缺失字段或输出非法时，对应最终 tag 为 `null`。
- 新增或变更 tag 时，同步更新 schema、registry、pipeline、resolver/auditor 和测试样例；只有全局接口变化才需要更新本文件。

## 输入 Schema

输入是封闭 raw-only schema：

```json
{
  "corpus": {
    "dataset_name": "",
    "source_urls": {
      "article": [],
      "github": [],
      "huggingface": [],
      "dataset_card": []
    },
    "native_metadata": {}
  },
  "sample": {
    "sample_id": "",
    "audio": {
      "path": ""
    },
    "text": {
      "transcript": ""
    },
    "native_metadata": {}
  }
}
```

约束：

- 顶层只允许 `corpus`、`sample`。
- `corpus` 只允许 `dataset_name`、`source_urls`、`native_metadata`。
- `source_urls` 只允许 `article`、`github`、`huggingface`、`dataset_card` 四个数组字段。
- `sample` 只允许 `sample_id`、`audio`、`text`、`native_metadata`。
- `sample.audio` 只允许 `path`。
- `sample.text` 只允许 `transcript`。
- `native_metadata` 只能保存数据集原始记录中已有键值。
- 数据集原生但无法放入固定字段的内容，只能放入对应层级的 `native_metadata`。
- 不允许写入 `uri`、`format`、`dataset_field`、`raw_documents`、`raw_metadata`、`metadata`、`provenance`、`manifest_path`、`row_index`、`source_split`、`raw_record`、`raw_text_fields` 或其他扩展字段。
- 不允许把网页正文、README 正文、dataset card 正文、字段解释、规范化字段名、运行时路径、split、行号、缓存键、工具输出或派生值写入 raw input。
- `sample.sample_id`、`sample.audio.path`、`sample.text.transcript` 必须来自数据集原生字段或原始 manifest；运行时生成的定位信息只能进入内部 provenance。

## 输出 Schema

公开输出是 tags-only JSON：

```json
{
  "basic_acoustic": {
    "duration_sec": null,
    "sample_rate_hz": null,
    "channels": null,
    "silence_segments": null,
    "nonspeech_ratio": null
  },
  "audio_quality": {
    "snr_db": null,
    "dnsmos_sig": null,
    "dnsmos_bak": null,
    "dnsmos_ovrl": null,
    "dnsmos_p808": null
  },
  "room_acoustic": {
    "far_field": null,
    "rt60_sec": null,
    "c50_db": null
  },
  "sound_field_scene": {
    "speech_music_events": null,
    "music_present": null,
    "external_noise_type": null,
    "noise_composition": null
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
    "asr_transcript": null
  },
  "language_content": {
    "language": null,
    "word_count": null,
    "punctuation": null,
    "repetition": null,
    "filler": null
  },
  "deterministic_derived": {
    "speaker_change_rate_per_min": null,
    "filler_ratio": null
  }
}
```

约束：

- 最终输出不得包含 raw input、`sample_id`、工具标识、方法、状态、来源、证据、置信度、warning、rationale、prompt、模型名或审计结论。
- Rec-RIR 生成的 RIR 波形不得进入公开输出；pipeline 只公开由该波形计算出的 `room_acoustic.rt60_sec` 和 `room_acoustic.c50_db`，波形保存为内部 artifact。
- `sound_field_scene.speech_music_events` 是 FireRed AED 检出的 `speech`、`singing`、`music` 类别数组，其中 `singing`、`music` 只在事件占比（event ratio）达到 `--firered-aed-min-singing-ratio` / `--firered-aed-min-music-ratio`（均默认 0.10，2026-08-25 起：caption_pairs_3000 上语音帧被帧级误判为歌声/音乐产生的短段占比都低于 0.10，而真实歌声/音乐占比远高）时进入数组；`sound_field_scene.music_present` 是该数组是否包含 `music` 的布尔值；`sound_field_scene.external_noise_type` 是 DASS 检出的 docs/DASS.md 类别键数组：`music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment`（人类声音与未归类标签永不公开），由全量 527 类向量中未被排除（主语音、Silence、声学场景、混响、回声）且达到 `--dass-threshold`（默认 0.25）的标签归组而来，按各类别最高分降序排列。成功但没有检出时输出空数组，工具失败时输出 `null`；模型分数、比例和时间段不进入公开输出。
- `sound_field_scene.noise_composition` 是 DASS 全量 527 类 sigmoid 输出按 docs/DASS.md 类别归组后的背景声组成对象，固定含 `music`、`animal`、`mechanical`、`nature`、`formless`、`channel_environment` 六个键，每键为按分数降序的标签名数组（每类最多 `--dass-composition-top-k` 个，默认 3；入组阈值 `--dass-composition-threshold`，默认 0.25，与 `external_noise_type` 的类别阈值对齐（2026-08-25 对齐：此前 0.30 与 0.25 之间有空档，会出现「有类别、无组成」；对齐后保证有类别的行组成非空））。音乐类别以 FireRed AED 为准：`music_present` 为 `false` 时输出空数组，为 `true` 或 AED 未运行（`null`）时输出 DASS 音乐类标签；人类声音与未归类标签只进内部 evidence（`category_events`），不进入公开输出。类别分数、证据和 AED 门控状态不进入公开输出。
- `speaker.profiles` 与 `speaker_count` 使用同一条按语言选择的 decision timeline，并严格按各说话人第一次发言的时间排列为 `speaker_1`、`speaker_2`……。每项为 `speaker_id`、`gender`（`male`/`female`/`null`）、`speech_rate`（`band`：`slow`/`normal`/`fast`/`variable`/`null`、`value`、`unit`：`zh_char_per_sec`/`word_per_min`/`null`）、`pitch`（`low`/`mid`/`high`/`variable`/`null`）、`speaker_volume`（`low`/`normal`/`loud`/`variable`/`null`）。性别模型按 FireRed LID 路由：`zh`/`en` 使用 Alice，`sw` 及已配置非洲语言使用 AfriSpeech，其余、mixed、unknown 或 LID failure 使用 Griko；单说话人直接输入整段音频，多说话人只拼接该说话人的无 overlap 活动片段，净时长小于 1 秒时 `gender=null`。性别模型失败只影响对应 `gender`。`unit` 为 `null` 时 `value` 必须为 `null`，画像数组为空（`[]`）表示确认没有语音，无法得到可靠时间轴或画像计算失败时为 `null`。不推断年龄、情绪或口音，也不从 pitch 派生性别。模型分数、原始 F0/RMS、区间、文本证据和 evidence id 只进入内部 artifact。
- `speaker.speaker_present` 是 `speaker_count` 的确定性派生值：`speaker_count > 0` 时为 `true`，`speaker_count == 0` 时为 `false`，`speaker_count == null` 时为 `null`。它不新增模型或独立 claim。
- `deterministic_derived.speaker_change_rate_per_min` 为 `round(speaker.speaker_change_count * 60 / basic_acoustic.duration_sec, 6)`；`deterministic_derived.filler_ratio` 为 `round(language_content.filler / language_content.word_count, 6)`。任一依赖缺失、非法或分母不为正时输出 `null`。
- `speaker.asr_transcript` 由语言和人数共识路由选中的 ASR 产生：中英文在 MOSS 与 DiariZen 的有效人数都 `<=1` 时取 FireRedASR2-AED，否则从 MOSS 全音频时间线中有效 segment 的文本按 `start_sec` 排序后以空格拼接；其他、混合或未知语言取 Qwen3-ASR-0.6B 的整段文本。该字段不带时间戳和 speaker ID，不得以 `sample.text.transcript` 补值；选中模型的文本缺失或非法时为 `null`，不会跨候选补位。官方 DiariZen timeline 独立决定其他语言的 C/M/O/X，DiariZen 失败不覆盖有效 ASR 文本。
- 字段缺失、无法可靠判断或对应程序输出非法时，字段值必须为 `null`。
- `audio_quality` 与 `room_acoustic` 是独立分组，字段不得跨组复制。Brouhaha C50（内部 `internal.brouhaha_c50_db`）只作为内部 evidence，不进入公开输出；公开的 `room_acoustic.c50_db` 只能来自 Rec-RIR 派生的 C50 估计。

## 工具开发

- 工具代码放在对应 `tagger/tools/<tag_group>/` 目录。
- 每个公开 tag 必须通过该分组的 registry 暴露给 pipeline。
- 工具输出先进入内部结果对象，再由 pipeline/auditor 归并成 tags-only 输出。
- 工具级运行顺序和失败联动不要写在根目录文档；需要说明时，在对应工具目录下新增简短说明文件。

## 下载相关
- 优先使用国内源
- 下载失败后终止，并提醒开发者手动下载上传
