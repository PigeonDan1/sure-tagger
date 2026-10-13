# 650 条结构性评测集：数据构成报告

- 数据版本：`structural-benchmark-subset-v1.0`
- 统计截止：2026-09-19
- 统计对象：`compare/dataset/benchmark/` 当前落盘的 650 条样本
- 配套技术说明：[`../dataset/benchmark/DATA_COMPOSITION.md`](../dataset/benchmark/DATA_COMPOSITION.md)、[`../dataset/benchmark/README.md`](../dataset/benchmark/README.md)、[`../dataset/benchmark/SCORING_STANDARD.md`](../dataset/benchmark/SCORING_STANDARD.md)

本文描述**实际进入评测的 650 条子集**，不以 AMI / FLEURS / DESED 等源数据集的原始总规模代替本子集统计。历史 T4 controlled render 已从当前 benchmark 禁用，不计入这 650 条。

---

## 1. 评测集定位

本集合用于音频结构性理解评测，而不是开放域 caption 语料。设计目标是：在固定配额下，同时覆盖

1. **自然多说话人会议**（说话人数、重叠、轮换、英语 ASR、非语音比例、性别）；
2. **多语言 ASR / 语种识别**（10 语种 + 英文 clean anchor）；
3. **声音事件与外部噪声**（闭世界多标签：speech / music / 六类 broad noise）。

抽样全部使用分层内的 **midpoint quantiles**，不设随机种子；重复构建会得到同一组 `sample_id`。系统输入统一为 16 kHz PCM WAV；GT、路径、源标签不得进入模型请求。

---

## 2. 总体构成

| Track | 来源 | 条数 | 输入时长 | 单条时长 | 音频格式 | 主要可评内容 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| T1 自然会议 | AMI Mix-Headset | 350 | 145.05 min | 15.90–94.52 s | 16 kHz PCM，348 条单声道 / 2 条立体声 | 说话人、重叠、轮换、英语 ASR、非语音比例、性别 |
| T2 语言 / ASR | FLEURS 100 + LibriSpeech 50 | 150 | 26.79 min | 2.18–23.67 s | 16 kHz mono PCM | 语种、ASR、词法 |
| T3 事件 / 噪声 | DESED 75 + FSD50K 75 | 150 | 21.21 min | 0.52–28.92 s | 16 kHz mono PCM | speech/music 事件、音乐存在、外部噪声类型 |
| **合计** |  | **650** | **193.05 min（约 3.22 h）** |  | 约 372 MB WAV |  |

配额固定为 350 / 150 / 150，构建器按 `exactly 650` 验收。音频文件数与清单条数一致：T1 350、T2 150、T3 150。

### 2.1 三个 track 的分工

- **T1** 提供自然会议的说话人结构与英语转写，是 speaker / overlap / change / gender / nonspeech 的主评测集。
- **T2** 提供 10 语种朗读语音和 LibriSpeech test-clean 英文锚点，是 LID / ASR 的主评测集；不含会议重叠或场景噪声 GT。
- **T3** 提供事件与 broad noise 的闭世界多标签，是 `speech_music_events`、`music_present`、`external_noise_type` 的主评测集。

### 2.2 模型可见信息

统一 `benchmark.manifest.jsonl` 是审计清单，不是模型请求体。正式 runner 只应暴露解码音频和 opaque `sample_id`。下列内容属于 GT / provenance，不能送给被测系统：

- AMI 转写、说话人 turn、词级时间戳、性别；
- FLEURS / LibriSpeech reference text 与语种；
- DESED 强标注事件、FSD50K 弱标签、T3 crosswalk 与 closed-world GT；
- 源路径、会议 ID、locale、license 等可由路径推断的封装字段。

---

## 3. T1：AMI 自然会议（350 条）

### 3.1 来源与切片

- 源语料：AMI Meeting Corpus，输入音频为 **Mix-Headset** 切片。
- 覆盖本地 utterance 树中的 **全部 170 个 meeting**。
- 切片原则：在官方 utterance 边界上切出约 20 s 量级的多 utterance 片段（实际 15.90–94.52 s，中位数 22.60 s，均值 24.87 s）。
- 350/350 条均有 reference transcript、speaker ID 和词级时间戳。
- 统一 manifest 已清空 `text.transcript` 与 `native_metadata`；完整 GT 只保留在 `t1-ami/manifest.jsonl`。

### 3.2 Meeting 场景分布

AMI 会议 ID 前缀对应场景类型。本子集按 meeting 全覆盖，因此样本数随各场景 meeting 数自然倾斜，**不是**按场景均匀抽样。

| 前缀 | 场景 | Meeting 数 | 样本数 | 时长合计 | 条均时长 | 条均说话人数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ES | Edinburgh scenario（遥控器设计） | 60 | 123 | 50.24 min | 24.51 s | 2.67 |
| TS | TNO scenario | 39 | 80 | 29.70 min | 22.27 s | 2.81 |
| IS | Idiap scenario | 38 | 79 | 31.11 min | 23.63 s | 2.86 |
| EN | Edinburgh natural | 16 | 33 | 15.18 min | 27.60 s | 2.61 |
| IN | Idiap natural | 10 | 20 | 13.02 min | 39.06 s | 3.15 |
| IB | Idiap briefing | 7 | 15 | 5.80 min | 23.19 s | 3.13 |
| **合计** |  | **170** | **350** | **145.05 min** | **24.87 s** | **2.79** |

自然语言会议（EN/IN）片段更长、说话人略多；scenario 会议（ES/IS/TS）构成样本主体。语言层面本 track 全部按 AMI 英语会议处理，没有片段级 code-switching GT。

### 3.3 抽样规则

方法名：`two_midpoint_quantiles_per_meeting_plus_third_for_ten_midpoint_meetings`。

- 每个 meeting 内按时间排序后取 midpoint quantiles，不使用随机状态。
- 160 个 meeting 各取 2 条（320 条），10 个 meeting 各加 1 条（30 条），凑满 350。
- 额外 1 条的 10 个 meeting 同样由 meeting 列表的 midpoint 确定，当前为：

`EN2003a`，`ES2004b`，`ES2008c`，`ES2012d`，`IB4001`，`IS1000a`，`IS1004c`，`IS1009a`，`TS3006b`，`TS3010d`。

每条 metadata 保存源 manifest 路径、原行号、manifest/record/audio SHA256，可回溯到完整 AMI 树。

### 3.4 说话人与时间结构

按 clip 内出现的唯一 speaker ID 和原始 turn 区间统计（构成描述，不是另一套评分口径）：

| Clip 内说话人数 | 条数 | 比例 |
| ---: | ---: | ---: |
| 1 | 45 | 12.9% |
| 2 | 84 | 24.0% |
| 3 | 122 | 34.9% |
| 4 | 99 | 28.3% |

- 说话人在场：350/350。
- 多说话人（≥2）：305/350（87.1%）。
- 观察到 speaker change：305/350。
- 按原始 turn 区间是否相交，288/350（82.3%）存在 overlap；62 条未观察到 overlap。
- overlap ratio 均值 0.139、中位数 0.105、最大 0.686。
- 非语音 / 静音比例均值 0.195、中位数 0.111，范围 0.001–0.921。
- 每条含 1–26 个原始 utterance（均值 6.19）；词数 2–412（均值 65.4，合计约 2.3 万词）。

这使 T1 成为多说话人、有重叠、有轮换的自然会议压力集，而不是单人朗读集。

### 3.5 性别 GT

`t1-ami/speaker_gender_gt/` 为 350 条补充了说话人性别。口径：

- 每个样本内，源 A/B/C/D/E 说话人按**该片段内首次开口时间**排序，映射为 `speaker_1`, `speaker_2`, …。
- 优先使用 AMI 官方 `participants.xml` 的 `sex`（750/975 个 sample-speaker 对）。
- 其余 225 对使用 Qwen3.5-Omni-Plus 在说话人语音片段上的回退标注。
- 官方元数据覆盖处，有 59 个 Qwen 标签被官方值覆盖。

| 统计 | 数量 |
| --- | ---: |
| 样本数 | 350 |
| 活跃 sample-speaker 对 | 975 |
| 男性 / 女性（说话人实例） | 666 / 309 |
| 样本内全男 / 全女 / 男女混合 | 164 / 63 / 123 |

性别 GT 可用于 `speaker.gender` 评测，但不能解释为 AMI 全库的性别先验；回退标签存在模型误差。

### 3.6 音频格式注意点

T1 全部为 16 kHz、16-bit PCM。其中 348 条单声道，2 条为立体声：

- `ES2010d_utterance_00017.wav`（26.35 s）
- `ES2010d_utterance_00029.wav`（20.54 s）

评测或特征提取若默认单声道，需明确这两条的通道处理，不要静默丢声道。

---

## 4. T2：多语言 ASR（150 条）

### 4.1 配额

| 子集 | 条数 | Split | 许可（源 metadata） | GT 强度 | 时长 |
| --- | ---: | --- | --- | --- | ---: |
| FLEURS | 100 | test | CC-BY-4.0 | D0 | 19.36 min（3.96–20.88 s） |
| LibriSpeech | 50 | test-clean | CC-BY-4.0 | D0 | 7.44 min（2.18–23.67 s） |
| **合计** | **150** |  |  |  | **26.79 min** |

FLEURS 从十个 locale 的上游面板中各取 10 条；LibriSpeech 从 100 条 clean panel 中取 50 条。分层内同样使用 midpoint quantiles。150/150 均有 reference transcript 与语言标签。输入均被 ffmpeg 转为 16 kHz mono `pcm_s16le`（bitexact）。

### 4.2 语种构成

| 来源 | 语言 | Locale / 配置 | 条数 |
| --- | --- | --- | ---: |
| FLEURS | 阿拉伯语 | `ar_eg` | 10 |
| FLEURS | 普通话（简体） | `cmn_hans_cn` | 10 |
| FLEURS | 英语（美国） | `en_us` | 10 |
| FLEURS | 西班牙语（拉丁美洲） | `es_419` | 10 |
| FLEURS | 法语 | `fr_fr` | 10 |
| FLEURS | 印地语 | `hi_in` | 10 |
| FLEURS | 日语 | `ja_jp` | 10 |
| FLEURS | 韩语 | `ko_kr` | 10 |
| FLEURS | 俄语 | `ru_ru` | 10 |
| FLEURS | 斯瓦希里语 | `sw_ke` | 10 |
| LibriSpeech | 英语 clean anchor | `en` | 50 |

英语合计 60 条（FLEURS 10 + LibriSpeech 50），其余 9 语种各 10 条。本子集没有片段级 code-switching 标注。

FLEURS 100 条的标注性别为男性 54、女性 46，各语种并不均衡（例如 `ar_eg` 10 条全为女性，`en_us` 9 女 1 男）。该性别字段来自 FLEURS 源标注，本 benchmark 未把 T2 性别作为正式评分轴。

### 4.3 LibriSpeech 使用边界

50 条 LibriSpeech 只来自 **2 名说话人**：`1089` 32 条、`1188` 18 条。这是 midpoint 抽样落在 100 条 panel 上的结果，不是说话人多样性采样。

因此 T2 的 LibriSpeech 部分只适合作为 **English clean ASR / 词法锚点**，不能用来评价说话人计数、性别或说话人变化。

T2 也不提供会议 overlap、环境事件、SNR 或房间声学 GT。

---

## 5. T3：事件与外部噪声（150 条）

### 5.1 来源

| 来源 | 条数 | 标注形态 | GT 强度 | Split | 时长 |
| --- | ---: | --- | --- | --- | ---: |
| DESED `synthetic21_train` | 75 | 强标注 soundscape（onset/offset） | D0 | train synthetic | 12.50 min，全部 10.00 s |
| FSD50K | 75 | clip-level 弱/verified 多标签 | W | dev 38 + eval 37 | 8.71 min（0.52–28.92 s） |
| **合计** | **150** |  |  |  | **21.21 min** |

抽样方法：`midpoint_quantiles_within_source_and_fsd50k_split`。输入同样转为 16 kHz mono PCM。

DESED 75 条共出现 10 种原生事件类型；单条含 1–8 个强标注事件（均值 3.09），clip 级独特标签 1–4 个。FSD50K 75 条共出现 87 种原生标签；单条 1–9 个标签（均值 3.53）。

### 5.2 闭世界口径

T3 只在冻结的 3 个 `speech_music_events` 类和 6 个 `external_noise_type` 类上采用

`native-label closed-world`（crosswalk 版本 `t3-native-label-closed-world-v1.0`）：

- 原生正标签经 crosswalk 映射为目标类正例；
- 没有正映射的目标类记为负例；
- **负例是评测假设，不是人工确认负例**，可能含漏标假负例。

`music_present` 由 GT event 物化，作为独立 boolean 计分。逐样本 class mask 见 `t3-event-noise/closed_world_gt.jsonl`，映射表见 `t3-event-noise/crosswalk.json`。

### 5.3 九类正负例

| Schema 字段 | 目标类 | 正例 | 闭世界负例 | 主要正例来源 |
| --- | --- | ---: | ---: | --- |
| `speech_music_events` | `speech` | 69 | 81 | DESED Speech；FSD50K Speech / Conversation 等 |
| `speech_music_events` | `music` | 38 | 112 | FSD50K Music / Musical instrument |
| `speech_music_events` | `singing` | **0** | 150 | 本子集无原生 singing 正标签 |
| `music_present` | true / false | 38 / 112 | — | 与 `speech_music_events.music` 物化一致 |
| `external_noise_type` | `mechanical` | 68 | 82 | Dishes、alarm、vehicle、typing、door 等 |
| `external_noise_type` | `music` | 38 | 112 | 与 music 事件共用 FSD50K 音乐标签 |
| `external_noise_type` | `animal` | 20 | 130 | Cat / Dog / Bird / Animal 等 |
| `external_noise_type` | `formless` | 20 | 130 | Frying、explosion、crackle、whoosh 等 |
| `external_noise_type` | `nature` | 13 | 137 | water / rain / thunder / ocean 等 |
| `external_noise_type` | `channel_environment` | **0** | 150 | 本子集无原生 channel/environment 正标签 |

多标签可共现，正例数不能相加当作样本数。`speech_music_events` 的集合分布为：仅 speech 69、仅 music 38、空集 43。噪声类型也可组合，例如 `mechanical+formless`、`animal+mechanical` 等。

**写作时必须写明：** `singing` 与 `channel_environment` 只有负例，可观察误报，但不能谈召回，也不能无说明地做 9 类 macro-F1。不能把“150 个负例”写成已经覆盖了这两类正声音。

### 5.4 许可

- DESED 75 条：dataset-specific DCASE / DESED 条款。
- FSD50K 75 条：CC BY 3.0 共 41 条，CC BY-NC 3.0 共 13 条，Sampling+ 1.0 共 13 条，CC0 共 8 条。

这些是源 metadata 记录，不构成新的统一再分发许可。

---

## 6. 目录结构

```
compare/dataset/benchmark/
├── README.md
├── DATA_COMPOSITION.md          # 字段级覆盖与评分 mask 技术说明
├── SCORING_STANDARD.md
├── benchmark.manifest.jsonl     # 650 条审计清单
├── benchmark.metadata.jsonl     # 与清单按 sample_id 对齐的 GT/provenance
├── benchmark.summary.json       # 配额、分层计数、SHA256
├── build_benchmark_subset.py    # 确定性构建与验收
├── t1-ami/
│   ├── audio/                   # 350 条 WAV
│   ├── manifest.jsonl           # 含 transcript / turn GT，不可送模型
│   ├── metadata.jsonl
│   └── speaker_gender_gt/       # 975 个说话人性别标签
├── t2-language-asr/
│   ├── audio/{fleurs,librispeech}/
│   ├── manifest.jsonl
│   └── metadata.jsonl
└── t3-event-noise/
    ├── audio/{desed,fsd50k}/
    ├── manifest.jsonl
    ├── metadata.jsonl
    ├── crosswalk.json
    └── closed_world_gt.jsonl
```

各 track 的 `manifest.jsonl` 中音频路径相对该 track 目录解析；统一 manifest 中路径相对 `benchmark/` 解析。

---

## 7. 结构标签覆盖（写作可用摘要）

当前冻结 field universe 里，**本 650 条实际提供有效 GT 的能力**可概括为：

| 能力组 | 有效范围 | 条数级分母 | 备注 |
| --- | --- | ---: | --- |
| 时长 / 采样率 / 声道 | 全集合 WAV header | 650 | 基本声学；T1 有 2 条立体声 |
| 非语音比例 | T1 官方 silence/speech 补集 | 350 | 不用 T2/T3 |
| 说话人数、在场、多人、轮换、重叠 | T1 turn | 350 | 含独立 boolean 与连续 ratio |
| 说话人性别 | T1 gender GT | 350 样本 / 975 人 | 官方元数据优先 |
| ASR 转写、语种、词法 | T1 + T2 | 500 | T1 英语会议；T2 10 语种 |
| 语音/音乐事件、音乐存在、外部噪声 | T3 闭世界 | 150 | `singing`、`channel_environment` 无正例 |
| SNR / DNSMOS / RT60 / C50 / far-field / 细粒度 noise composition | 无冻结 GT | 0 | 当前 650 条不可评 |
| 开放描述文本 | 无人工 caption GT | 0 | 描述轴若评，需另列参考侧 |

更细的 35 字段 mask 与计分公式见 `DATA_COMPOSITION.md` 与 `SCORING_STANDARD.md`。默认能力雷达与 29 轴评估方法见 `docs/benchmark-v1-29-metrics.md`。

---

## 8. 明确缺口与表述边界

写结果或数据集章节时，建议直接采用下列边界，避免把源数据集能力写成子集能力：

1. **总量就是 650，不是 AMI/FLEURS/DESED 全库。** 分层配额是设计选择，场景类（如 ES）样本更多只因为 meeting 更多。
2. **T4 不在本评测集中。** 受控混音、SNR、RT60 等历史 T4 标签不属于这 650 条。
3. **`singing` 与 `channel_environment` 无正例。** 只能谈误报，不能谈召回，也不能无说明地做 9 类 macro-F1。
4. **T3 负例不是人工确认负例。** 必须保留 closed-world 假设与可能假负例的说明。
5. **没有片段级 code-switching GT。** T1 为英语会议，T2 为单语朗读。
6. **LibriSpeech 50 条只有 2 名说话人**，不能当作说话人多样性集合。
7. **T3 FSD50K 是弱标签（W），DESED 是合成强标注（D0）**，两者不可当成同一标注精度。
8. **无 DNSMOS、far-field、房间声学、DASS 细粒度噪声成分、开放 caption 人工 GT。**
9. **许可异质。** 本地 bundle 完整不等于可以统一公开再分发。
10. **2 条 T1 立体声** 需要在预处理描述里点明。

---

## 9. 可直接写入论文的一句话概括

> 我们构建了一个 650 条的结构性音频评测子集：350 条覆盖 AMI 全部 170 场会议的 Mix-Headset 自然会议片段（约 2.42 小时），用于说话人结构、重叠、英语 ASR 与性别；150 条来自 FLEURS 10 语种（各 10 条）与 LibriSpeech test-clean（50 条），用于语种识别与 ASR；150 条来自 DESED 与 FSD50K（各 75 条），在冻结的 speech/music/噪声类上做原生标签闭世界多标签评测。全部样本按分层 midpoint quantiles 确定性抽样，输入音频约 3.22 小时。

如需更短：

> 650 clips / 3.22 h：AMI 会议 350，多语言 ASR 150，事件与噪声 150；确定性分层抽样，T4 受控混音不纳入。

---

## 10. 复现与验收

在共享目录根路径执行：

```bash
python3 compare/dataset/benchmark/build_benchmark_subset.py --validate-only
```

验收检查总数、分项配额、音频路径与 SHA256、WAV 可读性、AMI 170 个 meeting 回溯、FLEURS 语言配额和 FSD50K split。机器计数见 `compare/dataset/benchmark/benchmark.summary.json`。
