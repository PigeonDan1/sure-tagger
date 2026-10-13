# sure-tagger 与 captioner 比较方案

## 1. 目标

在同一批 `caption_pairs_3000` 样本上回答两个问题：

1. 两个系统分别能覆盖哪些标签能力，哪些是交集，哪些只属于一方。
2. 对语义交集，以 captioner 的结构化输出为参考标准，度量 sure-tagger 的一致性。

本方案中的“一致性”不是人工真值准确率。captioner 来自富描述文本，sure-tagger
直接分析音频，两者的证据源和标签粒度不同；因此主报告必须称为“相对 captioner
的一致性”，不能把所有分歧直接解释为 sure-tagger 错误。

## 2. 当前输入与现状

比较对象：

- sure-tagger：`/hpc_stor03/sjtu_home/weihan.chen/share/tagger`
- captioner：`/hpc_stor03/sjtu_home/weihan.chen/share/tag_qwen-omni-captioner`
- 评测数据：`/hpc_stor03/sjtu_home/weihan.chen/share/caption_pairs_3000`
- 比较工程：`/hpc_stor03/sjtu_home/weihan.chen/share/compare`

`/hpc_stor03/.../share` 与 `/mnt/cloudstorfs/.../share` 当前解析到同一目录。

截至 2026-08-26 的只读盘点结果：

- `caption_pairs_3000` 有 3000 个唯一 `.mp3` 和 3000 个同名 `.txt`。
- captioner 的 `tag_extracted_3000` 目录有 2979 个 JSON 文件，其中
  `_summary.json` 不是样本结果，所以实际匹配输入的结果为 2978 条。
- 当前有 22 个输入样本缺少 captioner 结果。
- `tag_qwen-omni-captioner/docs/coverage-full-report.md` 的分母是固定抽样的
  100 条，不是 3000 条；它可以说明 schema 字段行为，但不能直接作为本次
  3000 条实测覆盖率。
- sure-tagger 有两层公开接口，不能只看 `run_tagger.py` 的 tags-only JSONL：
  - 主 tags-only 结构有 29 个固定字段，其中 28 个已接入 pipeline，
    `room_acoustic.far_field` 是保留但尚未实现的字段；
  - speaker v2 的 `evaluation_output.speaker` / `public_adapter.speaker` 已于
    2026-08-26 接入 nullable `profiles`，公开逐说话人的 `speech_rate`、`pitch`
    和 `speaker_volume`。
- 当前 `run_tagger.py` 的聚合导出仍只从 speaker v2 拷贝六个旧 speaker 字段，
  没有把 `profiles` 写进 tags-only JSONL。这个导出差异不代表 sure-tagger 没有
  画像能力；正式比较必须另外读取 speaker v2 公共结果/`evaluation_output.json.gz`，
  或先修复聚合导出后再运行 3000 条。
- captioner 当前 schema 有 48 个非索引语义叶子；这个数字包含说话人画像和
  事件属性。两边粒度和公开接口不同，因此不能用 `29/48` 直接表示覆盖优劣。

在 22 条 captioner 缺失补齐前可以产出预览报告，但正式 headline 报告应等待
captioner 和 sure-tagger 都完成 3000 条。任何预览报告都必须显式显示缺失数，
不能只在两边成功结果的交集上静默缩小分母。

开源数据集、字段级 GT 覆盖矩阵及自然/合成 track 建议见
[`open-audio-dataset-survey.md`](open-audio-dataset-survey.md)；该调研用于后续独立
GT benchmark 设计，不替换本方案当前的 `caption_pairs_3000` 输入。

## 3. 比较的两种“覆盖面”

覆盖面必须拆成两层，不能混为一个百分比。

### 3.1 能力覆盖（静态 schema）

回答“系统设计上能输出什么”。以语义概念为单位审阅 schema、字段说明和模型
方法，不看某次运行是否为 `null`。

### 3.2 实测覆盖（3000 条数据）

回答“这次运行实际给出了多少可用结果”。对每个字段统计：

- `known`：字段存在且不为 `null`。
- `unknown`：字段缺失或值为 `null`。
- 空数组 `[]`：明确已知为空，算 `known`，不能当成缺失。
- `false` 和 `0`：有效值，不能当成缺失。
- 无结果、JSON 非法、schema 非法：样本级运行失败，另列失败率。

实体数组需要同时报告两种分母：

- 样本覆盖：多少样本至少有一个 profile/event 或明确给出空数组。
- 实体覆盖：profile/event 内各属性在所有实体中的 known 比例。

## 4. 静态能力覆盖结论

| 语义域 | captioner | sure-tagger | 关系 |
| --- | --- | --- | --- |
| 文件基础属性 | 当前 `attribute` 为空对象 | 时长、采样率、声道数 | sure-tagger 独有 |
| 语音与静音 | 粗粒度 `audio.tag`、主导内容 | speech/music 事件、静音段、静音比例 | 部分交集，粒度不同 |
| 音频质量 | 清晰度、保真度、噪声等级、底噪等级 | SNR、DNSMOS SIG/BAK/OVRL/P808 | 同一语义域，但一个是枚举、一个是连续数值 |
| 房间声学 | 室内外、房型、尺寸、表面、混响等级、回声等级 | RT60、C50、预留 far-field | 混响可做有条件比较，其余大多不同 |
| 音乐 | 是否存在及角色、时序、相对响度、旋律、节奏、音色 | music presence 和 speech/music/singing 事件 | 音乐存在部分可比；细节是 captioner 独有 |
| 环境声事件 | event type、family、位置、显著度、频带等 | DASS 六大类及具体 composition | 有粗粒度交集，需要固定 taxonomy 映射 |
| 说话人核心 | 人数、多人、切换、重叠 | 人数、多人、切换、重叠，外加 count/ratio | 四个字段可直接比较 |
| 说话人画像 | 性别、年龄、音高、音量、语速、距离、口音、情绪、音色、吐字、韵律、表达风格等 | speaker v2 已公开逐说话人的语速、相对音高、片段内相对音量；`speaker_descriptions[].gender` 提供结构化性别 | 语速/音高/音量/性别是直接交集；其余当前为 captioner 独有 |
| 互动和场景 | dialogue type、背景人声、audio scene | 无直接对应字段 | captioner 独有 |
| 语言内容 | 当前 acoustic schema 无对应字段 | 语言、主题、词数、标点、重复、filler | sure-tagger 独有 |

### 4.1 样本级直接交集（Tier A-S）

这四项含义和类型基本一致，作为正式“样本级核心一致性”范围：

| captioner 参考字段 | sure-tagger 字段 | 比较方式 |
| --- | --- | --- |
| `annotation[0].speaker.speaker_count` | `speaker.speaker_count` | 整数精确匹配 |
| `annotation[0].speaker.multi_speaker` | `speaker.multi_speaker` | 三态中的 known boolean 比较 |
| `annotation[0].speaker.speaker_change` | `speaker.speaker_change` | 三态中的 known boolean 比较 |
| `annotation[0].speaker.speaker_overlap` | `speaker.speaker_overlap` | 三态中的 known boolean 比较 |

### 4.2 说话人画像直接交集（Tier A-P）

speaker v2 当前公开结构为：

```json
{
  "profiles": [
    {
      "speaker_id": "speaker_1",
      "speech_rate": {
        "band": "normal",
        "value": 120.0,
        "unit": "word_per_min"
      },
      "pitch": "mid",
      "speaker_volume": "normal"
    }
  ]
}
```

与 captioner 可直接比较的画像字段有三项：

| captioner 参考字段 | sure-tagger 字段 | 公共值域交集 | 比较方式 |
| --- | --- | --- | --- |
| `speaker.profiles[].speed` | `speaker.profiles[].speech_rate.band` | `slow/normal/fast/variable` | 对齐 profile 后做枚举比较；captioner 的 `ultrafast/free` 另列为 sure 不可表达值 |
| `speaker.profiles[].pitch` | `speaker.profiles[].pitch` | `low/mid/high/variable` | 对齐 profile 后精确比较 |
| `speaker.profiles[].speaker_volume` | `speaker.profiles[].speaker_volume` | `low/normal/loud/variable` | 对齐 profile 后精确比较；sure 值是同片段内相对音量 |

sure-tagger 的 `speech_rate.value` 和 `unit` 是额外的数值能力；captioner 只有 speed
档位，没有数值参考，因此数值速率只做覆盖和分布统计，不计算一致率。

以下 captioner 画像字段目前仍是 captioner-only：`accent`、`age`、`emotion`、`distance`、
`stereo_position`、`amplification`、`voice_traits`、`articulation`、`prosody`、
`delivery_style`。性别现作为 `descriptive.speaker_descriptions[].gender` 的独立结构化标签
评估；sure-tagger 仍禁止用 pitch 反推年龄/性别，也不使用 caption 文本补画像。

#### Profile 对齐规则

两边的 `speaker_id` 都是 clip-local 索引，不是跨文件声纹身份。captioner 按首次出现
顺序建立 `speaker_1...`，sure-tagger 也按 decision timeline 的首次出现顺序建立
profile，因此只允许使用“首次出现序号”对齐，不能根据待评测的 pitch/speed/volume
寻找最相似配对，否则会人为抬高一致率。

画像结果分两层报告：

1. **高置信对齐**：两边 profile 数相同；双方 profile id 都是从 `speaker_1` 开始的
   连续序列；双方 known speaker count 相等且与 profile 数一致。作为画像主指标。
2. **宽松前缀对齐**：按共同存在的 `speaker_i` 配对，其余 profile 标记 missing/extra。
   只作覆盖和误差诊断，不并入主指标。

必须报告 `profile_alignment_sample_rate` 和 `profile_alignment_entity_rate`。无法对齐的
reference profile 不能被静默删除；在画像 effective agreement 中按未匹配处理。

### 4.3 可由字段投影得到的交集（Tier A2）

| captioner 投影 | sure-tagger 投影 | 限制 |
| --- | --- | --- |
| `"music" in annotation[0].audio.tag` | `sound_field_scene.music_present` | captioner 的 `audio.tag` 是正标签集合，缺少 `music` 不一定等于文本明确否定；正例召回是主指标，闭世界二分类只作次指标 |
| `"speech" in annotation[0].audio.tag` | `"speech" in sound_field_scene.speech_music_events` | 同样区分“未提及”和“明确不存在”；闭世界结果单列 |

Tier A2 不与 Tier A-S 或 Tier A-P 混成一个总分。

### 4.4 需要归一化或标定的交集（Tier B）

| captioner 参考字段 | sure-tagger 候选字段 | 建议分析 |
| --- | --- | --- |
| `room_acoustic.reverberation_level` | `room_acoustic.rt60_sec` | 枚举与连续值的有序相关；标定后可算加权 Kappa |
| `audio_quality.overall_fidelity` | `audio_quality.dnsmos_ovrl` / `dnsmos_p808` | 有序相关和分组分布，不做逐值相等 |
| `audio_quality.clarity_intelligibility` | `audio_quality.dnsmos_sig`、`room_acoustic.c50_db` | 分别做诊断相关，不能先合成一个“清晰度”值 |
| `audio_quality.noise_level` / `noise_floor` | `audio_quality.snr_db`、`dnsmos_bak` | 注意方向相反；做有序相关、分组箱线统计 |
| `sweeper_scene.external_noise_info.events` | `sound_field_scene.external_noise_type` / `noise_composition` | 先映射到双方都能表达的粗类，再算集合指标 |

Tier B 只作为扩展诊断，不能进入 Tier A-S 或 Tier A-P 的一致性分数。

### 4.5 不应强行比较的字段

- captioner 的 room type/size/surfaces/environment 与 sure-tagger 的 RT60/C50
  不是同一标签，不能由物理数值反推出房型或材质。
- captioner 的 echo level 与 sure-tagger 的 C50/RT60 不是同义字段。
- captioner 的 profile distance 是逐说话人的近远，sure-tagger 的
  `far_field` 是片段级预留字段，且当前恒为 `null`。
- captioner profile 中除 speed/pitch/speaker_volume 外的字段，以及 dialogue type、
  background speech、audio scene，在 sure-tagger 中没有直接公开对应项。
- sure-tagger 的基础文件属性、VAD、说话人切换次数、重叠比例、语言内容在
  captioner 当前 schema 中没有直接对应项。

## 5. 数据对齐与运行约束

### 5.1 唯一样本键

统一使用文件 stem 作为 `sample_id`，例如：

```text
0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2
```

canonical manifest 按 captioner 的 `docs/full_manifest_3000.jsonl` 的 `rank`
排序，并补充绝对音频路径、文本路径、文件大小和 SHA256。所有系统输出最终都
必须回到 `sample_id`，不能依赖目录遍历顺序。

### 5.2 sure-tagger 输出关联

sure-tagger 的 tags-only JSONL 不包含 `sample_id`，只保证与输入 manifest
行序一致。因此必须同时保存：

1. 本次运行使用的不可变 manifest；
2. tags-only JSONL；
3. 将两者按行严格 zip 后生成的 `sample_id -> tags` 归一化结果。

zip 前必须断言两文件行数完全相等；不能用短文件长度截断，也不能在中间过滤
失败行。

说话人画像不能从当前 tags-only JSONL 读取。画像数据源按优先级固定为：

1. 独立 speaker v2 运行的 `speaker_v2_shadow_results.jsonl` 中
   `speaker.profiles`；
2. 完整 tagger 运行为每条样本保存的 speaker v2
   `evaluation_output.json.gz` 中 `output.speaker.profiles`；
3. 如果后续修改了聚合导出，则使用通过回归验证、确实包含 `speaker.profiles` 的
   tags-only 新版本。

不能因为当前聚合器漏导出 `profiles`，就在能力矩阵中把画像判为 sure-tagger 不支持。
归一化结果还应保存画像来源字段 `speaker_profile_source`，便于审计。

### 5.3 不把 caption 当 transcript

`caption_pairs_3000/*.txt` 是 captioner 的富音频描述/参考证据，不是 ASR
transcript。为避免信息泄漏和无意义的语言标签，sure-tagger manifest 中：

```json
"text": {"transcript": ""}
```

sure-tagger 只读取对应 `.mp3`。caption 文本只用于 captioner 结果追溯和人工
分歧复核，不进入 sure-tagger 输入。

### 5.4 冻结运行版本

每次比较保存以下信息：

- canonical manifest 的 SHA256；
- captioner schema、prompt、结果目录快照及文件清单 SHA256；
- sure-tagger 代码/配置快照、完整命令、模型 profile 和阈值；
- 运行时间、机器/GPU、成功数、失败数和 warning 汇总；
- 比较脚本版本及 `field_mapping.yaml` 版本。

## 6. 预检门禁

比较脚本开始计算指标前必须生成 `preflight.json`，至少检查：

1. 3000 个唯一 `sample_id` 均有且只有一个 `.mp3` 和一个 `.txt`。
2. captioner 无重复样本、无 orphan、每个结果通过当前 schema。
3. sure-tagger manifest 恰好 3000 行且 sample_id 唯一。
4. sure-tagger tags 行数与 manifest 完全一致，每行是合法 JSON 对象。
5. speaker v2 画像结果能按 `sample_id` 唯一关联；`profiles` 为 `null`、`[]` 或合法
   profile 数组，profile id 唯一且满足 `speaker_[1-9][0-9]*`。
6. `speech_rate.band/value/unit`、`pitch`、`speaker_volume` 满足 speaker profile
   v0.1 合同，数值有限且 value/unit 空值联动合法。
7. 所有数值有限，boolean 不被当作 integer，数组/对象类型正确。
8. `null`、空数组、`false`、`0` 的语义按本方案处理。
9. manifest 中音频路径与 canonical manifest 的 sample_id 和 hash 一致。

正式报告门槛：两边样本级结果均为 3000/3000。未达到时仍可输出
`status=preliminary` 的报告，但必须列出全部缺失/非法 sample_id。

## 7. 字段归一化

所有映射写入版本化的 `compare/config/field_mapping.yaml`，比较代码不得散落
硬编码映射。

### 7.1 通用规则

- 字符串只做安全的格式归一化，例如 trim、lower、snake_case；不做语义猜测。
- captioner 的 `null` 保持 unknown，不映射为 false、0 或空数组。
- sure-tagger 的 `null` 保持 unknown，不用其他字段自动补齐。
- captioner reference known、sure-tagger unknown 时，计入“未覆盖”，不能从
  一致性分母中无声删除。
- 同一字段若存在多个候选 sure 指标，分别报告，不挑结果最好的一个作为主结果。
- profile 的 `speaker_id` 只用于片段内序号对齐，不作为画像一致性字段计分。
- captioner `speed=ultrafast/free` 超出 sure v0.1 值域：strict 指标计为不匹配，
  shared-vocabulary 指标另行排除并报告排除数。

### 7.2 事件 taxonomy

事件比较先投影到双方可表达的六个粗类：

```text
music, animal, mechanical, nature, formless, channel_environment
```

映射优先级：

1. captioner event `type` 的人工审核映射；
2. 明确且无歧义时使用 `family` fallback；
3. 无法确定时进入 `unmapped`，不硬塞到最相近类别。

`background_speech` 和 `human_non_speech` 等 sure-tagger 当前公开 DASS 输出明确
排除的类别，标记为 captioner-only，不作为 sure-tagger 的假阴性。报告必须同时
给出 captioner event 总数、可映射数和 unmapped 数，避免只在容易映射的子集上
产生虚高分。

### 7.3 连续值与有序枚举

Tier B 不直接比较字符串和浮点数。先把 captioner 等级固定成有序变量，再：

- 主报告给 Spearman 相关、每档样本数和 sure 数值分布；
- 若需要离散一致率，按 `sample_id` 的稳定 hash 划分 20% calibration / 80%
  evaluation，在 calibration 上学习单调阈值并冻结，在 evaluation 上计算加权
  Kappa 和混淆矩阵；
- 不允许在全部 3000 条上选出最优阈值后再用同一批数据报告准确率。

## 8. 指标定义

设总样本数为 `N=3000`。每个字段 `f` 统计：

- `N_caption_success`：有合法 captioner 样本结果；
- `N_sure_success`：有合法 sure-tagger 样本结果；
- `N_ref_known(f)`：captioner 对字段 `f` 给出参考值；
- `N_sure_known(f)`：sure-tagger 对对应字段给出值；
- `N_joint_known(f)`：两边均 known；
- `N_match(f)`：两边在归一化后匹配。

覆盖指标：

```text
captioner_field_coverage = N_ref_known / N
sure_field_coverage      = N_sure_known / N
joint_coverage           = N_joint_known / N
```

一致性同时给两个口径：

```text
conditional_agreement = N_match / N_joint_known
effective_agreement   = N_match / N_ref_known
```

`conditional_agreement` 表示“两边都有答案时多一致”；`effective_agreement` 将
sure-tagger 的缺失计为未匹配，防止只覆盖容易样本却得到很高分。两者必须并列。

### 8.1 Tier A-S 指标

speaker count：

- exact accuracy；
- MAE；
- `|error| <= 1` 比例；
- 完整计数混淆矩阵，长尾人数合并为 `4+` 仅用于展示，原始数据不合并。

boolean 字段：

- TP/FP/FN/TN；
- accuracy、precision、recall、F1；
- macro-F1、balanced accuracy；
- captioner 正/负样本数，避免类别不平衡掩盖问题。

样本级 headline 可以给一个 `speaker_core_effective_consistency`：四个 Tier A-S
字段的 `effective_agreement` 做等权宏平均。同时必须保留四个字段的单项结果。

### 8.2 Tier A-P 指标

先报告画像结构覆盖：

- captioner/sure 的 profiles known、empty、nonempty 样本数；
- 两边 profile entity 总数及每样本 profile count 分布；
- 高置信对齐的样本数、profile 实体数和对齐率；
- 宽松前缀对齐数、captioner-only profile 数、sure-only profile 数。

对 `speed band`、`pitch`、`speaker_volume` 分别报告：

- reference known、sure known、joint known、match；
- entity-level conditional/effective agreement；
- sample-macro conditional/effective agreement，防止多说话人样本权重过大；
- 完整混淆矩阵、macro-F1、balanced accuracy；
- `null`、unsupported reference value、profile unaligned 三类未匹配分别计数。

画像 headline 使用独立的 `speaker_profile_effective_consistency`：三个共同枚举字段的
sample-macro effective agreement 等权平均。它不能与 Tier A-S 四个样本级字段再平均
成一个数，因为两者分母分别是 sample 和 profile entity。

### 8.3 Tier A2 指标

- captioner 正例上的 sure-tagger recall；
- 两边正例率；
- 正例交集、captioner-only、sure-only、两边均无正例的计数；
- 闭世界二分类的混淆矩阵作为 secondary，并明确标注 closed-world。

### 8.4 Tier B 指标

- 有序字段：Spearman、分档统计、标定后 weighted Kappa；
- 多标签事件：micro/macro precision、recall、F1，sample-level Jaccard，exact
  set match，各粗类单独的 2x2 混淆矩阵；
- taxonomy 映射覆盖率和 unmapped 率必须与一致性一起展示。

## 9. 分歧复核

自动指标后进行一次分层人工复核，避免把参考侧错误或 taxonomy 问题都归因给
sure-tagger：

- 每个 Tier A-S 字段抽取最多 50 条分歧；稀有正例分歧优先全量查看。
- 三个 Tier A-P 画像字段分别抽取最多 50 个分歧 profile，并保留其所属完整样本
  的两边全部 profiles，检查是否由 speaker 序号错配导致。
- Tier A2 和事件类分别抽取 captioner-only、sure-only 各 50 条。
- 再随机抽取 100 条一致样本，检查“共同犯错”。
- 复核时同时听音频、看原始 caption 和两边原始 JSON。

每条分歧只标一个主原因：

```text
sure_error
captioner_reference_error
ontology_or_mapping_mismatch
evidence_granularity_mismatch
both_plausible
cannot_determine
```

按用户要求，自动 headline 仍以 captioner 为标准；人工复核结果作为解释性附录，
同时给出“若采用人工裁决”的敏感性分析，不覆盖原始自动分数。

## 10. 产物设计

后续实现建议生成以下目录：

```text
compare/
  config/
    field_mapping.yaml
  inputs/
    canonical_manifest.jsonl
  runs/<run_id>/
    metadata.json
    preflight.json
    normalized_captioner.jsonl
    normalized_sure_tagger.jsonl
    normalized_speaker_profiles.jsonl
    capability_coverage.md
    empirical_coverage.csv
    field_metrics.csv
    metrics.json
    disagreements.jsonl
    confusion/
    report.md
```

`report.md` 的固定顺序：

1. 运行与完整性摘要；
2. 能力覆盖矩阵和双方独有能力；
3. 3000 条实测 known/unknown 覆盖；
4. Tier A-S 样本级核心一致性；
5. Tier A-P 说话人画像一致性；
6. Tier A2 派生一致性；
7. Tier B 扩展诊断；
8. 分歧样本与人工复核；
9. 限制和结论。

## 11. 执行阶段

### 阶段 0：冻结输入

- 生成 3000 条 canonical manifest 和 hash。
- 补齐 captioner 当前缺少的 22 条并排除 `_summary.json`。
- 固化 captioner schema/prompt/result snapshot。

### 阶段 1：运行 sure-tagger

- 使用同一批 3000 个 `.mp3`，transcript 留空。
- 保存 manifest、完整 tags-only 输出、模型配置、命令和日志。
- 保持 deterministic speaker profile 默认开启；不得使用
  `--speaker-profile-disable`。
- 额外保存含 `speaker.profiles` 的 speaker v2 result JSONL，或完整保留每条样本的
  `evaluation_output.json.gz`。只保存当前 tags-only JSONL 不满足画像比较要求。
- 不因为某个 stage 失败删除整行；失败字段保持 `null`。

### 阶段 2：覆盖面报告

- 先生成静态能力矩阵。
- 再按全部 3000 条生成双方字段 known rate、运行成功率和失败原因。

### 阶段 3：一致性报告

- 先跑 Tier A-S，得到样本级核心结果。
- 再按首次出现顺序对齐 profile，跑 Tier A-P 的 speed/pitch/speaker_volume。
- 再跑 Tier A2 和 Tier B，所有映射和标定参数写入产物。

### 阶段 4：分歧复核

- 生成可播放的分歧清单。
- 完成人工原因分类和敏感性分析。

## 12. 验收标准

方案实现完成应满足：

- 任一样本可通过 `sample_id` 追溯到音频、caption、captioner JSON、sure JSON。
- 3000 条输入、系统成功数、字段 known 数、joint known 数均可核对。
- 空数组、false、0、null、文件缺失不会被混淆。
- Tier A-S 四个字段有完整单项指标、混淆矩阵和 effective agreement。
- Tier A-P 的语速档位、音高、音量有 profile 对齐率、entity-micro 和 sample-macro
  指标；无法对齐、超出值域和字段 null 分开统计。
- Tier A2 明确标注 closed-world 假设。
- Tier B 的 taxonomy 映射和标定过程可复现，且不发生评测集泄漏。
- 双方独有能力不被强行打成“不一致”。
- 报告中任何总分都能从公开的逐字段计数重新计算。

## 13. 推荐的最终结论形式

正式结论不使用一句“系统 A 覆盖率高于系统 B”概括所有维度，而分成三句话：

1. **能力互补性**：captioner 的画像更宽，覆盖年龄、性别、距离、口音、情绪、
   音色等；sure-tagger 已具备音频直接计算的语速、相对音高和片段内相对音量画像，
   并更擅长文件/VAD、物理声学、连续质量指标和语言内容。
2. **直接交集一致性**：分别报告 Tier A-S 四个说话人核心字段，以及 Tier A-P
   语速档位、音高、音量三个画像字段；样本级和实体级结果不合成一个总分。
3. **扩展交集表现**：音乐/语音正例、声事件粗类和连续声学指标分别报告，明确
   closed-world、taxonomy 和标定限制，不与核心总分混合。
