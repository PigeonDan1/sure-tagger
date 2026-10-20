# multi-data 标签驱动的现实场景分组方案

更新时间：2026-10-20

状态：方法与契约说明；当前脚本规则版本为 `cluster_rules_v0.5`

本包的场景聚类只实现 **k-medoids 的多 K、多 seed 搜索及确定性 split/merge**。
本文同时保留方法理论和后续扩展约束；agentic 搜索、完整消融矩阵、nested group CV、
bootstrap 稳定性和新样本 conformal assignment 属于设计说明，包内没有这些流程的自动实现。
具体命令和实际支持参数见 [聚类使用说明](../../clustering/README.md)。

## 1. 结论与边界

本方案不预先定义十个主类，也不先给类别命名。**只使用 `sure-tagger` 的结构化标签视图**
发现匿名分区（`C00`、`C01`、……）；分区冻结后，再依据每个分区的标签富集和声学统计
总结名称，并用 WER/CER 做事后诊断。精确类别数 `K` 由标签结构的模型选择产生，不预填
成 10；“多于 5 类”是可行性下界，“约 10 类”只作为搜索范围/软偏好，不能用来强行制造
没有证据的类别。

`sure-tagger` 的结构性标签主要描述**可观测的声学条件和说话人交互**，不是“会议室、
车内、直播间”等地点或业务域标签。因此最终名称也只能使用可观测措辞，并把地点/应用
场景作为有 metadata 或人工听审时的低置信假设。不能仅凭 `speaker_count=1` 写成
“播客”，也不能仅凭 `rt60`/`c50` 写成“大厅”。这条边界与
[`tag-schema.md`](tag-schema.md) 中的 evidence boundary 一致。

`cluster_scenes.py` 默认沿用校准过的音频标签白名单（`cluster_rules_v0.5`）。如需检查
全部 sure-tagger 结构化标签，可显式传入 `--use-all-sure-tagger-labels`（别名
`--use-all-sure-tagger-tags` / `--use-all-labels` / `--use-all-tags`）。该模式递归发现快照中的所有非空数值、
布尔、枚举和集合字段，保留全空字段的 schema 记录，并增加 `language` 轴；profile/
transcript 等数组字段按稳定通配路径聚合为集合或计数，避免样本间维度漂移。运行摘要会
记录 `tag_feature_mode`、实际 feature spec、轴和计数，规则版本为
`cluster_rules_v0.5+all_sure_tagger_labels_v1`。它只改变 sure-tagger 标签输入，不会自动
打开 WER/CER；`--use-wer-input` 仍是独立的显式实验开关。

本文的 tags-only 契约要求运行时指定 `--no-wer-filter`，且不启用 `--use-wer-input`。
脚本还保留 WER 候选筛选和 WER 输入选项；这些选项不属于本文约定的可发布 tags-only 流程。

**WER/CER 不是该流程的聚类输入。** 它们不得进入特征、距离、初始分区、split/merge、`K` 选择、
medoid 选择或新样本 assignment。本文约定的 production 模式是 `model_agnostic`：
`assignment_basis=audio_tags_only`，`partition_scope=shared`。WER/CER 只在分区冻结后
用于诊断分析，例如组内误差离散度、跨 ASR 比较和“该场景是否对某模型更难”。只表现为
“低/中/高 WER”且没有可解释标签画像的分组，不得被实现成聚类目标，也不得命名为现实
场景。旧版 `target_aware_joint`（把 `error_rate` 写入 `d_joint`）不再作为本任务方案；
如需对照，只能作为明确标注的消融，不能发布。

最短落地流程是：冻结 tags；构造 `d_tag`；在 `K>=6` 的匿名候选中搜索；用标签结构
hard gates 和目标函数选择并迭代 split/merge；最后固定 `cluster_id`，再总结画像、命名，
并在同一冻结分区上计算 WER/CER 诊断。任一步标签证据不足都应拒识或保留诊断结果，
不能为了凑到约 10 类而放宽门槛，也不能用误差同质性去补标签结构。

建议最终保存三层结果：

1. `cluster_id`：每条样本一个匿名、互斥的分区 ID；ID 与名称分离，便于版本化和复现。
2. `cluster_name` + `cluster_profile`：分组冻结后，根据富集标签和数值画像生成的
   可观测名称；名称不能反过来参与聚类。
3. `secondary_tags` 以及 `needs_review`/`unknown_reason`：保留多值属性、缺失和冲突，
   不把不确定样本强行塞进某个“干净场景”。

分组结果应写入独立的 `cluster_assignments.jsonl`，不要回写或覆盖原始 tags JSONL
及 WER/CER sidecar。原始标签仍按
[`sure-tagger-main/docs/tags-and-methods.md`](../../sure-tagger-main/docs/tags-and-methods.md)
解释；公司侧的 [`schema.json`](../../sure-tagger-main/schema.json) 和
`caption_extraction_schema.json` 是对齐/导出目标，并不是本次运行的场景枚举。
如果需要对接公司 schema，应增加版本化扩展对象或 sidecar，不要把动态生成的簇硬塞进
`audio.tag` 的 `speech/music/noise/audio_event/other` 枚举。

## 2. 输入数据与可复现性

聚类读取使用者准备的结构化标签；输入数据、模型运行记录和实验结果不随源码包提供。
一次拟合应冻结以下输入及其版本：

- `manifest.jsonl`：原始样本身份、来源和必要的 native metadata。
- `index_map.jsonl`：标签行与 `sample_id`、`global_index` 的映射。
- `tags.jsonl`：一行一个样本的结构化标签。
- 可选运行元数据：实际 tagger profile、producer/claim policy 版本和标签 schema 版本。
- 可选诊断输入 `sample_wer_cer.jsonl` 与 `sample_sidecar.jsonl`：仅在所选流程需要错误率诊断时读取。

如果 tags 本身没有 `sample_id`，必须用约定的行号和 index map 恢复样本身份，不能按文件名猜测。
先断言 manifest、index map 和 tags 的行数一致、映射唯一，再生成带 ID 的下游文件。
下游摘要记录实际输入文件的 SHA-256、预处理参数、随机种子和规则版本；不依赖特定目录或固定样本数。

### 2.1 输出完整性与标签质量

运行输出齐全不表示每个标签都高置信。模型失败、未运行或无法判断的字段应保持 `null`，
并在聚类摘要中报告各字段覆盖率和失败/降级来源；`null` 不表示“没有说话人”或“没有语言”。
是否适合拟合应依据可比标签覆盖和质量审计，不能只看上游运行状态。

### 2.2 标签信号的使用边界

字段支持度、缺失率、数值分布和冲突率用于检查特征是否可用，不预先定义类别数或类别名称。
`external_noise_type=[]` 只表示未检出，不能证明绝对安静；`speech_music_events=[]`
不能单独解释为静音。缺少可靠 `far_field` 证据时，不生成近场/远场类别。
`noise_composition.formless` 中的 `Silence` 也不能直接升级为外部噪声场景。
分区和命名应由实际标签结构决定，不能复用某次数据集的统计量作为固定规则。

### 2.3 WER/CER 诊断契约

错误视图应与 tags 使用同一套样本身份，按 `sample_id`/`global_index` 对齐。
`sample_wer_cer.jsonl` 是一行一个样本的嵌套对象；诊断前检查样本集合、映射唯一性及
缺失占位，不按 JSONL 行号猜测连接，也不使用假设的 `metric_kind=both` 宽表。
该文件可以包含 `global_index`、`sample_id`、`dataset`、`key`、`language`、`path`
和 `scores[]`。当前两个诊断入口接受的模型布局不同：

- `cluster_scenes.py` 内置诊断要求每条样本的 `scores[]` 严格包含以下 7 项，且顺序固定：
  `nvidia__parakeet-tdt-0.6b-v2`、`Qwen__Qwen3-ASR-1.7B`、
  `OpenMOSS-Team__MOSS-Transcribe-preview-2B`、`FunAudioLLM__SenseVoiceSmall`、
  `openai__whisper-large-v3-turbo`、`ibm-granite__granite-speech-4.1-2b`、
  `fun-asr-flash-2026-06-15`。以 [脚本中的 `MODEL_IDS`](../../clustering/scripts/cluster_scenes.py)
  为准；`--eval-models` 选择需要汇总的模型，不改变这项输入布局要求。未评测模型仍保留
  对应 `model_id` 的占位项，`error_rate=null`，`wer_valid=false` 且 `cer_valid=false`。
- 独立 [error_homogeneity.py](../../clustering/scripts/error_homogeneity.py) 按 `sample_id`
  连接分区，再按 `model_id` 取分数；不要求 7 项固定顺序，可只提供部分模型。实际汇总模型为
  脚本中的 `DIAGNOSTIC_MODELS`。调用前须确保分区与评分使用同一套样本 ID，每个评分样本
  只有一条记录、每个样本内每个 `model_id` 只有一项；缺失或无效分数只影响诊断覆盖率。

下面仅展示独立诊断可读取的单模型格式，不是内置诊断所需的完整 7 项布局；示例数据为虚构：

~~~json
{
  "global_index": 0,
  "sample_id": "example_sample_0001",
  "dataset": "example_corpus",
  "key": "example_recording_0001",
  "language": "en",
  "scores": [
    {
      "model_id": "Qwen__Qwen3-ASR-1.7B",
      "run_id": "example_eval_run",
      "metric_name": "wer",
      "error_rate": 0.1,
      "wer": 0.1,
      "cer": null,
      "wer_valid": true,
      "cer_valid": false,
      "counts": {"all": 10, "cor": 9, "sub": 1, "ins": 0, "del": 0}
    }
  ]
}
~~~

未收录、无参考或语言不兼容的模型组合保持缺失占位，`error_rate=null`，
`wer_valid=false` 且 `cer_valid=false`。一次诊断先冻结 `eval_asr_system_id` 和评分协议，
再按 `model_id` 精确取一项；匹配不到有效分数时标记 `error_view_missing=true`，
匹配到同一模型多条则拒绝诊断。禁止删行、取第一条、产生笛卡尔积或把缺失填成 0。

从选定的 `scores[]` 项抽取规范记录：`sample_id`、`model_id`、`metric_name`、
`error_rate`、`error_valid`、`n_ref` 和 edit counts。其中
`error_valid` 要求有限且非负的 `error_rate`、`metric_name=wer|cer`、至少一个有效标志为真，
以及 `n_ref=counts.all>0`。
每个样本对指定模型只选择一个有效错误率：`metric_name=wer|cer` 标明单位，
不再拼成 `(wer, cer)` 二维向量，也不假装缺失的另一指标为 0。
原始 `wer`/`cer`/valid 标志、prediction/reference 和报告来源保留用于追溯。
多个模型可以共存于同一输入，但必须在同一冻结分区上分别诊断，不能把不同模型的错误率混成一个通道。

英语 WER 与中文 CER 可以按此方法放入同一标量诊断通道；报告应同时给出语言、
`metric_name` 和参考长度分层结果，不能把两个指标解释成完全相同的测量单位。
缺失错误率的样本仍完整参与标签聚类，组内离散度只在有效子集上计算并报告覆盖率。

评测 provenance 应由 sidecar 保存或补齐：`asr_system_id`、`run_id`、`scoring_protocol`、
`scoring_config_id`、`pipeline_id`、`reference_language`、语言来源、`reference_version`
以及 edit/reference counts。当前 `cluster_scenes.py` 的 sidecar 加载器以
`(sample_id, model_id)` 为唯一键，模型字段也接受 `asr_system_id`；即使
`scoring_config_id` 不同，同一样本和模型的重复记录也会报错。需要比较多个评分配置时，
应先选定一个配置并生成本次诊断的 sidecar。同一 `scoring_config_id` 不允许对应
多套 tokenizer/normalizer。独立 `error_homogeneity.py` 不读取 sidecar。
若对齐视图不带 config ID，应回查 sidecar，不能省略评分版本审计。
包内诊断适配器所用评分协议为 `sure.eval.protocol.v1/standard_system`；兼容输入需保留
对应的 WER/CER pipeline 和规范化说明。

`scoring_config_id` 可由 protocol、pipeline、reference 版本、dataset ID 和规范化配置哈希得到。
不同来源可能具有不同 config，不能用一个 config ID 过滤全部样本。诊断摘要保存所用
ASR、协议、允许的 pipeline 集合和实际 config 集合；每条诊断样本记录自身 config ID。
重复复合键、多模型匹配、缺版本、负数和 `counts.all<=0` 应拒绝计算。

`error_rate` 可以因插入错误大于 1，不要截断到 `[0,1]`。语言分层优先使用
manifest/sidecar 的 native language，不把 tags 中预测的语言当作 ground truth。
缺失错误率不能改变 `cluster_id`；纯标签聚类流程不需要准备 WER/CER 输入。

## 3. 从 schema 到分组特征

分组输入只有结构化标签视图 `X_tag`。标签字段归入五个正交轴，每个轴都保留显式的
`is_unknown`；`null` 不能当成 `false`、`0` 或空数组。`descriptive.*` 自然语言、
数据集名称、source 名称以及全部 WER/CER 字段都不进入标签视图。默认
`tag_assignment_basis=audio_only` 时，任何由 reference 或待评 ASR 派生的
`language_content.word_count`、`punctuation`、`repetition`、`filler` 和 transcript
字段也排除。若 `language_content.language` 来自独立 LID/native metadata 且不使用待评
ASR，可以作为低权重语言 facet；否则它只用于事后按语言分层报告。若业务确实要按
文本/语言分组，单独运行 `text_aware` 版本并在结果中标明。

| 特征轴 | 主要字段 | 可支持的判断 | 不能支持的判断 |
| --- | --- | --- | --- |
| 内容/事件 | `sound_field_scene.speech_music_events`、`music_present`、`external_noise_type`、`noise_composition` | 语音、歌唱、音乐、机械/动物/自然等背景类别 | 具体地点、声源方位和出现时刻 |
| 交互结构 | `speaker_count`、`multi_speaker`、`speaker_change`、`speaker_overlap`、`overlap_ratio` | 单人、多人、轮流发言、重叠发言 | 会议、访谈等业务形式的确定事实 |
| 声学质量 | `snr_db`、DNSMOS、`nonspeech_ratio` | 噪声/质量难度、非语音占比 | “一定在户外”或具体设备型号 |
| 空间/片段 | `rt60_sec`、`c50_db`、`duration_sec`、`silence_segments` | 混响/清晰度、长短片段、稀疏语音 | 房型、墙面材质、远近场（当前字段未实现） |
| 语言/文本 | `language`、`word_count`、`filler`、`repetition`、`asr_transcript` | 语言覆盖、长短文本、语言现象 | 年龄、性别身份、主题对应的真实地点 |

语言、噪声和质量适合作为二级 facet。比如“泰语/低资源”是 ASR 覆盖 cohort，不
是物理声学场景；`profiles[].pitch`/`gender` 不能用来推断儿童或老年。

### 3.1 混合标签向量与距离

对每条样本按冻结的 `tag_assignment_basis` 构造可审计的 `x_i^tag`：

- 数值字段（时长、非语音比例、SNR、DNSMOS、RT60、C50、说话人数、重叠比例）用训练折
  的中位数/IQR 做稳健缩放，并为每个字段增加 missing indicator；IQR 退化时依次使用
  MAD/稳健标准差，仍为常数则从距离中删除并记录 `constant_tag_feature=true`。词数等文本
  字段仅在 `text_aware` 或其 provenance 明确允许时加入；
- 单值类别做 one-hot/Hamming 距离，多值事件/噪声集合做 Jaccard 距离；`speaker_count`
  可保留序数信息，同时提供离散分箱作为敏感性分析；
- 当前 basis 启用的 schema 轴先分别归一化，再按校准集确定轴权重，避免独立 LID 或一个
  高维数组轴独占距离；audio-only 中没有独立语言证据时，语言/文本轴不进入距离。缺失
  维度使用可用维度掩码，不把缺失当作“没有该属性”；
- missing indicator 的轴权重需要设上限；若一个簇主要由 producer 失败/字段缺失区分，
  标记 `missingness_driven=true` 并进入质量审计，不能把它命名成现实场景；
- 每个字段记录 producer provenance。由任一待评 ASR 的输出、中间表示、置信度、路由
  结果或 WER/CER 派生的字段默认排除出 `X_tag`，并标记 `target_model_derived=true`；
  即使保留作辅助分析，也不能充当 tag anchor 或 `Q_tag` 的独立标签证据；
- source、原始录音 ID、相邻切片只用于分层抽样、block split 和泄漏审计，不作为场景
  特征。若某簇几乎等于一个 source，应标记 `source_driven=true` 并降权或拒绝该分裂。

由上述分量得到缩放到 `[0,1]` 的 Gower 风格距离 `d_tag(i,j)`。精确字段白名单、
provenance、距离矩阵、缩放参数、轴权重和缺失策略必须写入规则版本，保证后续
assignment 可复现。若某样本对没有共同可比的标签维度，`d_tag` 记为 `NA`；若某样本
因此没有足够的可比标签邻居，则进入 `C_REVIEW_OUTLIER`/拒识。绝不能把分母为 0 的距离
填成 0。**不构造 `d_err` 或 `d_joint`。**

### 3.2 诊断用错误视图

每个样本对指定诊断模型最多选取一个有效错误率，因此诊断视图 `X_err` 是标量
`error_rate`：英语用 WER，中文用 CER，二者直接放进同一诊断通道比较。`metric_name`
只记录它是哪一种，不构成第二维，也不按固定阈值先切成“高/中/低错误率”。该视图
只在第 4.5 节使用。

对冻结的诊断 ASR，从该样本 `scores[]` 取出对应 `model_id` 的一项：

~~~text
error_i = score.error_rate
m_i     = score.metric_name          # wer | cer | null
valid_i = error_i is not null and (score.wer_valid or score.cer_valid)
n_ref_i = score.counts.all           # WER=参考词数，CER=参考字符数
u_i     = log1p(error_i)             # 仅用于诊断离散度，不进入距离
~~~

所有有效样本共用当前训练折上的**同一套**诊断尺度，不再按 WER/CER 或语言拆成两个
不可比维度。尺度只用于报告 `D_err`，不得回写到 `d_tag`、assignment 或 split/merge。
若 `valid=false`，该样本仍保留其 `cluster_id`，只是不进入该模型的离散度分子；计入
`error_view_missing`，不能把缺失当成零错误。

语言和 `metric_name` 只用于事后分层画像。不同 ASR 系统仍然不可比，不能把两个
`model_id` 的 `error_rate` 混进同一诊断通道。公平比较多个 ASR 时，必须使用**同一个
冻结的 tags-only 分区**分别计算各系统的诊断表。

## 4. 先分组，不先命名

当前实现按
`tag discovery -> k-medoids candidates -> deterministic split/merge -> freeze -> error diagnostic -> naming`
执行。agentic 候选提案和完整 held-out 审计是后续扩展设计，见第 4.3 节。
整个过程中只保留匿名簇 ID；schema 和标签预处理先冻结，分区本身尚未冻结。
WER/CER 从第一步到 freeze 都不可见；它们不能直接按错误率排序后把区间包装成场景，
也不能作为 split/merge 的接受条件。

### 4.1 发现候选分区

使用第 3.1 节的 `d_tag`，只对 k-medoids 的候选 `K` 和随机种子搜索；
全程不读取预先写好的场景名称，也不读取 WER/CER。当前 `--cluster-methods` 仅接受
`kmedoids`。其他算法的研究背景可见
[`multi-data-clustering-methods.md`](multi-data-clustering-methods.md)，不代表本包提供对应实现。
error-only 与 joint 对照仅属于后续消融设计，包内没有完整消融实验入口。

1. 在 `d_tag` 上运行 k-medoids 的多个 K 和多个 seed，保持同一距离与候选评价口径。
   可比标签邻居不足的样本先标为 `review_outlier`，不能静默当作有效零距离样本；
   未进入主簇的样本计入覆盖率和拒识统计。
2. 候选 `K` 从数据规模和最小有效样本数推导，例如
   `K_max=floor(N/n_min)`，而不是写死 10。`n_min` 只服务标签稳定性和来源覆盖；
   不要因为某模型 WER/CER 缺失而改变 `K` 搜索范围。
3. 每个候选只保留匿名 ID（`C00`, `C01`, ...）。此阶段不输出“机械背景”“会议”等
   类名，也不输出“高/低错误率”分箱。

后续公式中，`G_main` 表示匿名场景簇，`N_main` 是其中样本数，`K=|G_main|`。
所有没有进入 `G_main` 的 outlier、冲突和关键缺失样本统一计入 `n_non_main`，分别
报告原因并在目标函数中惩罚。不得把高误差样本丢进 review 来改善诊断表。

### 4.2 分区目标函数和选择原则

tags-only 流程选择分区只看标签指标；WER/CER 离散度不进入 `J(P)`，也不能作为 hard gate。
当前脚本计算标签紧致度、锚点惩罚、K 代价、小簇和非主簇惩罚，并用 leave-one-source-out
的标签 medoid 分类诊断 `Q_tag`。下面保留更完整的评价设计：随机基线 `R_tag`、
bootstrap/轴权重扰动 `Stability`、置信下界和 Pareto/one-standard-error 选择尚未实现；
公式中的 `lambda_s * (1 - Stability)` 因而不属于当前脚本的实际目标函数。

- `Q_tag(P)`：在 source/group-held-out 数据上，用低容量分类器仅凭 `X_tag` 预测
  `cluster_id`。主统计固定为 out-of-fold balanced accuracy `q_raw`；macro-F1 只作辅助。
  除宏平均外必须报告每簇 recall 及 bootstrap 下界。该分类器不负责最终 assignment，
  只衡量 tags-only 簇在留出组上是否可恢复。
- `A_unanchored(P)`：没有稳定标签锚点的主簇样本占比。标签锚点是相对全局具有足够
  lift/log-odds 或数值效应量、且在 block bootstrap 中重复出现的标签或标签组合；阈值在
  校准前冻结，WER/CER 本身不能充当锚点。

~~~text
q_chance(K) = 1 / K
Q_tag(P) = clip([q_raw(P) - q_chance(K)] / [1 - q_chance(K)], 0, 1)

C_tag(P) = sum_(g in G_main) [ n_g / N_main ] * median_i_in_g d_tag(x_i, medoid_g)
C_tag_ref(K) = best observed C_tag(P) over tags-only candidates with |P|=K
C_tag_rand(K) = median C_tag(P_random) over size-matched random partitions
denom_tag(K) = C_tag_rand(K) - C_tag_ref(K)
R_tag(P) = clip([C_tag(P) - C_tag_ref(K)] / denom_tag(K), 0, 1)
           # 仅在 denom_tag(K) > eps 时定义
B_small(P) = sum_(g in G_main) max(0, n_min - n_g) / N_main
B_non_main(P) = n_non_main / N
B_anchor(P) = A_unanchored(P) + max(0, q_min - Q_tag(P))

J(P) = lambda_t * C_tag(P)
     + lambda_a * B_anchor(P)
     + lambda_k * log(1 + K) / log(1 + K_max)
     + lambda_b * B_small(P)
     + lambda_r * B_non_main(P)
     + lambda_s * (1 - Stability(P))
~~~

若 `K<2`，或任一 held-out fold 的 train/test 缺少 production 类而无法计算对应 recall，
则 `Q_tag=NA`。优先重建保持 group 完整且类支持充分的 folds；仍不可计算时 fail-closed，
不能把缺类 recall 或 `Q_tag` 填成 0 后继续通过 hard gate。

`C_tag_ref(K)` 是在相同算法、seed 和计算预算下找到的 tags-only 参考最优，不宣称全局
最优；`C_tag_rand(K)` 的随机分区保留簇大小，并按 source/cohort 分层，避免形成过弱基线。
若 `denom_tag(K)<=eps`，记 `R_tag=NA`、`structure_uninformative=true`；这表示当前 tags
不能支持该 K 的现实场景结构，禁止用任何误差诊断把它放行，只能补标签或人工听审。

`K=1` 仅用于 `C_tag` 的全局基线；`Q_tag`、每簇分类指标、标签锚点、`B_anchor`、`R_tag`、
`Cut_tag` 和完整 `J` 均记为 `NA`，不参加 production 可行性判断或候选排序。

`Stability` 同时包含 source-block bootstrap 的 ARI/VI 和标签轴权重扰动后的一致性
（先转成同方向分数）。所有项先在校准集归一化到可比较范围。
`source_concentration`、每簇来源覆盖和最大来源占比既可作为硬约束，也应作为额外审计项；
若一个簇几乎等于某个 source，默认拒绝或标记 `source_driven=true`。若业务允许真实的
source-specific 场景，可以保留，但不能在未通过 leave-one-source-out 前宣称可泛化。
操作上，当 cluster/source NMI 或任一簇的最大 source 占比超过预注册阈值时置
`source_driven=true`；当仅用 missing indicators 的 held-out surrogate 已能以超过阈值的
balanced accuracy 预测 cluster ID 时置 `missingness_driven=true`。阈值均由 block
permutation/null 校准，不由最终结果临时决定。

当前单方法实现的 hard gates **只保留** `K>=K_min` 和 `C_tag` 可计算。簇大小、
来源纯度、`Q_tag`、`Cut_tag`、锚点等改为 `diagnostic_flags`，不再否决候选。
冻结分区后另报评估分 `S_ηK=η²-λ log2(max(K,K_ref)/K_ref)`（默认 `λ=0.3`、
`K_ref=6`），只描述组内误差相近与类数代价，**禁止**回写 `K`、assignment 或 split/merge。

以下更严格的 guardrail 属于扩展设计约束，当前实现尚未完整提供对应审计：
`B_non_main<=b_non_main_max`，未校准 cohort 比例不超过
`b_unseen_max`；每个 production 主簇至少有一个稳定标签锚点；
`missingness_driven=false` 且默认 `source_driven=false`（只有预注册的 source-specific
任务、有独立 metadata 证据并通过相应留出审计时才允许例外，且不得宣称跨来源泛化）；
`Q_tag>=q_min` 且每簇 recall 通过下限；`R_tag<=epsilon_tag`。再在 `d_tag` 的
mutual-kNN 图上计算被分区切断的近邻边权比例 `Cut_tag`：

~~~text
Cut_tag(P) = sum_((i,j) in G_tag) a_ij * 1[cluster_i != cluster_j]
             / sum_((i,j) in G_tag) a_ij
~~~

`G_tag` 是 `d_tag` 上的无向 mutual-kNN 图，每条互选边只计一次。默认
`k_graph=min(N_main-1, 20, max(5, round(sqrt(N_main))))`，边权固定为
`a_ij=exp(-(d_tag(i,j)/sigma_tag)^2)`，其中 `sigma_tag` 是 calibration 中正的 mutual-neighbor
距离中位数；`k_graph`/`sigma_tag` 随规则版本冻结。阈值按相同 K 的 tags-only/null 结果
校准，要求 `Cut_tag<=c_tag(K)`。若图无边或 `sigma_tag` 退化，记
`Cut_tag=NA`、`structure_uninformative=true` 并 fail-closed。标签签名完全相同的样本默认
must-link，或只允许预注册的小比例违例预算。这些约束定义标签结构可行域 `F_tag`：
算法只能在 `F_tag` 内比较 `J(P)`。不满足者拒绝，不能给它现实场景名。

不要只依赖某一组手工 lambda。然后在通过 guardrail 的候选上画 `C_tag`、`B_anchor`、
稳定性和 `K` 的 Pareto 前沿，选择达到最佳 `C_tag` 的一个标准误以内且 K 较小的分区。
这样 K 是验证结果；6、10 或 12 只是可能落在前沿上的数，而不是输入参数。所有
`lambda_*`、`n_min`、`q_min`、`epsilon_tag`、`c_tag(K)`、`b_non_main_max`、
`b_unseen_max`、最小来源数、最大来源纯度和 source/missingness 判定阈值、
`epsilon_split/merge` 都必须在校准折预注册；最终评估折不能重新调这些值。诊断用的
`n_error_min`、`D_err` 阈值不进入上述列表。

### 4.3 受约束的 agentic split/merge

本节描述尚未实现的 agentic 扩展设计；包内没有 agent/LLM 提案、beam 搜索或 nested CV
编排。当前脚本的 `refine_split_merge` 是确定性循环：split 用同一 k-medoids 将局部簇二分，
merge 检查标签画像和目标函数；候选接受逻辑不读取 WER/CER。

若将来实现 agentic 流程，agent 只负责扩大候选搜索，不能直接裁决类别。设计职责如下：
Tag-view agent 从标签子空间提出结构
split/merge；Leakage critic 检查 source、录音组、language/tokenizer、缺失模式和参考
长度捷径；Deterministic arbiter 统一计算第 4.2 节的 held-out 目标与 hard gates。
Error-view agent **没有写分区的权限**：它只能在分区冻结后标注 `error_heterogeneous`
等诊断旗标。Naming agent 必须等分区冻结后才运行。

每个 proposal 至少保存
`{op, affected_cluster_ids, candidate_assignment, tag_evidence,
deterministic_boundary, provenance}`。用有限宽度 beam 保留 Pareto 候选；只有
deterministic arbiter 可以接受操作，agent/LLM 无权覆盖 `F_tag`、样本量或泄漏约束。
proposal 中不得出现 `error_gain`、标准化 WER/CER 阈值或 `X_err` 作为分裂特征。

每个 outer fold 内再做 inner train/validation：只在 inner-train blocks 上拟合 tag
缩放参数、轴权重、medoid 和候选 split/merge；inner-validation 用于迭代选择，
outer-test 在参数冻结后只审计一次。outer-test 的 WER/CER 若被读取，只能写入诊断表，
不能改变 assignment。

split 接受条件：

- 子簇都达到 `n_min`，且来源覆盖没有恶化；
- 每个子簇都产生稳定标签锚点；同一次 split 的兄弟簇还必须具有**不同**的标签画像，
  用 held-out tags-only 二分类 balanced accuracy 的置信下界或预注册的 profile distance
  验证；
- `Q_tag`、`C_tag`、稳定性和最大来源占比不越过 guardrail；
- 不得因为组内 WER/CER 仍宽而继续拆。若标签结构已稳定、仅误差异质，保留原簇并标记
  `error_heterogeneous=true`。

对标签画像相近或规模过小的簇提出 merge 候选。只有合并后验证集 `J` 改善或在容忍
范围内、且没有稳定的标签差异被抹掉时才合并；两个簇 WER/CER 接近但标签结构稳定不同，
不因“错误率相近”而强行合并。两个簇标签相近但错误率不同，也不因“要把难例分开”
而拒绝合并。

~~~text
for outer_train_dev, outer_test in source_block_outer_folds(data):
    inner_results = []
    for inner_train, inner_val in source_block_inner_folds(outer_train_dev):
        prep = fit_tag_preprocess(inner_train.tags)
        inner_train_x = prep.transform(inner_train.tags)
        inner_val_x = prep.transform(inner_val.tags)
        for theta in candidate(K, seed, split_merge_limits):
            model = fit_tag_partition(inner_train_x, theta)
            model = refine_by_tag_split_merge(model, inner_train_x, theta)
            val_id = assign_with_tag_gate(model, inner_val_x)
            score = evaluate_J_and_tag_guardrails(val_id, inner_val)
            inner_results.append((theta, score))

    theta = one_standard_error_select(inner_results)
    outer_prep = fit_tag_preprocess(outer_train_dev.tags)
    outer_train_x = outer_prep.transform(outer_train_dev.tags)
    outer_test_x = outer_prep.transform(outer_test.tags)
    frozen_model = fit_partition(outer_train_x, theta)     # 参数选择不看 outer_test
    test_cluster_id = assign_with_tag_gate(frozen_model, outer_test_x)
    outer_scores.append(audit_structure(test_cluster_id, outer_test.tags))
    outer_diag.append(audit_posthoc_error_dispersion(     # 诊断，不回写 theta
        test_cluster_id, outer_test.WER_CER
    ))

report_structural_generalization(outer_scores)
report_posthoc_error_diagnostics(outer_diag)
selected_theta = select_by_group_cv(
    calibration_data.tags, preregistered_grid
)  # 不读取 WER/CER，也不按 outer 诊断事后挑配置
final_prep = fit_tag_preprocess(calibration_data.tags)
final_x = final_prep.transform(calibration_data.tags)
final_model = fit_final_partition(final_x, selected_theta)
final_names = summarize_and_name(final_model)             # 最后才命名
attach_error_diagnostics(final_model, calibration_data.WER_CER)
~~~

这里 outer-test 的 WER/CER 被读取，只因为诊断表需要它们；禁止的是用它们重新选择
`K`、阈值或 split/merge。由于 assignment 从不读取错误率，所有 `D_err` 都是
`posthoc_eval_dispersion`，不能写成 joint 拟合。

正式使用时采用 nested source/original-recording block CV；反复迭代只能查看 inner
validation 的标签目标。

### 4.4 类别数和稳定性

当前只对 k-medoids 多 K、多 seed 候选计算 `J`、`Q_tag`、`A_unanchored`、
簇支持度和来源集中度等指标；默认 K=6…12、seed=0/1/2，上限还受 `floor(N/n_min)` 约束。
标签 silhouette、bootstrap/轴权重扰动稳定性及 `K=1` 全局诊断基线是扩展验证设计，
不是当前搜索入口已生成的结果。所有 `NA` 指标不得填 0 后参与排序。

以下是扩展设计所要求的消融矩阵，包内未提供该矩阵的自动运行实现，也不包含实验结果。
实现后，各版本应使用相同 fold、候选 K 和有效样本口径：

| 版本 | 用途 |
| --- | --- |
| tags-only（完整方案） | 唯一可发布的场景分区 |
| leave-one-tag-axis-out | 检查结论是否依赖单一 schema 轴 |
| no-source-block / no-agentic-refine | 分别量化来源泄漏和 agentic 搜索的增益 |
| error-only / joint（禁止发布） | 对照：确认 production 与误差分桶的 ARI 不高 |

统一报告 `R_tag/Cut_tag`、`Q_tag` 及每簇 recall、tag silhouette、source/missingness
与 cluster ID 的 NMI、bootstrap ARI、K、小簇率和拒识率。production 结果与 error-only
分区的 ARI 若异常接近，应标记 `error_aligned=true` 并复核标签是否只是难度代理；
这不能反过来改成 joint 聚类。

完整验证设计建议使用 Pareto 前沿/one-standard-error 规则选最终 K；当前脚本按标签目标函数
选候选，再做确定性 split/merge。默认任务要求多于 5 类，故初始候选设 `K>=6`。
若另行实现上述验证流程，可从 `K in [6, min(15, floor(N/n_min))]` 起搜，
在最优值落在上边界时扩展，而不是把 10 写死。应先计算 `K_max=floor(N/n_min)`；
若 `K_max<6`，直接报告样本支持不足，不能执行 production 搜索。
在该扩展设计中保留 `K=1..5` 作为诊断基线：若所有 `K>=6` 候选都不能通过结构和稳定性门槛，
应明确报告“当前标签不足以支持多于 5 个现实场景”，不能凭空拆分，也不能用 WER/CER
把大簇切开。最终冻结的是 `cluster_rules_version`、距离配置、medoid/模型参数和
split/merge 历史，而不是一组预先命名的主类。

以下新样本 assignment、conformal 校准和漂移监控属于扩展设计；当前入口负责批量拟合，
没有实现这些在线分配接口。设计要求对后续新样本不重新聚类，并采用标签 assignment：

~~~text
A_tag(x_new) = {g: d_tag(x_new, tag_medoid_g) <= r_tag_g
                   and p_tag(g | x_new) >= p_min}
~~~

`r_tag_g`、`p_min` 用校准集冻结；`A_tag` 为空或标签覆盖不足时直接拒识。最终在
`A_tag` 内选择最近 tag medoid，并要求通过 conformal radius 与 top-2 margin，否则输出
`ambiguous`/`review_outlier`。其中 `p_tag` 来自用于计算 `Q_tag` 的低容量 tags-only
surrogate；最终分区冻结后重拟合并单独校准，概率校准集与 conformal 阈值校准集分离，
或使用严格的 cross-conformal 流程。**新样本有没有 WER/CER，都不改变 assignment。**
有错误率时只附加诊断字段。

新批次只有在监控到明显标签分布漂移后才重新拟合并递增版本。conformal/coverage 保证
还要求新批次与 calibration 近似 exchangeable；应监控 tag 的 MMD/PSI、outlier 率和
cohort scale 漂移，越过冻结阈值时只能拒识或重拟合升版，不能继续沿用旧覆盖保证。

### 4.5 冻结后的 WER/CER 诊断

分区冻结后，才允许读取 `sample_wer_cer.jsonl`。诊断目标是回答：“这个标签场景对某
个 ASR 有多难、组内是否同质”，而不是回过头改簇。

先对有效 `error_rate` 做 `u=log1p(error)`；每个 outer/inner fold 的诊断尺度只能由
对应 train/calibration 部分拟合并冻结，不能读取 validation/test 来改聚类，也不得用
它改诊断阈值后再去筛选簇。`error_rate` 保持原始比例单位，大于 1 仍合法。原始值用于
结果报告；缩放后的 `u` 只用于公平比较不同分区的诊断表。对已经冻结的分区
`P={G_1,...,G_K}` 定义：

~~~text
R_cal = robust_scale(u | valid calibration samples)
V = {valid evaluation samples}
n_valid = |V|
s_g = robust_spread(u | G_g ∩ V)          # g 属于 G_main
E = {g in G_main: n_(g) >= n_error_min}
N* = sum_(g in E) n_g
D_err(P) = sum_(g in E) (n_g / N*) * s_g / R_cal
coverage(P) = N* / n_valid
B_error_coverage(P) = 1 - coverage(P)
~~~

`robust_scale` 默认 IQR，退化时依次使用 MAD/稳健标准差；仍退化则标记
`metric_constant=true`，`D_err=NA`。若 `E` 为空，则 `D_err=NA`、`coverage=0`，
不伪装成零离散度。coverage 的分母是全部有效 evaluation 样本，包括最终进入
non-main 的样本。无效/缺失 `error_rate` 不填零。没有有效错误率时，聚类结果仍可
发布，只是该诊断模型的误差表标记为不可用。

`D_err` 越小表示组内 `error_rate` 越一致，这是**描述性**诊断。语言/`metric_name`
分层只作诊断表。若某簇有效数低于 `n_error_min`，该项标记
`insufficient_error_coverage`，不当作零离散度。`robust_spread` 默认使用 P90-P10，
并同时计算 IQR/MAD；每个簇还要报告原始 `error_rate` 的 median、IQR、MAD、P05/P95、
有效样本数、`metric_name` 构成和覆盖率。最大值/最小值只作诊断。另计算组内 Gini
mean difference 作为敏感性指标。与单纯全局一组相比，额外报告：

~~~text
gain = 1 - D_err(P) / D_err(P_one_cluster)
       # 使用相同 evaluation 样本、冻结 R_cal 和 coverage 口径
~~~

并通过 source/original-recording block bootstrap 给出置信区间。诊断按样本等权；
另提供按参考长度分层以及 pooled edit/reference micro 指标，但不能用长样本加权
结果替换逐样本同质性描述。若 `D_err(P_one_cluster)=0` 或无有效样本，`gain` 记为
`NA`。短参考的比率方差通常更大，所以每簇还要报告 `n_ref` 分布。

可选地用 cross-fitted 低容量分类器计算条件贡献，作为“簇是否只是难度分桶”的
诊断，不作为 production gate：

~~~text
I_tag_given_err = [CE_cv(Z | X_err) - CE_cv(Z | X_tag, X_err)] / H(Z)
I_err_given_tag = [CE_cv(Z | X_tag) - CE_cv(Z | X_tag, X_err)] / H(Z)
~~~

两式只在有效主簇支持数 `|supp(Z)|>=2` 且 `H(Z)>eps` 时定义。因为 `Z` 由 tags 产生，
`I_err_given_tag` 应接近 0；若明显大于预注册诊断阈值，标记 `error_aligned=true`，
表示标签场景与错误率共线，命名时更要避免把难度写成地点。`I_tag_given_err` 高则说明
错误率本身预测不了这些场景，这是期望的诊断结果。两项不可辨识时记 `NA`，不得填 0。

若某主簇 `D_err` 明显高于全局且 bootstrap 下界为正，标记
`error_heterogeneous=true`，进入听审/补标签队列，**不自动 split**。

### 4.6 使用模式

- **`model_agnostic`（本任务默认，也是唯一可发布模式）**：只使用 tags 建簇和
  assignment；WER/CER 仅用于事后画像和跨模型比较。输出必须记录
  `assignment_basis=audio_tags_only`、`partition_scope=shared`、
  `diagnostic_error.used_as_input=false`。
- **`target_aware_joint` / error-only**：本版本不再作为方案选项。若实现中仍保留
  代码路径，必须在输出里写 `mode` 为消融名且 `publishable=false`。

如果要公平比较多个 ASR 模型，所有模型必须在**同一个冻结的 tags-only 分区**上报告
`dispersion_evaluations[]`。不能针对每个待比较模型分别用其 WER/CER 重分组后再横向
比较，因为此时 `cluster_id` 已不是同一条件。

## 5. 聚类后总结和命名

先把算法产生的任意 label canonicalize：每簇取 `d_tag` 总和最小的实际样本为
medoid（并以 `sample_id` 破同分），再按 `(medoid_sample_id, cluster_profile_hash)` 排序生成
`C00/C01/...`；这里的 profile hash 只覆盖冻结的结构标签画像，不含名称或错误率。保存
原始 algorithm label 到 `cluster_id` 的映射及 partition hash；不得按
名称或 WER/CER 高低排序 ID。这样不同 seed 的 label permutation 不会伪装成版本变化。

只有在最终匿名分区冻结后才生成名称。对每个 `Cxx` 计算：样本数、来源覆盖、各结构
标签的平滑 log-odds/lift（相对全局）、数值字段的中位数和分位数、缺失率；然后再附加
WER/CER 原始统计、错误离散度和 `error_heterogeneous`。名称优先使用稳定且可观测的标签
组合，例如“多人交互 + 重叠较高”或“语音 + 机械事件富集”；“会议室”“车内”
“直播间”等地点/业务词只能作为单独的 `environment_hypothesis`，并注明证据来源。
**不得用 WER/CER 高低给簇命名**（例如“高错误率场景”），也不得把诊断难度改写成
地点。

名称只是展示层：保留 `cluster_id` 作为唯一主键，名称变更不改变样本归属；名称生成
可以由模板或语言模型完成，但输入只能是簇的标签画像，不能让模型重新决定分组，也不能
把误差表当作命名证据。人工审核后把 `cluster_name_source`、`cluster_name_version` 和
被采用的 top evidence 写入 summary。

下面的形式是**拟合完成后的示例**，不是预先规定的类别：

| 匿名簇 | 发现后的标签画像 | `error_rate` 诊断（事后） | 生成名称（示例） |
| --- | --- | --- | --- |
| `C03` | `speaker_count>=2`、`speaker_overlap=true` 富集 | 中位数中等，组内离散度低 | 多人交互 / 重叠发言 |
| `C07` | `speech` + `mechanical` 富集，SNR 偏低 | `error_rate` 中位数较高 | 机械背景下的低清晰度语音 |
| `C11` | 单人、短片段、无事件富集 | 错误率分布仍宽 | 短单人语音（误差异质，待复核） |

若新的数据得到完全不同的富集标签，应生成新的名称和版本；不能为了复用这三个名字
而修改聚类结果。

## 6. 二级标签（允许多标签）

建议把以下 facet 固定成小词表，簇划分变化时仍保持稳定：
这些是允许跨簇复用的二级描述轴，不是预先规定的主类，也不参与聚类或预先定义主类；
分区冻结后，它们的富集统计可以作为命名证据。WER/CER 分箱不是二级标签。

~~~json
{
  "content_values": ["speech", "singing", "music", "non_speech_candidate"],
  "interaction_values": ["single", "multi", "overlap", "unknown"],
  "language_group_values": ["zh", "en", "other_known", "unknown"],
  "noise_values": ["mechanical", "music", "animal", "nature", "channel_environment"],
  "quality_values": ["high", "medium", "low", "unknown"],
  "room_values": ["dry", "ordinary", "reverberant", "unknown"],
  "temporal_values": ["short_fragment", "long_form", "sparse_speech"]
}
~~~

具体映射建议如下：

- `language_group`：若有独立 native/LID 证据，`zh`/`zh-*` 归为 `zh`，`en` 归为 `en`，
  其它非空语言为 `other_known`，`null` 为 `unknown`；预测语言不可靠时只作诊断。
  它是语言覆盖 cohort，不是地点场景。
- `noise`：来自 `external_noise_type`；`noise_composition` 只用于展开具体声音。
  `formless: Silence` 不放进外部噪声 facet，避免把静音误报为噪声。
- `quality`：内部 `audio_quality.snr_db` 是连续值；本方案沿用公司 schema 中
  `snr_estimation` 的区间命名，定义四档操作分箱（`>20`、`10~20`、`0~10`、`<0 dB`），
  再按冻结的 DNSMOS 阈值合成 `low/medium/high`。这些是分组画像配置，不是内部 tags
  的额外枚举，阈值不应在每个来源单独漂移；阈值表及其 hash 必须进入
  `cluster_rules_version`。不要用 WER/CER 代替 `quality`。
- `room`：可先用 `rt60<0.3`、`0.3~0.6`、`>=0.6` 做干/普通/混响候选，并用
  `c50` 做一致性检查；这是操作分档，不等价于房型。
- `temporal`：例如 `duration_sec<3` 标 `short_fragment`，`duration_sec>=30` 标
  `long_form`；`nonspeech_ratio>=0.5` 可标 `sparse_speech`。`word_count>=80` 只有在
  明确不是 reference/待评 ASR 派生时才可作为 facet；否则只在 `text_aware` 分析中使用。
  对无空格语言不能把 `word_count=0` 当成无内容；应标注 tokenizer coverage。这些标签
  可以与任何簇并存。

## 7. 缺失值、冲突与 assignment basis

这是分组正确性的关键规则：

- `null` 表示未运行、失败或无法判断。它可以作为距离中的 missing indicator，降低
  某个簇的证据覆盖度，必要时标 `needs_review`，绝不能当成“没有噪声/没有说话人”。
  例如 `language=null` 不代表无语言。
- `[]` 表示 producer 成功但没有公开检出，属于已知空集合；仍需和 VAD、ASR、
  `speaker_present` 交叉检查。`speech_music_events=[]` 不能单独解释成静音。
- `false`、`0` 和 `overlap_ratio=0` 是有效观测，不能当缺失。
- `music_present`/事件、`speaker_count`/`multi_speaker`、`speaker_overlap`/
  `overlap_ratio` 之间做一致性审计。冲突样本保留原标签，设置
  `needs_review=true`，不要静默修正原始值。若仍保留在 `G_main`，它必须正常进入所有
  目标项；若排除，则统一映射为 `C_REVIEW_OUTLIER` 并计入 `B_non_main`。
- `far_field` 缺失或无可靠 producer 时，不要生成“近场”默认值。
- `speaker.profiles[].gender/pitch` 不能用来推断年龄、儿童/老年或身份；如需这些
  场景，必须有独立 metadata 或人工审核。

必须分别保存 `tag_assignment_basis`、整体 `assignment_basis` 和 `feature_provenance`。
本方案固定 `mode=model_agnostic`、`assignment_basis=audio_tags_only`，完全排除
WER/CER 和其它待评模型输出。由参考文本或待评 ASR 派生的字段（例如
`language_content.word_count`、`punctuation`、`repetition`，以及
`speaker.asr_transcript`）仍不得进入默认标签视图；它们可以作为独立的 `text_aware`
分析 facet。`duration`、VAD/非语音比例和纯音频声学字段可进入 `audio_only` 标签视图。

`descriptive.topic`、`descriptive.scene_hypotheses` 和 DeepSeek 生成的自然语言只可
作为簇画像解释/抽样复核。它们不是结构性事实，字段缺失时应保持未知；不得
覆盖已经冻结的 `cluster_id`。

### 7.1 对接公司 `environment`/`application_domain`（可选）

如果下游必须使用公司 schema 的环境枚举，建议另写
`environment_hypothesis`，不要直接填充 `annotation` 的事实字段。只有存在明确的
manifest/native metadata 或人工确认时，才把候选升级为正式环境标签；结构化标签最多
提供下面的弱证据：

| 公司环境候选 | 可引用的结构证据 | 仍然缺少的确认 |
| --- | --- | --- |
| 室内安静 | 最终簇 profile 中无已检出噪声、较高 SNR、较短 RT60 | “无噪声”不能证明在室内，更不能证明具体房间 |
| 室内带噪 | 最终簇的 `noise` facet 非空、语音事件已知 | 需要确认声源和地点 |
| 会议室开会 | 簇 profile 富集多人切换/重叠 | 需要 transcript、metadata 或人工听辨，不能由人数单独确定 |
| 车内/高速驾驶 | `mechanical` 且 composition 含 `Vehicle/Car` 等 | 机械声也可能来自其它设备，需 metadata/人工确认 |
| 录音棚录制 | 单人、质量高、低混响 | 声学指标不能证明“录音棚” |
| 音乐/歌唱（直播等） | 簇 profile 富集 `music_present`/`singing` | “直播”应用属性必须来自 metadata 或文本 |
| 医疗、政务、商超、厨房、地铁等 | 主题/背景声只能给很弱提示 | 必须有明确环境元数据或人工标注 |

同理，`application_domain`（教育、医疗、金融等）应优先读取原始数据集 metadata；
ASR 主题只能生成带置信度的 domain hypothesis。这样既能输出用户需要的现实场景视图，
又不会把 schema 未提供的事实伪装成 sure-tagger 标签。WER/CER 不能作为环境证据。

## 8. 输出契约与示例

推荐的 sidecar 行结构如下，`evidence_tags` 只引用实际存在且非 `null` 的路径：
`assignment_confidence` 综合标签覆盖度和簇归属稳定性，不是 WER/CER 准确率
或校准概率。`cluster_name` 是聚类冻结后的展示名称，不能作为 assignment 的输入。
schema validator 必须检查模式不变量：`mode=model_agnostic` 对应
`assignment_basis=audio_tags_only` 且 `diagnostic_error.used_as_input=false`；
若出现 `used_as_input=true`、`assignment_basis=audio_tags_plus_error`、`w_err` 或
`d_joint` 字段，视为契约失败。顶层不需要 `target_asr_system_id`；诊断用的
`eval_asr_system_id` 写在 `diagnostic_error` / `dispersion_evaluations[]` 内。
`scoring_config_id` 可以随来源变化，诊断摘要应保存允许的 config 集合，样本级 config
必须属于该集合，不能假定全部样本共用一个 config。下面 JSON 全部为虚构的格式示例，
不引用真实样本或评测运行。`diagnostic_error` 使用单一 `error_rate` + `metric_name`；
缺失保持 `null`，不得填 0。

本节为建议的扩展输出契约，包含尚未实现的 assignment gate、校准和 schema validator。
当前输出字段以脚本实际生成结果为准；不能把本示例视为这些扩展能力已经实现的证明。

~~~json
{
  "sample_id": "example_sample_0001",
  "mode": "model_agnostic",
  "partition_scope": "shared",
  "cluster_id": "C07",
  "cluster_name": "机械背景下的低清晰度语音",
  "cluster_name_status": "posthoc_reviewed",
  "cluster_name_source": "template_plus_human_review",
  "cluster_name_version": "cluster_names_v0.4",
  "cluster_profile_ref": "cluster_summary.json#C07",
  "secondary_tags": {
    "content": ["speech"],
    "interaction": "single",
    "language_group": "en",
    "noise": ["mechanical"],
    "quality": "medium",
    "room": "ordinary",
    "temporal": ["short_fragment"]
  },
  "assignment_confidence": "high",
  "assignment_approximate": false,
  "assignment_gate": {
    "tag_gate_passed": true,
    "tag_candidate_count": 2,
    "tag_radius_passed": true,
    "top2_margin": 0.18
  },
  "needs_review": false,
  "unknown_reason": [],
  "error_driven": false,
  "error_heterogeneous": false,
  "tag_anchor": {
    "status": "passed",
    "top_features": [
      "sound_field_scene.external_noise_type=mechanical",
      "audio_quality.snr_db=lower_than_global"
    ]
  },
  "assignment_basis": "audio_tags_only",
  "tag_assignment_basis": "audio_only",
  "feature_provenance": {
    "used": [
      "basic_acoustic.duration_sec",
      "basic_acoustic.nonspeech_ratio",
      "audio_quality.*",
      "room_acoustic.*",
      "sound_field_scene.*",
      "speaker.speaker_count",
      "speaker.speaker_overlap",
      "speaker.overlap_ratio"
    ],
    "excluded": [
      "wer_cer.error_rate",
      "reference_derived_text_except_metrics",
      "target_asr_transcript",
      "target_model_derived_tags"
    ]
  },
  "coverage": {
    "cluster_evidence": "complete",
    "unknown_tags": []
  },
  "evidence_tags": [
    "sound_field_scene.speech_music_events",
    "sound_field_scene.external_noise_type",
    "sound_field_scene.noise_composition.mechanical",
    "speaker.speaker_count",
    "audio_quality.snr_db"
  ],
  "cluster_rules_version": "cluster_rules_v0.5",
  "diagnostic_error": {
    "asr_system_id": "Qwen__Qwen3-ASR-1.7B",
    "model_id": "Qwen__Qwen3-ASR-1.7B",
    "run_id": "example_eval_run",
    "scoring_config_id": "scorecfg_sha256:<config_hash>",
    "scoring_protocol": "sure.eval.protocol.v1/standard_system",
    "pipeline_id": "asr.en.wer.whisper_norm_english_v1.wenet_wer_v1",
    "metric_name": "wer",
    "error_rate": 0.1,
    "error_valid": true,
    "n_ref": 10,
    "counts": {"all": 10, "cor": 9, "sub": 1, "ins": 0, "del": 0},
    "used_as_input": false,
    "error_view_missing": false,
    "assignment_role": "posthoc_diagnostic"
  }
}
~~~

对 `review_outlier`/冲突样本，要么输出 `cluster_id=null`，要么使用显式且不冒充现实场景的
`C_REVIEW_OUTLIER`；两种方式都必须 `needs_review=true`、计入 `B_non_main`，并在
summary 中按原因单独统计。

建议同时写 `cluster_summary.json`，其中 `clusters[].cluster_profile` 保存每个簇的 top
富集标签、数值分位数、缺失率、命名证据和事后误差画像；逐行 sidecar 只用
`cluster_profile_ref` 引用，避免重复。summary 还应至少包含：总行数、最终 `K`、各簇
数量/占比和来源覆盖、
`needs_review`/outlier 数量、每个 facet 的 unknown 率、每簇 `error_rate` 的有效数/
中位数/IQR/MAD/P05/P95 及 `metric_name` 构成、`D_err`/gain、`B_non_main`、
`B_error_coverage`、constant/insufficient 覆盖、`error_view_missing` 比例、`Q_tag` 及每簇 recall、
`A_unanchored`、`R_tag`、`Cut_tag`、条件贡献诊断、标签紧致度、bootstrap/轴权重扰动稳定性、
source/missingness NMI、source concentration、拒识率、assignment basis、
规则版本、输入 tags/manifest/`sample_wer_cer.jsonl`/sidecar 的 SHA-256、
实际执行过的扩展验证结果（例如第 4.4 节消融）和 split/merge 历史；
未执行项注明未实现或未运行。这样可以判断某一簇是稳定的标签
信号，还是某个模型、语言、来源、缺失模式或错误率共线造成的假分组。

所有 `D_err` 必须放在 `dispersion_evaluations[]` 的某一项内，不能作为
没有评估角色的裸指标。每一项必须包含 `dispersion_evaluation_role`，本版本枚举严格为：

- `posthoc_eval_dispersion`：在冻结的 tags-only 分区上评估某个 ASR 的 `error_rate`；
  必须保存 `eval_asr_system_id`、`eval_scoring_config_id`、`eval_run_id`、
  `E_eval_sha256` 和 `used_for_assignment=false`。这是默认且唯一允许的角色。
- 禁止 `joint_assignment_dispersion`。旧字段若出现，validator 失败。

一个 summary 可以同时包含多个 `posthoc_eval_dispersion`（每个 ASR 一项），但每项的
样本覆盖、cohort、指标值和置信区间必须分别计算，不能复用一个无 provenance 的
`D_err`。条件贡献不可辨识时还必须输出 `view_contribution_nonidentifiable=true`、
两项贡献为 `null`。

## 9. 设计验证清单与后续步骤

以下是完整方法的验证要求；nested CV、消融矩阵、bootstrap 和听审编排需另行实现，
本包没有自动执行这些扩展验证。运行当前代码不能等同于已完成全部清单。

1. 标签完成后冻结 manifest、tags、index map、`sample_wer_cer.jsonl`、
   `sample_sidecar.jsonl` 和 group map，检查行数、`sample_id`/`global_index` 对齐、
   sidecar 唯一键 `(sample_id, asr_system_id, scoring_config_id)`、参考文本单位和
   JSON schema。聚类拟合阶段不得打开 WER/CER 文件；诊断阶段按 `model_id` 过滤后每个
   `sample_id` 至多匹配一条有效分数，占位缺失保持 `null`，不要混用旧 run 的部分文件。
   英语 WER 与中文 CER 可收成同一个诊断 `error_rate` 通道并给出分层报告，任何语言的缺失均不得填 0。
2. 依据 source + 原始录音/meeting/talk/speaker 做 block split；来自同一录音或说话人的
   相邻切片不得随机分散到 train/dev/test。独立 group 太少时使用整 source 留出或
   group-k-fold，并明确有效独立组数。
3. 使用 nested group CV：inner-train 只拟合标签预处理、tags-only 候选分区和
   split/merge，inner-validation 选择 `K` 及其它超参数；outer-test 的 tags 只用于按
   冻结规则 assignment 和一次性结构审计。outer-test 的 WER/CER 只写诊断表，不能再反向
   调参。没有独立 outer group 时只能称为当前数据上的 post-hoc partition，不能声称对
   新来源稳定。
4. 对冻结分区报告每个诊断 ASR 的 `D_err`、within-MAD、P05-P95、有效覆盖率和相对
   全局的 gain；同时报告 pooled micro（按 `n_ref`）与 source/group macro，并附英/中
   分层诊断。每个离散度指标必须连同 `dispersion_evaluation_role=posthoc_eval_dispersion`
   和对应输入 provenance 一起输出，且 `used_for_assignment=false`。
5. 运行第 4.4 节的完整消融矩阵，并报告 `Q_tag` 及每簇 recall、`A_unanchored`、
   `R_tag`、`Cut_tag`、source/missingness NMI 和轴权重扰动稳定性。可选报告
   `I_tag_given_err/I_err_given_tag` 作为共线诊断，不作为选簇条件。production 分区
   与 error-only 的 ARI 不应异常接近。
6. 运行 source-block bootstrap/leave-one-source-out，检查簇稳定性、来源集中度、
   最小样本数和语言/tokenizer 分层。对短参考、空参考、无效 ASR 和异常数值单独计数，
   不把它们填成 0 或静默剔除，也不用它们改 `cluster_id`。
7. 只有最终匿名簇稳定后才生成名称，并抽样听审每个簇的代表样本和边界样本。名称、
   `environment_hypothesis` 和 `topic` 是后验解释，不能改变 `cluster_id`。误差诊断
   可以进入听审备注，不能成为命名依据。

该方案的目标是让分组在标签完成后可重复、可审计，并由实际标签结构决定保留多少个簇；
它不把数据源名称泄漏成“场景标签”，也不把 WER/CER 排序伪装成现实场景。
