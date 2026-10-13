# 多数据场景：全部 sure-tagger 标签聚类报告

更新时间：2026-09-15  
输入快照：`tags-gpu-20260912-202433`，1000 条音频，10 个来源，每个来源 100 条  
实验参考：[`WER_INPUT_EXPERIMENT.md`](../dataset/multi-data/WER_INPUT_EXPERIMENT.md)
可视化报告：[`all-labels-clustering-experiment.pdf`](../dataset/multi-data/all-labels-clustering-experiment.pdf)

## 1. 结论摘要

本次使用 `cluster_scenes.py --use-all-sure-tagger-labels`，将快照中发现的全部
sure-tagger 结构化标签纳入距离计算；WER/CER 没有参与 assignment，只在分区冻结后做
诊断。最终选择 k-medoids `K=6`、seed=2，输出到
`compare/dataset/multi-data/scene-clusters-all-labels/`。

结果不支持把“全部标签”直接替换成新的生产场景分区：

- 标签距离代价 `C_tag=0.0992`，高于原四轴 tags-only 的 `0.0705`；`Q_tag=0.4912`，低于基线的 `0.6064`。`J=0.212948`，基线为 `0.194730`。这些 J 值属于不同特征空间，不能当成同一尺度的绝对优劣，但方向说明新增标签没有改善标签空间内的紧度或来源留一复用性。
- 新分区与旧 tags-only 分区的 ARI=0.4377、NMI=0.5197，确实发生了明显重分配；与 PDF 中 tags+fun-asr WER/CER 分区的 ARI=0.2904、NMI=0.4040，说明它们不是同一套分桶。
- 新分区的 `source_driven=true`。其中 C02 为 82 条样本，81 条来自 GigaSpeech 泰语，最大来源占比 98.8%，超过脚本的 90% 来源占比阈值。新增语言、transcript 和 profile 标签后，标签空间更容易识别语料来源，而不是只识别可复用的声学场景。
- 冻结后错误率诊断没有统一提升：相对旧 tags-only，SenseVoice 的 η² 从 0.0604 升到 0.1074，Qwen 从 0.4826 降到 0.3984，Whisper 从 0.3660 降到 0.2804，fun-asr 从 0.2665 降到 0.2571。全部标签分区不是跨模型都更好的难度分桶。

因此，本次结果适合作为 `all_sure_tagger_labels` 的探索性/text-aware 变体和敏感性分析，
不应覆盖原来的共享 `audio_tags_only` 生产分区。若目标仍是跨来源、跨模型的现实声学场景，
建议继续使用默认四轴白名单；若确实要纳入语言或文本，应单独命名为 text-aware 版本，
并增加 source-holdout 检查。

## 2. 运行与特征

运行命令：

```bash
cd /hpc_stor03/sjtu_home/weihan.chen/share/compare/dataset/multi-data
python3 cluster_scenes.py \
  --use-all-sure-tagger-labels \
  --output-dir scene-clusters-all-labels
python3 error_homogeneity.py \
  --assignments scene-clusters-all-labels/cluster_assignments.jsonl \
  --cluster-summary scene-clusters-all-labels/cluster_summary.json \
  --output scene-clusters-all-labels/error_homogeneity.json
```

全量模式递归发现并记录了以下特征规格：

| 类型 | 数量 | 处理方式 |
| --- | ---: | --- |
| 数值 | 33 | median/IQR 稳健缩放；退化列不产生 observed 差异 |
| 布尔 | 7 | Hamming 距离 |
| 枚举/单值字符串 | 3 | Hamming 距离 |
| 集合/重复 profile 值 | 16 | Jaccard 距离；数组同时保留容器计数 |

使用的轴为 `content`、`interaction`、`quality`、`spatial`、`language`，各轴在有可比
特征时等权平均。`language` 轴包含 `language_content.*`、transcript 相关字段以及
未归入交互的 deterministic derived 字段；`speaker.profiles[]` 的 gender、pitch、
volume、语速和 speaker_id 作为重复值集合处理。全量 schema 中全为空的
`room_acoustic.far_field` 也被记录，但因两两均缺失不会贡献 observed 距离。

运行元数据标记为：

```text
cluster_rules_version = cluster_rules_v0.5+all_sure_tagger_labels_v1
tag_feature_mode      = all_sure_tagger_labels
assignment_basis      = all_sure_tagger_labels
partition_scope       = shared
production_publishable = true  # 仅表示现有 hard gates 通过
```

`production_publishable=true` 只通过了 `min_k` 和 `C_tag` 两个 hard gate；它不会覆盖
`metrics.source_driven=true` 这一诊断风险。

## 3. 聚类结果

### 3.1 与 PDF 对照

PDF 中的实验是原四轴 tags-only 与加入 fun-asr 错误率轴的模型特定分区。本次全部标签
运行没有使用 WER/CER，因此它应作为第三套独立分区阅读。

| 分区 | 输入 | K | J | Q_tag | fun-asr η² | fun-asr SηK |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 原 tags-only | 四轴音频标签 | 6 | 0.194730 | 0.6064 | 0.2665 | 0.2665 |
| **本次 all-labels** | 全部 sure-tagger 标签，含 language 轴 | **6** | **0.212948** | **0.4912** | **0.2571** | **0.2571** |
| PDF WER-input | 四轴标签 + fun-asr `error_rate` | 8 | 0.301523 | 0.5429 | 0.8576 | 0.7331 |

原 tags-only 与 PDF WER-input 的 ARI/NMI 为 0.3991/0.5264，与 PDF 中的报告一致；本次
all-labels 与原 tags-only 为 0.4377/0.5197，与 PDF WER-input 为 0.2904/0.4040。
ARI/NMI 只表示 assignment 变化，不表示哪套分区更正确。

PDF 的 fun-asr η²=0.8576、SηK=0.7331 是把同一个 fun-asr 错误率直接放进 assignment
后的 in-sample 结果；它可以说明错误轴确实改变了分桶，但不能作为独立泛化验证，也不
应与本次不含错误率输入的 all-labels 场景分区混用。

### 3.2 全量标签分区画像

下表中的 `language` 使用 sampled manifest 的 native language 做事后描述，不参与该表的
聚类输入统计。来源占比是每簇最大来源样本数除以簇大小。

| 簇 | n | 占比 | 主要语言构成 | 来源占比 | 观测标签画像 | Qwen 中位 error_rate |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| C00 | 277 | 27.7% | en 272 / zh 5 | 34.7% | 单人语音；常规长短混合 | 0.160 |
| C01 | 85 | 8.5% | en 41 / zh 44 | 41.2% | 多说话人/短片段信号富集 | 0.111 |
| C02 | 82 | 8.2% | th 81 / en 1 | **98.8%** | 含音乐和外部噪声；几乎等同 GigaSpeech 泰语来源 | 1.000 |
| C03 | 200 | 20.0% | en 100 / zh 100 | 50.0% | 长时片段，duration 中位数约 45.6 秒 | 0.0335 |
| C04 | 67 | 6.7% | th 19 / zh 5 / en 43 | 64.2% | 事件为空、短片段、稀疏语音；边界样本 | 1.000 |
| C05 | 289 | 28.9% | zh 246 / en 43 | 34.6% | 短片段；错误率分布异质，需复核 | 0.000 |

C03 是最容易解释的稳定结构：200 条样本全部来自 ContextASR 英文/中文两个来源，
长时长是强锚点。C02 则是本次最明显的失败模式：它的音乐/噪声标签与语言、transcript
组合一起把 GigaSpeech 泰语样本几乎整簇拉出。C05 虽然 Qwen 中位 CER 为 0，但其 IQR
为 0.2，不能命名为“天然简单场景”。簇名仍是模板化的观测描述，不是会议室、电话会等
物理地点真值。

## 4. 冻结后的跨模型诊断

`error_rate` 缺失不填 0。Qwen、SenseVoice、Whisper 各有 900 条有效分数（500 WER +
400 CER）；fun-asr 有 1000 条有效分数（500 WER + 500 CER）。`SηK` 使用
`SηK = η² - 0.3·log2(max(K,6)/6)`；本次 K=6，所以 SηK=η²。

| 模型 | 有效 n | all-labels η² | 随机 η² | 按来源 η² | all-labels SηK | 相对原 tags-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | 900 | 0.3984 | 0.0062 | 0.6261 | 0.3984 | -0.0842 |
| SenseVoice Small | 900 | 0.1074 | 0.0051 | 0.1784 | 0.1074 | +0.0470 |
| Whisper large-v3 turbo | 900 | 0.2804 | 0.0054 | 0.3920 | 0.2804 | -0.0856 |
| fun-asr-flash-2026-06-15 | 1000 | 0.2571 | 0.0052 | 0.4621 | 0.2571 | -0.0094 |

四个模型的 all-labels η² 都高于相同簇大小的随机基线，说明新分区仍然包含可检测的
误差结构；但是它们全部低于“按来源”对照，尤其 Qwen 和 Whisper。这与
`source_driven=true` 一致：新增标签带来的误差解释，不能简单归因于更好的现实场景
同质性，可能来自语言、语料来源和模型覆盖构成。

与 PDF 中加入 fun-asr 错误轴的结果相比：PDF 的 Qwen/SenseVoice/Whisper/fun-asr
`SηK` 分别为 0.604、-0.006、0.317、0.733；本次 all-labels 分别为 0.398、0.107、
0.280、0.257。PDF 分区的高 fun-asr 数值主要是输入变量回流造成的，不能据此断言
它比 all-labels 更像真实场景；跨模型下降/上升方向才是更有用的迁移诊断。

## 5. 风险与建议

1. **不要替换默认生产分区。** 默认 tags-only 的 `source_driven=false`、`Q_tag=0.6064`
   和较低 `C_tag=0.0705` 更符合跨来源共享场景的目标。all-labels 结果保留作敏感性分析。
2. **如果要保留 language/text/profile 标签，明确标记 text-aware。** 这套输入会把语言、
   transcript、profile 属性和来源绑定；至少应报告 source-holdout、language-holdout 和
   missingness 审计，不能只报告总体 η²。
3. **不要把 C02/C04/C05 的错误率中位数当场景难度真值。** C02 接近来源单一，C04/C05
   含短参考或错误率饱和效应；发布 ASR 结果时应单独报告有效 n 和短参考敏感性。
4. **WER-input 继续作为明确的 model-specific 消融。** PDF 中 K=8、fun-asr η²=0.8576
   的实验可用于展示错误轴的上限和信息回流风险，但不得与 shared cluster_id 混用。

## 6. 产物与复现

- 全部标签 assignment：`compare/dataset/multi-data/scene-clusters-all-labels/cluster_assignments.jsonl`
- 全部标签摘要：`compare/dataset/multi-data/scene-clusters-all-labels/cluster_summary.json`
- 全部标签运行元数据：`compare/dataset/multi-data/scene-clusters-all-labels/run_metadata.json`
- 冻结后错误率诊断：`compare/dataset/multi-data/scene-clusters-all-labels/error_homogeneity.json`
- 全部标签对比 PDF：`compare/dataset/multi-data/all-labels-clustering-experiment.pdf`
- 对比图：`compare/dataset/multi-data/images/all_labels_cluster_regression_three_models_horizontal.png`
- 已废弃的 WER-as-input 说明：`compare/dataset/multi-data/WER_INPUT_EXPERIMENT.md`

复现全部标签分区时使用上面的命令；复现默认分区则去掉
`--use-all-sure-tagger-labels` 并改用另一个输出目录。所有比较均按 `sample_id` 对齐，
不按 JSONL 行号跨文件猜测样本，不对缺失错误率补零。
