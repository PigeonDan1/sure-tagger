# multi-data 场景聚类：k-medoids

源码包提供一种聚类方法：在预计算标签相异度矩阵上运行 k-medoids。输入准备、运行命令与测试见 [聚类使用说明](../../clustering/README.md)；完整设计约束见 [场景分组方案](multi-data-scene-classification.md)。

## 距离与代表点

默认使用已校准的音频标签白名单，将数值、布尔和集合特征组织为 content、interaction、quality、spatial 四轴：

```text
d_tag(i,j) = mean(d_content, d_interaction, d_quality, d_spatial)
```

数值特征按 median/IQR 稳健标准化；缺失由有上限的 missingness 项处理。具体特征、截断和缺失规则见 [cluster_scenes.py](../../clustering/scripts/cluster_scenes.py) 中的 `default_feature_specs`、`axis_distance` 与 `tag_distance`。

每个簇的 medoid 是一个真实样本，以簇内距离和选择，不要求混合标签具有欧氏均值。`kmedoids` 采用距离加权初始化，交替执行最近 medoid 分配和簇内 medoid 更新。

## 搜索与细化

[cluster_methods.py](../../clustering/scripts/cluster_methods.py) 只生成 k-medoids 候选。默认搜索 K=6…12、随机种子 0/1/2；K 上界还受 `floor(N/n_min)` 限制。样本不足以支持 K 下界时，入口会报错。

每个候选计算标签紧致度 `C_tag`、来源留一可恢复性 `Q_tag`、标签锚点、来源占比、小簇与未分配比例。`evaluate_candidate` 以这些量组成目标 `J`，并分别记录硬门和诊断标志。`select_candidate` 优先使用通过硬门的候选，并在可选时排除来源主导候选，再按 `J` 等指标排序。没有通过硬门的候选时，输出只供诊断，不能视作可发布分区。

选中分区后，`refine_split_merge` 仅依据标签执行 split/merge；split 继续使用 k-medoids。最终重新计算 medoid，并按确定性规则生成 `C00`、`C01` 等匿名 ID。最终 K 可能不同于初选 K。

## 错误率的边界

常规场景聚类使用已有标签，不将 WER/CER 放入距离。使用说明中的命令显式设置 `--no-wer-filter --skip-diagnostics`，可完全独立于 ASR 评测数据运行。

CLI 保留默认开启的 `--wer-filter` 兼容选项：在标签距离冻结后，用已有 WER/CER 诊断过滤候选。若希望错误率仅用于最终分区的解释，应显式使用 `--no-wer-filter`。冻结分区后的独立诊断由 [error_homogeneity.py](../../clustering/scripts/error_homogeneity.py) 完成，不改变 `cluster_id`。

`--use-all-sure-tagger-labels` 扩展标签特征范围；`--use-wer-input` 会加入一个 ASR 系统的错误率轴。这些选项仍使用同一种 k-medoids 算法，但会改变距离和规则版本，不能与默认 tags-only 场景定义混用。

## 输出与复现

入口保存 `cluster_assignments.jsonl`、`cluster_summary.json` 和 `run_metadata.json` 等输出。运行元数据记录参数及输入指纹，便于使用相同输入和代码复现；这些文件由运行生成，保存在仓库外或被忽略的 `runs/` 目录。

聚类和错误率诊断仅依赖 Python 3.9+ 标准库。测试覆盖标签特征、k-medoids 与候选筛选，不需要模型权重或 GPU。实验输入、运行结果、对照表和历史报告在源码包外保存。
