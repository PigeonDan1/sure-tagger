# 多数据场景聚类：k-medoids

本源码包提供 k-medoids 场景聚类。聚类和错误率诊断使用 Python 3.9+ 标准库，读取已经生成的结构化标签，无需音频模型或 GPU。

## 文件

本目录与 `sure-tagger-main/` 同级；脚本位于 `scripts/`：

| 文件 | 用途 |
| --- | --- |
| `cluster_scenes.py` | 标签特征、混合类型距离、k-medoids、候选筛选、split/merge 和分区输出。split 使用相同的 k-medoids 算法。 |
| `cluster_methods.py` | k-medoids 候选生成与候选筛选的评分辅助函数。 |
| `error_homogeneity.py` | 冻结分区后的 WER/CER 诊断。 |

`tests/` 包含特征、k-medoids 和候选筛选测试。请保留脚本的目录结构，以便同目录导入。

## 准备输入

输入数据在源码包外准备。将 `INPUT_ROOT` 设置为自己的数据目录，布局如下：

```text
INPUT_ROOT/
├── sampled-100-per-source/
│   ├── tags.jsonl               # 必需：结构化标签
│   ├── index_map.jsonl          # 必需：样本 ID 与 global index
│   ├── manifest.jsonl           # 必需：抽样清单
│   ├── sample_wer_cer.jsonl     # WER/CER 诊断所需
│   └── tag_run_metadata.json    # 可选：标签运行元数据
└── wer-cer-results/
    └── sample_sidecar.jsonl     # 可选：ASR 评测元数据
```

标签、索引和清单的样本数与行序必须一致，`global_index` 从 0 连续编号。诊断输入还须按 `sample_id` 和 `global_index` 与标签对齐。音频路径可以保留在清单中，聚类不会打开音频。

## 运行聚类

在源码包根目录执行，并替换下面的 `INPUT_ROOT`：

```bash
export INPUT_ROOT=/path/to/multi-data
export CLUSTER_OUTPUT="$PWD/runs/clustering_kmedoids"

PYTHONDONTWRITEBYTECODE=1 python3 -u \
  clustering/scripts/cluster_scenes.py \
  --tags "$INPUT_ROOT/sampled-100-per-source/tags.jsonl" \
  --index-map "$INPUT_ROOT/sampled-100-per-source/index_map.jsonl" \
  --sampled-manifest "$INPUT_ROOT/sampled-100-per-source/manifest.jsonl" \
  --run-metadata "$INPUT_ROOT/sampled-100-per-source/tag_run_metadata.json" \
  --cluster-methods kmedoids --k-min 6 --k-max 12 --seeds 0,1,2 \
  --no-wer-filter --skip-diagnostics \
  --output-dir "$CLUSTER_OUTPUT"
```

命令仅使用声学标签指标选择候选，搜索 K=6…12、种子 0/1/2，再执行标签驱动的 split/merge。`--cluster-methods` 仅接受 `kmedoids`。结果目录包含 `cluster_assignments.jsonl`、`cluster_summary.json` 和 `run_metadata.json`。

## 可选 WER/CER 诊断

对已有分区计算错误率同质性，不改变聚类结果：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  clustering/scripts/error_homogeneity.py \
  --assignments "$CLUSTER_OUTPUT/cluster_assignments.jsonl" \
  --wer-cer "$INPUT_ROOT/sampled-100-per-source/sample_wer_cer.jsonl" \
  --cluster-summary "$CLUSTER_OUTPUT/cluster_summary.json" \
  --no-update-summary --output "$CLUSTER_OUTPUT/error_homogeneity.json"
```

若需要聚类入口同时输出诊断，可从聚类命令移除 `--skip-diagnostics`，增加 `--wer-cer "$INPUT_ROOT/sampled-100-per-source/sample_wer_cer.jsonl"` 和 `--sidecar "$INPUT_ROOT/wer-cer-results/sample_sidecar.jsonl"`；保留 `--no-wer-filter`，使 WER/CER 仅用于分区后的诊断。

## 检查与测试

在源码包根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  clustering/scripts/cluster_scenes.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s clustering/tests -p 'test_cluster*.py' -v
```
