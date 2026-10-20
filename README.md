# sure-tagger

从音频生成结构化标签、语言核验与描述，并完成结构评分和描述性语义评分。

- `sure-tagger-main/`：主程序、模型适配器、配置、schema、工具脚本及测试。
- `compare/`：完整流水线、合并与重试、DeepSeek prompt、两套评分和测试。
- [clustering/](clustering/README.md)：独立的 k-medoids 场景聚类、错误率诊断及测试。

## 完整流程

从仓库根目录运行单卡或八卡入口：

```bash
export PYTHON_BIN=/path/to/python
export BENCHMARK_ROOT=/path/to/benchmark
export RUNTIME_ROOT=/path/to/runtime
export GT_ROOT=/path/to/descriptive_gt
export DEEPSEEK_API_FILE=/private/deepseek-api.txt
export DEEPSEEK_WORKERS=500
export SCORING_WORKERS=500
export OUTPUT_ROOT=/path/to/new_output

bash compare/eval/run_sure_tagger_pipeline_1.sh --check-inputs
bash compare/eval/run_sure_tagger_pipeline_1.sh
# 八卡使用 compare/eval/run_sure_tagger_pipeline_8.sh
```

入口默认使用本仓库的 `sure-tagger-main/`。在该目录下配置外部 `models/` 和 `.runtime/`，可用符号链接挂接已有模型与 Python 环境。GPU wrappers 位于 `sure-tagger-main/scripts/gpu_wrappers/`；各模型的路径及运行环境配置见 [主程序说明](sure-tagger-main/README.md)。

运行顺序为：输入覆盖检查 → 分片 → 核心与辅助标签 → 失败重试与合并 → DeepSeek 语言核验和描述 → 描述合并 → 语义评分 → 结构评分和完整性检查。`--check-inputs` 不调用模型或 API。

默认 benchmark 为 650 条；复现原冻结 GT 的 648 条语义评分时设置 `ALLOW_FROZEN_GT_EXCLUSIONS=1`。该选项仅接受原有的两个 GT 缺失样本。数据结构与评分规则见 [benchmark 说明](compare/dataset/benchmark/README.md) 和 [评分标准](compare/dataset/benchmark/SCORING_STANDARD.md)。

模型、运行环境、数据、API 凭据及运行输出由本机提供。直接运行主 CLI 时通过 `--manifest` 指定自己的输入。演示音频及生成的 `demo/data.js` 需另行准备；可用 `sure-tagger-main/scripts/build_demo_data.py` 生成演示数据。

## 场景聚类

`clustering/` 与 `sure-tagger-main/` 同级，读取已有结构化标签，只提供 k-medoids 方法。聚类与分区后的 WER/CER 诊断仅依赖 Python 3.9+ 标准库，无需 GPU。输入数据由使用者在仓库外准备；输入布局、运行参数与命令见 [聚类使用说明](clustering/README.md)。

## 测试

```bash
cd sure-tagger-main
python -m unittest discover -s tests
cd ..
python -m unittest discover -s compare/eval/tests
python compare/eval/test_sure_tagger_pipeline_launchers.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s clustering/tests -p 'test_cluster*.py' -v
```
