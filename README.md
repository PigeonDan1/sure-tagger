<p align="center"><img src="./assets/readme/mark.svg" width="76" alt="SURE-Tagger：波形连接结构化标签的图标"></p>

<p align="center"><strong>SURE-Tagger</strong></p>

<h1 align="center">让 agent 真正听懂语音</h1>

<p align="center"><strong>从语音证据到结构化标签与可组合评测</strong></p>

<p align="center"><a href="#motivation">项目动机</a> · <a href="#demo">Demo</a> · <a href="#tag-bench">Tag-Bench</a> · <a href="#usage">使用方式</a> · <a href="#license">协议</a></p>

SURE-Tagger 将专项模型的语音证据整理为可查询的标签与描述，并在证据不足时保留 `null`。论文研究这些表示如何支持跨数据集的组合评测与目标数据选择；本仓库提供打标、描述和评分入口。

<p align="center"><a href="./assets/readme/video/suretagger-intro-zh.mp4"><img src="./assets/readme/video/suretagger-intro-zh-poster.jpg" width="100%" alt="SURE-Tagger 中文宣传视频；点击播放"></a></p>

<p align="center"><sub><a href="./assets/readme/video/suretagger-intro-zh.mp4">播放中文宣传视频</a> · 166 秒 · H.264/AAC · 960×544</sub></p>

<a id="motivation"></a>

## 项目动机

自由文本 caption 能概述一段语音，却无法说明未提及的属性究竟不存在，还是缺少证据。显式字段、补充描述和 `null` 使语音条件可以被共同查询与逐项评测。

<p align="center"><img src="./assets/readme/paper-figure-1.png" width="100%" alt="论文图 1：对比自由文本 caption 的属性遗漏与结构化表示中明确的值和 null"></p>

<p align="center"><sub>论文 Figure 1；图中的字段是概念节选，实际 CLI 字段见下方 demo。</sub></p>

<a id="demo"></a>

## Demo：从音频到标签

输入为音频和逐行 manifest，输出为 tags-only JSONL。下面摘自[标签文档中的示例](sure-tagger-main/docs/tags-and-methods.md#5-输出-json-示例)：

```json
{
  "basic_acoustic": {"duration_sec": 15.963062, "nonspeech_ratio": 0.674248},
  "sound_field_scene": {"speech_music_events": ["singing", "music"], "music_present": true},
  "speaker": {"speaker_count": null, "asr_transcript": "Good morning. Let us begin the meeting."},
  "language_content": {"language": "en", "word_count": 12}
}
```

[完整字段与输入格式](sure-tagger-main/docs/tags-and-methods.md) · [交互演示页面](sure-tagger-main/demo/index.html)（需自行准备音频和 `demo/data.js`，可用 [`build_demo_data.py`](sure-tagger-main/scripts/build_demo_data.py) 生成页面数据）

### 可下载的样本 / Downloadable samples

仓库附带两条带来源说明的 T2 样本，以及从 benchmark review fixture 提取的 Sure-Tagger / Captioner / Omni 对比标签。完整归属信息见 [`ATTRIBUTION.md`](assets/readme/tagbench/ATTRIBUTION.md)。

| 样本 | 场景 | 资源 |
| --- | --- | --- |
| `fleurs_t2_0055` | 普通话单人朗读，低混响、无音乐 | [音频](assets/readme/tagbench/audio/fleurs_t2_0055.wav) · [标签对比](assets/readme/tagbench/labels.json) |
| `librispeech_t2_0048` | 英语单人朗读，清晰语音、无外部噪声 | [音频](assets/readme/tagbench/audio/librispeech_t2_0048.wav) · [标签对比](assets/readme/tagbench/labels.json) |

[同一音频的三系统描述对照 / Same-audio comparison](assets/readme/tagbench/COMPARISON.md)

`labels.json` 中的 `A/B/C` 分别表示 Sure-Tagger、Captioner 和 Qwen3-Omni；这些记录只展示选中的 10 道对比题，不包含完整 benchmark GT。

<a id="tag-bench"></a>

## Tag-Bench

论文中的 Tag-Bench 将受控合成场景与现有数据集样本结合。合成部分从属性组合选定场景，由 agent 调度语音生成、时间安排和声学处理，再验证生成结果。

| 构成 | 来源或生成方式 | 样本数 |
| --- | --- | ---: |
| 受控合成 | Agent 调度工具生成并核验 | 500 |
| 现有数据集 · T1 | AMI 会议 | 350 |
| 现有数据集 · T2 | FLEURS、LibriSpeech | 150 |
| 现有数据集 · T3 | DESED、FSD50K | 150 |
| **合计** | **合成 + 现有数据集** | **1,150** |

<p align="center"><img src="./assets/readme/tagbench-workflow.png" width="100%" alt="论文图 3：从属性组合选择场景，经 agent 调度合成并核验语音，再与现有录音组成 Tag-Bench"></p>

<p align="center"><sub>论文 Figure 3。当前仓库的完整评测入口使用现有数据集子集；[数据构成](compare/docs/benchmark-650-dataset-composition.md)与[评分规则](compare/docs/benchmark-v1-29-metrics.md)另见文档。</sub></p>

README 只展示可公开归属的代表性样本。AMI、DESED、FSD50K 和受控 TTS reference-pool 音频的许可各不相同，完整音频与交互式 100 题评审站点应放在单独项目网页或受控发布位置。

<a id="usage"></a>

## 使用方式

先准备本地模型权重、Python 运行环境和自己的 manifest；配置见[主程序说明](sure-tagger-main/README.md)。在主程序目录运行：

```bash
cd sure-tagger-main
python3 scripts/run_tagger.py \
  --manifest /path/to/manifest.jsonl \
  --output /path/to/tags.jsonl
```

<details>
<summary>展开：完整评测流水线与测试命令</summary>

完整入口需要 benchmark、模型可见的 runtime 音频、描述性 GT 和仓库外的 API 凭据。各路径和 GPU 配置见[入口帮助](compare/eval/run_sure_tagger_pipeline.sh)；`--check-inputs` 不调用模型或 API，但仍要求凭据文件可读。

```bash
# 从仓库根目录运行
export PYTHON_BIN=/path/to/python
export BENCHMARK_ROOT=/path/to/benchmark
export RUNTIME_ROOT=/path/to/runtime
export GT_ROOT=/path/to/descriptive_gt
export DEEPSEEK_API_FILE=/private/deepseek-api.txt
export OUTPUT_ROOT=/path/to/new_output

bash compare/eval/run_sure_tagger_pipeline_1.sh --check-inputs
bash compare/eval/run_sure_tagger_pipeline_1.sh
# 八卡：bash compare/eval/run_sure_tagger_pipeline_8.sh
```

| 评测范围 | 样本数 | 入口设置 |
| --- | ---: | --- |
| 现有数据集结构性 benchmark | 650 | 默认输入 |
| 原冻结描述性 GT | 648 | `ALLOW_FROZEN_GT_EXCLUSIONS=1`，仅允许既定缺失样本 |

单卡与八卡运行请使用各自的 `OUTPUT_ROOT` / `STAGE_ROOT`。

```bash
cd sure-tagger-main
python3 -m unittest discover -s tests
cd ..
python3 -m unittest discover -s compare/eval/tests
python3 compare/eval/test_sure_tagger_pipeline_launchers.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s clustering/tests -p 'test_*.py' -v
```

</details>

主程序：[sure-tagger-main](sure-tagger-main/README.md) · 评分与比较：[compare](compare/docs/benchmark-v1-29-metrics.md) · 独立的 k-medoids 场景聚类：[clustering](clustering/README.md)

<a id="license"></a>

## License

仓库目前没有 `LICENSE` 文件，尚未声明开源协议；外部模型与数据遵循各自条款。
