<p align="center"><img src="./assets/readme/three-layer-audio-metadata-tag.png" width="112" alt="SURE-Tagger 图标：三层音频元数据标签"></p>

<h1 align="center">SURE-Tagger</h1>

<h2 align="center">一种语音的半结构化表征，让智能体真正听懂并分析声音</h2>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

<p align="center">
<a href="#motivation"><kbd>01</kbd> 项目动机</a> · <a href="#demo"><kbd>02</kbd> Demo</a> · <a href="#from-tags-to-analysis"><kbd>03</kbd> 场景分析</a> · <a href="#how-it-works"><kbd>04</kbd> 工作原理</a><br>
<a href="#tag-bench"><kbd>05</kbd> Tag-Bench</a> · <a href="#paper-results"><kbd>06</kbd> 论文结果</a> · <a href="#usage"><kbd>07</kbd> 使用方式</a> · <a href="#license"><kbd>08</kbd> 协议</a>
</p>

SURE-Tagger 将语音证据整理为可查询的标签与有证据支撑的描述。

https://github.com/user-attachments/assets/e2543c8b-6002-40fd-b9ba-c3a99d7cf16f

<p align="center"><sub>SURE-Tagger 中文介绍视频</sub></p>

<a id="motivation"></a>

## 项目动机

自由文本 caption 能概述一段语音，却无法说明未提及的属性究竟不存在，还是缺少证据。显式字段、补充描述和 `null` 使语音条件可以被共同查询与逐项评测。

<p align="center"><img src="./assets/readme/paper-figure-1.png" width="100%" alt="自由文本 caption 会遗漏属性；结构化标签区分已知值与缺失证据"></p>

<p align="center"><sub>结构化标签让缺失的证据变得明确。</sub></p>

<a id="demo"></a>

## Demo · 从音频到标签

半结构化标签将可查询的显式字段与简短描述放在一起；无法从证据确认的属性保留为 `null`。先[试听这段 AMI 会议音频](assets/readme/tagbench/samples/ES2012d_utterance_00018/audio.wav)，再看它的标签：

<table width="100%">
<thead><tr><th width="520">证据</th><th width="1080">结构化标签</th></tr></thead>
<tbody>
<tr><td><strong>基础声学</strong><br><sub>WAV 时长与语音区间</sub></td><td><code>basic_acoustic</code><br>时长 <strong>23.72 秒</strong> · 非语音占比 <strong>0.500</strong></td></tr>
<tr><td><strong>声音事件</strong><br><sub>FireRed AED</sub></td><td><code>sound_field_scene</code><br>检测到语音 · 音乐 <strong>false</strong></td></tr>
<tr><td><strong>说话人与转写</strong><br><sub>MOSS / Nemotron + ASR</sub></td><td><code>speaker</code><br><strong>4</strong> 位说话人 · 重叠语音 <strong>true</strong></td></tr>
<tr><td><strong>语言与文本</strong><br><sub>FireRed LID + 转写文本</sub></td><td><code>language_content</code><br>英语 · <strong>45</strong> 词</td></tr>
<tr><td><strong>证据支撑的描述</strong><br><sub>转写与说话人标签</sub></td><td><code>descriptive.caption</code><br>多人用英语讨论一件设备的外观与技术特点……</td></tr>
</tbody>
</table>

<sub>AMI 参考标注：四位说话人，存在重叠语音。</sub>

<details>
<summary>展开这段音频的标签节选</summary>

数值来自[完整的 SURE-Tagger 预测](assets/readme/tagbench/samples/ES2012d_utterance_00018/full_predictions.json)：

```json
{
  "basic_acoustic": {"duration_sec": 23.72, "nonspeech_ratio": 0.500422},
  "sound_field_scene": {"speech_music_events": ["speech"], "music_present": false},
  "speaker": {"speaker_count": 4, "speaker_overlap": true},
  "language_content": {"language": "en", "word_count": 45},
  "descriptive": {"topic": "Product/device evaluation: whether the device is flashy/fashionable and technologically innovative"}
}
```

</details>

[完整字段与输入格式](sure-tagger-main/docs/tags-and-methods.md)见标签文档。[交互演示页面](sure-tagger-main/demo/index.html)已内置下方五条音频；可[在本地运行](sure-tagger-main/demo/README.md)，也可用 [`build_demo_data.py`](sure-tagger-main/scripts/build_demo_data.py) 导入自己的音频与标签。

<a id="from-tags-to-analysis"></a>

## 从标签到场景分析

组合 `speaker.speaker_count > 1` 与 `speaker.speaker_overlap = true`，即可筛出可比较的音频。在另一组[冻结的场景聚类运行](clustering/README.md)中，仅凭音频标签得到下表分组；ASR WER/CER 只在分组固定后用于评测。

<table width="100%">
<thead><tr><th width="18%">分组 · 条数</th><th width="48%">代表音频</th><th width="34%">查看</th></tr></thead>
<tbody>
<tr><td><code>C00</code> · 341</td><td>较长的单人英语语音</td><td><a href="assets/readme/clustering-pool/C00/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C00/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C00/cluster_assignment.json">分组</a></td></tr>
<tr><td><code>C01</code> · 122</td><td>短中文语音 · 较低信噪比</td><td><a href="assets/readme/clustering-pool/C01/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C01/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C01/cluster_assignment.json">分组</a></td></tr>
<tr><td><code>C02</code> · 286</td><td>短单人中文朗读</td><td><a href="assets/readme/clustering-pool/C02/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C02/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C02/cluster_assignment.json">分组</a></td></tr>
<tr><td><code>C03</code> · 196</td><td>较长的单人英语语音 · 高信噪比</td><td><a href="assets/readme/clustering-pool/C03/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C03/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C03/cluster_assignment.json">分组</a></td></tr>
<tr><td><code>C04</code> · 17</td><td>极短边界片段 · 语音证据冲突</td><td><a href="assets/readme/clustering-pool/C04/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C04/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C04/cluster_assignment.json">分组</a></td></tr>
<tr><td><code>C05</code> · 38</td><td>泰语语音伴随音乐 · 低信噪比</td><td><a href="assets/readme/clustering-pool/C05/audio.wav">音频</a> · <a href="assets/readme/clustering-pool/C05/tags.json">标签</a> · <a href="assets/readme/clustering-pool/C05/cluster_assignment.json">分组</a></td></tr>
</tbody>
<tfoot><tr><td><strong>合计 · 1,000</strong></td><td colspan="2">六个基于标签形成的场景分组</td></tr></tfoot>
</table>

<sub>这些编号只对应本次冻结分组；音频用于展示分组方式，不代表 ASR 得分。[样本说明](assets/readme/clustering-pool/README.md) · [运行来源](assets/readme/clustering-pool/source_run.json) · [音频归属](assets/readme/clustering-pool/ATTRIBUTION.md)</sub>

<a id="how-it-works"></a>

## 工作原理

专项模型分别提取声学、时间、词汇和说话人证据。SURE-Tagger 核对不确定或冲突的信息，生成有类型的字段与证据支撑的描述。

<p align="center"><a href="./assets/readme/how-it-works.png"><img src="./assets/readme/how-it-works.png" width="100%" alt="专项语音模型产生证据，经核对形成结构化标签，再用于跨数据集的组合评测"></a></p>

<a id="tag-bench"></a>

## Tag-Bench

Tag-Bench 将受控合成场景与现有数据集样本结合。合成部分从属性组合选定场景，由 agent 调度语音生成、时间安排和声学处理，再验证生成结果。

<table width="100%">
<thead><tr><th width="520">构成</th><th width="850">来源或生成方式</th><th width="230" align="right">样本数</th></tr></thead>
<tbody>
<tr><td>受控合成</td><td>Agent 调度工具生成并核验</td><td align="right">500</td></tr>
<tr><td>现有数据集 · T1</td><td>AMI 会议</td><td align="right">350</td></tr>
<tr><td>现有数据集 · T2</td><td>FLEURS、LibriSpeech</td><td align="right">150</td></tr>
<tr><td>现有数据集 · T3</td><td>DESED、FSD50K</td><td align="right">150</td></tr>
<tr><td><strong>合计</strong></td><td><strong>合成 + 现有数据集</strong></td><td align="right"><strong>1,150</strong></td></tr>
</tbody>
</table>

<p align="center"><img src="./assets/readme/tagbench-workflow.png" width="100%" alt="属性组合引导受控语音生成和核验，并与现有数据集录音共同组成 Tag-Bench"></p>

<p align="center"><sub>受控合成与现有录音共同组成 Tag-Bench。</sub></p>

五段 Tag-Bench 音频均提供 SURE-Tagger、Captioner 与 Qwen3-Omni 对同一音频的完整预测：

<table width="100%">
<thead><tr><th width="350">样本</th><th width="570">试听重点</th><th width="680">资源</th></tr></thead>
<tbody>
<tr><td>AMI · <code>ES2012d</code></td><td>四人对话与重叠语音</td><td><a href="assets/readme/tagbench/samples/ES2012d_utterance_00018/audio.wav">试听</a> · <a href="assets/readme/tagbench/samples/ES2012d_utterance_00018/full_predictions.json">完整预测</a></td></tr>
<tr><td>AMI · <code>EN2001e</code></td><td>两人对话与短暂重叠</td><td><a href="assets/readme/tagbench/samples/EN2001e_utterance_00071/audio.wav">试听</a> · <a href="assets/readme/tagbench/samples/EN2001e_utterance_00071/full_predictions.json">完整预测</a></td></tr>
<tr><td>FLEURS · 俄语</td><td>语种与语音内容</td><td><a href="assets/readme/tagbench/samples/fleurs_t2_0347/audio.wav">试听</a> · <a href="assets/readme/tagbench/samples/fleurs_t2_0347/full_predictions.json">完整预测</a></td></tr>
<tr><td>DESED</td><td>语音与机械事件</td><td><a href="assets/readme/tagbench/samples/t3_desed_0112/audio.wav">试听</a> · <a href="assets/readme/tagbench/samples/t3_desed_0112/full_predictions.json">完整预测</a></td></tr>
<tr><td>FSD50K</td><td>音乐事件被打标器漏检</td><td><a href="assets/readme/tagbench/samples/t3_fsd50k_0045/audio.wav">试听</a> · <a href="assets/readme/tagbench/samples/t3_fsd50k_0045/full_predictions.json">完整预测</a></td></tr>
</tbody>
</table>

[五条样本与参考信息](assets/readme/tagbench/README.md) · [同音频输出对照](assets/readme/tagbench/COMPARISON.md) · [音频来源与归属](assets/readme/tagbench/ATTRIBUTION.md)

当前仓库的完整评测入口使用现有数据集子集；[数据构成](compare/docs/benchmark-650-dataset-composition.md)与[评分规则](compare/docs/benchmark-v1-29-metrics.md)另见文档。

<a id="paper-results"></a>

## 论文结果

雷达图对比现有录音与合成录音上的逐属性相对表现。每类数据的每个轴都以该轴最优方法归一化；SURE-Tagger 在多数说话人、时间与声学属性上表现领先。

<p align="center"><a href="./assets/readme/paper-results-rps.png"><img src="./assets/readme/paper-results-rps.png" width="100%" alt="SURE-Tagger 与两种 Qwen3-Omni 基线在现有录音和合成录音各属性上的相对表现雷达图"></a></p>

<a id="usage"></a>

## 使用方式

先按[主程序说明](sure-tagger-main/README.md)准备本地模型权重、Python 运行环境和自己的 manifest。在主程序目录运行：

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

<table width="100%">
<thead><tr><th width="590">评测范围</th><th width="200" align="right">样本数</th><th width="810">入口设置</th></tr></thead>
<tbody>
<tr><td>现有数据集结构性 benchmark</td><td align="right">650</td><td>默认输入</td></tr>
<tr><td>原冻结描述性 GT</td><td align="right">648</td><td><code>ALLOW_FROZEN_GT_EXCLUSIONS=1</code>，仅允许既定缺失样本</td></tr>
</tbody>
</table>

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

## 协议

仓库目前没有 `LICENSE` 文件，尚未声明源码开源协议；外部模型与数据遵循各自条款。演示音频的来源见[归属说明](assets/readme/tagbench/ATTRIBUTION.md)。
