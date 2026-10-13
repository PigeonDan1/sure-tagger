# sure-tagger bridge 使用文档

本目录是 sure-tagger 的语言层桥接实现。目前主流程已经改为 AMI utterance 级打标：从 `/hpc_stor03/sjtu_home/huifei.wang/dataset/all` 中读取每个会议的 `.jsonl` utterance 文件，生成统一 manifest，再对每条 utterance 输出语言层 tags。

## 当前数据

默认数据根目录：

```bash
/hpc_stor03/sjtu_home/huifei.wang/dataset/all
```

该目录下每个 `.jsonl` 文件对应一个 AMI meeting，例如 `ES2002a.jsonl`。每行是一条 utterance，当前适配器要求字段为：

```json
{
  "utt_id": "ES2002a_utt_00002",
  "audio_id": "ES2002a",
  "speaker": "B",
  "start": 55.415,
  "end": 77.456,
  "text": "Um well this is the kick-off meeting.",
  "words": [{"w": "Um", "start": 55.98, "end": 56.53}]
}
```

生成的 manifest 会把每条 utterance 映射为一个样本：

```text
sample_id = utt_id
audio.path = ""
text.transcript = text
native_metadata.utt_id = utt_id
native_metadata.audio_id = audio_id
native_metadata.speaker = speaker
native_metadata.start = start
native_metadata.end = end
native_metadata.text = text
native_metadata.words = words
```

注意：当前 `dataset/all` 只提供 utterance 时间戳和文本，manifest 中的 `sample.audio.path` 暂为空字符串；后续接声学层时再补真实音频路径。

## 可用标签

默认 utterance 流水线只跑确定性语言层标签：

```text
language, word_count, punctuation, filler, repetition
```

各标签含义：

| tag | 类型 | 说明 |
| --- | --- | --- |
| `language` | 确定性启发式 | 基于 Unicode/script 的文本语言识别 |
| `word_count` | 确定性统计 | 统计 token/word 数 |
| `punctuation` | 确定性统计 | 统计标点，并标记是否有句末标点 |
| `filler` | 词表规则 | 检测 `uh`、`um`、`erm` 等 filler |
| `repetition` | 文本规则 | 检测连续重复词或短语 |
| `topic` | 可选 | 基于 taxonomy 的主题分类，默认不随全量流水线开启 |

`topic` 可以用 `heuristic` 或模型 provider 跑。全量 AMI 大约八万多条 utterance，不建议不加限制直接跑模型 topic。

utterance 级 `topic` 默认启用短 utterance guard：如果目标文本只有很短的 acknowledgement/backchannel/filler，例如 `Yeah.`、`Mm-hmm.`、`Okay.`，会直接标为 `other/insufficient_context`，不调用模型，也不把上下文主题硬套到目标 utterance 上。

## 快速开始

进入 bridge 实现目录：

```bash
cd /hpc_stor03/sjtu_home/huifei.wang/sure-tagger/bridge/sure-tagger
```

先跑测试：

```bash
python3 -m unittest discover -s tests
```

对单个 meeting 跑 utterance 级确定性标签：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --run-name es2002a.deterministic \
  --skip-qa
```

输出会写到：

```text
outputs/ami_utterance/
```

核心输出文件：

```text
utterance_manifest.<run_name>.jsonl
utterance_tags.<run_name>.jsonl
build_utterance_manifest.<run_name>.report.json
tag.utterance.<run_name>.report.json
pipeline.utterance.<run_name>.report.json
bad_samples.build_utterance_manifest.<run_name>.jsonl
bad_samples.tag.utterance.<run_name>.jsonl
```

如果不加 `--skip-qa`，还会生成抽样检查文件：

```text
utterance_qa_samples.<run_name>.jsonl
```

## 常用命令

只构建 utterance manifest：

```bash
python3 -m sure_tagger.cli build-utterance-manifest \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --output outputs/ami_utterance/utterance_manifest.es2002a.jsonl
```

对已有 manifest 打标签：

```bash
python3 -m sure_tagger.cli tag \
  --manifest outputs/ami_utterance/utterance_manifest.es2002a.jsonl \
  --config configs/tags_language_mvp.yaml \
  --tags language,word_count,punctuation,filler,repetition \
  --output outputs/ami_utterance/utterance_tags.es2002a.jsonl \
  --run-id es2002a.deterministic \
  --workers 5
```

限制样本数，适合调试：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --run-name es2002a.limit200 \
  --limit 200
```

多个 meeting 用逗号分隔：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a,ES2003a \
  --run-name es2002a_es2003a.deterministic \
  --skip-qa
```

跑全量确定性标签：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --run-name full.deterministic \
  --skip-qa
```

## Topic 标签

`topic` 默认不在 `run-utterance-pipeline` 中开启。需要时显式加 `--include-topic`。

先用 heuristic 小样本检查：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --run-name es2002a.topic_heuristic.limit200 \
  --include-topic \
  --topic-provider heuristic \
  --limit 200 \
  --workers 5
```

只想检查 prompt、schema 和流水线，不调用真实模型：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --run-name es2002a.topic_dry_run.limit50 \
  --include-topic \
  --dry-run \
  --limit 50 \
  --workers 5
```

如果要使用配置里的模型 provider：

```bash
python3 -m sure_tagger.cli run-utterance-pipeline \
  --dataset ami_utterance \
  --root /hpc_stor03/sjtu_home/huifei.wang/dataset/all \
  --meetings ES2002a \
  --run-name es2002a.topic_model.limit50 \
  --include-topic \
  --limit 50 \
  --workers 5
```

topic 配置来自 `configs/tags_language_mvp.yaml`：

```text
taxonomy_path = configs/topic_taxonomy_general.yaml
schema_path = configs/topic_response_schema.json
cache = outputs/cache/topic_llm_cache.jsonl
context.meeting_window_sec = 120
context.speaker_neighbor_segments = 3
context.max_context_chars = 6000
short_utterance_guard.enabled = true
short_utterance_guard.max_tokens = 3
```

utterance 级 topic 会以目标 utterance 为主，同时给模型或 heuristic 提供同一 meeting 邻近窗口和同 speaker 邻近片段作为上下文。

`--workers` 控制 tag 阶段并发数，默认是 `1`。使用真实 API 跑 topic 时可以先试 `--workers 5`；如果网关限流或 fallback 增多，再降低并发数。

## 输出格式

manifest 每行是一个统一样本对象：

```json
{
  "corpus": {
    "dataset_name": "AMI",
    "source_urls": {
      "article": [],
      "github": [],
      "huggingface": [],
      "dataset_card": []
    },
    "native_metadata": {
      "annotation_release": "AMI utterance JSONL from dataset/all"
    }
  },
  "sample": {
    "sample_id": "ES2002a_utt_00002",
    "audio": {
      "path": ""
    },
    "text": {
      "transcript": "Um well this is the kick-off meeting."
    },
    "native_metadata": {
      "utt_id": "ES2002a_utt_00002",
      "audio_id": "ES2002a",
      "speaker": "B",
      "start": 55.415,
      "end": 77.456,
      "text": "Um well this is the kick-off meeting.",
      "words": [{"w": "Um", "start": 55.98, "end": 56.53}]
    }
  }
}
```

tags 每行对应一个 `sample_id`：

```json
{
  "sample_id": "ES2002a_utt_00002",
  "tags": {
    "language": {
      "value": "en",
      "confidence": 0.99,
      "method": "unicode_script_heuristic",
      "tool_version": "language_v0.1.0",
      "reliability": "L1"
    }
  },
  "pipeline": {
    "run_id": "es2002a.deterministic",
    "config": "configs/tags_language_mvp.yaml",
    "created_at": "2026-08-05T..."
  }
}
```

## 质量检查

检查 CLI：

```bash
python3 -m sure_tagger.cli --help
python3 -m sure_tagger.cli run-utterance-pipeline --help
```

跑 bridge 测试：

```bash
python3 -m unittest discover -s tests
```

从仓库根目录跑全仓测试：

```bash
cd /hpc_stor03/sjtu_home/huifei.wang/sure-tagger
python3 -m unittest discover -s tests
```

最近一次实测结果：

```text
ES2002a: 236 utterance manifest records, 236 tag records, 0 bad records
```

## 兼容说明

代码里仍保留 meeting-level 命令：

```text
build-manifest
build-meeting-manifest
run-meeting-pipeline
```

这些命令用于兼容旧流程。当前 bridge 语言层主线请使用：

```text
build-utterance-manifest
run-utterance-pipeline
```
