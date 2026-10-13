# 650 条结构性评测集（T1-T3）

本目录是第一版结构性 benchmark 的当前有效确定性缩减版，总量固定为 650 条。T4 controlled
数据集已禁用，不再进入后续构建、运行或评分；历史 T4 源数据仍保留在归档目录供审计。

面向写作的数据集构成报告见
[`../../docs/benchmark-650-dataset-composition.md`](../../docs/benchmark-650-dataset-composition.md)。
实际数据分布、结构字段、T3 九类正负例数及明确覆盖缺口见
[`DATA_COMPOSITION.md`](DATA_COMPOSITION.md)。

各结构化标签的 2/1/0、连续指标、ASR、多标签及 `score_rate` 口径见
[`SCORING_STANDARD.md`](SCORING_STANDARD.md)。
当前评分规范版本为 `benchmark-scoring-v0.8`；默认能力雷达包含 18 个能力轴，
其中 `basic_acoustic.nonspeech_ratio` 使用 T1 官方 `silence_ratio` GT 独立计分，并兼容预测端的
`basic_acoustic.silence_ratio` 字段；该字段按 `2*(1-abs(预测-GT))` 评分，不使用 `0.05`
容差除数；标点计数、句末标点和 filler 原始计数保留为诊断字段，不进入能力雷达。

| Track | 数量 | 构成 |
| --- | ---: | --- |
| T1 AMI | 350 | 覆盖全部 170 个 meeting；每个 meeting 2 条，确定性的 10 个 meeting 各加 1 条 |
| T2 language/ASR | 150 | FLEURS 100（10 种语言各 10）+ LibriSpeech 50 |
| T3 event/noise | 150 | DESED 75 + FSD50K 75（dev 38、eval 37） |
抽样使用各分层内的 midpoint quantiles，不使用随机状态，因此重复构建会选择相同的 `sample_id`。

## 目录与清单

- `benchmark.manifest.jsonl`：650 条用于审计的统一清单；其中保留路径/track 等封装字段，
  不能原样交给模型。正式 runner 必须生成只含解码音频和 opaque sample ID 的 model-facing
  runtime view，并剥离 reference transcript、事件标签、stems、recipe 及可由路径推断的标签。
- `benchmark.metadata.jsonl`：与统一 manifest 按 `sample_id` 对齐的 GT/provenance sidecar。
- `benchmark.summary.json`：计数、分层分布和统一清单 SHA256。
- `t1-ami/manifest.jsonl`：从各 meeting 原始 `manifest.jsonl` 抽出的 350 条对应记录，包含 AMI transcript/turn GT；不可送给被测系统。
- `t1-ami/metadata.jsonl`：每条记录的源 manifest 路径、原行号、manifest/record/audio SHA256。
- `t2-language-asr/`：150 条输入音频及 manifest/metadata。
- `t3-event-noise/`：150 条输入音频、原生标签 metadata、版本化 `crosswalk.json` 和
  `closed_world_gt.jsonl`；后者同时物化独立计分的 `sound_field_scene.music_present`。
各 track 自己的 `manifest.jsonl` 中音频路径均相对该 track 目录解析；统一 manifest 中路径相对本目录
解析。GT metadata、AMI reference manifest 和 T3 GT 均不得作为模型输入。

## T3 闭世界口径

T3 只对冻结的三个 `speech_music_events` 类和六个 `external_noise_type` 类采用
`native-label closed-world`：原生正标签经 `t3-native-label-closed-world-v1.0` crosswalk 映射为
正例，其余目标类直接记为负例。负例是评测假设，不是人工确认负例，可能包含漏标假负例；报告中
必须保留该说明。旧面板中的 `pending_human_full_review` 不会复制到本子集 metadata。

## 重建与验收

在共享目录根路径执行：

```bash
python3 compare/dataset/benchmark/build_benchmark_subset.py
python3 compare/dataset/benchmark/build_benchmark_subset.py --validate-only
```

构建器只清理本目录下三个生成 track 目录以及三个统一清单/汇总文件，不修改 AMI 完整数据或
`compare/evaluation/` 下的源面板。验收会检查总数、分项配额、所有音频路径和 SHA256、WAV 可读性、
AMI 170 个 meeting 的原 manifest 回溯、FLEURS 语言配额和 FSD50K split。

所有粗粒度、词面叶字段和派生目标都使用自己的 `gt_mask` 与 eligible 分母计分。参考侧由
transcript/rule 物化 GT 不代表 captioner 可由 ASR 或其它预测字段免费推导；被测系统的公式依赖
字段为 `null` 时，只把跨字段合同记为不可检查，不会清空目标预测或屏蔽该目标的能力分；依赖和
目标都已知时再单列公式一致性。

音频与第三方资源仍受各源数据集许可约束。本目录用于当前共享工作区内的评测整理，不表示获得新的
公开再分发授权。
