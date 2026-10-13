# Speaker v2 直接公开输出

> 更新日期：2026-09-11  
> 状态：certification gate 已移除；speaker profile v0.2 已加入语言路由性别字段。

## 输出行为

Speaker v2 不再提供 `SpeakerEvidenceConfig.certification_gate_enabled` 或
`--certification-gate-enable`。resolver 完成 claim 路由后，直接把可决策结果写入
`evaluation_output.speaker` 和 `public_adapter.speaker`：

```json
{
  "speaker_count": 2,
  "multi_speaker": true,
  "speaker_change_count": 1,
  "speaker_change": true,
  "overlap_ratio": 0.125,
  "speaker_overlap": true,
  "profiles": [
    {
      "speaker_id": "speaker_1",
      "gender": "female",
      "speech_rate": {
        "band": "normal",
        "value": 4.2,
        "unit": "zh_char_per_sec"
      },
      "pitch": "mid",
      "speaker_volume": "normal"
    }
  ]
}
```

`profiles` 使用与 `speaker_count` 相同的 decision timeline。没有可用时间轴时为
`null`，有效音频但没有语音时为 `[]`；不可靠的单个画像值也保持 `null`。首版只
发布语言感知语速、相对音高档位、片段内相对音量和模型判断的二元声音性别；
不推断年龄、情绪或口音，也不从 pitch 派生性别。

`evaluation_output.mode` 固定为 `direct`，`production_eligible` 和
`public_metadata_published` 固定为 `true`。`compat_metadata.json.gz` 同步保存同一份
speaker 对象，并设置 `published=true`。

这里有两层输出合同：standalone resolver 的 `public_adapter.speaker` 是上述六个
C/M/O/X 字段加 `profiles`；`run_record` 另行返回路由后的
`speaker_asr_transcript`。主 tagging adapter 在此基础上派生 `speaker_present`，并把
路由文本写入 `speaker.asr_transcript`，因此 tags-only 的完整 speaker 对象共九个
字段。选中分支的原始输出和路由诊断始终只在内部 evidence/fusion artifact 中。

## 空值语义

移除上线门禁不等于强制生成数值。以下情况仍输出 `null`：

- 语言路由选中的 runtime 或模型 pack 缺失、失败或不可用；
- 路由状态非法，无法唯一确定 decision source；
- 数值派生所需的 timeline 字段缺失或非法。

当前独立 ASR/diarization 输出按组件 fail closed：空或非法的 Qwen3/FireRed/MOSS
文本只使 ASR 及其依赖值为 `null`；缺失或非法的 MOSS/DiariZen timeline 使
C/M/O/X 与依赖 timeline 的 profiles 为 `null`。合法的 0 turns / 0 speakers
timeline 是“确认无语音”，不是缺失：公开
`speaker_count=0`、`speaker_present=false`，profiles 为 `[]`。分支之间始终不回退。

claim status、decision source、evidence id、policy version 和 policy hash 继续写入
artifact，供故障定位和质量分析使用，但不再作为发布阻断条件。

## CLI

运行方式不再包含 certification gate 参数：

```bash
python3.11 scripts/run_speaker_evidence_v2.py \
  --manifest <manifest.jsonl> \
  --output-dir <output_dir> \
  --profile quality-shadow
```

若只需回归六个旧字段，可显式使用 `--speaker-profile-disable`；此时画像 evidence
记录为 `missing`，不会影响旧 claim。

`legacy-shadow`、`quality-shadow` 和 `lean-shadow` 名称继续保留，用于模型集合和
claim policy 选择；名称中的 `shadow` 不再表示 public adapter 被禁用。
当前 policy 为 `speaker_v2.claim_policy.20260910.1`。standalone CLI 默认
`legacy-shadow`，主 tagging CLI 默认 `quality-shadow`。

standalone CLI 分别提供 `--firered-asr-enable/disable`、
`--qwen3-asr-enable/disable` 和 `--diarizen-enable/disable`。旧的
`--openasr-enable/disable` 仅作为兼容组别名。

## 语言路由 ASR/diarization

speaker stage 先调用 FireRed LID，再对每条样本并发运行 MOSS-Transcribe-Diarize 与
官方 DiariZen。语言路由仍决定 C/M/O/X 的 timeline：中文和英文使用 MOSS，其他、
混合或未知语言使用 DiariZen。中英文且两条有效 timeline 的人数都 `<=1` 时 ASR
使用 FireRedASR2-AED，中英文其它情况使用 MOSS；其他语言使用 Qwen3-ASR-0.6B。DiariZen
使用固定的 WavLM large-s80-v2、pyannote WeSpeaker、VBx 配置和 PLDA；各路独立校验，
不再依赖 OpenASR/ReDimNet/ForcedAligner 组合。主 tagging
输出的 `speaker.asr_transcript` 和 standalone 的 `speaker_asr_transcript` 承载
该选择的文本。ASR 策略为 `single_speaker_consensus_firered_for_zh_en_else_qwen3_asr`；
任一人数缺失或非法时不触发共识。选中 ASR 失败时相关公开文本为 `null`，不会
调用另一候选补位。语言路由选中的 `timeline_summary` / `raw_turns` 驱动 speaker
C/M/O/X；另一 timeline 只参与 ASR 人数共识，不改变 speaker claim。语言、selected source、原始输出、
模型 runtime provenance 和失败原因保存在 fusion/evidence artifact，不直接进入公开
tags 对象。FireRed 模型资产与历史
7 集对照见
[`asr双路评测报告_20260831.md`](../evalue/asr双路评测报告_20260831.md)。

policy 内四个 C/M/O/X claim rule 都使用 `selection="language_routed"`，且
`fallback_sources` 为空。resolver 必须收到样本的 `language_route` 对象；缺失、
未知或不受支持的 route 会直接报错，不能根据哪一路 evidence 恰好可用来猜测分支。
实际 claim route 始终记录 `language_selected_source`：该 source 可用时记录
`selection="language_route"`，不可用时记录 `selection="none"`、fallback reason
并产生 `null` claim，不会尝试另一 source。这一区分也避免把 policy 选择模式与
某条样本的最终结果混为一谈。

独立 speaker CLI 与主 tagging CLI 都提供 standalone FireRed LID 参数：
`--firered-lid-python`、`--firered-lid-model`、`--firered-lid-repo`、
`--firered-lid-use-gpu` 和 `--firered-lid-half`。LID 加载或推理失败时按 unknown
处理，ASR 进入 Qwen3-ASR-0.6B 分支，speaker timeline 进入官方 DiariZen 分支；
具体错误保存在 `asr_route.language_error`。

FireRed 只用于中英文；其他、混合、未知或 LID 失败的 ASR 由
Qwen3-ASR-0.6B 自动识别语言并转写。Qwen3 模型声明支持 30 种语言和 22 种中文
方言；DiariZen timeline 仍独立判断 speaker，不使用 ASR 文本补位。

## 历史 OpenASR 部署记录

以下 OpenASR 0.1.40 参数、pack 和命令只用于复现 2026-09-04 及更早产物，不属于
当前 tagging pipeline。当前链路使用独立 FireRedASR2-AED、Qwen3-ASR-0.6B 与官方
DiariZen adapter。

两套 CLI 也都提供 `--openasr-executable`、
`--openasr-firered-asr-model-pack`、`--diarizen-model-pack`、
`--redimnet-model-pack`、`--qwen3-forced-aligner-model-pack` 和
`--openasr-timeout-sec`。主 tagging CLI 的默认值来自 `tagger/local_config.py`；
forced-aligner 默认键为 `OPENASR_FORCED_ALIGNER_MODEL_PACK`。不显式传入 config 的
library 调用还可用 `TAGGER_OPENASR_RUNTIME`、`TAGGER_DIARIZEN_MODEL_PACK`、
`TAGGER_OPENASR_REDIMNET_PACK`、`TAGGER_OPENASR_FIRERED_ASR_PACK` 和
`TAGGER_OPENASR_FORCED_ALIGNER_PACK` 覆盖默认路径。底层 OpenASR 子进程使用
`OPENASR_FORCED_ALIGNER_PACK` 指向 aligner pack。

底层命令固定为：

```bash
OPENASR_DIARIZEN_PACK=models/diarizen-large-s80-v2/diarizen-large-s80-v2-fp16.oasr \
OPENASR_REDIMNET_PACK=models/redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr \
OPENASR_FORCED_ALIGNER_PACK=models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr \
OPENASR_DIARIZE_DEBUG=1 \
.runtime/openasr/0.1.40/openasr transcribe AUDIO \
  --model-pack models/firered-aed-l-v2/firered-aed-l-v2-q4_k.oasr \
  --diarize --offline --no-punctuate --format json
```

OpenASR binary 与 FireRed pack 为 Apache-2.0，DiariZen pack 为 CC BY-NC 4.0，
ReDimNet pack 为 MIT，Qwen3-ForcedAligner pack 为 Apache-2.0。四个 `.oasr` 都是
OpenASR 加载的 model pack；DiariZen pack 不能作为 standalone runtime 执行。
Qwen3-ForcedAligner pack 位于
`models/qwen3-forced-aligner-0.6b/qwen3-forced-aligner-0.6b-q4_k.oasr`，固定为
765,395,744 bytes、SHA-256
`5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b`、OpenASR
catalog revision `3a82628cf7036c2ad0128f2d35bd1cced09b3100`。缺少它时 offline
`transcribe --diarize` 整体失败，两类 OpenASR evidence 都记为 missing。

OpenASR runtime 本身也强固定，默认部署路径为
`.runtime/openasr/0.1.40/openasr`。进程内首次使用对应 resolved binary identity 时，
adapter 校验该 executable（包括 CLI/env 路径覆盖）必须为 42,268,384 bytes、SHA-256
`98d973d0d72b47198ecf17e4c00b2710d84c351b77d3c8024a7256bcea59a1a5`，且
`--version` stdout 必须精确等于 `openasr 0.1.40`。任一不符都在 ASR/diarization
前失败。成功结果的 `runtime` provenance 记录 `engine_version`、
`executable_verified`、`executable_sha256`、`executable_size_bytes` 和
`executable_version_output`；config 中的固定 digest 字段为
`runtime_executable_sha256`。
