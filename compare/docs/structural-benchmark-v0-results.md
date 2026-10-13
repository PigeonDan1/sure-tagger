# sure-tagger / captioner 结构性评测 v0 结果

更新时间：2026-09-02  
状态：历史 1,000 条 benchmark（含 T4）已物化并验收；当前有效 benchmark 已切换为 650 条 T1-T3，T4 禁用；暂不评 `descriptive.*`

本文记录 [`open-audio-dataset-survey.md`](open-audio-dataset-survey.md) 第一轮实际下载和运行结果。
本轮只评价六个原始结构分组与 `deterministic_derived`，不评价开放描述文本，也不把 raw caption
转换成标签后再反向当作模型原生输出。

## 1. 结论

1. 本轮需要的公开源树和衍生物已经落盘并完成本地完整性审计：ESC-50、LibriSpeech、FLEURS、
   DESED DCASE 2021 task-data、FSD50K、MUSAN、DEMAND 和 RIRS_NOISES 均有可复核路径；逐 artifact
   的 revision、SHA256、字节数、license caveat 和本地/官方状态见
   [`download_manifest.jsonl`](../dataset/open_benchmark/download_manifest.jsonl)，全树机器审计见
   [`open_benchmark_audit.json`](../dataset/open_benchmark/open_benchmark_audit.json)。
2. AMI 300 条历史输出已经按冻结 `structural-v0.1` 重新连接 GT 并计分。共同可评的说话人结构字段上，
   sure-tagger 明显优于 captioner 的旧版抽取结果；但两边原生输出都早于当前 schema，因此该结果只能称
   **legacy compatibility baseline**，不能代表两套系统按新 schema 运行后的最终排名。
3. Public Phase 0 使用 ESC-50、LibriSpeech、FLEURS 各 10 条、共 30 条 blind shared-audio：输入
   SHA256、WAV 头和系统输入路径全部一致。sure-tagger 的 CPU header stage 为 30/30；captioner
   真实音频 API 调用为 30/30 success。
4. Phase 0 的**运行链路通过**，但**schema go/no-go 未通过**：sure-tagger 当前仍直接输出六组
   tags-only，captioner 当前输出 raw caption；两边对完整当前结构 schema 的 native valid rate 都是
   0/30。因而这 30 条只能证明 audio-only 链路可运行，不能生成新的结构能力 head-to-head 排名。
5. 历史 benchmark 曾包含 1,000 条（AMI 350、T2 150、T3 150、T4 350）；T4 已禁用，当前有效
   benchmark 为 650 条 T1-T3。统一 audio-only manifest/metadata、AMI 对应 manifest 和 T3 版本化
   closed-world GT 继续按新构建器验收。主观质量 track 已移除，四个 `dnsmos_*` 字段不参与评测。

## 2. 下载结果

| 数据集 / subset | 实际状态 | 本地内容与验收 | 本轮用途 |
| --- | --- | --- | --- |
| AMI local utterance tree | 本地抽取完整 | 170 个 meeting、12,568 条 manifest 记录及对应 WAV；另保留历史 pilot 300 | 正式 T1 350 条覆盖全部 meeting；历史 300 条仍只作兼容基线 |
| ESC-50 commit `33c8ce9...` | 完成 | 647,303,505-byte archive，SHA256 `f3dabe6a...`；2,000 WAV、50 类、5 folds | T3 smoke test；10 条 Phase 0 |
| LibriSpeech `all/test.clean` commit `71cacbfb...` | 完成 | 350,452,636-byte parquet，SHA256 `7113aa4c...`；2,620 rows | T2 英语 clean anchor；10 条 Phase 0 |
| FLEURS commit `70bb2e84...` | 完成 | 十种语言的 10 个 TSV + 10 个 test archive，共 3,668,559,793 bytes、6,316 条音频；逐包 SHA256/gzip/tar 验收通过 | 400 条来源面板保留；正式 T2 使用 100 条（每语言 10） |
| DESED DCASE 2021 data `f7dc296e...`；代码 archive `0686178d...` | 完成（含显式拒收） | task-data 44,422 WAV；代码/metadata archive 6,172,092 bytes，SHA256 `bed197b6...`；task-data 无统一 archive hash | 250 条候选池保留；正式 T3 使用 75 条；1 个零帧 WAV rejected |
| FSD50K local extracted tree | 完成本地完整性 | 51,197 WAV（dev 40,966、eval 10,231），ground truth/vocabulary/license 字段对齐；源文件 license 混合 | 250 条候选池保留；正式 T3 使用 75 条（dev 38、eval 37）并物化闭世界 GT |
| MUSAN local extracted tree | 完成本地完整性 | 2,016 WAV，九类计数和各目录 LICENSE/README 对齐；OpenSLR archive/revision hash unknown | T4 noise/music source；再分发条款逐目录核对 |
| DEMAND local extracted tree | 完成本地完整性 | 560 WAV，18 场景、16 通道，`demand_noise_durations.scp` 560/560 对齐；官方 archive/license unknown | 仅作本地受控实验来源，不作许可确认 |
| RIRS_NOISES mirror `db444a69...` | 完成（归档及解压树） | `rirs_noises.zip` 1,311,166,223 bytes，SHA256 `3b50cfde...c05dc7`；解压树 61,260 WAV，zip/list/room_info 对齐 | T4 RIR source；第三方镜像与 OpenSLR 官方等价性未独立证明 |

这里的“完成”只针对表中冻结 subset 或本地 extracted tree 的完整性；不等于取得官方 archive
transport 证据或再分发许可。DESED 的代码 archive 与 DCASE 2021 task-data revision 分开记录；
RIRS 的 mirror commit/hash 也不自动证明与 OpenSLR release 等价。未知许可不作为可发布授权。

### 2.1 历史 1,000 条子集与当前 650 条版本

- **统一入口**：[`dataset/benchmark/README.md`](../dataset/benchmark/README.md)。当前
  `benchmark.manifest.jsonl`/metadata 各 650 行；历史含 T4 的 1,000 行产物仅作审计记录。
- **T1**：AMI 350 条，覆盖 170 个 meeting（160 个各 2 条、10 个各 3 条）。选中 WAV 对应的原始
  manifest 记录已抽入 `t1-ami/manifest.jsonl`，源 manifest 路径、原行号和 hashes 可回溯。
- **T2**：FLEURS 100 条（十种语言各 10）+ LibriSpeech 50 条，共 150。
- **T3**：DESED 75 + FSD50K 75（dev 38、eval 37）。`closed_world_gt.jsonl` 已按
  `t3-native-label-closed-world-v1.0` 物化 150 条完整九类 class mask；旧 review 占位状态未复制。
- **T4**：已禁用，不再进入 benchmark 构建、runtime 或 scorer；`evaluation/t4-controlled/` 历史
  源树保留供审计，不作为当前数据集。

## 3. 评测协议

- 字段 universe：[`field_universe.structural-v0.1.json`](../eval/field_universe.structural-v0.1.json)，
  共 35 个结构字段路径。
- GT 与被测系统输出物理隔离，按 `sample_id` 连接；AMI 的 GT builder 不 import 任一系统代码。
- `null`、missing、schema-invalid 和 runtime failure 均保留在 `eligible_gt` 分母中。
- 数值、粗粒度布尔投影和 `deterministic_derived` 均使用各自 GT mask 独立计分，不合并成一个
  无解释的总分。系统细粒度依赖为 `null` 时不屏蔽目标字段；仅把对应公式合同记为不可检查。
- 只有系统的目标字段和依赖字段都已知时才检查内部公式一致性；合同结果不覆盖系统预测，也不替代
  目标字段的能力分。
- Phase 0 的 runtime manifest 不含 transcript、language、event label、source filename 或其他 GT
  sidecar；metadata 独立存放，两个系统读取相同的音频字节。
- captioner raw caption 只用于验证 audio-only runtime，不解析为 schema prediction，也不评价
  描述文本质量。

## 4. AMI 300 条兼容基线

AMI 由 `legacy_200`（两个 meeting）和 `diverse_100`（100 个不同 meeting 各一条）组成；总体
300 条、102 个 group。下表只列主要共同字段和重要接口缺口，完整分 split 结果见
[`evaluation/ami-v0/report.md`](../evaluation/ami-v0/report.md)。

| 字段 | sure-tagger | captioner | 解释 |
| --- | ---: | ---: | --- |
| `speaker.speaker_count` | exact 78.0%；MAE 0.2333 | exact 32.0%；MAE 0.9967 | 300 条均留在分母；captioner 1 个 null |
| `speaker.multi_speaker` | F1 98.7%；e2e acc 97.7% | F1 82.7%；e2e acc 73.0% | captioner 更偏向漏报多说话人 |
| `speaker.speaker_change_count` | exact 45.0%；MAE 0.9267 | 接口不可用 | captioner 旧输出没有 count |
| `speaker.speaker_change` | F1 96.5%；e2e acc 94.0% | F1 78.8%；e2e acc 69.0% | captioner 有 18 个 null |
| `speaker.overlap_ratio` | MAE 0.0496；Spearman 0.8920 | 接口不可用 | sure-tagger 299/300 为已知值 |
| `speaker.speaker_overlap` | F1 94.8%；e2e acc 91.7% | F1 1.6%；e2e acc 17.0% | captioner 有 124 个 null，且 overlap 正例召回很低 |
| `language_content.language` | e2e accuracy 100.0% | 接口不可用 | 本 pilot 全为英语，不能代表多语言能力 |
| `sound_field_scene...speech` | positive recall 99.7% | positive recall 100.0% | AMI 这里只有 speech 正例，不能称闭世界 F1 |

接口与 schema 结果：

- sure-tagger 历史六组输出经 adapter 后覆盖 88.6% 的字段 universe；captioner 旧 annotation 经
  adapter 后覆盖 17.1%。两边 adapter-normalized valid 都是 300/300。
- 两边 **native current-schema valid 都是 0/300**。adapter 合法表示兼容层成功，不等于系统已经
  按当前合同输出。
- sure-tagger 历史结果缺后来新增的 `speaker.asr_transcript`、`speaker.speaker_present` 和两个
  `deterministic_derived` 字段；captioner 还缺 header、language、change count、overlap ratio 等字段。
- ASR 在这批历史接口中不可用。报告中的空 hypothesis 分母只用于保留 e2e 口径，不能解读成
  当前模型真实 WER 为 100%。

## 5. Public Phase 0

Pilot 由 ESC-50 10 条环境声、LibriSpeech 10 条英语朗读和 FLEURS 十语言各 1 条组成。所有输入
统一为 16 kHz、单声道、PCM16 WAV。

| 数据集 | 样本 | 音频 SHA / WAV | sure header | captioner runtime | sure native schema | captioner native schema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ESC-50 | 10 | 10/10 | duration/rate/channel 均 100% | 10/10 success | 0/10 | 0/10 |
| LibriSpeech | 10 | 10/10 | duration/rate/channel 均 100% | 10/10 success | 0/10 | 0/10 |
| FLEURS | 10 | 10/10 | duration/rate/channel 均 100% | 10/10 success | 0/10 | 0/10 |
| **合计** | **30** | **30/30** | **三字段均 100%** | **30/30 success** | **0/30** | **0/30** |

captioner 使用 `qwen-omni-turbo` 的 audio-only 调用。ESC-50 和 FLEURS 均一次成功；LibriSpeech
有 1 条首次请求超时后由 runner 自动重试成功。API key 只从本地 `api.txt` 读取，未写入配置、结果
或报告。

Phase 0 的两个结论必须分开：

- **Runtime gate: GO。** 30 条 blind audio 均可被两边读取，输入 hash 一致，captioner 无需
  reference text 即可完成真实调用。
- **Schema/head-to-head gate: NO-GO。** sure header 产物是六组 tags-only 且没有完整派生层；
  captioner 产物是带 `caption` 的 runtime record，不是目标标签对象。两边需先原生输出当前 35 字段
  合同，才能在 FLEURS transcript/language、LibriSpeech ASR 和 ESC-50 事件映射上计算公平的能力分。

## 6. 当前限制

1. AMI 是历史输出的离线兼容评测，不是今天重新执行两套完整模型的结果；系统版本和接口变化后需重跑。
2. 当前 7 张 GPU 均被其他用户任务占用，本轮没有抢占资源或启动 sure-tagger full inference，只运行
   CPU header stage。Phase 0 因此没有 silence、SNR、room、event、speaker、ASR/LID 的新预测。
3. T3 对 DESED/FSD50K 的九个冻结目标类采用预注册闭世界规则：原生标签映射为正例，没有正映射
   的目标类直接记负例，不再要求独立人工复核。该规则可能引入漏标假负例，必须在 T3 headline
   中显式披露；它不改变 ESC-50、其他 DESED weak subset 或其他 weak 数据集的 PU 口径。
4. 正式 T2 子集已经冻结（FLEURS 每语言 10 条、LibriSpeech 50 条），但尚未在本结果文档中
   声称新的模型 ASR/LID 排名；需先通过当前 native-schema gate，再运行统一 scorer。
5. T4 历史子集不再计入当前结果；`evaluation/t4-controlled/` 仅保留审计用途。主观质量 track 已删除，四个
   `dnsmos_*` 字段统一 `gt_mask=false`。
6. DESED task-data 保留 1 个已知 58-byte 零帧源 WAV 为 rejected；public-eval 许可证表有 55 个
   音频缺少行，均按 unknown 处理，不据此补写许可。

## 7. 产物与复现

- 下载清单：[`dataset/open_benchmark/download_manifest.jsonl`](../dataset/open_benchmark/download_manifest.jsonl)
- AMI 一键评测：[`eval/run_ami_v0.sh`](../eval/run_ami_v0.sh)
- AMI 详细报告：[`evaluation/ami-v0/report.md`](../evaluation/ami-v0/report.md)
- Public Phase 0 运行入口：[`eval/run_public_phase0.sh`](../eval/run_public_phase0.sh)
- Public Phase 0 审计器：[`eval/summarize_public_phase0.py`](../eval/summarize_public_phase0.py)
- Public Phase 0 机器结果：[`evaluation/public-phase0/public_phase0.summary.json`](../evaluation/public-phase0/public_phase0.summary.json)
- Public Phase 0 可读摘要：[`evaluation/public-phase0/public_phase0.summary.md`](../evaluation/public-phase0/public_phase0.summary.md)
- T2 面板与审计：[`evaluation/t2-panel/README.md`](../evaluation/t2-panel/README.md)、
  [`evaluation/t2-panel/t2_panel.audit.json`](../evaluation/t2-panel/t2_panel.audit.json)
- T3 候选池：[`evaluation/t3-event-panel/README.md`](../evaluation/t3-event-panel/README.md)
- 历史 T4 controlled render（仅审计）：[`evaluation/t4-controlled/t4_controlled.summary.json`](../evaluation/t4-controlled/t4_controlled.summary.json)、
  [`evaluation/t4-controlled/t4_controlled.resources.json`](../evaluation/t4-controlled/t4_controlled.resources.json)
- 当前 650 条 bundle：[`dataset/benchmark/README.md`](../dataset/benchmark/README.md)、
  [`dataset/benchmark/benchmark.summary.json`](../dataset/benchmark/benchmark.summary.json)
- 源树总览：[`dataset/open_benchmark/README.md`](../dataset/open_benchmark/README.md)、
  [`dataset/open_benchmark/open_benchmark_audit.json`](../dataset/open_benchmark/open_benchmark_audit.json)

复现命令：

```bash
# AMI 兼容基线
bash eval/run_ami_v0.sh

# Phase 0 默认只校验输入并补齐 sure header，不触发付费 API
bash eval/run_public_phase0.sh

# 显式允许 captioner API；已有 success item 会 resume，不覆盖
bash eval/run_public_phase0.sh --captioner

# 只重建 Phase 0 审计结果
python3 eval/summarize_public_phase0.py

# T2 正式面板审计
python3 eval/audit_t2_panel.py --root evaluation/t2-panel

# 构建当前 650 条 T1-T3 bundle；只复验时追加 --validate-only
python3 dataset/benchmark/build_benchmark_subset.py

# 源树完整性审计（RIRS archive 校验约需读取 1.3 GB）
python3 dataset/open_benchmark/audit_open_benchmark.py \
  --root dataset/open_benchmark --output open_benchmark_audit.json \
  --desed-license-table desed_public_eval_license_resolved.tsv --verify-archives

# 全部评测单测
python3 -m unittest discover -s eval/tests -v
```

## 8. 下一步准入条件

在把这些面板用于 headline scoring 前，先完成以下最小闭环：

1. sure-tagger 原生输出补齐 `deterministic_derived`，captioner 原生输出完整结构对象；两边对当前
   field universe 的 native schema-valid rate 达到预注册门槛。
2. 在本轮冻结的 30 条 shared-audio 上重跑完整推理，保持 manifest、audio SHA 和 GT 不变；运行失败、
   null 和 schema-invalid 继续留在分母中。
3. 在已冻结的 650 条 T1-T3 bundle 上产出 T2 ASR/LID、T3 native-label closed-world 指标。官方/可验证 release 或许可不足的源只能留在本地
   integrity track，不自动进入可再分发 benchmark。
4. 新结果必须与本兼容基线分开命名，不能覆盖历史产物后继续沿用同一版本号。
