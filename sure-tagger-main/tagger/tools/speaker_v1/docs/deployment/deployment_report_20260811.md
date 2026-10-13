# Sure-Tagger 多说话人 v2 Shadow 部署报告（历史基线 + 2026-08-13 增量）

> 部署日期：2026-08-11
>
> 根目录：`/hpc_stor03/sjtu_home/weihan.chen/share/tagger`
>
> 状态：engineering shadow 已部署，尚未达到 calibrated production

## 1. 部署结论

已完成一条保持 utterance-level 输入边界的多说话人证据融合链路。它对每条 sample 只读取当前 `sample.audio.path`，不会读取相邻 utterance、外部 recording 上下文或输入 transcript。native speaker metadata 仅可在融合完成后用于评分，不进入任何模型、假设或 resolver。

当前链路能稳定生成内部 evidence、speaker-text track、冲突、H1/H2/H_other 和人数分层状态。它仍以 `v2-shadow` 运行，公开适配器关闭，三个公开字段始终为：

```json
{
  "speaker": {
    "multi_speaker": null,
    "speaker_change": null,
    "speaker_overlap": null
  }
}
```

这次部署证明的是工程链路可运行、可审计和可回退，不代表模型已经完成跨域校准，也不代表 exact speaker count 已认证。

## 2. 已部署组件

| 组件 | 环境 | 模型/版本 | 在融合中的角色 | 状态 |
| --- | --- | --- | --- | --- |
| Orchestrator | `.runtime/speaker_orchestrator_py311_v1` | Python 3.11.5 | evidence、对齐、resolver、artifact | passed |
| MOSS | `.runtime/moss_transcribe_diarize_py311_torch280_cu128_v1` | 0.9B，权重 SHA256 `9a0ceb4a...6c4` | 独立 joint ASR + speaker timeline | passed |
| Sortformer | `.runtime/sortformer_nemo253_py311_torch260_cu124_v1` | NeMo 2.5.3，模型 SHA256 `b371afce...3329` | 第二条独立 speaker timeline | passed |
| Whisper Base | `.runtime/whisper_base_multilingual_py311_torch280_cu128_v1` | openai-whisper 20250625，SHA256 `ed3a0b6b...e34e` | 独立 lexical clock | passed |
| CAM++ | `.runtime/campplus_sv_py311_torch280_cu128_v1` | ModelScope v1.0.0，SHA256 `3388cf5f...ada8` | clean-region identity guard/诊断 | passed |
| FireRedVAD | `.runtime/fireredvad_rebuild_py310` | 本地模型 SHA256 `63f4fb1b...3d71` | speech coverage guard | passed in integrated demo |
| pyannote Community-1 | 2026-08-13 已接入 | gated model，本地离线 bundle | 第三条诊断 timeline | smoke passed；CC-BY-4.0 license review pending，不参与认证 |

模型依赖以 subprocess 隔离，未把 NeMo、Transformers、ModelScope 和 Whisper 混装到 orchestrator。Sortformer 最多支持 4 个 speaker；它不能提供 `>4` 的人数上界或完整负例证据。

MOSS 的 checkpoint custom code/config/tokenizer 与 helper source 已登记在 `.deploy/moss/model_manifest.json`。当前 adapter 运行时仅强制校验主 safetensors，尚未强制校验整个 bundle，且本地 helper source 没有可验证的 upstream commit；这是 G0 保持 partial pass 的原因之一。

## 3. 实现范围

主要代码入口：

- `scripts/run_speaker_evidence_v2.py`
- `tagger/pipelines/speaker_evidence.py`
- `tagger/tools/speaker_v2/`
- `tests/test_speaker_v2.py`

已实现的关键约束：

- typed evidence 与确定性 evidence ID；status、quality、音频 fingerprint 均进入身份计算。
- dependency-group 和完整 lineage closure 校验，拒绝同源包装器冒充独立证据。
- MOSS/Sortformer 两条 timeline 的人数、overlap 和 change 逐 claim 比较。
- Whisper 词级时间投影到两条 speaker timeline；文本只作诊断和边界歧义证据，不增加 speaker 事件票。
- CAM++ crop 排除全部 overlap activity；同一 timeline cluster 内比较只作一致性诊断，不能认证或淘汰人数假设。
- H1/H2/H_other 在定向取证前由确定性模板冻结；假设分支只能移除冲突并重新进入 resolver，不能直接 certify。
- 人数区分 `observed_lower_bound_candidate`、`supported_lower_bound`、certified lower/upper bound 和 exact。
- 同 bool 但事件位置不一致仍记为 conflict；`false` 需要两条具备完整负例能力的独立 timeline 和 joint-negative calibration。
- worker timeout/失败只移除对应证据，不把失败解释成 `false`；artifact 使用同目录临时文件和原子替换。
- 跨 sample、跨音频 evidence 被 resolver 拒绝。

## 4. AMI Demo

输入：

```text
ami_en2001a_utterances/audio/EN2001a_utterance_00000.wav
sample_id: EN2001a_utterance_00000
duration: 18.436 s
audio SHA256: 6328e506fecb9c7f718addcb9d2734b419ad6da19974cec66eb0d226fd20d46f
```

权威输出目录：

```text
ami_en2001a_utterances/outputs/speaker_v2_demo_20260811_final_r3
```

运行时设备：MOSS `cuda:2`，Whisper `cuda:3`，Sortformer `cuda:0`，CAM++/FireRedVAD CPU。Sortformer 在直接指定 `cuda:1` 时出现过 NeMo illegal memory access，因此该组合不作为已验证配置。

### 4.1 交叉验证结果

| Claim/证据 | MOSS | Sortformer/Whisper/CAM++ | 融合结论 |
| --- | --- | --- | --- |
| speaker count | 2 | Sortformer 3 | `conflicted`；observed lower bound 3，supported lower bound 2，exact `null` |
| multi-speaker | true | Sortformer true | `supported`；缺少校准 identity guard，不能公开 |
| overlap | false | Sortformer true，约 1.68 s | `conflicted` |
| change | true | Sortformer true，但事件匹配率 0 | `conflicted`，不是简单 bool 投票 |
| lexical | MOSS 自带 text | Whisper 21 个词级单元 | 对称文本差异诊断 `0.34375` |
| word-speaker assignment | Whisper 投影到 MOSS | 同一 Whisper 投影到 Sortformer | agreement `0.47619`，11 个词出现 ambiguity disagreement |
| identity | - | CAM++ score `0.23967`，decision `different` | 只比较到同一 Sortformer cluster，作为 cluster consistency 告警，不参与认证 |

人数冲突先冻结了三条解释：H1 假设 MOSS 的 2 人正确，H2 假设 Sortformer 的 3 人正确，H_other 表示两边均不完整或存在其它失败模式。因为本条样本没有形成跨 cluster、独立且已校准的 CAM++ partition，H1/H2 保持 `untested`，H_other 保持 `viable`。系统没有退化成“最后听 CAM++”或多数投票。

### 4.2 Native 事后评分

本次 demo 显式启用了 `--score-native`。AMI reference 为 3 人、有 overlap；评分发生在 `fusion_id` 生成之后，artifact 记录 `entered_resolver=false`。Sortformer 与参考一致，MOSS 少估 1 人且漏掉 overlap。

AMI 在 Sortformer 训练数据中，因此这个结果只能用作部署 smoke 和评分链路验证，不能作为 Sortformer 的无偏精度结论。

## 5. 验证结果

完整回归命令：

```bash
.runtime/speaker_orchestrator_py311_v1/bin/python -m unittest discover -s tests
```

结果：`97` 个测试通过，`0` failure，`0` error，`0` skip；unittest 耗时 `99.811s`。

测试日志仍有既有运行卫生问题：3 次 Rec-RIR 在 CPU 路径调用 CUDA 扩展的预期 traceback、18 条 subprocess still running warning、36 条 unclosed TextIOWrapper warning，以及 Brouhaha checkpoint/pyannote/torch 版本告警。它们未造成测试失败，但生产门禁前应单独关闭。

## 6. Artifact 与完整性

| Artifact | SHA256 |
| --- | --- |
| `fusion_artifact_v2.json.gz` | `892665fdf824ee8be1ada2a1c86f84de3355b571c8ca0510f81f2c761361b477` |
| `speaker_v2_shadow_results.jsonl` | `115fe0cd450adfa7af07ab6ec7ab756b0a70dbbd08780011c966f9c9f57da97c` |
| 部署前 snapshot | `e4e39125da4d8dbf4cfa74b2d9a5cac298dd286cdb8943c3849b43a2ccec773f` |
| 部署后 code snapshot | `de7aef47d382ea83002418bea573c931ba3b6f927bbb1eacb78867e47c263675` |

统一的环境、模型、代码、demo 和测试清单位于：

```text
.deploy/speaker_v2/deployment_manifest.json
```

## 7. 当前门禁

| Gate | 状态 | 说明 |
| --- | --- | --- |
| G0 资产 | pass for shadow | 已部署模型均有固定 hash/本地路径；Community-1 bundle 逐项 hash 通过，TorchCodec 解码路径被 adapter 绕过 |
| G1 adapter | pass for deployed set | 五路真实 demo 与 typed schema 通过 |
| G2 shadow | partial pass | speaker evidence 10/195 AMI 诊断已完成，v0 未改；跨域 shadow 与正式 acceptance 仍未完成 |
| G3 calibration | blocked | 无冻结 calibration/joint-negative profile，CAM++ 未做 AMI 英文域校准 |
| G4 production | blocked | G3 未通过，公开 adapter 必须继续关闭 |

## 8. 后续工作

1. 已完成 10 条与 195 条 AMI speaker evidence 诊断；后续应在多 meeting 数据上重复，并检查失败率、GPU 峰值、冲突类型和 artifact 完整性。
2. 补充非 AMI、非训练污染的英文数据，以及中文、多语、噪声、远场和 4 人以上切片。
3. 冻结各 claim calibration 与 joint-negative profile，报告错误认证率和 risk-coverage，不以普通准确率代替 certification 风险。
4. 完成 Community-1 CC-BY-4.0 license review 和跨域 calibration；token 不得写入日志、manifest 或 artifact。
5. 修复全量测试中的 orphan subprocess/file descriptor warning，再评估批量常驻 worker。
6. 评审是否向公开 schema 增加 `speaker_count: int|null`；在评审完成前只保留内部 claim。

## 9. 2026-08-13 完整 shadow 增量

在不改动上述历史记录的前提下，2026-08-13 已完成 v2 的完整工程 shadow 部署：

- 新增 `.runtime/speaker_pyannote4_py311_torch280_cu128_v1`，Python 3.11.5、Torch 2.8.0+cu128、pyannote.audio 4.0.0；`uv pip check` 检查 111 个包通过。
- 固定 Community-1 revision `3533c8cf8e...`，五个 bundle 文件逐项 SHA256 通过；本地 CC-BY-4.0 bundle 已离线加载，license review 仍为 `pending`，因此只提供 support/diagnostic evidence。
- pyannote CPU smoke 和 `cuda:0` smoke 均通过；adapter 用 SoundFile 预加载 waveform，绕过当前 TorchCodec `libnvrtc.so.13` 兼容性告警，并在 runtime 中记录 `torchcodec_bypassed=true`。
- v2 CLI 默认支持并启用 `--pyannote-*` 参数；worker、证据采集、resolver certification gate 和测试均已接入。
- 每条 sample 现在原子写入独立 evidence、alignment、speaker-text、comparison、hypothesis、certification、compat metadata 和 fusion artifact；每个批次写 `run_manifest.json`。
- 六路真实 smoke：`EN2001a_utterance_00045` 和 `EN2001a_utterance_00000` 均 `failure_count=0`；后者包含 1 个真实 CAM++ comparison。两条 smoke 的公开 adapter 均保持三个 `null`。
- v2 回归 `tests.test_speaker_v2` 为 `41/41` 通过；全仓 `unittest discover` 还会触发两个与本次部署无关的历史导入错误（旧 v0 registry 符号和 abandoned v3 包），未修改旧路线。

当前最新 smoke 输出：

```text
ami_en2001a_utterances/outputs/speaker_v2_complete_20260813_smoke
ami_en2001a_utterances/outputs/speaker_v2_complete_20260813_demo
```

评测目录和评测报告未参与本次部署修改。G3/G4 仍 blocked：未冻结跨域 calibration/joint-negative profile，未完成许可证审批和运行卫生清理。
