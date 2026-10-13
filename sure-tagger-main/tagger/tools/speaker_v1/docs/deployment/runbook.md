# Sure-Tagger 多说话人 v2 Shadow Runbook

> 适用版本：`speaker_fusion_artifact_v2.0-shadow.1`
>
> 根目录：`/hpc_stor03/sjtu_home/weihan.chen/share/tagger`

## 1. 运行边界

- 输入必须是 `development.md` 约定的 utterance-level JSONL manifest。
- 每条 sample 只读取自己的 `sample.audio.path`，不得拼接相邻 utterance。
- 默认不要传 `--score-native`。该参数只用于受控离线评测，且 native metadata 仍只能在融合完成后读取。
- 当前只运行 `v2-shadow`；它不会写公开 speaker metadata，公开值保持 `null`。
- 默认启用离线 pyannote Community-1 作为第三条诊断 timeline；其 `license_review_status` 默认是 `pending`，即使传入 calibration ID 也不会参与认证。
- 不要在稳定 `.runtime` 中执行 `pip install -U`。升级时创建带新版本名的环境并重新 healthcheck。

## 2. 路径与预检

```bash
TAGGER_ROOT=/hpc_stor03/sjtu_home/weihan.chen/share/tagger
ORCH_PY="$TAGGER_ROOT/.runtime/speaker_orchestrator_py311_v1/bin/python"

test -x "$ORCH_PY"
test -f "$TAGGER_ROOT/ami_en2001a_utterances/manifest.jsonl"
nvidia-smi
```

确认关键资产：

```bash
cd "$TAGGER_ROOT"
sha256sum \
  models/MOSS-Transcribe-Diarize-model/model-00000-of-00001.safetensors \
  models/speech_campplus_sv_zh-cn_16k-common-v1.0.0/campplus_cn_common.bin \
  models/speaker/openai/whisper-base/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt \
  models/speaker/nvidia/diar_streaming_sortformer_4spk-v2/diar_streaming_sortformer_4spk-v2.nemo \
  models/FireRedVAD/pretrained_models/FireRedVAD/VAD/model.pth.tar \
  models/brouhaha/brouhaha-vad/models/best/checkpoints/best.ckpt
```

预期模型 SHA256 依次为：

```text
9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4
3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8
ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e
b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329
63f4fb1b00a6b8607c118dd48efc18d5e40d67d99b7bf9aa7a8d61540cf23d71
9c237e4a7b1de8b456dbee25db853342bf374b19d8732b72b61356519e390ae1
```

MOSS 的 custom code/config/tokenizer 及 helper source 清单位于 `.deploy/moss/model_manifest.json`。当前 adapter 运行时只强制校验主 safetensors；bundle 其余文件已经盘点但尚未在加载时强制校验，因此 G0 仍是 partial pass。

## 3. 单条 Smoke

先选择一个新的输出目录，避免覆盖已有审计 artifact：

```bash
cd "$TAGGER_ROOT"

"$ORCH_PY" scripts/run_speaker_evidence_v2.py \
  --manifest ami_en2001a_utterances/manifest.jsonl \
  --sample-id EN2001a_utterance_00000 \
  --output-dir ami_en2001a_utterances/outputs/<new_run_id> \
  --moss-device cuda:2 \
  --whisper-device cuda:3 \
  --sortformer-device cuda:0 \
  --pyannote-device cpu \
  --campplus-device cpu \
  --fail-fast
```

这是已经验证过的设备组合。GPU 空闲情况变化时可以调整 MOSS/Whisper，但 Sortformer 直接使用 `cuda:1` 曾触发 NeMo illegal memory access，当前稳定配置为逻辑 `cuda:0`。

如果必须让 Sortformer 使用物理 GPU 1，应在独立进程外设置 `CUDA_VISIBLE_DEVICES=1`，并让它在进程内使用逻辑 `cuda:0`。当前统一 CLI 不支持给单个 worker 单独注入 `CUDA_VISIBLE_DEVICES`，不要在同一个多模型进程里强行套这个映射。

只在离线 demo 需要事后对照 AMI reference 时增加：

```text
--score-native
```

## 4. 查看结果

```bash
jq . \
  ami_en2001a_utterances/outputs/<new_run_id>/speaker_v2_shadow_results.jsonl

gzip -cd \
  ami_en2001a_utterances/outputs/<new_run_id>/artifacts/speaker_v2/EN2001a_utterance_00000/fusion_artifact_v2.json.gz \
  | jq '{sample_id, profile, claims, public_adapter, input_provenance}'
```

成功条件：

- sample `status` 为 `ok`，并存在 fusion artifact。
- `scope.level` 为 `utterance`，duration 与当前 WAV 一致。
- `input_provenance.native_metadata_entered_inference=false`。
- evidence 的 `sample_id` 和 `audio_sha256` 全部一致。
- `public_adapter.enabled=false`，三个公开 speaker 值均为 `null`。
- 冲突、supported 或 insufficient 都是合法结果；不能为了得到非空值而降低 resolver 门槛。
- `run_manifest.json`、`alignments/`、`speaker_text/`、`speaker_text_comparisons/`、`hypotheses/`、`certifications/` 和 `compat_metadata.json.gz` 均应存在；pyannote evidence 应记录 `model_asset_verified=true`、`license_review_status=pending`、`counts_for_certification=false`。

## 5. 小批 Shadow

前 10 条基础 smoke：

```bash
cd "$TAGGER_ROOT"

"$ORCH_PY" scripts/run_speaker_evidence_v2.py \
  --manifest ami_en2001a_utterances/manifest.jsonl \
  --max-samples 10 \
  --output-dir ami_en2001a_utterances/outputs/<new_10_sample_run_id> \
  --moss-device cuda:2 \
  --whisper-device cuda:3 \
  --sortformer-device cuda:0 \
  --pyannote-device cpu \
  --campplus-device cpu
```

`--max-samples 10` 只是 manifest 前 10 条，不是分层校准集。正式评估应显式传多个 `--sample-id`，覆盖 1/2/3/4+ speaker、overlap、短 backchannel、静音、噪声和不同时长。

批量运行先保持每个 GPU worker 并发为 1。speaker evidence 的 10 条/195 条验收结果见 evaluation 文档；Brouhaha VAD 对照使用独立 CPU 诊断命令，不能把它的成功状态当作 speaker claims 已验收。

Brouhaha VAD/声学诊断：

```bash
cd "$TAGGER_ROOT"
BROUHAHA_PY="$TAGGER_ROOT/.runtime/fireredvad_rebuild_py310/bin/python"
"$BROUHAHA_PY" scripts/run_brouhaha_silence_comparison.py \
  --manifest ami_en2001a_utterances/manifest.jsonl \
  --output ami_en2001a_utterances/outputs/<brouhaha_run_id>/comparison.jsonl
```

该命令只生成 FireRedVAD/Brouhaha coverage 对照。Brouhaha 的公开 `audio_quality.snr_db` 适配器（C50 为内部 evidence `internal.brouhaha_c50_db`）仍不产生 speaker count、overlap 或 change 票；checkpoint 兼容性 warning 必须保留在日志中。

## 6. 测试

```bash
cd "$TAGGER_ROOT"
"$ORCH_PY" -m unittest discover -s tests
```

2026-08-11 基线：`97/97` 通过。speaker evidence 已完成 10 条分层 smoke 和 195 条 AMI 单会议诊断，但这不等于跨域校准或 production acceptance。既有 Rec-RIR traceback、Brouhaha 兼容性 warning 和 orphan subprocess/file descriptor `ResourceWarning` 不会改变 unittest 退出码，但要保留日志并在生产门禁前修复。

## 7. 常见故障

| 现象 | 处理 |
| --- | --- |
| 某模型 timeout/worker 退出 | 检查对应 evidence 是否为 missing；不要把 missing 当成 `false`，修复后用新输出目录重跑 |
| 模型 hash mismatch | 停止运行，核对 `.deploy/speaker_v2/deployment_manifest_20260813.json`；不得联网静默更新 |
| Sortformer `illegal memory access` | 停止该 worker，回到逻辑 `cuda:0`；需要物理 GPU 1 时使用进程外映射和独立进程 |
| CUDA OOM | 降为单 worker/单模型，检查 GPU 占用；不要让自动重试结果成为新独立票 |
| CAM++ 没有 candidate pair | 合法 abstention；通常是 clean region 不足或只取得同 cluster 比较，不得补造 identity 票 |
| H1/H2 为 `untested` | 检查是否缺少独立、已校准、跨 cluster identity evidence；不要用原冲突 timeline 自证 |
| 结果一直 `conflicted` | 保留 conflict，查看 event-local alignment 和 speaker-text ambiguity；不能退化成多数投票 |
| pyannote 不可用 | 检查 `.deploy/pyannote/healthcheck.json` 和本地 bundle hash；adapter 使用 SoundFile waveform 绕过 TorchCodec，token 不进入运行参数 |
| Brouhaha 兼容性 warning | 当前 CPU smoke 可运行，但 checkpoint 为旧 pyannote/Torch 训练栈；保留 `diagnostic_pass/compatibility_risk`，完成回归前不得升为 production VAD |

## 8. 回退与审计

v0 入口和公开 metadata 未被本次部署修改。回退时停止调用 `scripts/run_speaker_evidence_v2.py` 即可，已有 v2 artifact 保留用于审计，不需要删除模型或环境。

每次运行至少保存：输入 manifest hash、代码 snapshot hash、deployment manifest、模型 hash、环境名、GPU/driver、完整命令、stdout/stderr、results JSONL 和 fusion artifact。只有完成 calibration 并通过 G3/G4 后，才能评审打开 certified-only public adapter。
