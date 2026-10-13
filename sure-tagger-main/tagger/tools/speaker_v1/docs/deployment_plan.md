# Sure-Tagger 多说话人 v2 部署计划

> 状态：v2 engineering shadow 已部署；生产发布与校准尚未完成
>
> 基线盘点 / 首次 shadow 部署日期：2026-08-11
>
> 共同维护根目录：`/hpc_stor03/sjtu_home/weihan.chen/share/tagger`
>
> 设计依据：[证据融合详细设计](evidence_fusion_design.md)、[汇报版 Pipeline](speaker_evidence_pipeline_brief.md)、[Speaker Metadata 标准](../speaker_metadata_standard.md)
>
> 部署实录：[部署报告](deployment/deployment_report_20260811.md)、[运行手册](deployment/runbook.md)

模型评测状态与 blocked 候选见：[候选状态登记](evaluation/candidate_status_20260812.md)、[AMI speaker 诊断](evaluation/ami_single_meeting_diagnostic_report_20260812.md)、[Brouhaha VAD 诊断](evaluation/brouhaha_vad_diagnostic_report_20260812.md)。

## 0. 实施进度（2026-08-11）

本文件保留原始目标架构和生产门禁。当前已经完成的是一条可运行、可审计的 utterance-level `v2-shadow`，不是 calibrated production：

- 已部署独立 orchestrator、MOSS、CAM++、Whisper Base、Streaming Sortformer 和复用的 FireRedVAD 环境/模型。
- 已实现 typed evidence、依赖闭包、双 timeline 对齐、lexical clock、count/三个 bool resolver、确定性 H1/H2/H_other、原子 artifact 和 shadow public adapter。
- 已在 `EN2001a_utterance_00000.wav` 上完成五路真实推理；完整回归为 `97/97` 通过。
- pyannote Community-1 已以本地离线 bundle 接入；当前用 Sortformer 和 Community-1 作为独立 timeline。Community-1 license review 仍 pending，因此只提供 support/diagnostic evidence。
- speaker evidence 的 10 条分层 smoke 与 195 条 AMI 单会议诊断已经完成；仍未完成跨域校准、joint-negative calibration 和生产门禁，因此公开 speaker 字段仍全部输出 `null`。
- speaker count 已作为内部 claim 输出 observed/supported/certified/exact 分层状态，但尚未认证 exact count，也未写入公开 schema。

## 1. 结论与部署策略

采用 side-by-side 部署，不直接重构或覆盖当前 speaker v0 路线：

1. 保留现有 `tagger/tools/speaker/`、`tagger/pipelines/tagging.py` 和三个公开 bool 的行为，作为可回滚基线。
2. 新建 `tagger/tools/speaker_v2/`、`tagger/pipelines/speaker_evidence.py` 和 `scripts/run_speaker_evidence_v2.py`，先只生成内部 evidence、hypothesis、certification 和兼容 metadata artifact。
3. 所有模型通过现有 subprocess worker 思路隔离运行。依赖不兼容时新建环境，禁止为了减少环境数量而混装 Torch/CUDA/Transformers 栈。
4. 现有 `.runtime` 只作为盘点参考。它包含旧路径和错误命名，不能直接宣称为共同维护的稳定部署环境；新环境必须在共享 `.runtime` 中从共享 `models` 重建。
5. 模型必须放在共享 `models`，固定 revision、SHA256、许可和加载代码；运行期默认离线，禁止从 Hub 漂移更新。
6. v2 首先以 `shadow` profile 运行，不写公开标签。完成校准后，只有 `certified` claim 才允许映射到公开值，其余一律为 `null`。
7. speaker count 作为独立 claim 部署：内部保存人数下界、上界和 exact 状态。当前公开 schema 仍只有三个 bool；是否新增 `speaker.speaker_count: int|null` 必须先完成接口变更评审。

## 2. 不可改变的输入与输出边界

### 2.1 输入

- 输入仍是 `development.md` 定义的 utterance-level raw-only sample。
- 每个工具只读取当前 `sample.audio.path` 指向的完整文件。
- 所有时间是当前 sample 的 `[0, duration_sec]` 相对时间。
- 禁止读取相邻 utterance、拼接同一 recording 的其它音频，或跨 sample 复用 speaker ID、embedding、text track 和上下文。
- `sample.text.transcript` 为空不能推导“无 speech”或三个 speaker 标签为 `false`；v2 的声学推理必须独立运行。
- native speaker annotation 只能注册为 `G_oracle_native`，用于 calibration/audit，不能进入 production resolver。

### 2.2 输出

当前公开输出保持：

```json
{
  "speaker": {
    "multi_speaker": null,
    "speaker_change": null,
    "speaker_overlap": null
  }
}
```

v2 的完整状态只写内部 artifact，包括：

- `speaker_count_lower_bound`
- `speaker_count_upper_bound`
- `speaker_count_exact`
- `certified | supported | conflicted | insufficient`
- evidence、dependency closure、hypothesis、falsifier、coverage 和 calibration profile

如果决定公开人数，新增字段建议为 `speaker.speaker_count: int|null`。上线前必须同步修改 `development.md`、output schema、registry、pipeline、auditor、测试和消费端；不能直接暴露当前 `recording_summary.speaker_count`。

## 3. 当前部署基线

### 3.1 机器与目录

| 项目 | 当前状态 | 部署影响 |
| --- | --- | --- |
| GPU | 4 x RTX 2080 Ti，每张 11 GB；driver 570.144 | 适合按模型分进程、逐模型常驻；不假设 BF16 或 FlashAttention 2 可用 |
| 系统 Python | 3.6.8 | 不得作为 v2 或环境构建解释器 |
| 文件系统 | 约 979 GB 可用 | 当前模型约 2.29 GB；空间不是短期瓶颈 |
| 环境目录 | `.runtime/` | 新环境只能放这里，使用准确且版本化的名称 |
| 模型目录 | `models/` | 新权重、源码和许可快照只能放这里 |
| 代码版本控制 | 当前根目录不是 Git working tree | 不能依赖 Git 回滚；代码部署前必须建立 release snapshot 或接入上游 Git |
| 权限 | 多数根目录为 `777`，speaker docs 部分为 `755` | 需统一共同维护 ACL；禁止继续依赖全员可写 |

GPU 空闲量只是盘点时快照，不能写死为长期分配。当前 `local_config.py` 中 `MOSS_DIARIZE_DEVICE="cuda:1"` 只可视为本机旧默认值；v2 必须由每次 run 的资源配置指定实际设备。

### 3.2 当前代码能力

已经存在并可复用：

- `scripts/run_tagger.py` 和 sample-level manifest 驱动。
- `tagger/tools/subprocess_runner.py`、`subprocess_worker.py` 的常驻 JSONL subprocess 模型隔离机制。
- MOSS 本地适配器、多通道 mixdown/channel purity、channel activity、native metadata 归一化。
- speaker v0 metadata、artifact 写入和三个公开 bool 派生。
- speaker 相关单测以及 `--sample-id`、`--only-tags`、补标入口。

本次已实现：

- MOSS、Sortformer、CAM++、Whisper、FireRedVAD 和离线 pyannote Community-1 adapter；Brouhaha VAD/SNR/C50 已完成独立 CPU smoke 和 AMI 诊断，但仅作为声学辅助证据；Silero/Paraformer 未纳入本次部署。
- evidence lineage/dependency closure、sample timebase、timeline comparison、speaker/lexical projection。
- speaker-text track、四状态 resolver、人数 observed/support/certified/exact 分层、H1/H2/H_other 确定性模板。
- `fusion_artifact_v2`、逐 evidence artifact、模型 hash 校验、worker timeout/故障移除和 shadow adapter。

仍未实现或未完成：

- 跨域 calibration registry、joint-negative calibration 和 production certification profile。
- speaker claims 的跨域校准集评测仍未完成；Brouhaha 的 VAD 对照已完成，但正式 VAD/SNR/C50 榜仍待多 session gold。
- 公开 `speaker_count` schema 及任何 v2 production 写回。

部署后完整回归为 `97 tests OK`。运行中仍出现既有 Brouhaha 训练栈警告、Rec-RIR CPU/CUDA traceback 和 subprocess `ResourceWarning`；这些没有造成测试失败，但仍是生产化前要关闭的运行卫生问题，不能把“测试返回 OK”解释为全部模型已经 production healthy。

### 3.3 当前环境

| 环境 | 实际栈 | 处理决定 |
| --- | --- | --- |
| `.runtime/moss_transcribe_diarize_py312` | 实际 Python 3.11.5；Torch 2.8.0+cu128；Transformers 5.14；MOSS 0.1.0 | 名称错误且 editable install 指向非共享旧目录；冻结，不原地修，重建新环境 |
| `.runtime/fireredvad_rebuild_py310` | Python 3.10.20；Torch 2.2.2 CPU；FireRedVAD、pyannote 3.3、SpeechBrain、ModelScope | 可作参考/CPU smoke；不能当作 GPU Community-1 部署环境 |
| `.runtime/recrir_py310_torch271` | Python 3.10.20；Torch 2.7.1+cu126；Transformers 4.44；Mamba 扩展 | 保留给现有 Rec-RIR；不与 MOSS 或 speaker v2 混装 |
| `.runtime/panns_py310` | 继承旧非共享 ReCRiR 环境 | 不可迁移；以后单独重建，但不属于 speaker v2 首要任务 |
| `.runtime/uv_bootstrap_py311` | Python 3.11.5；uv 0.12.2 | 只作为新环境构建工具 |

本次新增并验收的隔离环境：

| 环境 | 实际栈 | 状态 |
| --- | --- | --- |
| `.runtime/speaker_orchestrator_py311_v1` | Python 3.11.5；无 GPU 模型包 | healthcheck passed |
| `.runtime/moss_transcribe_diarize_py311_torch280_cu128_v1` | Python 3.11.5；Torch 2.8.0+cu128；Transformers 5.15.0 | GPU demo passed；无旧 editable path |
| `.runtime/campplus_sv_py311_torch280_cu128_v1` | Python 3.11.5；Torch 2.8.0+cu128；ModelScope 1.39.1 | CPU smoke passed |
| `.runtime/whisper_base_multilingual_py311_torch280_cu128_v1` | Python 3.11.5；Torch 2.8.0+cu128；openai-whisper 20250625 | FP16 GPU healthcheck passed |
| `.runtime/sortformer_nemo253_py311_torch260_cu124_v1` | Python 3.11.5；Torch 2.6.0+cu124；NeMo 2.5.3 | GPU healthcheck passed |

高风险旧路径包括：

- `panns_py310/bin/python` 和部分 CLI/shebang 指向不含 `/share/` 的旧目录。
- MOSS editable finder 指向 `/mnt/cloudstorfs/sjtu_home/weihan.chen/tagger/models/...`，而不是共同维护的 `share/tagger/models/...`。
- 当前没有可靠的 requirements/uv lock/Conda lock，不能精确重建已有环境。

部署原则是“新建、验收、原子发布”，不是在共享环境里就地执行 `pip install -U` 或修 shebang。

### 3.4 当前模型

| 模型 | 状态 | 作用 | 部署判断 |
| --- | --- | --- | --- |
| MOSS-Transcribe-Diarize 0.9B | 已部署 | joint ASR + speaker timeline | shared runtime 与真实 GPU demo passed |
| FireRedVAD | 已复用 | speech coverage 候选 | integrated demo passed；不提供人数、身份或 overlap 票 |
| Brouhaha | 已存在；checkpoint SHA256 `9c237e4a...e390ae1` | VAD/SNR/C50 | 195/195 AMI VAD 诊断成功；只作 coverage/声学诊断，不是 speaker timeline；旧 checkpoint/runtime 兼容性仍需回归 |
| SpeechBrain ECAPA-TDNN | candidate pool | identity embedding baseline | blocked：共享 `models/` 无 VoxCeleb checkpoint，本轮未形成可离线评测 profile |
| PANNs Cnn14 | 已存在 | clip-level sound diagnostic | 不能作为 frame-level speaker event 证据 |
| DNSMOS、Rec-RIR | 已存在 | 音质/混响诊断 | 不参与 speaker 认证 |
| pyannote Community-1 | 已接入 shadow | 独立 powerset timeline 和 exclusive audit view | 本地离线 bundle、CPU/GPU smoke passed；CC-BY-4.0，license review pending，未进入认证 |
| CAM++ | 已部署 | identity guard | CPU smoke/integrated demo passed；AMI 英文域未校准 |
| Silero VAD | 未部署 | 可选 coverage guard | 本次以 FireRedVAD+Brouhaha 对照代替，不阻塞 shadow |
| Whisper base | 已部署 | lexical clock | FP16 GPU healthcheck/integrated demo passed |
| Paraformer-zh | 未部署 | 中文 lexical clock 候选 | 待中文 profile 数据到位后评估 |
| Streaming Sortformer 4spk v2 | 已部署 | 第二条 timeline/conflict witness | CC-BY-4.0；真实 GPU demo passed；最多 4 speaker |

现有 MOSS 主权重 SHA256：

```text
9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4
```

MOSS 本次以固定权重 hash 和共享源码版本标识加载，新环境不是 editable install，healthcheck 未发现旧非共享路径。后续升级仍必须同时冻结权重、Transformers custom code、processor 和 Python package revision；不能让 `trust_remote_code=True` 在运行时选择漂移代码。

## 4. 目标运行拓扑

```text
speaker v2 orchestrator（无模型、轻依赖）
        |
        +-- MOSS subprocess ------------------ existing joint evidence
        +-- pyannote + Silero subprocess ----- timeline + coverage
        +-- CAM++ subprocess ----------------- identity guard
        +-- Paraformer 或 Whisper subprocess - lexical evidence
        +-- Sortformer subprocess ------------ conflict-only witness, Phase 2
        |
        v
evidence store -> alignment -> per-claim resolver
        |                         |
        |                         +-- material conflict
        |                               -> deterministic H1/H2/H_other
        |                               -> frozen prediction/falsifier
        |                               -> targeted independent acquisition
        |                               -> re-enter resolver
        v
fusion/certification artifact -> shadow result -> certified-only public adapter
```

关键约束：

- CAM++ 只回答候选片段是否同一 speaker，不是人数、overlap 或 change 的第三票。
- ASR text 只承担 lexical presence、assignment disagreement、boundary/backchannel/crosstalk guard，不增加 speaker event 票。
- H1/H2/H_other 由 versioned deterministic template 生成；假设引擎只能淘汰解释和触发取证，不能直接 certify。
- 同一模型重跑、同 checkpoint wrapper、共享上游的复合 track 和 separation 后原模型不能伪装成独立证据。

## 5. 环境部署计划

### 5.1 环境拆分

以下表格保留部署前的原始环境拆分方案；实际落盘名称以 3.3 节和 deployment manifest 为准。名称中的实际 Python/Torch/CUDA 版本须准确，不得继续使用占位 `x`：

| 新环境 | 内容 | 原因 |
| --- | --- | --- |
| `speaker_orchestrator_py311_v1` | schema、artifact、alignment、resolver、测试；不装 GPU 模型 | 稳定主进程，避免模型依赖污染 |
| `moss_transcribe_diarize_py311_torch280_cu128_v1` | 从共享 MOSS 源码安装；Torch 2.8 / Transformers 5.x | 替换旧路径环境；MOSS 与其它栈隔离 |
| `speaker_pyannote_py310_torch2x_cu12x_v1` | Community-1、Silero、音频 I/O | 第二 timeline 与 coverage 可共享兼容 Torch；先以兼容性 smoke 为准 |
| `speaker_campplus_py310_torch2x_cu12x_v1` | ModelScope/3D-Speaker CAM++ | 避免 ModelScope 依赖影响 pyannote/FunASR |
| `speaker_funasr_py310_torch2x_cu12x_v1` | Paraformer-zh；Phase 2 可隔离评估 SenseVoice/`fa-zh` | 中文 ASR 生态独立 |
| `speaker_whisper_py311_torch2x_cu12x_v1` | Whisper base 与 word timestamp probe | 只在 multilingual profile 部署 |
| `speaker_sortformer_py310_torch2x_cu12x_v1` | NeMo Streaming Sortformer v2 | Phase 2；NeMo 依赖单独隔离 |

本次实际映射：`speaker_campplus_*` 落为 `campplus_sv_py311_torch280_cu128_v1`，`speaker_whisper_*` 落为 `whisper_base_multilingual_py311_torch280_cu128_v1`，`speaker_sortformer_*` 落为 `sortformer_nemo253_py311_torch260_cu124_v1`，pyannote 落为 `speaker_pyannote4_py311_torch280_cu128_v1`；FunASR 环境未创建。

如果 pyannote 与 Silero 的依赖或 device 策略冲突，拆成 `speaker_pyannote_*` 和 `speaker_vad_*`。如果同一环境 `pip check`、CUDA smoke 或真实 utterance 推理任一失败，不通过强制降级包版本来迁就另一个模型，直接拆环境。

### 5.2 构建与发布规范

使用共享 bootstrap uv 的绝对路径，不调用系统 Python 3.6：

```bash
TAGGER_ROOT=/hpc_stor03/sjtu_home/weihan.chen/share/tagger
RUNTIME_ROOT="$TAGGER_ROOT/.runtime"
UV_BIN="$RUNTIME_ROOT/uv_bootstrap_py311/bin/uv"

"$UV_BIN" --version
```

每个环境必须执行：

1. 在 `.runtime/.build-<env>-<timestamp>/` 创建临时环境。
2. 从 `tagger/tools/speaker/requirements/<env>.in` 和 `<env>.lock` 安装；这些文件随代码 release 版本化。依赖下载优先国内源，失败后停止并由维护者手工放入缓存。
3. 执行 dependency check、import check、CUDA/CPU tensor check 和一条真实 utterance inference。
4. 保存 `python-version.txt`、`packages.lock`、`healthcheck.json` 和构建命令；其中不得包含 token。
5. 检查所有 shebang、`.pth`、editable finder、`direct_url.json` 不包含旧非共享路径。
6. 通过后在安装锁保护下原子改名为稳定环境；发布后改为维护组可读、发布者可写。

共享安装必须串行化，例如使用 `.runtime/.speaker-install.lock`。禁止两个维护者同时修改同一稳定环境。

环境健康检查至少包括：

```bash
rg -l '/sjtu_home/weihan\.chen/tagger/' \
  "$RUNTIME_ROOT"/*/bin \
  "$RUNTIME_ROOT"/*/pyvenv.cfg

"$RUNTIME_ROOT/<env>/bin/python" -c \
'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`rg` 命中旧路径时，该环境不能发布。RTX 2080 Ti 上的每个模型必须用 FP16/FP32 实测，不能因 CUDA runtime 支持而假设 BF16、FlashAttention 2 或某个 fused kernel 可用。

## 6. 模型落盘与治理

### 6.1 路径约定

保留现有模型目录，不移动或覆盖。新 speaker 模型统一放在：

```text
models/speaker/<model_slug>/<revision>/
```

每个 revision 目录至少包含：

```text
MODEL_MANIFEST.json
LICENSE.txt 或 license snapshot
README/model-card snapshot
权重与必要配置
```

`MODEL_MANIFEST.json` 必填：

- upstream model ID、download URL、revision/commit。
- 每个权重和 custom code 文件的 SHA256。
- code license、checkpoint license、gated/attribution 状态。
- 输入采样率、channel、speaker slot 等 applicability。
- 对应 runtime 环境名、adapter version 和 capability list。
- 状态：`present_unverified | smoke_passed | calibrated | production_approved | blocked`。

代码 registry 同时保存 manifest 路径和预期 hash。运行时 hash 不匹配必须失败，不允许静默联网修复。

### 6.2 下载顺序

Phase 0：

1. 先修复 MOSS 的可复现 bundle，不重新下载其它模型。
2. 明确 MOSS 权重 custom code 与源码 package 的匹配 revision。
3. 重建 MOSS 共享环境并完成一条 AMI utterance 的两次离线重复推理。

Phase 1：

1. Silero VAD。
2. CAM++。
3. pyannote Community-1；先由授权账号接受 gated 条件，token 只用于下载，不写文件、日志或 artifact。
4. 中文/中英 profile 下载 Paraformer-zh；multilingual profile 下载 Whisper base。按实际数据 profile 部署，不要求第一天同时启用两套 ASR。

Phase 2：

1. Streaming Sortformer 4spk v2。
2. SenseVoice、Zipformer、ECAPA、`fa-zh`、SepFormer 等只在明确实验需求和许可通过后下载。

不要下载 offline Sortformer v1；其 CC-BY-NC-4.0 不进入默认部署。v2.1 在 NVIDIA Open Model License 完成审批前只能标记为 `blocked/audit_only`。

## 7. 代码部署工作包

### WP0：契约与 feature flag

- 固定 v2 semantics version、material activity、overlap absolute/ratio threshold、change 的 floor-transfer 定义。
- 增加 `speaker_pipeline_version = v0 | v2-shadow | v2`；默认保持 `v0`。
- 定义 exact speaker count、上下界和 `int|null` 兼容 metadata。
- 明确 native oracle 和 production evidence 的 profile 隔离。

验收：v2 关闭时，现有公开输出和 artifact 行为不变。

### WP1：证据 adapter 与 registry

- 为 MOSS、pyannote、Silero、CAM++ 和 selected ASR 实现独立 adapter。
- adapter 只输出 typed raw evidence，不直接输出公开标签。
- 实现 model/capability/lineage registry 和 dependency closure 校验。
- 每条 evidence 保存 sample-relative scope、模型 hash、环境、device、预处理和质量状态。

验收：每个模型对一条真实 utterance 可独立离线运行，schema 校验通过；模型失败只产生 missing evidence，不产生 `false`。

### WP2：坐标、对齐与 artifact

- 实现 audio probe/timebase、speaker alignment、lexical alignment。
- 保留模型原始 timeline；派生 track 不得覆盖 raw evidence。
- 实现 atomic evidence store 和 `fusion_artifact_v2`。
- artifact 主键只用 `sample_id`，targeted crop 回映到 sample-relative time。

验收：重复运行得到相同 evidence key 和确定性对齐；没有跨 sample 引用。

### WP3：人数与三个标签 resolver

- 分别实现 `speaker_count_exact`、`multi_speaker`、`speaker_overlap`、`speaker_change` resolver。
- 输出 `certified | supported | conflicted | insufficient`，不使用简单多数票。
- `false` 必须有两个完整、具备负例能力的来源、coverage 和 joint-negative calibration。
- exact count 同时验证人数下界和上界。

人数规则：

```text
multi_speaker=true   if certified lower_bound >= 2
multi_speaker=false  if certified upper_bound == 1
speaker_count=K      if certified lower_bound == upper_bound == K
```

例如确认至少两人、但无法区分两人或三人时：

```json
{
  "speaker_count_lower_bound": 2,
  "speaker_count_upper_bound": 3,
  "speaker_count_exact": null,
  "multi_speaker": true
}
```

验收：exact count conflicted 时仍可独立认证 `multi_speaker=true`；不得以同一 timeline 的派生字段互相认证。

### WP4：确定性假设检验与定向取证

- Phase 1 固定实现 count mismatch、overlap region mismatch、change/backchannel 三类模板。
- H1/H2/H_other 必须 claim-local、event-local；prediction/falsifier 在取证前冻结。
- validator 必须通过完整 dependency closure 独立性检查。
- 分支淘汰后只能 `remove_conflict/reenter_resolver`，不能直接 certify。

验收：H1/H2 都错时进入 H_other；只剩一个未被反驳分支但认证角色不全时仍输出 `null`。

### WP5：集成与兼容输出

- 新入口先写 `outputs/<run_id>/artifacts/speaker_v2/` 和 shadow report。
- 兼容 metadata 仅从 certified claim 构建；完整 timeline 未认证时拒绝输出伪完整 `segments/speakers`。
- 产品批准后，才把 v2 certified adapter 接入 `scripts/run_tagger.py`。

验收：`v0`、`v2-shadow`、`v2` 三种 profile 可并存；删除 v2 输出目录不会影响 v0 回滚。

## 8. Artifact 与并发策略

目标目录：

```text
outputs/<run_id>/
  run_manifest.json
  tags.jsonl
  artifacts/speaker_v2/<sample_id>/
    evidence/<evidence_id>.json.gz
    audio/<derived_audio_id>.wav
    alignments/<alignment_id>.json.gz
    speaker_text/<track_id>.json.gz
    speaker_text_comparisons/<comparison_id>.json.gz
    hypotheses/<case_id>.json.gz
    certifications/<sample_id>.json.gz
    fusion_artifact_v2.json.gz
    compat_metadata.json.gz
```

当前 `v2-shadow.1` 使用简化结构：每个 sample 写 `evidence/*.json.gz` 与一个 `fusion_artifact_v2.json.gz`；speaker-text、comparison、hypothesis 和 public adapter 状态内嵌在 fusion artifact 中。`run_manifest`、独立 alignment/certification/compat 文件是后续批量与 production 目标，不能把尚未生成的文件记为已部署。

并发要求：

- 文件先写同目录临时文件，`fsync` 后原子 rename。
- evidence cache key 至少包含 audio hash、sample-relative scope、model revision/hash、adapter/config version 和 preprocessing。
- 同一 sample 使用细粒度 lock；失败任务不能留下看似完成的 artifact。
- 模型 stderr、内部 warning 和运行指标进入 run log，不进入 tags-only output。
- transcript/lexical unit 按 PII 策略设置访问和 retention；默认短于声学 evidence 保留期。

## 9. GPU 与执行策略

初期不建立复杂调度器，使用显式 worker/device 分配：

- 一个 GPU worker 同时常驻一个主要模型，先把并发设为 1。
- Silero 优先 CPU；CAM++ 先测 CPU，性能不足再独占 GPU worker。
- MOSS、pyannote、selected ASR 和 Sortformer 不在同一进程加载。
- 当前 engineering shadow 为建立双 timeline 基线，对每条 sample 默认调用 Sortformer；未来 production 成本优化可改为 material conflict 后定向调用。无论何种调度，Sortformer 最多 4 speaker，不能提供超过模型适用范围的人数上界或完整负例票。
- GPU 由运行参数或 run manifest 指定，禁止继续把物理 `cuda:1` 作为团队级默认。
- 每个 adapter smoke 必须记录模型加载显存、单 utterance 峰值、推理耗时和 OOM 恢复行为。

若同一节点并行处理多个 sample，先按模型做批处理/常驻 worker，而不是让每个 sample 重复加载模型。任何自动重试必须保留相同 model/config lineage；重试结果不能算新的独立票。

## 10. 数据、校准与验收

### 10.1 本地可用数据

- `ami_en2001a_utterances/`：195 条 10.5-76.4 秒 utterance-level WAV，可用于真实多人/overlap smoke 和离线评测。
- `phase2_asr_sample/`：40 条、8 个 corpus，可用于输入兼容、噪声和非语音 smoke。
- `phase1_asr_samples/`：小型回归样本。

AMI manifest 中存在人工 speaker/time native metadata。运行 v2 production/shadow inference 时必须屏蔽；只在独立 scoring 阶段读取，避免 oracle leakage。

AliMeeting、AISHELL-4、VoxConverse 等跨域校准集当前未在该共享目录中确认存在。没有这些数据时，只能完成工程 smoke 和 AMI 初步校准，不能宣称中文、多语、远场和多人数切片已 production-approved。

### 10.2 测试层级

1. **Schema/unit**：输入封闭性、timestamp 越界、dependency closure、typed evidence、count lower/upper、四状态 resolver。
2. **Failure injection**：merge/split、同 bool 不同区域、backchannel、串音、局部 crop false、H_other、跨标签循环、预算耗尽。
3. **Adapter smoke**：每个模型至少一条真实 utterance；离线重复两次，输出 schema 和时间范围稳定。
4. **Shadow batch**：先 10 条人工挑选样本，再跑 195 条 AMI；不改变 public tags。
5. **Calibration**：按 language、layout、duration、SNR、speaker count、overlap ratio 分层，产生 model 和 joint-negative profile。
6. **Full regression**：当前 `97/97` 测试通过；orphan subprocess 和未关闭文件句柄 warning 仍是 production 前单独门禁，不能因 unittest 返回成功而忽略。

### 10.3 发布门禁

| Gate | 必须满足 |
| --- | --- |
| G0 资产 | 环境无旧路径；模型 revision/hash/license 完整；离线加载成功 |
| G1 adapter | typed schema、scope、capability 和失败返回通过；无模型把 absence 自动变 false |
| G2 shadow | v0 输出零回归；v2 artifact 可复现；native oracle 未进入 resolver |
| G3 calibration | 每个 certified claim 有冻结的 calibration profile；negative 有 joint profile；slice 报告完整 |
| G4 production | 只发布 certified；abstention、冲突率、成本、错误认证率达到项目冻结阈值；rollback 演练通过 |

认证精度目标和最低 slice 样本量应由项目 owner 在 calibration 前冻结。没有足够 ground truth 时，提高 abstention，而不是临时降低认证门槛。

## 11. 分阶段实施顺序

| 阶段 | 主要任务 | 可交付物 | 退出条件 |
| --- | --- | --- | --- |
| P0 资产治理 | 共同维护权限、代码 snapshot/Git、MOSS bundle、count contract、许可确认 | inventory、hash、lock、semantics manifest | G0；MOSS 从共享路径离线 smoke |
| P1 baseline evidence | orchestrator、MOSS、pyannote、Silero、CAM++、selected ASR adapter | raw evidence artifact、model health report | 单条和 10 条 smoke 通过 |
| P2 fusion shadow | alignment、count/三个标签 resolver、H1/H2/H_other、compat adapter | `v2-shadow` batch、failure tests | G1/G2；v0 零回归 |
| P3 calibration | AMI + 新增中文/多语数据；负例联合校准；Sortformer 条件调用 | calibration registry、risk-coverage、ablation | G3；各 slice 有明确通过/blocked 状态 |
| P4 gated rollout | certified-only adapter、监控、回滚演练 | v2 release manifest、运行手册 | G4；先小批，再扩大 |

依赖顺序不能颠倒：在 MOSS 路径、模型 hash、count 语义和 oracle 隔离未解决前，不开始生产标签融合；在 negative calibration 未完成前，不认证任何 `false`。

## 12. 共同维护与回滚

### 12.1 共同维护

- 当前目录不是 Git working tree。P0 必须选择：接入团队上游 Git，或对每次代码发布生成带 SHA256 的只读 release snapshot。没有其中之一，不部署代码变更。
- `.runtime`、`models` 和 `outputs` 被 `.gitignore` 排除，必须分别维护 inventory/lock/checksum，不能依赖代码仓记录资产状态。
- 统一维护组 ACL 和 default ACL；禁止用 `777` 作为长期协作方案。
- 环境和模型安装使用全局安装锁；稳定版本不原地修改，升级创建新版本目录。
- 每个 run 保存 code revision/release hash、环境名、模型 manifest hash、GPU/driver、profile 和 calibration version。

### 12.2 回滚

- 默认 feature flag 保持 `v0`，v2 首先只有 `v2-shadow`。
- v2 正式接入后，回滚只切换 `speaker_pipeline_version=v0`，不删除环境、模型或 artifact。
- 新旧 semantics、metadata 和 artifact 使用不同 version，不覆盖旧文件。
- 回滚演练必须验证：同一 manifest 使用 v0 得到基线结果；v2 worker 全部停止；tags-only 输出没有 v2 内部字段泄漏。

## 13. 上线前待确认事项

以下决定会改变部署产物，P0 必须明确：

1. 是否新增公开 `speaker.speaker_count: int|null`；如果不新增，只在 v2 artifact 和兼容 metadata 保存 certified exact count。
2. `speaker_change` 是否采用 v2 的 floor transfer 语义；若采用，必须更换 semantics version，不能静默覆盖 v0。
3. pyannote Community-1、Sortformer v2、Paraformer/Whisper 的具体 revision 和许可证/attribution 是否批准。
4. speaker transcript/lexical unit 的访问、加密、retention 和删除策略。
5. certified precision、最低 slice 样本量、最大 abstention 和单 utterance 计算预算。
6. 共享目录的 Git/release snapshot 方案，以及共同维护 ACL 的负责人。

## 14. 首批实施清单

P0 完成顺序：

- [x] 建立带 SHA256 的部署前/后 release snapshot 机制（当前目录仍未接入 Git）。
- [x] 对本次新增代码、文档和 `.deploy` 资产设置维护组可写；稳定环境/模型保持维护组可读执行，避免原地改包或改权重。
- [x] 生成本次 environment/model inventory 与 checksum deployment manifest。
- [x] 重建 `moss_transcribe_diarize_py311_torch280_cu128_v1`，消除旧 editable path。
- [x] 冻结本次使用的 MOSS 权重 hash 与共享源码版本标识。
- [x] v2-shadow 内部人数语义已冻结；公开 `speaker_count` 接口仍待产品/schema 评审。
- [x] v2 `speaker_change` 使用独立 shadow semantics，不覆盖 v0。
- [x] 完成本地 pyannote Community-1 bundle、独立 runtime、模型 hash、CPU/GPU offline smoke；认证仍受 gated license review 和 calibration profile 双门禁限制。
- [x] 创建 `speaker_orchestrator_py311_v1` 和模型专属隔离环境/依赖记录。
- [x] 完成 1 条 AMI offline shadow demo；inference 阶段禁用 native oracle。
- [x] 完成 10 条 AMI 分层 speaker-evidence smoke；结果见 evaluation 报告。
- [x] 完成 195 条 AMI speaker-evidence 诊断与分层报告；结果是单 meeting descriptive，不是正式 leaderboard。
- [x] 完成 10/195 条 Brouhaha VAD 与 FireRedVAD coverage 对照；正式 VAD/SNR/C50 榜仍待多 session gold。

adapter 和 resolver 的 engineering shadow 已完成。完成跨域校准、Community-1 许可证审批和运行卫生清单后，再进入 production rollout；任何模型仅达到 `present_unverified` 或 `diagnostic_measured` 时，不得用于 certification。
