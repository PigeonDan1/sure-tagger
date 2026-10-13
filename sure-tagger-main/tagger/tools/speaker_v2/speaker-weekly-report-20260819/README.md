# Speaker Intelligence 技术汇报

> 历史快照说明：本页面和随附 `report-data.js` 固化的是 2026-08-19 的 shadow 运行。2026-08-20 已移除 certification gate 并启用 public adapter 直接输出；2026-09-04 起，旧 Sortformer/Pyannote/MOSS claim policy 又被 FireRed LID 驱动的中文 MOSS、其他语言 OpenASR 单分支路由取代。历史数据中的 `production_eligible=false`、六字段 `null` 和模型路由均不代表当前代码行为。

本目录是 2026-08-19 Speaker 技术进展的独立静态汇报网站，包含四个主栏目：

1. Speaker v2 六个公开标签的 quality-shadow pipeline
2. 冻结 AMI 1k 数据集上的 SURE 模型测评矩阵
3. Qwen-captioner / Qwen3-8B 派生 pseudo-GT 的 100 条一致性评测
4. AMI EN2001a 五条真实 quality-shadow demo

## 重新生成数据

```bash
cd /hpc_stor03/sjtu_home/weihan.chen/share/tagger/tagger/tools/speaker_v2/speaker-weekly-report-20260819
python3 scripts/build_report_data.py
```

生成器会读取原始 Markdown、JSON/JSONL、gzip fusion artifact 和 WAV header，校验公开六字段、评测分母与 demo public adapter 边界，并复制页面使用的九条真实音频。

## 本地启动

```bash
cd /hpc_stor03/sjtu_home/weihan.chen/share/tagger/tagger/tools/speaker_v2/speaker-weekly-report-20260819
python3 -m http.server 8765 --bind 0.0.0.0
```

如果浏览器与服务运行在同一台机器，打开 `http://127.0.0.1:8765/`。页面不依赖 CDN 或构建工具。

如果服务运行在远端节点，本机浏览器中的 `127.0.0.1` 指向本机，不能直接访问远端服务。可以打开节点可达地址，或先建立 SSH 端口转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 <远端 SSH 主机>
```

保持该 SSH 进程运行，再在本机浏览器打开 `http://127.0.0.1:8765/`。

## 数据边界

- 六字段公开对象始终单独展示；该次 2026-08-19 quality-shadow 运行的 public adapter 六值均为 `null`。
- 该历史快照中的非空 candidate 仅属于 `evaluation_output`，并标记为 `unsafe_for_publication=true`、`production_eligible=false`；当前代码已改为直接发布。
- native metadata、transcript 和 Qwen-derived reference 只用于推理后的评测展示，不进入 speaker resolver。
- 页面中的完整 metadata 是对应 `fusion_artifact_v2.json.gz` 的完整 JSON 对象。
