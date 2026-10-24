# DEMAND 环境背景噪声库

当前背景噪声库位于工作区的 `background_noise_library/`，使用 DEMAND（Diverse
Environments Multichannel Acoustic Noise Database）真实环境录音，不再使用此前的程序
合成 office/airport/station/train/cafe/street WAV。

## 当前数据源

- Hugging Face：`SPARCO-project/benchmark_DEMAND_noise`
- 固定 revision：`9494656cfa837f881a718db97111c35d331a6e5b`
- 原始数据 DOI：`10.5281/zenodo.1227121`
- 格式：16 kHz、单声道、PCM-16 WAV
- 覆盖：17 个 DEMAND 环境，每个环境 5 条，共 85 条
- 每条 12 秒，由同一原始录音中 3 个连续的 4 秒片段直接拼接
- 总时长：1020 秒（17 分钟）

没有对噪声做降噪、增强或响度归一化。向语音混合时，pipeline 会根据目标 `snr_db`
缩放完整的场景级噪声轨。

用户描述的“6800 个 12 秒 Opus、16k/48k split、约 916 MB”镜像目前无法从 Hugging
Face 公共数据集目录核实。当前选择的镜像明确来自 DEMAND、覆盖全部 17 个环境，而且可
以文件粒度下载，不需要拉取 10 GB 的完整多通道 Parquet 镜像。该差异也记录在库内
`manifest.json`。

## 环境覆盖

| 类别 | DEMAND 环境 |
|---|---|
| 家庭 | `DKITCHEN`、`DLIVING`、`DWASHING` |
| 自然 | `NFIELD`、`NPARK`、`NRIVER` |
| 办公 | `OHALLWAY`、`OMEETING`、`OOFFICE` |
| 公共场所 | `PCAFETER`、`PRESTO`、`PSTATION`、`SPSQUARE` |
| 交通 | `STRAFFIC`、`TBUS`、`TCAR`、`TMETRO` |

DEMAND 没有独立的机场类别。机场大厅评测可优先使用 `PSTATION`、`PCAFETER` 或
`SPSQUARE` 作为公共空间近似，但清单中不会将它们错误标注成 airport。

## 目录结构

```text
background_noise_library/
├── README.md
├── manifest.json
├── selected_segments.csv
└── demand_16k/<ENV>/<ENV>_ch01_clip00.wav
```

`manifest.json` 保存每条成品的环境、类别、RMS、峰值、源时间区间、源文件列表和
SHA-256。`selected_segments.csv` 保存下载的全部 4 秒源片段元数据。

## 重新下载

下载脚本不使用 `git clone` 或 `snapshot_download`，只请求清单中选定的文件。默认使用
国内镜像端点，并在全部下载和校验通过后原子替换旧库：

```bash
../rednote-hilab__dots.tts-base/.venv/bin/python \
  scripts/prepare_background_noise.py --replace
```

调整每个环境的样本数量：

```bash
../rednote-hilab__dots.tts-base/.venv/bin/python \
  scripts/prepare_background_noise.py \
  --clips-per-environment 8 \
  --replace
```

## 在场景中使用

```yaml
snr_db: 12.0
noise_path: ../../background_noise_library/demand_16k/OOFFICE/OOFFICE_ch01_clip00.wav
```

一条 session 只能使用一个 `noise_path`；pipeline 会在所有说话人完成 RIR 混合后统一
加入这一条场景噪声。噪声短于 session 时会循环，长于 session 时会截取。

## 许可证注意事项

镜像说明提示 DEMAND 的许可证表述存在差异：Zenodo 元数据可能显示 CC BY 4.0，其他
资料也可能写作 CC BY-SA 3.0。重新分发噪声或发布衍生混音前，应以 DEMAND 官方页面的
当前条款为准。库内保留了来源、DOI、固定 revision 和这一提示。
