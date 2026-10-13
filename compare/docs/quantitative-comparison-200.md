# sure-tagger 与 captioner 定量对比（200 条预览）

更新时间：2026-08-27  
状态：`preliminary`

## 1. 数据与对齐

- sure-tagger 输出：`/hpc_stor03/sjtu_home/weihan.chen/share/sure-tagger-main/outputs/caption_pairs_3000_tags_200.jsonl`
- sure-tagger 结果：200 行，按 `caption_pairs_3000_manifest.jsonl` 的前 200 行顺序解释。
- captioner 结果目录：`/hpc_stor03/sjtu_home/weihan.chen/share/tag_qwen-omni-captioner/tag_extracted_3000`
- 200 个样本中找到 197 个 captioner JSON；缺少 3 个：
  - `0.000_30.000_obj_000af06e-ad87-4fde-a8a8-6c8ccee6f8a3`
  - `0.000_30.000_obj_1186a1a8-80f5-4ade-9501-553fe4a3e64e`
  - `1050.000_1080.000_obj_200f06d1-c89b-4f9f-b323-027bd5e37d5b`

captioner 作为参考侧，不等同于人工真值。所有百分比保留一位小数；字段的
`conditional agreement` 只在双方均有值的样本/实体上计算，`effective agreement`
把 sure-tagger 未覆盖的参考值算作未匹配。

## 2. 结论摘要

1. 说话人数是当前最稳定的直接定量标签：190 条双方均有值，精确一致率 74.2%，平均绝对误差 0.30 人，误差不超过 1 人的比例为 96.8%。
2. `speaker_change` 的布尔一致率为 89.6%，`multi_speaker` 为 85.6%；`speaker_overlap` 的表面一致率为 70.5%，但正例 precision 只有 16.4%，存在明显过报。
3. 说话人画像的 profile 对齐率只有 70.0%。在高置信对齐实体上，语速一致率 67.7%、音高 37.0%、音量 9.2%；音量主要被 sure-tagger 的 `variable` 输出拉低。
4. 归一化后的连续声学指标与 captioner 枚举的 Spearman 相关整体较弱：整体保真度约 0.13–0.15，清晰度约 0.23（DNSMOS SIG），混响约 0.02；噪声等级与 SNR 为负相关（-0.24），方向符合“噪声越大、SNR 越低”，但相关性仍弱。

## 3. Tier A-S：说话人核心标签

### 3.1 说话人数

| 标签 | captioner known | sure-tagger known | joint | 精确一致 | 精确一致率 | MAE | `|error| <= 1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `speaker_count` | 190 | 200 | 190 | 141 | 74.2% | 0.30 | 96.8% |

captioner 的人数分布为 `1/2/3/4/5/6 = 111/65/9/3/1/1`；sure-tagger 为
`0/1/2/3/4 = 12/116/56/14/2`。190 条联合样本的误差分布为：

| sure - captioner | -3 | -2 | -1 | 0 | +1 | +2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 样本数 | 2 | 3 | 26 | 141 | 17 | 1 |

### 3.2 布尔标签

| 标签 | joint | match | conditional / effective | TP | FP | FN | TN | precision | recall | F1 | macro-F1 | balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi_speaker` | 194 | 166 | 85.6% | 60 | 9 | 19 | 106 | 87.0% | 75.9% | 81.1% | 84.7% | 84.1% |
| `speaker_change` | 193 | 173 | 89.6% | 63 | 7 | 13 | 110 | 90.0% | 82.9% | 86.3% | 89.0% | 88.5% |
| `speaker_overlap` | 193 | 136 | 70.5% | 11 | 56 | 1 | 125 | 16.4% | 91.7% | 27.8% | 54.6% | 80.4% |

`speaker_overlap` 的高 recall 不能解读为整体表现好：captioner 仅有 12 个正例，而
sure-tagger 报告了 69 个正例，其中 56 个是 captioner-only 的反向分歧。

## 4. Tier A-P：说话人画像标签

两边 profile 只按 `speaker_1`、`speaker_2` 的首次出现顺序配对，不根据待比较的
音高/语速/音量寻找最相似 profile。200 条中：captioner profile 结果 197 条，
其中空数组 7 条、非空 190 条；sure-tagger profile 结果 200 条，其中空数组 13 条、
非空 187 条。满足高置信条件的样本为 140/200（70.0%），涉及 185 个 profile
实体。宽松前缀配对共 252 个实体，另有 captioner-only 39 个、sure-only 25 个，
这些未对齐实体不应从 effective 指标中静默删除。

| 标签 | captioner known 实体 | sure known 实体 | 高置信 joint | 高置信 match | 实体 conditional | 高置信实体 effective | 全部参考实体 effective | 样本 macro effective（全部样本） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `speed` ↔ `speech_rate.band` | 287 | 215 | 155 | 105 | 67.7% | 57.1% | 36.6% | 47.6% |
| `pitch` | 288 | 252 | 173 | 64 | 37.0% | 34.8% | 22.2% | 27.5% |
| `speaker_volume` | 290 | 252 | 174 | 16 | 9.2% | 8.6% | 5.5% | 5.4% |

高置信实体 effective 的分母是高置信样本内的 captioner known profile；“全部参考实体
effective”额外把非高置信样本和未对齐 profile 算作未匹配。样本 macro effective 对每个
有 captioner profile 属性的样本等权，避免多说话人样本支配结果。

在共同枚举值中，sure-tagger 的音量有 198 个 `variable` 与 captioner 的 `normal`
配对，解释了音量一致率很低。语速主要分歧是 captioner `normal`、sure-tagger
`fast`（46 个）；音高主要分歧是 captioner `mid`、sure-tagger `low`（123 个）。

## 5. Tier A2：语音/音乐正例

该部分以 captioner 正标签为主参考；“未提及”不自动当作明确否定。下表只在 197
条双方均有 captioner 结果的样本上计算，闭世界准确率仅作 secondary 指标。

| 标签 | joint | captioner 正例 | sure 正例 | TP | FP | FN | TN | captioner 正例 recall | 闭世界准确率 | 闭世界 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `music_present` | 197 | 9 | 48 | 8 | 40 | 1 | 148 | 88.9% | 79.2% | 28.1% |
| `speech_present` | 197 | 192 | 189 | 188 | 1 | 4 | 4 | 97.9% | 97.5% | 98.7% |

音乐的 40 个 FP 说明 sure-tagger 的音乐检出比 captioner 的正标签更宽，不能只用
闭世界准确率下结论。

## 6. Tier B：归一化连续声学指标

这里把 captioner 枚举映射为有序等级，再与 sure-tagger 连续值计算 Spearman 相关；
没有在这 200 条上拟合阈值，也没有报告 weighted Kappa。映射为：

- 混响：`none < slight < moderate < strong` ↔ `rt60_sec`。
- 保真度：`low < moderate < high` ↔ `dnsmos_ovrl`、`dnsmos_p808`。
- 清晰度：`沉闷/模糊 < 清晰 < 极清晰` ↔ `dnsmos_sig`、`c50_db`。
- 噪声：`none/absent < faint < low < moderate < dominant/high`，与 SNR、DNSMOS BAK 比较时注意方向相反。

| captioner 标签 | sure-tagger 数值 | 可计算样本 | Spearman ρ | 说明 |
| --- | --- | ---: | ---: | --- |
| `reverberation_level` | `rt60_sec` | 196 | 0.019 | 基本无单调对应关系；不能由 RT60 直接反推 captioner 房间等级。 |
| `overall_fidelity` | `dnsmos_ovrl` | 196 | 0.146 | 弱正相关。 |
| `overall_fidelity` | `dnsmos_p808` | 197 | 0.135 | 弱正相关。 |
| `clarity_intelligibility` | `dnsmos_sig` | 194 | 0.234 | 弱正相关；captioner 的“清晰”占 173/194。 |
| `clarity_intelligibility` | `c50_db` | 194 | -0.061 | 基本无相关；不把 C50 当作 captioner 清晰度等价标签。 |
| `noise_level` | `snr_db` | 197 | -0.238 | 弱负相关，方向与噪声增大导致 SNR 降低一致。 |
| `noise_floor` | `dnsmos_bak` | 197 | -0.298 | 弱负相关；DNSMOS BAK 越高通常表示背景质量越好。 |

这些 ρ 只用于当前 200 条预览的诊断，不能替代完整 3000 条上的 calibration/evaluation
流程。特别是 `room_acoustic` 的房型、表面材质、echo level 不在此表中，因为它们与
RT60/C50 不是同义标签。

## 7. 限制与下一步

- 当前是前 200 条预览，且 captioner 缺少 3 条；不能作为 3000 条 headline 结论。
- sure-tagger 的 tags-only JSONL 没有 `sample_id`，本报告依赖 manifest 行序；重新运行时应保存不可变 manifest 并执行行数相等断言。
- `null` 按 unknown 处理；`[]`、`false`、`0` 是有效结果，不当作缺失。
- profile 的全局 effective 指标已将未对齐 profile 计为未匹配；高置信指标和宽松前缀诊断不能混为一个总分。
- 正式报告应先补齐 captioner 缺失样本，再在全部 3000 条上固定映射，按 20% calibration / 80% evaluation 学习连续值阈值，并分别报告 Tier A-S、Tier A-P、Tier A2 与 Tier B，不合成单一总分。
