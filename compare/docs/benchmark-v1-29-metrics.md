# Benchmark v1.0 二十九指标评估方法

更新时间：2026-09-19  
数据版本：`structural-benchmark-subset-v1.0`（650 条，T1–T3）  
结构评分规范：`benchmark-scoring-v1.0`  
描述性评分：DeepSeek 0–2 连续分，再归一化到 $\left[0,1\right]$  
实现：`eval/score_benchmark_v1.py`、`eval/run_descriptive_semantic_scoring_batch.py`

本文对应报告 [`docs/report/pdf/benchmark-v1-current.pdf`](report/pdf/benchmark-v1-current.pdf) 中的 **29 个可比评分轴**：结构 19 轴 + 描述 10 轴。每个轴都给出含义、评测范围、GT 构造、单样本公式和聚合公式。报告中的百分数为得分率 $\times 100\%$。

---

## 1. 评测对象

| Track | 评测组 $\mathcal{G}$ | 数据 | 条数 | 主要可评标签 |
| --- | --- | --- | ---: | --- |
| T1 Natural meeting | `T1_natural_meeting_utterance` | AMI | 350 | 说话人、重叠、ASR、语言、词法、非语音比例、性别 |
| T2 Language / ASR | `T2_multilingual_fleurs` | FLEURS，10 语种各 10 条 | 100 | ASR、语言、词法 |
| T2 Language / ASR | `T2_clean_librispeech` | LibriSpeech | 50 | ASR、语言、词法（英文 clean anchor） |
| T3 Event / noise | `T3_events_noise@DESED` | DESED | 75 | 语音/音乐事件、外部噪声 |
| T3 Event / noise | `T3_events_noise@FSD50K` | FSD50K | 75 | 语音/音乐事件、音乐存在、外部噪声 |
| 合计 | 5 个评测组 |  | **650** | 约 3.2 小时音频 |

T4 controlled render 完全排除。无有效 GT 的字段记为 `score_rate = null`，**不记 0 分，不进入 29 轴**。

29 轴分层如下。

| 层 | 轴数 | 标签 |
| --- | ---: | --- |
| L1 词汇内容 | 6 | `speaker.asr_transcript`，`language_content.language`，`language_content.word_count`，`language_content.repetition.has_repetition`，`language_content.repetition.repetition_count`，`deterministic_derived.filler_ratio` |
| L2 说话人 | 5 | `speaker.speaker_count`，`speaker.speaker_present`，`speaker.multi_speaker`，`speaker.speaker_overlap`，`speaker.gender` |
| L3 时间结构 | 4 | `speaker.speaker_change_count`，`speaker.speaker_change`，`speaker.overlap_ratio`，`deterministic_derived.speaker_change_rate_per_min` |
| L5 声学场景与事件 | 4 | `basic_acoustic.nonspeech_ratio`，`sound_field_scene.speech_music_events`，`sound_field_scene.music_present`，`sound_field_scene.external_noise_type` |
| D 描述性 | 10 | `caption`，`topic`，`background_condition`，`dominant_content_impression`，`clarity_intelligibility_impression`，`background_sound_summary`，`descriptive_phrases`，`facets`，`speaker_descriptions`，`scene_hypotheses` |

---

## 2. 统一计分框架

### 2.1 预测状态

对样本 $i$、字段 $f$，先把预测规范成五种状态之一：

| 状态 $\sigma_{i,f}$ | 含义 | 默认单样本分 $s_{i,f}$ |
| --- | --- | ---: |
| `result_known` | 字段存在且通过 schema 规范化 | 按该字段公式计算，落在 $\left[0,2\right]$ |
| `result_null` | 合法显式弃答 `null` | $1$ |
| `result_missing` | 字段缺失 | $0$ |
| `schema_invalid` | 类型/枚举/范围非法 | $0$ |
| `runtime_failed` | 该样本无预测或运行失败 | $0$ |

除特别说明外：

$$
s_{i,f}=
\begin{cases}
2, & \sigma_{i,f}=\mathrm{known}\ \text{且预测正确}\\
1, & \sigma_{i,f}=\mathrm{null}\\
0, & \text{错误、缺失、非法或失败}
\end{cases}
$$

接近 $50\%$ 的得分**可能来自显式 `null` 的半分基线**，不等于 $50\%$ 识别准确率。

### 2.2 有效样本

只对同时满足以下条件的样本计分：

1. 该字段 `gt_mask = true`；
2. GT 合法；
3. 无输入泄漏。

记字段 $f$ 在评测组 $g$ 中的有效样本集为 $\mathcal{S}_{f,g}$，$N_{f,g}=|\mathcal{S}_{f,g}|$。若 $N_{f,g}=0$，则该组该字段无得分。

### 2.3 单样本到得分率

结构字段的原始分为 $s\in[0,2]$。组内微平均得分率为

$$
r_{f,g}=\frac{\sum_{i\in\mathcal{S}_{f,g}} s_{i,f}}{2N_{f,g}}\in[0,1].
$$

二值、多标签、语言、ASR、词法字段会先做类别/语言平衡，再得到 $r_{f,g}$，见第 3 节。

数值量化使用 round-half-even 到 6 位小数，记为 $\operatorname{q6}(\cdot)$。报告展示时再乘 $100\%$。

### 2.4 标签优先两级宏平均（结构轴）

**禁止**直接对「评测组 $\times$ 标签」格子做一次平均。官方口径是 label-first macro：

1. 同一标签先对所有可评分评测组等权平均

$$
r_f=\frac{1}{|\mathcal{G}_f|}\sum_{g\in\mathcal{G}_f} r_{f,g};
$$

2. 再对该层所有可评分标签等权平均

$$
R_L=\frac{1}{|\mathcal{F}_L|}\sum_{f\in\mathcal{F}_L} r_f.
$$

$\mathcal{G}_f$ 只包含该标签有定义得分的评测组。例如 `speaker.gender` 只有 T1；`music_present` 只有 FSD50K（DESED 全为负例，不产生能力分）。

### 2.5 29 轴总宏平均

描述轴先把 DeepSeek 的 $\left[0,2\right]$ 分除以 2，得到 $r_f\in[0,1]$。然后 19 个结构轴与 10 个描述轴**标签等权**：

$$
R_{\mathrm{total}}=\frac{1}{29}\left(\sum_{f\in\mathcal{F}_{\mathrm{struct}}} r_f+\sum_{f\in\mathcal{F}_{\mathrm{desc}}} r_f\right).
$$

层均、结构 19 轴均、描述 10 轴均同理，都是标签等权，不是样本加权。

---

## 3. 通用评分算子

下面几个算子被多个指标复用。记预测为 $p$、GT 为 $g$。仅当 $\sigma=\mathrm{known}$ 时使用这些公式；$\sigma=\mathrm{null}$ 时 $s=1$，其余 $s=0$。

### 3.1 精确匹配

用于整数、布尔、语言类别：

$$
s^{\mathrm{eq}}(p,g)=
\begin{cases}
2, & p=g\\
0, & p\neq g.
\end{cases}
$$

### 3.2 绝对比例误差（无容差除数）

用于 $p,g\in[0,1]$ 的比例字段：

$$
s^{\mathrm{abs}}(p,g)=2\bigl(1-\lvert p-g\rvert\bigr).
$$

性质：误差 $0$ 得 $2$，误差 $0.5$ 得 $1$，误差 $1$ 得 $0$。

### 3.3 带容差的连续值

容差为 $T>0$：

$$
s^{\mathrm{cont}}(p,g;T)=2\max\left(0,\ 1-\frac{\lvert p-g\rvert}{T}\right).
$$

误差 $0$ 得 $2$，误差 $T/2$ 得 $1$，误差 $\ge T$ 得 $0$。

### 3.4 对称归一化绝对误差

用于计数和速率，避免零值除零，并对多算/少算一视同仁：

$$
s^{\mathrm{nae}}(p,g)=2\left(1-\frac{\lvert p-g\rvert}{\max(p,g,1)}\right).
$$

完全相等得 $2$。若 $p=g=0$，分母取 $1$，仍得 $2$。

### 3.5 正负类平衡（布尔字段）

设正类 $\mathcal{S}^+=\{i:g_i=\mathrm{true}\}$，负类 $\mathcal{S}^-=\{i:g_i=\mathrm{false}\}$，$N^\pm=\lvert\mathcal{S}^\pm\rvert$。

$$
r^+=\frac{\sum_{i\in\mathcal{S}^+} s_i}{2N^+},\qquad
r^-=\frac{\sum_{i\in\mathcal{S}^-} s_i}{2N^-}.
$$

$$
r=
\begin{cases}
\dfrac{r^++r^-}{2}, & N^+>0\ \text{且}\ N^->0\\
r^+, & N^+>0\ \text{且}\ N^-=0\\
\text{undefined}, & N^+=0.
\end{cases}
$$

只有负样本的类别**只统计误报，不产生能力分**。这是 `music_present` 在 DESED 上不进入 29 轴分母的原因。

### 3.6 多标签类别宏平均

固定类别集 $\mathcal{C}$。对每个类别 $c\in\mathcal{C}$，把「$c$ 是否出现」当成布尔问题，按 3.5 得到 $r_c$。令有正例的类别

$$
\mathcal{C}^+=\{c\in\mathcal{C}:N_c^+>0\}.
$$

$$
r=\frac{1}{\lvert\mathcal{C}^+\rvert}\sum_{c\in\mathcal{C}^+} r_c.
$$

无正例类别（本版的 `singing`、`channel_environment`）不进入宏平均。

### 3.7 语言/词法分组宏平均

对依赖 tokenizer 或语言的字段，先按 GT 语言 $\ell$ 分组得到 $r_{f,g,\ell}$，再

$$
r_{f,g}=\frac{1}{\lvert\mathcal{L}_{f,g}\rvert}\sum_{\ell\in\mathcal{L}_{f,g}} r_{f,g,\ell}.
$$

ASR 的分组键是 $(\ell,\ \mathrm{WER\ 或\ CER})$。

---

## 4. 文本规范化与词法规则

L1 多个指标共享同一套冻结规则：`text-normalization-v1`、`text-lexical-rules-v1`。

### 4.1 规范化 $\operatorname{Norm}(\cdot)$

对原始字符串 $x$：

1. Unicode NFKC 规范化并转小写；
2. 所有 Unicode 标点类别 `P*` 的字符替换为空格；
3. 空白折叠为一个空格并去掉首尾空白。

### 4.2 切分单元 $\operatorname{Units}(x,\ell)$

$$
\operatorname{Units}(x,\ell)=
\begin{cases}
\text{去掉空白后的字符序列}, & \ell\in\{\mathrm{zh},\mathrm{ja},\mathrm{ko}\}\\
\operatorname{Norm}(x)\ \text{按空白切分的词序列}, & \text{其他语言}.
\end{cases}
$$

中日韩用字错误率 CER，其余语言用词错误率 WER。

### 4.3 相邻重复次数 $\operatorname{Rep}(\mathbf{t})$

对 token 序列 $\mathbf{t}=(t_1,\ldots,t_m)$，按 n-gram 长度 $n\in\{3,2,1\}$ 依次贪心扫描：若 $\mathbf{t}[i:i+n)=\mathbf{t}[i+n:i+2n)$，则计数加 $1$ 并跳过 $2n$ 个 token，否则 $i\leftarrow i+1$。三个尺度的计数相加：

$$
\operatorname{Rep}(\mathbf{t})=\sum_{n\in\{3,2,1\}} \#\{\text{相邻重复 }n\text{-gram}\}.
$$

### 4.4 英文填充词

仅英语启用。词表

$$
\mathcal{F}_{\mathrm{en}}=\{\texttt{ah},\texttt{er},\texttt{erm},\texttt{hmm},\texttt{mm},\texttt{oh},\texttt{uh},\texttt{um},\texttt{yeah}\}.
$$

$$
\mathrm{filler}(\mathbf{t})=\sum_{t\in\mathbf{t}}\mathbf{1}[t\in\mathcal{F}_{\mathrm{en}}].
$$

其他语言没有冻结 filler 词表，不评 `filler_ratio`。

---

## 5. T1 说话人时间线 GT

L2、L3 以及 `nonspeech_ratio` 的 GT 都从 AMI utterance turn 按冻结规则物化，版本 `speaker-gt-benchmark-v1-seed` / `ami-turn-complement-v1`。

记音频时长为 $D$（来自 WAV header），utterance 集合为 $\{(k,a,b)\}$，其中 $k$ 为说话人 ID，$\left[a,b\right]$ 为时间区间。

### 5.1 清洗与按人合并

1. 裁剪到 $\left[0,D\right]$，丢弃时长 $<0.1\,\mathrm{s}$ 的片段；
2. 同一说话人的区间按时间排序，若相邻间隙 $\le 0.3\,\mathrm{s}$ 则合并；
3. 合并后总时长 $<0.1\,\mathrm{s}$ 的说话人丢弃。

得到每位说话人的不重叠区间集合 $\mathcal{I}_k$。

### 5.2 语音并集与重叠

令全部区间 $\mathcal{I}=\bigcup_k\mathcal{I}_k$。$\operatorname{Union}(\mathcal{I})$ 为区间并集时长。重叠时长只统计**至少两名说话人同时活跃且持续 $\ge 0.1\,\mathrm{s}$** 的部分，记为 $\operatorname{Overlap}(\{\mathcal{I}_k\};0.1)$。

### 5.3 说话人切换

把所有合并区间按起始时间排序为 $(a_j,b_j,k_j)$。若相邻两段说话人不同，且

$$
a_{j}-b_{j-1}\le 1.0\,\mathrm{s},
$$

则在 $a_j$ 记一次切换（同一时刻只计一次）。切换次数为 $C$。

派生量：

$$
\begin{aligned}
K &= \lvert\{k:\mathcal{I}_k\neq\emptyset\}\rvert,\\
U &= \operatorname{Union}(\mathcal{I}),\\
O &= \operatorname{Overlap}(\{\mathcal{I}_k\};0.1),\\
\rho_{\mathrm{ns}} &= \operatorname{clip}_{[0,1]}\bigl(1-U/D\bigr),\\
\rho_{\mathrm{ov}} &= O/U \quad (U>0),\\
\lambda &= \operatorname{q6}(60C/D).
\end{aligned}
$$

---

## 6. L1 词汇内容层（6 轴）

### 6.1 `speaker.asr_transcript`

**含义。** 整段音频的语音转写。

**评测范围。** T1 350 + T2 150，共 500 条有 reference 的样本。三个评测组分别计分后再等权平均。

**GT。** T1 用 AMI transcript；T2 FLEURS 用 `reference_text_raw`；T2 LibriSpeech 用官方 text。

**单样本。** 令 $\mathbf{r}=\operatorname{Units}(g,\ell)$，$\mathbf{h}=\operatorname{Units}(p,\ell)$，$d=\operatorname{Lev}(\mathbf{r},\mathbf{h})$ 为 Levenshtein 编辑距离（插入/删除/替换代价均为 1）。

$$
e=\min\left(\frac{d}{\max(\lvert\mathbf{r}\rvert,1)},1\right),\qquad
s=2(1-e).
$$

$\ell\in\{\mathrm{zh},\mathrm{ja},\mathrm{ko}\}$ 用 CER，否则用 WER。显式 `null` 得 $1$。

**聚合。** 在评测组 $g$ 内先按 $(\ell,\ \mathrm{WER/CER})$ 分组做微平均，再对分组等权：

$$
r_{g}=\frac{1}{\lvert\mathcal{H}_g\rvert}\sum_{h\in\mathcal{H}_g}\frac{\sum_{i\in h}s_i}{2\lvert h\rvert},\qquad
r=\frac{r_{\mathrm{T1}}+r_{\mathrm{FLEURS}}+r_{\mathrm{LS}}}{3}.
$$

辅助量（不进入 29 轴）：known/e2e 的 WER 或 CER、规范化后完全匹配率。

---

### 6.2 `language_content.language`

**含义。** 音频语言，规范为

$$
\mathcal{L}=\{\mathrm{ar},\mathrm{zh},\mathrm{en},\mathrm{es},\mathrm{fr},\mathrm{hi},\mathrm{ja},\mathrm{ko},\mathrm{ru},\mathrm{sw}\}.
$$

别名映射包括 `en_us\to en`、`cmn_hans_cn\to zh`、`ja_jp\to ja` 等；不在集合中的预测视为非法。

**评测范围。** T1（语料级英语）、T2 FLEURS（clip/locale GT）、T2 LibriSpeech（英语）。三种 GT 语义不同，**先分组再宏平均，不直接合并样本**。

**单样本。**

$$
s=
\begin{cases}
2, & \hat{\ell}=\ell\\
1, & \sigma=\mathrm{null}\\
0, & \text{其他}.
\end{cases}
$$

**聚合。** 组内先按 GT 语言做精确匹配微平均，再语言等权，最后三个评测组等权。T1 只有英语，语言宏平均退化为英语准确率；FLEURS 对 10 语种等权。

---

### 6.3 `language_content.word_count`

**含义。** 按 4.2 的 tokenizer 统计的非标点词数（CJK 为非空白字符数）。

**评测范围。** T1/T2 中 tokenizer 与 reference 可用的样本。GT 由 reference transcript 物化：$g=\lvert\operatorname{Units}(x,\ell)\rvert$。

**单样本。** $p,g$ 为非负整数。

$$
s=s^{\mathrm{nae}}(p,g)=2\left(1-\frac{\lvert p-g\rvert}{\max(p,g,1)}\right).
$$

**聚合。** 组内按语言宏平均，再跨 T1 / FLEURS / LibriSpeech 等权。另报 exact accuracy、MAE、归一化绝对误差，不进入 29 轴。

---

### 6.4 `language_content.repetition.has_repetition`

**含义。** 是否存在第 4.3 节定义的相邻重复 token n-gram。

**评测范围。** T1/T2。GT：$g=\mathbf{1}[\operatorname{Rep}(\mathbf{t})>0]$。

**单样本。** 布尔精确匹配，`null` 得 $1$。

**聚合。** 组内按语言做正负类平衡，再语言宏平均，再跨评测组等权。与 `repetition_count` **独立计分**。

---

### 6.5 `language_content.repetition.repetition_count`

**含义。** 冻结 n-gram 规则统计出的相邻重复次数，非负整数。

**评测范围。** T1/T2。GT：$g=\operatorname{Rep}(\mathbf{t})$。

**单样本。**

$$
s=
\begin{cases}
2, & p=g\\
1, & \sigma=\mathrm{null}\\
0, & \text{其他}.
\end{cases}
$$

**聚合。** 组内按语言微平均后语言等权，再跨评测组等权。注意这里是**精确整数匹配**，不用 NAE。

---

### 6.6 `deterministic_derived.filler_ratio`

**含义。** 填充词数占词数的比例，范围 $\left[0,1\right]$。

**评测范围。** 仅英语且 `word_count>0`：T1 全部英语；T2 LibriSpeech；T2 FLEURS 中的英语子集。其他语言无冻结 filler 词表。

**GT。**

$$
g=\operatorname{q6}\left(\frac{\mathrm{filler}(\mathbf{t})}{\lvert\mathbf{t}\rvert}\right),\qquad \lvert\mathbf{t}\rvert>0.
$$

**单样本。** 连续值评分，容差 $T=0.05$：

$$
s=2\max\left(0,\ 1-\frac{\lvert p-g\rvert}{0.05}\right).
$$

误差 $0$ 得 $2$，误差 $0.025$ 得 $1$，误差 $\ge 0.05$ 得 $0$。`null` 得 $1$。

**聚合。** 词法语言分组后跨评测组等权。原始 `language_content.filler` 计数只作诊断，不进入 29 轴。

---

## 7. L2 说话人层（5 轴）

除 `speaker.gender` 外，本层均只评 T1 350 条。`speaker.*` 只统计当前音频中作为主要语音、可区分并参与当前交互的说话人；电视/广播等背景人声不计入。

### 7.1 `speaker.speaker_count`

**含义。** 片段内不同说话人数 $K$。

**GT。** 第 5 节合并后的说话人个数。

**单样本。** 非负整数精确匹配：

$$
s=
\begin{cases}
2, & p=K\\
1, & \sigma=\mathrm{null}\\
0, & \text{其他}.
\end{cases}
$$

**聚合。** 样本微平均 $r=\sum s_i/(2N)$。与 `speaker_present`、`multi_speaker` 独立计分；若两者都已知，另做合同检查 $K>0$、$K>1$，合同失败不影响本轴分数。

---

### 7.2 `speaker.speaker_present`

**含义。** 是否存在说话人。

**GT。** $g=\mathbf{1}[K>0]$。

**单样本。** 布尔精确匹配，`null` 得 $1$。

**聚合。** 正负类平衡（3.5）。T1 几乎全为正例，负类很少，平衡后对漏检更敏感。

---

### 7.3 `speaker.multi_speaker`

**含义。** 是否至少有两个不同说话人。

**GT。** $g=\mathbf{1}[K>1]$。

**单样本 / 聚合。** 同 7.2，正负类平衡。与 `speaker_count` 独立；两者都已知时检查 $K>1$。

---

### 7.4 `speaker.speaker_overlap`

**含义。** 是否存在多人同时发言。

**GT。** $g=\mathbf{1}[O\ge 0.1]$，必须使用与 `overlap_ratio` 相同的重叠/collar 规则。

**单样本 / 聚合。** 布尔精确匹配 + 正负类平衡。只评 T1。

---

### 7.5 `speaker.gender`

**含义。** 按首次发言顺序排列的性别序列，元素只能是 `male` 或 `female`。

**评测范围。** 只评 T1 AMI 的有效性别 GT，版本 `ami-speaker-gender-gt-v1`。不得根据音高、音色或语速推断。

**GT。** 来自 `t1-ami/speaker_gender_gt/speaker_gender_gt.jsonl`。说话人必须按首次出现顺序编号为 `speaker_1,\ldots,speaker_K`，得到参考序列

$$
\mathbf{g}=(g_1,\ldots,g_K),\qquad g_j\in\{\mathrm{male},\mathrm{female}\}.
$$

本数据集中 $\mathbf{g}$ 非空。

**预测读取。** 规范表面是 `descriptive.speaker_descriptions[].gender`。若 `speaker.profiles[]` 每项都带 `gender`，则该兼容表面优先。条目按 `speaker_id` 的数字顺序排序。任一性别为 `null` 则整段记为显式弃答；缺字段记缺失；非法 ID/重复 ID/非法性别记 schema 非法。

**单样本。** 对 known 预测 $\mathbf{p}=(p_1,\ldots,p_M)$，计算序列 Levenshtein 距离 $d=\operatorname{Lev}(\mathbf{g},\mathbf{p})$（插入、删除、替换代价 1）。

$$
s=
\begin{cases}
2\max\left(0,\ 1-\dfrac{d}{K}\right), & \sigma=\mathrm{known},\ K>0\\
2\cdot\mathbf{1}[M=0], & \sigma=\mathrm{known},\ K=0\\
1, & \sigma=\mathrm{null}\\
0, & \text{其他}.
\end{cases}
$$

插入过多可使 $d>K$，此时分数被截到 $0$。完全序列匹配得 $2$。

**聚合。** 样本微平均

$$
r=\frac{\sum_i s_i}{2N_{\mathrm{T1}}}.
$$

辅助量：exact sequence accuracy、substitutions/deletions/insertions、$d/K$ 编辑错误率。

---

## 8. L3 时间结构层（4 轴）

全部只评 T1。

### 8.1 `speaker.speaker_change_count`

**含义。** 按第 5.3 节冻结时间线统计的说话人切换次数 $C$。

**单样本。**

$$
s=s^{\mathrm{nae}}(p,C)=2\left(1-\frac{\lvert p-C\rvert}{\max(p,C,1)}\right).
$$

`null` 得 $1$，缺失或非法得 $0$。

**聚合。** 样本微平均。另报 exact accuracy、MAE、归一化绝对误差。

---

### 8.2 `speaker.speaker_change`

**含义。** 音频中是否发生过说话人切换。

**GT。** $g=\mathbf{1}[C>0]$。

**单样本 / 聚合。** 布尔精确匹配 + 正负类平衡。与 count 独立；两者都已知时检查 $C>0$。

---

### 8.3 `speaker.overlap_ratio`

**含义。** 多人重叠发言时长占有效语音并集时长的比例：

$$
g=\operatorname{q6}(O/U)\in[0,1],\qquad U>0.
$$

**单样本。** 不使用容差除数：

$$
s=2\bigl(1-\lvert p-g\rvert\bigr).
$$

误差 $0$ 得 $2$，误差 $0.5$ 得 $1$，误差 $1$ 得 $0$。因 $p,g\in[0,1]$，分数仍在 $\left[0,2\right]$。

**聚合。** 样本微平均。T1 使用冻结 overlap/collar 规则（活跃重叠持续 $\ge 0.1\,\mathrm{s}$）。

---

### 8.4 `deterministic_derived.speaker_change_rate_per_min`

**含义。** 每分钟说话人切换次数。

**GT。** 时长 $D>0$ 时

$$
g=\operatorname{q6}\left(\frac{60C}{D}\right).
$$

**单样本。** 与切换次数相同的对称 NAE：

$$
s=2\left(1-\frac{\lvert p-g\rvert}{\max(p,g,1)}\right).
$$

**聚合。** 样本微平均。另报公式一致性：若系统同时给出 $C$ 与 $D$，检查 $p$ 是否等于 $\operatorname{q6}(60C_{\mathrm{pred}}/D_{\mathrm{pred}})$。合同检查不覆盖能力分。

---

## 9. L5 声学场景与事件层（4 轴）

`audio_quality.snr_db`、`room_acoustic.rt60_sec`、`room_acoustic.c50_db` 在 scorer 中保留连续值规则，但当前无冻结 GT，**不进入本层 4 轴**。

### 9.1 `basic_acoustic.nonspeech_ratio`

**含义。** 未检测到人类语音的时间比例，范围 $\left[0,1\right]$。它不是物理静音比：音乐、环境声、噪声也算 non-speech。官方 GT 名称为 `silence_ratio`。

**评测范围。** 仅 T1 350 条。规则版本 `ami-turn-complement-v1`。

**GT。** AMI 合并 utterance turn 的补集：

$$
g=\operatorname{q6}\bigl(\operatorname{clip}_{[0,1]}(1-U/D)\bigr).
$$

**预测。** 兼容同义字段 `basic_acoustic.silence_ratio`；两者同时存在时以 `nonspeech_ratio` 为准。值必须落在 $\left[0,1\right]$，否则非法。

**单样本。**

$$
s=2\bigl(1-\lvert p-g\rvert\bigr).
$$

**聚合。** 样本微平均。本轮路由变化不影响该轴。

---

### 9.2 `sound_field_scene.speech_music_events`

**含义。** 音频中是否出现 `speech`、`music`、`singing` 的多标签数组。`speech` 可比 `speaker.*` 更宽，背景电视人声也可触发 `speech`，但不自动增加说话人数。

**评测范围。** T3 150 条。DESED 与 FSD50K 分成两个评测组。闭世界口径 `t3-native-label-closed-world-v1.0`：原生正标签经 crosswalk 映射为正例，其余目标类直接记负例。负例是评测假设，**不是人工确认负例**，可能含假负例。

固定类别

$$
\mathcal{C}_{\mathrm{sme}}=\{\mathrm{speech},\mathrm{music},\mathrm{singing}\}.
$$

本子集 `singing` 无正例；DESED 的 `music` 也无正例。

**单样本（逐类）。** 对类别 $c$，$g_c=\mathbf{1}[c\in G]$，$p_c=\mathbf{1}[c\in P]$（known 时）。按 2/1/0 计分后做正负平衡得到 $r_c$。

**聚合。** 组内对 $\mathcal{C}^+$ 宏平均，再对 DESED 与 FSD50K 等权。当前：

- DESED 的 $\mathcal{C}^+\approx\{\mathrm{speech}\}$；
- FSD50K 的 $\mathcal{C}^+\approx\{\mathrm{speech},\mathrm{music}\}$。

数组必须是上述三类的无重复子集，否则 schema 非法。`[]` 表示确认三类都不出现，**不等于** `null`。

---

### 9.3 `sound_field_scene.music_present`

**含义。** 音频中是否存在音乐。

**评测范围。** T3。与事件数组独立计分；两者都已知时另查一致性：`music_present` 应等于事件集是否包含 `music`。

**GT。** 由 closed-world 事件物化。

**单样本 / 聚合。** 布尔精确匹配 + 正负类平衡。

DESED 75 条全部为负例，按 3.5 节 $N^+=0$ 规则 **不产生能力分**。因此该轴只来自 FSD50K（38 正 / 37 负）。

---

### 9.4 `sound_field_scene.external_noise_type`

**含义。** 外部噪声大类多标签数组。固定六类：

$$
\mathcal{C}_{\mathrm{noise}}=\{\mathrm{music},\mathrm{animal},\mathrm{mechanical},\mathrm{nature},\mathrm{formless},\mathrm{channel\_environment}\}.
$$

**评测范围。** T3 150 条，DESED / FSD50K 分开。`channel_environment` 本子集无正例，只统计误报。`music` 在噪声轴上表示音乐类噪声/背景音乐，与 `speech_music_events` 的 `music` 独立计分。

**单样本 / 聚合。** 与 9.2 相同的逐类 2/1/0 + 正负平衡 + 有正例类别宏平均，再跨两个 source 组等权。

---

## 10. D 描述性标签层（10 轴）

描述性字段是开放自然语言，**不走**结构 scorer 的 2/1/0 规则。由 DeepSeek（`deepseek-v4-pro`）逐样本、逐字段打连续分，提示词见 `eval/deepseek_descriptive_semantic_scoring_prompt.txt`。

### 10.1 输入与输出合同

每个样本构造 JSON：`ground_truth` 与 `prediction` 各含下列 10 个字段；GT 额外可带 `uncertainties` 作为辅助上下文，**不评分**。忽略 `confidence`、`evidence_tags`、`usability_notes` 以及说话人 `gender`（性别已在结构轴 7.5 单独评）。

模型必须只输出：

```json
{"scores": {十个字段名: 0–2 的有限数字}}
```

校验：字段集合必须恰好为 10 个；分数有限且 $\in[0,2]$；预测为 `null` 的字段必须恰好为 $1.00$。通过后 round 到 2 位小数。

当前官方聚合只用评分成功样本：650 条中 648 条成功，2 条缺描述性 GT，不记 0 分。

### 10.2 分数锚点

记字段 $f$ 上 DeepSeek 原始分为 $u_{i,f}\in[0,2]$。锚点是连续标尺，不是只能打 0/1/2：

| $u$ | 语义 |
| --- | --- |
| $2.00$ | 中心命题等价，覆盖完整；允许无害措辞/翻译/语序差异 |
| $1.70$–$1.99$ | 非常接近，仅有次要遗漏或无害增补 |
| $1.25$–$1.69$ | 有共同中心，但重要细节缺失、过宽或轻度不支持 |
| $1.01$–$1.24$ | 弱相关，大部分缺失或错位 |
| $1.00$ | **预测为 `null` 的固定中性分**，无论 GT 是否为 null |
| $0.01$–$0.99$ | 中心基本错误，或有实质矛盾，仅残留少量有效信息 |
| $0.00$ | 无关、相反或完全矛盾。仅因措辞不同或细节缺失不得打 0 |

评判顺序：先抓中心命题，再看实质矛盾，最后看次要覆盖。中心正确但次要 GT 细节缺失，通常落在 $1.25$–$1.69$，而不是 $<1$。更宽但仍为真的预测，可以高于更细但引入矛盾的预测。GT 为 `null` 时：预测 `null` 得 $1.00$；非空预测默认中性 $1.00$，仅当与可用 GT/uncertainty 明显不兼容才低于 $1$，且不得给 $2.00$。

GT 是非穷尽语义参考，不是检查清单。预测中多出来、但与 GT 兼容的事实，不因 GT 未写而判为无支持。

### 10.3 字段定义与比较规则

对成功样本 $i=1,\ldots,M$（$M=648$），字段均分与归一化：

$$
\bar u_f=\frac{1}{M}\sum_{i=1}^{M} u_{i,f},\qquad
r_f=\frac{\bar u_f}{2},\qquad
\text{报告百分数}=100r_f.
$$

十个字段等权构成描述层：

$$
R_D=\frac{1}{10}\sum_{f\in\mathcal{F}_{\mathrm{desc}}} r_f.
$$

#### 10.3.1 `caption`

整段音频一至三句综合描述，概括主导内容、背景声、录音听感和说话交互。评估整体摘要是否同义，不要求复述 GT 每一句。

#### 10.3.2 `topic`

主要语义主题短语。允许更宽但仍正确的主题或同义压缩。没有足够语音内容时 GT 可为 `null`，不能仅凭语言、人数或声学条件猜主题。

#### 10.3.3 `background_condition`

背景整体状态，如「背景整体较安静」「持续存在明显噪声」。比较安静/嘈杂等中心命题；相反断言（安静 vs 明显噪声）应显著降分。

#### 10.3.4 `dominant_content_impression`

主导可听内容：语音、歌唱、音乐、噪声、静音或混合。不评话题、情绪或说话风格。语音主导 vs 音乐主导属于中心矛盾。

#### 10.3.5 `clarity_intelligibility_impression`

语音清晰度与可懂度听感。「识别出语言」或「有转写」不等于「清晰易懂」。清楚 vs 难以听清属于中心矛盾。

#### 10.3.6 `background_sound_summary`

背景声自然语言总结，只覆盖已标注或直接可听的声音，不补充地点、方位等未提供细节。与 `background_condition` 独立计分：前者偏具体声源，后者偏整体状态。

以上 6 个标量字段比较时权衡：

1. 主对象/内容是否相同；
2. 交互、噪声、笑声、重叠、音乐、可懂度等重要属性是否一致；
3. 是否遗漏、增添或矛盾于实质事实。

主对象权重大于次要属性。预测中出现 GT 未写但与定性描述兼容的技术细节，视为中性。

#### 10.3.7 `descriptive_phrases`

去重短短语数组，视为**无序语义集合**。将预测短语匹配到最近的 GT 概念，允许释义和翻译。同时看重要概念的召回与预测概念的精度。漏掉次要短语只轻微降分；漏掉主导概念或加入矛盾短语则大幅降分。兼容同义增补不重罚。空数组 `[]` 是「确认没有可报告短语」的实质主张，当 GT 有短语时**不等于** `null`，应按缺失覆盖打到 $<1$。按语义重要性加权，不按条数加权。

#### 10.3.8 `facets`

可扩展描述维度数组。用 `name` 与 `description` 的联合语义角色匹配，**不要求名字字面相同**；忽略 `confidence` 与 `evidence_tags`；顺序无关。部分重要 facet 匹配给部分分。无关、无支持或矛盾的额外 facet 降分；一条预测 facet 合并两条相近 GT facet 且保住中心意思，允许。GT 列表非穷尽，兼容的额外 facet 中性。

#### 10.3.9 `speaker_descriptions`

逐主要说话人的自然语言描述。比较 `speaker_id`、`description`、`descriptive_phrases`；忽略置信度、证据路径和 `gender`。有 ID 时用 ID 对齐，但属性对错优先于格式。相关属性：相对音高、语速、响度、稳定性、笑声、轮转、重叠。漏掉一名说话人或把实质属性安错人，按比例降分；说话人级属性错误重于无害措辞差异。

#### 10.3.10 `scene_hypotheses`

谨慎的场景或交互假设。比较底层可能性，不比较置信度或证据元数据；顺序无关。同一可能性但不确定措辞不同，视为强匹配。更宽但仍兼容的假设给部分到高分；无支持的场所、关系、身份、设备主张降分。不要求复现每一条 GT 假设，但漏掉主假设、尤其是矛盾要罚。可参考 `ground_truth.uncertainties`，避免在 GT 已标明不确定时过度惩罚保守预测。

---

## 11. 二十九轴总览公式

结构轴 $r_f$ 来自第 2.4 节标签优先宏平均；描述轴 $r_f=\bar u_f/2$。

$$
\begin{aligned}
R_{\mathrm{L1}}&=\frac{1}{6}\sum_{f\in\mathcal{F}_{\mathrm{L1}}} r_f,\\
R_{\mathrm{L2}}&=\frac{1}{5}\sum_{f\in\mathcal{F}_{\mathrm{L2}}} r_f,\\
R_{\mathrm{L3}}&=\frac{1}{4}\sum_{f\in\mathcal{F}_{\mathrm{L3}}} r_f,\\
R_{\mathrm{L5}}&=\frac{1}{4}\sum_{f\in\mathcal{F}_{\mathrm{L5}}} r_f,\\
R_{\mathrm{struct}}&=\frac{1}{19}\sum_{f\in\mathcal{F}_{\mathrm{struct}}} r_f,\\
R_D&=\frac{1}{10}\sum_{f\in\mathcal{F}_{\mathrm{desc}}} r_f,\\
R_{\mathrm{total}}&=\frac{1}{29}\sum_{f\in\mathcal{F}_{\mathrm{29}}} r_f.
\end{aligned}
$$

雷达图有两种显示：

- **Absolute**：$100r_f\in[0,100]$；
- **RPS**（relative-to-best）：对同一轴三套系统中的最高分 $r_f^\star$，

$$
\mathrm{RPS}_f=\frac{r_f}{r_f^\star}\in[0,1]
\quad(r_f^\star>0).
$$

---

## 12. 指标一览（公式速查）

| # | 轴 | 层 | 有效样本 | 单样本公式（known） | 组内聚合 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `speaker.asr_transcript` | L1 | T1+T2 | $2(1-\min(d/\max(\lvert r\rvert,1),1))$ | 语言$\times$WER/CER 宏平均 |
| 2 | `language_content.language` | L1 | T1+T2 | $2\cdot\mathbf{1}[\hat\ell=\ell]$ | 语言宏平均 |
| 3 | `language_content.word_count` | L1 | T1+T2 | $2(1-\lvert p-g\rvert/\max(p,g,1))$ | 语言宏平均 |
| 4 | `repetition.has_repetition` | L1 | T1+T2 | 布尔 2/0 | 语言内正负平衡 |
| 5 | `repetition.repetition_count` | L1 | T1+T2 | 整数精确 2/0 | 语言宏平均 |
| 6 | `filler_ratio` | L1 | 英语且词数$>0$ | $2\max(0,1-\lvert p-g\rvert/0.05)$ | 语言宏平均 |
| 7 | `speaker.speaker_count` | L2 | T1 | 整数精确 2/0 | 样本微平均 |
| 8 | `speaker.speaker_present` | L2 | T1 | 布尔 2/0 | 正负平衡 |
| 9 | `speaker.multi_speaker` | L2 | T1 | 布尔 2/0 | 正负平衡 |
| 10 | `speaker.speaker_overlap` | L2 | T1 | 布尔 2/0 | 正负平衡 |
| 11 | `speaker.gender` | L2 | T1 | $2\max(0,1-d/K)$ | 样本微平均 |
| 12 | `speaker_change_count` | L3 | T1 | 对称 NAE | 样本微平均 |
| 13 | `speaker.speaker_change` | L3 | T1 | 布尔 2/0 | 正负平衡 |
| 14 | `speaker.overlap_ratio` | L3 | T1 | $2(1-\lvert p-g\rvert)$ | 样本微平均 |
| 15 | `speaker_change_rate_per_min` | L3 | T1 且 $D>0$ | 对称 NAE | 样本微平均 |
| 16 | `nonspeech_ratio` | L5 | T1 | $2(1-\lvert p-g\rvert)$ | 样本微平均 |
| 17 | `speech_music_events` | L5 | T3 | 逐类 2/0 | 有正例类宏平均 |
| 18 | `music_present` | L5 | T3 FSD50K | 布尔 2/0 | 正负平衡 |
| 19 | `external_noise_type` | L5 | T3 | 逐类 2/0 | 有正例类宏平均 |
| 20–29 | 10 个描述字段 | D | 648 条成功 | DeepSeek $u\in[0,2]$ | 成功样本均分后 $r=\bar{u}/2$ |

所有 known 公式之外，合法 `null` 一律 $s=1$，再除以满分 $2$ 进入 $r$。

---

## 13. 不进入 29 轴的字段

下列字段仍可能出现在预测 schema 中，但不进入本报告 29 轴。

| 角色 | 字段 | 原因 |
| --- | --- | --- |
| 合同 | `basic_acoustic.duration_sec` / `sample_rate_hz` / `channels` | 与 WAV header 对齐；时长容差 $0.001\,\mathrm{s}$，采样率与声道精确匹配。计入合同检查，不进能力雷达 |
| 诊断 | `language_content.filler` | 只审计原始计数；能力分用 `filler_ratio` |
| 诊断 | `punctuation.punctuation_count`，`has_terminal_punctuation` | 规则保留，不进雷达 |
| 诊断 | `silence_segments`，`snr_db`，DNSMOS 四项，`far_field`，`rt60_sec`，`c50_db`，`noise_composition`，`speaker.profiles` | 无冻结能力 GT 或定义未冻结 |
| 空层 | Layer 4 副语言/非言语 | 无冻结 GT，记 n/a |
| 描述辅助 | `usability_notes`，`uncertainties` | 不评分；后者仅校准描述性宽严 |

跨字段公式一致性（如 `speaker_present` 是否等于 `speaker_count>0`、`filler_ratio` 是否等于 `filler/word_count`）只作为合同审计，不覆盖能力分。预测中出现未知 sample ID、JSONL 无法解析时，该次提交视为 `input_contract_failed`，需修复后重跑。

---

## 14. 可复现来源

| 内容 | 路径 |
| --- | --- |
| 结构评分实现 | `eval/score_benchmark_v1.py` |
| 描述性评分实现 | `eval/run_descriptive_semantic_scoring_batch.py` |
| 描述性评分提示词 | `eval/deepseek_descriptive_semantic_scoring_prompt.txt` |
| 结构评分表（无性别、无描述） | `dataset/benchmark/SCORING_STANDARD.md` |
| 数据构成 | `dataset/benchmark/DATA_COMPOSITION.md` |
| 字段语义 | `docs/tag-schema-concise.md` |
| T3 闭世界映射 | `dataset/benchmark/t3-event-noise/crosswalk.json` |
| T1 性别 GT | `dataset/benchmark/t1-ami/speaker_gender_gt/speaker_gender_gt.jsonl` |
| 报告生成 | `eval/generate_benchmark_pdf.py`，`eval/generate_benchmark_radar.py` |

当前报告分数文件：

- `docs/report/result/sure-tagger/score.json`
- `docs/report/result/qwen-captioner/score.json`
- `docs/report/result/qwen3-omni-v2-multi-api12/structural_score.json`
- `docs/report/result/descriptive/*/summary.json`
