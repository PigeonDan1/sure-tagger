# multi-data 场景聚类方法调研

更新时间：2026-09-17
状态：`draft`，配套 [`multi-data-scene-classification.md`](multi-data-scene-classification.md) 第 4.1 节
实现：[`cluster_methods.py`](../dataset/multi-data/sampled-100-per-source/scripts/cluster_methods.py)、[`cluster_ensemble.py`](../dataset/multi-data/sampled-100-per-source/scripts/cluster_ensemble.py)、[`run_flexible_clustering.py`](../dataset/multi-data/sampled-100-per-source/scripts/run_flexible_clustering.py)
对比结果：[`report/METHOD_COMPARE.md`](../dataset/multi-data/report/METHOD_COMPARE.md)、[`result/scene-clusters-flexible/summary.md`](../dataset/multi-data/result/scene-clusters-flexible/summary.md)

本文只讨论**在已经冻结的 tags-only 距离 `d_tag` 上如何分区**。不重新定义特征，不把 WER/CER、来源名或参考文本写进距离。目标是：从文献和常用实践中列出适合 mixed-type 相异度矩阵的方法，说明各自假设、对本任务的适配方式，以及本仓库实际落地的范围。

## 1. 任务约束

当前样本约 1000 条、10 个来源。每条样本是 sure-tagger 的混合类型标签：数值（时长、SNR、RT60 等）、布尔、集合（噪声类型、音乐事件）。距离不是欧氏向量上的 `||x-y||`，而是四轴等权平均的 `d_tag`：

```text
d_tag(i,j) = mean(d_content, d_interaction, d_quality, d_spatial)
```

缺失用有上限的 missingness 项处理，数值轴做 median/IQR 稳健标准化。因此候选算法必须满足：

1. **直接吃预计算相异度矩阵**，或能从 `d_tag` 构造对称相似度 / kNN 图。不能假设特征在欧氏空间里可加、可求均值。
2. **不读取 WER/CER。** 错误率是模型属性，不是音频场景。允许在分区生成之后用 η² / SηK 筛选候选，不允许把它当作第五轴、分裂特征或命名依据。
3. **K 有下界和规模约束。** 生产候选要求 `K>=6`，主簇 `n_min≈30`，小簇与未分配样本计入 `J` 的惩罚项。
4. **需要可解释代表点。** 后续 `Q_tag`、split/merge、新样本 assignment 都依赖 medoid。没有原型的方法，事后也要在 `d_tag` 上重算 medoid。
5. **不能 source-driven。** 单一来源占比过高的分区，即使内部更紧，也不能当共享场景。

这些约束直接排除了一批常见算法，也解释了为什么“先把标签拼成向量再 k-means”不是默认方案。

## 2. 选型原则

混合类型数据的教科书路线是 **Gower 相异度 + PAM/层次聚类**（Kaufman & Rousseeuw, *Finding Groups in Data*）。本任务的 `d_tag` 就是带轴聚合的 Gower 变体，所以优先考虑相异度矩阵方法，而不是高斯混合或 Ward。

按输入需求把方法分成四档：

| 档 | 输入 | 对本任务 | 例子 |
| --- | --- | --- | --- |
| A. 原生相异度 | `d_tag` 矩阵 | 首选 | k-medoids、HAC、DBSCAN、AP |
| B. 相似度 / 图 | `S=exp(-d^2)` 或 kNN 图 | 可试 | 谱聚类、Louvain、Leiden |
| C. 嵌入后再划分 | MDS/UMAP 得到欧氏坐标 | 可作敏感性分析，审计成本高 | MDS+k-means、UMAP+HDBSCAN |
| D. 不适配 | 需要欧氏均值、协方差或类别原型重估 | 不作为生产候选 | k-means、GMM、Ward、BIRCH |

档 C 会额外引入嵌入随机性和尺度，medoid 不再是原距离上的最优点，本轮不作为主搜索。档 D 只有在先把 `d_tag` 强行欧氏化之后才能跑，结论不可直接解释为标签场景。

## 3. 方法族

### 3.1 划分法：k-medoids / PAM

**想法。** 选 K 个真实样本当 medoid，其余点归到最近 medoid，迭代替换 medoid 使簇内距离和下降。PAM++ / greedy 初始化用 `d^2` 加权抽样，类似 k-means++。

**为什么适合。** medoid 是原始样本，不需要特征均值；对离群点和缺失稳健；天然输出可审计代表点。生态学、临床混合数据里，Gower+PAM 是默认组合。

**变体。**

- PAM：精确局部搜索，N=1000 可承受。
- CLARA / CLARANS：大样本抽样版。当前 N 不必上。
- k-prototypes / k-modes（Huang）：直接在混合特征上交替更新，但会绕开已经校准的 `d_tag` 轴权重，本轮不采用。

**实现。** `cluster_scenes.kmedoids`，K=6…12，种子 0/1/2。这是生产初选方法。

**风险。** 依赖初始种子和 K；球形簇假设强于密度方法。用多种子和后续 split/merge 缓解。

### 3.2 层次凝聚：average / complete / weighted / single

**想法。** 每点自成一簇，按 Lance–Williams 公式合并，在 K∈[6,12] 处切树。

| linkage | 簇间距离 | 形态 | 预期 |
| --- | --- | --- | --- |
| single | min | 易 chaining | 主簇被细丝粘成一团 |
| complete | max | 偏紧、偏圆 | 可能切碎来源块 |
| average (UPGMA) | 尺寸加权平均 | 折中 | 文档原方案之一 |
| weighted (WPGMA) | 等权平均 | 对大小不敏感 | 与 average 接近 |
| Ward | 合并后方差增量 | 要欧氏质心 | **不适用** |

**为什么适合。** 无随机种子、切点可复现，适合作为 PAM 的对照。协议原文就要求“层次聚类多个切点 + k-medoids 多种子”。

**实现。** `hierarchical_cuts()`。默认搜索 average / complete；weighted、single 作对照。Ward 不实现。

**风险。** 贪心合并不可回退；single/average 在本数据上 J 明显差于 PAM。complete 更紧，但容易 source-driven。

### 3.3 密度法：DBSCAN / HDBSCAN / OPTICS

**想法。** 不先指定 K，按密度连通扩簇。噪声标为 -1，对应协议里的 `review_outlier`。

- **DBSCAN**（Ester et al.）：全局 `eps` + `min_samples`。实现简单，密度差大时一个 eps 不够。
- **HDBSCAN**（Campello et al.）：互达距离 MST + 稳定簇抽取，可变密度。协议 4.1 明确要求密度不均时补充。
- **OPTICS**：有序可达图，可导出多层 DBSCAN。与 HDBSCAN 信息重叠，本轮未单列。

**为什么适合。** 标签空间里短音频、带乐、低 SNR 可能是稀疏尾部，不一定是球形簇。噪声点可以进 review，不必强行塞进主簇。

**本任务的额外约束。** 生产要 `K>=6` 且主簇够大。密度方法常给出过少、过多或大量噪声。实现上对主簇做 `force_k_range`：过小合并、过少则对最大簇 PAM 二分。噪声默认保留为 -1，不静默贴到最近簇；`dbscan_reassign` 只作为对照。

**实现。** `dbscan_labels` 用 pairwise 距离分位数作 eps 网格；`hdbscan_labels` 用互达距离上的 single-linkage 切树，丢掉小于 `min_cluster_size` 的块。

**风险。** eps 网格是启发式；force-K 会削弱“密度聚类”的原意，只作为把结果送进同一套 J/Q_tag 门控的手段。HDBSCAN 在本快照上未抽出稳定主簇。

### 3.4 图与 exemplar：谱聚类、Louvain、Affinity Propagation

**谱聚类。** 由 `d_tag` 建 kNN 高斯相似度，对规范化 Laplacian 取 Fiedler 向量递归二分，直到目标 K。适合流形/带状结构，不保证 medoid 可解释，也不优化 `C_tag`。

**Louvain / Leiden。** 在 mutual kNN 图上最大化模块度。簇数由分辨率决定，不是输入。社区可能变成“来源团”而不是声学场景。Leiden 比 Louvain 更保证连通，本轮先实现 Louvain。

**Affinity Propagation**（Frey & Dueck）。每点当 exemplar 候选，用 `-d_tag` 作相似度，preference 控制簇数。优点是自动 K、exemplar 就是样本；缺点是簇数常远大于 12，压缩后 J 仍然差，计算也是稠密 `N²`。

**实现。** `recursive_spectral_labels`、`knn_louvain_labels`、`affinity_propagation_labels`。

**风险。** 图方法对 kNN 的 k、核宽、preference 敏感；模块度偏好同等大小社区，和 `n_min`、来源平衡不是一回事。

### 3.5 考虑过、本轮不作为主搜索的方法

| 方法 | 不作为主搜索的原因 |
| --- | --- |
| k-means / MiniBatch k-means / BIRCH | 需要欧氏均值，不能直接用 `d_tag` |
| GMM / VAE 聚类 | 需要坐标和协方差；标签缺失与集合特征对不上 |
| Ward HAC | 合并准则是欧氏方差 |
| k-prototypes | 会替换已校准的轴距离 |
| MDS/PCoA + k-means | 先把相异度嵌到欧氏空间，medoid 与 `d_tag` 不再一致 |
| UMAP + HDBSCAN | 常用探索工具，但嵌入随机、超参多，不适合作为可复现生产分区 |
| FANNY（模糊 PAM） | 相异度上可用，输出是软隶属，和当前硬 `cluster_id` 协议不一致 |
| DIANA（分裂层次） | 与 PAM 二分 split/merge 重复 |
| Leiden | 本轮 Louvain 已足够判断图社区是否 source-driven |

这些方法以后若做敏感性分析，必须先声明：嵌入或欧氏化之后的簇，不能直接替换 tags-only 的共享 `cluster_id`。

### 3.6 更灵活、不绑死簇形的路线

上一节的 PAM / HAC / DBSCAN / 谱聚类都还是**一种固定生成器**：先选算法族，再出一张扁平硬分区。场景本身未必是互斥、等大、球形的。更灵活的意思通常有四层，可以叠加，不必一次全上。

| 灵活在哪 | 固定形式在做什么 | 更灵活的对象 |
| --- | --- | --- |
| 簇形 | 假设圆、链或密度球 | 不同区域用不同生成器，或根本不假设形状 |
| 簇数 K | 先搜 6…12 再切一刀 | 保留树 / 后验，按用途再切 |
| 隶属 | 每点恰好一个 `cluster_id` | 软隶属、重叠、拒识 |
| 距离 | 四轴永远等权平均 | 轴权重、多视图、局部度量可学 |

下面按**可落地的做法**写，不改 `d_tag` 的定义，也不把 WER/CER 写进距离。实现时应单独开规则小版本，例如 `cluster_rules_v0.5+ensemble_v1`，不能静默覆盖现网 `result/scene-clusters/`。

#### 3.6.1 证据累积 / 聚类集成

目标：不再从 68 个候选里“选一个生成器的赢家”，而是让生成器当投票人，抽出它们共同同意的块。依据是 Fred & Jain 的 evidence accumulation clustering (EAC)。

**输入。** 冻结后的 tags-only `d_tag`（N×N），以及 `generate_method_partitions()` 已经能产出的一批硬标签。默认票仓：

- k-medoids：K=6…12，种子 0/1/2
- average / complete linkage：同一组 K 切点
- DBSCAN：eps 分位数网格；噪声标 -1，**这一张票不把噪声强行归进主簇**
- spectral：K=6…12

一张票要进入共现，必须先过结构硬门（`passed=true`），并且 `source_driven=false`。过碎（K>k_max）、过空（主簇数 < 2）或未分配比例 > `b_non_main_max` 的票丢掉，避免噪声方法把共识拉碎。WER/CER 此时仍不可见。

**共现矩阵。** 对每对样本 (i,j)，i<j：

```text
n_valid_ij = 进入票仓、且 i 与 j 都有非负标签的票数
C_ij = (# 两票同簇) / n_valid_ij     若 n_valid_ij >= n_vote_min
C_ij = NA                              否则
C_ii = 1
```

`n_vote_min` 预注册，建议 `max(5, round(0.3 * n_ballots))`。NA 在后续距离里当缺失，走现有 missingness cap，不填 0（填 0 会把“很少一起被标”当成“强不同”）。噪声点与任何点的同簇次数都不计。N=1000 时 C 约 50 万对，可直接存上三角。

可选加权：票权 `w = 1 / K` 或 `w = Q_tag`，避免高 K 碎片票主导共现。权重必须来自标签门控，不能来自 η²。

**共识距离与再聚类。**

```text
d_cons(i,j) = 1 - C_ij     # C 为 NA 则 d_cons 为 NA
```

在 `d_cons` 上跑 **average-linkage 切点 + PAM 多种子**，K 搜索范围与主实验相同。不要用 single linkage（EAC 原文常用它，但在来源团上会 chaining）。得到的标签仍要在**原始 `d_tag`** 上重算 medoid、`J`、`Q_tag`、来源占比——共识只负责提案，评价必须回到音频标签距离，否则会优化“算法互相同意”而不是“标签紧致”。

**筛选。** 与主实验同一套：结构门控 → 非 source-driven → 事后 mean SηK 过滤 → J-band。集成分区是额外一个 `method=eac_consensus` 候选，不是自动赢家。若它的 J、Q_tag 不优于 PAM，生产仍留 PAM，只把共现矩阵留下做稳定性诊断。

**产物。**

- `coassociation.npy` / `coassociation.json`：C 的上三角 + 票仓清单（method、K、seed、是否入票）
- `consensus_assignments.jsonl`：共识硬标签
- `vote_disagreement.jsonl`：`1 - max_g C_i(g)` 或 top-2 共现差，作为后续软隶属
- 与 PAM 的 ARI/NMI：若 ARI 很高，说明集成没有新结构；若很低但 Q_tag 也低，说明共识在切来源或噪声

**消融。** leave-one-algorithm-out 重算 C，看共识对 PAM/DBSCAN 是否过敏；把 source-driven 票放进去对比一次，确认门控有用。

#### 3.6.2 保留场景树，而不是只冻 6 个桶

目标：主对象改成一棵匿名树，扁平 `cluster_id` 只是树上的一个 cut。评测用粗层，听审和边界分析用细层。

**建树，两条可并行。**

1. **标签树：** 在 `d_tag` 上做 average 或 complete linkage，存完整合并历史 `(i, j, height, size)`。不要只用 K=6…12 的快照。
2. **共识树：** 若已有 3.6.1 的 `d_cons`，在其上做 average linkage。这棵树表示“算法们何时同意分开”，往往比单次 HAC 稳。

HDBSCAN condensed tree 是第三条：互达距离 MST 上按稳定度抽簇。本快照上它没抽出主簇，仍应把 condensed tree 存下来，供看哪些分支从未稳定。

**节点画像，而不是先命名。** 每个内部节点（簇）计算：

- n、来源占比、是否 `source_driven`
- 相对全局的标签锚点（与现在 `tag_anchors` 相同）
- 到父节点的 profile 是否**可区分**（兄弟节点的 tags-only 二分类 balanced accuracy 下界，或锚点集合是否不同）
- 节点内 `C_tag`（相对该节点 medoid）

只有同时满足 `n>=n_min`、有锚点、兄弟可区分、非 source-driven 的节点，才进入“合法场景节点”集合 `G*`。其它节点可以存在，但不能当生产 cut 的叶子。

**选生产 cut。** 从根往下，在 `G*` 里选一组不相交叶子覆盖尽量多样本：

```text
合法 cut = 一组节点 {g} ⊆ G*
  互不祖先，并集尽量覆盖 N
  K = |{g}| >= 6（样本支持不足则 fail-closed，不硬拆）
  未覆盖样本进 C_REVIEW_OUTLIER，计入 B_non_main
```

在所有合法 cut 上算原来的 `J`（距离仍是 `d_tag`）。用 Pareto / one-standard-error 选一个发布 cut，**同时把整棵树和 `G*` 写进 `cluster_summary`**。换评测任务时可以另选 cut，不必重跑生成器。

**样本侧输出。** 每条样本除 `cluster_id`（发布 cut 上的叶子）外，保存：

```text
tree_path: [root, ..., leaf]
cut_ids: { "k6": "C02", "k8": "C02a", ... }   # 仅合法节点
```

命名仍在冻结 cut 之后做。树上未入 `G*` 的节点禁止取业务名。

**和现网的关系。** 当前 PAM+merge 的 C00–C05 应能在树上找到对应节点或近邻 cut；若对不上，报告树与 PAM 的 ARI，而不是强制改名对齐。

#### 3.6.3 多视图共识（四轴不要先平均死）

目标：content / interaction / quality / spatial 各自保留结构，最后再合成，避免一条“重叠+带乐”被标量 `d_tag` 捏成四不像。

**单轴距离。** 现成的 `axis_distance()` 已经按轴输出标量。对每个轴 a∈{content, interaction, quality, spatial} 建 `d_a`，缺失规则与主实验相同。某一轴上两人都不信息（该轴 features 全缺）则 `d_a=NA`，不填 0。

**单轴分区。** 每个 `d_a` 上跑 PAM（K=4…8 即可，单轴不要求 K>=6）和 average linkage。同样丢掉 source-driven 票。得到四组标签 `P_content, …`。

**合成，两种做法只做一种当主，另一种当对照。**

1. **晚融合（推荐先做）：** 四组 `P_a` 当作 3.6.1 的票仓子集，再算共现、共识。解释直观：两点经常在多轴上被分到一起，才算同一场景。
2. **轴权重学习（标签可复原性，禁止看 WER）：**  
   `d_w = Σ_a w_a d_a`，`w_a>=0`，`Σ w_a=1`。  
   在 inner-train 上对 w 做单纯形搜索或 leave-one-axis-out：使 held-out `Q_tag` 或 `C_tag` 最好。outer-test 冻结 w。w 写入 `preprocess.axis_weights`。  
   leave-one-axis-out 若某轴被拿掉后分区几乎不变，说明该轴当前没贡献，应降权而不是删 schema。

**禁止。** 用各轴与 WER 的相关、η² 去学 w。那会把质量轴偷偷加成“难度轴”。

**产物。** 每条样本可附四轴最近 medoid 和轴内距离，便于解释“为什么进 C02”：例如 interaction 近重叠簇、content 近带乐簇。发布 hard id 仍来自合成分区。

#### 3.6.4 软隶属、拒识与混合体制

硬 `cluster_id` 继续给下游评测用；灵活放在附属字段，不替换主 ID。

**由距离得到隶属。** 发布 cut 的 medoid 为 m_g，对样本 i：

```text
s_ig = exp( - d_tag(i, m_g) / tau_soft )
p_ig = s_ig / Σ_g s_ig
margin = p_i(1st) - p_i(2nd)
```

`tau_soft` 在校准折上预注册（可用 `d_tag` 的中位最近邻）。`margin < epsilon_margin` 则 `assignment_approximate=true`，`cluster_id` 仍取 argmax，但 `needs_review=true`。这与现有 `assignment_gate.top2_margin` 对齐。

**混合体制（稠密核 + 尾部拒识）。**

1. 对每个 i 算 kNN 半径 `r_i = d_tag` 到第 k 邻（k 与协议 `k_graph` 相同）。
2. `r_i <= q_core`（如全局 r 的 0.8 分位）进核心集 Core；其余进 Tail。
3. **只在 Core 上**跑 PAM / EAC / 树 cut，得到主簇 medoid。
4. Tail 中，若 `min_g d_tag(i,m_g) <= r_assign` 则软归属，否则 `C_REVIEW_OUTLIER`。
5. `B_non_main` 照常惩罚；禁止为了让 η² 好看而把高误差点丢进 Tail。

DBSCAN 的噪声策略与此一致：噪声默认 -1，只有显式 `dbscan_reassign` 才贴最近核。

#### 3.6.5 受约束的 agentic split/merge（协议 4.3 的做法）

这是“搜索分区本身”，不是再加一个参数族。当前 `refine_split_merge()` 只是确定性的 PAM 二分 + 锚点不同则接受，是非 agentic 消融。完整做法：

**角色。**

| 角色 | 可以做什么 | 不可以做什么 |
| --- | --- | --- |
| Tag-view proposer | 根据锚点、缺失、子矩阵结构提出 split/merge/reassign | 读 WER、改 hard gate |
| Leakage critic | 标 source / group / language / 缺失捷径 | 因误差异质要求拆簇 |
| Deterministic arbiter | 算 J、Q_tag、n_min、锚点差、来源占比，接受或拒绝 | 被 LLM 覆盖 |
| Error-view | 冻结后打 `error_heterogeneous` | 写 `cluster_id` |
| Naming | 冻结后根据锚点模板起名 | 在 freeze 前起名 |

**一轮迭代。** 从当前合法分区 P 开始，beam 宽 B（预注册，如 4）：

1. Proposer 对每个足够大的簇（`n>=2 n_min`）提出 2-way PAM split，或对锚点相同/过小的簇对提出 merge。每个 proposal 必须带 `tag_evidence`（将产生的锚点、n、来源占比）。
2. Critic 若发现某一子簇 `max_source_share>=source_max_share`，或 must-link 组被拆超预算，否决该 proposal。
3. Arbiter 在 inner-val 上算 ΔJ、锚点是否不同、Q_tag 是否越界。只有 `ΔJ <= -epsilon_refine` 且兄弟画像不同且不越 guardrail 才接受。
4. 接受后入 beam；超出宽度时按 inner-val 的 J 截断。
5. 连续一轮无接受则停。WER 全程不在 proposal 字段里。

**must-link / cannot-link。** 同一 `group_id` 默认 must-link（标签签名相同则强制）。cannot-link 不按来源两两连边（那会爆炸），而是约束**簇级**最大来源占比，在 assign 时若把某点并入会打破约束，则改赋次近 medoid 或拒识。这比事后 `source_driven=true` 更早生效。

#### 3.6.6 不建议当成“更灵活”的做法

- 把 WER 当第五轴或分裂依据：换了一种更死的难度分箱。
- 无约束深度学习聚类 / GMM：输入对不上 `d_tag`，medoid 不可审计。
- 无门控的集成：共识会把来源和语言当成场景。
- 为灵活而放出 K=90 的 AP/Louvain：那是碎片，不是多分辨率场景。

#### 3.6.7 建议落地顺序

1. **共现集成（3.6.1）**：复用现成票仓，产物是一个新候选 + 共现矩阵。**已实现。**
2. **在 `d_cons` 或 `d_tag` 上留树（3.6.2）**：发布 cut 仍可与现网 6 簇对照。**已实现（合法 cut 本快照全被 source-driven 挡住）。**
3. **四轴晚融合（3.6.3）**：解释“凭哪一轴进簇”。未做。
4. **p_tag / margin / Core-Tail（3.6.4）**：硬 ID 旁记录 `p_tag` / `margin` / `vote_disagreement`。附属字段已写出，未改 hard id。
5. **真正的 proposer+critic+arbiter（3.6.5）**：替换现在的贪心 refine。未做。

每一步都用同一套 `J` / `Q_tag` / 来源门控验收；事后 SηK 只筛选、不回写。若某步不优于现网 PAM 六簇，就停在诊断快照，不替换 `result/scene-clusters/`。

入口：`python3 run_flexible_clustering.py --output-dir result/scene-clusters-flexible`。票仓默认 k-medoids / average / complete / spectral，外加噪声保持 `-1` 的 raw DBSCAN。`n_vote_min` 按**票数**计，不是权重和；权重只进入 `C_ij` 的分子分母。

#### 3.6.8 本轮 EAC / 场景树结果

1000 条、tags-only `d_tag`、权重 `w=1/K`。52 张生成票里留下 28 张（complete linkage 全部 source-driven 被丢；raw DBSCAN 大多因 `K<6` 未过硬门）。共现矩阵 499500 对全部有定义（`n_vote_min=8`）。完整表见 [`summary.md`](../dataset/multi-data/result/scene-clusters-flexible/summary.md)。

| 分区 | K | J | Q_tag | mean SηK | source_driven | ARI vs PAM |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| PAM refine（现网） | 6 | 0.1947 | 0.6064 | 0.3157 | no | 1.00 |
| EAC consensus refine | 6 | 0.1805 | 0.5344 | 0.3211 | no | 0.87 |
| 共识树 cut | 6 | 0.4885 | 0.6664 | 0.2961 | yes | 0.79 |

读法：

- **J 更低不等于可替换。** EAC 在 `d_cons` 上的 k-medoids（K=6, seed=0）再映回 `d_tag`，J 从 0.1947 降到 0.1805（超过 1% 门槛），但来源留一 `Q_tag` 从 0.606 掉到 0.534。块更“算法们都同意”，跨来源却更难用标签 medoid 复现。生产分区仍留 PAM。
- **ARI=0.87 / NMI=0.85**：集成没有挖出和 PAM 正交的新几何，只是把 PAM 的小簇（17 / 38）重新切成 88 / 94 / 43。这是稳定性诊断，不是新场景。
- **共识树上的扁平 cut 全部 source-driven**，J 也回到 0.49 附近，和单次 average linkage 一样偏链。树仍然写出（`scene_tree.json`、样本 `tree_path` / `cut_ids`），但不能当发布叶子。
- **事后 SηK 略高**（0.321 vs 0.316）不参与替换判定。WER/CER 从未进入 `d_tag` 或 `C`。

因此 3.6.1–3.6.2 停在诊断快照 `result/scene-clusters-flexible/`。下一步若继续，应做 3.6.3 四轴晚融合或 3.6.5 的受约束 split/merge，而不是把 EAC 的低 J 写进生产。

## 4. 本仓库落地

`cluster_methods.generate_method_partitions()` 在同一张 `d_tag` 上产出匿名标签。`cluster_scenes.py` 默认搜索：

```text
kmedoids, average_linkage, complete_linkage, dbscan, spectral
```

对照实验另加 weighted linkage、HDBSCAN、affinity、kNN Louvain。所有方法共享后续步骤：

1. 在 `d_tag` 上算 medoid、`J`、`Q_tag`、来源占比、silhouette。
2. 不 source-driven 的候选进入池。
3. 打开 WER/CER **只做筛选**：丢掉 mean SηK 低于 `max(0.10, 30th percentile)` 的候选。
4. 在 J-band（`J <= 1.05 J_min + 0.01`）里取最低 J，并列时看 Q_tag、SηK。
5. 对选中分区做 tags-only split/merge，再 canonicalize 成 `C00…`。

WER 从不改 `d_tag`，也不作为 split/merge 接受条件。这与协议第 3–4 节一致，只是把“多种算法出候选”从文档补成实现。

## 5. 本轮对照摘要

1000 条、K∈[6,12]、68 个候选的完整表见 [`METHOD_COMPARE.md`](../dataset/multi-data/report/METHOD_COMPARE.md)。结论只在这里复述选型含义：

- **k-medoids 仍是唯一同时满足低 J、高 Q_tag、非 source-driven 的方法。** 初选 K=7，merge 后 K=6，与现网 `result/scene-clusters/` 一致。
- **DBSCAN 是唯一接近的备选**，事后 SηK 略高，但 J 落在 5% band 外，不能替换共享分区。
- **谱聚类能压低 J，但 Q_tag 从约 0.62 掉到 0.30**，说明切出来的块在来源留一上不可复用。
- **层次聚类、AP、Louvain** 要么 chaining / 过碎，要么变成来源社区。
- **HDBSCAN** 在当前 `min_cluster_size` 下没有稳定主簇。

因此：生产分区继续 PAM；搜索列表保留 DBSCAN 和谱聚类作监视，避免以后数据密度变化时漏掉非球形结构；不要把 WER 轴实验当作场景定义。证据累积（3.6.8）同样不能替换 PAM：J 虽低，Q_tag 掉了 0.07，且与 PAM 的 ARI 高达 0.87。

## 6. 复现

```bash
cd /hpc_stor03/sjtu_home/weihan.chen/share/compare/dataset/multi-data
python3 compare_clustering_methods.py \
  --output-dir result/scene-clusters-method-compare
python3 run_flexible_clustering.py \
  --output-dir result/scene-clusters-flexible
python3 cluster_scenes.py \
  --cluster-methods kmedoids,average_linkage,complete_linkage,dbscan,spectral \
  --wer-filter \
  --output-dir result/scene-clusters
```

规则版本仍是 `cluster_rules_v0.5`。新增的是候选生成器，不是新的距离定义。
