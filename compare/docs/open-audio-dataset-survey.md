# sure-tagger / captioner 结构性标签评测：开源音频数据集调研

更新时间：2026-09-06  
范围：650 条 T1-T3 结构能力 benchmark 设计；暂不评 `descriptive.*`。

本文是 [`tag-schema.md`](tag-schema.md) 的配套 benchmark 设计，不修改 schema。第一阶段评
六个原始结构分组和 `deterministic_derived` 中已冻结的能力字段；所有 `descriptive.*` 均不参与样本选择、GT
coverage、能力指标或系统排名。系统即使继续输出这些字段，本轮也只检查 JSON/schema 合法性。

benchmark-v1 的默认能力雷达包含 18 个 Headline 叶字段；字段角色、单项评分和
`score_rate` 以 [`SCORING_STANDARD.md`](../dataset/benchmark/SCORING_STANDARD.md)
（`benchmark-scoring-v0.8`）为准。GT 可以从 reference transcript 或冻结规则物化，
但这不表示 captioner 可以免费从其它预测字段推导出对应能力；预测端仍须逐字段独立评测。

这个收缩会得到更可靠的方案：多数结构字段可以对接独立的时间标注、转写、文件头、源轨或 RIR，
而开放文本没有唯一答案，也很容易把 caption、transcript 和事件标签混为一谈。
代价是仍然不存在一个数据集覆盖全部结构字段，因此必须采用**多 track、逐字段 mask、分开报告**，
不能追求一个全字段总分。

文中“公开数据集”包括可直接下载、需注册/申请，以及仅公开 URL/标注的资源。是否可公开重打包
音频必须以冻结版本内的 manifest、逐文件 license 和使用协议为准。

## 1. 范围与推荐方案

### 1.1 第一版 benchmark

历史第一版 benchmark 曾冻结为 1,000 条；T4 已禁用，当前正式 benchmark 为 **650 条 T1-T3 结构能力主评测**。旧 AMI 300 条继续只作为兼容基线；
正式 T1 已从完整 AMI utterance 树重新抽取 350 条，并同步保留每条音频对应的原始 manifest 记录。

正式取数前先做 **Phase 0 go/no-go**：抽取 20--50 条 shared-audio pilot，确认 captioner 在
`text_sha256=null`、没有 reference/sidecar 的情况下能够运行，并达到预先登记的 schema-valid 和
结构字段有效输出门槛。若不能通过，后续数据仍可用于两个系统各自的独立 GT 报告，但不存在
650 条 head-to-head headline；不能在运行后改用富文本输入再保留“直接比较”的名称。

| Track | 第一版建议 | 预算 | 主要结构 GT | 报告定位 |
| --- | --- | ---: | --- | --- |
| T1 自然会议与语音结构 | AMI utterance 350；覆盖全部 170 个 meeting，每个至少 2 条 | 350 | speaker count/change/overlap、ASR、language、词面统计；`ami-turn-complement-v1` 官方 `silence_ratio` GT | 自然录音 headline；原始 manifest 只留在 GT 侧 |
| T2 多语言与干净 ASR | FLEURS 100（10 种语言各 10）；LibriSpeech 50 个英文 clean anchor | 150 | transcript、language、词面统计 | FLEURS panel 与英文 anchor 分报；不声称覆盖全部 102 种语言 |
| T3 事件与背景声 | DESED 75 + FSD50K 75（dev 38、eval 37），按冻结 crosswalk 采用闭世界解释 | 150 | clip-level speech/music/noise/event；映射到目标类的标签为正例，其余目标类为负例 | 事件 headline；单列闭世界假设造成的潜在漏标风险 |
| T4 可控声学（已禁用） | 历史来源池和旧产物保留供审计 | 0 | 不进入当前 GT、runtime 或 scorer | 不参与当前 benchmark |

各 track 不是要求每条样本都有所有 GT。完整字段指标只评 `gt_mask=true` 且没有输入泄漏的字段；
多标签的部分 GT 只按 class mask 进入对应 per-class 指标。
自然录音、公开半合成和自建合成三类结果必须分表，不做跨 track 的单一总分。

主观质量 track 已从本版计划删除。`audio_quality.dnsmos_sig/bak/ovrl/p808` 保留为 schema 接口字段，
但本轮统一 `gt_mask=false`，不下载专用数据、不建立 GT，也不进入指标或排名。

T1 当前采用完整 AMI utterance 树已有的切片单位：170 个 meeting 各按时间顺序取两个 midpoint
quantile，另对确定性的 10 个 meeting 各取第三条，共 350 条。`t1-ami/manifest.jsonl` 保存对应的
原始 transcript/turn 记录，只能作为 GT；统一系统输入 manifest 已清空这些字段。后续若扩展到固定
30 s 窗口，须另起版本并冻结尾段、padding、stride 和边界归属规则，不能与本版混报。

### 1.3 字段分级

同一能力组内的数值字段、布尔投影和派生字段都独立计分并展示。某系统即使细粒度字段为 `null`，
只要粗粒度/派生字段有输出，仍在该字段自己的 eligible 分母中获得相应得分。跨字段一致性另列合同
报告，不用细字段覆盖粗字段；能力组聚合必须公布固定字段权重，不能临时选择对某系统更有利的层级。

| 级别 | 字段 | 本轮处理 |
| --- | --- | --- |
| Headline 主指标 | `basic_acoustic.nonspeech_ratio`（T1 官方 GT 名为 `silence_ratio`）；`sound_field_scene.speech_music_events/music_present/external_noise_type`；`speaker.speaker_count/speaker_present/multi_speaker/speaker_change_count/speaker_change/overlap_ratio/speaker_overlap`；`speaker.asr_transcript`；`language_content.language`、`language_content.word_count`、`language_content.repetition.has_repetition`、`language_content.repetition.repetition_count`；`deterministic_derived.speaker_change_rate_per_min/filler_ratio` | 只在可审计 GT 上逐字段计分；细字段为 `null` 不屏蔽同组粗字段；当前雷达不包含 T4 专属 SNR/room 轴 |
| 条件/诊断 | `language_content.punctuation.punctuation_count`、`language_content.punctuation.has_terminal_punctuation`、`language_content.filler` | 保留原始结构输出和 GT 规则用于审计，但不进入能力分数、分母或雷达图 |
| 条件指标/诊断 | `basic_acoustic.silence_segments`；`sound_field_scene.noise_composition`；`speaker.profiles` 的 speech rate/pitch/volume | 有可审计 GT 时同样独立计分；否则报告 proxy、相关性、测量一致性或覆盖率 |
| 合同校验 | `basic_acoustic.duration_sec/sample_rate_hz/channels`；`speaker.profiles` 容器、`speaker_id` 和 `speech_rate.unit`；所有已知依赖与粗粒度/派生字段之间的公式 | 合同报告与能力分并行；依赖缺失为不可检查，不把目标字段判错或清空 |
| 当前占位 | `room_acoustic.far_field` | 共同评测只检查 `boolean \| null` 类型并将能力 mask 关闭；sure-tagger 当前应为 `null` 是系统特定实现预期，不能据此把 captioner 的合法 boolean 判成 schema 违例 |
| 排除 | 四个 `audio_quality.dnsmos_*`；全部 `descriptive.*` | 不进入本轮 GT、指标、分母和排名；只检查 JSON/schema 合法性 |

T1 的非语音比例采用官方 `silence_ratio` GT：按 `ami-turn-complement-v1` 合并 AMI utterance
turn，再以时间补集占整段时长的比例计分。评分 schema 仍使用 `basic_acoustic.nonspeech_ratio`，
并兼容预测端同义键 `basic_acoustic.silence_ratio`。该 ratio 口径不等于物理低能量静音；
`silence_segments` 仍需冻结区间 GT、端点和 collar 规则，当前只作条件/诊断。

词面字段的 GT 可由 canonical reference transcript 按冻结 tokenizer、标点、repetition 和
filler 规则物化，但 `word_count`、punctuation、repetition、filler 的预测值是独立能力输出。
captioner 不需要先提供 `speaker.asr_transcript` 或其它依赖字段，才能获得这些字段自己的
能力分；依赖缺失只影响另列的合同检查。

### 1.4 输入模态与公平性门槛

只有下列 **shared-audio 协议**可以产生 sure-tagger 与 captioner 的直接排名：

1. 两个系统收到逐字节相同的 `system_input_audio`，以 SHA256 核验。
2. 两边都不得收到 reference transcript、caption、事件标签、原生 annotation、source stem、RIR、
   合成 recipe 或任何 GT sidecar。系统内部从同一音频自行 ASR 属于系统能力，不算泄漏。
3. GT 必须来自数据集、独立测量、独立人工复核或受控 recipe，不能用任一被测系统的输出充当 GT。
4. 同一字段使用相同的样本 mask、归一化和容差；系统失败或输出 `null` 仍留在分母中。

若 captioner 的原生接口接收 reference text，而 sure-tagger 只接收音频，则改记为
`input_mode=native_input`：两套系统可以分别对独立 GT 报告，但不得给 head-to-head 总排名。
输入文本已经直接包含或同义透露答案的字段必须在该系统的 `leakage_mask` 中排除。报告必须同时
展示 `input_mode`、音频 hash、文本来源/hash 和最终可评字段，不能把不同模态结果并排后称为优劣。

若 captioner 当前只能从人工 rich description 抽取标签，它在本方案中属于**文本信息抽取系统**，
不是 audio tagger。此时仍可用独立 GT 检查其输出，但结果回答的是“给定文本后能否正确结构化”，
不能回答“从音频识别结构标签是否优于 sure-tagger”。若输入文本来自 sure-tagger 输出，则所有被
提供的结构路径都属于泄漏，只能评流水线合同，不能再当独立系统比较。

### 1.5 GT 类型约定

| 标记 | 含义 | 可否用于 headline |
| --- | --- | --- |
| D0 | 数据集原生标注或系统实际输入文件的事实，可按稳定 ID 连接 | 可以 |
| D1 | 由 D0 时间戳/转写按冻结规则确定性派生，输出使用 `gt_derived_` 命名 | 可以，但必须公布规则版本 |
| W | weak/positive-unlabeled 标签；未标注不等于 negative | 只能报正例 recall/coverage，除非另有完整复核 |
| C | 可复现 recipe 控制且渲染后验收的 GT | 只用于 controlled track |
| H | 独立人工听测或人工完整复核 | 可以，但报告标注协议和一致性 |
| M | 独立算法测量或另一个模型的评分 | 仅诊断，不能称人工 GT |
| -- | 没有可验证对应标注 | 不可评 |

T3 是本轮对 `W` 的预注册例外：只对冻结的三类 `speech_music_events` 和六类
`external_noise_type` 使用闭世界解释。原生标签经版本化 crosswalk 映射到某目标类时记为正例；
没有任何原生正标签映射到该目标类时直接记为负例。源标注缺失、音频损坏或 crosswalk 无法解析的
样本不制造标签，而是从该类分母排除。报告必须将结果命名为 `native-label closed-world`，不能将
这些派生负例描述成人工确认负例。

按照 schema，`null` 表示未知；`false`、`0` 和 `[]` 是已知值。coverage 必须把 GT 未定义、
明确负例/空集合、系统输出未知、解析失败和整条运行失败分别统计。

## 2. 高价值数据集清单

### 2.1 说话人、turn、转写和远场

| 数据集与官方入口 | 规模与原生 GT | 对 schema 的映射 | 授权/访问与局限 | 建议用途 |
| --- | --- | --- | --- | --- |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | 约 100 h；会议数量会因 release 和切分口径不同（常见统计约 100--171 场）；头戴麦和阵列等多通道；手工 utterance、speaker、起止时间、逐词时间戳和 transcript | D0：duration/channel/transcript；D1：`speaker_count`、`multi_speaker`、change、overlap、speech/non-speech 区间、word/punctuation；由 turn 补集得到的只能叫 non-speech/silence proxy，除非 release 另有 VAD；阵列录音只作远场压力条件 | 许可应以当前官网和发布包的 `LICENSE` 为准（历史版本常见 CC BY 4.0，不能只凭旧页面断言）；音频是会议录音，事件、RT60、主观质量没有人工 GT；speaker ID 是匿名且 clip-local | **P0 核心集**。现有 300 条 manifest 可用于 pilot；正式样本按 meeting 分组抽取 |
| [AliMeeting / OpenSLR 119](https://www.openslr.org/119/) | 约 118.75 h 中文会议，约 200+ session；近讲/远场多通道；transcript、VAD/diarization 时间标注 | D0/D1：中文语言、转写、speaker、turn、重叠、静音、channel；有条件地比较 `speech_rate` | OpenSLR 页面和压缩包中的 license/使用条款优先；发布版可能要求研究用途或署名，不能据名称推断可商用 | 中文远场结构性 track；与 AMI 分开报，避免语言和麦克风分布混淆 |
| [AISHELL-4 / OpenSLR 111](https://www.openslr.org/111/) | 约 120 h、约 200+ 中文会议，4--8 人，多通道远场；转写和 diarization/RTTM | D0/D1：`language=zh`、transcript、speaker count/change/overlap、VAD/silence、channel；可派生词统计 | 以 OpenSLR 的当前 license 和下载协议为准；不同 AISHELL 版本条款不应混用；通常没有事件、SNR、RT60/C50 或主观质量 GT | 中文会议的独立测试域；按会议划分 train/test，不能随机切片泄漏 |
| [CHiME-6](https://github.com/chimechallenge/chime6) | 约 40 h、20 名说话人、6 个住宅场景；多麦远场 dinner-party；transcript、speaker/diarization、重叠和阵列元数据 | D0/D1：多通道、duration、speaker/turn/overlap、speech/non-speech proxy、transcript；住宅/远场只可作为场景元数据，不能反推 RT60 | 挑战赛注册、下载协议和研究用途条款；音频/annotation 的再分发要单独核对 | 远场、多人重叠和背景家居声压力集；不作为唯一主 benchmark |
| [LibriCSS](https://github.com/chenzhuo1011/libri_css) | 约 10 h、7 麦远场连续语音；由 LibriSpeech 语句按 0/10/20/30/40% overlap 条件构造，带 source/transcript 和 segment 标签 | D0/D1：已知 speaker、turn、overlap ratio/边界、speech activity、channel；可做 `speaker_change`/`overlap` 的受控曲线 | 属于 LibriSpeech 衍生集，代码和音频包的发布/再分发条款需同时核对；它是半合成条件，不代表自然会议 | 小规模 overlap calibration；与 AMI/CHiME 的自然场景结果分开报 |
| [ICSI Meeting Corpus](https://www1.icsi.berkeley.edu/Speech/mr/) | 约 72 h；约 53 场完整转写会议（总录音包含更多 session）；多通道、speaker turns、逐词/对话标注 | D0/D1：speaker/turn/change/overlap、transcript、词统计；可做会议场景弱标签 | 通常需签署 ICSI 数据使用协议，属于“可申请研究数据”而非无条件开放；逐版确认 | AMI 的跨语料复核；预算允许时加入 |
| [DIHARD III](https://dihardchallenge.github.io/dihard3/) | 多域 diarization challenge 音频，包含会议、电话、家庭、广播等域；官方提供 RTTM/UEM 和评测协议，版本规模随 challenge 变化 | D0/D1：speech activity、speaker count/change/overlap、turn 边界和 channel/domain 切片；没有统一的逐词 transcript 或房间/SNR GT | challenge 数据需按具体年份注册并接受下载/使用条款；各 domain 的标注深度不同，不能把所有 RTTM 当同一质量等级 | 跨域 diarization 诊断；与 AMI 的自然会议 headline 分开报告 |
| [VoxConverse](https://github.com/joonson/voxconverse) | 来自 YouTube 访谈/节目，多说话人；官方提供 RTTM/UEM 和分段，版本规模随 release 变化 | D0/D1：duration、speech activity、speaker count/change/overlap；通常无可靠逐词 transcript、room 或 SNR | 标注/脚本可公开，但原始 YouTube 音频受平台条款和版权约束，不能把“URL 可见”当作可再分发许可 | 真实节目 diarization 压力集；记录失效 URL 和可复现下载比例 |
| [LibriSpeech / OpenSLR 12](https://www.openslr.org/12/) | 约 1,000 h、2,484 名英语朗读者、16 kHz；utterance segment、speaker/chapter、规范 transcript | D0：duration/sample rate/channel/transcript/language/speaker；D1：word/punctuation（原文本标点有限）；filler/repetition 只能按词面规则计数，不是 disfluency GT；每段基本单人且近似干净 | **CC BY 4.0**（底层 LibriVox 文本/录音另有来源说明）；适合公开基线 | 干净 ASR/语言 baseline；不能用来评估 overlap、环境噪声、房间或 event |
| [LibriTTS / OpenSLR 60](https://www.openslr.org/60/) | 约 585 h、约 2,400 名说话人、24 kHz；raw/normalized text、speaker/chapter 和 utterance 对齐信息 | D0/D1：transcript、language、speaker、duration；只有在额外 forced alignment 后才派生逐词 timing/speech rate；无 turn/overlap/event | 通常 CC BY 4.0；按 OpenSLR 压缩包中的 NOTICE 使用 | 补充 LibriSpeech 的文本和语速实验；仍是单人朗读域 |
| [Multilingual LibriSpeech (MLS) / OpenSLR 94](https://www.openslr.org/94/) | 约 50,000 h、8 种语言；有声书 segment、speaker 和 transcript | D0/D1：language、transcript、duration、speaker、词统计；以单人朗读为主、相对干净但录音质量有变异；不作为严格 clean acoustic GT | 通常 CC BY 4.0；语言子集/版本要在 manifest 记录 | 多语言 `language`/ASR 覆盖率；不覆盖声场 |
| [Common Voice](https://commonvoice.mozilla.org/datasets) | Mozilla 众包，多语言且版本持续更新；用户提交的 prompt/reference text、locale、client_id、时长、投票和部分自报人口统计 | D0/W：language/prompt text/duration/header/client proxy；只有额外核验发音与文本一致后才作为 transcript reference；D1：word/punctuation；filler/repetition 只能按词面规则计数，不是 disfluency GT；自报 age/gender/accent 不应直接填入当前 schema（schema 没有这些结构字段） | 数据集发布通常为 **CC0 1.0**，但必须以所下载版本的 `LICENSE`/locale 条款为准；client_id 不是实名 speaker | 语言和短句鲁棒性；按 client_id 做 speaker-disjoint split，不能把同一用户随机拆到 train/test |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 102 种语言的朗读句；官方 train/dev/test 音频和 transcription，规模随版本 manifest | D0/D1：language、transcript、duration、词/标点；通常没有 turn/overlap/room/event | 以 Hugging Face dataset card 的当前 license 和原始来源说明为准（常见发布为 CC BY 4.0）；不要仅引用缓存脚本 | 语言识别/ASR 横向测试；不是声学场景 benchmark |
| [TED-LIUM 3 / OpenSLR 51](https://www.openslr.org/51/) | 约 452 h TED 演讲，带 talk/segment transcript、时间边界和演讲元数据 | D0/D1：英语 transcript、duration、speaker（通常单人）、word/punctuation；无自然 overlap、环境事件或 RT60 | 发布包常见为 **CC BY-NC-ND 3.0** 或附带来源条款；检查当前 OpenSLR NOTICE，不能商用或任意改编 | P1 讲座/独白 ASR 域；不使用 talk title/description 参与本轮评测 |

### 2.2 事件和音乐/背景声

| 数据集与官方入口 | 规模与原生 GT | 对 schema 的映射 | 授权/访问与局限 | 建议用途 |
| --- | --- | --- | --- | --- |
| [DESED / DCASE Task 4 (2021)](https://dcase.community/challenge2021/task-sound-event-detection-in-domestic-environments) | 家庭环境 10 s clip；合成子集有强 onset/offset，真实子集多为 weak labels；版本和类别数随 challenge 年份变化 | D0/W：T3 选用 synthetic-strong；原始事件经冻结 crosswalk 映射到九个目标类，未映射为正例的目标类按 T3 闭世界规则记为负例 | 每届 DCASE 的下载和 license 不同，必须固定年份、manifest 和条款；事件标注不是完整 SAD，不能生成 `silence_segments` | **正式 T3 75 条**（上游池 250）。当前 schema 没有事件时间线；结果须标明 `native-label closed-world` |
| [FSD50K](https://zenodo.org/records/4060432) | 51,197 个 Freesound clip、约 100 h、200 个 AudioSet ontology 类；人工验证的 clip-level 多标签，不提供统一 onset/offset | W/T3 closed-world：原始标签可映射 `speech_music_events` 三类和 `external_noise_type` 六个 broad classes；没有正映射的目标类直接记为负例 | Freesound 文件是逐文件 CC0/CC BY/CC BY-NC 等混合许可；闭世界规则可能引入漏标假负例；AudioSet/FSD 标签不能直接充当 DASS 细类 `noise_composition` | **正式 T3 75 条**（dev 38、eval 37；上游池 250）。版本化 GT 已物化，报告须单列闭世界局限 |
| [AudioSet](https://research.google.com/audioset/)（[ontology](https://github.com/audioset/ontology)） | 约 2.08M 个 10 s YouTube segment、527 类、多标签；平衡/非平衡 split | W：可映射 speech/singing/music 和部分 broad noise classes；通常没有 onset/offset | ontology/annotation 可引用；音频是 YouTube URL，版权、ToS、失效率和区域限制必须单独报告 | P1 taxonomy/正例扩展；只算可取回样本，未取回或未标注都不能当负例 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2,000 个 5 s 环境声、50 类、5-fold；clip-level 主类标签 | D0/W：部分主类可映射 animal/nature/mechanical/formless；其余类是否缺失仍需 mask | **CC BY-NC 3.0**；非商业限制 | 小型 event smoke test；按官方 folds，不用于 speaker/ASR |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 个不超过 4 s 城市声 clip、10 类、预定义 10 folds | D0/W：部分城市声可映射 mechanical/animal/channel_environment；主类标签不证明其他声源不存在 | **CC BY-NC 3.0**；按官方 folds，不能跨 fold 随机重采样 | P1 城市噪声 smoke test，不声称完整多标签 GT |
| [MUSAN / OpenSLR 17](https://www.openslr.org/17/) | 约 109 h speech/music/noise 素材，按 source 类别组织；常用于可控混音 | D0：source speech/music/noise；C：混音后的 source activity、SNR 和已加入事件 | 多来源素材，具体来源和许可以压缩包内 README/NOTICE 为准；逐文件核对 | T4 合成组件，不作为自然场景 GT |

### 2.3 SNR、房间和受控混合

| 数据集/资源与官方入口 | 规模与原生 GT | 对 schema 的映射 | 授权/访问与局限 | 建议用途 |
| --- | --- | --- | --- | --- |
| [REVERB Challenge](https://reverb2014.audiolabs-erlangen.de/) | 真实/模拟 reverberant speech、RIR、room/source/mic 配置；WSJCAM0 语音和参考 transcript；具体文件数按 release | D0/D1：RIR 可按固定算法计算 RT60/C50；source-mic distance、array/channel、reverberant speech；当前不直接评 `far_field` | Challenge 数据有注册和发布条款，底层 WSJ/WSJCAM0 也有来源限制；不能无条件再分发 | T4 房间资源。只用 RIR/元数据作物理 GT，不能由 RT60 反推房型或表面 |
| [OpenAIR library](https://www.openairlib.net/) | 真实空间的 impulse responses 及 room/source/microphone metadata，覆盖礼堂、教堂、教室等 | D0/D1：RIR、采样率、channel、距离；D1：RT60、C50、DRR 等（公式和频带版本化）；无 speech/transcript/event | 每个 IR 的 license/署名不同，逐文件核对；部分仅限研究 | 与 LibriSpeech/MLS 等 clean stem 卷积，构建可审计房间子集 |
| [WHAM / WHAMR](https://wham.whisper.ai/) | 在 WSJ0-2mix 两说话人混合基础上加入真实噪声；WHAMR 再加入混响，保留 source/mixture 和条件 metadata | D0/D1/C：由 source stems 和 schedule 得到 speaker activity/overlap；目标/实测 SNR 分开；WHAMR 的 RIR 可派生 RT60/C50 | WSJ0 源音频受 LDC/原始条款约束，noise 另有来源许可；衍生包条款优先 | 公开半合成的多人 overlap、SNR、混响校准；与自然会议分报 |
| [LibriMix](https://github.com/JorisCos/LibriMix) | 基于 LibriSpeech 的 2/3 speaker 合成混音；保留 source stems、transcript、speaker 和混音 metadata | C：speaker count、schedule/overlap、source activity 和 speaker-relative mixture level；只有另有独立 background-noise stem 时才形成 schema 的 speech-vs-background SNR | 合成包和源 LibriSpeech 的许可证分别核对；这是 controlled track，不代表自然会议分布 | 快速校准 overlap 和文本回连；clean 2/3-speaker relative SNR 不填 `audio_quality.snr_db` |
| [SMS-WSJ](https://github.com/fgnt/sms_wsj) | 空间化多说话人 WSJ 仿真，带 RIR、位置、增益、SNR 和 source/transcript metadata | C：speaker schedule、overlap、SNR 和 mic geometry；D1/C：从 RIR 计算 RT60/C50；当前不把 geometry 映射为 `far_field` | WSJ/LDC 源音频受限；代码/recipe 可公开不等于音频可再分发 | 获批后做空间混音与房间参数 stress test |
| [DEMAND](https://zenodo.org/records/1227121) | 多种真实环境噪声录音，按场景和通道组织；不含目标语音 | D0：noise source/domain；C：与 clean speech 混音后的 SNR 和 broad noise presence；只有严格 DASS crosswalk 才评 `noise_composition` | 逐版本核对 Zenodo 和源录音许可；场景名不能直接当 room、RT60 或 DASS 细类 GT | 作为合成噪声池和自然噪声切片诊断 |
| [RIRS_NOISES / OpenSLR 28](https://www.openslr.org/28/) | 真实/模拟 RIR 与噪声素材，常用于语音增强和混响仿真 | C/D1：RIR、SNR、RT60/C50（按固定测量器重算）；source class 作为 provenance | OpenSLR 包内各文件/脚本条款优先；文件名参数不能未经验收当物理真值 | 与 clean speech 组合的房间/噪声补集 |
| [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) | clean/noisy speech pairs，常见 SNR 0/5/10/15 dB；约万级 utterance（版本不同） | D0：clean/noisy、目标 SNR、speaker、transcript（若随版本）；D1：speech/silence；无 event/turn/room | 以 Edinburgh DataShare 当前版本的 license、署名和下载条款为准，不从旧镜像推断 | 小型 SNR/去噪校准集，不作完整 schema benchmark |

### 2.4 针对缺口的补充集（按需申请）

这些数据集不是首批必选，但对 filler/disfluency 和独立 speaker profile 很有价值；
多数有更严格的访问协议，不能因为论文或代码公开就当作无条件开放。

| 数据集与官方入口 | 适合补的字段 | 访问和主要限制 |
| --- | --- | --- |
| [Switchboard](https://catalog.ldc.upenn.edu/LDC97S62) | 电话对话的 speaker turn、逐词 transcript、filled pause、重复和修正；可 D1 派生 punctuation/filler/repetition | LDC 授权/付费协议；不适合公开再分发 benchmark 音频 |
| [Buckeye Corpus](https://buckeyecorpus.osu.edu/) | 自然会话、speaker、词级时间和 disfluency/phonetic labels，可诊断 filler、语速和边界 | 研究者申请/使用协议；说话人隐私和版本条款需核对 |
| [DisfluencyBank / TalkBank](https://talkbank.org/disfluency/) | 自然会话的转写和 disfluency 层（filled pause、repetition、repair 等，具体子语料不同） | 可补 `filler`、`repetition`、词/句边界和语速；优先使用其原生 disfluency 标签。TalkBank 各子项目的音频、转写和使用条款不同；先核对可下载范围和再分发限制 |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 约 110 位说话人、多口音朗读、transcript、speaker 元数据；可补 language/speech-rate/profile stress | DataShare 发布条款和版本 license 优先；没有自然 overlap、事件或房间 GT |

## 3. Schema 覆盖矩阵

下表覆盖本轮参与评测的非 `descriptive` 字段，并显式列出被排除的 `dnsmos_*`。“最合适的数据集”
只表示可能形成可审计 GT，最终是否进入分母仍由样本级 `gt_mask` 决定。

| Schema 字段组 | 级别 | 最合适的数据集/来源 | GT 与 mask 边界 |
| --- | --- | --- | --- |
| `basic_acoustic.duration_sec`、`basic_acoustic.sample_rate_hz`、`basic_acoustic.channels` | 合同 | 所有可取得音频的集 | D0；从**系统实际收到的文件**读取 header/帧数。原始多通道源与 16 kHz mono 输入分别保存，不能拿 source header 给转码后的输入评分 |
| `basic_acoustic.nonspeech_ratio` | Headline | 当前 T1 AMI 官方 `silence_ratio`；后续可扩展原生 SAD/VAD 的 AliMeeting/AISHELL-4/DIHARD | T1 350 条按 `ami-turn-complement-v1` 设置 `gt_mask=true`；预测端兼容 `silence_ratio` 同义键。其他数据集仍须冻结非语音定义、端点规则和分母后才能开启 mask |
| `basic_acoustic.silence_segments` | 条件/诊断 | 原生 SAD/VAD 的 AMI/AliMeeting/AISHELL-4/DIHARD；受控 source activity | D0/D1/C；需额外冻结区间、端点和 collar 规则后才可启用 interval 指标；当前只报 proxy/coverage，`gt_mask=false` 时不计分 |
| `audio_quality.snr_db` | Headline | WHAM/WHAMR、带独立 background-noise stem 的 LibriMix/SMS-WSJ 配置、VoiceBank-DEMAND、自建混音 | D0/C；保留 requested/official SNR，但连续指标优先使用从渲染 speech/background stems 按冻结算法复算的 measured SNR。自然录音或只有 speaker-relative level 时 mask=false |
| 四个 `audio_quality.dnsmos_*` | 排除 | -- | 本版不建立可测 GT；全部 `gt_mask=false`，只保留 schema 合法性检查 |
| `room_acoustic.far_field` | 当前占位 | schema 类型合同 + 系统特定实现合同 | 共同 schema 只校验 `boolean \| null`；sure-tagger 当前预期 `null` 单列检查，不适用于 captioner。即使数据集有距离/mic 元数据，也要等 near/far 定义冻结后才能评能力 |
| `room_acoustic.rt60_sec`、`room_acoustic.c50_db` | Headline | REVERB、OpenAIR、RIRS_NOISES、SMS-WSJ、WHAMR、自建 RIR 卷积 | D1/C；从保留的 RIR 按同一采样率、频带、截断和能量衰减算法计算，记录算法版本和失败；不能由数值反推房型或材质 |
| `sound_field_scene.speech_music_events` | Headline | DESED、FSD50K、MUSAN/受控混合 | D0/C/W；目标值域只有 `speech/singing/music` 三类。T3 对 DESED/FSD50K 使用冻结的 native-label closed-world 映射，可算 clip-level multilabel F1；其他 PU weak 集仍只报正例 recall/coverage |
| `sound_field_scene.music_present` | Headline | 与上一字段相同 | GT 从原生/受控事件独立物化；按 boolean F1/accuracy 计分。系统的事件数组为 `null` 不屏蔽本字段；两者都已知时另查一致性 |
| `sound_field_scene.external_noise_type` | Headline | FSD50K、DESED/ESC-50/UrbanSound8K 映射、受控混合 | D1/C/W；映射目标是 `music/animal/mechanical/nature/formless/channel_environment` 六个 broad classes，保存原类和 crosswalk 版本。T3 中没有正映射的目标类记为负例；该规则不自动推广到其他 weak 数据集 |
| `sound_field_scene.noise_composition` | 条件/诊断 | 受控混合；人工复核且能严格映射到 DASS 细类的子集 | C/H；对象值是 DASS 具体标签数组，不能直接把 AudioSet/FSD taxonomy 当 D0。crosswalk 不等价或只知道 broad class 时 mask=false |
| `speaker.speaker_count`、`speaker.multi_speaker` | Headline | AMI、AliMeeting、AISHELL-4、CHiME-6、DIHARD、LibriCSS/LibriMix/SMS-WSJ；干净朗读单人片段 | D0/D1/C；只统计 clip 内被完整标注的 speaker。未标注背景人声使人数成为下界时 mask=false；count 与 boolean 同组只聚合一次 |
| `speaker.speaker_present` | Headline | 由 GT speaker activity/count 物化 | D1/C；按 boolean F1/accuracy 独立计分。系统 count 为 `null` 不屏蔽本字段；两者都已知时另查 `count>0` 一致性 |
| `speaker.speaker_change_count`、`speaker.speaker_change`、`speaker.overlap_ratio`、`speaker.speaker_overlap` | Headline | AMI、AliMeeting、AISHELL-4、CHiME-6、DIHARD、LibriCSS/LibriMix/SMS-WSJ | D1/C；冻结 turn 排序、相邻同 speaker 合并、短段、边界 collar 和 overlap 分母。数值与布尔字段同组报告，并额外做内部一致性检查 |
| `speaker.profiles`、`speaker.profiles[].speaker_id` | 合同/诊断 | 有完整 speaker timeline 的会议或受控集 | 容器形状、ID 格式/唯一性属于合同。逐 speaker 诊断只有在系统 artifact 提供独立 timeline、可按时间重叠对齐 GT 时成立；匹配 cost 不得使用待评分的 speed/pitch/volume。否则只报 sample-level 属性 multiset/coverage，不称逐 speaker accuracy |
| `speaker.profiles[].speech_rate`、`speaker.profiles[].speech_rate.band`、`speaker.profiles[].speech_rate.value`、`speaker.profiles[].speech_rate.unit` | 条件/诊断 | AMI/AliMeeting/AISHELL-4、LibriTTS/受控语音（需 speaker turn + transcript/word time） | D1；冻结 tokenizer、语音时长分母、overlap 处理、语言单位和 band 阈值。只有 utterance 边界而无逐词时间时标为 approximate；`unit` 另做合同检查 |
| `speaker.profiles[].pitch`、`speaker.profiles[].speaker_volume` | 诊断 | 独立 F0/RMS/LUFS 测量；未来专门语料 | M/D1；大多数集没有人工分档真值。冻结测量器和相对分档基准后只报一致性，不把测量结果称人工 GT，也不推断性别/年龄 |
| `speaker.asr_transcript` | Headline | AMI、AliMeeting、AISHELL-4、LibriSpeech/MLS、FLEURS | D0；reference transcript 只在 GT 侧，按语言冻结 normalization 后算 WER/CER。Common Voice prompt 只有经核听一致后才可升级为 transcript reference |
| `language_content.language` | Headline | FLEURS、MLS、AMI/AliMeeting/AISHELL-4、LibriSpeech | D0/D1；冻结 ISO/locale 映射；样本级标签不能评价片段级 code-switching。输入文本暴露 locale/language 时对该系统 mask=false |
| `language_content.word_count`、`language_content.repetition.has_repetition`、`language_content.repetition.repetition_count` | Headline | 上述可靠 reference transcript 集；Switchboard/Buckeye/DisfluencyBank 仅辅助核验 | D1；GT 必须对 reference transcript 运行与 schema 相同版本的 tokenizer、n-gram repetition 规则；预测端逐字段独立计分，不要求 `speaker.asr_transcript` 或其它依赖 known |
| `language_content.punctuation.punctuation_count`、`language_content.punctuation.has_terminal_punctuation`、`language_content.filler` | 条件/诊断 | 上述可靠 reference transcript 集；Switchboard/Buckeye/DisfluencyBank 仅辅助核验 | 保留用于规则审计和覆盖率检查，但不进入本轮能力分数、分母或雷达图 |
| `deterministic_derived.speaker_change_rate_per_min`、`deterministic_derived.filler_ratio` | Headline | GT 输入字段已知的样本 | D1；GT 严格按冻结公式生成；切换率使用对称归一化绝对误差，filler 比例使用连续容差公式，并报告 MAE/Spearman。系统依赖为 `null` 不屏蔽目标；依赖齐全时另查公式一致性 |

## 4. 统一 manifest 与转换方案

每个 corpus 先保留原始字段，再生成与模型输出物理隔离的 canonical JSONL。manifest 必须同时
记录源音频和系统实际输入；结构字段中的 duration/sample rate/channels 对后者评分。下面是
shared-audio 样本的最小示例：

```json
{
  "sample_id": "ami_EN2001a_window_00030",
  "corpus": "AMI",
  "track": "T1_natural_meeting",
  "split": "test",
  "native_id": "EN2001a_window_00030",
  "segment_start_sec": 0.0,
  "segment_end_sec": 15.8235,
  "input_mode": "shared_audio",
  "mask_policy": {
    "field_universe": "tag-schema-2026-09-01:non-descriptive",
    "gt_mask_default": false,
    "shared_audio_leakage_default": false,
    "native_input_leakage_default": true
  },
  "source_audio": {
    "path": "source/ami_EN2001a_window_00030_multichannel.wav",
    "sha256": "source-sha256",
    "duration_sec": 15.8235,
    "sample_rate_hz": 48000,
    "channels": 8
  },
  "system_input_audio": {
    "path": "inputs/ami_EN2001a_window_00030.wav",
    "sha256": "input-sha256",
    "duration_sec": 15.8235,
    "sample_rate_hz": 16000,
    "channels": 1
  },
  "system_inputs": {
    "sure-tagger": {"audio_sha256": "input-sha256", "text_sha256": null},
    "captioner": {"audio_sha256": "input-sha256", "text_sha256": null}
  },
  "leakage_mask_by_system": {
    "sure-tagger": {},
    "captioner": {}
  },
  "native_metadata": {
    "meeting_id": "EN2001a",
    "segments": [
      {
        "speaker": "A",
        "start_sec": 0.12,
        "end_sec": 4.0,
        "text": "...",
        "words": [{"w": "...", "start_sec": 0.12, "end_sec": 0.41}]
      },
      {
        "speaker": "B",
        "start_sec": 3.1,
        "end_sec": 6.0,
        "text": "...",
        "words": []
      },
      {
        "speaker": "A",
        "start_sec": 7.0,
        "end_sec": 10.0,
        "text": "...",
        "words": []
      },
      {
        "speaker": "B",
        "start_sec": 11.0,
        "end_sec": 15.0,
        "text": "...",
        "words": []
      }
    ],
    "rttm_path": null,
    "event_labels": null,
    "event_intervals": null,
    "snr_target_db": null,
    "rir_id": null,
    "rt60_sec": null,
    "c50_db": null
  },
  "ground_truth": {
    "values": {
      "basic_acoustic.duration_sec": 15.8235,
      "basic_acoustic.sample_rate_hz": 16000,
      "basic_acoustic.channels": 1,
      "basic_acoustic.nonspeech_ratio": null,
      "speaker.speaker_count": 2,
      "speaker.multi_speaker": true,
      "speaker.speaker_change_count": 3,
      "speaker.speaker_change": true,
      "speaker.overlap_ratio": 0.069876,
      "speaker.speaker_overlap": true,
      "speaker.asr_transcript": "...",
      "language_content.language": "en",
      "language_content.word_count": 31,
      "language_content.filler": 1
    },
    "gt_mask": {
      "basic_acoustic.duration_sec": true,
      "basic_acoustic.sample_rate_hz": true,
      "basic_acoustic.channels": true,
      "basic_acoustic.nonspeech_ratio": false,
      "speaker.speaker_count": true,
      "speaker.multi_speaker": true,
      "speaker.speaker_change_count": true,
      "speaker.speaker_change": true,
      "speaker.overlap_ratio": true,
      "speaker.speaker_overlap": true,
      "speaker.asr_transcript": true,
      "language_content.language": true,
      "language_content.word_count": true,
      "language_content.filler": true
    },
    "label_mask_by_field": {},
    "weak_positive_labels_by_field": {},
    "evaluable_labels_by_field": {},
    "evaluable_fields": [
      "basic_acoustic.duration_sec",
      "basic_acoustic.sample_rate_hz",
      "basic_acoustic.channels",
      "speaker.speaker_count",
      "speaker.multi_speaker",
      "speaker.speaker_change_count",
      "speaker.speaker_change",
      "speaker.overlap_ratio",
      "speaker.speaker_overlap",
      "speaker.asr_transcript",
      "language_content.language",
      "language_content.word_count",
      "language_content.filler"
    ],
    "strength_by_field": {
      "basic_acoustic.duration_sec": "D0",
      "basic_acoustic.sample_rate_hz": "D0",
      "basic_acoustic.channels": "D0",
      "speaker.speaker_count": "D1",
      "speaker.multi_speaker": "D1",
      "speaker.speaker_change_count": "D1",
      "speaker.speaker_change": "D1",
      "speaker.overlap_ratio": "D1",
      "speaker.speaker_overlap": "D1",
      "speaker.asr_transcript": "D0",
      "language_content.language": "D0",
      "language_content.word_count": "D1",
      "language_content.filler": "D1"
    },
    "source_by_field": {
      "basic_acoustic.duration_sec": "system_input_audio_header",
      "basic_acoustic.sample_rate_hz": "system_input_audio_header",
      "basic_acoustic.channels": "system_input_audio_header",
      "speaker.speaker_count": "AMI_words_and_segments_release",
      "speaker.multi_speaker": "AMI_words_and_segments_release",
      "speaker.speaker_change_count": "AMI_words_and_segments_release",
      "speaker.speaker_change": "AMI_words_and_segments_release",
      "speaker.overlap_ratio": "AMI_words_and_segments_release",
      "speaker.speaker_overlap": "AMI_words_and_segments_release",
      "speaker.asr_transcript": "AMI_reference_transcript",
      "language_content.language": "AMI_corpus_metadata",
      "language_content.word_count": "AMI_reference_transcript",
      "language_content.filler": "AMI_reference_transcript"
    },
    "rule_version_by_field": {
      "basic_acoustic.duration_sec": "audio-header-v1",
      "basic_acoustic.sample_rate_hz": "audio-header-v1",
      "basic_acoustic.channels": "audio-header-v1",
      "speaker.speaker_count": "speaker-gt-v0.1",
      "speaker.multi_speaker": "speaker-gt-v0.1",
      "speaker.speaker_change_count": "speaker-gt-v0.1",
      "speaker.speaker_change": "speaker-gt-v0.1",
      "speaker.overlap_ratio": "speaker-gt-v0.1",
      "speaker.speaker_overlap": "speaker-gt-v0.1",
      "speaker.asr_transcript": "AMI-native",
      "language_content.language": "language-map-v0.1",
      "language_content.word_count": "text-rules-v0.1",
      "language_content.filler": "text-rules-v0.1"
    }
  },
  "contract_expectations": {
    "common_reference": {
      "speaker.speaker_present": true,
      "deterministic_derived.speaker_change_rate_per_min": 11.375486,
      "deterministic_derived.filler_ratio": 0.032258
    },
    "by_system": {
      "sure-tagger": {"room_acoustic.far_field": null},
      "captioner": {}
    }
  },
  "license": {
    "dataset_url": "https://...",
    "audio_license": "see per-file license",
    "attribution": "..."
  },
  "annotation": {
    "status": "partial",
    "source": "AMI release manifest/annotation",
    "rule_version": "gt-rules-v0.1"
  }
}
```

`gt_mask` 是 GT 是否独立、完整且口径可评的权威值；`evaluable_fields` 是所有 `true` key 的物化
列表，校验器必须保证两者完全一致。`null` 通常对应 mask=false；已知的 `false`、`0`、`[]` 对应
mask=true。粗粒度/派生 GT 从 **GT 依赖**生成并开启自己的 mask；它不要求任一系统同时输出这些
依赖。示例中的 `contract_expectations` 只检查系统内部一致性：仅当该系统目标和依赖都已知时重算，
依赖缺失记为不可检查；目标字段仍独立进入能力分。

多标签数组/对象还需要 `label_mask_by_field`，例如分别记录 speech/singing/music 或六个 broad
noise class 是否有明确正负标注。只有目标值域全部已知时，整个字段的 `gt_mask` 才能为 true 并
计算普通 multilabel F1；只有部分 class 可评时按 class mask 报 per-class 指标；只有弱正例时放入
`weak_positive_labels_by_field`，仅进入 positive recall/coverage，不能制造负例。
`evaluable_labels_by_field` 是 class mask 中已知正/负类的物化列表，校验规则与
`evaluable_fields` 相同。native text 若泄漏多标签字段中的任一答案，默认保守地屏蔽整个字段；只有
实现并审计 label-level leakage mask 后才允许保留未泄漏类。

T3 的九个冻结目标类按上一节的闭世界例外全部物化 class mask：正映射为 positive，其余为
negative；因此不会写入 `weak_positive_labels_by_field`。这一处理是评测假设，不改变源数据集的
原生标注类型，也不把派生负例升级为人工 GT。

示例采用显式的稀疏 mask policy：缺失 `gt_mask` key 等价于 false；每个 `gt_mask=true` 字段都必须
具有 strength/source/rule-version provenance。shared-audio 在 input adapter
和 hash 审计通过后，缺失 leakage key 等价于 false；native-input 缺失 leakage key 则保守地等价于
true。runner 在评分前必须按冻结的 `field_universe` 展开并保存完整 dense mask。任何
`gt_mask=true` 但缺少 GT value、未知系统名、未知字段路径或 field-universe 版本不符都使 manifest
校验失败，不能依赖不同实现各自猜测缺省值。

转换时遵守以下规则：

1. **键和完整性**：用 `corpus + native_id + segment start/end` 形成稳定 `sample_id`，并按固定小数格式
   序列化边界；保存原始 manifest、下载 URL、SHA256、文件 license。禁止用目录遍历顺序对齐。
2. **源与输入分离**：保留 `source_audio`；另生成统一 `system_input_audio`，记录转码命令和 hash。
   duration/sample rate/channels 的期望值来自后者；房间、source stem 和原生多通道信息仍指向前者。
3. **切片**：有 native utterance/RTTM 时优先使用原边界；event clip 保留官方窗口。增加上下文时
   保存 `context_left/right_sec`，不得改写 GT 边界或让相邻 split 的内容泄漏。
4. **字段 mask**：每个 `gt_mask=true` 字段分别记录 value、strength、source 和 rule version，多标签
   字段再记 class mask。除 T3 已预注册的九类闭世界映射外，不能因为 corpus“通常有标注”就给
   全体样本开启 mask；弱事件正例和人数下界均按样本处理。T1 的 `silence_ratio` 仅对已冻结
   `ami-turn-complement-v1` 的 350 条样本开启 mask。
5. **D1 派生器**：冻结 speaker 去重/change/overlap、SAD、tokenizer、punctuation、filler、repetition、
   SNR 和 RIR 测量规则；派生中间值以 `gt_derived_*` 命名并保留日志。
6. **本体映射**：保留 DESED/FSD50K/AudioSet 原始 class，另存 `schema_event_map_vN`；三类
   `speech_music_events`、六类 `external_noise_type` 和 DASS 细类 mapping 分开版本化。T3 在前两组
   九类内执行“有正映射为正，否则为负”；源标注缺失、文件损坏或 crosswalk 解析失败仍须单列排除。
   语言另存 `language_map_vN`，DASS 细类不得套用 T3 的 broad-class 闭世界规则。
7. **双系统输入隔离**：分别生成 sure-tagger/captioner adapter。shared-audio 时 adapter 只能暴露同一
   音频；canonical 中的 transcript、event、RIR、stems、recipe 和 contract expectation 都不得发送。
8. **native input 审计**：若系统必须接收文本，另存文本来源/hash 和每系统 `leakage_mask`；完整字段
   分母为 `gt_mask && !leakage_mask`，class 指标使用对应 class mask，并改为非直接比较报告。
9. **划分**：按 speaker、meeting、YouTube source、Freesound uploader、RIR 和 noise source 分组后划分，
   防止同一说话人/原始录音出现在 train 和 test。

## 5. 建议的评测指标与分母

| 能力组 | 主指标 | 约束 |
| --- | --- | --- |
| 合同与完整性 | JSON/schema valid rate；header exact/tolerance；类型、值域、排序、唯一性、公式和空值传播 violation count | 单列，不进入能力得分。`duration_sec` 允许冻结的 frame-level tolerance；采样率/通道 exact |
| Speaker structure | `speaker_count` exact + MAE；`multi_speaker`、change、overlap 的 precision/recall/F1；`speaker_change_count` 和每分钟切换率使用对称归一化绝对误差并报告 exact/MAE；`overlap_ratio` MAE/Spearman | 当前 schema 没有 `who spoke when`，不能报告 DER；数值和布尔投影分别展示但能力聚合只计一次 |
| Non-speech | `nonspeech_ratio` 的连续 `score_rate`/MAE；`silence_segments` collar 后 interval IoU、coverage/false alarm（当前诊断） | 当前 T1 使用 `ami-turn-complement-v1` 官方 `silence_ratio` GT；T2/T3 没有该连续 GT，不进入此字段分母 |
| ASR/LID | 英语等空格语言 WER，中文 CER，其他语言按冻结 tokenizer 报 WER/CER；language accuracy/macro-F1 | `asr_transcript=null`、runtime failure 或非法文本按空 hypothesis 计全删除，并另报 valid-transcript rate；同时给 raw/normalized 口径，按语言、时长、重叠和 SNR 分层 |
| 词面能力 | `word_count` 使用对称归一化绝对误差；`repetition_count` 使用精确 2/1/0；重复布尔字段报告 F1/平衡 `score_rate`；`filler_ratio` 使用连续容差公式 | GT 从 reference transcript 规则物化，但预测端是独立 Headline 能力输出，不要求 ASR/dependency known；标点计数、句末标点和 filler 原始计数仅作诊断，不进入评分。与 ASR 高度相关时仍须在汇总层公布固定权重，避免重复计权 |
| Events/noise | T3 native-label closed-world 与受控集计算 clip-level multilabel micro/macro P/R/F1 和 per-class AP；`music_present` 单独算 boolean F1/accuracy；其他 weak PU 集只算 positive recall/coverage | 多标签与粗粒度字段分别计分；系统事件数组为 `null` 时仍可凭 `music_present` 获得粗粒度能力分。两者都已知时另做一致性检查 |
| SNR/room | SNR、RT60、C50 的 valid rate、MAE、median AE、bias、Spearman；按 requested 条件画 error curve | 连续指标用渲染后的 measured GT；requested 值仅用于分层。自然与 controlled 绝不混报 |
| Speaker profiles | speech-rate 数值误差/band F1；pitch/volume 测量一致性；profile coverage/unmatched rate | 逐 speaker 指标必须由独立 system timeline/artifact 按时间对齐 GT，matching cost 禁止使用待评分属性；否则只报 sample-level multiset/coverage，不 exact 比较 ID |

每个字段和 track 至少同时报告以下分母：

- `eligible_gt`：完整字段上为 `gt_mask=true` 且该系统 `leakage_mask=false` 的样本数；per-class
  指标改用该 class 的 known-positive/known-negative mask，weak-positive 另设分母。
- `explicit_negative_or_empty`：GT 为 `false`、`0` 或 `[]` 的已知样本数。
- `result_known`、`result_null`、`schema_invalid`、`runtime_failed`：系统端互斥状态。
- `metric_n`：实际进入某个数值计算的样本数。它不得替代 `eligible_gt` 出现在 headline。

分类/exact 任务中，eligible 样本上的 `null`/失败按 abstention/错误保留；count MAE 和连续指标同时报告
`valid_prediction_rate = metric_n / eligible_gt` 和有效输出上的误差，不能只报后者。两系统的直接差异
使用相同 eligible 样本做 paired bootstrap，按 meeting/speaker/source 分组重采样并给 95% CI。

最终报告按 `natural`、`public_semi_synthetic`、`controlled_render` 分三组，再按 speaker、ASR/LID、
词面、event、acoustics 展示。默认能力雷达使用 18 个字段级 `score_rate`；词面轴按
`(track, language, tokenizer_version, lexical_rule_version)` 兼容组先做等权宏平均，其余跨兼容
track/语言组同样保留 coverage，不用 `null` 替换 0，也不计算跨字段或跨 track 的单一 overall score。

## 6. 覆盖缺口与合成数据方案

自然开源集通常无法在同一音频上同时给出准确 SNR、RIR 派生 RT60/C50、完整事件负例、
多人 schedule 和逐说话人 profile。`dnsmos_*` 不属于本版评测范围，也不能由混音 recipe 生成 GT。
因此先生成并审计了 **2,000 条** controlled render 来源池，再按 `(speaker_count, noise_role,
room_type)` 联合分层等比例抽取 **350 条**进入正式 benchmark，用来补组合覆盖和边界条件；其
machine-audited 产物与自然源树分开保存。

合成流程如下：

1. 从 LibriSpeech/LibriTTS/MLS 的相对干净 speech stem 取 0--4 名说话人和可靠 transcript；AMI
   保留在 natural track，不当 dry stem。每个 speaker 使用独立 gain，已知 schedule 定义 count、
   turn、change、overlap 和 speech activity；0-speaker 样本专门覆盖明确无人声。
2. 从 OpenAIR、REVERB 或 RIRS_NOISES 选择 RIR；卷积后保留原 RIR，并用同一版本化算法计算
   RT60/C50。需要给样本级 room 字段开 mask 时，各 speech source 使用同一 RIR；若使用多个
   source-position RIR，则必须冻结聚合定义，否则 room 字段 mask=false。distance 和 mic condition
   可作分层元数据，但当前不写成 `far_field` GT。
3. 从 MUSAN、DEMAND 或许可兼容且已人工复核的 FSD50K/Freesound 子集加入噪声/音乐，按 SNR
   网格混音；用渲染后的 target/noise stems 复算 measured SNR。0-speaker 或 clean（无 noise stem）
   样本没有 schema 接受的有限 SNR GT，`snr_db` mask=false；`clean` 只作实验条件，不写 Infinity
   或任意上限。
4. 对每个样本保存 stems、混音矩阵、增益、RIR、SNR、采样率/通道和随机种子；
   event GT 只写实际加入且验收为 audible 的 source。onset/offset 保留作 provenance，但本轮
   转换为 clip-level schema 值；静默 stem 不算事件。
5. 用作 closed-world event GT 的每条 source stem 都要按完整 schema 类表做人工或经验证的全类审计，
   并把 known positive/negative/unknown 沿 class mask 传播到混音。只有文件夹标签或只确认加入事件
   时，其他类仍是 unknown。broad noise class 与 DASS fine label 分开保存；只有等价且人工确认的
   DASS crosswalk 才允许给 `noise_composition` 开 mask，不能从文件夹名猜测细类。
来源池使用 fractional-factorial 或分层抽样覆盖下列因素，正式版固定总量 350，不展开完整笛卡尔积。
按 speech speaker、noise/music source 和 RIR/room 联合做 disjoint split：

| 因素 | 建议档位 |
| --- | --- |
| speaker count | 0（少量合同边界）、1、2、3、4 |
| speaker change count | 0、1、5 |
| overlap ratio | 0、0.05、0.20、0.40 |
| SNR (dB) | clean、20、10、5、0、-5 |
| noise class | `animal`、`mechanical`、`nature`、`music`、`formless`、`channel_environment` |
| music role/pattern | foreground/background/mixed；brief/intermittent/continuous |
| measured RT60 (s) | 约 0.1、0.3、0.6、1.0（以 RIR 实测值为 GT） |
| microphone condition | near/far 只作 stress metadata；多通道时再取 center/left/right，不评当前 `far_field` |
| event | speech、singing、music 和映射后的噪声类；保存 onset/offset，但评 clip-level presence |

每个合成样本至少保存 `recipe_version`、`seed`、源样本 ID/hash、speaker schedule、source/event
区间、target/noise/mix stems、每路 RIR、distance/gain/position、requested 与 measured SNR、
RT60/C50、通道和许可证。推荐 sidecar 形态：

```json
{
  "sample_id": "syn_000001",
  "system_input_audio_sha256": "...",
  "recipe_version": "acoustic-bench-v0.1",
  "seed": 12345,
  "sources": [
    {"role": "speech", "id": "librispeech/...", "sha256": "...", "speaker_id": "src_a"},
    {"role": "noise", "id": "musan/music/...", "sha256": "...", "broad_class": "music"}
  ],
  "speaker_schedule": [{"speaker_id": "src_a", "start_sec": 0.0, "end_sec": 3.2}],
  "event_presence": ["speech", "music"],
  "requested": {"snr_db": 10.0, "overlap_ratio": 0.0},
  "measured": {"snr_db": 9.87, "rt60_sec": 0.31, "c50_db": 8.4},
  "rir": {"id": "openair/...", "sha256": "...", "distance_m": 1.2},
  "mic_condition": "near",
  "license_refs": ["..."],
  "annotation_source": "controlled rendering recipe",
  "gt_strength": "C",
  "derivation_version": "gt-rules-v0.1"
}
```

`requested` 只用于设计和分层；连续指标使用渲染后的 `measured`。测量、削波、响度、source
audibility 或 hash 验收失败的样本进入 rejected 清单，不能以请求档位代填。recipe 与 sidecar
不得送入任一被测系统。

合成数据适合验证确定性字段和边界条件（0/1 speaker、无/有 overlap、空数组、0 dB SNR、
极短非语音等），但不能替代自然录音的外部有效性；历史 T4 不再给当前 headline。

## 7. 授权与可复现检查清单

1. **YouTube 来源**：AudioSet、VoxConverse，以及附录中的 AudioCaps/部分 WavCaps 只提供
   URL/ID 或受限缓存；记录检索日期、可取回比例和 hash，不把无再分发权的音频打包进 benchmark。
2. **Freesound/多来源素材**：FSD50K、MUSAN，以及附录中的 Clotho/WavCaps 可能具有逐文件
   异质 license；按目标用途过滤冲突条款，保留作者、URL、license 和 attribution。
3. **挑战赛/LDC**：CHiME、REVERB、ICSI、VoiceBank-DEMAND、部分 Ali/AISHELL 版本有
   注册或研究协议；`open source code` 不等于 `open audio`。
4. **CC 条款**：CC BY 需要署名，NC 禁止商用，ND 禁止衍生，SA 要求同许可；把数据、
   annotation、ontology、RIR、脚本和渲染产物的许可证分开记录。
5. **版本冻结**：manifest、`gt_mask`、crosswalk、派生器、音频转换、input adapter、模型和
   评测脚本都存 SHA256/版本号；报告中注明下载失败、缺失标签、人工排除和 license 排除数。
6. **隐私与切分**：会议/对话数据按协议处理个人信息；按 speaker/meeting/source 分组切分，
   发布 benchmark 时优先发布 ID、hash、转换脚本和评测标注，而不是复制受限音频。
7. **训练污染**：记录两个系统已知训练集、公开模型卡和可比对的 source ID/hash；公开 test 不自动
   等于 unseen。发现精确或同源重叠时单列 contaminated 结果，不能静默并入 headline。

重建 benchmark 使用 `python3 compare/dataset/benchmark/build_benchmark_subset.py`；仅验收时追加 `--validate-only`。

## 8. 暂缓的描述性数据集

下列调研材料保留供第二阶段使用，但**不进入本轮 650 条核心预算、`gt_mask`、指标或排名**。
它们的 caption 也不得代替 transcript、事件精确标签、speaker 标注或房间物理 GT。

| 数据集 | 公开内容与价值 | 本轮暂缓原因 |
| --- | --- | --- |
| [Clotho](https://zenodo.org/records/3490684) | 15--30 s Freesound clip，每 clip 5 条人工 caption；常见 v1 为 5,929 clips/29,645 captions，v2 常见约 6,974 clips/34,870 captions，须以冻结 manifest 为准 | 多参考开放文本没有唯一结构答案；源 Freesound 音频还需逐文件核对 license |
| [AudioCaps](https://audiocaps.github.io/) | 约 50k 个 10 s AudioSet clip；训练多为单 caption，dev/test 有多参考 | 音频依赖 YouTube ID，存在失效、版权和地域差异；caption 中提到事件不等于完整事件 GT |
| [MusicCaps](https://github.com/google-research/musiccaps) | 约 5.5k 个 10 s 音乐片段及人工音乐描述，覆盖乐器、风格、节奏等 | 适合未来音乐 caption，不足以评当前结构字段；音频 URL 与 caption/metadata 的许可分离 |
| [WavCaps](https://github.com/XinhaoMei/WavCaps) | 约 400k 音频/文本对，汇集 Freesound、BBC、AudioSet、SoundBible 等来源 | 很多 caption 是自动或 LLM 生成，不是独立人工 GT；来源和 license 高度异质 |

若以后恢复 `descriptive.*`，应新建独立 track：冻结输出语言和输入模态，保留多个人工参考，
另做 factuality/unsupported-claim 人工审计，并继续用结构 GT 核查描述中的事实。该 track 不与本轮
结构能力分数相加。RAVDESS、IEMOCAP、CREMA-D 的情绪标注也只在 schema 新增 emotion/affect
结构字段后再考虑，不能塞入当前 speaker profile 或描述字段来增加“覆盖率”。

## 9. 官方入口索引（便于复核）

- AMI：<https://groups.inf.ed.ac.uk/ami/corpus/>
- ICSI：<https://www1.icsi.berkeley.edu/Speech/mr/>（AMI 项目镜像/说明页也可能提供下载入口）
- AliMeeting：<https://www.openslr.org/119/>
- AISHELL-4：<https://www.openslr.org/111/>
- CHiME-6：<https://github.com/chimechallenge/chime6>
- DIHARD III：<https://dihardchallenge.github.io/dihard3/>
- VoxConverse：<https://github.com/joonson/voxconverse>
- LibriSpeech / LibriTTS / MLS：<https://www.openslr.org/12/>、<https://www.openslr.org/60/>、<https://www.openslr.org/94/>
- Common Voice：<https://commonvoice.mozilla.org/datasets>
- FLEURS：<https://huggingface.co/datasets/google/fleurs>
- TED-LIUM：<https://www.openslr.org/51/>
- Clotho：<https://zenodo.org/records/3490684>
- AudioCaps：<https://audiocaps.github.io/>
- MusicCaps：<https://github.com/google-research/musiccaps>
- WavCaps：<https://github.com/XinhaoMei/WavCaps>
- FSD50K：<https://zenodo.org/records/4060432>
- AudioSet / ontology：<https://research.google.com/audioset/>、<https://github.com/audioset/ontology>
- ESC-50 / UrbanSound8K：<https://github.com/karolpiczak/ESC-50>、<https://urbansounddataset.weebly.com/urbansound8k.html>
- DESED/DCASE 2021：<https://dcase.community/challenge2021/task-sound-event-detection-in-domestic-environments>
- Microsoft DNS Challenge：<https://github.com/microsoft/DNS-Challenge>
- REVERB / OpenAIR：<https://reverb2014.audiolabs-erlangen.de/>、<https://www.openairlib.net/>
- WHAM/WHAMR：<https://wham.whisper.ai/>（代码/脚本镜像：<https://github.com/mpariente/wham>）
- VoiceBank-DEMAND：<https://datashare.ed.ac.uk/handle/10283/2791>
- LibriMix：<https://github.com/JorisCos/LibriMix>
- SMS-WSJ：<https://github.com/fgnt/sms_wsj>
- DEMAND：<https://zenodo.org/records/1227121>
- RIRS_NOISES：<https://www.openslr.org/28/>
- MUSAN：<https://www.openslr.org/17/>
- RAVDESS：<https://zenodo.org/records/1188976>
- Switchboard：<https://catalog.ldc.upenn.edu/LDC97S62>
- Buckeye：<https://buckeyecorpus.osu.edu/>
- DisfluencyBank / TalkBank：<https://talkbank.org/disfluency/>
- VCTK：<https://datashare.ed.ac.uk/handle/10283/3443>
- IEMOCAP：<https://sail.usc.edu/iemocap/>
- CREMA-D：<https://github.com/CheyneyComputerScience/CREMA-D>
