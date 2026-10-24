# v4 多语种与语言变体计划

- 基于 v3 的 468 条记录扩展到 500 条，原 L2-L5 声学设计继承。
- 24 条中文样本加入中英 code-switch，24 条英文样本加入 repetition+filler。
- 新增 ar、es、fr、hi、ja、ko、ru、sw 各 2 条单语样本。
- 新增 16 条多语种样本，覆盖按说话人分语种和同一说话人跨轮次切换。
- filler 严格沿用 sure-tagger 当前英文词表；中文口语词不冒充 filler GT。
- 多语种样本的样本级 language 为 null，逐轮 language 和 language_spans 保留在合成元数据中。
- 参考音色：AMI、AliMeeting、AISHELL-3，以及 FLEURS 每种新增语言 2 个音色。
