# Captioner -> 完整 schema 单次 DeepSeek 抽取

这份 prompt 用于把 captioner 的一段原始文本转换成完整 schema 对象。输出包含
field universe 中的 35 个 structural 路径、`deterministic_derived` 和全部 12 个
`descriptive.*` 字段；当前只暂缓评测 descriptive，不删除或省略它们。

## 输入

每次请求只传一个样本的纯文本。用户消息就是 txt 文件全文，不传 JSON wrapper、文件名、
路径、音频或 header：

```text
<txt 文件的原始 UTF-8 内容>
```

纯文本中出现的 `9-second`、`stereo` 等是 captioner 的观测声称，不是音频真值；脚本不会
读取同名 mp3，也不会把文件名中的 `start_end` 数字当作时长。`sample_id` 不在模型输入中，
默认输出为 `null`；如需关联样本，可由调用方通过 CLI 在 API 返回后注入外部 ID。

## Prompt

直接使用同目录的 [`../eval/captioner_deepseek_extraction_prompt.txt`](../eval/captioner_deepseek_extraction_prompt.txt)
作为 `system` message。它要求模型在内部完成“通读观测 -> 逐字段取证 -> 否定/不确定性
处理 -> 跨字段比对 -> 派生公式 -> schema 校验”，但只返回最终 JSON，不返回思维链。

## API 请求建议

```json
{
  "model": "deepseek-v4-pro",
  "messages": [
    {"role": "system", "content": "<prompt 文件全文>"},
    {"role": "user", "content": "<txt 文件全文，且只有这一段文本>"}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 20000,
  "stream": false,
  "response_format": {"type": "json_object"},
  "thinking": {"type": "enabled"}
}
```

`thinking.type=enabled` 保留隐藏的逐字段核验/逻辑推理，符合本任务的质量优先要求；网关
当前 prompt 在该网关的完整 schema + 隐藏推理路径建议至少预留 20k token；本次 smoke 使用
`max_tokens=20000` 成功。若批量吞吐优先，可改为 `thinking.type=disabled` 并把上限降到
4096，但这会关闭隐藏推理，应该在运行摘要中记录。
用户消息不加入任何 `/no_think` 或其它前缀，以保证输入确实只有原文；思考模式只由请求的
`thinking` 开关控制。
同一输入在 enabled/disabled 两种路径的抽取结果可能不同；正式比较时应固定一种模式，并
保存脚本输出中的模型、模式、prompt SHA-256 和 usage，避免把两种运行结果混作同一系统。

网关若不支持 strict `json_schema` response format，不要依赖服务端约束；请求后在本地
检查 JSON、字段集合、类型/范围，并把可检查的跨字段公式单独记入合同报告。四个粗粒度/派生
字段会保留模型的独立输出，不再由后处理覆盖：

```text
speaker_present = (speaker_count > 0)
music_present = (speech_music_events 已知且包含 music)
speaker_change_rate_per_min = round(speaker_change_count * 60 / duration_sec, 6)
filler_ratio = round(filler / word_count, 6)
```

这四个字段都独立计分。caption 明确给出“存在说话人/音乐”或精确 rate/ratio 时，即使 count、
event set 或公式输入为 `null`，目标字段仍可为非 `null`。只有依赖和目标都已知时才检查公式一致性；
依赖缺失记为合同不可检查，不会清空目标或阻止其计分。模型返回的 `null` 也不会被后处理成
`false`、`0` 或 `[]`。

## 输出

输出顶层键为 `sample_id`、版本字段、七个结构分组和 `descriptive`。结构字段严格对应
当前 scorer；`descriptive` 保留在结果中（仅做类型、证据路径和结构校验），暂不进入 GT、指标和排名。不要将 `noise`/`audio_event`
直接放进 `speech_music_events`，也不要把 gender、age、emotion 等无法映射的字段塞进
`speaker.profiles`。

这次抽取的结果同时适合做结构恢复和描述生成诊断。由于原始 caption 多为摘要而非完整
ASR，`asr_transcript` 和词面统计应保持保守；不能把摘要或质量形容词伪装成精确
transcript、SNR、DNSMOS、RT60 或 C50，但可以在 descriptive 中忠实复述其定性描述和假设。

## 单样本 smoke test

已使用用户指定的
`0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2.txt` 运行一次
`deepseek-v4-pro`（`thinking=enabled`，`max_tokens=20000`），并在本地完成结构字段、
descriptive 类型、证据路径和派生公式校验后，将结果写入
[`../evaluation/captioner-v0/deepseek-text-single.json`](../evaluation/captioner-v0/deepseek-text-single.json)。
该历史输出已通过当时版本的 validator；使用的历史 prompt SHA-256 为
`3063216c8b5fab66bc019d437f891df69de05a503162b4b5a85f9943a940c62b`。当前已改为粗粒度/派生
字段独立输出和独立计分的新 prompt，其 SHA-256 为
`46714183a626c13dc0b4d92fbe645671947a09550140d9e0477d166035b481c3`；旧 smoke 输出不能视为
通过当前 prompt 的复测。输入 SHA-256 为
`61a45676cc2efc2f0fff9be47b2001559b38346439ab6f654ccd9cc8b15cdfe2`，API usage 为
`prompt_tokens=5339`、`completion_tokens=12001`、`reasoning_tokens=10812`。这里只使用
文本本身；文本声称的时长/立体声属于 captioner 观测，不能当作真实音频 header。这只是链路
smoke test，不代表已经对全部 3000 条样本批量抽取，也不代表 descriptive 已评测。

可复现命令（API 配置默认读取 `api/deepseek-api.txt`，密钥不会写入输出）：

```bash
python3 eval/extract_captioner_deepseek_once.py \
  --input-file /hpc_stor03/sjtu_home/weihan.chen/share/caption_pairs_3000/0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2.txt \
  --sample-id 0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2 \
  --thinking enabled --max-tokens 20000 --timeout 300 \
  --output evaluation/captioner-v0/deepseek-text-single.json
```

## 全量批处理

批量 runner 对每行输入只读取 `input_text`（也兼容 Qwen 原始结果中的
`caption`），不会把 `sample_id`、音频路径或其它 JSON 字段拼入 user message：

```bash
python3 eval/run_captioner_deepseek_batch.py \
  --input-manifest evaluation/benchmark-v1/captioner_deepseek_input.jsonl \
  --output-dir evaluation/benchmark-v1/captioner-deepseek \
  --workers 10 --retries 4 --thinking enabled
```

每个样本完成后原子写入 `items/` checkpoint；可安全重跑同一命令续跑。下游使用
`results.jsonl`（仅包含通过本地完整 schema 校验的预测），失败项见
`failures.jsonl`，运行配置和覆盖率见 `summary.json`。
