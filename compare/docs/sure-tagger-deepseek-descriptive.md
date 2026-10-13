# sure-tagger v1 -> 语言核验与描述性标签单次 DeepSeek 抽取

这条链路接在 sure-tagger 的 tags-only 输出之后。一次 DeepSeek 调用生成 `language_verification` 和全部 12 个 `descriptive.*` 字段。脚本只允许语言核验覆盖 `language_content.language`，其他结构性字段作为只读观测保留。

当前仅保留提示词 `v1`，批处理默认使用它，也可显式传入 `--prompt-version v1`。2026-10-09 已清理后续提示词实验、专用脚本和结果目录；底层 `speaker_v2` 是当前音频工具模块，与提示词版本无关。

## Prompt 与输入

Prompt 文件为 [`../eval/sure_tagger_deepseek_language_descriptive_prompt.txt`](../eval/sure_tagger_deepseek_language_descriptive_prompt.txt)。每次请求只处理一个 sure-tagger JSON/JSONL 行，user message 是该行七组结构性 JSON 的紧凑表示，不传音频、caption、文件名或其它样本。旧的 descriptive-only prompt 仍供 streaming 入口及历史结果复现使用。

模型必须在内部执行“完整通读观测 -> 语言核验 -> 逐字段取证 -> 跨字段逻辑比对 -> 区分事实/综合印象/假设 -> 逐键校验”的隐式思考流程；不返回 chain-of-thought，只返回语言核验和描述性字段。`evidence_tags` 只能引用当前样本非 null 的 35 个 structural exact path。

## 请求与拼接

API 配置默认读取 [`../api/deepseek-api.txt`](../api/deepseek-api.txt)，默认使用 `thinking=enabled`、`temperature=0`、`max_tokens=20000`。除语言核验授权的 `language_content.language` 外，脚本不重算或修改结构性字段；`speaker_present`、`music_present` 和 `deterministic_derived` 也原样保留。

```bash
python3 eval/extract_sure_tagger_deepseek_once.py \
  --input-tags /hpc_stor03/sjtu_home/weihan.chen/share/sure-tagger-main/outputs/smoke_structural_complete.jsonl \
  --sample-id 0.000_10.049_obj_03dce036-8982-4663-b85f-6719b7a690a2 \
  --thinking enabled \
  --max-tokens 20000 \
  --timeout 300 \
  --output evaluation/sure-tagger-v0/descriptive-single.json
```

## 全量批处理

批量 runner 支持每行 `{sample_id, input_text}`，其中 `input_text` 可以是结构性
JSON 字符串；也支持直接提供 `tags` 或七个 structural 分组。发送前 runner
只重建七组 structural JSON，因此 `sample_id`、音频路径、GT、caption 和其它
metadata 不会进入 DeepSeek user message：

```bash
python3 eval/run_sure_tagger_deepseek_batch.py \
  --input-manifest evaluation/benchmark-v1/sure_tagger_deepseek_input.jsonl \
  --output-dir evaluation/benchmark-v1/sure-tagger-deepseek \
  --workers 10 --retries 4 --thinking enabled
```

`items/` 保存逐样本原子 checkpoint；`results.jsonl` 只包含 schema 校验通过的
完整 prediction envelope，`failures.jsonl` 保存失败原因，`summary.json` 记录
成功/失败覆盖率和请求配置。中断后重复同一命令即可续跑。

DeepSeek 返回值为含 `language_verification` 和 `descriptive` 的对象；脚本也兼容旧的 12 键 descriptive-only 对象及仅含 `descriptive` 的包装。语言核验为 null 时沿用上游语言，非 null 时按校验规则覆盖，并写入独立的语言核验记录。脚本随后执行描述性字段类型、profile ID、confidence 和 evidence path 校验，并写出：

```json
{
  "sample_id": "调用方注入的样本 ID",
  "schema_version": "structural-prediction-v0.1",
  "field_universe_version": "structural-v0.1",
  "basic_acoustic": "原 sure-tagger 值",
  "audio_quality": "原 sure-tagger 值",
  "room_acoustic": "原 sure-tagger 值",
  "sound_field_scene": "原 sure-tagger 值",
  "speaker": "原 sure-tagger 值",
  "language_content": "仅 language 可由语言核验覆盖，其余保留原值",
  "deterministic_derived": "原 sure-tagger 值",
  "descriptive": "DeepSeek 返回的 12 个字段"
}
```

`descriptive.*` 当前保留但暂不进入 structural GT、指标或排名。模型失败、响应不合法或证据不足时应终止该行并保留原结构性输出，不用描述性文本反向补齐结构性 null。
