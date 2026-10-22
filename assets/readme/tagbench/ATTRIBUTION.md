# Audio Sources And Attribution

These five WAV files are byte-for-byte copies of selected clips from the
`compare/dataset/benchmark/demo-web/audio/` review fixture. The accompanying
`reference.json` files record SHA-256 digests and only the benchmark references
needed to interpret each example.

| Sample | Source | Source terms recorded in benchmark metadata |
| --- | --- | --- |
| `ES2012d_utterance_00018` | AMI Meeting Corpus, meeting `ES2012d` | No unified redistribution license is recorded in the local benchmark metadata; consult the AMI source terms. |
| `EN2001e_utterance_00071` | AMI Meeting Corpus, meeting `EN2001e` | No unified redistribution license is recorded in the local benchmark metadata; consult the AMI source terms. |
| `fleurs_t2_0347` | FLEURS, `ru_ru` | `CC-BY-4.0` |
| `t3_desed_0112` | DESED `synthetic21_train`, source `1098.wav` | `dataset-specific; see DESED/DCASE release terms` |
| `t3_fsd50k_0045` | FSD50K dev, source `644.wav` | `http://creativecommons.org/publicdomain/zero/1.0/` (CC0) |

The structural and descriptive predictions in `full_predictions.json` come
from the three 650-sample report exports under `compare/docs/report/result/`:

- `sure-tagger/result_with_id_descriptive.jsonl`
- `qwen-captioner/results_t1_t3.jsonl`
- `qwen3-omni-v2-multi-api12/final_results.jsonl`

The five per-sample `demo_questions.json` files come from
`compare/dataset/benchmark/demo-web/data/quiz.json`. They contain the original
L1/L2/L3/L5/D review fields, whereas `full_predictions.json` retains each
system's complete per-sample output. These predictions and audio files do not
grant a new license to the source datasets.

For DESED and FSD50K, the event references use a native-label closed-world
mapping. A negative target class under this mapping is not necessarily a
human-verified absence of that sound.
