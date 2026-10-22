# Tag-Bench Demo-Web Samples

Five samples selected from the 20-audio `demo-web` review fixture. For every
sample, `audio.wav` is the unchanged source clip, `full_predictions.json`
contains the complete SURE-Tagger, Captioner, and Qwen3-Omni outputs,
`demo_questions.json` retains all five L1/L2/L3/L5/D review records, and
`reference.json` records available benchmark references and the audio SHA-256.

| Sample | Source | Focus | Audio | Complete labels | Review questions | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| `ES2012d_utterance_00018` | AMI | Four speakers and overlap | [WAV](samples/ES2012d_utterance_00018/audio.wav) | [JSON](samples/ES2012d_utterance_00018/full_predictions.json) | [JSON](samples/ES2012d_utterance_00018/demo_questions.json) | [JSON](samples/ES2012d_utterance_00018/reference.json) |
| `EN2001e_utterance_00071` | AMI | Two speakers and brief overlap | [WAV](samples/EN2001e_utterance_00071/audio.wav) | [JSON](samples/EN2001e_utterance_00071/full_predictions.json) | [JSON](samples/EN2001e_utterance_00071/demo_questions.json) | [JSON](samples/EN2001e_utterance_00071/reference.json) |
| `fleurs_t2_0347` | FLEURS | Russian speech and language tags | [WAV](samples/fleurs_t2_0347/audio.wav) | [JSON](samples/fleurs_t2_0347/full_predictions.json) | [JSON](samples/fleurs_t2_0347/demo_questions.json) | [JSON](samples/fleurs_t2_0347/reference.json) |
| `t3_desed_0112` | DESED | Speech alongside a mechanical event | [WAV](samples/t3_desed_0112/audio.wav) | [JSON](samples/t3_desed_0112/full_predictions.json) | [JSON](samples/t3_desed_0112/demo_questions.json) | [JSON](samples/t3_desed_0112/reference.json) |
| `t3_fsd50k_0045` | FSD50K | Short music event | [WAV](samples/t3_fsd50k_0045/audio.wav) | [JSON](samples/t3_fsd50k_0045/full_predictions.json) | [JSON](samples/t3_fsd50k_0045/demo_questions.json) | [JSON](samples/t3_fsd50k_0045/reference.json) |

[Comparison notes](COMPARISON.md) · [Source attribution and data terms](ATTRIBUTION.md)

The three prediction objects are keyed by `sure_tagger`, `captioner`, and
`qwen3_omni`. They retain the source report's per-sample schema, including
explicit `null` fields. The review questions are a separate field selection;
their `audio` paths retain the original `demo-web` layout, so use the WAV links
above for these packaged copies. They are not the complete prediction objects.
Available references differ by
track, and the T3 closed-world event labels may contain unverified negatives.
