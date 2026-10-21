# Tag-Bench Demo Assets

This directory contains two representative audio clips and the corresponding
Sure-Tagger / Captioner / Omni comparison records extracted from the local
benchmark review fixture.

| File | Source | License note |
| --- | --- | --- |
| `audio/fleurs_t2_0055.wav` | FLEURS, `cmn_hans_cn` | The benchmark metadata records CC-BY-4.0. Preserve the upstream attribution when redistributing. |
| `audio/librispeech_t2_0048.wav` | LibriSpeech clean | The benchmark metadata records CC-BY-4.0. Preserve the upstream attribution when redistributing. |
| `labels.json` | `compare/dataset/benchmark/demo-web/data/quiz.json` | Derived evaluation labels; A = Sure-Tagger, B = Captioner, C = Qwen3-Omni. |

The clips are copied from the current benchmark demo fixture without changing
their sample content. The full benchmark contains sources with heterogeneous
terms, including AMI, DESED, and FSD50K; those sources are intentionally not
included in this public README asset bundle.

For a larger interactive comparison, see the local `demo-web` fixture and the
project-page plan. Do not treat this compact bundle as a replacement for the
full benchmark or its ground-truth manifests.
