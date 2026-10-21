<p align="center"><img src="./assets/readme/mark.svg" width="76" alt="SURE-Tagger icon: a waveform connected to structured tags"></p>

<h1 align="center">SURE-Tagger</h1>

<h2 align="center">Help agents understand speech</h2>

<p align="center"><strong>From audio evidence to structured tags and compositional evaluation</strong></p>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><a href="#motivation">Motivation</a> · <a href="#demo">Demo</a> · <a href="#tag-bench">Tag-Bench</a> · <a href="#usage">Usage</a> · <a href="#license">License</a></p>

SURE-Tagger turns evidence from specialized speech models into queryable tags and descriptions. It keeps `null` when evidence is insufficient. The research explores how these representations support evaluation across datasets and targeted data selection; this repository provides tagging, description, and scoring entry points.

<p align="center"><a href="./assets/readme/video/suretagger-intro-zh.mp4"><img src="./assets/readme/video/suretagger-intro-zh-poster.jpg" width="100%" alt="Watch the SURE-Tagger introduction video in Chinese"></a></p>

<p align="center"><sub><a href="./assets/readme/video/suretagger-intro-zh.mp4">Watch the introduction video</a> · Chinese narration</sub></p>

<a id="motivation"></a>

## Motivation

A free-form caption can summarize a recording, but it cannot tell whether an omitted property is absent or simply unsupported by evidence. Explicit fields, descriptions, and `null` make speech conditions easier to query and evaluate individually.

<p align="center"><img src="./assets/readme/paper-figure-1.png" width="100%" alt="Paper Figure 1: free-form captions omit properties, while structured speech representations distinguish values from unknowns"></p>

<p align="center"><sub>Paper Figure 1. The diagram shows selected concepts; the actual CLI fields appear in the demo below.</sub></p>

<a id="demo"></a>

## Demo · Audio to tags

Given audio and a JSONL manifest, the tagger writes tags-only JSONL. This excerpt comes from the [tag documentation](sure-tagger-main/docs/tags-and-methods.md#5-输出-json-示例):

```json
{
  "basic_acoustic": {"duration_sec": 15.963062, "nonspeech_ratio": 0.674248},
  "sound_field_scene": {"speech_music_events": ["singing", "music"], "music_present": true},
  "speaker": {"speaker_count": null, "asr_transcript": "Good morning. Let us begin the meeting."},
  "language_content": {"language": "en", "word_count": 12}
}
```

Two attributed Tag-Bench clips let you hear the input and inspect the same-audio comparison across SURE-Tagger, Captioner, and Qwen3-Omni:

| Clip | What to listen for | Files |
| --- | --- | --- |
| Mandarin · FLEURS | Single-speaker reading; quiet background | [Listen / download](assets/readme/tagbench/audio/fleurs_t2_0055.wav) |
| English · LibriSpeech | Single-speaker reading; no labeled external noise | [Listen / download](assets/readme/tagbench/audio/librispeech_t2_0048.wav) |

[Compare the three descriptions](assets/readme/tagbench/COMPARISON.md) · [Machine-readable labels](assets/readme/tagbench/labels.json) · [Audio attribution](assets/readme/tagbench/ATTRIBUTION.md)

For the complete fields and input format, see the [tag guide](sure-tagger-main/docs/tags-and-methods.md). The [interactive demo](sure-tagger-main/demo/index.html) needs your own audio and `demo/data.js`; [`build_demo_data.py`](sure-tagger-main/scripts/build_demo_data.py) prepares that data.

<a id="tag-bench"></a>

## Tag-Bench

Tag-Bench combines selected recordings from existing datasets with controlled synthetic scenarios. For the latter, an agent chooses attribute combinations, schedules speech generation and acoustic processing, then validates the resulting clips.

| Component | Source or process | Clips |
| --- | --- | ---: |
| Controlled synthesis | Agent-orchestrated generation and validation | 500 |
| Existing data · T1 | AMI meetings | 350 |
| Existing data · T2 | FLEURS, LibriSpeech | 150 |
| Existing data · T3 | DESED, FSD50K | 150 |
| **Total** | **Synthesis + existing data** | **1,150** |

<p align="center"><img src="./assets/readme/tagbench-workflow.png" width="100%" alt="Paper Figure 3: select attribute combinations, generate and validate speech, then combine it with existing recordings for Tag-Bench"></p>

<p align="center"><sub>Paper Figure 3. The repository's full evaluation launcher currently uses the existing-data subset; see its <a href="compare/docs/benchmark-650-dataset-composition.md">composition</a> and <a href="compare/docs/benchmark-v1-29-metrics.md">scoring rules</a>.</sub></p>

<a id="usage"></a>

## Usage

Prepare local model weights, a Python environment, and your own manifest as described in the [tagger guide](sure-tagger-main/README.md). Then run from the tagger directory:

```bash
cd sure-tagger-main
python3 scripts/run_tagger.py \
  --manifest /path/to/manifest.jsonl \
  --output /path/to/tags.jsonl
```

<details>
<summary>Full evaluation pipeline and test commands</summary>

The full launcher needs a benchmark, model-accessible runtime audio, descriptive ground truth, and external API credentials. See the [launcher](compare/eval/run_sure_tagger_pipeline.sh) for paths and GPU configuration. `--check-inputs` does not call models or APIs, but still requires the credential file to be readable.

```bash
# From the repository root
export PYTHON_BIN=/path/to/python
export BENCHMARK_ROOT=/path/to/benchmark
export RUNTIME_ROOT=/path/to/runtime
export GT_ROOT=/path/to/descriptive_gt
export DEEPSEEK_API_FILE=/private/deepseek-api.txt
export OUTPUT_ROOT=/path/to/new_output

bash compare/eval/run_sure_tagger_pipeline_1.sh --check-inputs
bash compare/eval/run_sure_tagger_pipeline_1.sh
# Eight GPUs: bash compare/eval/run_sure_tagger_pipeline_8.sh
```

| Evaluation scope | Clips | Setting |
| --- | ---: | --- |
| Existing-data structural benchmark | 650 | Default inputs |
| Original frozen descriptive ground truth | 648 | `ALLOW_FROZEN_GT_EXCLUSIONS=1` for the predefined missing clips only |

Use separate `OUTPUT_ROOT` / `STAGE_ROOT` directories for one-GPU and eight-GPU runs.

```bash
cd sure-tagger-main
python3 -m unittest discover -s tests
cd ..
python3 -m unittest discover -s compare/eval/tests
python3 compare/eval/test_sure_tagger_pipeline_launchers.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s clustering/tests -p 'test_*.py' -v
```

</details>

Explore the [tagger](sure-tagger-main/README.md), [scoring and comparison](compare/docs/benchmark-v1-29-metrics.md), and standalone [k-medoids scene clustering](clustering/README.md).

<a id="license"></a>

## License

The repository does not currently include a `LICENSE` file, so no source-code license has been declared. External models and datasets follow their own terms; see the [demo audio attribution](assets/readme/tagbench/ATTRIBUTION.md).
