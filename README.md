<p align="center"><img src="./assets/readme/three-layer-audio-metadata-tag.png" width="112" alt="SURE-Tagger icon: three layered audio metadata tags"></p>

<h1 align="center">SURE-Tagger</h1>

<h2 align="center">Making speech interpretable to agents through rich, semi-structured representations.</h2>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
<a href="#motivation"><kbd>01</kbd> Motivation</a> · <a href="#demo"><kbd>02</kbd> Demo</a> · <a href="#from-tags-to-analysis"><kbd>03</kbd> Analysis</a> · <a href="#how-it-works"><kbd>04</kbd> How it works</a><br>
<a href="#tag-bench"><kbd>05</kbd> Tag-Bench</a> · <a href="#paper-results"><kbd>06</kbd> Results</a> · <a href="#usage"><kbd>07</kbd> Usage</a> · <a href="#license"><kbd>08</kbd> License</a>
</p>

SURE-Tagger turns speech evidence into queryable tags and evidence-linked descriptions.

https://github.com/user-attachments/assets/e2543c8b-6002-40fd-b9ba-c3a99d7cf16f

<p align="center"><sub>Introduction video · Chinese narration</sub></p>

<a id="motivation"></a>

## Motivation

A free-form caption can summarize a recording, but it cannot tell whether an omitted property is absent or simply unsupported by evidence. Explicit fields, descriptions, and `null` make speech conditions easier to query and evaluate individually.

<p align="center"><img src="./assets/readme/paper-figure-1.png" width="100%" alt="Free-form captions omit attributes; structured tags distinguish known values from missing evidence"></p>

<p align="center"><sub>Structured tags make missing evidence explicit.</sub></p>

<a id="demo"></a>

## Demo · Audio to tags

Semi-structured tags pair explicit, queryable fields with short descriptions; `null` marks what the evidence cannot establish. Hear this [AMI meeting clip](assets/readme/tagbench/samples/ES2012d_utterance_00018/audio.wav) and inspect its tags:

<table width="100%">
<thead><tr><th width="520">Evidence</th><th width="1080">Structured tags</th></tr></thead>
<tbody>
<tr><td><strong>Acoustic timing</strong><br><sub>WAV duration + speech regions</sub></td><td><code>basic_acoustic</code><br>Duration <strong>23.72 s</strong> · Non-speech ratio <strong>0.500</strong></td></tr>
<tr><td><strong>Sound events</strong><br><sub>FireRed AED</sub></td><td><code>sound_field_scene</code><br>Speech detected · Music <strong>false</strong></td></tr>
<tr><td><strong>Speaker + ASR</strong><br><sub>MOSS / Nemotron + ASR</sub></td><td><code>speaker</code><br><strong>4</strong> speakers · Overlap <strong>true</strong></td></tr>
<tr><td><strong>Language + text</strong><br><sub>FireRed LID + transcript</sub></td><td><code>language_content</code><br>English · <strong>45</strong> words</td></tr>
<tr><td><strong>Evidence-linked description</strong><br><sub>Transcript + speaker tags</sub></td><td><code>descriptive.caption</code><br>A short multi-speaker English conversation evaluates a device…</td></tr>
</tbody>
</table>

<sub>AMI reference: four speakers with overlapping turns.</sub>

<details>
<summary>View this clip's tag excerpt</summary>

Values from the [complete SURE-Tagger prediction](assets/readme/tagbench/samples/ES2012d_utterance_00018/full_predictions.json):

```json
{
  "basic_acoustic": {"duration_sec": 23.72, "nonspeech_ratio": 0.500422},
  "sound_field_scene": {"speech_music_events": ["speech"], "music_present": false},
  "speaker": {"speaker_count": 4, "speaker_overlap": true},
  "language_content": {"language": "en", "word_count": 45},
  "descriptive": {"topic": "Product/device evaluation: whether the device is flashy/fashionable and technologically innovative"}
}
```

</details>

For the complete fields and input format, see the [tag guide](sure-tagger-main/docs/tags-and-methods.md). The [interactive demo](sure-tagger-main/demo/index.html) includes the five clips below; [run it locally](sure-tagger-main/demo/README.md), or use [`build_demo_data.py`](sure-tagger-main/scripts/build_demo_data.py) with your own audio and tags.

<a id="from-tags-to-analysis"></a>

## From tags to analysis

Combine `speaker.speaker_count > 1` with `speaker.speaker_overlap = true` to retrieve comparable clips. In a separate [frozen scene-clustering run](clustering/README.md), audio tags alone formed the groups below; ASR WER/CER is assessed only after assignment.

<table width="100%">
<thead><tr><th width="18%">Group · clips</th><th width="48%">Representative clip</th><th width="34%">Explore</th></tr></thead>
<tbody>
<tr><td><code>C00</code> · 341</td><td>Long single-speaker English speech</td><td><a href="assets/readme/clustering-pool/C00/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C00/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C00/cluster_assignment.json">Group</a></td></tr>
<tr><td><code>C01</code> · 122</td><td>Short Mandarin speech · lower SNR</td><td><a href="assets/readme/clustering-pool/C01/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C01/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C01/cluster_assignment.json">Group</a></td></tr>
<tr><td><code>C02</code> · 286</td><td>Short single-speaker Mandarin reading</td><td><a href="assets/readme/clustering-pool/C02/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C02/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C02/cluster_assignment.json">Group</a></td></tr>
<tr><td><code>C03</code> · 196</td><td>Long single-speaker English speech · high SNR</td><td><a href="assets/readme/clustering-pool/C03/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C03/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C03/cluster_assignment.json">Group</a></td></tr>
<tr><td><code>C04</code> · 17</td><td>Very short boundary clip · conflicting speech evidence</td><td><a href="assets/readme/clustering-pool/C04/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C04/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C04/cluster_assignment.json">Group</a></td></tr>
<tr><td><code>C05</code> · 38</td><td>Thai speech with music · low SNR</td><td><a href="assets/readme/clustering-pool/C05/audio.wav">Audio</a> · <a href="assets/readme/clustering-pool/C05/tags.json">Tags</a> · <a href="assets/readme/clustering-pool/C05/cluster_assignment.json">Group</a></td></tr>
</tbody>
<tfoot><tr><td><strong>Total · 1,000</strong></td><td colspan="2">Six tag-derived scene groups</td></tr></tfoot>
</table>

<sub>These IDs belong to one frozen run, not universal scene labels. The clips illustrate grouping rather than ASR scores. <a href="assets/readme/clustering-pool/README.md">Sample notes</a> · <a href="assets/readme/clustering-pool/source_run.json">Run provenance</a> · <a href="assets/readme/clustering-pool/ATTRIBUTION.md">Audio attribution</a></sub>

<a id="how-it-works"></a>

## How it works

Specialized observers collect acoustic, temporal, lexical, and speaker evidence. SURE-Tagger reconciles uncertainty and conflicts into typed fields and evidence-linked descriptions.

<p align="center"><a href="./assets/readme/how-it-works.png"><img src="./assets/readme/how-it-works.png" width="100%" alt="Specialized speech observers feed an evidence record, which is reconciled into structured tags for compositional evaluation"></a></p>

<a id="tag-bench"></a>

## Tag-Bench

Tag-Bench combines selected recordings from existing datasets with controlled synthetic scenarios. For the latter, an agent chooses attribute combinations, schedules speech generation and acoustic processing, then validates the resulting clips.

<table width="100%">
<thead><tr><th width="520">Component</th><th width="850">Source or process</th><th width="230" align="right">Clips</th></tr></thead>
<tbody>
<tr><td>Controlled synthesis</td><td>Agent-orchestrated generation and validation</td><td align="right">500</td></tr>
<tr><td>Existing data · T1</td><td>AMI meetings</td><td align="right">350</td></tr>
<tr><td>Existing data · T2</td><td>FLEURS, LibriSpeech</td><td align="right">150</td></tr>
<tr><td>Existing data · T3</td><td>DESED, FSD50K</td><td align="right">150</td></tr>
<tr><td><strong>Total</strong></td><td><strong>Synthesis + existing data</strong></td><td align="right"><strong>1,150</strong></td></tr>
</tbody>
</table>

<p align="center"><img src="./assets/readme/tagbench-workflow.png" width="100%" alt="Attribute combinations guide controlled speech generation and validation alongside selected recordings from existing datasets"></p>

<p align="center"><sub>Controlled synthesis and existing recordings together form Tag-Bench.</sub></p>

Explore five Tag-Bench clips with complete same-audio predictions from SURE-Tagger, Captioner, and Qwen3-Omni:

<table width="100%">
<thead><tr><th width="350">Clip</th><th width="570">Listen for</th><th width="680">Files</th></tr></thead>
<tbody>
<tr><td>AMI · <code>ES2012d</code></td><td>Four speakers and overlap</td><td><a href="assets/readme/tagbench/samples/ES2012d_utterance_00018/audio.wav">Listen</a> · <a href="assets/readme/tagbench/samples/ES2012d_utterance_00018/full_predictions.json">Full predictions</a></td></tr>
<tr><td>AMI · <code>EN2001e</code></td><td>Two speakers and brief overlap</td><td><a href="assets/readme/tagbench/samples/EN2001e_utterance_00071/audio.wav">Listen</a> · <a href="assets/readme/tagbench/samples/EN2001e_utterance_00071/full_predictions.json">Full predictions</a></td></tr>
<tr><td>FLEURS · Russian</td><td>Language and speech content</td><td><a href="assets/readme/tagbench/samples/fleurs_t2_0347/audio.wav">Listen</a> · <a href="assets/readme/tagbench/samples/fleurs_t2_0347/full_predictions.json">Full predictions</a></td></tr>
<tr><td>DESED</td><td>Speech with a mechanical event</td><td><a href="assets/readme/tagbench/samples/t3_desed_0112/audio.wav">Listen</a> · <a href="assets/readme/tagbench/samples/t3_desed_0112/full_predictions.json">Full predictions</a></td></tr>
<tr><td>FSD50K</td><td>Music event missed by the tagger</td><td><a href="assets/readme/tagbench/samples/t3_fsd50k_0045/audio.wav">Listen</a> · <a href="assets/readme/tagbench/samples/t3_fsd50k_0045/full_predictions.json">Full predictions</a></td></tr>
</tbody>
</table>

[All five samples and references](assets/readme/tagbench/README.md) · [Same-audio comparison](assets/readme/tagbench/COMPARISON.md) · [Audio attribution](assets/readme/tagbench/ATTRIBUTION.md)

The repository's full evaluation launcher currently uses the existing-data subset; see its [composition](compare/docs/benchmark-650-dataset-composition.md) and [scoring rules](compare/docs/benchmark-v1-29-metrics.md).

<a id="paper-results"></a>

## System performance

The radar plots compare attribute-level relative performance on existing and synthetic recordings. Each axis is normalized to its best method within that source; SURE-Tagger performs strongest across most speaker, temporal, and acoustic attributes.

<p align="center"><a href="./assets/readme/paper-results-rps.png"><img src="./assets/readme/paper-results-rps.png" width="100%" alt="Relative performance scores across speech attributes for SURE-Tagger and two Qwen3-Omni baselines on existing and synthetic recordings"></a></p>

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

<table width="100%">
<thead><tr><th width="590">Evaluation scope</th><th width="200" align="right">Clips</th><th width="810">Setting</th></tr></thead>
<tbody>
<tr><td>Existing-data structural benchmark</td><td align="right">650</td><td>Default inputs</td></tr>
<tr><td>Original frozen descriptive ground truth</td><td align="right">648</td><td><code>ALLOW_FROZEN_GT_EXCLUSIONS=1</code> for the predefined missing clips only</td></tr>
</tbody>
</table>

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
