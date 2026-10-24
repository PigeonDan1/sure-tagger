<p align="center"><img src="./assets/readme/three-layer-audio-metadata-tag.png" width="112" alt="SURE-Tagger icon: three layered audio metadata tags"></p>

<h1 align="center">SURE-Tagger</h1>

<h2 align="center">Making speech interpretable and computable to agents<br><sub>— through rich, semi-structured representations</sub></h2>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
<a href="#motivation"><img src="./assets/readme/nav/motivation.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>Motivation</ins></a><img src="./assets/readme/nav/spacer.png" width="20" height="1" alt=""><a href="#sure-tagger"><img src="./assets/readme/nav/sure-tagger.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>SURE-Tagger</ins></a><img src="./assets/readme/nav/spacer.png" width="74" height="1" alt=""><a href="#how-it-works"><img src="./assets/readme/nav/how-it-works.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>How&nbsp;it&nbsp;works</ins></a><img src="./assets/readme/nav/spacer.png" width="98" height="1" alt=""><a href="#usage"><img src="./assets/readme/nav/usage.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>Usage</ins></a><img src="./assets/readme/nav/spacer.png" width="28" height="1" alt="">
</p>

<p align="center">
<a href="#tag-bench"><img src="./assets/readme/nav/tag-bench.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>Tag-Bench</ins></a><img src="./assets/readme/nav/spacer.png" width="12" height="1" alt=""><a href="#system-performance"><img src="./assets/readme/nav/system-performance.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>System&nbsp;performance</ins></a><img src="./assets/readme/nav/spacer.png" width="12" height="1" alt=""><a href="#compositional-benchmark"><img src="./assets/readme/nav/compositional-benchmark.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>Compositional&nbsp;benchmark</ins></a><img src="./assets/readme/nav/spacer.png" width="12" height="1" alt=""><a href="#references"><img src="./assets/readme/nav/references.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>References</ins></a><img src="./assets/readme/nav/spacer.png" width="12" height="1" alt=""><a href="#license"><img src="./assets/readme/nav/license.svg" width="22" height="22" align="middle" alt="">&nbsp;<ins>License</ins></a>
</p>

SURE-Tagger turns speech evidence into queryable tags and evidence-linked descriptions.

https://github.com/user-attachments/assets/da8797a9-47d8-4c20-9781-eb336b8e4ba8

<p align="center"><sub>Introduction video · English narration</sub></p>

<a id="motivation"></a>

## Motivation

A free-form caption can summarize a recording, but it cannot tell whether an omitted property is absent or simply unsupported by evidence. Explicit fields, descriptions, and `null` make speech conditions easier to query and evaluate individually.

<p align="center"><img src="./assets/readme/paper-figure-1.png" width="100%" alt="Free-form captions omit attributes; structured tags distinguish known values from missing evidence"></p>

<p align="center"><sub>Structured tags make missing evidence explicit.</sub></p>

<a id="sure-tagger"></a>
<a id="demo"></a>

## SURE-Tagger · Audio to tags

Semi-structured tags pair explicit, queryable fields with short descriptions; `null` marks what the evidence cannot establish. Hear this [AMI meeting clip](https://me-chuan.github.io/sure-tagger/sure-tagger-main/demo/?lang=en&sample=ES2012d_utterance_00018#listen) and inspect its tags:

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

For the complete fields and input format, see the [tag guide](sure-tagger-main/docs/tags-and-methods.md). The [interactive demo](https://me-chuan.github.io/sure-tagger/) includes the five clips below; [run it locally](sure-tagger-main/demo/README.md), or use [`build_demo_data.py`](sure-tagger-main/scripts/build_demo_data.py) with your own audio and tags.

<a id="how-it-works"></a>

## How it works

Specialized observers collect acoustic, temporal, lexical, and speaker evidence. SURE-Tagger reconciles uncertainty and conflicts into typed fields and evidence-linked descriptions.

<p align="center"><a href="./assets/readme/how-it-works.png"><img src="./assets/readme/how-it-works.png" width="100%" alt="Specialized speech observers feed an evidence record, which is reconciled into structured tags for compositional evaluation"></a></p>

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

<a id="tag-bench"></a>

## Tag-Bench

Tag-Bench combines selected recordings from existing datasets with 500 controlled synthetic scenes. The released [LayeredSpeech-500](LayeredSpeech-500/README.md) pipeline follows frozen L1–L5 scene scripts to generate and validate the synthetic half.

<table width="100%">
<thead><tr><th width="520">Component</th><th width="850">Source or process</th><th width="230" align="right">Clips</th></tr></thead>
<tbody>
<tr><td>Controlled synthesis</td><td>LayeredSpeech-500 scene scripts and renderer</td><td align="right">500</td></tr>
<tr><td>Existing data · T1</td><td>AMI meetings</td><td align="right">350</td></tr>
<tr><td>Existing data · T2</td><td>FLEURS, LibriSpeech</td><td align="right">150</td></tr>
<tr><td>Existing data · T3</td><td>DESED, FSD50K</td><td align="right">150</td></tr>
<tr><td><strong>Total</strong></td><td><strong>Synthesis + existing data</strong></td><td align="right"><strong>1,150</strong></td></tr>
</tbody>
</table>

<p align="center"><img src="./assets/readme/tagbench-workflow.png" width="100%" alt="Attribute combinations guide controlled speech generation and validation alongside selected recordings from existing datasets"></p>

<p align="center"><sub>Controlled synthesis and existing recordings together form Tag-Bench.</sub></p>

### Generation agent

The [frozen scene scripts](LayeredSpeech-500/data/scene_scripts_v4.jsonl) specify language, dialogue, speakers, overlap, and acoustic conditions. [The runner](LayeredSpeech-500/scripts/reproduce.py) executes **prepare → base TTS → edit → derive → render → QC**, then exports mixed audio, speaker stems, RTTM, manifests, and ground truth. The released workflow runs these scripts without a live LLM planning step.

On Linux with Python 3.10–3.12, install the [release dependencies](LayeredSpeech-500/README.md#quick-start), then try the offline fixture:

```bash
cd LayeredSpeech-500
python scripts/verify_release.py
python scripts/reproduce.py --profile fixture --output runs/fixture
```

The fixture makes non-linguistic test audio; paper WAVs are not bundled. Full reproduction also needs CUDA, FFmpeg, separately licensed reference audio, RIRs, music, and pinned Dots models. Run `python scripts/reproduce.py --profile full --check`, then `python scripts/reproduce.py --profile full --gpus 0 --output runs/v4_multilingual`. See the [full reproduction guide](LayeredSpeech-500/docs/REPRODUCE.md) for setup and expected QC results.

Hear five Tag-Bench clips and compare SURE-Tagger, Captioner, and Qwen3-Omni predictions on the same audio.

<h3 align="center"><a href="https://me-chuan.github.io/sure-tagger/sure-tagger-main/demo/?lang=en#listen">▶ Listen here! ↗</a></h3>

<p align="center"><sub><a href="assets/readme/tagbench/COMPARISON.md">Same-audio comparison</a> · <a href="assets/readme/tagbench/ATTRIBUTION.md">Audio attribution</a></sub></p>

<a id="system-performance"></a>
<a id="paper-results"></a>

## System performance

On Tag-Bench, SURE-Tagger scores higher than Qwen3-Omni-Captioner on rich speech annotation. All methods use the same language model to construct the final schema. The overall score covers 29 structured and descriptive axes; higher is better.

<table width="100%">
<thead><tr><th width="830">Data</th><th width="385" align="right">SURE-Tagger</th><th width="385" align="right">Captioner</th></tr></thead>
<tbody>
<tr><td>Existing recordings</td><td align="right"><strong>76.28%</strong></td><td align="right">51.23%</td></tr>
<tr><td>Synthetic scenes</td><td align="right"><strong>78.88%</strong></td><td align="right">63.54%</td></tr>
<tr><td>Combined</td><td align="right"><strong>77.28%</strong></td><td align="right">58.20%</td></tr>
</tbody>
</table>

<p align="center"><a href="./assets/readme/paper-results-rps.png"><img src="./assets/readme/paper-results-rps.png" width="100%" alt="Attribute-level relative performance on existing and synthetic recordings for SURE-Tagger, Qwen3-Omni-Captioner, and Qwen3-Omni-Flash"></a></p>

<p align="center"><sub>Attribute-level scores are normalized to the best method within each data source.</sub></p>

<a id="compositional-benchmark"></a>
<a id="from-tags-to-analysis"></a>

## Compositional benchmark

The same tags can select conditions across datasets or group clips into speech scenes. We cluster 1,000 recordings from ten datasets with audio-side tags and k-medoids; dataset identity, language, transcripts, and ASR errors do not determine the groups.

### Scene clusters

The table shows representative clips from six groups in one frozen run. Each clip links to its audio, tags, and assignment; group names describe this run only.

<table width="100%">
<thead><tr><th width="290">Group · clips</th><th width="770">Representative clip</th><th width="540">Explore</th></tr></thead>
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

<details>
<summary>Run clustering on your own tags</summary>

Prepare `tags.jsonl`, `index_map.jsonl`, and `manifest.jsonl` as described in the [clustering guide](clustering/README.md). From the repository root:

```bash
export INPUT_ROOT=/path/to/multi-data
export CLUSTER_OUTPUT=/path/to/clustering-output

python3 clustering/scripts/cluster_scenes.py \
  --tags "$INPUT_ROOT/sampled-100-per-source/tags.jsonl" \
  --index-map "$INPUT_ROOT/sampled-100-per-source/index_map.jsonl" \
  --sampled-manifest "$INPUT_ROOT/sampled-100-per-source/manifest.jsonl" \
  --cluster-methods kmedoids --k-min 6 --k-max 12 --seeds 0,1,2 \
  --no-wer-filter --skip-diagnostics --output-dir "$CLUSTER_OUTPUT"
```

The script writes assignments, a cluster summary, and run metadata. After freezing the groups, [`error_homogeneity.py`](clustering/scripts/error_homogeneity.py) can compute WER/CER diagnostics; error rates do not enter clustering.

</details>

ASR error by scene (%; lower is better):

<table width="100%">
<thead><tr><th align="left">ASR model</th><th align="right">C00<br><sub>Mid-length, dry</sub></th><th align="right">C01<br><sub>Short, low SNR</sub></th><th align="right">C02<br><sub>Short, moderate reverb</sub></th><th align="right">C03<br><sub>Long, clean</sub></th><th align="right">C04<br><sub>Very short, sparse</sub></th><th align="right">C05<br><sub>Short, noisy + music</sub></th></tr></thead>
<tbody>
<tr><td>Parakeet-TDT</td><td align="right">15.47</td><td align="right">48.06</td><td align="right">14.84</td><td align="right">20.18</td><td align="right">100.00</td><td align="right">100.00</td></tr>
<tr><td>Qwen3-ASR-1.7B</td><td align="right">14.70</td><td align="right">21.53</td><td align="right">12.59</td><td align="right">5.12</td><td align="right">86.36</td><td align="right">17.26</td></tr>
<tr><td>MOSS-Transcribe</td><td align="right">12.50</td><td align="right">36.43</td><td align="right">12.83</td><td align="right">15.03</td><td align="right">85.71</td><td align="right">91.30</td></tr>
<tr><td>SenseVoiceSmall</td><td align="right">10.04</td><td align="right">25.00</td><td align="right">10.00</td><td align="right">13.67</td><td align="right">18.18</td><td align="right">34.78</td></tr>
<tr><td>Whisper large-v3 turbo</td><td align="right">17.82</td><td align="right">51.55</td><td align="right">18.35</td><td align="right">17.82</td><td align="right">104.55</td><td align="right">130.43</td></tr>
<tr><td>Granite Speech 4.1</td><td align="right">15.51</td><td align="right">24.81</td><td align="right">16.04</td><td align="right">22.06</td><td align="right">33.33</td><td align="right">73.91</td></tr>
<tr><td>FunASR-Flash</td><td align="right">13.90</td><td align="right">19.85</td><td align="right">13.03</td><td align="right">11.25</td><td align="right">86.36</td><td align="right">13.57</td></tr>
</tbody>
</table>

<sub>Scenes expose different recognition weaknesses. Evaluation coverage varies with each model's language support.</sub>

### Composed conditions and ASR errors

Tags such as short duration, low SNR, and speaker overlap can also be combined into targeted subsets. The heatmap compares ASR WER on a common cohort of 500 English recordings; small subsets are exploratory.

<p align="center"><a href="./assets/readme/asr-condition-heatmap-cropped.svg"><img src="./assets/readme/asr-condition-heatmap-cropped.svg" width="100%" alt="WER heatmap for Qwen, Whisper, and FunASR under individual and composed speech conditions"></a></p>

<p align="center"><sub>WER rises sharply for some combined conditions; colors show the error rate and dots identify each tag combination. Counts below the columns show subset size.</sub></p>

### Using scene results to improve Qwen ASR

Condition-level errors guide candidate selection for about one hour of Qwen3-ASR-1.7B fine-tuning. SURE-Tagger-guided pool selection lowers error from 5.50% to 4.71% ± 0.08%, a 14.3% relative reduction; the same strategy with Captioner-derived tags reaches 6.17% ± 1.02% (three sampling seeds).

<table width="100%">
<thead><tr><th width="560" align="left">Selection</th><th width="600" align="right">Error (%) · mean ± SD</th><th width="440" align="right">Δ vs. original</th></tr></thead>
<tbody>
<tr><td>Original · no fine-tuning</td><td align="right">5.50</td><td align="right">—</td></tr>
<tr><td>Random</td><td align="right">6.17 ± 0.34</td><td align="right">+12.3%</td></tr>
<tr><td><strong>Pool / SURE-Tagger</strong></td><td align="right"><strong>4.71 ± 0.08</strong></td><td align="right"><strong>−14.3%</strong></td></tr>
<tr><td>Pool / Captioner</td><td align="right">6.17 ± 1.02</td><td align="right">+12.4%</td></tr>
<tr><td>Eval / SURE-Tagger</td><td align="right">4.96 ± 0.27</td><td align="right">−9.7%</td></tr>
<tr><td>Eval / Captioner</td><td align="right">6.83 ± 0.61</td><td align="right">+24.4%</td></tr>
</tbody>
</table>

<sub>One-hour (3,600 s) selection budget; three sampling seeds. Error combines English WER and Chinese/Thai CER by reference units across the same 1,000-recording evaluation set.</sub>

<details>
<summary>How the fine-tuning data is selected</summary>

<table width="100%">
<thead><tr><th width="340" align="left">Step</th><th width="1260" align="left">Cluster-weighted selection</th></tr></thead>
<tbody>
<tr><td>1 · Score</td><td>Reuse evaluation clusters and errors (<code>Eval</code>) or cluster and score the candidate pool with the original ASR model (<code>Pool</code>).</td></tr>
<tr><td>2 · Assign</td><td>Assign each candidate to its nearest representation-space cluster medoid.</td></tr>
<tr><td>3 · Weight</td><td>Set the sampling weight to <code>(cluster error rate + 10⁻³)²</code>.</td></tr>
<tr><td>4 · Select</td><td>Sample without replacement up to the 3,600-second audio budget.</td></tr>
</tbody>
</table>

</details>

The repository's full evaluation launcher currently uses the existing-data subset; see its [composition](compare/docs/benchmark-650-dataset-composition.md) and [scoring rules](compare/docs/benchmark-v1-29-metrics.md).

<a id="references"></a>

## References

- **Interspeech 2026** · Jing Peng et al., [A Unified and Reproducible Experimentation Framework for Speech Understanding](https://www.isca-archive.org/interspeech_2026/peng26e_interspeech.pdf).
- **arXiv 2026** · Jing Peng et al., [SURE-EVAL: A Systematic and Unified Agentic Framework for Reproducible Evaluation](https://arxiv.org/abs/2609.39030).
- **Website** · [Open-Bench — Open Bench, Open Innovation](https://www.open-bench.net/home).

<a id="license"></a>

## License

The repository root has no `LICENSE` file, so no repository-wide source-code license has been declared. [LayeredSpeech-500's project-owned code](LayeredSpeech-500/LICENSE) is Apache-2.0. External models and datasets follow their own terms; see the [release data licenses](LayeredSpeech-500/docs/DATA_LICENSES.md) and [demo audio attribution](assets/readme/tagbench/ATTRIBUTION.md).
