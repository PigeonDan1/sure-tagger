# sure-tagger

ASR dataset sample-level tagging toolkit. The project currently focuses on
acoustic, sound-field, speaker, and language-content tags from raw-only
sample inputs.

## Repository Layout

- `tagger/`: core schemas, tag tools, and pipelines.
- `docs/`: user-facing documentation, including the
  [tag definitions, methods, and JSON examples](docs/tags-and-methods.md).
- `scripts/`: runnable pipeline entry points.
- `tests/`: unit and smoke tests.
- `phase1_asr_samples/`: small local demo samples and manifests.
- `ami_en2001a_utterances/`: utterance-level AMI multi-speaker test manifest
  and locally generated audio cuts.
- `bridge/sure-tagger/`: language-layer prototype and related notes.
- `development.md`: current development constraints and public input/output
  schema.

## Git-ignored Local Artifacts

Large or machine-local files are intentionally not tracked:

- `.runtime/`: local Python/runtime environments.
- `models/`: third-party model repositories and checkpoints.
- `data`: shared dataset symlink.
- `outputs/`, `bridge/sure-tagger/outputs/`,
  `phase1_asr_samples/outputs/`,
  `ami_en2001a_utterances/outputs/`: generated run outputs.
- `ami_en2001a_utterances/audio/`: generated EN2001a utterance WAV files.
- Rec-RIR waveform artifacts are written under the selected output directory's
  `artifacts/rir/` subdirectory by default.
- Speaker v2 evidence, alignment, and fusion artifacts are written under
  `artifacts/speaker_v2/<row-key>/`.
- `api.txt`, `.env*`, `.codex/`: local secrets and private config.

## Quick Commands

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the sample-level tagging pipeline on the bundled sample manifest:

```bash
python3 scripts/run_tagger.py
```

Run the full pipeline for one sample only:

```bash
python3 scripts/run_tagger.py \
  --manifest phase2_asr_sample/manifest.jsonl \
  --output outputs/one_sample_tags.jsonl \
  --sample-id EN2001a_utterance_00000
```

Supplement selected tags into an existing tags-only output:

```bash
python3 scripts/run_tagger.py \
  --manifest phase2_asr_sample/manifest.jsonl \
  --input-tags outputs/phase2_full_pipeline_tags.jsonl \
  --output outputs/phase2_asr_patch.jsonl \
  --sample-id EN2001a_utterance_00000 \
  --only-tags speaker.asr_transcript \
  --missing-only
```

Compare Brouhaha C50 against Rec-RIR-derived C50 on the bundled manifest:

```bash
python3 scripts/run_c50_method_comparison.py
```

Run only the speaker stage with the default `quality-shadow` profile:

```bash
python3 scripts/run_tagger.py \
  --manifest phase1_asr_samples/manifest.jsonl \
  --output phase1_asr_samples/outputs/sample_tags.jsonl \
  --only-tags speaker \
  --speaker-profile quality-shadow
```

The main pipeline calls speaker v2 directly; there is no legacy MOSS enable gate,
native-metadata speaker route, or channel-activity fallback. Every speaker run
starts MOSS and NVIDIA Nemotron 3 Diarization together. Language routing
uses full-audio FireRed LID on the complete clip, matching the
`benchmark-v1-current.pdf` route. Chinese (`zh` or `zh-*`) and English
(`en` or `en-*`) then use MOSS for the published timeline; every other language,
including mixed or unknown language, uses Nemotron. Speaker-count, overlap, and
change decisions use the selected timeline. For ASR,
non-Chinese/English, mixed, and unknown languages always use Qwen3-ASR-0.6B.
Chinese/English uses FireRedASR2-AED when valid MOSS and Nemotron counts are
both at most one, and MOSS otherwise. A missing or invalid count does not form
the single-speaker consensus. Use `--speaker-segment-lid` to opt into the older
speaker-first longest-clean-segment vote, or `--speaker-segment-lid-shadow-only`
with that flag to record the segment proposal without using it for routing.
Speaker profiles are numbered by first speech onset on that selected timeline.
Their `gender` field uses Alice for Chinese and English, AfriSpeech for Swahili
and the configured African-language codes, and Griko for all other languages.
Single-speaker audio is sent to the classifier as a whole. For multi-speaker
audio, each speaker's non-overlap activity is cropped and concatenated in
memory; less than 1 second of clean speech publishes `gender: null`.
ECAPA and Brouhaha remain auxiliary evidence as configured by
`tagger/pipelines/speaker_evidence.py`; Sortformer and Pyannote are no longer
speaker-claim sources in the language-routed chain. `--speaker-profile lean-shadow`
selects the lower-cost profile. `--speaker-v2-skip-model-verification` only skips
pinned model asset hash checks; it does not download models or disable inference.

The active claim policy is `speaker_v2.claim_policy.20261008.1`. Its C/M/O/X
rules use `selection="language_routed"`; the resolver therefore requires the
sample's `language_route` object. A missing or unsupported route is an error,
and each rule has an empty fallback list. The route always records
`language_selected_source`: it uses `selection="language_route"` when that
source is usable, or `selection="none"`, a failure reason, and a `null` claim
when it is unavailable. It never tries the other branch. The
standalone speaker CLI exposes independent `--firered-asr-enable/disable`,
`--qwen3-asr-enable/disable`, and `--nemotron-enable/disable` controls.
`--openasr-enable/disable` remains only as a backward-compatible group alias.
The main pipeline accepts `--nemotron-model-dir`, `--nemotron-python`,
`--nemotron-device`, and `--nemotron-timeout-sec`; GPU inference is the default.

Model-backed tools require local model directories/checkpoints. General acoustic
tools read `tagger/local_config.py`; speaker v2 defaults are defined in
`tagger/pipelines/speaker_evidence.py`.

DASS uses the project-local CUDA runtime by default. Provision or verify it with
`bash scripts/setup_dass_gpu.sh`; use `--dass-use-cpu` only for an explicit CPU
debugging run.

The active other-language deployment is:

- `models/Qwen3-ASR-0.6B/` with
  `.runtime/qwen3_asr_0_6b_py311_torch280_cu128_v1/bin/python` for ASR.
- `models/speaker/nvidia/Nemotron-3-Diarization/model.safetensors` with
  `.runtime/nemotron3_diarization_py311_transformers519_torch280_v1/bin/python`
  for diarization.
- `models/speaker/alice_voice_gender_classifier/onnx-q8-v2/`,
  `models/speaker/afrispeech_gender_id/2026-09-11/`, and
  `models/speaker/griko_gender_cls_ecapa_voxceleb/2025-07-31/` for the
  language-routed profile gender field.

The NVIDIA model is pinned to
`nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e`.
The active adapter uses the local Transformers checkpoint, takes channel 0 of
multichannel audio, and resamples to mono 16 kHz. Model inference runs on GPU
by default. Exact files, SHA-256 values, runtime versions, and deployment
healthcheck results are recorded in
`models/speaker/nvidia/Nemotron-3-Diarization/deployment/deployment_manifest.json`.

The former OpenASR 0.1.40 combined adapter and its `.oasr` packs remain in the
tree only for historical artifact compatibility. They are not selected by the
current claim policy or the active tagging pipeline.

`language_content.topic` and its API/CLI integration are no longer part of
sure-tagger. A downstream language model may generate an open descriptive
`topic` phrase from the deterministic tags and `speaker.asr_transcript`.

See [the tag and method documentation](docs/tags-and-methods.md) for the public
fields, preprocessing, and JSON examples. Tool-specific setup details are kept
in `tagger/tools/basic_acoustic/README.md`,
`tagger/tools/sound_field_scene/README.md`,
`tagger/tools/speaker_v2/docs/direct_public_output_20260820.md`, and
`tagger/tools/language_content/README.md`.

## Speaker Pipeline

The `speaker` stage directly runs speaker v2 and publishes seven scalar fields:
`speaker_count`, derived `speaker_present`, `multi_speaker`,
`speaker_change_count`, `speaker_change`, `overlap_ratio`, and
`speaker_overlap`, plus the `profiles` array and `asr_transcript` string. Every
run executes MOSS and NVIDIA Nemotron 3 Diarization together. Full-audio
FireRed LID selects the route before claims are resolved. Chinese and English publish the MOSS
timeline; other, mixed, non-Latin, unknown, or failed-LID samples publish the
Nemotron timeline. Other, mixed, unknown, or failed-LID samples always use
Qwen3-ASR-0.6B for ASR. For Chinese/English, valid counts no greater than one
from both systems select FireRedASR2-AED; otherwise ASR uses MOSS. There is no
ASR failure fallback. None of these paths reads
`sample.text.transcript`. Selected outputs, both count decisions, language, and route reason
remain in the internal artifact under
`artifacts/speaker_v2/<row-key>/`. See
[`tagger/tools/speaker_v2/evalue/asr双路评测报告_20260831.md`](tagger/tools/speaker_v2/evalue/asr双路评测报告_20260831.md)
for the historical ASR comparison; its pre-2026-09-04 routing results do not
describe the current language-routed chain.
