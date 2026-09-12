# Sound-Field And Scene Tools

## FireRed AED Event Detection

The FireRed AED adapter uses the official non-streaming AED model at
`models/FireRedVAD/pretrained_models/FireRedVAD/AED` in the configured
FireRed Python runtime. The upstream model recognizes exactly three overlapping
event classes: `speech`, `singing`, and `music`. This registered tool publishes
the detected class names and a backward-compatible music decision.

The public schema exposes detected event names and keeps the existing music
boolean for compatibility:

| FireRed AED result | Public tag |
| --- | --- |
| Detected classes in fixed `speech`, `singing`, `music` order | `sound_field_scene.speech_music_events` |
| No detected class | `sound_field_scene.speech_music_events = []` |
| `music` segments whose event ratio meets `min_music_ratio` | `sound_field_scene.music_present = true` |
| No `music` segment, or only segments below the ratio floor | `sound_field_scene.music_present = false` |
| Model, dependency, inference, or validation failure | Both fields are `null` |

`singing` and `music` participate in `speech_music_events` only when their
event ratios meet the config floors `min_singing_ratio` / `min_music_ratio`
(both default `0.10`; CLI `--firered-aed-min-singing-ratio` /
`--firered-aed-min-music-ratio`). Calibrated on caption_pairs_3000
(2026-08-25): frame-level singing/music false positives on speech produce
short segments whose ratio stays below `0.10`, while real singing or music
occupies far more of the clip. The gates keep `speech_music_events` and
`music_present` mutually consistent; gated segments remain visible in the
internal evidence (`event_segments`, `event_ratios`, `event_gates`).

Validated `speech`/`singing`/`music` segments and upstream thresholded-frame
ratios are retained as internal tool evidence and do not enter the tags-only
public output. General background-sound composition is owned by the DASS
tool below.

Run an AED-only smoke test:

```bash
python3 - <<'PY'
from tagger.tools.sound_field_scene.firered_aed_detector import run

for result in run("models/FireRedVAD/assets/event.wav", duration_sec=22.016):
    print(result.tag_path, result.value)
    print(result.evidence["event_segments"])
PY
```

## PANNs Background-Sound Detection（已废弃，2026-08-25）

The PANNs adapter uses the upstream AudioSet Cnn14 model pinned at commit
`d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`. It normalizes audio to 32 kHz
mono, runs non-overlapping 10-second chunks, and takes the maximum eligible
event probability across the sample.

The panns stage and its public `sound_field_scene.sound` field were removed
on 2026-08-25 — `sound_field_scene.noise_composition` (DASS) supersedes them.
The tool module, model, and runtime are kept for future cross-validation
evidence work, but the stage is not registered, cannot be selected via
`--only-tags`, and its output must not enter public tags.

Set up the pinned source and dedicated environment (if ever needed for
evidence work):

```bash
bash scripts/setup_panns_background.sh
```

The official checkpoint host is not reachable from this server. Manually
upload `Cnn14_mAP=0.431.pth` to
`models/audioset_tagging_cnn/checkpoints/Cnn14_mAP=0.431.pth`, then rerun the
setup command so the checkpoint is validated.

Run a detector-only smoke test after the checkpoint is present:

```bash
python3 - <<'PY'
from tagger.tools.sound_field_scene.panns_background_detector import run

result = run("phase1_asr_samples/audio/wham_noise_only_011a010d_mix.wav")
print(result.tag_path, result.value)
print(result.evidence["winning_event"])
PY
```

## DASS Noise-Type Detection

The DASS adapter uses the local `saurabhati/DASS_medium_AudioSet_48.9`
checkpoint (49M params, AudioSet-2M mAP 48.9) copied into
`models/DASS/saurabhati__DASS_medium_AudioSet_48.9`. Its subprocess runtime is
the project-local `.runtime/dass_runtime_py311/bin/python`, provisioned by
`scripts/setup_dass_gpu.sh` with CUDA-enabled PyTorch and TorchAudio. Audio is normalized to
16 kHz mono and run in non-overlapping 10.24-second chunks; the maximum
eligible probability across chunks is kept.

`sound_field_scene.external_noise_type` is a score-ranked list of
docs/DASS.md category keys — `music`, `animal`, `mechanical`, `nature`,
`formless`, `channel_environment` — one key per category that has any
eligible label at least `0.25` in the full 527-class sigmoid vector. The
default exclusion policy (primary speech, silence, acoustic scenes,
reverberation, echo) applies here too, so e.g. `Silence` never flags a
clean-speech sample as `formless`; human and unclassified labels never
surface regardless. Categories are ordered by their best label score,
highest first.
The default was lowered from the AudioSet multi-label convention `0.50`
after calibrating on the phase2 sample set: DASS-medium sigmoid scores for
real noise classes are soft (roughly 0.1–0.45), while clean speech stays
below 0.15, so `0.25` recovers real noise labels without false positives on
clean speech. The default exclusion policy — primary
speech, silence, room/outdoor scene labels, reverberation, and echo are not
background noise — affects only the ranked top-event evidence; it is
all-or-nothing, so pass `--no-exclusion` to keep every AudioSet class
eligible so the raw class distribution stays visible. A successful
inference with no category at the threshold produces `[]`; model failure
produces `null`. Scores and chunk details remain internal evidence.

`sound_field_scene.noise_composition` expands each `external_noise_type`
category into its concrete labels, produced by the same inference. The full
527-class sigmoid vector (unaffected by the exclusion policy) is bucketed
per `docs/DASS.md` by the mapping in `dass_categories.py` into six public
keys — `music`, `animal`, `mechanical`, `nature`, `formless`,
`channel_environment` — plus evidence-only `human` and `other` buckets. Each
public key holds a score-ranked list of at most `--dass-composition-top-k`
(default 3) labels at or above `--dass-composition-threshold` (default 0.25,
aligned with the category threshold since 2026-08-25, so every present
category has a non-empty bucket). The music bucket is gated by
FireRed AED `music_present`: `false` empties it, `true` or an unavailable AED
(`null`) keeps the DASS music labels. Per-category scores, the `human` and
`other` buckets, and the gate state stay in internal evidence
(`category_events`, `music_gate`) — the channel/environment bucket there is
the reserved supplementary-evidence source for future far-field and
reverberation tags.

Provision or verify the GPU runtime before running the pipeline:

```bash
bash scripts/setup_dass_gpu.sh
```

The setup is idempotent. It checks `nvidia-smi` and the local checkpoint,
reuses a healthy runtime without contacting a package index, installs the
pinned dependencies only when they are missing or incompatible, and runs a
real CUDA forward pass through the 527-class checkpoint.

Run a detector-only smoke test:

```bash
python3 - <<'PY'
from tagger.tools.sound_field_scene.dass_noise_type_detector import run

for result in run("phase1_asr_samples/audio/wham_noise_only_011a010d_mix.wav"):
    print(result.tag_path, result.value)
print("winning:", result.evidence["winning_event"])
PY
```

The full tagging pipeline enables FireRed AED and DASS by default, and DASS
uses the GPU runtime by default. Use `--dass-use-cpu` only for an explicit
CPU debugging run. `--dass-threshold` (default 0.25) overrides the DASS
threshold; `--no-exclusion` keeps every AudioSet class eligible.
