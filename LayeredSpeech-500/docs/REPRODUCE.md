# Reproduction Guide

## Environment

- Python 3.10, 3.11, or 3.12
- Linux recommended
- `./scripts/bootstrap.sh` or `pip install -r requirements-lock.txt && pip install -e . --no-deps`
- CUDA is required by the Dots full profile; fixture mode is CPU-only
- FFmpeg is required for MP3 decoding and rate calibration in the full profile

The dependency set is intentionally split: the renderer is lightweight, while
the Dots runtime and CUDA/PyTorch stack are downloaded separately from the model
repositories listed in `data/models.lock.yaml`.

## Profiles

`fixture` runs the offline deterministic renderer with synthetic speech-shaped
audio. It must work without credentials, external recordings, or model weights.

`full` runs the v4 scene pipeline. It requires:

```text
external_assets/
├── refs/ami_headset_100/audio_enhanced/
├── refs/alimeeting_headset_100/audio_enhanced/
├── refs/aishell3_speaker_pool/audio_enhanced/
├── fleurs/audio_enhanced_deepfilternet/
├── fleurs/transcripts/
├── rir/Audio/
└── music/
models/
├── dots_tts_base/checkpoints/
├── dots_tts_edit/checkpoints/
└── runtime/.venv/
```

Measured RIR and real music are deliberately user-supplied. The release does not
pretend that their licenses are known or grant redistribution rights.
`data/asset_hashes.json` lists the 315 exact input files used by the paper run:
283 reference WAVs, 16 FLEURS transcripts, 14 measured RIRs and two music files.
`python scripts/verify_assets.py --root external_assets` checks both size and
SHA-256. Resource preparation scripts in the original research workspace are
not a public download service; this release makes no claim that upstream URLs
will continue to serve identical files.

Run `python scripts/download_models.py --accept-model-terms` to download and
SHA-256-check the two pinned Dots checkpoints, then `./scripts/setup_dots_runtime.sh` to install
the pinned upstream Dots runtime in an isolated environment. `--dry-run` and
`--check` inspect prerequisites without downloading or synthesizing audio.

## Determinism

The frozen scene file is `data/scene_scripts_v4.jsonl` and the generation seed is
`20260911`. The release fixes the renderer seed, scene ordering, source key
derivation, sample-rate matrix, and output schema. Different CUDA, PyTorch,
vocoder, or driver versions can produce small waveform differences; acceptance is
based on manifest structure, counts, QC, and aggregate statistics rather than
byte-identical WAV hashes.

## Expected Result

- 500 scenes
- approximately 3.7784 hours of audio
- 413 scenes at 24 kHz, 72 at 16 kHz, and 15 at 8 kHz
- historical hard QC failures: 0
- historical warnings: 324, including 304 speech-speed proxy mismatches

The warning count is part of the historical result. A passed QC report means no
structural hard failure; it does not claim that every requested acoustic proxy
is exact. The full profile has not been rerun from this public package.
