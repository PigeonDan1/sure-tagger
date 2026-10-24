# LayeredSpeech-500

This is the standalone research release for the 500-scene multilingual L1-L5
speech benchmark. It contains the scene design, deterministic acoustic renderer,
Ground Truth schema, manifests, and reproduction tools.

The release intentionally does not contain generated WAV files, model weights,
reference recordings, measured RIRs, or background music. Those resources have
separate upstream licenses and are described in `docs/DATA_LICENSES.md` and
`docs/MODEL_LICENSES.md`.

## Quick Start

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
python scripts/verify_release.py
python scripts/reproduce.py --profile fixture --output runs/fixture
```

The fixture profile is fully offline and produces non-linguistic speech-shaped
test audio. It verifies the renderer and output contract; it is not a substitute
for the TTS model or the paper data.

## Full Reproduction

Full reproduction requires legally obtained reference recordings, RIRs and music.
The project records their exact file hashes but does not redistribute them or
silently substitute other audio. Download the pinned model checkpoints only
after reviewing their separate terms:

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py --accept-model-terms
./scripts/setup_dots_runtime.sh
# Place authorized input audio under external_assets/ (layout in docs/REPRODUCE.md).
python scripts/verify_assets.py --root external_assets
python scripts/reproduce.py --profile full --check
python scripts/reproduce.py --profile full --gpus 0 --output runs/v4_multilingual
```

See `docs/REPRODUCE.md` for the required directory layout, seed, model settings,
asset checks, and expected 500-scene QC summary. `full` was not executed as
part of the public package check because the inputs are not redistributed.

## Release Boundary

Project-owned code is Apache-2.0. External data and model licenses are not changed
by this repository. In particular, AliMeeting and some environmental/reference
assets may be non-commercial or require separate attribution. Do not redistribute
external files unless their upstream terms explicitly allow it.

The optional sure-tagger and Qwen Omni evaluation steps are not required for the
main generation or QC flow. They require separate installations, credentials, or
service access and are documented as optional evaluation profiles.

The checked-in `manifests/` files describe the historical paper run. They are
provenance records, not runnable input paths; a new run writes its own manifests
under `runs/`.
