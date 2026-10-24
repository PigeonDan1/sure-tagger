# Model Licenses

Model weights are downloaded separately and are not covered by the project
Apache-2.0 license.

| Model | Role | Policy |
| --- | --- | --- |
| dots.tts-base | Base TTS | Pin `rednote-hilab/dots.tts-base` to the recorded revision; model card reports Apache-2.0 |
| dots.tts.edit | Rate/prosody edit | Pin `dots-studio/dots.tts.edit` to the recorded revision; model card reports `license: other`, so review its terms before use |
| dots.tts runtime | Loads both checkpoints | Install pinned source commit; code repository reports Apache-2.0 |

`data/models.lock.yaml` records the observed checkpoint and runtime revisions.
`data/model_hashes.json` records the SHA-256 values of the two checkpoint
snapshots used in the research workspace. The downloader verifies the returned
revision and all recorded files before writing `MODEL_SOURCE.json`; full-profile
preflight repeats the hash checks.
