# LayeredSpeech-500 audio examples / 合成音频示例

These five examples come from the historical 500-scene LayeredSpeech-500 run.
Each `audio.wav` is an unmodified rendered mixture. `ground_truth.json` is the
complete record for that scene from the checked-in
[`ground_truth.jsonl`](../../../LayeredSpeech-500/manifests/ground_truth/ground_truth.jsonl).
`review.json` is the matching record from the local 24-sample review site: it
contains field-level GT, SURE-Tagger, Captioner and Qwen3-Omni comparisons.
Its audio path points to the packaged `audio.wav`.

| Scene | Controlled conditions | Audio | GT | Comparison |
| --- | --- | --- | --- | --- |
| `SCN-0006` | Mandarin, one speaker, transit ambience, 15 dB SNR | [WAV](SCN-0006/audio.wav) | [JSON](SCN-0006/ground_truth.json) | [JSON](SCN-0006/review.json) |
| `SCN-0198` | Mandarin, two speakers, overlap and background music | [WAV](SCN-0198/audio.wav) | [JSON](SCN-0198/ground_truth.json) | [JSON](SCN-0198/review.json) |
| `SCN-0360` | Mandarin, three speakers, overlap, music and 5 dB SNR | [WAV](SCN-0360/audio.wav) | [JSON](SCN-0360/ground_truth.json) | [JSON](SCN-0360/review.json) |
| `SCN-0367` | Mandarin/English code-switching, three speakers, overlap | [WAV](SCN-0367/audio.wav) | [JSON](SCN-0367/ground_truth.json) | [JSON](SCN-0367/review.json) |
| `SCN-0469` | Arabic, one speaker, open-space ambience, 5 dB SNR | [WAV](SCN-0469/audio.wav) | [JSON](SCN-0469/ground_truth.json) | [JSON](SCN-0469/review.json) |

These are individual examples, not a balanced sample of the 500 scenes or a
substitute for the complete benchmark. The ground-truth tags come from render
controls and measurements; descriptive reference fields are unavailable in
the strict no-inference GT and are shown separately in the review record.
`nonspeech_ratio` means time without scheduled human speech, not physical
silence. A code-switching scene cannot be represented by one language label.

## Provenance and use / 来源与使用范围

The mixtures were generated with Dots TTS and the project renderer using
reference recordings and environmental inputs from several upstream sources.
The project owns the orchestration code, but it does not grant a new or common
license to AMI, AliMeeting, AISHELL-3, FLEURS, measured RIRs, noise or music
used during generation. These five excerpts are provided for inspection of
this project; check [data and asset licenses](../../../LayeredSpeech-500/docs/DATA_LICENSES.md)
and upstream terms before further reuse or redistribution. The standalone
`LayeredSpeech-500/` source release still excludes the full audio corpus.

Exact source-audio SHA-256 values:

| Scene | SHA-256 |
| --- | --- |
| `SCN-0006` | `532ac2f1d4333446ebe30791ba5da24133e80590268518800b01917b400d4fb2` |
| `SCN-0198` | `40f8e727cc7774406bce9851951a57663e7c512af8d5a36832c4f514149a160f` |
| `SCN-0360` | `c0e6aac2672f9848baeebb81d679d2ae0ff18589f098eaad296805f679da19a7` |
| `SCN-0367` | `c8928b8a38920791d8467269e2b1322dc02bd5722a195d31b60ddf6f7f39f85e` |
| `SCN-0469` | `568592d75a5df429b5e6a253d40ffbb6ab49977bde9d9a55e9eee029a2429f78` |
