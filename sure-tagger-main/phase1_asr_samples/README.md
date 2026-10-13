# Phase 1 ASR Representative Samples

This folder contains copied representative samples from the shared `data` tree.

## Files

- `audio/*.wav`: copied audio files for local testing.
- `manifest.jsonl`: raw-only input records aligned with `development.md` (`corpus` + `sample`).
- `kaldi/text`: extracted original utterance ids and transcript strings.
- `kaldi/utt2spk`: extracted original utterance ids and speaker ids.
- `kaldi/wav.scp`: local wav mapping for the 7 samples that have ASR transcripts.
- `local_audio.scp`: local wav mapping for all 8 copied audio files, including the WHAM no-transcript sample.

## Raw-Only Notes

`manifest.jsonl` does not contain inferred language, topic, noise type, duration, sample rate, channel count, confidence, evidence, warnings, or selection rationale.

The manifest keeps only raw source fields and provenance:

- source dataset name from the source path group,
- source audio path,
- original transcript text where present,
- original `utt2spk` value where present,
- source manifest path, source split, row index, and raw source lines.

Local copied audio paths are kept outside the raw input in `local_audio.scp`.
