# Basic Acoustic Model Tools

## VAD

`basic_acoustic.silence_segments` now uses a metadata-first route. If
`sample.native_metadata` contains usable segment annotations, the pipeline runs
`native_metadata_vad.py` and does not call FireRed VAD for that sample.

Recognized fields:

- `silence_segments`
- `speech_segments`
- `vad_segments`
- `segments`
- `utterances`
- `words`

Segments may use `start`/`end` or `start_sec`/`end_sec`. If segment times appear
to be absolute recording times and `sample.native_metadata.start` is available,
the tool shifts them into sample-relative time before clipping to
`duration_sec`. When no usable metadata segment exists, the pipeline falls back
to FireRed VAD and then derives `basic_acoustic.nonspeech_ratio` from the final
non-speech segments. The ratio is the complement of detected speech coverage,
not a physical silence or low-energy measurement.

The Brouhaha and DNSMOS quality estimators used to live in this directory and
now sit under `tagger/tools/audio_quality/` with their own README.
