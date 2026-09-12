# Language Content Tools

Language-content tools run inside the root tagging pipeline. Deterministic tools
use `sample.text.transcript`, or the speaker-v2 language-routed ASR transcript
when the input text is empty. FireRed LID drives that route: Chinese (`zh` or
`zh-*`) and English (`en` or `en-*`) use MOSS joint ASR/diarization; other
languages use FireRedASR2-AED text plus the official DiariZen timeline. Mixed,
unknown, and failed LID results use the other-language branch.
A selected branch never falls back to the other.

This routing rule is broader than the verified ASR capability. The pinned
FireRed and Qwen3-ForcedAligner catalogs advertise the English/Chinese family.
English passed an end-to-end smoke, but a real French sample returned valid JSON
with incorrect English text while its DiariZen turn remained usable; an explicit
`--language fr` request failed with exit code 6. Other-language routed ASR text
is not production-validated and can pass structural validation despite
being semantically wrong.

Implemented fields:

- `language_content.language`: FireRed LID audio language ID. Its `zh`/`zh-*`
  or `en`/`en-*` result selects MOSS; all other or unknown results select the
  official DiariZen + FireRed branch. LID failure
  leaves this field `null` while the speaker route proceeds as unknown.
- `language_content.word_count`: simple multilingual token word count.
- `language_content.punctuation`: punctuation count and terminal punctuation flag.
- `language_content.repetition`: consecutive repeated word/ngram detection.
- `language_content.filler`: lexicon count for fillers such as `uh`, `um`, `hmm`,
  and `yeah`.

The routed transcript is published separately as `speaker.asr_transcript`. For
Chinese and English it is built from MOSS timeline segment text; for other
languages it is the FireRedASR2-AED text. The official DiariZen timeline
independently supplies the other-language speaker timeline. In both
branches the public transcript is ordered by time, has no timestamps or speaker
IDs, and is `null` when the selected ASR output has no valid text. A non-empty
`sample.text.transcript` is still the preferred input for the four deterministic
language-content fields, but it never populates `speaker.asr_transcript`.
`language_content.language` never uses either transcript: with empty input text
it reuses the exact FireRed LID observation and evidence already produced for
speaker routing, without rerunning LID or inferring language from ASR text.

`language_content.topic` is not registered or emitted by sure-tagger. Topic is
an open descriptive phrase inferred by the downstream language model from the
deterministic tags and ASR text; it has no fixed taxonomy in this layer. The
legacy `topic.py` module remains only for historical pipeline reproducibility.
