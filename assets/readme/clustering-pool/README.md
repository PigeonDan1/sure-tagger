# Clustered Scene Exemplars

These six audio clips are representative records from the 1,000-sample
`audio_tags_only` scene-clustering run. The run produced six frozen clusters
(`C00`–`C05`) using `cluster_rules_v0.5`, k-medoids, `K=6`, and no WER/CER input
to assignment.

| Cluster | Sample | Profile | Audio | Full structured tags | Assignment |
| --- | --- | --- | --- | --- | --- |
| C00 · 341 samples | `multi_000525_CCFBDCI-001982_EN` | Clean, long single-speaker English weather report | [WAV](C00/audio.wav) | [JSON](C00/tags.json) | [JSON](C00/cluster_assignment.json) |
| C01 · 122 samples | `multi_000454_Elderly0014S0046W0055__s0_000055` | Short Mandarin speech with lower SNR | [WAV](C01/audio.wav) | [JSON](C01/tags.json) | [JSON](C01/cluster_assignment.json) |
| C02 · 286 samples | `multi_000901_aishell1_test_BAC009S0766W0141` | Short single-speaker Mandarin read speech | [WAV](C02/audio.wav) | [JSON](C02/tags.json) | [JSON](C02/cluster_assignment.json) |
| C03 · 196 samples | `multi_000513_CCFBDCI-001454_EN` | Long, high-SNR single-speaker English travel conversation | [WAV](C03/audio.wav) | [JSON](C03/tags.json) | [JSON](C03/cluster_assignment.json) |
| C04 · 17 samples | `multi_000803_AMI_EN2002a_H02_FEO072_0033513_0033555` | 0.42-second non-speech boundary case | [WAV](C04/audio.wav) | [JSON](C04/tags.json) | [JSON](C04/cluster_assignment.json) |
| C05 · 38 samples | `multi_000347_gigaspeech2_th_test__v1.0.1__00047__0-79` | Thai speech with music and low SNR | [WAV](C05/audio.wav) | [JSON](C05/tags.json) | [JSON](C05/cluster_assignment.json) |

C01 and C04 use the medoids recorded in the cluster summary. C00, C02, and C03
use same-cluster members with less sensitive example content. The C05 medoid
is a contradictory 0.18-second boundary clip whose current tags do not show
music; this bundle instead uses a C05 member tagged with speech and music. Each
`sample_info.json` records the selection rationale.

Every sample directory also includes `cluster_profile.json` and
`sample_info.json`. The latter records sample ID, source key, selection type,
audio properties, SHA-256, and upstream URL where available.

## Run provenance

`source_run.json` identifies the exact tag snapshot and assignment file used:

- Input tag snapshot SHA-256: `7fe87530179fdf575c1c17946ecb30aea006e5e0570404db5a6ba9e3391f354d`
- Cluster assignment SHA-256: `4d7191752f27c8150382f4e6621b69d5ad16593142db9ad344d33d7ab10b4dbd`
- Input pool: 1,000 samples; frozen partition: 6 clusters
- Assignment basis: audio tags only; error rates were post-hoc diagnostics

## Source terms

The samples originate from SlideSpeech, BAAI SeniorTalk, AISHELL-1,
ContextASR-Bench Speech, AMI, and GigaSpeech2. The pooled local benchmark does
not establish one redistribution license for all six sources. See
[`ATTRIBUTION.md`](ATTRIBUTION.md) and each sample's `sample_info.json`; upstream
terms apply and may restrict redistribution or reuse.

These cluster IDs describe this frozen 1,000-sample run. They are not labels
from the original datasets or universal assignments for other audio.
