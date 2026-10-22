# Same-Audio System Comparisons

Each row compares outputs for **one identical audio file**. Values below are
examples, not aggregate benchmark scores. Open a sample's
`full_predictions.json` for all structural and descriptive fields; see
[`README.md`](README.md) for the audio and the original review questions.

| Sample | Reference | SURE-Tagger | Captioner | Qwen3-Omni |
| --- | --- | --- | --- | --- |
| `ES2012d_utterance_00018` | AMI: 4 speakers, overlap | 4 speakers; overlap `true` | 2 speakers; overlap `null` | 2 speakers; overlap `false` |
| `EN2001e_utterance_00071` | AMI: 2 speakers, overlap | 2 speakers; overlap `true` | Count and overlap `null` | 2 speakers; overlap `false` |
| `fleurs_t2_0347` | FLEURS: Russian (`ru`) | Language `ru`; adds acoustic metrics | Language `null`; describes content | Language `ru`; describes content |
| `t3_desed_0112` | DESED: speech and mechanical event | Speech detected; external noise `[]` | No structured event classes | Speech detected; no structured noise class |
| `t3_fsd50k_0045` | FSD50K: music event | `music_present=false`; event list `[]` | No structured music field | No structured music field |

The two event references above are derived from source annotations with the
benchmark's closed-world crosswalk. The FSD50K example illustrates a missed
music tag; the DESED example illustrates a missing mechanical noise tag.
Neither example should be presented as an overall system ranking.
