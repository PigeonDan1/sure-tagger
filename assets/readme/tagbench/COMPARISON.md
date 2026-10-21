# Same Audio, Different Descriptions

The two rows below use the same audio file for all systems. `Sure-Tagger`
combines structured acoustic and speaker evidence with a description; the
Captioner mainly describes the semantic content and may leave acoustic fields
empty.

## `fleurs_t2_0055`

| System | Description focus |
| --- | --- |
| Sure-Tagger | Single-speaker Mandarin reading; quiet background; low reverberation; SNR/DNSMOS and `far_field=null` are reported with evidence boundaries. |
| Captioner | Mentions the boundary-dispute text and the 1967 borders; does not provide the same acoustic measurements. |
| Qwen3-Omni | Summarizes the Chinese speech topic and describes it as quiet. |

## `librispeech_t2_0048`

| System | Description focus |
| --- | --- |
| Sure-Tagger | Single-speaker English literary reading; speech-only event; no external noise labels; low reverberation indicators and measured quality fields. |
| Captioner | Identifies a possible story or novel scene involving Brother Hickey, without the structured acoustic fields. |
| Qwen3-Omni | Describes a single English narrative reading and its content. |

The machine-readable records are in [`labels.json`](labels.json), where `A`,
`B`, and `C` correspond to Sure-Tagger, Captioner, and Qwen3-Omni.
