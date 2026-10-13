# Room Acoustic Tools

## Rec-RIR (RIR estimation, RT60, C50)

The Rec-RIR adapters (`rir_estimator.py`, `rt60_estimator.py`,
`c50_estimator.py`) use the upstream Rec-RIR model at
`models/Rec-RIR/config/Rec-RIR.toml` with checkpoint
`models/Rec-RIR/ckpt/epoch35.tar`, run in the configured Python runtime
`.runtime/recrir_py310_torch271/bin/python`.

Public outputs:

| Estimator | Public tag | Method |
| --- | --- | --- |
| RIR | internal artifact only | Estimated room impulse response |
| RT60 | `room_acoustic.rt60_sec` | T20 linear fit over the `-5 dB` to `-25 dB` range of the Schroeder energy decay curve, extrapolated to `-60 dB` |
| C50 | `room_acoustic.c50_db` | `10*log10(E_early/E_late)` around the direct sound: the 50 ms after the maximum-magnitude direct sound versus the remaining late energy |

The estimated RIR waveform never enters the public JSON; it is saved as an
internal artifact under the output directory's `artifacts/rir/` and only the
derived `room_acoustic.rt60_sec` and `room_acoustic.c50_db` are published.
Brouhaha's direct C50 prediction is a different source kept separately in
internal evidence (`internal.brouhaha_c50_db`) and is never mixed with
Rec-RIR C50.

Note: Rec-RIR inference requires GPU in the current deployment environment;
on CPU-only hosts the tool fails and the two public fields are left `null`.
