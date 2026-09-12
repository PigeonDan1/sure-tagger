# Audio Quality Tools

## Brouhaha SNR

The Brouhaha adapter (`brouhaha_signal_estimator.py`) runs the upstream
Brouhaha model at
`models/brouhaha/brouhaha-vad/models/best/checkpoints/best.ckpt` with the
configured Python runtime `.runtime/fireredvad_rebuild_py310/bin/python`.

`audio_quality.snr_db` is the arithmetic mean of the valid per-frame SNR
predictions. Brouhaha also predicts per-frame C50, but that value only enters
internal evidence (`internal.brouhaha_c50_db`) for cross-validation against
Rec-RIR C50 — it never enters the public tags-only output.

## DNSMOS

The DNSMOS adapter follows Microsoft's
[`DNSMOS/dnsmos_local.py`](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
implementation at commit `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`. It
publishes four regular no-reference MOS values by default:

| Upstream output | Public tag |
| --- | --- |
| `SIG` | `audio_quality.dnsmos_sig` |
| `BAK` | `audio_quality.dnsmos_bak` |
| `OVRL` | `audio_quality.dnsmos_ovrl` |
| `P808_MOS` | `audio_quality.dnsmos_p808` |

The local model checkout is `models/DNS-Challenge`, and the configured Python
is `.runtime/recrir_py310_torch271/bin/python`. This reuses the existing
`librosa`, `numpy`, and `soundfile` packages. The additional runtime dependency
was installed from the Tsinghua PyPI mirror:

```bash
.runtime/recrir_py310_torch271/bin/python -m pip install \
  --only-binary=:all: \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  onnxruntime==1.23.2
```

The model paths and subprocess interpreter are fixed in
`tagger/local_config.py`. Standard DNSMOS is the pipeline default. Pass
`--dnsmos-personalized` to `scripts/run_tagger.py` to select the upstream
personalized primary model; P.808 scoring is unchanged.

Run a DNSMOS-only smoke test without invoking the other model tools:

```bash
python3 - <<'PY'
from tagger.tools.audio_quality.dnsmos_quality_estimator import run

results = run("phase1_asr_samples/audio/wsj_clean_short_40go030s.wav")
for result in results:
    print(result.tag_path, result.value)
PY
```

The adapter converts multichannel input to mono, resamples to 16 kHz, repeats
clips shorter than the official 9.01-second window, and averages overlapping
one-second-hop predictions exactly as the upstream local evaluator does. Any
missing dependency/model, unreadable audio, invalid ONNX output, or score
outside `[1, 5]` leaves the affected public tag as `null`.
