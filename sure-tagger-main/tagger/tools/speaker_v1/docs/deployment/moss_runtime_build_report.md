# MOSS shared runtime build report

Status: PASS
Date: 2026-08-11 (Asia/Shanghai)

## Published runtime

- Final path: `/hpc_stor03/sjtu_home/weihan.chen/share/tagger/.runtime/moss_transcribe_diarize_py311_torch280_cu128_v1`
- Source: `/hpc_stor03/sjtu_home/weihan.chen/share/tagger/models/MOSS-Transcribe-Diarize`
- Temporary build path: `/hpc_stor03/sjtu_home/weihan.chen/share/tagger/.runtime/.build-moss_transcribe_diarize_py311_torch280_cu128_v1-7JYkfFr9`
- Published size: 7.1 GiB (`du -sh`)
- Final root permissions: `0750`, owner `huifei.wang`, group `sjtu`
- Existing `.runtime/moss_transcribe_diarize_py312` was not modified.
- No model files or model weights were modified.
- The build phase did not run model inference. A post-publication demo inference is recorded below.

The source tree contains an existing `moss_transcribe_diarize.egg-info` whose files are owned by another maintainer. An editable setuptools build attempted to refresh this metadata and failed on permissions. To avoid changing model/source files, the successful build used an rsync source snapshot under the temporary runtime, excluding `*.egg-info` and `__pycache__`, and installed it non-editably. The clean snapshot remains inside the final runtime at `_build_source/MOSS-Transcribe-Diarize` for provenance. The installed package code is self-contained in `site-packages`.

## Build and fallback commands

Environment variables below contain only shared filesystem paths; no credentials were used.

```bash
RUNTIME_ROOT=/hpc_stor03/sjtu_home/weihan.chen/share/tagger/.runtime
UV_BIN="$RUNTIME_ROOT/uv_bootstrap_py311/bin/uv"
SOURCE_DIR=/hpc_stor03/sjtu_home/weihan.chen/share/tagger/models/MOSS-Transcribe-Diarize
ENV_NAME=moss_transcribe_diarize_py311_torch280_cu128_v1
FINAL_DIR="$RUNTIME_ROOT/$ENV_NAME"
MOSS_BUILD_DIR="$RUNTIME_ROOT/.build-$ENV_NAME-7JYkfFr9"

flock -n "$RUNTIME_ROOT/.speaker-install.lock" bash --noprofile --norc
"$UV_BIN" venv --relocatable \
  --python "$RUNTIME_ROOT/uv_bootstrap_py311/bin/python" \
  "$MOSS_BUILD_DIR"
```

The required offline attempt was made first. It failed because the uv cache did not contain the Torch wheel/metadata:

```bash
"$UV_BIN" pip install --offline \
  --python "$MOSS_BUILD_DIR/bin/python" \
  --torch-backend=cu128 --link-mode=copy --compile-bytecode \
  "torch==2.8.0" "torchaudio==2.8.0" -e "$SOURCE_DIR"
```

The first normal-network fallback used the official PyTorch CUDA 12.8 index. It failed because `download-r2.pytorch.org` timed out after retries:

```bash
"$UV_BIN" pip install \
  --python "$MOSS_BUILD_DIR/bin/python" \
  --torch-backend=cu128 --link-mode=copy --compile-bytecode \
  "torch==2.8.0" "torchaudio==2.8.0" -e "$SOURCE_DIR"
```

PyPI was reachable. The successful build used the normal PyPI Linux Torch 2.8.0 distribution, whose runtime was then verified as CUDA 12.8:

```bash
STAGE_DIR="$MOSS_BUILD_DIR/_build_source/MOSS-Transcribe-Diarize"
mkdir -p "$STAGE_DIR"
rsync -a --exclude='*.egg-info/' --exclude='__pycache__/' \
  "$SOURCE_DIR/" "$STAGE_DIR/"

UV_HTTP_TIMEOUT=300 "$UV_BIN" pip install \
  --python "$MOSS_BUILD_DIR/bin/python" \
  --link-mode=copy --compile-bytecode \
  "torch==2.8.0" "torchaudio==2.8.0" "$STAGE_DIR"
```

Generated `build/` and `*.egg-info` files were removed only from the temporary snapshot. `diff -qr`, excluding those generated paths and `__pycache__`, confirmed the snapshot matched the shared source. `direct_url.json` was set to the final in-runtime snapshot path before publication. It is non-editable and contains only a `/share/tagger/` path.

```bash
"$UV_BIN" pip check --python "$MOSS_BUILD_DIR/bin/python"
mv -- "$MOSS_BUILD_DIR" "$FINAL_DIR"
"$UV_BIN" pip check --python "$FINAL_DIR/bin/python"
chmod 0750 "$FINAL_DIR"
```

## Verification

- `uv pip check`: all 81 installed packages compatible, before and after rename.
- Python: 3.11.5.
- Torch: 2.8.0+cu128.
- Torchaudio: 2.8.0+cu128.
- CUDA runtime reported by Torch: 12.8.
- `torch.cuda.is_available()`: true.
- Device: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5.
- CUDA tensor smoke: `sum(square([1, 2, 3])) == 14.0` on `cuda:0`.
- Imports passed: package root, configuration, modeling, processing, inference utilities, Torch, Torchaudio, Transformers, PyAV and Librosa.
- Relocatable `mtd-subtitle --help` passed after atomic rename.
- Old non-shared path scan passed: no `/sjtu_home/weihan.chen/tagger/` reference in scripts, Python source, `.pth` or `direct_url.json`.
- Editable artifact scan passed: no `__editable__*` artifact.
- Final `direct_url.json` exists, is non-editable, resolves to an existing directory, contains `/sjtu_home/weihan.chen/share/tagger/`, and contains no `.build-*` path.

## Demo inference verification

The shared runtime subsequently completed a real utterance-level shadow run:

- Sample: `EN2001a_utterance_00000`.
- Audio: `/hpc_stor03/sjtu_home/weihan.chen/share/tagger/ami_en2001a_utterances/audio/EN2001a_utterance_00000.wav`.
- Device and dtype: `cuda:0`, FP16.
- Result: success.
- First standalone inference: 44.158 seconds.
- Integrated shadow run: 26.219943 seconds.
- MOSS-observed speaker count: 2.
- Shadow count state: supported observation only; `certified_speaker_count=null` because no independent count certificate exists yet.
- Result: `ami_en2001a_utterances/outputs/speaker_v2_demo_20260811/speaker_v2_shadow_results.jsonl`.
- Fusion artifact: `ami_en2001a_utterances/outputs/speaker_v2_demo_20260811/artifacts/speaker_v2/EN2001a_utterance_00000/fusion_artifact_v2.json.gz`.

Source fingerprints:

```text
181e406a953d1b6b1a18971dada4427732597fa592252fe8e63080169192d72b  pyproject.toml
989087110e877549ba7879e985d72f2d6e505524384cba51b1ce81c3b71711da  moss_transcribe_diarize/__init__.py
```

## Exact package freeze

The authoritative machine-readable freeze is stored with the runtime at `.runtime/moss_transcribe_diarize_py311_torch280_cu128_v1/packages.lock`.

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
audioread==3.1.0
av==18.0.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.4.9
click==8.4.2
decorator==5.3.1
fastapi==0.141.1
filelock==3.32.2
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.27.0
idna==3.18
jinja2==3.1.6
joblib==1.5.3
lazy-loader==0.5
librosa==0.11.0
llvmlite==0.48.0
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
moss-transcribe-diarize @ file:///hpc_stor03/sjtu_home/weihan.chen/share/tagger/.runtime/moss_transcribe_diarize_py311_torch280_cu128_v1/_build_source/MOSS-Transcribe-Diarize
mpmath==1.3.0
msgpack==1.2.1
narwhals==2.24.0
networkx==3.6.1
numba==0.66.0
numpy==2.4.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
packaging==26.3
platformdirs==4.11.2
pooch==1.9.0
pycparser==3.0
pydantic==2.13.4
pydantic-core==2.46.4
pygments==2.20.0
python-multipart==0.0.32
pyyaml==6.0.3
regex==2026.7.19
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.17.1
setuptools==84.0.0
shellingham==1.5.4
soundfile==0.14.0
soxr==1.1.0
starlette==1.6.0
sympy==1.14.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.8.0
torchaudio==2.8.0
tqdm==4.70.0
transformers==5.15.0
triton==3.4.0
typer==0.27.1
typing-extensions==4.16.0
typing-inspection==0.4.3
urllib3==2.7.0
uvicorn==0.52.1
```
