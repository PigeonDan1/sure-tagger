#!/usr/bin/env bash
set -euo pipefail

# Provision and validate the project-local DASS runtime.  The script is
# intentionally conservative: an existing healthy runtime is reused and no
# package index is contacted in that case.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${DASS_MODEL_DIR:-${PROJECT_ROOT}/models/DASS/saurabhati__DASS_medium_AudioSet_48.9}"
RUNTIME_DIR="${DASS_RUNTIME_DIR:-${PROJECT_ROOT}/.runtime/dass_runtime_py311}"
RUNTIME_PYTHON="${DASS_RUNTIME_PYTHON:-${RUNTIME_DIR}/bin/python}"
REQUIREMENTS_PATH="${DASS_REQUIREMENTS_PATH:-${PROJECT_ROOT}/tagger/tools/sound_field_scene/requirements-dass.txt}"
PYPI_INDEX="${DASS_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${DASS_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

fail() {
    printf 'DASS GPU setup failed: %s\n' "$1" >&2
    exit 1
}

printf 'Checking NVIDIA driver...\n'
command -v nvidia-smi >/dev/null 2>&1 || \
    fail "nvidia-smi is not installed or is not on PATH"
GPU_LIST="$(nvidia-smi -L 2>/dev/null || true)"
[[ -n "${GPU_LIST}" ]] || \
    fail "nvidia-smi found no usable GPU; run this on a GPU allocation"
printf '%s\n' "${GPU_LIST}"

[[ -f "${REQUIREMENTS_PATH}" ]] || \
    fail "DASS requirements file is missing: ${REQUIREMENTS_PATH}"

required_model_files=(
    "config.json"
    "preprocessor_config.json"
    "model.safetensors"
    "configuration_dass.py"
    "modeling_dass.py"
    "feature_extraction_dass.py"
)
for filename in "${required_model_files[@]}"; do
    [[ -s "${MODEL_DIR}/${filename}" ]] || \
        fail "DASS model file is missing: ${MODEL_DIR}/${filename}"
done

find_base_python() {
    local candidate
    local candidates=()
    if [[ -n "${DASS_BASE_PYTHON:-}" ]]; then
        candidates+=("${DASS_BASE_PYTHON}")
    fi
    candidates+=(
        "${PROJECT_ROOT}/.runtime/python311/miniconda/bin/python"
        "${PROJECT_ROOT}/.runtime/python311/miniconda/bin/python3.11"
    )
    if command -v python3.11 >/dev/null 2>&1; then
        candidates+=("$(command -v python3.11)")
    fi
    if command -v python3 >/dev/null 2>&1; then
        candidates+=("$(command -v python3)")
    fi
    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] || continue
        if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ -e "${RUNTIME_DIR}" || -L "${RUNTIME_DIR}" ]]; then
    [[ -d "${RUNTIME_DIR}" ]] || \
        fail "runtime path exists but is not a directory: ${RUNTIME_DIR}"
    [[ -x "${RUNTIME_PYTHON}" ]] || \
        fail "runtime exists but its Python is not executable: ${RUNTIME_PYTHON}"
else
    BASE_PYTHON="$(find_base_python || true)"
    [[ -n "${BASE_PYTHON}" ]] || \
        fail "Python 3.11 is required to create ${RUNTIME_DIR}; set DASS_BASE_PYTHON"
    mkdir -p "$(dirname "${RUNTIME_DIR}")"
    printf 'Creating runtime with %s...\n' "${BASE_PYTHON}"
    "${BASE_PYTHON}" -m venv --system-site-packages "${RUNTIME_DIR}" || \
        fail "could not create virtual environment at ${RUNTIME_DIR}"
    [[ -x "${RUNTIME_PYTHON}" ]] || \
        fail "virtual environment was created without ${RUNTIME_PYTHON}"
fi

if ! "${RUNTIME_PYTHON}" -m pip --version >/dev/null 2>&1; then
    printf 'Bootstrapping pip in the DASS runtime...\n'
    "${RUNTIME_PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || \
        fail "pip is unavailable in ${RUNTIME_PYTHON}; install pip and rerun"
fi

runtime_dependencies_healthy() {
    "${RUNTIME_PYTHON}" <<'PY'
import importlib
import sys

expected = {
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "4.49.0",
    "librosa": "0.11.0",
    "numpy": "2.4.6",
    "soundfile": "0.14.0",
    "safetensors": None,
}
for name, version in expected.items():
    module = importlib.import_module(name)
    actual = str(getattr(module, "__version__", ""))
    if name in ("torch", "torchaudio"):
        actual = actual.split("+", 1)[0]
    if version is not None and actual != version:
        raise RuntimeError("%s version is %s, expected %s" % (name, actual, version))

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot access CUDA (version=%s, compiled_cuda=%s)"
        % (torch.__version__, torch.version.cuda)
    )
if torch.cuda.device_count() < 1:
    raise RuntimeError("PyTorch reports no CUDA devices")
probe = torch.ones(1, device="cuda")
torch.cuda.synchronize(probe.device)
print(
    "DASS runtime dependencies healthy: torch=%s, cuda=%s, devices=%d"
    % (torch.__version__, torch.version.cuda, torch.cuda.device_count())
)
PY
}

pytorch_cuda_stack_healthy() {
    "${RUNTIME_PYTHON}" <<'PY'
import torch
import torchaudio

torch_version = str(torch.__version__).split("+", 1)[0]
audio_version = str(torchaudio.__version__).split("+", 1)[0]
if torch_version != "2.7.1" or audio_version != "2.7.1":
    raise RuntimeError(
        "expected torch/torchaudio 2.7.1, found %s/%s"
        % (torch.__version__, torchaudio.__version__)
    )
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("PyTorch cannot access a CUDA device")
PY
}

if ! runtime_dependencies_healthy; then
    printf 'DASS dependencies are incomplete or not CUDA-enabled; installing pinned requirements...\n'
    # Repair torch and torchaudio together. Explicit local versions plus a
    # forced reinstall prevent an existing CPU wheel with the same public
    # version from being treated as already satisfied.
    if ! pytorch_cuda_stack_healthy; then
        printf 'Installing CUDA-enabled torch/torchaudio 2.7.1 from the cu126 index...\n'
        "${RUNTIME_PYTHON}" -m pip install \
            --disable-pip-version-check \
            --force-reinstall \
            --index-url "${PYTORCH_INDEX}" \
            --extra-index-url "${PYPI_INDEX}" \
            "torch==2.7.1+cu126" \
            "torchaudio==2.7.1+cu126" || \
            fail "CUDA-enabled torch/torchaudio installation failed"
    fi
    "${RUNTIME_PYTHON}" -m pip install \
        --disable-pip-version-check \
        --upgrade-strategy only-if-needed \
        --index-url "${PYPI_INDEX}" \
        --extra-index-url "${PYTORCH_INDEX}" \
        --requirement "${REQUIREMENTS_PATH}" || \
        fail "dependency installation failed; inspect pip output and rerun"
    runtime_dependencies_healthy || \
        fail "installed dependencies still cannot access CUDA"
else
    printf 'Pinned DASS dependencies already present; skipping pip install.\n'
fi

printf 'Loading DASS checkpoint and running a CUDA forward pass...\n'
"${RUNTIME_PYTHON}" - "${MODEL_DIR}" <<'PY'
import sys

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_dir = sys.argv[1]
device = torch.device("cuda:0")
model = AutoModelForAudioClassification.from_pretrained(
    model_dir, trust_remote_code=True
).to(device).eval()
extractor = AutoFeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
max_length = int(getattr(extractor, "max_length", 0) or 0)
if max_length <= 0:
    raise RuntimeError("DASS feature extractor has no positive max_length")
waveform = np.zeros(max_length * 160, dtype=np.float32)
inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
with torch.no_grad():
    logits = model(**inputs).logits
torch.cuda.synchronize(device)
if tuple(logits.shape) != (1, 527):
    raise RuntimeError("unexpected DASS output shape: %s" % (tuple(logits.shape),))
print(
    "DASS CUDA forward pass healthy: device=%s, model_outputs=%s"
    % (device, tuple(logits.shape))
)
PY

printf '\nDASS GPU runtime is ready.\n'
printf 'Runtime: %s\n' "${RUNTIME_PYTHON}"
printf 'Model:   %s\n' "${MODEL_DIR}"
