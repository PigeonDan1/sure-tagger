#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/qiuqiangkong/audioset_tagging_cnn.git"
UPSTREAM_COMMIT="d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
REPO_DIR="${PROJECT_ROOT}/models/audioset_tagging_cnn"
CHECKPOINT_PATH="${REPO_DIR}/checkpoints/Cnn14_mAP=0.431.pth"
BASE_PYTHON="${PROJECT_ROOT}/.runtime/recrir_py310_torch271/bin/python"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime/panns_py310"
RUNTIME_PYTHON="${RUNTIME_DIR}/bin/python"
REQUIREMENTS_PATH="${PROJECT_ROOT}/tagger/tools/sound_field_scene/requirements-panns.txt"
PYPI_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

fail() {
    printf 'PANNs setup failed: %s\n' "$1" >&2
    exit 1
}

if [[ ! -x "${BASE_PYTHON}" ]]; then
    fail "base Python is missing: ${BASE_PYTHON}"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    if [[ -e "${REPO_DIR}" ]]; then
        fail "${REPO_DIR} exists but is not a git checkout; move it aside and rerun"
    fi
    mkdir -p "$(dirname "${REPO_DIR}")"
    git clone "${UPSTREAM_URL}" "${REPO_DIR}" || \
        fail "could not clone upstream source; download it manually and rerun"
    git -C "${REPO_DIR}" checkout --detach "${UPSTREAM_COMMIT}" || \
        fail "could not check out pinned upstream commit"
else
    CURRENT_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
    if [[ "${CURRENT_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
        fail "upstream checkout is ${CURRENT_COMMIT}; expected ${UPSTREAM_COMMIT}"
    fi
fi

if [[ ! -x "${RUNTIME_PYTHON}" ]]; then
    "${BASE_PYTHON}" -m venv --system-site-packages "${RUNTIME_DIR}"
fi

if ! "${RUNTIME_PYTHON}" -m pip --version >/dev/null 2>&1; then
    "${BASE_PYTHON}" -m venv --upgrade --system-site-packages "${RUNTIME_DIR}"
fi

"${RUNTIME_PYTHON}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --index-url "${PYPI_INDEX}" \
    --requirement "${REQUIREMENTS_PATH}" || \
    fail "dependency installation failed; install the requirements manually"

"${RUNTIME_PYTHON}" - "${REPO_DIR}" <<'PY'
import sys

repo_dir = sys.argv[1]
sys.path.insert(0, repo_dir + "/pytorch")
import librosa  # noqa: F401
import torch  # noqa: F401
import torchlibrosa  # noqa: F401
from models import Cnn14  # noqa: F401
PY

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    printf '%s\n' \
        "PANNs source and runtime are ready, but the checkpoint is missing." \
        "Manually upload Cnn14_mAP=0.431.pth to:" \
        "  ${CHECKPOINT_PATH}" \
        "Official source:" \
        "  https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" >&2
    exit 2
fi

CHECKPOINT_SIZE="$(stat -c '%s' "${CHECKPOINT_PATH}")"
if (( CHECKPOINT_SIZE < 300000000 )); then
    fail "checkpoint is too small (${CHECKPOINT_SIZE} bytes): ${CHECKPOINT_PATH}"
fi

"${RUNTIME_PYTHON}" - "${CHECKPOINT_PATH}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if not isinstance(checkpoint, dict):
    raise SystemExit("checkpoint must contain an object")
state_dict = checkpoint.get("model")
if not isinstance(state_dict, dict) or not state_dict:
    raise SystemExit("checkpoint is missing the model state dict")
PY

printf 'PANNs background detector is ready.\n'
printf 'Repository: %s\n' "${REPO_DIR}"
printf 'Runtime: %s\n' "${RUNTIME_PYTHON}"
printf 'Checkpoint: %s\n' "${CHECKPOINT_PATH}"
