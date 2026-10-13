#!/usr/bin/env bash
# Single-GPU full benchmark entry; select a physical GPU with GPU_ID (default 0).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_COUNT=1
export GPU_ID="${GPU_ID:-0}"
export SAMPLE_WORKERS="${SAMPLE_WORKERS:-1}"
exec bash "${SCRIPT_DIR}/run_sure_tagger_pipeline.sh" "$@"
