#!/usr/bin/env bash
# Eight-GPU full benchmark entry. Keep all stages in the shared launcher.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_COUNT=8
export GPU_ID=0
export SAMPLE_WORKERS="${SAMPLE_WORKERS:-2}"
exec bash "${SCRIPT_DIR}/run_sure_tagger_pipeline.sh" "$@"
