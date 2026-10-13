#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_ami_gt.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/adapt_outputs.py" --system sure-tagger
"${PYTHON_BIN}" "${SCRIPT_DIR}/adapt_outputs.py" --system captioner
"${PYTHON_BIN}" "${SCRIPT_DIR}/score_structural.py"

# Atomic writers create mode-0600 temporary files. Keep the shared-workspace
# evaluation artifacts readable after every successful rebuild.
find "${SCRIPT_DIR}/../evaluation/ami-v0" -maxdepth 1 -type f \
    -exec chmod 0644 {} +
