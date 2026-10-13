#!/usr/bin/env bash
set -u
set -o pipefail

# Compatibility entry point. The historical launcher started core and aux
# workers concurrently on each GPU; delegate to the memory-safe serial runner
# so every GPU has at most one top-level worker.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_sure_tagger_7.sh" "$@"
