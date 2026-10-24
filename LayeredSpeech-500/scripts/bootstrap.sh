#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'
"$python_bin" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python scripts/verify_release.py
