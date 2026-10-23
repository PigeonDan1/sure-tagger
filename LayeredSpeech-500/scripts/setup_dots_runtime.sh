#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3}"
runtime_root="${DOTS_RUNTIME_ROOT:-models/runtime}"
repo_url="$("$python_bin" -c 'import yaml; print(yaml.safe_load(open("data/models.lock.yaml"))["runtime"]["repository"])')"
commit="$("$python_bin" -c 'import yaml; print(yaml.safe_load(open("data/models.lock.yaml"))["runtime"]["commit"])')"

if [[ ! -d "$runtime_root/source/.git" ]]; then
  git clone "$repo_url" "$runtime_root/source"
fi
git -C "$runtime_root/source" fetch origin "$commit"
git -C "$runtime_root/source" checkout --detach "$commit"

"$python_bin" -m venv "$runtime_root/.venv"
"$runtime_root/.venv/bin/python" -m pip install --upgrade pip
"$runtime_root/.venv/bin/python" -m pip install -e "$runtime_root/source" \
  -c "$runtime_root/source/constraints/recommended.txt"

echo "Dots runtime ready at $runtime_root/.venv"
