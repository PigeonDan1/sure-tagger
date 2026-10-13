#!/usr/bin/env bash
# Shared single/eight-GPU inference, v1 descriptions, and benchmark scoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${COMPARE_ROOT}/evaluation/benchmark-v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${COMPARE_ROOT}/dataset/benchmark}"
GT_ROOT="${GT_ROOT:-${COMPARE_ROOT}/evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-8}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-650}"
DEFAULT_SAMPLE_WORKERS=2
[[ "${GPU_COUNT}" != "1" ]] || DEFAULT_SAMPLE_WORKERS=1
SAMPLE_WORKERS="${SAMPLE_WORKERS:-${DEFAULT_SAMPLE_WORKERS}}"
DEEPSEEK_WORKERS="${DEEPSEEK_WORKERS:-300}"
SCORING_WORKERS="${SCORING_WORKERS:-300}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/sure_tagger_pipeline_$(date +%Y%m%d_%H%M%S)_g${GPU_COUNT}_w${SAMPLE_WORKERS}_ds${DEEPSEEK_WORKERS}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}_staged}"
DEEPSEEK_ROOT="${DEEPSEEK_ROOT:-${OUTPUT_ROOT}_deepseek}"
DESCRIPTIVE_SCORE_ROOT="${DESCRIPTIVE_SCORE_ROOT:-${OUTPUT_ROOT}/descriptive_score}"
SCORING_PROMPT="${SCORING_PROMPT:-${SCRIPT_DIR}/deepseek_descriptive_semantic_scoring_prompt.txt}"
DEEPSEEK_API_FILE="${DEEPSEEK_API_FILE:-}"
SCORING_API_FILE="${SCORING_API_FILE:-${DEEPSEEK_API_FILE}}"
SKIP_STRUCTURAL="${SKIP_STRUCTURAL:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
ALLOW_FROZEN_GT_EXCLUSIONS="${ALLOW_FROZEN_GT_EXCLUSIONS:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: DEEPSEEK_API_FILE=/private/deepseek-api.txt bash run_sure_tagger_pipeline_8.sh
       DEEPSEEK_API_FILE=/private/deepseek-api.txt GPU_ID=3 bash run_sure_tagger_pipeline_1.sh
       DEEPSEEK_API_FILE=/private/deepseek-api.txt bash run_sure_tagger_pipeline_8.sh --check-inputs

Runs the 650-sample audio-only benchmark on eight GPUs (0-7) or one GPU, then v1 DeepSeek
language verification/descriptions, strict merge, descriptive semantic scoring,
and the existing benchmark-scoring-v1.0 structural/contract scoring.

The _1.sh entry selects GPU_COUNT=1, GPU_ID=0, SAMPLE_WORKERS=1 by default.
GPU_ID is a physical nvidia-smi index; child processes see it as cuda:0.
The _8.sh entry selects GPU_COUNT=8, SAMPLE_WORKERS=2 by default.
Both entries set CUDA_VISIBLE_DEVICES explicitly; use GPU_ID for single-GPU selection.
Core and aux run sequentially in separate processes on each selected GPU.
Defaults: DEEPSEEK_WORKERS=300, SCORING_WORKERS=300,
PROMPT_VERSION=v1. Credentials must be supplied through DEEPSEEK_API_FILE;
SCORING_API_FILE optionally selects a separate credential file.

Configurable paths (use absolute paths): SURE_TAGGER_ROOT, BENCHMARK_ROOT,
RUNTIME_ROOT (runtime.manifest.jsonl + audio/), GT_ROOT (items/ checkpoints),
OUTPUT_ROOT, STAGE_ROOT, DEEPSEEK_ROOT, DESCRIPTIVE_SCORE_ROOT, SCORING_PROMPT.
PYTHON_BIN and the structural launcher's model/runtime environment overrides
are passed through. Models, runtime environments, data and API files are external.

Set SKIP_STRUCTURAL=1 and OUTPUT_ROOT to continue a completed structural run.
Set RESUME_EXISTING=1 with the same paths to reuse validated checkpoints.
Use separate OUTPUT_ROOT/STAGE_ROOT paths for single- and eight-GPU runs.
--check-inputs validates benchmark/runtime/GT coverage without model/API calls.
Missing predictions or missing semantic GT fail this launcher by default.
Set ALLOW_FROZEN_GT_EXCLUSIONS=1 to retain the published v1 semantic denominator
of 648: only ES2002b_utterance_00012 and IN1016_utterance_00044 may lack GT.
These exclusions and the actual denominator are recorded in the final summary.
The scoring rules and field exclusions themselves are unchanged.
For a prepared benchmark subset, set EXPECTED_SAMPLE_COUNT to its exact size
(default 650). Benchmark/runtime/GT IDs must still match; this option does not
select samples or allow missing predictions. Subset scores are not full-benchmark scores.

Outputs: result_with_id_descriptive.jsonl, tags_with_descriptive.jsonl,
score.json / score.md, descriptive_score/summary.json, pipeline.full.summary.json.
HELP
    exit 0
fi
CHECK_INPUTS=0
if [[ "${1:-}" == "--check-inputs" && "$#" -eq 1 ]]; then
    CHECK_INPUTS=1
elif [[ "$#" -gt 0 ]]; then
    printf 'unknown argument: %s\n' "$1" >&2
    exit 2
fi
if [[ "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "8" ]]; then
    printf 'GPU_COUNT must be 1 or 8\n' >&2
    exit 2
fi
if [[ ! "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'GPU_ID must be a nonnegative physical GPU index\n' >&2
    exit 2
fi
for value in "${SAMPLE_WORKERS}" "${DEEPSEEK_WORKERS}" "${SCORING_WORKERS}" "${EXPECTED_SAMPLE_COUNT}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'SAMPLE_WORKERS, DEEPSEEK_WORKERS, SCORING_WORKERS and EXPECTED_SAMPLE_COUNT must be positive integers\n' >&2
        exit 2
    fi
done
for value in "${SKIP_STRUCTURAL}" "${RESUME_EXISTING}" "${ALLOW_FROZEN_GT_EXCLUSIONS}"; do
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        printf 'SKIP_STRUCTURAL, RESUME_EXISTING and ALLOW_FROZEN_GT_EXCLUSIONS must be 0 or 1\n' >&2
        exit 2
    fi
done
if [[ "${PROMPT_VERSION}" != "v1" ]]; then
    printf 'this launcher requires PROMPT_VERSION=v1\n' >&2
    exit 2
fi
if ! PYTHON_BIN="$(command -v "${PYTHON_BIN}")"; then
    printf 'PYTHON_BIN must identify an available Python executable\n' >&2
    exit 2
fi
if [[ -z "${DEEPSEEK_API_FILE}" || ! -r "${DEEPSEEK_API_FILE}" || ! -r "${SCORING_API_FILE}" ]]; then
    printf 'set DEEPSEEK_API_FILE (and optionally SCORING_API_FILE) to readable external credential files\n' >&2
    exit 2
fi
if [[ ! -s "${SCORING_PROMPT}" || ! -s "${SCRIPT_DIR}/sure_tagger_deepseek_language_descriptive_prompt.txt" ]]; then
    printf 'required v1 generation/scoring prompt is missing or empty\n' >&2
    exit 2
fi
if [[ "${CHECK_INPUTS}" == "0" && "${RESUME_EXISTING}" == "0" ]]; then
    if [[ -e "${DEEPSEEK_ROOT}" || -e "${DESCRIPTIVE_SCORE_ROOT}" || -e "${OUTPUT_ROOT}/result_with_id_descriptive.jsonl" ]]; then
        printf 'downstream output already exists; choose fresh paths or set RESUME_EXISTING=1\n' >&2
        exit 3
    fi
fi

export SURE_TAGGER_ROOT RUNTIME_ROOT BENCHMARK_ROOT GT_ROOT PYTHON_BIN
export OUTPUT_ROOT STAGE_ROOT DEEPSEEK_ROOT DESCRIPTIVE_SCORE_ROOT
export SAMPLE_WORKERS RESUME_EXISTING DEEPSEEK_WORKERS SCORING_WORKERS PROMPT_VERSION
export GPU_COUNT GPU_ID EXPECTED_SAMPLE_COUNT
export ALLOW_FROZEN_GT_EXCLUSIONS
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Reuse the existing readers and field universes. These checks enforce complete
# execution; they do not turn absent/null predictions into new scoring rules.
validate_coverage() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import os
import sys
from pathlib import Path

from finalize_sure_tagger import _validate_tags, atomic_write
from run_descriptive_semantic_scoring_batch import SCORED_FIELDS, load_ready_ground_truth
from score_benchmark_v1 import CAPABILITY_FIELDS, CONTRACT_FIELDS, build_ground_truth, keyed, load_jsonl

mode = sys.argv[1]
benchmark_root = Path(os.environ["BENCHMARK_ROOT"])
runtime_root = Path(os.environ["RUNTIME_ROOT"])
output_root = Path(os.environ["OUTPUT_ROOT"])
expected = set(keyed(load_jsonl(benchmark_root / "benchmark.manifest.jsonl"),
                     lambda row: row.get("sample", {}).get("sample_id"), "benchmark"))
expected_count = int(os.environ["EXPECTED_SAMPLE_COUNT"])
if len(expected) != expected_count:
    raise SystemExit("expected %d benchmark samples, got %d" % (expected_count, len(expected)))
if mode in {"structural", "complete"}:
    finalized = json.loads((output_root / "finalize.summary.json").read_text())
    if finalized.get("gpu_count") != int(os.environ["GPU_COUNT"]):
        raise SystemExit("structural GPU count differs from this launcher; "
                         "resume with the original entry or use fresh output/stage paths")

def require_coverage(rows, label, required=None):
    required = expected if required is None else required
    actual = set(rows)
    if actual != required:
        raise SystemExit("%s sample coverage mismatch: missing=%d extra=%d; missing IDs=%s" %
                         (label, len(required - actual), len(actual - required),
                          sorted(required - actual)[:10]))

def results(path, label, required=None):
    rows = keyed(load_jsonl(path), lambda row: row.get("sample_id"), label)
    require_coverage(rows, label, required)
    return rows

def semantic_coverage():
    gt_root = Path(os.environ["GT_ROOT"])
    gt_config_path = gt_root / "run_config.json"
    if gt_config_path.is_file():
        gt_config = json.loads(gt_config_path.read_text())
        if gt_config.get("sample_count") != len(expected):
            raise SystemExit("descriptive GT run_config sample_count differs from benchmark; "
                             "prepare a matching GT subset before running")
    ready, failed = load_ready_ground_truth(gt_root / "items", expected)
    excluded = expected - set(ready)
    if excluded and os.environ["ALLOW_FROZEN_GT_EXCLUSIONS"] == "1":
        # Published exclusions: compare/docs/benchmark-v1-29-metrics.md and
        # compare/docs/descriptive-semantic-scoring-grok-4.7.md. No other
        # missing IDs are accepted by this reproduction mode.
        frozen = {"ES2002b_utterance_00012", "IN1016_utterance_00044"}
        gt_summary = json.loads((gt_root / "summary.json").read_text())
        if (excluded != frozen or set(failed) != frozen or
                gt_summary.get("sample_count") != len(expected) or
                gt_summary.get("success_count") != len(ready) or
                gt_summary.get("failed_count") != len(frozen) or
                set(gt_summary.get("failed_sample_ids", [])) != frozen):
            raise SystemExit("GT coverage differs from the frozen v1 exclusions")
    else:
        require_coverage(ready, "descriptive GT")
        if failed:
            raise SystemExit("descriptive GT contains %d failed records" % len(failed))
    return set(ready), sorted(excluded)

if mode == "preflight":
    require_coverage(build_ground_truth(benchmark_root), "structural GT")
    runtime = keyed(load_jsonl(runtime_root / "runtime.manifest.jsonl"),
                    lambda row: row.get("sample", {}).get("sample_id"), "runtime")
    require_coverage(runtime, "runtime")
    if not (runtime_root / "audio").is_dir():
        raise SystemExit("runtime audio directory is missing")
    semantic_ids, excluded = semantic_coverage()
    print("semantic GT denominator=%d/%d; frozen exclusions=%s" %
          (len(semantic_ids), len(expected), excluded))
elif mode == "structural":
    for sample_id, row in results(output_root / "result_with_id.jsonl", "structural output").items():
        _validate_tags(row.get("tags"), sample_id)
elif mode == "complete":
    results(output_root / "result_with_id_descriptive.jsonl", "merged output")
    deepseek_root = Path(os.environ["DEEPSEEK_ROOT"])
    semantic_root = Path(os.environ["DESCRIPTIVE_SCORE_ROOT"])
    results(deepseek_root / "results.jsonl", "DeepSeek output")
    semantic_ids, excluded = semantic_coverage()
    semantic_rows = results(semantic_root / "results.jsonl", "semantic scores", semantic_ids)
    for sample_id, row in semantic_rows.items():
        if row.get("status") != "success" or set(row.get("scores", {})) != set(SCORED_FIELDS):
            raise SystemExit("incomplete semantic scores for %s" % sample_id)
    deepseek = json.loads((deepseek_root / "summary.json").read_text())
    semantic = json.loads((semantic_root / "summary.json").read_text())
    structural = json.loads((output_root / "score.json").read_text())
    if deepseek.get("success_count") != len(expected) or deepseek.get("failed_count") != 0:
        raise SystemExit("DeepSeek summary is incomplete")
    expected_status = "complete_with_missing_ground_truth" if excluded else "complete"
    if (semantic.get("status") != expected_status or
            semantic.get("scoring_success_count") != len(semantic_ids) or
            semantic.get("scoring_failed_count") != 0 or
            semantic.get("score_denominator_sample_count") != len(semantic_ids) or
            semantic.get("excluded_missing_ground_truth_count") != len(excluded) or
            sorted(semantic.get("excluded_missing_ground_truth_sample_ids", [])) != excluded):
        raise SystemExit("semantic scoring summary is incomplete")
    if (structural.get("prediction_row_count") != len(expected) or
            structural.get("missing_result_rows") != 0 or
            structural.get("input_contract_failed") is not False):
        raise SystemExit("structural scoring summary is incomplete")
    summary = {
        "status": "complete_with_frozen_gt_exclusions" if excluded else "complete",
        "sample_count": len(expected), "gpu_count": int(os.environ["GPU_COUNT"]),
        "evaluation_scope": "full_benchmark" if expected_count == 650 else "benchmark_subset",
        "gpu_ids": ([int(os.environ["GPU_ID"])] if os.environ["GPU_COUNT"] == "1"
                    else list(range(8))),
        "sample_workers_per_gpu": int(os.environ["SAMPLE_WORKERS"]),
        "semantic_score_denominator_sample_count": len(semantic_ids),
        "excluded_missing_ground_truth_count": len(excluded),
        "excluded_missing_ground_truth_sample_ids": excluded,
        "prompt_version": os.environ["PROMPT_VERSION"],
        "deepseek_workers": int(os.environ["DEEPSEEK_WORKERS"]),
        "scoring_workers": int(os.environ["SCORING_WORKERS"]),
        "structural_capability_fields": list(CAPABILITY_FIELDS),
        "structural_contract_fields": list(CONTRACT_FIELDS),
        "descriptive_semantic_fields": list(SCORED_FIELDS),
        "structural_score": str((output_root / "score.json").resolve()),
        "semantic_score": str((semantic_root / "summary.json").resolve()),
        "deepseek_summary": str((deepseek_root / "summary.json").resolve()),
        "merged_predictions": str((output_root / "result_with_id_descriptive.jsonl").resolve()),
    }
    atomic_write(output_root / "pipeline.full.summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
else:
    raise SystemExit("unknown validation mode")
print("%s coverage validated: %d samples" % (mode, len(expected)))
PY
}

validate_coverage preflight
[[ "${CHECK_INPUTS}" == "0" ]] || exit 0

if [[ "${SKIP_STRUCTURAL}" == "0" ]]; then
    bash "${SCRIPT_DIR}/run_sure_tagger_nemotron_benchmark.sh"
fi
validate_coverage structural
mkdir -p "${OUTPUT_ROOT}/logs"
MASTER_LOG="${OUTPUT_ROOT}/pipeline.full.log"
trap 'status=$?; printf "[%s] full pipeline failed (exit=%s); see %s/logs\n" "$(date --iso-8601=seconds)" "${status}" "${OUTPUT_ROOT}" >&2; exit "${status}"' ERR

deepseek_status=1
for attempt in 1 2 3; do
    printf '[%s] DeepSeek v1 descriptions attempt=%s workers=%s\n' \
        "$(date --iso-8601=seconds)" "${attempt}" "${DEEPSEEK_WORKERS}" | tee -a "${MASTER_LOG}"
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/run_sure_tagger_deepseek_batch.py" \
        --input-manifest "${OUTPUT_ROOT}/result_with_id.jsonl" \
        --output-dir "${DEEPSEEK_ROOT}" \
        --api-file "${DEEPSEEK_API_FILE}" \
        --prompt-version "${PROMPT_VERSION}" \
        --workers "${DEEPSEEK_WORKERS}" \
        --retries 4 --timeout 300 --max-tokens 20000 --thinking enabled \
        >>"${OUTPUT_ROOT}/logs/deepseek.log" 2>&1; then
        deepseek_status=0
        break
    fi
done
[[ "${deepseek_status}" -eq 0 ]]

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_sure_tagger_descriptive.py" \
    --structural-input "${OUTPUT_ROOT}/result_with_id.jsonl" \
    --deepseek-results "${DEEPSEEK_ROOT}/results.jsonl" \
    --output-dir "${OUTPUT_ROOT}" | tee -a "${MASTER_LOG}"

printf '[%s] descriptive semantic scoring workers=%s\n' \
    "$(date --iso-8601=seconds)" "${SCORING_WORKERS}" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_descriptive_semantic_scoring_batch.py" \
    --ground-truth-dir "${GT_ROOT}" \
    --predictions "${OUTPUT_ROOT}/result_with_id_descriptive.jsonl" \
    --output-dir "${DESCRIPTIVE_SCORE_ROOT}" \
    --api-file "${SCORING_API_FILE}" --prompt "${SCORING_PROMPT}" \
    --workers "${SCORING_WORKERS}" \
    --timeout 300 --max-tokens 12000 --thinking enabled --retries 3 --poll-interval 5 \
    >>"${OUTPUT_ROOT}/logs/deepseek_scoring.log" 2>&1

"${PYTHON_BIN}" "${SCRIPT_DIR}/score_benchmark_v1.py" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --predictions "${OUTPUT_ROOT}/result_with_id_descriptive.jsonl" \
    --output "${OUTPUT_ROOT}/score.json" --report "${OUTPUT_ROOT}/score.md" \
    --system sure-tagger-nemotron-3-pipeline+deepseek-v1 \
    --structural-prediction-owner sure-tagger | tee -a "${MASTER_LOG}"
validate_coverage complete | tee -a "${MASTER_LOG}"
printf '[%s] complete output=%s deepseek=%s semantic_scores=%s\n' \
    "$(date --iso-8601=seconds)" "${OUTPUT_ROOT}" "${DEEPSEEK_ROOT}" "${DESCRIPTIVE_SCORE_ROOT}" | tee -a "${MASTER_LOG}"
