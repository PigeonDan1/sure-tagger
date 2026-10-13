#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
WORKERS="${WORKERS:-200}"
MODEL="${MODEL:-grok-4.7}"
API_FILE="${API_FILE:-${COMPARE_ROOT}/api/grok.txt}"
GT_DIR="${GT_DIR:-${COMPARE_ROOT}/evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3}"
PROMPT="${PROMPT:-${SCRIPT_DIR}/deepseek_descriptive_semantic_scoring_prompt.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${COMPARE_ROOT}/evaluation/descriptive-semantic-scoring-${MODEL}}"
SYSTEMS="${SYSTEMS:-sure-tagger captioner omni}"
BACKGROUND="${BACKGROUND:-0}"
DRY_RUN="${DRY_RUN:-0}"

SURE_TAGGER_PREDICTIONS="${SURE_TAGGER_PREDICTIONS:-${COMPARE_ROOT}/evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl}"
CAPTIONER_PREDICTIONS="${CAPTIONER_PREDICTIONS:-${COMPARE_ROOT}/docs/report/result/qwen-captioner/results_t1_t3.jsonl}"
OMNI_PREDICTIONS="${OMNI_PREDICTIONS:-${COMPARE_ROOT}/docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl}"

mkdir -p "${OUTPUT_ROOT}"

run_one() {
    local name="$1"
    local predictions="$2"
    local system_label="$3"
    local output_dir="${OUTPUT_ROOT}/${name}"
    local log_file="${output_dir}/run.log"
    local pid_file="${output_dir}/run.pid"
    mkdir -p "${output_dir}"

    printf '[%s] start system=%s model=%s workers=%s predictions=%s output=%s\n' \
        "$(date --iso-8601=seconds)" "${system_label}" "${MODEL}" "${WORKERS}" \
        "${predictions}" "${output_dir}" | tee -a "${log_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_descriptive_semantic_scoring_batch.py" \
            --ground-truth-dir "${GT_DIR}" \
            --predictions "${predictions}" \
            --output-dir "${output_dir}" \
            --prompt "${PROMPT}" \
            --api-file "${API_FILE}" \
            --model "${MODEL}" \
            --system "${system_label}" \
            --workers "${WORKERS}" \
            --timeout 300 \
            --max-tokens 12000 \
            --thinking enabled \
            --retries 3 \
            --poll-interval 5 \
            --dry-run 2>&1 | tee -a "${log_file}"
        return "${PIPESTATUS[0]}"
    fi

    if [[ "${BACKGROUND}" == "1" ]]; then
        setsid -f bash -c "
            echo \$\$ > '${pid_file}'
            exec '${PYTHON_BIN}' '${SCRIPT_DIR}/run_descriptive_semantic_scoring_batch.py' \
                --ground-truth-dir '${GT_DIR}' \
                --predictions '${predictions}' \
                --output-dir '${output_dir}' \
                --prompt '${PROMPT}' \
                --api-file '${API_FILE}' \
                --model '${MODEL}' \
                --system '${system_label}' \
                --workers '${WORKERS}' \
                --timeout 300 \
                --max-tokens 12000 \
                --thinking enabled \
                --retries 3 \
                --poll-interval 5 \
                >> '${log_file}' 2>&1
        "
        sleep 0.3
        printf '[%s] background pid=%s log=%s\n' \
            "$(date --iso-8601=seconds)" "$(cat "${pid_file}" 2>/dev/null || echo unknown)" "${log_file}" \
            | tee -a "${log_file}"
        return 0
    fi

    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_descriptive_semantic_scoring_batch.py" \
        --ground-truth-dir "${GT_DIR}" \
        --predictions "${predictions}" \
        --output-dir "${output_dir}" \
        --prompt "${PROMPT}" \
        --api-file "${API_FILE}" \
        --model "${MODEL}" \
        --system "${system_label}" \
        --workers "${WORKERS}" \
        --timeout 300 \
        --max-tokens 12000 \
        --thinking enabled \
        --retries 3 \
        --poll-interval 5 \
        2>&1 | tee -a "${log_file}"
}

status=0
for name in ${SYSTEMS}; do
    case "${name}" in
        sure-tagger|sure-tagger-v1)
            run_one "sure-tagger-v1" "${SURE_TAGGER_PREDICTIONS}" "sure-tagger-v1+${MODEL}" || status=$?
            ;;
        captioner|qwen-captioner)
            run_one "qwen-captioner" "${CAPTIONER_PREDICTIONS}" "qwen-captioner+${MODEL}" || status=$?
            ;;
        omni|qwen3-omni)
            run_one "qwen3-omni" "${OMNI_PREDICTIONS}" "qwen3-omni+${MODEL}" || status=$?
            ;;
        *)
            printf 'unknown system: %s\n' "${name}" >&2
            status=2
            ;;
    esac
done

exit "${status}"
