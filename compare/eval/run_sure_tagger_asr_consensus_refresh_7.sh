#!/usr/bin/env bash
set -euo pipefail

# Refresh ASR-owned fields with the current MOSS + DiariZen single-speaker
# consensus route. Existing speaker structure is restored from BASELINE after
# inference so model reruns cannot change unrelated public fields. Exactly one
# top-level worker runs per GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="${COMPARE_ROOT}/evaluation/benchmark-v1"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-7}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-${BENCHMARK_ROOT}/runtime.manifest.jsonl}"
BASELINE="${BASELINE:-${BENCHMARK_ROOT}/sure_tagger_moss_en_zh_reuse_20260907/result_with_id.jsonl}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARK_ROOT}/sure_tagger_asr_consensus_original_chain_20260908}"
FINAL_ROOT="${FINAL_ROOT:-${BENCHMARK_ROOT}/sure_tagger_asr_consensus_original_chain_final_20260908}"
LOG_ROOT="${RUN_ROOT}/logs"
OUTPUT_ROOT="${RUN_ROOT}/outputs"
ARTIFACT_ROOT="${RUN_ROOT}/artifacts"

if [[ "${GPU_COUNT}" != "7" ]]; then
    printf 'This launcher requires GPU_COUNT=7, got %s\n' "${GPU_COUNT}" >&2
    exit 2
fi
if [[ -e "${RUN_ROOT}" || -e "${FINAL_ROOT}" ]]; then
    printf 'Refusing to overwrite an existing run: %s or %s\n' \
        "${RUN_ROOT}" "${FINAL_ROOT}" >&2
    exit 3
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sure_tagger_asr_refresh.py" \
    --runtime-manifest "${RUNTIME_MANIFEST}" \
    --baseline "${BASELINE}" \
    --output-dir "${RUN_ROOT}" \
    --gpu-count "${GPU_COUNT}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${ARTIFACT_ROOT}" "${FINAL_ROOT}"

pids=()
names=()
terminate_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap 'terminate_children; exit 143' INT TERM

cd "${SURE_TAGGER_ROOT}"
for gpu in $(seq 0 6); do
    name=$(printf '%02d' "${gpu}")
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        export TOKENIZERS_PARALLELISM=false
        printf 'gpu=%s top_level_workers=1 profile=refresh-shadow\n' "${gpu}"
        exec "${PYTHON_BIN}" scripts/run_tagger.py \
            --manifest "${RUN_ROOT}/manifests/manifest_${name}.jsonl" \
            --input-tags "${RUN_ROOT}/baseline/tags_${name}.jsonl" \
            --output "${OUTPUT_ROOT}/tags_${name}.jsonl" \
            --artifact-dir "${ARTIFACT_ROOT}/shard_${name}" \
            --speaker-profile refresh-shadow \
            --only-tags speaker,firered_lid,language_deterministic,deterministic_derived \
            --firered-asr-device cuda:0 \
            --firered-lid-python outputs/gpu_wrappers/firered_lid_gpu_python
    ) >"${LOG_ROOT}/worker_${name}.log" 2>&1 &
    pids+=("$!")
    names+=("gpu_${name}")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '%s finished\n' "${names[$index]}"
    else
        code=$?
        printf '%s failed (exit %s)\n' "${names[$index]}" "${code}" >&2
        status=1
    fi
done
pids=()
names=()
if [[ "${status}" != "0" ]]; then
    printf '%s\n' 'At least one GPU shard failed; finalization and scoring skipped.' >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/finalize_sure_tagger.py" \
    --runtime-manifest "${RUNTIME_MANIFEST}" \
    --manifest-dir "${RUN_ROOT}/manifests" \
    --tags-dir "${OUTPUT_ROOT}" \
    --output-dir "${FINAL_ROOT}" \
    --gpu-count "${GPU_COUNT}"

# The speaker-v2 run must still collect both diarizers for ASR consensus, but
# this refresh owns only the transcript and its language-derived fields. Keep
# the previously validated speaker structure and profiles unchanged.
"${PYTHON_BIN}" "${SCRIPT_DIR}/restore_sure_tagger_asr_result.py" \
    --baseline "${BASELINE}" \
    --refreshed "${FINAL_ROOT}/result_with_id.jsonl" \
    --output-dir "${FINAL_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/score_benchmark_v1.py" \
    --predictions "${FINAL_ROOT}/result_with_id.jsonl" \
    --output "${FINAL_ROOT}/score.json" \
    --report "${FINAL_ROOT}/score.md" \
    --system sure-tagger-asr-consensus-refresh \
    --structural-prediction-owner sure-tagger
