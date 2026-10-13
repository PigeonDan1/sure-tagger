#!/usr/bin/env bash
set -euo pipefail

# Refresh only ASR/diarization-owned structural fields for the active T1-T3
# runtime. Seven top-level processes are launched, exactly one per GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="${COMPARE_ROOT}/evaluation/benchmark-v1"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-7}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-${BENCHMARK_ROOT}/runtime.manifest.jsonl}"
BASELINE="${BASELINE:-${BENCHMARK_ROOT}/sure_tagger/result_with_id.jsonl}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARK_ROOT}/sure_tagger_official_refresh_20260907}"
FINAL_ROOT="${FINAL_ROOT:-${BENCHMARK_ROOT}/sure_tagger_official_refresh_final_20260907}"
LOG_ROOT="${RUN_ROOT}/logs"
ARTIFACT_ROOT="${RUN_ROOT}/artifacts"

force=0
[[ "${1:-}" == "--force" || "${FORCE:-0}" == "1" ]] && force=1
if [[ "${GPU_COUNT}" != "7" ]]; then
    printf 'This benchmark launcher requires GPU_COUNT=7, got %s\n' "${GPU_COUNT}" >&2
    exit 2
fi

if [[ "${force}" == "1" ]]; then
    archive_stamp=$(date '+%Y%m%d_%H%M%S')
    if [[ -e "${RUN_ROOT}" ]]; then
        mv "${RUN_ROOT}" "${RUN_ROOT}_attempt_${archive_stamp}"
    fi
    if [[ -e "${FINAL_ROOT}" ]]; then
        mv "${FINAL_ROOT}" "${FINAL_ROOT}_attempt_${archive_stamp}"
    fi
fi

prepare_args=(
    --runtime-manifest "${RUNTIME_MANIFEST}"
    --baseline "${BASELINE}"
    --output-dir "${RUN_ROOT}"
    --gpu-count "${GPU_COUNT}"
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sure_tagger_asr_refresh.py" "${prepare_args[@]}"

mkdir -p "${LOG_ROOT}" "${ARTIFACT_ROOT}" "${FINAL_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/split_sure_tagger_asr_routes.py" \
    --root "${RUN_ROOT}" --gpu-count "${GPU_COUNT}"

pids=()
names=()
terminate_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap 'terminate_children; exit 143' INT TERM

run_route() {
    local route="$1" profile="$2" gpu name lid_gpu_arg
    pids=()
    names=()
    mkdir -p "${RUN_ROOT}/routes/${route}/outputs"
    for gpu in $(seq 0 6); do
        name=$(printf '%02d' "${gpu}")
        # LID is a routing model and is deliberately kept on CPU. This leaves
        # enough VRAM for the branch model pair on every card, including GPU 6
        # which has a pre-existing external process.
        lid_gpu_arg=()
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export CUDA_DEVICE_ORDER=PCI_BUS_ID
            export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export OMP_NUM_THREADS=2
            export MKL_NUM_THREADS=2
            export TOKENIZERS_PARALLELISM=false
            printf 'route=%s gpu=%s top_level_workers=1\n' "${route}" "${gpu}"
            exec "${PYTHON_BIN}" scripts/run_tagger.py \
                --manifest "${RUN_ROOT}/routes/${route}/manifests/manifest_${name}.jsonl" \
                --input-tags "${RUN_ROOT}/routes/${route}/baseline/tags_${name}.jsonl" \
                --output "${RUN_ROOT}/routes/${route}/outputs/tags_${name}.jsonl" \
                --artifact-dir "${ARTIFACT_ROOT}/${route}_shard_${name}" \
                --speaker-profile "${profile}" \
                --only-tags speaker,firered_lid,language_deterministic,deterministic_derived \
                --firered-asr-device cuda:0 \
                --firered-lid-python outputs/gpu_wrappers/firered_lid_gpu_python \
                "${lid_gpu_arg[@]}"
        ) >"${LOG_ROOT}/${route}_${name}.log" 2>&1 &
        pids+=("$!")
        names+=("${route}_gpu_${name}")
    done

    local status=0 index code
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
    return "${status}"
}

cd "${SURE_TAGGER_ROOT}"
if ! run_route official refresh-official; then
    printf '%s\n' 'Other-language FireRed/DiariZen route failed; MOSS route skipped.' >&2
    exit 1
fi
if ! run_route moss refresh-moss; then
    printf '%s\n' 'Chinese/English MOSS route failed; finalization skipped.' >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_sure_tagger_asr_routes.py" \
    --root "${RUN_ROOT}" --final-root "${RUN_ROOT}/merged_shards" \
    --gpu-count "${GPU_COUNT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/finalize_sure_tagger.py" \
    --runtime-manifest "${RUNTIME_MANIFEST}" \
    --manifest-dir "${RUN_ROOT}/manifests" \
    --tags-dir "${RUN_ROOT}/merged_shards" \
    --output-dir "${FINAL_ROOT}" \
    --gpu-count "${GPU_COUNT}"
