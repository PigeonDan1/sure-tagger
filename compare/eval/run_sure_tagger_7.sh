#!/usr/bin/env bash
set -u
set -o pipefail

# Run the full sure-tagger profile in two serial stages.  There is at most one
# top-level worker per GPU at any time; serializing the stages lets each group
# of model subprocesses release its CUDA allocations before the next group is
# loaded.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${COMPARE_ROOT}/evaluation/benchmark-v1"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-7}"
GPU_WRAPPER_ROOT="${GPU_WRAPPER_ROOT:-${SURE_TAGGER_ROOT}/outputs/gpu_wrappers}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/sure_tagger}"
STAGE_ROOT="${STAGE_ROOT:-${RUNTIME_ROOT}/sure_tagger_staged}"
MANIFEST_ROOT="${STAGE_ROOT}/manifests"
CORE_OUT="${STAGE_ROOT}/core"
AUX_OUT="${STAGE_ROOT}/aux"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' "Usage: $0 [--force]" \
        "  FORCE=1 or --force permits replacing existing stage/final outputs." \
        "  One worker is launched per GPU; core and aux stages run serially." \
        "  Requests and manifests are audio-only."
    exit 0
fi

force=0
[[ "${1:-}" == "--force" || "${FORCE:-0}" == "1" ]] && force=1

for wrapper in firered_gpu_python firered_lid_gpu_python dnsmos_gpu_python dass_gpu_python; do
    [[ -x "${GPU_WRAPPER_ROOT}/${wrapper}" ]] || {
        printf 'missing GPU Python wrapper: %s\n' "${GPU_WRAPPER_ROOT}/${wrapper}" >&2
        exit 2
    }
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CORE_OUT}/artifacts" "${AUX_OUT}/artifacts"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sure_tagger_stages.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --output-dir "${MANIFEST_ROOT}" \
    --audio-root "${RUNTIME_ROOT}/audio" \
    --gpu-count "${GPU_COUNT}"

for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    name=$(printf '%02d' "${gpu}")
    core_manifest="${MANIFEST_ROOT}/core/manifest_${name}.jsonl"
    aux_manifest="${MANIFEST_ROOT}/aux/manifest_${name}.jsonl"
    core_output="${CORE_OUT}/tags_${name}.jsonl"
    aux_output="${AUX_OUT}/tags_${name}.jsonl"
    [[ -s "${core_manifest}" ]] || { printf 'missing manifest: %s\n' "${core_manifest}" >&2; exit 2; }
    [[ -s "${aux_manifest}" ]] || { printf 'missing manifest: %s\n' "${aux_manifest}" >&2; exit 2; }
    if [[ "${force}" != "1" ]]; then
        [[ ! -e "${core_output}" ]] || { printf 'refusing to overwrite: %s\n' "${core_output}" >&2; exit 3; }
        [[ ! -e "${aux_output}" ]] || { printf 'refusing to overwrite: %s\n' "${aux_output}" >&2; exit 3; }
    fi
done
if [[ "${force}" != "1" && -e "${OUTPUT_ROOT}/tags.jsonl" ]]; then
    printf 'refusing to overwrite final output: %s\n' "${OUTPUT_ROOT}/tags.jsonl" >&2
    exit 3
fi

if [[ "${force}" == "1" ]]; then
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        name=$(printf '%02d' "${gpu}")
        : > "${CORE_OUT}/tags_${name}.jsonl"
        : > "${AUX_OUT}/tags_${name}.jsonl"
        rm -rf "${CORE_OUT}/artifacts/shard_${name}" "${AUX_OUT}/artifacts/shard_${name}"
        mkdir -p "${CORE_OUT}/artifacts/shard_${name}" "${AUX_OUT}/artifacts/shard_${name}"
    done
    rm -f "${OUTPUT_ROOT}/tags.jsonl" "${OUTPUT_ROOT}/result_with_id.jsonl" \
        "${OUTPUT_ROOT}/finalize.summary.json"
fi

pids=()
names=()
terminate_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap 'terminate_children; exit 143' INT TERM

launch_core() {
    local gpu="$1" name
    name=$(printf '%02d' "${gpu}")
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        export TOKENIZERS_PARALLELISM=false
        export TAGGER_ECAPA_PYTHON="${GPU_WRAPPER_ROOT}/firered_gpu_python"
        printf 'stage=core gpu=%s worker_count=1 audio_only=true profile=quality-shadow\n' "${gpu}"
        exec "${PYTHON_BIN}" scripts/run_tagger.py \
            --manifest "${MANIFEST_ROOT}/core/manifest_${name}.jsonl" \
            --output "${CORE_OUT}/tags_${name}.jsonl" \
            --artifact-dir "${CORE_OUT}/artifacts/shard_${name}" \
            --speaker-profile quality-shadow \
            --only-tags language_deterministic,audio_probe,silence,speaker,brouhaha,firered_lid,deterministic_derived \
            --firered-asr-device cuda:0 \
            --diarizen-batch-size 8 \
            --firered-vad-use-gpu \
            --firered-lid-use-gpu \
            --brouhaha-use-gpu \
            --firered-vad-python "${GPU_WRAPPER_ROOT}/firered_gpu_python" \
            --firered-lid-python "${GPU_WRAPPER_ROOT}/firered_lid_gpu_python" \
            --brouhaha-python "${GPU_WRAPPER_ROOT}/firered_gpu_python"
    ) >"${LOG_ROOT}/core_${name}.log" 2>&1 &
    pids+=("$!")
    names+=("core_gpu_${name}")
}

launch_aux() {
    local gpu="$1" name
    name=$(printf '%02d' "${gpu}")
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        export TOKENIZERS_PARALLELISM=false
        export TAGGER_ECAPA_PYTHON="${GPU_WRAPPER_ROOT}/firered_gpu_python"
        printf 'stage=aux gpu=%s worker_count=1 audio_only=true profile=quality-shadow\n' "${gpu}"
        exec "${PYTHON_BIN}" scripts/run_tagger.py \
            --manifest "${MANIFEST_ROOT}/aux/manifest_${name}.jsonl" \
            --input-tags "${CORE_OUT}/tags_${name}.jsonl" \
            --output "${AUX_OUT}/tags_${name}.jsonl" \
            --artifact-dir "${AUX_OUT}/artifacts/shard_${name}" \
            --speaker-profile quality-shadow \
            --only-tags dnsmos,firered_aed,dass,recrir \
            --missing-only \
            --firered-aed-use-gpu \
            --recrir-use-gpu \
            --dnsmos-use-gpu \
            --dass-use-gpu \
            --firered-aed-python "${GPU_WRAPPER_ROOT}/firered_gpu_python" \
            --dnsmos-python "${GPU_WRAPPER_ROOT}/dnsmos_gpu_python" \
            --dass-python "${GPU_WRAPPER_ROOT}/dass_gpu_python"
    ) >"${LOG_ROOT}/aux_${name}.log" 2>&1 &
    pids+=("$!")
    names+=("aux_gpu_${name}")
}

wait_stage() {
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
for gpu in $(seq 0 $((GPU_COUNT - 1))); do launch_core "${gpu}"; done
if ! wait_stage; then
    printf '%s\n' 'sure-tagger core stage failed; aux stage skipped' >&2
    exit 1
fi

# The core subprocesses are all closed by run_tagger before this point.  Only
# then load the independent acoustic-scene models for the aux stage.
for gpu in $(seq 0 $((GPU_COUNT - 1))); do launch_aux "${gpu}"; done
if ! wait_stage; then
    printf '%s\n' 'sure-tagger aux stage failed; finalization skipped' >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/finalize_sure_tagger.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --manifest-dir "${MANIFEST_ROOT}/core" \
    --tags-dir "${AUX_OUT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --gpu-count "${GPU_COUNT}"
