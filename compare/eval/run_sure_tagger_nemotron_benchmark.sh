#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${COMPARE_ROOT}/evaluation/benchmark-v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${COMPARE_ROOT}/dataset/benchmark}"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-8}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-650}"
DEFAULT_SAMPLE_WORKERS=2
[[ "${GPU_COUNT}" != "1" ]] || DEFAULT_SAMPLE_WORKERS=1
SAMPLE_WORKERS="${SAMPLE_WORKERS:-${DEFAULT_SAMPLE_WORKERS}}"
GPU_WRAPPER_ROOT="${GPU_WRAPPER_ROOT:-${SURE_TAGGER_ROOT}/scripts/gpu_wrappers}"
NEMOTRON_MODEL_DIR="${NEMOTRON_MODEL_DIR:-${SURE_TAGGER_ROOT}/models/speaker/nvidia/Nemotron-3-Diarization}"
NEMOTRON_PYTHON="${NEMOTRON_PYTHON:-${SURE_TAGGER_ROOT}/.runtime/nemotron3_diarization_py311_transformers519_torch280_v1/bin/python}"
NEMOTRON_TIMEOUT_SEC="${NEMOTRON_TIMEOUT_SEC:-1800}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/sure_tagger_nemotron3_$(date +%Y%m%d_%H%M%S)_g${GPU_COUNT}_w${SAMPLE_WORKERS}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}_staged}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
MANIFEST_ROOT="${STAGE_ROOT}/manifests"
CORE_OUT="${STAGE_ROOT}/core"
AUX_OUT="${STAGE_ROOT}/aux"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MASTER_LOG="${OUTPUT_ROOT}/pipeline.log"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        "Usage: $0" \
        "  Runs the audio-only 650-sample benchmark with GPU_COUNT=8 (GPUs 0-7)" \
        "  or GPU_COUNT=1 (physical GPU_ID=0 by default; overrides CUDA_VISIBLE_DEVICES)." \
        "  Each GPU pipelines core then aux in separate processes." \
        "  SAMPLE_WORKERS defaults to 2 for eight GPUs and 1 for a single GPU." \
        "  Scores structural labels only, including speaker.gender from speaker profiles." \
        "  EXPECTED_SAMPLE_COUNT defaults to 650; set it explicitly for a prepared subset." \
        "  Set OUTPUT_ROOT to choose a run directory. Set RESUME_EXISTING=1 with the" \
        "  same OUTPUT_ROOT to validate and resume completed GPU shards."
    exit 0
fi
if [[ "$#" -gt 0 ]]; then
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
if [[ ! "${SAMPLE_WORKERS}" =~ ^[1-9][0-9]*$ || ! "${NEMOTRON_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ || ! "${EXPECTED_SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'SAMPLE_WORKERS, NEMOTRON_TIMEOUT_SEC and EXPECTED_SAMPLE_COUNT must be positive integers\n' >&2
    exit 2
fi
if [[ "${RESUME_EXISTING}" != "0" && "${RESUME_EXISTING}" != "1" ]]; then
    printf 'RESUME_EXISTING must be 0 or 1\n' >&2
    exit 2
fi
if [[ "${RESUME_EXISTING}" != "1" && ( -e "${OUTPUT_ROOT}" || -e "${STAGE_ROOT}" ) ]]; then
    printf 'run output already exists; choose a fresh OUTPUT_ROOT or set RESUME_EXISTING=1\n' >&2
    exit 3
fi
if [[ ! -f "${RUNTIME_ROOT}/runtime.manifest.jsonl" || ! -d "${RUNTIME_ROOT}/audio" ]]; then
    printf 'audio-only benchmark runtime is missing under %s\n' "${RUNTIME_ROOT}" >&2
    exit 2
fi
if [[ "$(wc -l < "${RUNTIME_ROOT}/runtime.manifest.jsonl")" -ne "${EXPECTED_SAMPLE_COUNT}" ]]; then
    printf 'expected %s audio-only benchmark rows\n' "${EXPECTED_SAMPLE_COUNT}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -x "${NEMOTRON_PYTHON}" || ! -f "${NEMOTRON_MODEL_DIR}/model.safetensors" ]]; then
    printf 'missing Python executable or local Nemotron checkpoint\n' >&2
    exit 2
fi
for wrapper in firered_gpu_python firered_lid_gpu_python dnsmos_gpu_python dass_gpu_python; do
    if [[ ! -x "${GPU_WRAPPER_ROOT}/${wrapper}" ]]; then
        printf 'missing GPU Python wrapper: %s\n' "${GPU_WRAPPER_ROOT}/${wrapper}" >&2
        exit 2
    fi
done
if ! command -v nvidia-smi >/dev/null || ! available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader)"; then
    printf 'unable to query NVIDIA GPUs with nvidia-smi\n' >&2
    exit 2
fi
selected_gpus=(0 1 2 3 4 5 6 7)
[[ "${GPU_COUNT}" != "1" ]] || selected_gpus=("${GPU_ID}")
for selected_gpu in "${selected_gpus[@]}"; do
    if ! rg -q "^[[:space:]]*${selected_gpu}[[:space:]]*$" <<<"${available_gpus}"; then
        printf 'requested physical GPU %s is unavailable\n' "${selected_gpu}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CORE_OUT}" "${AUX_OUT}"
printf '[%s] preparing %s duration-balanced audio-only shards; physical GPUs=%s\n' \
    "$(date --iso-8601=seconds)" "${GPU_COUNT}" "${selected_gpus[*]}" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sure_tagger_stages.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --output-dir "${MANIFEST_ROOT}" \
    --audio-root "${RUNTIME_ROOT}/audio" \
    --gpu-count "${GPU_COUNT}" \
    --workers-per-gpu "${SAMPLE_WORKERS}" \
    --balance-by-duration | tee -a "${MASTER_LOG}"

configure_gpu_environment() {
    # $1 is the logical shard index; single-GPU shard 00 may run on GPU_ID=3.
    export CUDA_VISIBLE_DEVICES="${selected_gpus[$1]}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export TAGGER_ECAPA_PYTHON="${GPU_WRAPPER_ROOT}/firered_gpu_python"
    export TAGGER_GENDER_DEVICE=cuda:0
    export TAGGER_GENDER_ALICE_PYTHON="${GPU_WRAPPER_ROOT}/dnsmos_gpu_python"
    export TAGGER_GENDER_AFRISPEECH_PYTHON="${GPU_WRAPPER_ROOT}/dnsmos_gpu_python"
    export TAGGER_GENDER_GRIKO_PYTHON="${GPU_WRAPPER_ROOT}/firered_gpu_python"
    export TAGGER_GRIKO_RUNTIME_CACHE="${STAGE_ROOT}/gender_cache/gpu_$1"
}

valid_shard() {
    local stage="$1" name="$2" expected_count output_dir
    output_dir="${CORE_OUT}"
    [[ "${stage}" == "aux" ]] && output_dir="${AUX_OUT}"
    expected_count=$(wc -l < "${MANIFEST_ROOT}/${stage}/manifest_${name}.jsonl")
    [[ -s "${output_dir}/tags_${name}.jsonl" ]] || return 1
    PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c \
        'import sys; from finalize_sure_tagger import _read_tags; _read_tags(sys.argv[1], int(sys.argv[2]))' \
        "${output_dir}/tags_${name}.jsonl" "${expected_count}"
}

run_core() {
    local gpu="$1" name attempt workers log_path expected_count
    name=$(printf '%02d' "${gpu}")
    expected_count=$(wc -l < "${MANIFEST_ROOT}/core/manifest_${name}.jsonl")
    if [[ "${RESUME_EXISTING}" == "1" ]] && \
        rg -q '"processed_sample_count": '"${expected_count}" "${LOG_ROOT}"/core_"${name}"_attempt_*.log 2>/dev/null && \
        valid_shard core "${name}"; then
        printf '[%s] gpu=%s core validated for resume\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
        return 0
    fi
    for attempt in 1 2 3; do
        workers="${SAMPLE_WORKERS}"
        [[ "${attempt}" -gt 1 ]] && workers=1
        log_path="${LOG_ROOT}/core_${name}_attempt_${attempt}.log"
        if (
            configure_gpu_environment "${gpu}"
            printf 'stage=core gpu=%s sample_workers=%s model_workers=1 attempt=%s\n' "${gpu}" "${workers}" "${attempt}"
            exec "${PYTHON_BIN}" scripts/run_tagger.py \
                --manifest "${MANIFEST_ROOT}/core/manifest_${name}.jsonl" \
                --output "${CORE_OUT}/tags_${name}.jsonl" \
                --artifact-dir "${CORE_OUT}/artifacts/shard_${name}" \
                --workers "${workers}" \
                --speaker-profile quality-shadow \
                --only-tags language_deterministic,audio_probe,silence,speaker,brouhaha,firered_lid,deterministic_derived \
                --firered-asr-device cuda:0 \
                --qwen3-asr-device cuda:0 \
                --nemotron-model-dir "${NEMOTRON_MODEL_DIR}" \
                --nemotron-python "${NEMOTRON_PYTHON}" \
                --nemotron-device cuda:0 \
                --nemotron-timeout-sec "${NEMOTRON_TIMEOUT_SEC}" \
                --firered-vad-use-gpu \
                --firered-lid-use-gpu \
                --brouhaha-use-gpu \
                --firered-vad-python "${GPU_WRAPPER_ROOT}/firered_gpu_python" \
                --firered-lid-python "${GPU_WRAPPER_ROOT}/firered_lid_gpu_python" \
                --brouhaha-python "${GPU_WRAPPER_ROOT}/firered_gpu_python"
        ) >"${log_path}" 2>&1 && valid_shard core "${name}"; then
            return 0
        fi
        printf '[%s] gpu=%s core attempt=%s failed validation; retrying\n' "$(date --iso-8601=seconds)" "${gpu}" "${attempt}" | tee -a "${MASTER_LOG}"
    done
    return 1
}

run_aux() {
    local gpu="$1" name attempt workers log_path expected_count
    name=$(printf '%02d' "${gpu}")
    expected_count=$(wc -l < "${MANIFEST_ROOT}/aux/manifest_${name}.jsonl")
    if [[ "${RESUME_EXISTING}" == "1" ]] && \
        rg -q '"processed_sample_count": '"${expected_count}" "${LOG_ROOT}"/aux_"${name}"_attempt_*.log 2>/dev/null && \
        valid_shard aux "${name}"; then
        printf '[%s] gpu=%s aux validated for resume\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
        return 0
    fi
    for attempt in 1 2 3; do
        workers="${SAMPLE_WORKERS}"
        [[ "${attempt}" -gt 1 ]] && workers=1
        log_path="${LOG_ROOT}/aux_${name}_attempt_${attempt}.log"
        if (
            configure_gpu_environment "${gpu}"
            printf 'stage=aux gpu=%s sample_workers=%s model_workers=1 attempt=%s\n' "${gpu}" "${workers}" "${attempt}"
            exec "${PYTHON_BIN}" scripts/run_tagger.py \
                --manifest "${MANIFEST_ROOT}/aux/manifest_${name}.jsonl" \
                --input-tags "${CORE_OUT}/tags_${name}.jsonl" \
                --output "${AUX_OUT}/tags_${name}.jsonl" \
                --artifact-dir "${AUX_OUT}/artifacts/shard_${name}" \
                --workers "${workers}" \
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
        ) >"${log_path}" 2>&1 && ! rg -qi 'out of memory|cuda oom|memoryerror' "${log_path}" && valid_shard aux "${name}"; then
            return 0
        fi
        printf '[%s] gpu=%s aux attempt=%s failed validation; retrying\n' "$(date --iso-8601=seconds)" "${gpu}" "${attempt}" | tee -a "${MASTER_LOG}"
    done
    return 1
}

run_gpu_pipeline() {
    local gpu="$1"
    printf '[%s] gpu=%s core start\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
    run_core "${gpu}" || return 1
    printf '[%s] gpu=%s core done; aux start\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
    run_aux "${gpu}" || return 1
    printf '[%s] gpu=%s aux done\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
}

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
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    run_gpu_pipeline "${gpu}" &
    pids+=("$!")
    names+=("gpu_${gpu}")
done
stage_status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        printf '%s failed after retries\n' "${names[$index]}" | tee -a "${MASTER_LOG}" >&2
        stage_status=1
    fi
done
[[ "${stage_status}" -eq 0 ]] || exit 1
pids=()
names=()

retry_input="${STAGE_ROOT}/retry_samples_round_1.tsv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/find_sure_tagger_retry_samples.py" \
    --artifact-root "${CORE_OUT}/artifacts" \
    --required-source nemotron_3_diarization \
    --expected-manifest-dir "${MANIFEST_ROOT}/core" >"${retry_input}"
for round in 1 2 3; do
    [[ -s "${retry_input}" ]] || break
    printf '[%s] speaker retry round=%s samples=%s\n' \
        "$(date --iso-8601=seconds)" "${round}" "$(wc -l < "${retry_input}")" | tee -a "${MASTER_LOG}"
    retry_artifacts="${STAGE_ROOT}/retries/round_${round}"
    pids=()
    names=()
    for shard_number in $(seq 0 $((GPU_COUNT - 1))); do
        name=$(printf '%02d' "${shard_number}")
        mapfile -t sample_ids < <(awk -F '\t' -v shard="shard_${name}" '$1 == shard {print $2}' "${retry_input}")
        [[ "${#sample_ids[@]}" -gt 0 ]] || continue
        gpu=$(( (shard_number + round - 1) % GPU_COUNT ))
        sample_args=()
        for sample_id in "${sample_ids[@]}"; do
            sample_args+=(--sample-id "${sample_id}")
        done
        (
            configure_gpu_environment "${gpu}"
            exec "${PYTHON_BIN}" scripts/run_tagger.py \
                --manifest "${MANIFEST_ROOT}/core/manifest_${name}.jsonl" \
                --input-tags "${AUX_OUT}/tags_${name}.jsonl" \
                --output "${AUX_OUT}/tags_${name}.jsonl" \
                --artifact-dir "${retry_artifacts}/shard_${name}" \
                --workers 1 \
                "${sample_args[@]}" \
                --speaker-profile quality-shadow \
                --only-tags speaker,firered_lid,language_deterministic,deterministic_derived \
                --firered-asr-device cuda:0 \
                --qwen3-asr-device cuda:0 \
                --nemotron-model-dir "${NEMOTRON_MODEL_DIR}" \
                --nemotron-python "${NEMOTRON_PYTHON}" \
                --nemotron-device cuda:0 \
                --nemotron-timeout-sec "${NEMOTRON_TIMEOUT_SEC}" \
                --firered-vad-use-gpu \
                --firered-lid-use-gpu \
                --brouhaha-use-gpu \
                --firered-vad-python "${GPU_WRAPPER_ROOT}/firered_gpu_python" \
                --firered-lid-python "${GPU_WRAPPER_ROOT}/firered_lid_gpu_python" \
                --brouhaha-python "${GPU_WRAPPER_ROOT}/firered_gpu_python"
        ) >"${LOG_ROOT}/retry_round_${round}_shard_${name}.log" 2>&1 &
        pids+=("$!")
        names+=("retry_${round}_${name}")
    done
    retry_status=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            printf '%s failed\n' "${names[$index]}" | tee -a "${MASTER_LOG}" >&2
            retry_status=1
        fi
    done
    [[ "${retry_status}" -eq 0 ]] || exit 1
    next_retry="${STAGE_ROOT}/retry_samples_round_$((round + 1)).tsv"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/find_sure_tagger_retry_samples.py" \
        --artifact-root "${retry_artifacts}" \
        --required-source nemotron_3_diarization \
        --expected-samples-tsv "${retry_input}" >"${next_retry}"
    retry_input="${next_retry}"
done
if [[ -s "${retry_input}" ]]; then
    printf 'speaker failures remain after three retry rounds: %s\n' "${retry_input}" >&2
    exit 1
fi
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    valid_shard aux "$(printf '%02d' "${gpu}")"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/finalize_sure_tagger.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --manifest-dir "${MANIFEST_ROOT}/core" \
    --tags-dir "${AUX_OUT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --gpu-count "${GPU_COUNT}" | tee -a "${MASTER_LOG}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/score_benchmark_v1.py" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --predictions "${OUTPUT_ROOT}/result_with_id.jsonl" \
    --output "${OUTPUT_ROOT}/score_structural.json" \
    --report "${OUTPUT_ROOT}/score_structural.md" \
    --system sure-tagger-nemotron-3-diarization \
    --structural-prediction-owner sure-tagger | tee -a "${MASTER_LOG}"
printf '[%s] complete; structural score: %s\n' \
    "$(date --iso-8601=seconds)" "${OUTPUT_ROOT}/score_structural.md" | tee -a "${MASTER_LOG}"
