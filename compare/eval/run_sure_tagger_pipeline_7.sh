#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${COMPARE_ROOT}/evaluation/benchmark-v1"
SURE_TAGGER_ROOT="${SURE_TAGGER_ROOT:-${COMPARE_ROOT}/../sure-tagger-run-20260916}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
GPU_COUNT="${GPU_COUNT:-7}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-4}"
DIARIZEN_BATCH_SIZE="${DIARIZEN_BATCH_SIZE:-32}"
RESUME_COMPLETED_CORE="${RESUME_COMPLETED_CORE:-0}"
RESUME_COMPLETED_AUX="${RESUME_COMPLETED_AUX:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
GPU_WRAPPER_ROOT="${GPU_WRAPPER_ROOT:-${COMPARE_ROOT}/../sure-tagger-main/scripts/gpu_wrappers}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/sure_tagger_full_20260916_clean_lid_pipeline}"
STAGE_ROOT="${STAGE_ROOT:-${RUNTIME_ROOT}/sure_tagger_full_20260916_clean_lid_pipeline_staged}"
DEEPSEEK_ROOT="${DEEPSEEK_ROOT:-${RUNTIME_ROOT}/sure_tagger_full_20260916_clean_lid_pipeline_deepseek}"
DESCRIPTIVE_SCORE_ROOT="${DESCRIPTIVE_SCORE_ROOT:-${OUTPUT_ROOT}/descriptive_score}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
MANIFEST_ROOT="${STAGE_ROOT}/manifests"
CORE_OUT="${STAGE_ROOT}/core"
AUX_OUT="${STAGE_ROOT}/aux"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MASTER_LOG="${OUTPUT_ROOT}/pipeline.log"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        "Usage: $0" \
        "  Seven GPU processes, each with a shared-model sample pipeline." \
        "  Defaults: SAMPLE_WORKERS=4, DIARIZEN_BATCH_SIZE=32, DeepSeek workers=100, PROMPT_VERSION=v1." \
        "  Existing final outputs are never overwritten."
    exit 0
fi

if [[ "${GPU_COUNT}" != "7" ]]; then
    printf 'This launcher requires GPU_COUNT=7, got %s\n' "${GPU_COUNT}" >&2
    exit 2
fi
if [[ "${RESUME_EXISTING}" != "1" && ( -e "${OUTPUT_ROOT}/result_with_id.jsonl" || -e "${DEEPSEEK_ROOT}/results.jsonl" ) ]]; then
    printf 'refusing to overwrite an existing completed run under %s or %s\n' \
        "${OUTPUT_ROOT}" "${DEEPSEEK_ROOT}" >&2
    exit 3
fi
for wrapper in firered_gpu_python firered_lid_gpu_python dnsmos_gpu_python dass_gpu_python; do
    if [[ ! -x "${GPU_WRAPPER_ROOT}/${wrapper}" ]]; then
        printf 'missing GPU Python wrapper: %s\n' "${GPU_WRAPPER_ROOT}/${wrapper}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CORE_OUT}" "${AUX_OUT}"
printf '[%s] preparing duration-balanced manifests\n' "$(date --iso-8601=seconds)" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sure_tagger_stages.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --output-dir "${MANIFEST_ROOT}" \
    --audio-root "${RUNTIME_ROOT}/audio" \
    --gpu-count "${GPU_COUNT}" \
    --balance-by-duration | tee -a "${MASTER_LOG}" || exit 1

configure_gpu_environment() {
    local gpu="$1"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export TOKENIZERS_PARALLELISM=false
    export TAGGER_ECAPA_PYTHON="${GPU_WRAPPER_ROOT}/firered_gpu_python"
}

run_core() {
    local gpu="$1" name attempt workers log_path expected_count
    name=$(printf '%02d' "${gpu}")
    if [[ "${RESUME_COMPLETED_CORE}" == "1" && -s "${CORE_OUT}/tags_${name}.jsonl" ]]; then
        expected_count=$(wc -l < "${MANIFEST_ROOT}/core/manifest_${name}.jsonl")
        if rg -q '"processed_sample_count": '"${expected_count}" "${LOG_ROOT}"/core_"${name}"_attempt_*.log &&
            PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c \
                'import sys; from finalize_sure_tagger import _read_tags; _read_tags(sys.argv[1], int(sys.argv[2]))' \
                "${CORE_OUT}/tags_${name}.jsonl" "${expected_count}"; then
            printf '[%s] gpu=%s core validated; resuming at aux\n' \
                "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
            return 0
        fi
    fi
    for attempt in 1 2 3; do
        workers="${SAMPLE_WORKERS}"
        [[ "${attempt}" -gt 1 ]] && workers=1
        log_path="${LOG_ROOT}/core_${name}_attempt_${attempt}.log"
        (
            configure_gpu_environment "${gpu}"
            printf 'stage=core gpu=%s sample_workers=%s model_workers=1 diarizen_batch=%s attempt=%s\n' \
                "${gpu}" "${workers}" "${DIARIZEN_BATCH_SIZE}" "${attempt}"
            exec "${PYTHON_BIN}" scripts/run_tagger.py \
                --manifest "${MANIFEST_ROOT}/core/manifest_${name}.jsonl" \
                --output "${CORE_OUT}/tags_${name}.jsonl" \
                --artifact-dir "${CORE_OUT}/artifacts/shard_${name}" \
                --workers "${workers}" \
                --speaker-profile quality-shadow \
                --only-tags language_deterministic,audio_probe,silence,speaker,brouhaha,firered_lid,deterministic_derived \
                --firered-asr-device cuda:0 \
                --diarizen-batch-size "${DIARIZEN_BATCH_SIZE}" \
                --firered-vad-use-gpu \
                --firered-lid-use-gpu \
                --brouhaha-use-gpu \
                --firered-vad-python "${GPU_WRAPPER_ROOT}/firered_gpu_python" \
                --firered-lid-python "${GPU_WRAPPER_ROOT}/firered_lid_gpu_python" \
                --brouhaha-python "${GPU_WRAPPER_ROOT}/firered_gpu_python"
        ) >"${log_path}" 2>&1
        if [[ "$?" -eq 0 ]]; then
            return 0
        fi
        printf '[%s] core shard=%s attempt=%s failed; retrying with one sample worker\n' \
            "$(date --iso-8601=seconds)" "${name}" "${attempt}" | tee -a "${MASTER_LOG}"
    done
    return 1
}

run_aux() {
    local gpu="$1" name attempt workers log_path expected_count
    name=$(printf '%02d' "${gpu}")
    if [[ "${RESUME_COMPLETED_AUX}" == "1" && -s "${AUX_OUT}/tags_${name}.jsonl" ]]; then
        expected_count=$(wc -l < "${MANIFEST_ROOT}/aux/manifest_${name}.jsonl")
        if rg -q '"processed_sample_count": '"${expected_count}" "${LOG_ROOT}"/aux_"${name}"_attempt_*.log &&
            PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c \
                'import sys; from finalize_sure_tagger import _read_tags; _read_tags(sys.argv[1], int(sys.argv[2]))' \
                "${AUX_OUT}/tags_${name}.jsonl" "${expected_count}"; then
            printf '[%s] gpu=%s aux validated; resuming after aux\n' \
                "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
            return 0
        fi
    fi
    for attempt in 1 2 3; do
        workers="${SAMPLE_WORKERS}"
        [[ "${attempt}" -gt 1 ]] && workers=1
        log_path="${LOG_ROOT}/aux_${name}_attempt_${attempt}.log"
        (
            configure_gpu_environment "${gpu}"
            printf 'stage=aux gpu=%s sample_workers=%s model_workers=1 attempt=%s\n' \
                "${gpu}" "${workers}" "${attempt}"
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
        ) >"${log_path}" 2>&1
        if [[ "$?" -eq 0 ]] && ! rg -qi 'out of memory|cuda oom|memoryerror' "${log_path}"; then
            return 0
        fi
        printf '[%s] aux shard=%s attempt=%s had a process/OOM failure; retrying\n' \
            "$(date --iso-8601=seconds)" "${name}" "${attempt}" | tee -a "${MASTER_LOG}"
    done
    return 1
}

run_gpu_pipeline() {
    local gpu="$1" name
    name=$(printf '%02d' "${gpu}")
    printf '[%s] gpu=%s core start\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
    run_core "${gpu}" || return 1
    printf '[%s] gpu=%s core done, aux start\n' "$(date --iso-8601=seconds)" "${gpu}" | tee -a "${MASTER_LOG}"
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
for gpu in $(seq 0 6); do
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
    --artifact-root "${CORE_OUT}/artifacts" >"${retry_input}"
for round in 1 2 3; do
    [[ -s "${retry_input}" ]] || break
    printf '[%s] speaker/OOM retry round=%s samples=%s\n' \
        "$(date --iso-8601=seconds)" "${round}" "$(wc -l < "${retry_input}")" | tee -a "${MASTER_LOG}"
    retry_artifacts="${STAGE_ROOT}/retries/round_${round}"
    retry_status=0
    pids=()
    names=()
    for shard_number in $(seq 0 6); do
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
                --diarizen-batch-size "${DIARIZEN_BATCH_SIZE}" \
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
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            printf '%s failed\n' "${names[$index]}" | tee -a "${MASTER_LOG}" >&2
            retry_status=1
        fi
    done
    [[ "${retry_status}" -eq 0 ]] || exit 1
    next_retry="${STAGE_ROOT}/retry_samples_round_$((round + 1)).tsv"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/find_sure_tagger_retry_samples.py" \
        --artifact-root "${retry_artifacts}" >"${next_retry}"
    retry_input="${next_retry}"
done
if [[ -s "${retry_input}" ]]; then
    printf 'speaker/OOM failures remain after three retry rounds: %s\n' "${retry_input}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/finalize_sure_tagger.py" \
    --runtime-manifest "${RUNTIME_ROOT}/runtime.manifest.jsonl" \
    --manifest-dir "${MANIFEST_ROOT}/core" \
    --tags-dir "${AUX_OUT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --gpu-count "${GPU_COUNT}" | tee -a "${MASTER_LOG}" || exit 1

deepseek_status=1
for attempt in 1 2 3; do
    printf '[%s] DeepSeek descriptive attempt=%s workers=100\n' \
        "$(date --iso-8601=seconds)" "${attempt}" | tee -a "${MASTER_LOG}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_sure_tagger_deepseek_batch.py" \
        --input-manifest "${OUTPUT_ROOT}/result_with_id.jsonl" \
        --output-dir "${DEEPSEEK_ROOT}" \
        --prompt-version "${PROMPT_VERSION}" \
        --workers 100 \
        --retries 4 \
        --timeout 300 \
        --max-tokens 20000 \
        --thinking enabled >>"${LOG_ROOT}/deepseek.log" 2>&1
    deepseek_status=$?
    [[ "${deepseek_status}" -eq 0 ]] && break
done
[[ "${deepseek_status}" -eq 0 ]] || exit 1

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_sure_tagger_descriptive.py" \
    --structural-input "${OUTPUT_ROOT}/result_with_id.jsonl" \
    --deepseek-results "${DEEPSEEK_ROOT}/results.jsonl" \
    --output-dir "${OUTPUT_ROOT}" | tee -a "${MASTER_LOG}" || exit 1

printf '[%s] DeepSeek descriptive scoring workers=100\n' \
    "$(date --iso-8601=seconds)" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_descriptive_semantic_scoring_batch.py" \
    --ground-truth-dir "${COMPARE_ROOT}/evaluation/qwen3_5_omni_plus_descriptive_gt_official_assisted_benchmark_t1_t3" \
    --predictions "${OUTPUT_ROOT}/result_with_id_descriptive.jsonl" \
    --output-dir "${DESCRIPTIVE_SCORE_ROOT}" \
    --workers 100 \
    --timeout 300 \
    --max-tokens 12000 \
    --thinking enabled \
    --retries 3 \
    --poll-interval 5 \
    >>"${LOG_ROOT}/deepseek_scoring.log" 2>&1 || exit 1

"${PYTHON_BIN}" "${SCRIPT_DIR}/score_benchmark_v1.py" \
    --benchmark-root "${COMPARE_ROOT}/dataset/benchmark" \
    --predictions "${OUTPUT_ROOT}/result_with_id_descriptive.jsonl" \
    --output "${OUTPUT_ROOT}/score.json" \
    --report "${OUTPUT_ROOT}/score.md" \
    --system sure-tagger-clean-lid-pipeline+deepseek-v4-pro \
    --structural-prediction-owner sure-tagger | tee -a "${MASTER_LOG}" || exit 1
printf '[%s] complete output=%s deepseek=%s\n' \
    "$(date --iso-8601=seconds)" "${OUTPUT_ROOT}" "${DEEPSEEK_ROOT}" | tee -a "${MASTER_LOG}"
