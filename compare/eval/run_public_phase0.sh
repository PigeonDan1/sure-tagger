#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${COMPARE_ROOT}/.." && pwd)"
PHASE_ROOT="${COMPARE_ROOT}/evaluation/public-phase0"
TAGGER_ROOT="${WORKSPACE_ROOT}/sure-tagger-main"
CAPTIONER_RUNNER="${COMPARE_ROOT}/dataset/Ami/captioner_result/run_qwen_omni_captioner.py"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.9}"
CAPTIONER_API_FILE="${CAPTIONER_API_FILE:-${COMPARE_ROOT}/dataset/Ami/captioner_result/api.txt}"

RUN_CAPTIONER=0
OVERWRITE_HEADER=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--captioner] [--overwrite-header]" \
        "" \
        "By default, validate all 30 inputs and create only missing sure-tagger" \
        "header-only outputs. --captioner explicitly enables Qwen Omni API calls." \
        "Existing successful captioner items are resumed without overwriting them." \
        "The final audit exits 2 when Phase 0 artifacts are still incomplete."
}

while (($#)); do
    case "$1" in
        --captioner)
            RUN_CAPTIONER=1
            ;;
        --overwrite-header)
            OVERWRITE_HEADER=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# Fail before either system runs if a manifest, metadata record, or runner glob
# points at audio bytes different from the frozen pilot input.
"${PYTHON_BIN}" - "${PHASE_ROOT}" "${TAGGER_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

phase_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
from tagger.input_schema import validate_input_record

batches = (
    ("ESC-50_phase0", "esc50_pilot.manifest.jsonl", "esc50_pilot.metadata.jsonl", "audio/esc50"),
    (
        "LibriSpeech-test-clean-public-phase0",
        "librispeech_manifest.jsonl",
        "librispeech_pilot_metadata.jsonl",
        "audio",
    ),
    ("FLEURS", "fleurs_pilot.manifest.jsonl", "fleurs_pilot.metadata.jsonl", "audio/fleurs"),
)


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_audio(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (phase_root / path).resolve()


for dataset, manifest_name, metadata_name, audio_dir_name in batches:
    manifest = load_jsonl(phase_root / manifest_name)
    metadata = load_jsonl(phase_root / metadata_name)
    if len(manifest) != 10 or len(metadata) != 10:
        raise SystemExit("%s must contain exactly 10 manifest and metadata rows" % dataset)
    metadata_by_id = {row["sample_id"]: row for row in metadata}
    if len(metadata_by_id) != 10:
        raise SystemExit("%s metadata sample_id values are not unique" % dataset)
    manifest_audio = []
    for record in manifest:
        validate_input_record(record)
        if record["corpus"]["dataset_name"] != dataset:
            raise SystemExit("%s manifest dataset_name mismatch" % dataset)
        sample = record["sample"]
        sample_id = sample["sample_id"]
        truth = metadata_by_id.get(sample_id)
        if truth is None:
            raise SystemExit("%s has no metadata for %s" % (dataset, sample_id))
        input_path = resolve_audio(sample["audio"]["path"])
        metadata_path = resolve_audio(truth["system_input_audio"]["path"])
        if input_path != metadata_path or not input_path.is_file():
            raise SystemExit("%s audio path mismatch for %s" % (dataset, sample_id))
        if sha256(input_path) != truth["system_input_audio"]["sha256"]:
            raise SystemExit("%s audio SHA256 mismatch for %s" % (dataset, sample_id))
        manifest_audio.append(input_path)
    runner_audio = sorted((phase_root / audio_dir_name).glob("*.wav"))
    if sorted(manifest_audio) != [path.resolve() for path in runner_audio]:
        raise SystemExit("%s runner audio glob does not exactly match the manifest" % dataset)
    print("input gate ok: %s (10/10)" % dataset)
PY

run_header() {
    local manifest="$1"
    local output="$2"
    local artifact_dir="$3"
    local expected_lines
    local observed_lines

    if [[ -e "${output}" && "${OVERWRITE_HEADER}" -eq 0 ]]; then
        expected_lines="$(wc -l < "${manifest}")"
        observed_lines="$(wc -l < "${output}")"
        if [[ "${observed_lines}" -ne "${expected_lines}" ]]; then
            printf 'existing header output is incomplete: %s (%s/%s rows)\n' \
                "${output}" "${observed_lines}" "${expected_lines}" >&2
            exit 1
        fi
        printf 'skip existing header output: %s\n' "${output}"
        return
    fi

    (
        cd "${TAGGER_ROOT}"
        "${PYTHON_BIN}" scripts/run_tagger.py \
            --manifest "${manifest}" \
            --output "${output}" \
            --artifact-dir "${artifact_dir}" \
            --only-tags basic_acoustic.duration_sec,basic_acoustic.sample_rate_hz,basic_acoustic.channels
    )
}

run_header \
    "${PHASE_ROOT}/esc50_pilot.manifest.jsonl" \
    "${PHASE_ROOT}/esc50_sure_header.tags.jsonl" \
    "${PHASE_ROOT}/artifacts/esc50_sure_header"
run_header \
    "${PHASE_ROOT}/librispeech_manifest.jsonl" \
    "${PHASE_ROOT}/librispeech_sure_header.tags.jsonl" \
    "${PHASE_ROOT}/artifacts/librispeech_sure_header"
run_header \
    "${PHASE_ROOT}/fleurs_pilot.manifest.jsonl" \
    "${PHASE_ROOT}/fleurs_sure_header.tags.jsonl" \
    "${PHASE_ROOT}/artifacts/fleurs_sure_header"

summarize_and_finish() {
    local status=0
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_public_phase0.py" \
        --root "${PHASE_ROOT}" || status=$?
    # Pilot builders and atomic writers may create mode-0600 files. These
    # artifacts contain no API key and belong to the shared evaluation set.
    find "${PHASE_ROOT}" -type f -exec chmod 0644 {} +
    exit "${status}"
}

if [[ "${RUN_CAPTIONER}" -eq 0 ]]; then
    printf '%s\n' 'captioner disabled; pass --captioner to enable API calls'
    summarize_and_finish
fi

if [[ ! -f "${CAPTIONER_API_FILE}" ]]; then
    printf 'captioner API file not found: %s\n' "${CAPTIONER_API_FILE}" >&2
    exit 1
fi

run_captioner() {
    local dataset="$1"
    local audio_dir="$2"
    local output_dir="$3"

    "${PYTHON_BIN}" "${CAPTIONER_RUNNER}" \
        --audio-dir "${audio_dir}" \
        --output-dir "${output_dir}" \
        --dataset "${dataset}" \
        --workers 2 \
        --retries 2 \
        --timeout 180 \
        --max-tokens 500 \
        --api-file "${CAPTIONER_API_FILE}"
}

# The captioner runner deliberately omits --overwrite. Existing successful
# item/caption pairs are reused, while failed or missing samples are resumed.
run_captioner \
    "ESC-50_phase0" \
    "${PHASE_ROOT}/audio/esc50" \
    "${PHASE_ROOT}/captioner_raw/esc50"
run_captioner \
    "LibriSpeech-test-clean-public-phase0" \
    "${PHASE_ROOT}/audio" \
    "${PHASE_ROOT}/captioner_raw/librispeech"
run_captioner \
    "FLEURS" \
    "${PHASE_ROOT}/audio/fleurs" \
    "${PHASE_ROOT}/captioner_raw/fleurs"

summarize_and_finish
