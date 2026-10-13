"""Utterance-level speaker evidence pipeline used by the v2 shadow profile."""

from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from tagger.input_schema import validate_input_record
from tagger.tools.acoustic_io import probe_audio_info
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
    FireRedVadConfig,
    run as run_firered_vad,
)
from tagger.tools.audio_quality.brouhaha_signal_estimator import (
    BrouhahaConfig,
)
from tagger.tools.language_content.firered_lid_detector import (
    FireRedLidConfig,
    run as run_firered_lid,
)
from tagger.tools.language_content.speaker_segment_lid import (
    SpeakerSegmentLidConfig,
    TOOL_VERSION as SPEAKER_SEGMENT_LID_VERSION,
    run as run_speaker_segment_lid,
)
from tagger.tools.speaker.moss_diarizer import (
    MossDiarizeConfig,
    parse_moss_text,
    run as run_moss_diarizer,
)
from tagger.tools.speaker_v2.firered_asr import (
    FireRedAsrConfig,
    run as run_firered_asr,
)
from tagger.tools.speaker_v2.qwen3_asr import (
    CHECKPOINT_SHA256 as QWEN3_ASR_MODEL_SHA256,
    SOURCE_VERSION as QWEN3_ASR_SOURCE_VERSION,
    Qwen3AsrConfig,
    run as run_qwen3_asr,
)
from tagger.tools.speaker_v2.nemotron3_diarization import (
    CONFIG_SHA256 as NEMOTRON_CONFIG_SHA256,
    DEFAULT_MODEL_DIR as DEFAULT_NEMOTRON_MODEL_DIR,
    DEFAULT_SUBPROCESS_PYTHON as DEFAULT_NEMOTRON_PYTHON,
    DEPENDENCY_GROUPS as NEMOTRON_DEPENDENCY_GROUPS,
    MODEL_SHA256 as NEMOTRON_MODEL_SHA256,
    MODEL_SLOT_KEY as NEMOTRON_MODEL_SLOT_KEY,
    MODEL_VERSION as NEMOTRON_MODEL_VERSION,
    PROCESSOR_CONFIG_SHA256 as NEMOTRON_PROCESSOR_CONFIG_SHA256,
    SOURCE_KIND as NEMOTRON_SOURCE_KIND,
    SOURCE_NAME as NEMOTRON_SOURCE_NAME,
    Nemotron3DiarizationConfig,
    diarize as diarize_nemotron,
)
from tagger.tools.speaker_v2.diarizen_official import (
    CODE_VERSION as DIARIZEN_OFFICIAL_CODE_VERSION,
    CONFIG_SHA256 as DIARIZEN_OFFICIAL_CONFIG_SHA256,
    DEFAULT_EMBEDDING_MODEL as DEFAULT_DIARIZEN_OFFICIAL_EMBEDDING_MODEL,
    DEFAULT_MODEL_DIR as DEFAULT_DIARIZEN_OFFICIAL_MODEL_DIR,
    DEFAULT_SOURCE_DIR as DEFAULT_DIARIZEN_OFFICIAL_SOURCE_DIR,
    DEFAULT_SUBPROCESS_PYTHON as DEFAULT_DIARIZEN_OFFICIAL_PYTHON,
    DEPENDENCY_GROUPS as DIARIZEN_OFFICIAL_DEPENDENCY_GROUPS,
    MODEL_SHA256 as DIARIZEN_OFFICIAL_MODEL_SHA256,
    MODEL_SLOT_KEY as DIARIZEN_OFFICIAL_MODEL_SLOT_KEY,
    MODEL_VERSION as DIARIZEN_OFFICIAL_MODEL_VERSION,
    PLDA_SHA256 as DIARIZEN_OFFICIAL_PLDA_SHA256,
    SOURCE_KIND as DIARIZEN_OFFICIAL_SOURCE_KIND,
    SOURCE_NAME as DIARIZEN_OFFICIAL_SOURCE_NAME,
    WESPEAKER_SHA256 as DIARIZEN_OFFICIAL_WESPEAKER_SHA256,
    WESPEAKER_VERSION as DIARIZEN_OFFICIAL_WESPEAKER_VERSION,
    XVEC_TRANSFORM_SHA256 as DIARIZEN_OFFICIAL_XVEC_TRANSFORM_SHA256,
    OfficialDiariZenConfig,
    diarize as diarize_official_diarizen,
)
from tagger.tools.speaker_v2.diarizen_timeline import (
    DEFAULT_DIARIZEN_PACK,
    DEFAULT_FIRERED_ASR_PACK,
    DEFAULT_FORCED_ALIGNER_PACK,
    DEFAULT_REDIMNET_PACK,
    DEFAULT_RUNTIME_EXECUTABLE,
    FIRERED_ASR_MODEL_VERSION as OPENASR_FIRERED_ASR_MODEL_VERSION,
    FIRERED_ASR_PACK_SIZE_BYTES as OPENASR_FIRERED_ASR_PACK_SIZE_BYTES,
    FIRERED_ASR_PACK_SHA256 as OPENASR_FIRERED_ASR_PACK_SHA256,
    FORCED_ALIGNER_MODEL_VERSION,
    FORCED_ALIGNER_PACK_SIZE_BYTES,
    FORCED_ALIGNER_PACK_SHA256,
    MODEL_PACK_SIZE_BYTES as DIARIZEN_MODEL_PACK_SIZE_BYTES,
    MODEL_PACK_SHA256 as DIARIZEN_MODEL_PACK_SHA256,
    MODEL_SLOT_KEY as OPENASR_MODEL_SLOT_KEY,
    MODEL_VERSION as DIARIZEN_MODEL_VERSION,
    OPENASR_EXECUTABLE_SHA256,
    OPENASR_EXECUTABLE_SIZE_BYTES,
    OPENASR_VERSION,
    REDIMNET_MODEL_VERSION,
    REDIMNET_PACK_SIZE_BYTES,
    REDIMNET_PACK_SHA256,
    DiariZenTimelineConfig,
    diarize as diarize_diarizen,
)
from tagger.tools.speaker_v2.artifacts import (
    write_run_manifest,
    write_sample_artifacts,
)
from tagger.tools.speaker_v2.campplus_identity import (
    CHECKPOINT_SHA256 as CAMPPLUS_CHECKPOINT_SHA256,
    CampPlusIdentityConfig,
    compare as compare_campplus_regions,
)
from tagger.tools.speaker_v2.brouhaha_coverage import (
    BINARIZATION_PARAMETERS as BROUHAHA_BINARIZATION_PARAMETERS,
    DEPENDENCY_GROUPS as BROUHAHA_DEPENDENCY_GROUPS,
    MODEL_ID as BROUHAHA_MODEL_ID,
    MODEL_SHA256 as BROUHAHA_MODEL_SHA256,
    estimate_coverage as estimate_brouhaha_coverage,
    verify_model_asset as verify_brouhaha_model_asset,
)
from tagger.tools.speaker_v2.contracts import (
    build_evidence,
    build_missing_evidence,
)
from tagger.tools.speaker_v2.hypotheses import (
    build_count_hypothesis_case,
    evaluate_count_hypothesis_case,
)
from tagger.tools.speaker_v2.ecapa_identity import (
    CHECKPOINT_SHA256 as ECAPA_CHECKPOINT_SHA256,
    HYPERPARAMS_SHA256 as ECAPA_HYPERPARAMS_SHA256,
    MODEL_VERSION as ECAPA_MODEL_VERSION,
    EcapaIdentityConfig,
    compare as compare_ecapa_regions,
)
from tagger.tools.speaker_v2.lexical import (
    project_asr_track,
    speaker_assignment_comparison,
    speaker_text_track,
    text_track_comparison,
)
from tagger.tools.speaker_v2.pyannote_community1 import (
    MODEL_REVISION as PYANNOTE_MODEL_REVISION,
    MODEL_SHA256 as PYANNOTE_MODEL_SHA256,
    MODEL_VERSION as PYANNOTE_MODEL_VERSION,
    PyannoteCommunity1Config,
    diarize as diarize_pyannote,
    verify_model_assets as verify_pyannote_model_assets,
)
from tagger.tools.speaker_v2.profiles import (
    claim_policy_hash,
    expand_profile,
    validate_claim_policy,
)
from tagger.tools.speaker_v2.resolver import resolve
from tagger.tools.speaker_v2.speaker_profile import (
    PROFILE_SCHEMA_VERSION as SPEAKER_PROFILE_SCHEMA_VERSION,
    compute_speaker_profiles,
)
from tagger.tools.speaker_v2.speaker_gender import (
    MODEL_METADATA as SPEAKER_GENDER_MODEL_METADATA,
    SCHEMA_VERSION as SPEAKER_GENDER_SCHEMA_VERSION,
    SpeakerGenderConfig,
    infer_speaker_genders,
)
from tagger.tools.speaker_v2.sortformer_timeline import (
    CHECKPOINT_SHA256 as SORTFORMER_CHECKPOINT_SHA256,
    MAXIMUM_SPEAKERS as SORTFORMER_MAXIMUM_SPEAKERS,
    SortformerTimelineConfig,
    diarize as diarize_sortformer,
)
from tagger.tools.speaker_v2.timeline import summarize_timeline
from tagger.tools.speaker_v2.whisper_lexical import (
    CHECKPOINT_SHA256 as WHISPER_CHECKPOINT_SHA256,
    WhisperLexicalConfig,
    transcribe as transcribe_whisper,
)
from tagger.tools.subprocess_runner import close_subprocess_workers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOSS_PYTHON = PROJECT_ROOT / ".runtime" / (
    "moss_transcribe_diarize_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_MOSS_MODEL = PROJECT_ROOT / "models" / "MOSS-Transcribe-Diarize-model"
DEFAULT_CAMPPLUS_PYTHON = PROJECT_ROOT / ".runtime" / (
    "campplus_sv_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_CAMPPLUS_MODEL = PROJECT_ROOT / (
    "models/speech_campplus_sv_zh-cn_16k-common-v1.0.0"
)
DEFAULT_WHISPER_PYTHON = PROJECT_ROOT / ".runtime" / (
    "whisper_base_multilingual_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_WHISPER_MODEL = PROJECT_ROOT / "models" / "speaker" / "openai" / (
    "whisper-base"
) / WHISPER_CHECKPOINT_SHA256 / "base.pt"
DEFAULT_SORTFORMER_PYTHON = PROJECT_ROOT / ".runtime" / (
    "sortformer_nemo253_py311_torch260_cu124_v1/bin/python"
)
DEFAULT_SORTFORMER_MODEL = PROJECT_ROOT / "models" / "speaker" / "nvidia" / (
    "diar_streaming_sortformer_4spk-v2"
) / "diar_streaming_sortformer_4spk-v2.nemo"
DEFAULT_PYANNOTE_PYTHON = PROJECT_ROOT / ".runtime" / (
    "speaker_pyannote4_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_PYANNOTE_MODEL = PROJECT_ROOT / "models" / "speaker" / (
    "pyannote_community_1"
)
DEFAULT_ECAPA_PYTHON = PROJECT_ROOT / ".runtime" / (
    "fireredvad_rebuild_py310/bin/python"
)
DEFAULT_ECAPA_MODEL = PROJECT_ROOT / "models" / "speaker" / (
    "speechbrain_ecapa_voxceleb"
)
DEFAULT_GENDER_ALICE_MODEL = PROJECT_ROOT / "models" / "speaker" / (
    "alice_voice_gender_classifier/onnx-q8-v2/model_quantized.onnx"
)
DEFAULT_GENDER_AFRISPEECH_MODEL_DIR = PROJECT_ROOT / "models" / "speaker" / (
    "afrispeech_gender_id/2026-09-11"
)
DEFAULT_GENDER_GRIKO_MODEL_DIR = PROJECT_ROOT / "models" / "speaker" / (
    "griko_gender_cls_ecapa_voxceleb/2025-07-31"
)
DEFAULT_GENDER_ONNX_PYTHON = PROJECT_ROOT / ".runtime" / (
    "diarizen_official_py310/bin/python"
)
DEFAULT_GENDER_GRIKO_PYTHON = PROJECT_ROOT / ".runtime" / (
    "fireredvad_rebuild_py310/bin/python"
)
MOSS_MODEL_SHA256 = (
    "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"
)
MOSS_SOURCE_VERSION = "OpenMOSS/MOSS-Transcribe-Diarize-0.9B@shared-20260811"
FIRERED_ASR_MODEL_SHA256 = (
    "4677cbd30988d63ed3e777f6a42a1e5260a3865317f6e15e488bef40954f7054"
)
# Keep the evidence provenance identical to the adapter and deployment
# manifest; the source tree records the corresponding FireRedTeam release.
FIRERED_ASR_SOURCE_VERSION = "ModelScope:xukaituo/FireRedASR2-AED@shared-20260831"
FIRERED_LID_SOURCE_VERSION = "FireRedASR2S/FireRedLID@shared-20260831"
FIRERED_VAD_MODEL_SHA256 = (
    "63f4fb1b00a6b8607c118dd48efc18d5e40d67d99b7bf9aa7a8d61540cf23d71"
)
CAMPPLUS_SOURCE_VERSION = "iic/speech_campplus_sv_zh-cn_16k-common@v1.0.0"
WHISPER_SOURCE_VERSION = "openai/whisper-base@official-base.pt"
SORTFORMER_SOURCE_VERSION = (
    "nvidia/diar_streaming_sortformer_4spk-v2@"
    "6dbf0d69730bfee097056692b86525a0a23b32f9"
)
_MODEL_HASH_CACHE = {}
_MODEL_HASH_CACHE_LOCK = threading.Lock()


class SpeakerEvidenceConfig:
    def __init__(
        self,
        moss_config=None,
        firered_asr_config=None,
        qwen3_asr_config=None,
        vad_config=None,
        campplus_config=None,
        whisper_config=None,
        sortformer_config=None,
        pyannote_config=None,
        ecapa_config=None,
        brouhaha_config=None,
        gender_config=None,
        profile_id="legacy-shadow",
        claim_policy=None,
        expanded_run_profile=None,
        enable_moss=None,
        enable_firered_asr=None,
        enable_qwen3_asr=None,
        enable_vad=None,
        enable_campplus=None,
        enable_whisper=None,
        enable_sortformer=None,
        enable_pyannote=None,
        enable_ecapa=None,
        enable_brouhaha=None,
        enable_gender=None,
        enable_speaker_profile=True,
        score_native=False,
        verify_model_assets=True,
        nemotron_config=None,
        firered_lid_config=None,
        enable_nemotron=None,
        enable_openasr=None,
        speaker_segment_lid_config=None,
    ):
        expanded = expand_profile(profile_id)
        policy = copy.deepcopy(
            expanded["claim_policy"] if claim_policy is None else claim_policy
        )
        validate_claim_policy(policy)
        if str(policy.get("profile_id")) != str(expanded["profile_id"]):
            raise ValueError("claim_policy profile_id does not match profile_id")
        policy["policy_hash"] = claim_policy_hash(policy)
        self.profile_id = str(expanded["profile_id"])
        self.claim_policy = policy
        self.profile_model_defaults = {
            str(name): bool(enabled)
            for name, enabled in expanded["models"].items()
        }
        self.moss_config = moss_config or MossDiarizeConfig()
        self.nemotron_config = nemotron_config or Nemotron3DiarizationConfig(
            model_dir=os.environ.get(
                "TAGGER_NEMOTRON_MODEL_DIR",
                str(DEFAULT_NEMOTRON_MODEL_DIR),
            ),
            subprocess_python=os.environ.get(
                "TAGGER_NEMOTRON_PYTHON",
                str(DEFAULT_NEMOTRON_PYTHON),
            ),
        )
        self.firered_asr_config = firered_asr_config or FireRedAsrConfig(
            enable_lid=False,
        )
        self.qwen3_asr_config = qwen3_asr_config or Qwen3AsrConfig()
        self.firered_lid_config = firered_lid_config or FireRedLidConfig()
        self.speaker_segment_lid_config = (
            speaker_segment_lid_config or SpeakerSegmentLidConfig()
        )
        self.vad_config = vad_config or FireRedVadConfig()
        self.campplus_config = campplus_config
        self.whisper_config = whisper_config
        self.sortformer_config = sortformer_config
        self.pyannote_config = pyannote_config
        self.ecapa_config = ecapa_config
        self.brouhaha_config = brouhaha_config
        self.gender_config = gender_config or SpeakerGenderConfig(
            alice_model_path=os.environ.get(
                "TAGGER_GENDER_ALICE_MODEL", str(DEFAULT_GENDER_ALICE_MODEL)
            ),
            afrispeech_model_dir=os.environ.get(
                "TAGGER_GENDER_AFRISPEECH_MODEL_DIR",
                str(DEFAULT_GENDER_AFRISPEECH_MODEL_DIR),
            ),
            griko_model_dir=os.environ.get(
                "TAGGER_GENDER_GRIKO_MODEL_DIR",
                str(DEFAULT_GENDER_GRIKO_MODEL_DIR),
            ),
            ecapa_model_dir=os.environ.get(
                "TAGGER_ECAPA_MODEL", str(DEFAULT_ECAPA_MODEL)
            ),
            alice_python=os.environ.get(
                "TAGGER_GENDER_ALICE_PYTHON", str(DEFAULT_GENDER_ONNX_PYTHON)
            ),
            afrispeech_python=os.environ.get(
                "TAGGER_GENDER_AFRISPEECH_PYTHON",
                str(DEFAULT_GENDER_ONNX_PYTHON),
            ),
            griko_python=os.environ.get(
                "TAGGER_GENDER_GRIKO_PYTHON", str(DEFAULT_GENDER_GRIKO_PYTHON)
            ),
            verify_model_assets=verify_model_assets,
        )
        configs = {
            "moss": self.moss_config,
            "firered_asr": self.firered_asr_config,
            "qwen3_asr": self.qwen3_asr_config,
            "nemotron": self.nemotron_config,
            "vad": self.vad_config,
            "campplus": self.campplus_config,
            "whisper": self.whisper_config,
            "sortformer": self.sortformer_config,
            "pyannote": self.pyannote_config,
            "ecapa": self.ecapa_config,
            "brouhaha": self.brouhaha_config,
            "gender": self.gender_config,
        }
        firered_override = enable_firered_asr
        qwen3_override = enable_qwen3_asr
        nemotron_override = enable_nemotron
        if enable_openasr is not None:
            if firered_override is None:
                firered_override = bool(enable_openasr)
            if qwen3_override is None:
                qwen3_override = bool(enable_openasr)
            if nemotron_override is None:
                nemotron_override = bool(enable_openasr)
        overrides = {
            "moss": enable_moss,
            "firered_asr": firered_override,
            "qwen3_asr": qwen3_override,
            "nemotron": nemotron_override,
            "vad": enable_vad,
            "campplus": enable_campplus,
            "whisper": enable_whisper,
            "sortformer": enable_sortformer,
            "pyannote": enable_pyannote,
            "ecapa": enable_ecapa,
            "brouhaha": enable_brouhaha,
            "gender": enable_gender,
        }
        self.model_disabled_reasons = {}
        actual_models = {}
        for name, model_config in configs.items():
            profile_default = bool(expanded["models"][name])
            requested = (
                profile_default
                if overrides[name] is None
                else bool(overrides[name])
            )
            enabled = bool(requested and model_config is not None)
            setattr(self, "enable_%s" % name, enabled)
            actual_models[name] = enabled
            if enabled:
                continue
            if overrides[name] is False:
                reason = "disabled_by_cli_or_run_override"
            elif not requested:
                reason = "disabled_by_run_profile:%s" % self.profile_id
            else:
                reason = "model_configuration_unavailable"
            self.model_disabled_reasons[name] = reason
        # Compatibility group summary only. The model calls are independent.
        self.enable_openasr = bool(
            self.enable_firered_asr
            and self.enable_qwen3_asr
            and self.enable_nemotron
        )
        if expanded_run_profile is not None:
            supplied_profile = copy.deepcopy(expanded_run_profile)
            if str(supplied_profile.get("profile_id")) != self.profile_id:
                raise ValueError("expanded_run_profile profile_id does not match")
            supplied_policy = supplied_profile.get("claim_policy")
            if supplied_policy is not None:
                validate_claim_policy(supplied_policy)
                if claim_policy_hash(supplied_policy) != claim_policy_hash(policy):
                    raise ValueError(
                        "expanded_run_profile claim_policy does not match"
                    )
            supplied_defaults = supplied_profile.get(
                "profile_model_defaults"
            )
            if supplied_defaults is not None and {
                str(name): bool(enabled)
                for name, enabled in supplied_defaults.items()
            } != self.profile_model_defaults:
                raise ValueError(
                    "expanded_run_profile profile_model_defaults do not match"
                )
            self.expanded_run_profile = supplied_profile
        else:
            self.expanded_run_profile = copy.deepcopy(expanded)
        self.expanded_run_profile["models"] = actual_models
        self.expanded_run_profile["profile_model_defaults"] = copy.deepcopy(
            self.profile_model_defaults
        )
        self.expanded_run_profile["claim_policy"] = copy.deepcopy(policy)
        self.score_native = bool(score_native)
        self.enable_speaker_profile = bool(enable_speaker_profile)
        self.verify_model_assets = bool(verify_model_assets)


def default_speaker_evidence_config(
    profile_id="quality-shadow",
    vad_config=None,
    brouhaha_config=None,
    verify_model_assets=True,
    firered_asr_config=None,
    qwen3_asr_config=None,
    nemotron_config=None,
    firered_lid_config=None,
    speaker_segment_lid_config=None,
):
    """Build the shared speaker-v2 configuration used by the main pipeline."""

    ecapa_python = os.environ.get(
        "TAGGER_ECAPA_PYTHON", str(DEFAULT_ECAPA_PYTHON)
    )
    firered_asr_config = firered_asr_config or FireRedAsrConfig(
        enable_lid=False,
    )
    nemotron_config = nemotron_config or Nemotron3DiarizationConfig(
        model_dir=os.environ.get(
            "TAGGER_NEMOTRON_MODEL_DIR",
            str(DEFAULT_NEMOTRON_MODEL_DIR),
        ),
        subprocess_python=os.environ.get(
            "TAGGER_NEMOTRON_PYTHON",
            str(DEFAULT_NEMOTRON_PYTHON),
        ),
    )
    return SpeakerEvidenceConfig(
        moss_config=MossDiarizeConfig(
            model=str(DEFAULT_MOSS_MODEL),
            subprocess_python=str(DEFAULT_MOSS_PYTHON),
            device="cuda:0",
            torch_dtype="float16",
            max_new_tokens=2048,
        ),
        firered_asr_config=firered_asr_config,
        qwen3_asr_config=qwen3_asr_config or Qwen3AsrConfig(),
        nemotron_config=nemotron_config,
        firered_lid_config=firered_lid_config or FireRedLidConfig(),
        speaker_segment_lid_config=(
            speaker_segment_lid_config or SpeakerSegmentLidConfig()
        ),
        vad_config=vad_config or FireRedVadConfig(),
        campplus_config=CampPlusIdentityConfig(
            model_dir=str(DEFAULT_CAMPPLUS_MODEL),
            subprocess_python=str(DEFAULT_CAMPPLUS_PYTHON),
            device="cpu",
        ),
        whisper_config=WhisperLexicalConfig(
            model_path=str(DEFAULT_WHISPER_MODEL),
            subprocess_python=str(DEFAULT_WHISPER_PYTHON),
            device="cuda:0",
        ),
        sortformer_config=SortformerTimelineConfig(
            model_path=str(DEFAULT_SORTFORMER_MODEL),
            subprocess_python=str(DEFAULT_SORTFORMER_PYTHON),
            device="cuda:0",
        ),
        pyannote_config=PyannoteCommunity1Config(
            model_dir=str(DEFAULT_PYANNOTE_MODEL),
            subprocess_python=str(DEFAULT_PYANNOTE_PYTHON),
            device="cuda:0",
        ),
        ecapa_config=EcapaIdentityConfig(
            model_dir=str(DEFAULT_ECAPA_MODEL),
            subprocess_python=ecapa_python,
            device="cuda:0",
        ),
        brouhaha_config=brouhaha_config or BrouhahaConfig(),
        profile_id=profile_id,
        verify_model_assets=verify_model_assets,
    )


def run_manifest(
    manifest_path,
    output_dir,
    config,
    sample_ids=None,
    max_samples=None,
    fail_fast=False,
    workers=1,
    resume=False,
    model_workers=1,
    model_worker_overrides=None,
):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    selected = set(sample_ids or [])
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    model_workers = max(1, int(model_workers))
    slots = {
        "moss_diarize_estimate": model_workers,
        OPENASR_MODEL_SLOT_KEY: model_workers,
        NEMOTRON_MODEL_SLOT_KEY: model_workers,
        "firered_asr_estimate": model_workers,
        "qwen3_asr_estimate": model_workers,
        "whisper_lexical_estimate": model_workers,
        "sortformer_timeline_estimate": model_workers,
        "firered_vad_detect": model_workers,
        "campplus_identity_estimate": model_workers,
        "pyannote_community1_estimate": model_workers,
        "ecapa_identity_estimate": model_workers,
        "brouhaha_estimate": model_workers,
        "speaker_gender_estimate": model_workers,
    }
    for key, value in (model_worker_overrides or {}).items():
        slots[key] = max(1, int(value))
    context = {
        "_subprocess_workers_lock": threading.Lock(),
        "_subprocess_worker_slots": slots,
    }
    results = []
    processed = 0
    result_path = output_dir / "speaker_v2_shadow_results.jsonl"
    progress_path = output_dir / ".speaker_v2_shadow_progress.jsonl"
    completed = (
        _read_completed_results(
            result_path,
            progress_path,
            expected_run_profile=config.profile_id,
            expected_policy_hash=claim_policy_hash(config.claim_policy),
        )
        if resume
        else {}
    )
    resumed_sample_count = 0
    pending = []
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_input_record(record)
                    sample_id = record["sample"]["sample_id"]
                except Exception as exc:
                    if fail_fast:
                        raise
                    results.append(
                        {
                            "sample_id": "manifest_line_%06d" % line_number,
                            "status": "failed",
                            "error_type": exc.__class__.__name__,
                            "message": str(exc),
                            "manifest_line": line_number,
                        }
                    )
                    processed += 1
                    continue
                if selected and sample_id not in selected:
                    continue
                if max_samples is not None and processed >= int(max_samples):
                    break
                if sample_id in completed:
                    results.append(completed[sample_id])
                    resumed_sample_count += 1
                else:
                    pending.append((line_number, record, sample_id))
                processed += 1

        if workers == 1:
            for item in pending:
                results.append(
                    _run_manifest_item(
                        item,
                        manifest_path.parent,
                        output_dir,
                        config,
                        context,
                        fail_fast,
                    )
                )
                _append_jsonl(progress_path, results[-1])
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(
                        _run_manifest_item,
                        item,
                        manifest_path.parent,
                        output_dir,
                        config,
                        context,
                        fail_fast,
                    ): item
                    for item in pending
                }
                for future in as_completed(future_to_item):
                    results.append(future.result())
                    _append_jsonl(progress_path, results[-1])
    finally:
        close_subprocess_workers(context)

    results.sort(key=lambda item: item.get("sample_id", ""))
    _write_jsonl_atomic(result_path, results)
    if progress_path.exists():
        progress_path.unlink()
    summary = {
        "profile": "v2-shadow",
        "run_profile": config.profile_id,
        "expanded_run_profile": copy.deepcopy(config.expanded_run_profile),
        "profile_model_defaults": copy.deepcopy(
            config.profile_model_defaults
        ),
        "claim_policy": copy.deepcopy(config.claim_policy),
        "policy_version": config.claim_policy["policy_version"],
        "policy_hash": claim_policy_hash(config.claim_policy),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "processed_sample_count": processed,
        "success_count": sum(item.get("status") == "ok" for item in results),
        "failure_count": sum(item.get("status") != "ok" for item in results),
        "workers": workers,
        "model_workers": model_workers,
        "model_worker_overrides": slots,
        "resumed_sample_count": resumed_sample_count,
        "results": results,
    }
    run_manifest_path = write_run_manifest(
        output_dir,
        manifest_path,
        sha256_file(manifest_path),
        config,
        summary,
        sample_ids=sample_ids,
        max_samples=max_samples,
        fail_fast=fail_fast,
        resume=resume,
    )
    summary["run_manifest"] = str(run_manifest_path)
    return summary


def _run_manifest_item(
    item, manifest_dir, output_dir, config, context, fail_fast
):
    line_number, record, sample_id = item
    try:
        return run_record(
            record,
            manifest_dir,
            output_dir,
            config,
            context=context,
        )
    except Exception as exc:
        if fail_fast:
            raise
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "manifest_line": line_number,
        }


def _read_completed_results(*paths, **expected):
    expected_run_profile = expected.get("expected_run_profile")
    expected_policy_hash = expected.get("expected_policy_hash")
    completed = {}
    for path in paths:
        if not Path(path).is_file():
            continue
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") != "ok" or not item.get("sample_id"):
                    continue
                if (
                    expected_run_profile is not None
                    and item.get("run_profile") != expected_run_profile
                ):
                    continue
                if (
                    expected_policy_hash is not None
                    and item.get("policy_hash") != expected_policy_hash
                ):
                    continue
                completed[item["sample_id"]] = item
    return completed


def run_record(
    record,
    manifest_dir,
    output_dir,
    config,
    context=None,
    artifact_root=None,
    artifact_sample_id=None,
    speaker_segment_lid_config=None,
):
    """Run inference without exposing native annotations to any adapter."""

    validate_input_record(record)
    sample = record["sample"]
    sample_id = sample["sample_id"]
    audio_path = resolve_sample_audio_path(sample["audio"]["path"], manifest_dir)
    audio_info = probe_audio_info(audio_path)
    duration_sec = float(audio_info.duration_sec)
    audio_sha256 = sha256_file(audio_path)

    # This is the complete inference view. Native metadata and the supplied
    # transcript are deliberately absent; both ASR paths generate their own
    # text from the sample audio.
    inference_scope = {
        "sample_id": sample_id,
        "audio_path": str(audio_path),
        "audio_sha256": audio_sha256,
        "duration_sec": duration_sec,
        "sample_rate_hz": audio_info.sample_rate_hz,
        "channels": audio_info.channels,
    }
    evidence = []
    # MOSS and Nemotron are collected before any language route. Language is
    # decided on the full audio by default. Speaker-segment LID is opt-in and
    # falls back to full-audio FireRed LID when it cannot be published.
    moss_evidence, nemotron_evidence = _collect_moss_and_nemotron_evidence(
        inference_scope,
        config,
        context=context,
    )
    segment_config = (
        speaker_segment_lid_config
        or getattr(config, "speaker_segment_lid_config", None)
        or SpeakerSegmentLidConfig()
    )
    if segment_config.enabled:
        speaker_segment_evidence = collect_speaker_segment_lid_evidence(
            inference_scope,
            nemotron_evidence,
            moss_evidence,
            None,
            context=context,
            config=segment_config,
            lid_config=config.firered_lid_config,
            artifact_dir=artifact_root,
            artifact_sample_id=artifact_sample_id or sample_id,
        )
        if speaker_segment_evidence is not None and speaker_segment_evidence.get(
            "quality", {}
        ).get("usable"):
            evidence.append(speaker_segment_evidence)
        else:
            if speaker_segment_evidence is not None:
                evidence.append(speaker_segment_evidence)
            if segment_config.fallback_to_full_audio:
                lid_evidence = collect_firered_lid_evidence(
                    inference_scope,
                    config.firered_lid_config,
                    context=context,
                )
                evidence.append(lid_evidence)
    else:
        lid_evidence = collect_firered_lid_evidence(
            inference_scope,
            config.firered_lid_config,
            context=context,
        )
        evidence.append(lid_evidence)
    language_route = _language_route_from_evidence(evidence)
    count_consensus = _single_speaker_count_consensus(
        (moss_evidence, nemotron_evidence)
    )
    firered_required = bool(
        _is_moss_route(language_route["language_route"])
        and count_consensus["all_at_most_one"]
    )
    qwen3_required = not _is_moss_route(language_route["language_route"])
    if firered_required and config.enable_firered_asr:
        firered_evidence = _collect_asr_evidence(
            "firered_asr",
            inference_scope,
            config.firered_asr_config,
            context=context,
            verify_model_asset=config.verify_model_assets,
        )
    elif firered_required:
        firered_evidence = _missing_asr_evidence(
            "firered_asr",
            inference_scope,
            "FireRed ASR disabled by run configuration",
        )
    else:
        firered_evidence = _missing_asr_evidence(
            "firered_asr",
            inference_scope,
            "not required for this Chinese/English ASR route",
        )
    if qwen3_required and config.enable_qwen3_asr:
        qwen3_evidence = _collect_asr_evidence(
            "qwen3_asr",
            inference_scope,
            config.qwen3_asr_config,
            context=context,
            verify_model_asset=config.verify_model_assets,
        )
    elif qwen3_required:
        qwen3_evidence = _missing_asr_evidence(
            "qwen3_asr",
            inference_scope,
            "Qwen3 ASR disabled by run configuration",
        )
    else:
        qwen3_evidence = _missing_asr_evidence(
            "qwen3_asr",
            inference_scope,
            "not required for this Chinese/English ASR route",
        )
    evidence.extend(
        (moss_evidence, firered_evidence, qwen3_evidence, nemotron_evidence)
    )

    if config.whisper_config is not None:
        if config.enable_whisper:
            evidence.append(
                collect_whisper_evidence(
                    inference_scope,
                    config.whisper_config,
                    context=context,
                    verify_model_asset=config.verify_model_assets,
                )
            )
        else:
            evidence.append(
                build_missing_evidence(
                    sample_id,
                    duration_sec,
                    "lexical_timeline",
                    "whisper_base_lexical_clock",
                    WHISPER_SOURCE_VERSION,
                    "asr",
                    ["lexical_timeline"],
                    ["G_whisper_base_official"],
                    "disabled by run configuration",
                )
            )
    if config.sortformer_config is not None:
        if config.enable_sortformer:
            evidence.append(
                collect_sortformer_evidence(
                    inference_scope,
                    config.sortformer_config,
                    context=context,
                    verify_model_asset=config.verify_model_assets,
                )
            )
        else:
            evidence.append(
                build_missing_evidence(
                    sample_id,
                    duration_sec,
                    "speaker_timeline",
                    "nvidia_streaming_sortformer_4spk_v2",
                    SORTFORMER_SOURCE_VERSION,
                    "end_to_end_streaming_diarizer",
                    [
                        "speaker_timeline",
                        "speaker_count_candidate",
                        "overlap_candidate",
                        "change_candidate",
                    ],
                    ["G_nvidia_sortformer_4spk_v2"],
                    "disabled by run configuration",
                    applicability={"maximum_speakers": 4},
                )
            )
    if config.pyannote_config is not None:
        if config.enable_pyannote:
            evidence.append(
                collect_pyannote_evidence(
                    inference_scope,
                    config.pyannote_config,
                    context=context,
                    verify_model_asset=config.verify_model_assets,
                )
            )
        else:
            evidence.append(
                build_missing_evidence(
                    sample_id,
                    duration_sec,
                    "speaker_timeline",
                    "pyannote_community_1",
                    PYANNOTE_MODEL_VERSION,
                    "powerset_speaker_diarization",
                    [
                        "speaker_timeline",
                        "speaker_count_candidate",
                        "overlap_candidate",
                        "change_candidate",
                    ],
                    ["G_pyannote_community1"],
                    "disabled by run configuration",
                    applicability={
                        "audio_sha256": audio_sha256,
                        "model_revision": PYANNOTE_MODEL_REVISION,
                    },
                )
            )
    if config.enable_vad:
        evidence.append(
            collect_vad_evidence(
                inference_scope,
                config.vad_config,
                context=context,
                verify_model_asset=config.verify_model_assets,
            )
        )
    else:
        evidence.append(
            build_missing_evidence(
                sample_id,
                duration_sec,
                "speech_coverage",
                "firered_vad",
                "FireRedVAD@shared-20260811",
                "vad",
                ["speech_coverage"],
                ["G_firered_vad"],
                "disabled by run configuration",
            )
        )

    if config.enable_brouhaha:
        evidence.append(
            collect_brouhaha_evidence(
                inference_scope,
                config.brouhaha_config,
                context=context,
                verify_model_asset=config.verify_model_assets,
            )
        )

    language_routed_claims = any(
        rule.get("selection") == "language_routed"
        for rule in config.claim_policy.get("claims", {}).values()
    )
    selected_timeline_source = (
        "moss_transcribe_diarize"
        if _is_moss_route(language_route["language_route"])
        else NEMOTRON_SOURCE_NAME
    )
    timelines = [
        item
        for item in evidence
        if "speaker_timeline" in item.get("capabilities", [])
        and item.get("status") in ("observed", "estimated")
        and item.get("quality", {}).get("usable", True) is not False
        and (
            not language_routed_claims
            or item.get("source", {}).get("name") == selected_timeline_source
        )
    ]
    profile_timeline = _select_identity_candidate_timeline(
        timelines, config.claim_policy
    )
    coverage_evidence = _select_profile_coverage_evidence(evidence)
    if not config.enable_speaker_profile:
        profile_bundle = {
            "profiles": None,
            "details": {"status": "disabled_by_run_configuration"},
        }
    else:
        try:
            profile_bundle = compute_speaker_profiles(
                profile_timeline,
                coverage_evidence=coverage_evidence,
                text_evidence=evidence,
                audio_path=audio_path,
                duration_sec=duration_sec,
                sample_rate_hz=audio_info.sample_rate_hz,
            )
        except Exception as exc:
            profile_bundle = {
                "profiles": None,
                "details": {
                    "status": "failed",
                    "reason": "%s: %s" % (exc.__class__.__name__, exc),
                },
            }
    if not config.enable_speaker_profile:
        gender_bundle = {
            "schema_version": SPEAKER_GENDER_SCHEMA_VERSION,
            "status": "speaker_profile_disabled",
            "detected_language": language_route.get("detected_language"),
            "selected_model": None,
            "speakers": [],
        }
    elif not config.enable_gender:
        gender_bundle = {
            "schema_version": SPEAKER_GENDER_SCHEMA_VERSION,
            "status": "disabled_by_run_configuration",
            "detected_language": language_route.get("detected_language"),
            "selected_model": None,
            "speakers": [],
        }
    else:
        try:
            gender_bundle = infer_speaker_genders(
                profile_timeline,
                audio_path,
                language_route.get("detected_language"),
                config=config.gender_config,
                context=context,
            )
        except Exception as exc:
            gender_bundle = {
                "schema_version": SPEAKER_GENDER_SCHEMA_VERSION,
                "status": "failed",
                "detected_language": language_route.get("detected_language"),
                "selected_model": None,
                "speakers": [],
                "reason": "%s: %s" % (exc.__class__.__name__, exc),
            }
    gender_model = gender_bundle.get("selected_model")
    gender_dependencies = ["G_speaker_gender_language_routed"]
    if gender_model in SPEAKER_GENDER_MODEL_METADATA:
        gender_dependencies.append(
            SPEAKER_GENDER_MODEL_METADATA[gender_model]["dependency_group"]
        )
    gender_status = (
        "observed"
        if gender_bundle.get("status") in ("observed", "no_speech")
        else "missing"
    )
    gender_evidence = build_evidence(
        sample_id=sample_id,
        duration_sec=duration_sec,
        evidence_type="speaker_gender",
        source_name="speaker_gender_language_routed",
        source_version=SPEAKER_GENDER_SCHEMA_VERSION,
        source_kind="language_routed_speaker_gender_classifier",
        capabilities=["speaker_gender"],
        dependency_groups=gender_dependencies,
        payload=gender_bundle,
        status=gender_status,
        quality={
            "usable": gender_status == "observed",
            "model_required": True,
            "reason": gender_bundle.get("status"),
        },
        lineage={
            "parent_evidence_ids": (
                [profile_timeline["evidence_id"]]
                if profile_timeline is not None
                else []
            )
        },
        applicability={
            "audio_sha256": audio_sha256,
            "detected_language": language_route.get("detected_language"),
            "selected_model": gender_model,
            "gender_schema_version": SPEAKER_GENDER_SCHEMA_VERSION,
        },
    )
    evidence.append(gender_evidence)
    _merge_profile_genders(profile_bundle, gender_bundle)
    profile_parents = []
    if profile_timeline is not None:
        profile_parents.append(profile_timeline["evidence_id"])
    if coverage_evidence is not None:
        profile_parents.append(coverage_evidence["evidence_id"])
    profile_text_parents = _profile_text_parent_ids(
        profile_timeline, evidence
    )
    profile_parents.extend(profile_text_parents)
    profile_parents.append(gender_evidence["evidence_id"])
    profile_status = "observed" if profile_bundle["profiles"] is not None else "missing"
    profile_quality = {
        "usable": profile_bundle["profiles"] is not None,
        "deterministic_acoustic_fields": True,
        "model_required": bool(config.enable_gender),
        "reason": profile_bundle["details"].get("status"),
    }
    evidence.append(
        build_evidence(
            sample_id=sample_id,
            duration_sec=duration_sec,
            evidence_type="speaker_profile_acoustic",
            source_name="speaker_profile_deterministic",
            source_version=SPEAKER_PROFILE_SCHEMA_VERSION,
            source_kind="deterministic_acoustic_adapter",
            capabilities=[
                "speaker_profile",
                "speech_rate",
                "pitch",
                "speaker_volume",
                "gender",
            ],
            dependency_groups=["G_speaker_profile_deterministic"],
            payload={
                "profiles": profile_bundle["profiles"],
                "details": profile_bundle["details"],
                "decision_timeline_evidence_id": (
                    profile_timeline.get("evidence_id")
                    if profile_timeline is not None
                    else None
                ),
            },
            status=profile_status,
            quality=profile_quality,
            lineage={"parent_evidence_ids": profile_parents},
            applicability={
                "audio_sha256": audio_sha256,
                "sample_rate_hz": audio_info.sample_rate_hz,
                "channels": audio_info.channels,
                "profile_schema_version": SPEAKER_PROFILE_SCHEMA_VERSION,
            },
        )
    )
    hypothesis = build_count_hypothesis_case(
        sample_id, timelines, all_evidence=evidence
    )
    identity_candidate = _select_identity_candidate_timeline(
        timelines, config.claim_policy
    )
    if config.enable_ecapa:
        if identity_candidate is None:
            evidence.append(
                _missing_ecapa_evidence(
                    inference_scope,
                    config.ecapa_config,
                    "no usable predicted timeline for region selection",
                )
            )
        else:
            evidence.append(
                collect_ecapa_evidence(
                    inference_scope,
                    identity_candidate,
                    config.ecapa_config,
                    context=context,
                    verify_model_asset=config.verify_model_assets,
                )
            )
    if config.enable_campplus and timelines:
        # Identity acquisition is candidate-driven by the first baseline
        # timeline, or by the higher-count source after a count hypothesis has
        # been frozen. Its lineage records that selection dependency explicitly.
        candidate_timeline = timelines[0]
        if hypothesis is not None:
            target_id = hypothesis["conflict"][
                "higher_count_source_evidence_id"
            ]
            candidate_timeline = next(
                item for item in timelines if item["evidence_id"] == target_id
            )
        evidence.append(
            collect_campplus_evidence(
                inference_scope,
                candidate_timeline,
                config.campplus_config,
                context=context,
                verify_model_asset=config.verify_model_assets,
            )
        )
    identity = [
        item
        for item in evidence
        if "speaker_identity_comparison" in item.get("capabilities", [])
    ]
    hypothesis = evaluate_count_hypothesis_case(hypothesis, identity)
    hypotheses = [hypothesis] if hypothesis is not None else []
    fusion = resolve(
        sample_id,
        duration_sec,
        evidence,
        hypotheses=hypotheses,
        claim_policy=config.claim_policy,
        profile_id=config.profile_id,
        speaker_profiles=profile_bundle["profiles"],
        language_route=language_route,
    )
    fusion["input_provenance"] = {
        "audio_sha256": audio_sha256,
        "duration_sec": round(duration_sec, 6),
        "sample_rate_hz": audio_info.sample_rate_hz,
        "channels": audio_info.channels,
        "native_metadata_entered_inference": False,
        "input_transcript_entered_resolver": False,
    }
    native_text_tracks = [speaker_text_track(item) for item in timelines]
    lexical_evidence = [
        item
        for item in evidence
        if "lexical_timeline" in item.get("capabilities", [])
        and item.get("status") in ("observed", "estimated")
        and item.get("quality", {}).get("usable", True)
    ]
    projected_text_tracks = [
        project_asr_track(asr, timeline)
        for asr in lexical_evidence
        for timeline in timelines
    ]
    text_comparisons = []
    for native in native_text_tracks:
        if not native.get("units"):
            continue
        for projected in projected_text_tracks:
            if (
                projected.get("units")
                and native["source_evidence_id"]
                in projected["source_evidence_ids"]
            ):
                text_comparisons.append(
                    text_track_comparison(native, projected)
                )
    timeline_evidence_ids = {item["evidence_id"] for item in timelines}
    for left_index, left in enumerate(projected_text_tracks):
        for right in projected_text_tracks[left_index + 1 :]:
            left_sources = set(left["source_evidence_ids"])
            right_sources = set(right["source_evidence_ids"])
            shared_asr = (
                left_sources.intersection(right_sources)
                - timeline_evidence_ids
            )
            if shared_asr:
                text_comparisons.append(
                    speaker_assignment_comparison(left, right)
                )
    fusion["speaker_text_tracks"] = native_text_tracks + projected_text_tracks
    fusion["speaker_text_comparisons"] = text_comparisons
    asr_selection = _select_speaker_asr(
        evidence, language_route=language_route
    )
    # Keep every ASR candidate in the fusion artifact. Count consensus may
    # select FireRed only for Chinese/English; other languages require Qwen3.
    fusion["asr_route"] = copy.deepcopy(asr_selection["route"])
    fusion["language_route"] = copy.deepcopy(language_route)
    fusion["asr_candidates"] = copy.deepcopy(asr_selection["candidates"])
    speaker_asr_transcript = asr_selection["text"]

    # Evaluation happens strictly after evidence collection and resolution.
    if config.score_native:
        fusion["evaluation_only"] = score_against_native_after_inference(
            record,
            timelines,
            duration_sec,
        )
    paths = write_sample_artifacts(
        output_dir,
        artifact_sample_id or sample_id,
        evidence,
        fusion,
        artifact_root=artifact_root,
    )
    return {
        "sample_id": sample_id,
        "status": "ok",
        "fusion_artifact": paths["fusion_artifact"],
        "artifacts": paths,
        "claims": fusion["claims"],
        "evaluation_output": fusion["evaluation_output"],
        "speaker": fusion["public_adapter"]["speaker"],
        "run_profile": config.profile_id,
        "policy_version": config.claim_policy["policy_version"],
        "policy_hash": claim_policy_hash(config.claim_policy),
        "native_scored": bool(config.score_native),
        "speaker_asr_transcript": speaker_asr_transcript,
        "asr_selected_source": asr_selection["route"].get("selected_source"),
        "language_route": asr_selection["route"].get("language_route"),
        "language_source": language_route.get("language_source"),
        "language_route_reason": asr_selection["route"].get("reason"),
        "detected_language": language_route.get("detected_language"),
        "language_confidence": language_route.get("language_confidence"),
        "language_error": language_route.get("language_error"),
        "language_evidence_id": language_route.get("lid_evidence_id"),
        "language_published_in_speaker_v2": True,
        "selected_timeline": _selected_timeline_output(
            profile_timeline,
            duration_sec,
        ),
        "asr_route": asr_selection["route"],
    }


def collect_speaker_segment_lid_evidence(
    scope,
    nemotron_evidence,
    moss_evidence,
    full_audio_evidence,
    context=None,
    config=None,
    lid_config=None,
    artifact_dir=None,
    artifact_sample_id=None,
):
    """Run speaker-first LID on a language-neutral timeline.

    The Nemotron timeline is preferred because it is not selected by
    language. MOSS is only used when Nemotron is unavailable. The resulting
    evidence is usable for routing only when the speaker-segment policy
    publishes a non-fallback result; the full-audio observation remains the
    explicit fallback candidate.
    """

    config = config or SpeakerSegmentLidConfig()
    timeline_evidence = _select_language_detection_timeline(
        nemotron_evidence, moss_evidence
    )
    full_audio = _lid_candidate(
        [full_audio_evidence] if isinstance(full_audio_evidence, dict) else []
    )
    baseline = {
        "language": full_audio.get("language"),
        "confidence": full_audio.get("language_confidence"),
        "evidence_id": full_audio.get("evidence_id"),
        "language_route": None,
        "error": full_audio.get("language_error"),
    }
    selected_timeline = _selected_timeline_output(
        timeline_evidence, scope["duration_sec"]
    )
    parent_ids = [
        full_audio_evidence.get("evidence_id")
        if isinstance(full_audio_evidence, dict)
        else None
    ]
    if timeline_evidence is not None:
        parent_ids.append(timeline_evidence.get("evidence_id"))
    parent_ids = [value for value in parent_ids if value]
    if selected_timeline is None:
        return build_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "language_identification",
            "speaker_segment_firered_lid",
            SPEAKER_SEGMENT_LID_VERSION,
            "speaker_aware_audio_language_identifier",
            ["language_route"],
            ["G_firered_lid", "G_speaker_segment_lid"],
            {},
            status="missing",
            quality={
                "usable": False,
                "speaker_first": True,
                "reason": "no language-neutral speaker timeline is available",
            },
            lineage={"parent_evidence_ids": parent_ids},
            applicability={"audio_sha256": scope["audio_sha256"]},
        )
    try:
        result = run_speaker_segment_lid(
            scope["audio_path"],
            selected_timeline,
            baseline,
            context=context,
            config=config,
            lid_config=lid_config,
            artifact_dir=artifact_dir,
            sample_id=artifact_sample_id or scope["sample_id"],
        )
    except Exception as exc:  # noqa: BLE001 - route falls back to full LID.
        return build_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "language_identification",
            "speaker_segment_firered_lid",
            SPEAKER_SEGMENT_LID_VERSION,
            "speaker_aware_audio_language_identifier",
            ["language_route"],
            ["G_firered_lid", "G_speaker_segment_lid"],
            {},
            status="missing",
            quality={
                "usable": False,
                "speaker_first": True,
                "reason": "%s: %s" % (exc.__class__.__name__, exc),
            },
            lineage={"parent_evidence_ids": parent_ids},
            applicability={"audio_sha256": scope["audio_sha256"]},
        )

    result_evidence = result.evidence if isinstance(result.evidence, dict) else {}
    selected = bool(
        result.value
        and result_evidence.get("decision") == "longest_clean_segment"
    )
    payload = {
        "language": result.value,
        "language_confidence": result.confidence,
        "decision": result_evidence.get("decision"),
        "fallback_reason": result_evidence.get("fallback_reason"),
        "timeline": selected_timeline,
        "speaker_segment_lid": result_evidence,
    }
    return build_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "language_identification",
        "speaker_segment_firered_lid",
        SPEAKER_SEGMENT_LID_VERSION,
        "speaker_aware_audio_language_identifier",
        ["language_route"],
        ["G_firered_lid", "G_speaker_segment_lid"],
        payload,
        status="estimated" if selected else "missing",
        quality={
            "usable": selected,
            "speaker_first": True,
            "full_audio_fallback": not selected,
            "reason": None if selected else result_evidence.get("fallback_reason"),
        },
        lineage={"parent_evidence_ids": parent_ids},
        applicability={
            "audio_sha256": scope["audio_sha256"],
            "timeline_source": selected_timeline.get("source"),
        },
    )


def _select_language_detection_timeline(nemotron_evidence, moss_evidence):
    """Choose a timeline without consulting the language route."""

    candidates = [nemotron_evidence, moss_evidence]
    usable = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("status") not in ("observed", "estimated"):
            continue
        if item.get("quality", {}).get("usable", True) is False:
            continue
        if "speaker_timeline" not in item.get("capabilities", []):
            continue
        usable.append(item)
    for item in usable:
        summary = item.get("payload", {}).get("timeline_summary", {})
        segments = summary.get("activity_segments") or summary.get("segments")
        if isinstance(segments, list) and segments:
            return item
    return usable[0] if usable else None


def collect_firered_lid_evidence(scope, config, context=None):
    """Collect the full-audio FireRed LID fallback observation."""

    runtime = {
        "python": getattr(config, "subprocess_python", ""),
        "model_path": getattr(config, "model_dir", ""),
        "source_path": getattr(config, "repo_dir", ""),
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    started = time.time()
    try:
        result = run_firered_lid(
            scope["audio_path"], context=context, config=config
        )
        language = result.value
        if not isinstance(language, str) or not language.strip():
            raise ValueError("FireRed LID returned an empty language")
        confidence = result.confidence
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="language_identification",
            source_name="firered_lid",
            source_version=FIRERED_LID_SOURCE_VERSION,
            source_kind="audio_language_identifier",
            capabilities=["language_route"],
            dependency_groups=["G_firered_lid"],
            payload={
                "language": language.strip(),
                "language_confidence": confidence,
            },
            status="estimated",
            quality={"usable": True, "full_scope_invocation": True},
            runtime=runtime,
            applicability={"audio_sha256": scope["audio_sha256"]},
        )
    except Exception as exc:  # noqa: BLE001 - unknown uses the Nemotron route.
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "language_identification",
            "firered_lid",
            FIRERED_LID_SOURCE_VERSION,
            "audio_language_identifier",
            ["language_route"],
            ["G_firered_lid"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability={"audio_sha256": scope["audio_sha256"]},
        )


def collect_nemotron_evidence(scope, config, context=None, verify_model_asset=True):
    """Collect the local Nemotron 3 speaker timeline."""

    model_dir = getattr(config, "model_dir", "")
    runtime = {
        "engine": "TransformersNemotron3Diarization",
        "python": getattr(config, "subprocess_python", ""),
        "model_dir": model_dir,
        "model_version": NEMOTRON_MODEL_VERSION,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": NEMOTRON_MODEL_SHA256 if verify_model_asset else None,
        "config_sha256": NEMOTRON_CONFIG_SHA256 if verify_model_asset else None,
        "processor_config_sha256": (
            NEMOTRON_PROCESSOR_CONFIG_SHA256 if verify_model_asset else None
        ),
        "preprocessing": "channel0_mono_resample_16khz_full_sample",
        "maximum_speakers": 8,
    }
    started = time.time()
    try:
        if verify_model_asset:
            _verify_model_asset(model_dir, "model.safetensors", NEMOTRON_MODEL_SHA256)
            _verify_model_asset(model_dir, "config.json", NEMOTRON_CONFIG_SHA256)
            _verify_model_asset(
                model_dir, "processor_config.json", NEMOTRON_PROCESSOR_CONFIG_SHA256
            )
        result = diarize_nemotron(
            scope["audio_path"], config=config, context=context
        )
        if not isinstance(result, dict):
            raise ValueError("Nemotron returned a non-object result")
        result_runtime = result.get("runtime", {})
        if not isinstance(result_runtime, dict):
            raise ValueError("Nemotron runtime must be an object")
        runtime.update(result_runtime)
        segments = result.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Nemotron timeline is invalid")
        summary = summarize_timeline(segments, scope["duration_sec"])
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_timeline",
            source_name=NEMOTRON_SOURCE_NAME,
            source_version=NEMOTRON_MODEL_VERSION,
            source_kind=NEMOTRON_SOURCE_KIND,
            capabilities=[
                "speaker_timeline",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
            ],
            dependency_groups=list(NEMOTRON_DEPENDENCY_GROUPS),
            payload={
                "timeline_summary": summary,
                "raw_turns": copy.deepcopy(segments),
                "model": NEMOTRON_MODEL_VERSION,
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "empty_timeline": not bool(segments),
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return _missing_nemotron_evidence(
            scope,
            "%s: %s" % (exc.__class__.__name__, exc),
            config,
            runtime=runtime,
            applicability=applicability,
        )


def _missing_nemotron_evidence(
    scope, reason, config, runtime=None, applicability=None
):
    applicability = dict(applicability or {})
    applicability.setdefault("audio_sha256", scope["audio_sha256"])
    applicability.setdefault("model_dir", getattr(config, "model_dir", ""))
    return build_missing_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "speaker_timeline",
        NEMOTRON_SOURCE_NAME,
        NEMOTRON_MODEL_VERSION,
        NEMOTRON_SOURCE_KIND,
        [
            "speaker_timeline",
            "speaker_count_candidate",
            "overlap_candidate",
            "change_candidate",
        ],
        list(NEMOTRON_DEPENDENCY_GROUPS),
        reason,
        runtime=runtime,
        applicability=applicability,
    )


def collect_official_diarizen_evidence(
    scope, config, context=None, verify_model_asset=True
):
    """Collect a speaker timeline from the pinned official DiariZen pipeline."""

    model_dir = getattr(config, "model_dir", "")
    embedding_model = getattr(config, "embedding_model", "")
    source_dir = getattr(config, "source_dir", "")
    runtime = {
        "engine": "DiariZenPipeline",
        "python": getattr(config, "subprocess_python", ""),
        "model_dir": model_dir,
        "embedding_model": embedding_model,
        "source_dir": source_dir,
        "model_version": DIARIZEN_OFFICIAL_MODEL_VERSION,
        "embedding_version": DIARIZEN_OFFICIAL_WESPEAKER_VERSION,
        "code_version": DIARIZEN_OFFICIAL_CODE_VERSION,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": (
            DIARIZEN_OFFICIAL_MODEL_SHA256 if verify_model_asset else None
        ),
        "embedding_sha256": (
            DIARIZEN_OFFICIAL_WESPEAKER_SHA256
            if verify_model_asset
            else None
        ),
        "config_sha256": (
            DIARIZEN_OFFICIAL_CONFIG_SHA256 if verify_model_asset else None
        ),
        "code_version": DIARIZEN_OFFICIAL_CODE_VERSION,
        "preprocessing": "official_sdm_channel0_full_sample_short_zero_pad",
        "embedding_exclude_overlap": True,
        "clustering": "VBxClustering",
    }
    started = time.time()
    try:
        if verify_model_asset:
            _verify_model_asset(
                model_dir, "pytorch_model.bin", DIARIZEN_OFFICIAL_MODEL_SHA256
            )
            _verify_model_asset(
                model_dir, "config.toml", DIARIZEN_OFFICIAL_CONFIG_SHA256
            )
            _verify_model_asset(
                model_dir,
                "plda/plda.npz",
                DIARIZEN_OFFICIAL_PLDA_SHA256,
            )
            _verify_model_asset(
                model_dir,
                "plda/xvec_transform.npz",
                DIARIZEN_OFFICIAL_XVEC_TRANSFORM_SHA256,
            )
            embedding_path = Path(str(embedding_model)).expanduser()
            _verify_model_asset(
                embedding_path.parent,
                embedding_path.name,
                DIARIZEN_OFFICIAL_WESPEAKER_SHA256,
            )
        result = diarize_official_diarizen(
            scope["audio_path"], config=config, context=context
        )
        if not isinstance(result, dict):
            raise ValueError("official DiariZen returned a non-object result")
        result_runtime = result.get("runtime", {})
        if not isinstance(result_runtime, dict):
            raise ValueError("official DiariZen runtime must be an object")
        runtime.update(result_runtime)
        segments = result.get("segments")
        if not isinstance(segments, list):
            raise ValueError("official DiariZen timeline is invalid")
        summary = summarize_timeline(segments, scope["duration_sec"])
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_timeline",
            source_name=DIARIZEN_OFFICIAL_SOURCE_NAME,
            source_version=DIARIZEN_OFFICIAL_MODEL_VERSION,
            source_kind=DIARIZEN_OFFICIAL_SOURCE_KIND,
            capabilities=[
                "speaker_timeline",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
            ],
            dependency_groups=list(DIARIZEN_OFFICIAL_DEPENDENCY_GROUPS),
            payload={
                "timeline_summary": summary,
                "raw_turns": copy.deepcopy(segments),
                "official_model_chain": {
                    "segmenter": DIARIZEN_OFFICIAL_MODEL_VERSION,
                    "speaker_embedder": DIARIZEN_OFFICIAL_WESPEAKER_VERSION,
                    "clustering": "VBxClustering",
                    "plda_sha256": DIARIZEN_OFFICIAL_PLDA_SHA256,
                    "embedding_exclude_overlap": True,
                    "code": DIARIZEN_OFFICIAL_CODE_VERSION,
                },
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "empty_timeline": not bool(segments),
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return _missing_official_diarizen_evidence(
            scope,
            "%s: %s" % (exc.__class__.__name__, exc),
            config,
            runtime=runtime,
            applicability=applicability,
        )


def _missing_official_diarizen_evidence(
    scope, reason, config, runtime=None, applicability=None
):
    applicability = dict(applicability or {})
    applicability.setdefault("audio_sha256", scope["audio_sha256"])
    applicability.setdefault("model_dir", getattr(config, "model_dir", ""))
    applicability.setdefault(
        "embedding_model", getattr(config, "embedding_model", "")
    )
    return build_missing_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "speaker_timeline",
        DIARIZEN_OFFICIAL_SOURCE_NAME,
        DIARIZEN_OFFICIAL_MODEL_VERSION,
        DIARIZEN_OFFICIAL_SOURCE_KIND,
        [
            "speaker_timeline",
            "speaker_count_candidate",
            "overlap_candidate",
            "change_candidate",
        ],
        list(DIARIZEN_OFFICIAL_DEPENDENCY_GROUPS),
        reason,
        runtime=runtime,
        applicability=applicability,
    )


def collect_diarizen_evidence(
    scope, config, context=None, verify_model_asset=True
):
    """Legacy OpenASR combined collector retained for artifact compatibility."""

    diarizen_hash = DIARIZEN_MODEL_PACK_SHA256 if verify_model_asset else None
    redimnet_hash = REDIMNET_PACK_SHA256 if verify_model_asset else None
    firered_hash = (
        OPENASR_FIRERED_ASR_PACK_SHA256 if verify_model_asset else None
    )
    forced_aligner_hash = (
        FORCED_ALIGNER_PACK_SHA256 if verify_model_asset else None
    )
    runtime = {
        "engine": "OpenASR",
        "engine_version": OPENASR_VERSION,
        "executable": getattr(config, "runtime_executable", ""),
        "executable_expected_sha256": OPENASR_EXECUTABLE_SHA256,
        "executable_expected_size_bytes": OPENASR_EXECUTABLE_SIZE_BYTES,
        "diarizen_pack": getattr(config, "diarizen_pack", ""),
        "redimnet_pack": getattr(config, "redimnet_pack", ""),
        "firered_asr_pack": getattr(config, "firered_asr_pack", ""),
        "forced_aligner_pack": getattr(config, "forced_aligner_pack", ""),
        "diarizen_pack_sha256": diarizen_hash,
        "redimnet_pack_sha256": redimnet_hash,
        "firered_asr_pack_sha256": firered_hash,
        "forced_aligner_pack_sha256": forced_aligner_hash,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "openasr_version": OPENASR_VERSION,
        "diarizen_pack_sha256": diarizen_hash,
        "redimnet_pack_sha256": redimnet_hash,
        "firered_asr_pack_sha256": firered_hash,
        "forced_aligner_pack_sha256": forced_aligner_hash,
        "preprocessing": "openasr_offline_full_sample",
    }
    started = time.time()
    try:
        if verify_model_asset:
            _verify_pack_asset(
                config.diarizen_pack,
                DIARIZEN_MODEL_PACK_SHA256,
                DIARIZEN_MODEL_PACK_SIZE_BYTES,
            )
            _verify_pack_asset(
                config.redimnet_pack,
                REDIMNET_PACK_SHA256,
                REDIMNET_PACK_SIZE_BYTES,
            )
            _verify_pack_asset(
                config.firered_asr_pack,
                OPENASR_FIRERED_ASR_PACK_SHA256,
                OPENASR_FIRERED_ASR_PACK_SIZE_BYTES,
            )
            _verify_pack_asset(
                config.forced_aligner_pack,
                FORCED_ALIGNER_PACK_SHA256,
                FORCED_ALIGNER_PACK_SIZE_BYTES,
            )
        result = diarize_diarizen(
            scope["audio_path"], config, context=context
        )
        if not isinstance(result, dict):
            raise ValueError("OpenASR returned a non-object result")
        result_runtime = result.get("runtime", {})
        if not isinstance(result_runtime, dict):
            raise ValueError("OpenASR result runtime must be an object")
        runtime.update(result_runtime)
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
    except Exception as exc:  # noqa: BLE001 - command failure invalidates both.
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        reason = "%s: %s" % (exc.__class__.__name__, exc)
        return (
            _missing_openasr_firered_evidence(
                scope,
                reason,
                config,
                runtime=runtime,
                applicability=applicability,
            ),
            _missing_diarizen_evidence(
                scope,
                reason,
                config,
                runtime=runtime,
                applicability=applicability,
            ),
        )

    text = result.get("text")
    if isinstance(text, str) and text.strip():
        try:
            asr_evidence = build_evidence(
                sample_id=scope["sample_id"],
                duration_sec=scope["duration_sec"],
                evidence_type="asr_transcript",
                source_name="fireredasr2_aed",
                source_version=OPENASR_FIRERED_ASR_MODEL_VERSION,
                source_kind="openasr_asr",
                capabilities=["asr_transcript"],
                dependency_groups=[
                    "G_openasr_firered_aed_l_v2_q4_k",
                    "G_openasr_qwen3_forced_aligner_0_6b_q4_k",
                ],
                payload={
                    "text": text.strip(),
                    "asr_transcript": text.strip(),
                    "not_a_speaker_timeline": True,
                    "not_a_speaker_event_vote": True,
                },
                status="estimated",
                quality={
                    "usable": True,
                    "full_scope_invocation": True,
                    "model_asset_verified": bool(verify_model_asset),
                    "timestamp_status": "none",
                },
                runtime=runtime,
                applicability=applicability,
            )
        except Exception as exc:  # noqa: BLE001 - timeline stays independent.
            asr_evidence = _missing_openasr_firered_evidence(
                scope,
                "%s: %s" % (exc.__class__.__name__, exc),
                config,
                runtime=runtime,
                applicability=applicability,
            )
    else:
        reason = runtime.get("asr_error")
        if not reason:
            reason = (
                "OpenASR returned an empty FireRed transcript"
                if isinstance(text, str)
                else "OpenASR stdout FireRed transcript is invalid"
            )
        asr_evidence = _missing_openasr_firered_evidence(
            scope,
            reason,
            config,
            runtime=runtime,
            applicability=applicability,
        )

    segments = result.get("segments")
    if isinstance(segments, list):
        try:
            summary = summarize_timeline(segments, scope["duration_sec"])
            timeline_evidence = build_evidence(
                sample_id=scope["sample_id"],
                duration_sec=scope["duration_sec"],
                evidence_type="speaker_timeline",
                source_name="diarizen_large_s80_v2",
                source_version=DIARIZEN_MODEL_VERSION,
                source_kind="openasr_external_diarization",
                capabilities=[
                    "speaker_timeline",
                    "speaker_count_candidate",
                    "overlap_candidate",
                    "change_candidate",
                ],
                dependency_groups=[
                    "G_openasr_diarizen_large_s80_v2",
                    "G_openasr_redimnet2_b6_cn",
                    "G_openasr_firered_stream_vad",
                    "G_openasr_qwen3_forced_aligner_0_6b_q4_k",
                ],
                payload={
                    "timeline_summary": summary,
                    "raw_turns": copy.deepcopy(segments),
                    "openasr_model_chain": {
                        "segmenter": DIARIZEN_MODEL_VERSION,
                        "speaker_embedder": REDIMNET_MODEL_VERSION,
                        "forced_aligner": FORCED_ALIGNER_MODEL_VERSION,
                        "clustering": "OpenASR-0.1.40-automatic",
                    },
                },
                status="estimated",
                quality={
                    "usable": True,
                    "full_scope_invocation": True,
                    "model_asset_verified": bool(verify_model_asset),
                    "empty_timeline": not bool(segments),
                },
                runtime=runtime,
                applicability=applicability,
            )
        except Exception as exc:  # noqa: BLE001 - ASR stays independent.
            timeline_evidence = _missing_diarizen_evidence(
                scope,
                "%s: %s" % (exc.__class__.__name__, exc),
                config,
                runtime=runtime,
                applicability=applicability,
            )
    else:
        timeline_evidence = _missing_diarizen_evidence(
            scope,
            runtime.get("timeline_error")
            or "OpenASR diarization trace is invalid",
            config,
            runtime=runtime,
            applicability=applicability,
        )
    return asr_evidence, timeline_evidence


def _missing_openasr_firered_evidence(
    scope, reason, config, runtime=None, applicability=None
):
    del config
    return build_missing_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "asr_transcript",
        "fireredasr2_aed",
        OPENASR_FIRERED_ASR_MODEL_VERSION,
        "openasr_asr",
        ["asr_transcript"],
        [
            "G_openasr_firered_aed_l_v2_q4_k",
            "G_openasr_qwen3_forced_aligner_0_6b_q4_k",
        ],
        reason,
        runtime=runtime,
        applicability=applicability,
    )


def _missing_diarizen_evidence(
    scope, reason, config, runtime=None, applicability=None
):
    applicability = dict(applicability or {})
    applicability.setdefault("audio_sha256", scope["audio_sha256"])
    applicability.setdefault(
        "diarizen_pack", getattr(config, "diarizen_pack", "")
    )
    return build_missing_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "speaker_timeline",
        "diarizen_large_s80_v2",
        DIARIZEN_MODEL_VERSION,
        "openasr_external_diarization",
        [
            "speaker_timeline",
            "speaker_count_candidate",
            "overlap_candidate",
            "change_candidate",
        ],
        [
            "G_openasr_diarizen_large_s80_v2",
            "G_openasr_redimnet2_b6_cn",
            "G_openasr_firered_stream_vad",
            "G_openasr_qwen3_forced_aligner_0_6b_q4_k",
        ],
        reason,
        runtime=runtime,
        applicability=applicability,
    )


def _verify_pack_asset(path, expected_sha256, expected_size):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError("model pack is missing: %s" % path)
    actual_size = path.stat().st_size
    if actual_size != int(expected_size):
        raise ValueError(
            "model pack size mismatch for %s: expected %s, got %s"
            % (path, expected_size, actual_size)
        )
    _verify_model_asset(path.parent, path.name, expected_sha256)


def _collect_moss_and_nemotron_evidence(scope, config, context=None):
    """Run the two speaker-count systems together for every sample."""

    results = {}
    tasks = {}
    if config.enable_moss:
        tasks["moss"] = (
            _collect_asr_evidence,
            (
                "moss",
                scope,
                config.moss_config,
                context,
                config.verify_model_assets,
            ),
        )
    else:
        results["moss"] = _missing_asr_evidence(
            "moss", scope, "disabled by run configuration"
        )

    if config.enable_nemotron:
        tasks["nemotron"] = (
            collect_nemotron_evidence,
            (
                scope,
                config.nemotron_config,
                context,
                config.verify_model_assets,
            ),
        )
    else:
        results["nemotron"] = _missing_nemotron_evidence(
            scope,
            "Nemotron disabled by run configuration",
            config.nemotron_config,
        )

    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                name: executor.submit(function, *arguments)
                for name, (function, arguments) in tasks.items()
            }
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate either path.
                    reason = "%s: %s" % (exc.__class__.__name__, exc)
                    if name == "moss":
                        results[name] = _missing_asr_evidence(
                            "moss", scope, reason
                        )
                    else:
                        results[name] = _missing_nemotron_evidence(
                            scope, reason, config.nemotron_config
                        )

    return results["moss"], results["nemotron"]


def _collect_asr_evidence(source, scope, config, context, verify_model_asset):
    """Run one ASR collector and convert unexpected failures to evidence."""

    try:
        if source == "moss":
            return collect_moss_evidence(
                scope,
                config,
                context=context,
                verify_model_asset=verify_model_asset,
            )
        if source == "firered_asr":
            return collect_firered_asr_evidence(
                scope,
                config,
                context=context,
                verify_model_asset=verify_model_asset,
            )
        if source == "qwen3_asr":
            return collect_qwen3_asr_evidence(
                scope,
                config,
                context=context,
                verify_model_asset=verify_model_asset,
            )
        raise ValueError("unknown ASR collector: %s" % source)
    except Exception as exc:  # noqa: BLE001 - preserve the other ASR path.
        return _missing_asr_evidence(
            source,
            scope,
            "%s: %s" % (exc.__class__.__name__, exc),
        )


def _missing_asr_evidence(source, scope, reason):
    if source == "moss":
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "speaker_timeline",
            "moss_transcribe_diarize",
            MOSS_SOURCE_VERSION,
            "joint_asr_diarizer",
            [
                "speaker_timeline",
                "joint_speaker_text",
                "speaker_count_candidate",
            ],
            ["G_moss_td_0_9b"],
            reason,
        )
    if source == "firered_asr":
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "asr_transcript",
            "fireredasr2_aed",
            FIRERED_ASR_SOURCE_VERSION,
            "asr",
            ["asr_transcript", "lexical_timeline"],
            ["G_firered_asr2_aed"],
            reason,
        )
    if source == "qwen3_asr":
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "asr_transcript",
            "qwen3_asr_0_6b",
            QWEN3_ASR_SOURCE_VERSION,
            "asr",
            [
                "asr_transcript",
                "lexical_timeline",
                "lexical_presence_diagnostic",
            ],
            ["G_qwen3_asr_0_6b"],
            reason,
        )
    raise ValueError("unknown ASR source: %s" % source)


def _speaker_asr_transcript(timelines):
    """Return the routed ASR text (compatibility helper for older callers)."""

    return _select_speaker_asr(timelines)["text"]


def _select_speaker_asr(evidence, language_route=None):
    """Select an ASR without sending non-Chinese/English audio to FireRed."""

    moss = _asr_candidate(evidence, "moss_transcribe_diarize")
    firered = _asr_candidate(evidence, "fireredasr2_aed")
    qwen3 = _asr_candidate(evidence, "qwen3_asr_0_6b")
    language_route = language_route or _language_route_from_evidence(evidence)
    moss_route = _is_moss_route(language_route["language_route"])
    routed = moss if moss_route else qwen3
    routed_model = "moss" if moss_route else "qwen3_asr"
    count_consensus = _single_speaker_count_consensus(evidence)
    consensus_selected = bool(
        moss_route and count_consensus["all_at_most_one"]
    )
    required = firered if consensus_selected else routed
    selected_model = "firered" if consensus_selected else routed_model
    if required["usable"]:
        if consensus_selected:
            reason = (
                "MOSS and Nemotron both report at most one speaker; "
                "selected FireRed transcript"
            )
        else:
            reason = "%s; selected routed %s transcript" % (
                language_route["reason"],
                selected_model,
            )
    else:
        if consensus_selected:
            reason = (
                "MOSS and Nemotron both report at most one speaker; "
                "required FireRed transcript unavailable; failed closed"
            )
        else:
            reason = (
                "%s; required %s transcript unavailable; failed closed"
                % (language_route["reason"], selected_model)
            )

    route = {
        "schema_version": "speaker_asr_route.v5",
        "strategy": "single_speaker_consensus_firered_for_zh_en_else_qwen3_asr",
        "language_route": language_route["language_route"],
        "language_source": language_route["language_source"],
        "detected_language": language_route.get("detected_language"),
        "language_confidence": language_route.get("language_confidence"),
        "language_error": language_route.get("language_error"),
        "selection_basis": (
            "single_speaker_consensus"
            if consensus_selected
            else "language_route"
        ),
        "single_speaker_consensus": count_consensus,
        "original_required_source": routed["source"],
        "original_required_model": routed_model,
        "consensus_override_applied": bool(consensus_selected and moss_route),
        "required_source": required["source"],
        "required_model": selected_model,
        "selected_source": required["source"] if required["usable"] else None,
        "selected_model": selected_model if required["usable"] else None,
        "reason": reason,
        "fallback": False,
        "failed_closed": not required["usable"],
    }
    return {
        "text": required["text"] if required["usable"] else "",
        "route": route,
        "candidates": {"moss": moss, "firered": firered, "qwen3_asr": qwen3},
    }


def _single_speaker_count_consensus(evidence):
    candidates = {
        "moss": _speaker_count_candidate(
            evidence, "moss_transcribe_diarize"
        ),
        "nemotron": _speaker_count_candidate(
            evidence, NEMOTRON_SOURCE_NAME
        ),
    }
    return {
        "all_at_most_one": all(
            candidate["usable"] and candidate["speaker_count"] <= 1
            for candidate in candidates.values()
        ),
        "candidates": candidates,
    }


def _speaker_count_candidate(evidence, source_name):
    for item in evidence or []:
        if item.get("source", {}).get("name") != source_name:
            continue
        summary = item.get("payload", {}).get("timeline_summary", {})
        count = (
            summary.get("observed_speaker_count")
            if isinstance(summary, dict)
            else None
        )
        valid_count = (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        )
        status = item.get("status", "missing")
        usable = (
            status in ("observed", "estimated")
            and item.get("quality", {}).get("usable", True) is not False
            and valid_count
        )
        return {
            "source": source_name,
            "evidence_id": item.get("evidence_id"),
            "status": status,
            "usable": bool(usable),
            "speaker_count": count if valid_count else None,
        }
    return {
        "source": source_name,
        "evidence_id": None,
        "status": "missing",
        "usable": False,
        "speaker_count": None,
    }


def _asr_candidate(evidence, source_name):
    for item in evidence or []:
        if item.get("source", {}).get("name") != source_name:
            continue
        payload = item.get("payload", {})
        text = ""
        language = None
        language_confidence = None
        language_error = None
        if source_name == "moss_transcribe_diarize":
            text = payload.get("asr_transcript", "")
            if not isinstance(text, str) or not text.strip():
                text = _join_segment_text(
                    payload.get("timeline_summary", {}).get("segments", [])
                )
            if not text:
                raw_text = payload.get("model_output_text", "")
                if isinstance(raw_text, str) and raw_text.strip():
                    text = _join_segment_text(parse_moss_text(raw_text))
        else:
            text = payload.get("asr_transcript", payload.get("text", ""))
            language = payload.get("language") or payload.get("lang")
            language_confidence = payload.get(
                "language_confidence", payload.get("lang_confidence")
            )
            language_error = payload.get("language_error")
            if not text:
                text = _firered_text(payload)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        status = item.get("status", "estimated")
        usable = (
            status in ("observed", "estimated")
            and item.get("quality", {}).get("usable", True) is not False
            and bool(text.strip())
        )
        return {
            "source": source_name,
            "evidence_id": item.get("evidence_id"),
            "status": status,
            "usable": bool(usable),
            "text": text.strip(),
            "language": language,
            "language_confidence": language_confidence,
            "language_error": language_error,
            "quality": copy.deepcopy(item.get("quality", {})),
        }
    return {
        "source": source_name,
        "evidence_id": None,
        "status": "missing",
        "usable": False,
        "text": "",
        "language": None,
        "language_confidence": None,
        "language_error": None,
        "quality": {},
    }


def _classify_asr_language(text="", language=None, language_confidence=None):
    """Map standalone FireRed LID output to the original result route."""

    del text
    normalized = " ".join(str(language or "").strip().lower().split())
    confidence_note = ""
    if (
        isinstance(language_confidence, (int, float))
        and not isinstance(language_confidence, bool)
    ):
        confidence_note = " (confidence %.3f)" % float(language_confidence)
    if _is_chinese_lid(normalized):
        return (
            "chinese",
            "standalone FireRed LID is %s%s" % (normalized, confidence_note),
            "firered_lid",
        )
    if _is_english_lid(normalized):
        return (
            "english",
            "standalone FireRed LID is %s%s" % (normalized, confidence_note),
            "firered_lid",
        )
    if normalized:
        return (
            "non_chinese_or_unknown",
            "standalone FireRed LID is %s%s" % (normalized, confidence_note),
            "firered_lid",
        )
    return (
        "non_chinese_or_unknown",
        "standalone FireRed LID is unavailable; unknown routes other-language branch",
        "firered_lid_unavailable",
    )


def _is_chinese_lid(language):
    normalized = " ".join(str(language or "").strip().lower().split())
    return bool(
        normalized == "zh"
        or normalized.startswith("zh-")
        or normalized.startswith("zh_")
        or normalized.startswith("zh ")
    )


def _is_english_lid(language):
    normalized = " ".join(str(language or "").strip().lower().split())
    return bool(
        normalized == "en"
        or normalized.startswith("en-")
        or normalized.startswith("en_")
        or normalized in ("english", "en english")
    )


def _is_moss_route(route):
    return str(route or "") in ("chinese", "english")


def _language_route_from_evidence(evidence):
    lid = _lid_candidate(evidence)
    route, reason, source = _classify_asr_language(
        language=lid.get("language"),
        language_confidence=lid.get("language_confidence"),
    )
    if lid.get("usable") and lid.get("source"):
        source = lid["source"]
        if source == "speaker_segment_firered_lid":
            reason = "speaker-segment FireRed LID selected %s" % (
                str(lid.get("language")).strip().lower()
            )
    if source == "speaker_segment_firered_lid":
        strategy = (
            "speaker_segment_lid_first_then_full_audio_fallback; "
            "moss_or_firered_for_chinese_or_english_else_nemotron_plus_qwen3_asr"
        )
    else:
        strategy = (
            "full_audio_firered_lid; "
            "moss_or_firered_for_chinese_or_english_else_nemotron_plus_qwen3_asr"
        )
    return {
        "schema_version": "speaker_language_route.v3",
        "strategy": strategy,
        "language_route": route,
        "language_source": source,
        "detected_language": lid.get("language"),
        "language_confidence": lid.get("language_confidence"),
        "language_error": lid.get("language_error"),
        "lid_evidence_id": lid.get("evidence_id"),
        "timeline_source": (
            "moss_transcribe_diarize"
            if _is_moss_route(route)
            else NEMOTRON_SOURCE_NAME
        ),
        "asr_source": (
            "moss_transcribe_diarize"
            if _is_moss_route(route)
            else "qwen3_asr_0_6b"
        ),
        "reason": reason,
        "fallback": False,
    }


def _lid_candidate(evidence):
    preferred_sources = ("speaker_segment_firered_lid", "firered_lid")
    for source_name in preferred_sources:
        for item in evidence or []:
            if item.get("source", {}).get("name") != source_name:
                continue
            payload = item.get("payload", {})
            language = payload.get("language")
            status = item.get("status", "missing")
            usable = (
                status in ("observed", "estimated")
                and item.get("quality", {}).get("usable", True) is not False
                and isinstance(language, str)
                and bool(language.strip())
            )
            if not usable:
                continue
            return {
                "evidence_id": item.get("evidence_id"),
                "status": status,
                "usable": True,
                "language": language.strip(),
                "language_confidence": payload.get("language_confidence"),
                "language_error": None,
                "source": source_name,
            }
    for item in evidence or []:
        if item.get("source", {}).get("name") not in preferred_sources:
            continue
        payload = item.get("payload", {})
        language = payload.get("language")
        status = item.get("status", "missing")
        usable = (
            status in ("observed", "estimated")
            and item.get("quality", {}).get("usable", True) is not False
            and isinstance(language, str)
            and bool(language.strip())
        )
        return {
            "evidence_id": item.get("evidence_id"),
            "status": status,
            "usable": bool(usable),
            "language": language.strip() if usable else None,
            "language_confidence": payload.get("language_confidence"),
            "language_error": (
                None if usable else item.get("quality", {}).get("reason")
            ),
            "source": item.get("source", {}).get("name"),
        }
    return {
        "evidence_id": None,
        "status": "missing",
        "usable": False,
        "language": None,
        "language_confidence": None,
        "language_error": "standalone FireRed LID evidence is missing",
        "source": "firered_lid_unavailable",
    }


def _join_segment_text(segments):
    indexed_segments = [
        (index, segment)
        for index, segment in enumerate(segments or [])
        if (
            isinstance(segment, dict)
            and isinstance(segment.get("start_sec"), (int, float))
            and not isinstance(segment.get("start_sec"), bool)
            and isinstance(segment.get("text"), str)
            and segment.get("text").strip()
        )
    ]
    indexed_segments.sort(
        key=lambda item: (item[1]["start_sec"], item[0])
    )
    return " ".join(
        str(segment.get("text", "")).strip()
        for _, segment in indexed_segments
    ).strip()


def collect_moss_evidence(
    scope, config, context=None, verify_model_asset=True
):
    verified_hash = MOSS_MODEL_SHA256 if verify_model_asset else None
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model,
        "model_sha256": verified_hash,
        "device": config.device,
        "torch_dtype": config.torch_dtype,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    try:
        if verify_model_asset:
            _verify_model_asset(
                config.model,
                "model-00000-of-00001.safetensors",
                MOSS_MODEL_SHA256,
            )
        result = run_moss_diarizer(
            scope["audio_path"],
            duration_sec=scope["duration_sec"],
            context=context,
            config=config,
        )
        asr_transcript = _join_segment_text(result.value.get("segments", []))
        summary = summarize_timeline(
            result.value.get("segments", []), scope["duration_sec"]
        )
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="joint_speaker_timeline_and_text",
            source_name="moss_transcribe_diarize",
            source_version=MOSS_SOURCE_VERSION,
            source_kind="joint_asr_diarizer",
            capabilities=[
                "speaker_timeline",
                "joint_speaker_text",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
            ],
            dependency_groups=["G_moss_td_0_9b"],
            payload={
                "timeline_summary": summary,
                "asr_transcript": asr_transcript,
                "model_output_text": result.value.get("raw_text", ""),
                "text_dependency_note": (
                    "speaker, timestamp, and text share one joint model group"
                ),
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "calibration_profile_id": None,
                "joint_negative_profile_id": None,
                "model_asset_verified": bool(verify_model_asset),
            },
            runtime=runtime,
            applicability={
                "sample_rate_hz": scope["sample_rate_hz"],
                "channels": scope["channels"],
                "audio_sha256": scope["audio_sha256"],
                "model_sha256": verified_hash,
                "max_new_tokens": config.max_new_tokens,
                "torch_dtype": config.torch_dtype,
                "prompt_sha256": hashlib.sha256(
                    config.prompt.encode("utf-8")
                ).hexdigest(),
                "preprocessing": "openmoss_processor_16khz_full_sample",
            },
        )
    except Exception as exc:
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "joint_speaker_timeline_and_text",
            "moss_transcribe_diarize",
            MOSS_SOURCE_VERSION,
            "joint_asr_diarizer",
            [
                "speaker_timeline",
                "joint_speaker_text",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
            ],
            ["G_moss_td_0_9b"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability={
                "audio_sha256": scope["audio_sha256"],
                "model_sha256": verified_hash,
                "max_new_tokens": config.max_new_tokens,
                "torch_dtype": config.torch_dtype,
                "prompt_sha256": hashlib.sha256(
                    config.prompt.encode("utf-8")
                ).hexdigest(),
                "preprocessing": "openmoss_processor_16khz_full_sample",
            },
        )


def collect_firered_asr_evidence(
    scope, config, context=None, verify_model_asset=True
):
    """Collect FireRedASR2-AED as an ASR-only, non-diarization source.

    FireRed is deliberately kept out of the speaker timeline vote. Its text is
    eligible only for Chinese/English audio when both MOSS and Nemotron report
    at most one speaker. Other languages use Qwen3-ASR-0.6B instead.
    """

    verified_hash = FIRERED_ASR_MODEL_SHA256 if verify_model_asset else None
    model_dir = getattr(config, "model_dir", "")
    source_dir = getattr(config, "source_dir", "")
    runtime = {
        "python": getattr(config, "subprocess_python", ""),
        "model_path": model_dir,
        "source_path": source_dir,
        "model_sha256": verified_hash,
        "device": getattr(config, "device", "auto"),
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": verified_hash,
        "source_dir": source_dir,
        "beam_size": getattr(config, "beam_size", None),
        "use_half": bool(getattr(config, "use_half", False)),
        "preprocessing": "soundfile_pcm16_16khz_mono",
        "not_a_speaker_timeline": True,
    }
    started = time.time()
    try:
        if verify_model_asset:
            _verify_model_asset(
                Path(model_dir).expanduser(),
                "model.pth.tar",
                FIRERED_ASR_MODEL_SHA256,
            )
        runtime_config = config
        if isinstance(config, FireRedAsrConfig) and (
            config.verify_lid_model_asset != bool(verify_model_asset)
        ):
            runtime_config_record = config.to_record()
            runtime_config_record["verify_lid_model_asset"] = bool(
                verify_model_asset
            )
            runtime_config = FireRedAsrConfig(**runtime_config_record)
        result = run_firered_asr(
            scope["audio_path"], config=runtime_config, context=context
        )
        raw = result.value if hasattr(result, "value") else result
        if not isinstance(raw, dict):
            raise ValueError("FireRed ASR returned a non-object result")
        text = _firered_text(raw)
        units = _firered_lexical_units(raw, scope["duration_sec"])
        runtime.update(raw.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        language, language_confidence = _firered_language(raw)
        stable_raw_output = copy.deepcopy(raw)
        # Runtime timing/device fields are volatile and belong in the evidence
        # runtime envelope, not in the content used to derive evidence_id.
        stable_raw_output.pop("runtime", None)
        payload = {
            "text": text,
            "asr_transcript": text,
            "confidence": raw.get("confidence"),
            "language": language,
            "language_confidence": language_confidence,
            "language_error": raw.get("language_error"),
            "sentences": raw.get("sentences", []),
            "timestamps": raw.get("timestamps", raw.get("timestamp", [])),
            "lexical_units": units,
            "raw_output": stable_raw_output,
            "not_a_speaker_timeline": True,
            "not_a_speaker_event_vote": True,
        }
        usable = bool(text or units)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="asr_transcript",
            source_name="fireredasr2_aed",
            source_version=FIRERED_ASR_SOURCE_VERSION,
            source_kind="asr",
            capabilities=[
                "asr_transcript",
                "lexical_timeline",
                "lexical_presence_diagnostic",
            ],
            dependency_groups=["G_firered_asr2_aed"],
            payload=payload,
            status="estimated",
            quality={
                "usable": usable,
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "timestamp_status": "native" if units else "none",
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "asr_transcript",
            "fireredasr2_aed",
            FIRERED_ASR_SOURCE_VERSION,
            "asr",
            ["asr_transcript", "lexical_timeline", "lexical_presence_diagnostic"],
            ["G_firered_asr2_aed"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def collect_qwen3_asr_evidence(
    scope, config, context=None, verify_model_asset=True
):
    """Collect Qwen3-ASR-0.6B text for non-Chinese/English routes."""

    verified_hash = QWEN3_ASR_MODEL_SHA256 if verify_model_asset else None
    model_dir = getattr(config, "model_dir", "")
    runtime = {
        "python": getattr(config, "subprocess_python", ""),
        "model_path": model_dir,
        "model_sha256": verified_hash,
        "device": getattr(config, "device", "cuda:0"),
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": verified_hash,
        "forced_aligner_model_dir": getattr(
            config, "forced_aligner_model_dir", ""
        ),
        "return_time_stamps": bool(
            getattr(config, "return_time_stamps", False)
        ),
        "torch_dtype": getattr(config, "torch_dtype", None),
        "max_new_tokens": getattr(config, "max_new_tokens", None),
        "preprocessing": "pcm16_16khz_mono",
        "language_hint": None,
        "language_detection": "qwen3_asr_automatic",
        "not_a_speaker_timeline": True,
    }
    started = time.time()
    try:
        if verify_model_asset:
            _verify_model_asset(
                Path(model_dir).expanduser(),
                "model.safetensors",
                QWEN3_ASR_MODEL_SHA256,
            )
        result = run_qwen3_asr(
            scope["audio_path"], config=config, context=context
        )
        raw = result.value if hasattr(result, "value") else result
        if not isinstance(raw, dict):
            raise ValueError("Qwen3 ASR returned a non-object result")
        text = raw.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()
        language = raw.get("language") or raw.get("lang")
        if not isinstance(language, str) or not language.strip():
            language = None
        else:
            language = language.strip()
        runtime.update(raw.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        stable_raw_output = copy.deepcopy(raw)
        stable_raw_output.pop("runtime", None)
        lexical_units = _qwen_lexical_units(raw, scope["duration_sec"])
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="asr_transcript",
            source_name="qwen3_asr_0_6b",
            source_version=QWEN3_ASR_SOURCE_VERSION,
            source_kind="asr",
            capabilities=[
                "asr_transcript",
                "lexical_timeline",
                "lexical_presence_diagnostic",
            ],
            dependency_groups=["G_qwen3_asr_0_6b"],
            payload={
                "text": text,
                "asr_transcript": text,
                "language": language,
                "timestamps": copy.deepcopy(raw.get("timestamps", [])),
                "lexical_units": lexical_units,
                "raw_output": stable_raw_output,
                "not_a_speaker_timeline": True,
                "not_a_speaker_event_vote": True,
            },
            status="estimated",
            quality={
                "usable": bool(text or lexical_units),
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "timestamp_status": "native" if lexical_units else "none",
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "asr_transcript",
            "qwen3_asr_0_6b",
            QWEN3_ASR_SOURCE_VERSION,
            "asr",
            [
                "asr_transcript",
                "lexical_timeline",
                "lexical_presence_diagnostic",
            ],
            ["G_qwen3_asr_0_6b"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def _firered_text(raw):
    text = raw.get("text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    sentences = raw.get("sentences", [])
    if isinstance(sentences, list):
        parts = [
            item.get("text", "").strip()
            for item in sentences
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item.get("text", "").strip()
        ]
        return " ".join(parts).strip()
    return ""


def _firered_language(raw):
    language = raw.get("language") or raw.get("lang")
    confidence = raw.get("language_confidence", raw.get("lang_confidence"))
    if not language:
        sentences = raw.get("sentences", [])
        if isinstance(sentences, list):
            for item in sentences:
                if isinstance(item, dict) and item.get("lang"):
                    language = item.get("lang")
                    confidence = item.get("lang_confidence", confidence)
                    break
    if not isinstance(language, str) or not language.strip():
        return None, None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    return language.strip(), confidence


def _firered_lexical_units(raw, duration_sec):
    """Normalize FireRed native timestamp variants to lexical.py's schema."""

    candidates = raw.get("timestamps", raw.get("timestamp", []))
    if not isinstance(candidates, list):
        candidates = []
    units = []
    for index, item in enumerate(candidates):
        token = start = end = None
        if isinstance(item, dict):
            token = item.get("text", item.get("token"))
            start = item.get("start_s", item.get("start_sec", item.get("start")))
            end = item.get("end_s", item.get("end_sec", item.get("end")))
            if start is None and item.get("start_ms") is not None:
                start = float(item["start_ms"]) / 1000.0
            if end is None and item.get("end_ms") is not None:
                end = float(item["end_ms"]) / 1000.0
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            token, start, end = item[0], item[1], item[2]
        if not isinstance(token, str) or not token.strip():
            continue
        try:
            start = max(0.0, float(start))
            end = min(float(duration_sec), float(end))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        units.append(
            {
                "unit_id": "firered_%06d" % index,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "text": token.strip(),
            }
        )
    units.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["unit_id"]))
    return units


def _qwen_lexical_units(raw, duration_sec):
    """Normalize Qwen3 native forced-aligner spans to lexical.py's schema."""

    candidates = raw.get("timestamps", raw.get("timestamp", []))
    if not isinstance(candidates, list):
        candidates = []
    units = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        token = item.get("text", item.get("token"))
        start = item.get(
            "start_sec",
            item.get("start_s", item.get("start_time", item.get("start"))),
        )
        end = item.get(
            "end_sec",
            item.get("end_s", item.get("end_time", item.get("end"))),
        )
        if not isinstance(token, str) or not token.strip():
            continue
        try:
            start = max(0.0, float(start))
            end = min(float(duration_sec), float(end))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        units.append(
            {
                "unit_id": "qwen3_%06d" % index,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "text": token.strip(),
            }
        )
    units.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["unit_id"]))
    return units


def collect_whisper_evidence(
    scope, config, context=None, verify_model_asset=True
):
    verified_hash = WHISPER_CHECKPOINT_SHA256 if verify_model_asset else None
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_path,
        "model_sha256": verified_hash,
        "device": config.device,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": verified_hash,
        "language_hint": config.language,
        "word_timestamps_requested": config.word_timestamps,
        "portable_word_timing": config.portable_word_timing,
        "preprocessing": "librosa_16khz_mono_full_sample",
    }
    started = time.time()
    try:
        if verify_model_asset:
            model_path = Path(config.model_path).expanduser()
            _verify_model_asset(
                model_path.parent,
                model_path.name,
                WHISPER_CHECKPOINT_SHA256,
            )
        result = transcribe_whisper(
            scope["audio_path"], config, context=context
        )
        lexical_units = []
        for raw in result.get("lexical_units", []):
            start_sec = max(0.0, float(raw["start_sec"]))
            end_sec = min(scope["duration_sec"], float(raw["end_sec"]))
            if end_sec <= start_sec:
                continue
            unit = dict(raw)
            unit["start_sec"] = round(start_sec, 6)
            unit["end_sec"] = round(end_sec, 6)
            lexical_units.append(unit)
        runtime.update(result.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        usable = bool(lexical_units or str(result.get("text", "")).strip())
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="lexical_timeline",
            source_name="whisper_base_lexical_clock",
            source_version=WHISPER_SOURCE_VERSION,
            source_kind="asr",
            capabilities=[
                "lexical_timeline",
                "lexical_presence_diagnostic",
                "boundary_ambiguity_guard",
            ],
            dependency_groups=["G_whisper_base_official"],
            payload={
                "text": str(result.get("text", "")),
                "language": result.get("language"),
                "lexical_units": lexical_units,
                "not_a_speaker_timeline": True,
                "not_a_speaker_event_vote": True,
            },
            status="estimated",
            quality={
                "usable": usable,
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "calibration_profile_id": None,
                "timestamp_status": (
                    "experimental_attention_dtw"
                    if config.word_timestamps
                    else "coarse_decode_segments"
                ),
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "lexical_timeline",
            "whisper_base_lexical_clock",
            WHISPER_SOURCE_VERSION,
            "asr",
            [
                "lexical_timeline",
                "lexical_presence_diagnostic",
                "boundary_ambiguity_guard",
            ],
            ["G_whisper_base_official"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def collect_sortformer_evidence(
    scope, config, context=None, verify_model_asset=True
):
    verified_hash = SORTFORMER_CHECKPOINT_SHA256 if verify_model_asset else None
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_path,
        "model_sha256": verified_hash,
        "device": config.device,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": verified_hash,
        "maximum_speakers": SORTFORMER_MAXIMUM_SPEAKERS,
        "trusted_speaker_upper_bound_leq_4": False,
        "can_supply_global_speaker_upper_bound": False,
        "can_supply_complete_negative": False,
        "frame_stride_sec": 0.08,
        "preprocessing": "nemo_16khz_full_sample_streaming",
    }
    started = time.time()
    try:
        if verify_model_asset:
            model_path = Path(config.model_path).expanduser()
            _verify_model_asset(
                model_path.parent,
                model_path.name,
                SORTFORMER_CHECKPOINT_SHA256,
            )
        result = diarize_sortformer(
            scope["audio_path"], config, context=context
        )
        summary = summarize_timeline(
            result.get("segments", []), scope["duration_sec"]
        )
        runtime.update(result.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_timeline",
            source_name="nvidia_streaming_sortformer_4spk_v2",
            source_version=SORTFORMER_SOURCE_VERSION,
            source_kind="end_to_end_streaming_diarizer",
            capabilities=[
                "speaker_timeline",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
                "positive_event_witness",
            ],
            dependency_groups=["G_nvidia_sortformer_4spk_v2"],
            payload={
                "timeline_summary": summary,
                "frame_probabilities": result.get("probabilities", {}),
                "maximum_speakers": SORTFORMER_MAXIMUM_SPEAKERS,
                "not_a_global_speaker_upper_bound": True,
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "model_asset_verified": bool(verify_model_asset),
                "calibration_profile_id": None,
                "joint_negative_profile_id": None,
                "joint_negative_claims": [],
                "positive_event_capability": True,
                "negative_capability": False,
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "speaker_timeline",
            "nvidia_streaming_sortformer_4spk_v2",
            SORTFORMER_SOURCE_VERSION,
            "end_to_end_streaming_diarizer",
            [
                "speaker_timeline",
                "speaker_count_candidate",
                "overlap_candidate",
                "change_candidate",
                "positive_event_witness",
            ],
            ["G_nvidia_sortformer_4spk_v2"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def collect_vad_evidence(
    scope, config, context=None, verify_model_asset=True
):
    verified_hash = FIRERED_VAD_MODEL_SHA256 if verify_model_asset else None
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_dir,
        "model_sha256": verified_hash,
        "device": "cpu" if not config.use_gpu else "cuda",
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    try:
        if verify_model_asset:
            _verify_model_asset(
                config.model_dir,
                "model.pth.tar",
                FIRERED_VAD_MODEL_SHA256,
            )
        result = run_firered_vad(
            scope["audio_path"],
            scope["duration_sec"],
            context=context,
            config=config,
        )
        speech = result.evidence.get("speech_segments", [])
        speech_duration = sum(
            max(0.0, item["end_sec"] - item["start_sec"]) for item in speech
        )
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speech_coverage",
            source_name="firered_vad",
            source_version="FireRedVAD@shared-20260811",
            source_kind="vad",
            capabilities=["speech_coverage"],
            dependency_groups=["G_firered_vad"],
            payload={
                "speech_segments": speech,
                "silence_segments": result.value,
                "speech_coverage_ratio": round(
                    speech_duration / scope["duration_sec"], 6
                ),
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "calibration_profile_id": None,
                "negative_capability": False,
                "model_asset_verified": bool(verify_model_asset),
            },
            runtime=runtime,
            applicability={
                "audio_sha256": scope["audio_sha256"],
                "model_sha256": verified_hash,
                "speech_threshold": config.speech_threshold,
                "min_speech_frame": config.min_speech_frame,
                "min_silence_frame": config.min_silence_frame,
                "preprocessing": "firered_16khz_mono_full_sample",
            },
        )
    except Exception as exc:
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "speech_coverage",
            "firered_vad",
            "FireRedVAD@shared-20260811",
            "vad",
            ["speech_coverage"],
            ["G_firered_vad"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability={
                "audio_sha256": scope["audio_sha256"],
                "model_sha256": verified_hash,
                "speech_threshold": config.speech_threshold,
                "min_speech_frame": config.min_speech_frame,
                "min_silence_frame": config.min_silence_frame,
                "preprocessing": "firered_16khz_mono_full_sample",
            },
        )


def collect_brouhaha_evidence(
    scope, config, context=None, verify_model_asset=True
):
    """Collect Brouhaha as coverage-only evidence.

    Brouhaha's SNR/C50 outputs are intentionally not copied into this record,
    and this evidence never declares a speaker-event capability.
    """

    verified_hash = BROUHAHA_MODEL_SHA256 if verify_model_asset else None
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_path,
        "model_sha256": verified_hash,
        "device": "cuda" if config.use_gpu else "cpu",
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_id": BROUHAHA_MODEL_ID,
        "model_sha256": verified_hash,
        "binarization": dict(BROUHAHA_BINARIZATION_PARAMETERS),
        "preprocessing": "brouhaha_pyannote_full_sample",
        "not_a_speaker_event_vote": True,
    }
    try:
        if verify_model_asset:
            verify_brouhaha_model_asset(config.model_path)
        result = estimate_brouhaha_coverage(
            scope["audio_path"],
            scope["duration_sec"],
            config=config,
            context=context,
        )
        runtime.update(result.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speech_coverage",
            source_name="brouhaha_vad",
            source_version=str(config.model_version),
            source_kind="vad",
            capabilities=["speech_coverage"],
            dependency_groups=list(BROUHAHA_DEPENDENCY_GROUPS),
            payload={
                "raw_speech_segments": result["raw_speech_segments"],
                "speech_segments": result["speech_segments"],
                "silence_segments": result["silence_segments"],
                "speech_duration_sec": result["speech_duration_sec"],
                "speech_coverage_ratio": result["speech_coverage_ratio"],
                "binarization": result["binarization"],
                "boundary_postprocess": result["boundary_postprocess"],
                "not_a_speaker_timeline": True,
                "not_a_speaker_event_vote": True,
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "calibration_profile_id": (
                    "ami_utterance_1k_v1_brouhaha_upstream_0_78_v1"
                ),
                "empty_prediction_is_valid": True,
                "model_asset_verified": bool(verify_model_asset),
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "speech_coverage",
            "brouhaha_vad",
            str(config.model_version),
            "vad",
            ["speech_coverage"],
            list(BROUHAHA_DEPENDENCY_GROUPS),
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def collect_pyannote_evidence(
    scope, config, context=None, verify_model_asset=True
):
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_dir,
        "device": config.device,
        "full_sample_invocation": True,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_revision": PYANNOTE_MODEL_REVISION,
        "model_sha256": PYANNOTE_MODEL_SHA256 if verify_model_asset else None,
        "sample_rate_hz": scope["sample_rate_hz"],
        "channels": scope["channels"],
        "preprocessing": "soundfile_preloaded_waveform_no_torchcodec",
        "full_sample_invocation": True,
    }
    capabilities = [
        "speaker_timeline",
        "speaker_count_candidate",
        "overlap_candidate",
        "change_candidate",
    ]
    try:
        if verify_model_asset:
            verify_pyannote_model_assets(config.model_dir)
        result = diarize_pyannote(
            scope["audio_path"], config, context=context
        )
        raw_summary = summarize_timeline(
            result.get("raw_segments", []),
            scope["duration_sec"],
            min_activity_sec=config.min_activity_sec,
        )
        exclusive_summary = summarize_timeline(
            result.get("exclusive_segments", []),
            scope["duration_sec"],
            min_activity_sec=config.min_activity_sec,
        )
        runtime.update(result.get("runtime", {}))
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        certification_eligible = bool(config.calibration_profile_id) and (
            config.license_review_status == "approved"
        )
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_timeline",
            source_name="pyannote_community_1",
            source_version=PYANNOTE_MODEL_VERSION,
            source_kind="powerset_speaker_diarization",
            capabilities=capabilities,
            dependency_groups=["G_pyannote_community1"],
            payload={
                "timeline_summary": raw_summary,
                "exclusive_timeline_summary": exclusive_summary,
                "exclusive_timeline_is_derived_audit_view": True,
            },
            status="estimated",
            quality={
                "usable": True,
                "full_scope_invocation": True,
                "calibration_profile_id": config.calibration_profile_id,
                "joint_negative_profile_id": (
                    config.joint_negative_profile_id
                ),
                "joint_negative_claims": (
                    ["multi_speaker", "speaker_overlap", "speaker_change"]
                    if config.joint_negative_profile_id
                    else []
                ),
                "license_id": "CC-BY-4.0",
                "license_review_status": config.license_review_status,
                "counts_for_certification": certification_eligible,
                "model_asset_verified": bool(verify_model_asset),
            },
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_total_sec"] = round(time.time() - started, 6)
        return build_missing_evidence(
            scope["sample_id"],
            scope["duration_sec"],
            "speaker_timeline",
            "pyannote_community_1",
            PYANNOTE_MODEL_VERSION,
            "powerset_speaker_diarization",
            capabilities,
            ["G_pyannote_community1"],
            "%s: %s" % (exc.__class__.__name__, exc),
            runtime=runtime,
            applicability=applicability,
        )


def collect_campplus_evidence(
    scope,
    timeline_evidence,
    config,
    context=None,
    verify_model_asset=True,
):
    verified_hash = CAMPPLUS_CHECKPOINT_SHA256 if verify_model_asset else None
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_dir,
        "model_sha256": verified_hash,
        "device": config.device,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "model_sha256": verified_hash,
        "threshold": config.threshold,
        "min_region_duration_sec": config.min_region_duration_sec,
        "max_regions_per_speaker": config.max_regions_per_speaker,
        "language_domain": "checkpoint_primarily_zh_demo_en_uncalibrated",
    }
    lineage = {"parent_evidence_ids": [timeline_evidence["evidence_id"]]}
    try:
        if verify_model_asset:
            _verify_model_asset(
                config.model_dir,
                "campplus_cn_common.bin",
                CAMPPLUS_CHECKPOINT_SHA256,
            )
        result = compare_campplus_regions(
            scope["audio_path"],
            timeline_evidence["payload"]["timeline_summary"],
            config,
            context=context,
        )
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_identity_matrix",
            source_name="campplus_speaker_verification",
            source_version=CAMPPLUS_SOURCE_VERSION,
            source_kind="speaker_embedding",
            capabilities=["speaker_identity_comparison"],
            dependency_groups=["G_campplus_sv_v1_0_0"],
            payload={
                "candidate_source_evidence_id": timeline_evidence["evidence_id"],
                "regions": result.get("regions", []),
                "comparisons": result.get("comparisons", []),
                "not_a_timeline": True,
                "not_a_speaker_count_vote": True,
            },
            status="estimated",
            quality={
                "usable": bool(result.get("comparisons")),
                "calibration_profile_id": None,
                "threshold_status": "upstream_default_uncalibrated_for_ami",
                "model_group_independent": True,
                "candidate_selection_independent": False,
                "validator_dependency_closure_independent": False,
                "candidate_selection_source": timeline_evidence["evidence_id"],
                "model_asset_verified": bool(verify_model_asset),
            },
            lineage=lineage,
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_identity_matrix",
            source_name="campplus_speaker_verification",
            source_version=CAMPPLUS_SOURCE_VERSION,
            source_kind="speaker_embedding",
            capabilities=["speaker_identity_comparison"],
            dependency_groups=["G_campplus_sv_v1_0_0"],
            payload={
                "candidate_source_evidence_id": timeline_evidence["evidence_id"],
                "comparisons": [],
            },
            status="missing",
            quality={
                "usable": False,
                "reason": "%s: %s" % (exc.__class__.__name__, exc),
                "calibration_profile_id": None,
                "candidate_selection_independent": False,
                "validator_dependency_closure_independent": False,
            },
            lineage=lineage,
            runtime=runtime,
            applicability=applicability,
        )


def collect_ecapa_evidence(
    scope,
    timeline_evidence,
    config,
    context=None,
    verify_model_asset=True,
):
    """Collect ECAPA comparisons from blind predicted-timeline regions."""

    checkpoint_hash = ECAPA_CHECKPOINT_SHA256 if verify_model_asset else None
    hyperparams_hash = (
        ECAPA_HYPERPARAMS_SHA256 if verify_model_asset else None
    )
    started = time.time()
    runtime = {
        "python": config.subprocess_python,
        "model_path": config.model_dir,
        "model_sha256": checkpoint_hash,
        "device": config.device,
        "audio_sha256": scope["audio_sha256"],
    }
    applicability = {
        "audio_sha256": scope["audio_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "hyperparams_sha256": hyperparams_hash,
        "threshold": config.threshold,
        "calibration_profile_id": config.calibration_profile_id or None,
        "min_region_duration_sec": config.min_region_duration_sec,
        "max_regions_per_speaker": config.max_regions_per_speaker,
        "candidate_region_source": "predicted_timeline_nonoverlap_only",
        "atomic_calibration_not_production_region_calibration": True,
    }
    lineage = {"parent_evidence_ids": [timeline_evidence["evidence_id"]]}
    try:
        if verify_model_asset:
            _verify_model_asset(
                config.model_dir,
                "embedding_model.ckpt",
                ECAPA_CHECKPOINT_SHA256,
            )
            _verify_model_asset(
                config.model_dir,
                "hyperparams.yaml",
                ECAPA_HYPERPARAMS_SHA256,
            )
        result = compare_ecapa_regions(
            scope["audio_path"],
            timeline_evidence["payload"]["timeline_summary"],
            config,
            context=context,
        )
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_identity_matrix",
            source_name="speechbrain_ecapa_voxceleb",
            source_version=ECAPA_MODEL_VERSION,
            source_kind="speaker_embedding",
            capabilities=["speaker_identity_comparison"],
            dependency_groups=["G_speechbrain_ecapa_voxceleb"],
            payload={
                "candidate_source_evidence_id": timeline_evidence["evidence_id"],
                "regions": result.get("regions", []),
                "comparisons": result.get("comparisons", []),
                "score_kind": result.get("score_kind", "cosine_similarity"),
                "not_a_timeline": True,
                "not_a_speaker_count_vote": True,
                "native_metadata_entered_inference": False,
            },
            status="estimated",
            quality={
                "usable": bool(result.get("comparisons")),
                "calibration_profile_id": config.calibration_profile_id or None,
                "calibration_scope": "atomic_clean_region_dev",
                "predicted_region_gate_passed": False,
                "counts_for_certification": False,
                "model_group_independent": True,
                "candidate_selection_independent": False,
                "validator_dependency_closure_independent": False,
                "candidate_selection_source": timeline_evidence["evidence_id"],
                "model_asset_verified": bool(verify_model_asset),
            },
            lineage=lineage,
            runtime=runtime,
            applicability=applicability,
        )
    except Exception as exc:
        runtime["elapsed_sec"] = round(time.time() - started, 6)
        return build_evidence(
            sample_id=scope["sample_id"],
            duration_sec=scope["duration_sec"],
            evidence_type="speaker_identity_matrix",
            source_name="speechbrain_ecapa_voxceleb",
            source_version=ECAPA_MODEL_VERSION,
            source_kind="speaker_embedding",
            capabilities=["speaker_identity_comparison"],
            dependency_groups=["G_speechbrain_ecapa_voxceleb"],
            payload={
                "candidate_source_evidence_id": timeline_evidence["evidence_id"],
                "comparisons": [],
            },
            status="missing",
            quality={
                "usable": False,
                "reason": "%s: %s" % (exc.__class__.__name__, exc),
                "calibration_profile_id": config.calibration_profile_id or None,
                "counts_for_certification": False,
                "candidate_selection_independent": False,
                "validator_dependency_closure_independent": False,
            },
            lineage=lineage,
            runtime=runtime,
            applicability=applicability,
        )


def _missing_ecapa_evidence(scope, config, reason):
    return build_missing_evidence(
        scope["sample_id"],
        scope["duration_sec"],
        "speaker_identity_matrix",
        "speechbrain_ecapa_voxceleb",
        ECAPA_MODEL_VERSION,
        "speaker_embedding",
        ["speaker_identity_comparison"],
        ["G_speechbrain_ecapa_voxceleb"],
        reason,
        applicability={
            "audio_sha256": scope["audio_sha256"],
            "checkpoint_sha256": ECAPA_CHECKPOINT_SHA256,
            "hyperparams_sha256": ECAPA_HYPERPARAMS_SHA256,
            "threshold": config.threshold,
            "calibration_profile_id": config.calibration_profile_id or None,
            "candidate_region_source": "predicted_timeline_nonoverlap_only",
        },
    )


def _select_identity_candidate_timeline(timelines, claim_policy):
    usable_by_source = {}
    for item in timelines:
        if not item.get("quality", {}).get("usable", True):
            continue
        usable_by_source.setdefault(item["source"]["name"], []).append(item)
    rule = claim_policy["claims"]["speaker_count"]
    for field in ("primary_sources", "fallback_sources"):
        for source_name in rule[field]:
            candidates = usable_by_source.get(source_name, [])
            if candidates:
                return candidates[0]
    return None


def _selected_timeline_output(timeline_evidence, duration_sec):
    """Expose routed activity and overlap intervals for downstream audio tools."""

    if not isinstance(timeline_evidence, dict):
        return None
    payload = timeline_evidence.get("payload", {})
    summary = payload.get("timeline_summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        return None
    raw_segments = summary.get("activity_segments") or summary.get("segments") or []
    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if (
            not speaker_id
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
        ):
            continue
        segments.append(
            {
                "speaker_id": str(speaker_id),
                "start_sec": round(max(0.0, start), 6),
                "end_sec": round(min(float(duration_sec), end), 6),
            }
        )
    segments = [item for item in segments if item["end_sec"] > item["start_sec"]]
    segments.sort(
        key=lambda item: (
            item["start_sec"],
            item["end_sec"],
            item["speaker_id"],
        )
    )
    raw_overlap = summary.get("overlap_activity_segments")
    if not isinstance(raw_overlap, list):
        raw_overlap = summary.get("overlap_segments") or []
    overlap_segments = []
    for item in raw_overlap:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
        ):
            continue
        start = max(0.0, start)
        end = min(float(duration_sec), end)
        if end > start:
            overlap_segments.append(
                {
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                }
            )
    overlap_segments.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    source = timeline_evidence.get("source", {})
    return {
        "source": source.get("name") if isinstance(source, dict) else None,
        "evidence_id": timeline_evidence.get("evidence_id"),
        "duration_sec": round(float(duration_sec), 6),
        "activity_segments": segments,
        "overlap_activity_segments": overlap_segments,
    }


def _select_profile_coverage_evidence(evidence):
    """Select one speech-coverage source for profile duration accounting."""

    usable = [
        item
        for item in evidence
        if "speech_coverage" in item.get("capabilities", [])
        and item.get("status") in ("observed", "estimated")
        and item.get("quality", {}).get("usable", True)
    ]
    for source_name in ("brouhaha_vad", "firered_vad"):
        for item in usable:
            if item.get("source", {}).get("name") == source_name:
                return item
    return usable[0] if usable else None


def _merge_profile_genders(profile_bundle, gender_bundle):
    """Project model results onto profiles without exposing scores publicly."""

    profiles = profile_bundle.get("profiles")
    if not isinstance(profiles, list):
        return
    predictions = {}
    for item in gender_bundle.get("speakers", []) or []:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        gender = item.get("gender")
        if speaker_id and gender in ("male", "female"):
            predictions[str(speaker_id)] = gender
    for profile in profiles:
        if isinstance(profile, dict):
            profile["gender"] = predictions.get(str(profile.get("speaker_id")))
    details = profile_bundle.setdefault("details", {})
    details["gender_schema_version"] = gender_bundle.get("schema_version")
    details["gender_status"] = gender_bundle.get("status")
    details["gender_model"] = gender_bundle.get("selected_model")


def _profile_text_parent_ids(decision_timeline, evidence):
    if decision_timeline is not None:
        summary = decision_timeline.get("payload", {}).get(
            "timeline_summary", {}
        )
        if not isinstance(summary, dict):
            return []
        if any(
            str(item.get("text", "")).strip()
            for item in summary.get("segments", [])
            if isinstance(item, dict)
        ):
            return []
    for item in evidence:
        if "joint_speaker_text" not in item.get("capabilities", []):
            continue
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        summary = payload.get("timeline_summary", {})
        if not isinstance(summary, dict):
            continue
        if any(
            str(segment.get("text", "")).strip()
            for segment in summary.get("segments", [])
            if isinstance(segment, dict)
        ):
            return [item["evidence_id"]]
    return [
        item["evidence_id"]
        for item in evidence
        if "lexical_timeline" in item.get("capabilities", [])
    ]


def score_against_native_after_inference(record, timelines, duration_sec):
    native = record.get("sample", {}).get("native_metadata", {})
    utterances = native.get("utterances", [])
    reference_segments = []
    for item in utterances:
        if not isinstance(item, dict):
            continue
        reference_segments.append(
            {
                "start_sec": item.get("start"),
                "end_sec": item.get("end"),
                "speaker_id": item.get("speaker"),
                "text": item.get("text", ""),
            }
        )
    if not reference_segments:
        return {
            "available": False,
            "entered_resolver": False,
            "reason": "native speaker reference unavailable",
        }
    reference = summarize_timeline(reference_segments, duration_sec)
    estimates = []
    for item in timelines:
        summary = item["payload"]["timeline_summary"]
        estimates.append(
            {
                "evidence_id": item["evidence_id"],
                "source": item["source"]["name"],
                "observed_speaker_count": summary["observed_speaker_count"],
                "speaker_count_error": (
                    summary["observed_speaker_count"]
                    - reference["observed_speaker_count"]
                ),
                "multi_speaker_correct": (
                    (summary["observed_speaker_count"] >= 2)
                    == (reference["observed_speaker_count"] >= 2)
                ),
                "overlap_bool_correct": (
                    summary["overlap_observed"]
                    == reference["overlap_observed"]
                ),
                "change_candidate_bool_correct": (
                    summary["change_observed"]
                    == reference["change_observed"]
                ),
            }
        )
    return {
        "available": True,
        "role": "G_oracle_native_evaluation_only",
        "entered_resolver": False,
        "evaluated_after_fusion_id": True,
        "reference": {
            "observed_speaker_count": reference["observed_speaker_count"],
            "multi_speaker": reference["observed_speaker_count"] >= 2,
            "overlap_observed": reference["overlap_observed"],
            "change_candidate_observed": reference["change_observed"],
        },
        "estimates": estimates,
    }


def resolve_sample_audio_path(audio_path, manifest_dir):
    path = Path(str(audio_path)).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (Path(manifest_dir) / path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise IOError("sample audio does not exist: %s" % resolved)
    return resolved


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_asset(model_root, relative_path, expected_sha256):
    asset_path = (Path(str(model_root)).expanduser() / relative_path).resolve()
    if not asset_path.is_file():
        raise IOError("pinned model asset does not exist: %s" % asset_path)
    stat = asset_path.stat()
    cache_key = (
        str(asset_path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    with _MODEL_HASH_CACHE_LOCK:
        actual = _MODEL_HASH_CACHE.get(cache_key)
        if actual is None:
            actual = sha256_file(asset_path)
            _MODEL_HASH_CACHE[cache_key] = actual
    if actual != expected_sha256:
        raise ValueError(
            "model asset SHA256 mismatch for %s: expected %s, got %s"
            % (asset_path, expected_sha256, actual)
        )
    return actual


def _write_jsonl_atomic(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("w", encoding="utf-8") as sink:
        for item in records:
            sink.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            sink.write("\n")
    temporary.replace(path)


def _append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        sink.write("\n")
        sink.flush()
