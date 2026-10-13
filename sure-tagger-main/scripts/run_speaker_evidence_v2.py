#!/usr/bin/env python3
"""Run the utterance-level speaker evidence pipeline in v2-shadow mode."""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.pipelines.speaker_evidence import (  # noqa: E402
    SpeakerEvidenceConfig,
    run_manifest,
)
from tagger.local_config import (  # noqa: E402
    BROUHAHA_MODEL_PATH,
    BROUHAHA_MODEL_VERSION,
    BROUHAHA_REPO_DIR,
    FIRERED_LID_MODEL_DIR,
    FIRERED_LID_PYTHON,
    FIRERED_LID_REPO_DIR,
    FIRERED_ASR_MODEL_DIR,
    FIRERED_ASR_PYTHON,
    FIRERED_ASR_SOURCE_DIR,
    FIRERED_ASR_TIMEOUT_SEC,
    QWEN3_ASR_MODEL_DIR,
    QWEN3_FORCED_ALIGNER_MODEL_DIR,
    QWEN3_ASR_PYTHON,
    QWEN3_ASR_TIMEOUT_SEC,
    NEMOTRON_MODEL_DIR,
    NEMOTRON_PYTHON,
    NEMOTRON_TIMEOUT_SEC,
)
from tagger.tools.basic_acoustic.brouhaha_signal_estimator import (  # noqa: E402
    BrouhahaConfig,
)
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (  # noqa: E402
    FireRedVadConfig,
)
from tagger.tools.speaker.moss_diarizer import MossDiarizeConfig  # noqa: E402
from tagger.tools.language_content.firered_lid_detector import (  # noqa: E402
    FireRedLidConfig,
)
from tagger.tools.speaker_v2.nemotron3_diarization import (  # noqa: E402
    Nemotron3DiarizationConfig,
)
from tagger.tools.speaker_v2.firered_asr import FireRedAsrConfig  # noqa: E402
from tagger.tools.speaker_v2.qwen3_asr import Qwen3AsrConfig  # noqa: E402
from tagger.tools.speaker_v2.speaker_gender import (  # noqa: E402
    SpeakerGenderConfig,
)
from tagger.tools.speaker_v2.campplus_identity import (  # noqa: E402
    CampPlusIdentityConfig,
)
from tagger.tools.speaker_v2.ecapa_identity import (  # noqa: E402
    EcapaIdentityConfig,
)
from tagger.tools.speaker_v2.profiles import (  # noqa: E402
    available_profiles,
    expand_profile,
)
from tagger.tools.speaker_v2.whisper_lexical import (  # noqa: E402
    CHECKPOINT_SHA256 as WHISPER_CHECKPOINT_SHA256,
    WhisperLexicalConfig,
)
from tagger.tools.speaker_v2.pyannote_community1 import (  # noqa: E402
    PyannoteCommunity1Config,
)
from tagger.tools.speaker_v2.sortformer_timeline import (  # noqa: E402
    SortformerTimelineConfig,
)


DEFAULT_MOSS_PYTHON = ROOT / ".runtime" / (
    "moss_transcribe_diarize_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_MOSS_MODEL = ROOT / "models" / "MOSS-Transcribe-Diarize-model"
DEFAULT_FIRERED_LID_PYTHON = ROOT / FIRERED_LID_PYTHON
DEFAULT_FIRERED_LID_MODEL = ROOT / FIRERED_LID_MODEL_DIR
DEFAULT_FIRERED_LID_REPO = ROOT / FIRERED_LID_REPO_DIR
DEFAULT_FIRERED_ASR_MODEL = ROOT / FIRERED_ASR_MODEL_DIR
DEFAULT_FIRERED_ASR_SOURCE = ROOT / FIRERED_ASR_SOURCE_DIR
DEFAULT_FIRERED_ASR_PYTHON = ROOT / FIRERED_ASR_PYTHON
DEFAULT_QWEN3_ASR_MODEL = ROOT / QWEN3_ASR_MODEL_DIR
DEFAULT_QWEN3_FORCED_ALIGNER_MODEL = ROOT / QWEN3_FORCED_ALIGNER_MODEL_DIR
DEFAULT_QWEN3_ASR_PYTHON = ROOT / QWEN3_ASR_PYTHON
DEFAULT_NEMOTRON_MODEL = ROOT / NEMOTRON_MODEL_DIR
DEFAULT_NEMOTRON_PYTHON = ROOT / NEMOTRON_PYTHON
DEFAULT_VAD_PYTHON = ROOT / ".runtime" / "fireredvad_rebuild_py310/bin/python"
DEFAULT_VAD_MODEL = ROOT / "models" / "FireRedVAD" / (
    "pretrained_models/FireRedVAD/VAD"
)
DEFAULT_CAMPPLUS_PYTHON = ROOT / ".runtime" / (
    "campplus_sv_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_CAMPPLUS_MODEL = ROOT / (
    "models/speech_campplus_sv_zh-cn_16k-common-v1.0.0"
)
DEFAULT_WHISPER_PYTHON = ROOT / ".runtime" / (
    "whisper_base_multilingual_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_WHISPER_MODEL = ROOT / "models" / "speaker" / "openai" / (
    "whisper-base"
) / WHISPER_CHECKPOINT_SHA256 / "base.pt"
DEFAULT_SORTFORMER_PYTHON = ROOT / ".runtime" / (
    "sortformer_nemo253_py311_torch260_cu124_v1/bin/python"
)
DEFAULT_SORTFORMER_MODEL = ROOT / "models" / "speaker" / "nvidia" / (
    "diar_streaming_sortformer_4spk-v2"
) / "diar_streaming_sortformer_4spk-v2.nemo"
DEFAULT_PYANNOTE_PYTHON = ROOT / ".runtime" / (
    "speaker_pyannote4_py311_torch280_cu128_v1/bin/python"
)
DEFAULT_PYANNOTE_MODEL = ROOT / "models" / "speaker" / (
    "pyannote_community_1"
)
DEFAULT_ECAPA_PYTHON = ROOT / ".runtime" / "fireredvad_rebuild_py310/bin/python"
DEFAULT_ECAPA_MODEL = ROOT / "models" / "speaker" / "speechbrain_ecapa_voxceleb"
DEFAULT_BROUHAHA_PYTHON = ROOT / ".runtime" / (
    "fireredvad_rebuild_py310/bin/python"
)
DEFAULT_BROUHAHA_REPO = ROOT / BROUHAHA_REPO_DIR
DEFAULT_BROUHAHA_MODEL = ROOT / BROUHAHA_MODEL_PATH
DEFAULT_GENDER_ALICE_MODEL = ROOT / (
    "models/speaker/alice_voice_gender_classifier/onnx-q8-v2/"
    "model_quantized.onnx"
)
DEFAULT_GENDER_AFRISPEECH_MODEL_DIR = ROOT / (
    "models/speaker/afrispeech_gender_id/2026-09-11"
)
DEFAULT_GENDER_GRIKO_MODEL_DIR = ROOT / (
    "models/speaker/griko_gender_cls_ecapa_voxceleb/2025-07-31"
)
DEFAULT_GENDER_ONNX_PYTHON = ROOT / (
    ".runtime/diarizen_official_py310/bin/python"
)
DEFAULT_GENDER_GRIKO_PYTHON = ROOT / (
    ".runtime/fireredvad_rebuild_py310/bin/python"
)


def _add_model_toggle(parser, option, destination):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--%s-enable" % option,
        dest=destination,
        action="store_true",
        help="Explicitly enable this model after profile expansion.",
    )
    group.add_argument(
        "--%s-disable" % option,
        dest=destination,
        action="store_false",
        help="Explicitly disable this model after profile expansion.",
    )
    parser.set_defaults(**{destination: None})


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile",
        choices=available_profiles(),
        default="legacy-shadow",
        help="Versioned speaker-v2 model and claim-routing profile.",
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel sample tasks; model subprocesses are shared.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful samples already present in the result JSONL.",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=1,
        help="常驻模型子进程数；每增加一路会增加一份模型显存。",
    )
    for option in (
        "moss",
        "openasr",
        "firered-asr",
        "qwen3-asr",
        "nemotron",
        "whisper",
        "sortformer",
        "pyannote",
        "firered-vad",
        "campplus",
        "ecapa",
        "brouhaha",
        "gender",
    ):
        parser.add_argument(
            "--%s-workers" % option,
            type=int,
            default=None,
            help="Override subprocess pool size for this model.",
        )
    parser.add_argument(
        "--score-native",
        action="store_true",
        help=(
            "Score only after inference. Native annotations never enter the "
            "evidence resolver."
        ),
    )
    parser.add_argument(
        "--skip-model-verification",
        action="store_true",
        help="Skip pinned model hash verification.",
    )
    _add_model_toggle(parser, "moss", "moss_enabled")
    parser.add_argument("--moss-python", default=str(DEFAULT_MOSS_PYTHON))
    parser.add_argument("--moss-model", default=str(DEFAULT_MOSS_MODEL))
    parser.add_argument("--moss-device", default="cuda:0")
    parser.add_argument("--moss-torch-dtype", default="float16")
    parser.add_argument("--moss-max-new-tokens", type=int, default=2048)
    parser.add_argument("--moss-prompt", default="")
    _add_model_toggle(parser, "openasr", "openasr_enabled")
    _add_model_toggle(parser, "firered-asr", "firered_asr_enabled")
    _add_model_toggle(parser, "qwen3-asr", "qwen3_asr_enabled")
    _add_model_toggle(parser, "nemotron", "nemotron_enabled")
    parser.add_argument("--firered-asr-model", default=str(DEFAULT_FIRERED_ASR_MODEL))
    parser.add_argument("--firered-asr-source", default=str(DEFAULT_FIRERED_ASR_SOURCE))
    parser.add_argument("--firered-asr-python", default=str(DEFAULT_FIRERED_ASR_PYTHON))
    parser.add_argument("--firered-asr-device", default="cuda:0")
    parser.add_argument("--firered-asr-half", action="store_true")
    parser.add_argument(
        "--firered-asr-timeout-sec", type=int, default=FIRERED_ASR_TIMEOUT_SEC
    )
    parser.add_argument("--qwen3-asr-model", default=str(DEFAULT_QWEN3_ASR_MODEL))
    parser.add_argument(
        "--qwen3-forced-aligner-model",
        default=str(DEFAULT_QWEN3_FORCED_ALIGNER_MODEL),
        help="Qwen3-ForcedAligner model directory used for native timestamps.",
    )
    parser.add_argument("--qwen3-asr-python", default=str(DEFAULT_QWEN3_ASR_PYTHON))
    parser.add_argument("--qwen3-asr-device", default="cuda:0")
    parser.add_argument(
        "--qwen3-asr-torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--qwen3-asr-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--qwen3-asr-timeout-sec", type=int, default=QWEN3_ASR_TIMEOUT_SEC
    )
    parser.add_argument("--nemotron-model-dir", default=str(DEFAULT_NEMOTRON_MODEL))
    parser.add_argument("--nemotron-python", default=str(DEFAULT_NEMOTRON_PYTHON))
    parser.add_argument("--nemotron-device", default="cuda:0")
    parser.add_argument(
        "--nemotron-timeout-sec", type=int, default=NEMOTRON_TIMEOUT_SEC
    )
    parser.add_argument(
        "--firered-lid-python", default=str(DEFAULT_FIRERED_LID_PYTHON)
    )
    parser.add_argument(
        "--firered-lid-model", default=str(DEFAULT_FIRERED_LID_MODEL)
    )
    parser.add_argument(
        "--firered-lid-repo", default=str(DEFAULT_FIRERED_LID_REPO)
    )
    parser.add_argument("--firered-lid-use-gpu", action="store_true")
    parser.add_argument("--firered-lid-half", action="store_true")
    _add_model_toggle(parser, "vad", "vad_enabled")
    parser.add_argument("--vad-python", default=str(DEFAULT_VAD_PYTHON))
    parser.add_argument("--vad-model", default=str(DEFAULT_VAD_MODEL))
    _add_model_toggle(parser, "campplus", "campplus_enabled")
    parser.add_argument("--campplus-python", default=str(DEFAULT_CAMPPLUS_PYTHON))
    parser.add_argument("--campplus-model", default=str(DEFAULT_CAMPPLUS_MODEL))
    parser.add_argument("--campplus-device", default="cpu")
    parser.add_argument("--campplus-threshold", type=float, default=0.5)
    parser.add_argument(
        "--campplus-min-region-duration-sec", type=float, default=0.8
    )
    _add_model_toggle(parser, "whisper", "whisper_enabled")
    parser.add_argument("--whisper-python", default=str(DEFAULT_WHISPER_PYTHON))
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER_MODEL))
    parser.add_argument("--whisper-device", default="cuda:0")
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--whisper-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--whisper-no-word-timestamps",
        action="store_true",
        help="Use coarse Whisper decode segments instead of word intervals.",
    )
    _add_model_toggle(parser, "sortformer", "sortformer_enabled")
    parser.add_argument(
        "--sortformer-python", default=str(DEFAULT_SORTFORMER_PYTHON)
    )
    parser.add_argument(
        "--sortformer-model", default=str(DEFAULT_SORTFORMER_MODEL)
    )
    parser.add_argument("--sortformer-device", default="cuda:0")
    parser.add_argument("--sortformer-timeout-sec", type=int, default=600)
    _add_model_toggle(parser, "pyannote", "pyannote_enabled")
    parser.add_argument(
        "--pyannote-python", default=str(DEFAULT_PYANNOTE_PYTHON)
    )
    parser.add_argument("--pyannote-model", default=str(DEFAULT_PYANNOTE_MODEL))
    parser.add_argument("--pyannote-device", default="cuda:0")
    parser.add_argument("--pyannote-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--pyannote-min-activity-sec", type=float, default=0.10
    )
    parser.add_argument("--pyannote-calibration-profile-id", default=None)
    parser.add_argument("--pyannote-joint-negative-profile-id", default=None)
    parser.add_argument(
        "--pyannote-license-review-status",
        choices=("pending", "approved"),
        default="pending",
    )
    _add_model_toggle(parser, "ecapa", "ecapa_enabled")
    parser.add_argument("--ecapa-python", default=str(DEFAULT_ECAPA_PYTHON))
    parser.add_argument("--ecapa-model", default=str(DEFAULT_ECAPA_MODEL))
    parser.add_argument("--ecapa-device", default="cpu")
    parser.add_argument("--ecapa-threshold", type=float, default=None)
    parser.add_argument(
        "--ecapa-min-region-duration-sec", type=float, default=0.8
    )
    parser.add_argument("--ecapa-max-regions-per-speaker", type=int, default=2)
    parser.add_argument("--ecapa-torch-num-threads", type=int, default=4)
    parser.add_argument("--ecapa-timeout-sec", type=int, default=300)
    parser.add_argument("--ecapa-runtime-cache-dir", default="")
    parser.add_argument("--ecapa-calibration-profile-id", default=None)
    _add_model_toggle(parser, "brouhaha", "brouhaha_enabled")
    _add_model_toggle(parser, "gender", "gender_enabled")
    parser.add_argument(
        "--gender-alice-model", default=str(DEFAULT_GENDER_ALICE_MODEL)
    )
    parser.add_argument(
        "--gender-afrispeech-model-dir",
        default=str(DEFAULT_GENDER_AFRISPEECH_MODEL_DIR),
    )
    parser.add_argument(
        "--gender-griko-model-dir",
        default=str(DEFAULT_GENDER_GRIKO_MODEL_DIR),
    )
    parser.add_argument(
        "--gender-alice-python", default=str(DEFAULT_GENDER_ONNX_PYTHON)
    )
    parser.add_argument(
        "--gender-afrispeech-python", default=str(DEFAULT_GENDER_ONNX_PYTHON)
    )
    parser.add_argument(
        "--gender-griko-python", default=str(DEFAULT_GENDER_GRIKO_PYTHON)
    )
    parser.add_argument("--gender-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--speaker-profile-disable",
        dest="speaker_profile_enabled",
        action="store_false",
        default=True,
        help="Disable deterministic phase 0/1 speaker profiles.",
    )
    parser.add_argument("--brouhaha-python", default=str(DEFAULT_BROUHAHA_PYTHON))
    parser.add_argument("--brouhaha-model", default=str(DEFAULT_BROUHAHA_MODEL))
    parser.add_argument("--brouhaha-repo", default=str(DEFAULT_BROUHAHA_REPO))
    parser.add_argument("--brouhaha-model-version", default=BROUHAHA_MODEL_VERSION)
    parser.add_argument("--brouhaha-use-gpu", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    firered_asr_enabled = (
        args.openasr_enabled
        if args.firered_asr_enabled is None
        else args.firered_asr_enabled
    )
    qwen3_asr_enabled = (
        args.openasr_enabled
        if args.qwen3_asr_enabled is None
        else args.qwen3_asr_enabled
    )
    nemotron_enabled = (
        args.openasr_enabled
        if args.nemotron_enabled is None
        else args.nemotron_enabled
    )
    expanded_run_profile = expand_profile(args.profile)
    expanded_run_profile["profile_model_defaults"] = dict(
        expanded_run_profile["models"]
    )
    model_overrides = {
        "moss": args.moss_enabled,
        "firered_asr": firered_asr_enabled,
        "qwen3_asr": qwen3_asr_enabled,
        "nemotron": nemotron_enabled,
        "vad": args.vad_enabled,
        "campplus": args.campplus_enabled,
        "whisper": args.whisper_enabled,
        "sortformer": args.sortformer_enabled,
        "pyannote": args.pyannote_enabled,
        "ecapa": args.ecapa_enabled,
        "brouhaha": args.brouhaha_enabled,
        "gender": args.gender_enabled,
    }
    expanded_models = dict(expanded_run_profile["models"])
    applied_overrides = {}
    for model_name, override in model_overrides.items():
        if override is None:
            continue
        expanded_models[model_name] = bool(override)
        applied_overrides[model_name] = bool(override)
    expanded_run_profile["models"] = expanded_models
    expanded_run_profile["model_overrides"] = applied_overrides
    moss_config = MossDiarizeConfig(
        model=args.moss_model,
        subprocess_python=args.moss_python,
        device=args.moss_device,
        torch_dtype=args.moss_torch_dtype,
        max_new_tokens=args.moss_max_new_tokens,
        prompt=args.moss_prompt,
    )
    firered_asr_config = FireRedAsrConfig(
        model_dir=args.firered_asr_model,
        source_dir=args.firered_asr_source,
        subprocess_python=args.firered_asr_python,
        device=args.firered_asr_device,
        use_half=args.firered_asr_half,
        timeout_sec=args.firered_asr_timeout_sec,
        enable_lid=False,
    )
    qwen3_asr_config = Qwen3AsrConfig(
        model_dir=args.qwen3_asr_model,
        forced_aligner_model_dir=args.qwen3_forced_aligner_model,
        subprocess_python=args.qwen3_asr_python,
        device=args.qwen3_asr_device,
        torch_dtype=args.qwen3_asr_torch_dtype,
        max_new_tokens=args.qwen3_asr_max_new_tokens,
        timeout_sec=args.qwen3_asr_timeout_sec,
        return_time_stamps=True,
    )
    nemotron_config = Nemotron3DiarizationConfig(
        model_dir=args.nemotron_model_dir,
        subprocess_python=args.nemotron_python,
        device=args.nemotron_device,
        timeout_sec=args.nemotron_timeout_sec,
    )
    firered_lid_config = FireRedLidConfig(
        model_dir=args.firered_lid_model,
        repo_dir=args.firered_lid_repo,
        subprocess_python=args.firered_lid_python,
        use_gpu=args.firered_lid_use_gpu,
        use_half=args.firered_lid_half,
    )
    vad_config = FireRedVadConfig(
        model_dir=args.vad_model,
        subprocess_python=args.vad_python,
        use_gpu=False,
    )
    campplus_config = CampPlusIdentityConfig(
        model_dir=args.campplus_model,
        subprocess_python=args.campplus_python,
        device=args.campplus_device,
        threshold=args.campplus_threshold,
        min_region_duration_sec=args.campplus_min_region_duration_sec,
    )
    whisper_config = WhisperLexicalConfig(
        model_path=args.whisper_model,
        subprocess_python=args.whisper_python,
        device=args.whisper_device,
        language=args.whisper_language,
        word_timestamps=not args.whisper_no_word_timestamps,
        timeout_sec=args.whisper_timeout_sec,
    )
    sortformer_config = SortformerTimelineConfig(
        model_path=args.sortformer_model,
        subprocess_python=args.sortformer_python,
        device=args.sortformer_device,
        timeout_sec=args.sortformer_timeout_sec,
    )
    pyannote_config = PyannoteCommunity1Config(
        model_dir=args.pyannote_model,
        subprocess_python=args.pyannote_python,
        device=args.pyannote_device,
        timeout_sec=args.pyannote_timeout_sec,
        min_activity_sec=args.pyannote_min_activity_sec,
        calibration_profile_id=args.pyannote_calibration_profile_id,
        joint_negative_profile_id=args.pyannote_joint_negative_profile_id,
        license_review_status=args.pyannote_license_review_status,
    )
    ecapa_kwargs = {
        "model_dir": args.ecapa_model,
        "subprocess_python": args.ecapa_python,
        "device": args.ecapa_device,
        "min_region_duration_sec": args.ecapa_min_region_duration_sec,
        "max_regions_per_speaker": args.ecapa_max_regions_per_speaker,
        "torch_num_threads": args.ecapa_torch_num_threads,
        "timeout_sec": args.ecapa_timeout_sec,
        "runtime_cache_dir": args.ecapa_runtime_cache_dir,
    }
    if args.ecapa_threshold is not None:
        ecapa_kwargs["threshold"] = args.ecapa_threshold
    if args.ecapa_calibration_profile_id is not None:
        ecapa_kwargs["calibration_profile_id"] = (
            args.ecapa_calibration_profile_id
        )
    ecapa_config = EcapaIdentityConfig(**ecapa_kwargs)
    brouhaha_config = BrouhahaConfig(
        model_path=args.brouhaha_model,
        repo_dir=args.brouhaha_repo,
        use_gpu=args.brouhaha_use_gpu,
        model_version=args.brouhaha_model_version,
        subprocess_python=args.brouhaha_python,
    )
    gender_config = SpeakerGenderConfig(
        alice_model_path=args.gender_alice_model,
        afrispeech_model_dir=args.gender_afrispeech_model_dir,
        griko_model_dir=args.gender_griko_model_dir,
        ecapa_model_dir=args.ecapa_model,
        alice_python=args.gender_alice_python,
        afrispeech_python=args.gender_afrispeech_python,
        griko_python=args.gender_griko_python,
        timeout_sec=args.gender_timeout_sec,
        verify_model_assets=not args.skip_model_verification,
    )
    config = SpeakerEvidenceConfig(
        moss_config=moss_config,
        firered_asr_config=firered_asr_config,
        qwen3_asr_config=qwen3_asr_config,
        nemotron_config=nemotron_config,
        firered_lid_config=firered_lid_config,
        vad_config=vad_config,
        campplus_config=campplus_config,
        whisper_config=whisper_config,
        sortformer_config=sortformer_config,
        pyannote_config=pyannote_config,
        ecapa_config=ecapa_config,
        brouhaha_config=brouhaha_config,
        gender_config=gender_config,
        enable_moss=args.moss_enabled,
        enable_firered_asr=firered_asr_enabled,
        enable_qwen3_asr=qwen3_asr_enabled,
        enable_nemotron=nemotron_enabled,
        enable_vad=args.vad_enabled,
        enable_campplus=args.campplus_enabled,
        enable_whisper=args.whisper_enabled,
        enable_sortformer=args.sortformer_enabled,
        enable_pyannote=args.pyannote_enabled,
        enable_ecapa=args.ecapa_enabled,
        enable_brouhaha=args.brouhaha_enabled,
        enable_gender=args.gender_enabled,
        enable_speaker_profile=args.speaker_profile_enabled,
        profile_id=args.profile,
        claim_policy=expanded_run_profile["claim_policy"],
        expanded_run_profile=expanded_run_profile,
        score_native=args.score_native,
        verify_model_assets=not args.skip_model_verification,
    )
    summary = run_manifest(
        args.manifest,
        args.output_dir,
        config,
        sample_ids=args.sample_id,
        max_samples=args.max_samples,
        fail_fast=args.fail_fast,
        workers=args.workers,
        resume=args.resume,
        model_workers=args.model_workers,
        model_worker_overrides={
            key: value
            for key, value in {
                "moss_diarize_estimate": args.moss_workers,
                "firered_asr_estimate": (
                    args.firered_asr_workers
                    if args.firered_asr_workers is not None
                    else args.openasr_workers
                ),
                "qwen3_asr_estimate": (
                    args.qwen3_asr_workers
                    if args.qwen3_asr_workers is not None
                    else args.openasr_workers
                ),
                "nemotron3_diarization_estimate": (
                    args.nemotron_workers
                    if args.nemotron_workers is not None
                    else args.openasr_workers
                ),
                "whisper_lexical_estimate": args.whisper_workers,
                "sortformer_timeline_estimate": args.sortformer_workers,
                "pyannote_community1_estimate": args.pyannote_workers,
                "firered_vad_detect": args.firered_vad_workers,
                "campplus_identity_estimate": args.campplus_workers,
                "ecapa_identity_estimate": args.ecapa_workers,
                "brouhaha_estimate": args.brouhaha_workers,
                "speaker_gender_estimate": args.gender_workers,
            }.items()
            if value is not None
        },
    )
    printable = dict(summary)
    printable.pop("results", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
