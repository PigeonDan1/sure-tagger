"""Local runtime paths.

Fill FIRERED_VAD_MODEL_DIR with the local FireRedVAD non-streaming VAD model
directory before running silence tagging.

Fill FIRERED_AED_MODEL_DIR with the local FireRedVAD non-streaming AED model
directory before running music tagging.

Fill FIRERED_LID_* with the FireRedASR2S LID repo, pretrained model dir, and
dedicated Python environment before running spoken-language tagging.

Fill PANNS_* with the pinned AudioSet tagging repository, Cnn14 checkpoint, and
dedicated Python environment before running background-sound tagging.

Fill DASS_* with the local DASS AudioSet checkpoint and its Python environment
before running background-noise-type tagging. The checkpoint was deployed by
sure-harness and copied into models/DASS. The project-local runtime is created
by scripts/setup_dass_gpu.sh and includes the CUDA-enabled PyTorch build.

Fill BROUHAHA_MODEL_PATH with the local Brouhaha checkpoint before running
SNR/C50 tagging. The default location is inside this project.

Fill RECRIR_* with the local Rec-RIR repository, config, and checkpoint before
running sound-field RIR/RT60/C50 tagging. The default location is inside this
project.

Fill DNSMOS_* with the local Microsoft DNSMOS ONNX models before running
no-reference speech quality tagging. The defaults match the sparse checkout in
models/DNS-Challenge.

Fill MOSS_DIARIZE_* with the local OpenMOSS MOSS-Transcribe-Diarize Python
environment before running speaker diarization. This project calls the local
model directly instead of requiring a hosted API service.

FIRERED_ASR_* configures the Chinese/English single-speaker override.
QWEN3_ASR_* configures non-Chinese/English transcription. NEMOTRON_*
configures the independent speaker timeline used for other languages.

TOPIC_* is retained only for the archived topic module and
`tagger/pipelines/tagging_original.py`. The active pipeline neither registers
topic nor exposes topic CLI options.

Model tools can run in subprocesses so incompatible model dependencies do not
need to share one Python environment.
"""


FIRERED_VAD_MODEL_DIR = "models/FireRedVAD/pretrained_models/FireRedVAD/VAD"
FIRERED_VAD_PYTHON = ".runtime/fireredvad_rebuild_py310/bin/python"
FIRERED_AED_MODEL_DIR = "models/FireRedVAD/pretrained_models/FireRedVAD/AED"
FIRERED_AED_PYTHON = ".runtime/fireredvad_rebuild_py310/bin/python"

FIRERED_LID_REPO_DIR = "models/FireRedASR2S/examples_infer/lid"
FIRERED_LID_MODEL_DIR = "models/FireRedASR2S/pretrained_models/FireRedLID"
FIRERED_LID_PYTHON = ".runtime/fireredlid_runtime_py311/bin/python"

PANNS_REPO_DIR = "models/audioset_tagging_cnn"
PANNS_CHECKPOINT_PATH = (
    "models/audioset_tagging_cnn/checkpoints/Cnn14_mAP=0.431.pth"
)
PANNS_MODEL_VERSION = (
    "github:qiuqiangkong/audioset_tagging_cnn"
    "@d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
)
PANNS_PYTHON = ".runtime/panns_py310/bin/python"

DASS_MODEL_DIR = "models/DASS/saurabhati__DASS_medium_AudioSet_48.9"
DASS_MODEL_VERSION = (
    "huggingface:saurabhati/DASS_medium_AudioSet_48.9"
    "@250cdd3ba4c43d790c64d7e7a48f9c5ac881a5d2"
)
# Keep this project-relative: a user's sure-harness home path is not portable
# across hosts (and can silently resolve to a different account's directory).
DASS_PYTHON = ".runtime/dass_runtime_py311/bin/python"

BROUHAHA_REPO_DIR = "models/brouhaha/brouhaha-vad"
BROUHAHA_MODEL_PATH = "models/brouhaha/brouhaha-vad/models/best/checkpoints/best.ckpt"
BROUHAHA_MODEL_VERSION = "github:marianne-m/brouhaha-vad@9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891"
BROUHAHA_PYTHON = ".runtime/fireredvad_rebuild_py310/bin/python"

RECRIR_REPO_DIR = "models/Rec-RIR"
RECRIR_CONFIG_PATH = "models/Rec-RIR/config/Rec-RIR.toml"
RECRIR_CHECKPOINT_PATH = "models/Rec-RIR/ckpt/epoch35.tar"
RECRIR_MODEL_VERSION = "github:Audio-WestlakeU/Rec-RIR@27d03a98bc9a5504a76f377147f36dc7ad169ac6"
RECRIR_PYTHON = ".runtime/recrir_py310_torch271/bin/python"

DNSMOS_PRIMARY_MODEL_PATH = "models/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
DNSMOS_P808_MODEL_PATH = "models/DNS-Challenge/DNSMOS/DNSMOS/model_v8.onnx"
DNSMOS_PERSONALIZED_MODEL_PATH = (
    "models/DNS-Challenge/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_MODEL_VERSION = "github:microsoft/DNS-Challenge@591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
DNSMOS_PYTHON = ".runtime/recrir_py310_torch271/bin/python"

MOSS_DIARIZE_ENDPOINT = ""
MOSS_DIARIZE_MODEL = "models/MOSS-Transcribe-Diarize-model"
MOSS_DIARIZE_TIMEOUT_SEC = 900
MOSS_DIARIZE_MAX_NEW_TOKENS = 65536
MOSS_DIARIZE_API_KEY = ""
MOSS_DIARIZE_PYTHON = ".runtime/moss_transcribe_diarize_py312/bin/python"
MOSS_DIARIZE_DEVICE = "cuda:1"
MOSS_DIARIZE_TORCH_DTYPE = "float16"
MOSS_DIARIZE_PROMPT = ""

FIRERED_ASR_MODEL_DIR = "models/FireRedASR2-AED"
FIRERED_ASR_SOURCE_DIR = "models/FireRedASR2S"
FIRERED_ASR_PYTHON = (
    ".runtime/fireredasr2_aed_py311_torch280_cu128_v1/bin/python"
)
FIRERED_ASR_TIMEOUT_SEC = 1800

QWEN3_ASR_MODEL_DIR = "models/Qwen3-ASR-0.6B"
QWEN3_FORCED_ALIGNER_MODEL_DIR = "models/Qwen3-ForcedAligner-0.6B"
QWEN3_ASR_PYTHON = (
    ".runtime/qwen3_asr_0_6b_py311_torch280_cu128_v1/bin/python"
)
QWEN3_ASR_TIMEOUT_SEC = 1800

NEMOTRON_MODEL_DIR = "models/speaker/nvidia/Nemotron-3-Diarization"
NEMOTRON_PYTHON = (
    ".runtime/nemotron3_diarization_py311_transformers519_torch280_v1/bin/python"
)
NEMOTRON_TIMEOUT_SEC = 1800

SPEAKER_CHANNEL_WINDOW_SEC = 0.05
SPEAKER_CHANNEL_ENERGY_THRESHOLD = 200.0
SPEAKER_CHANNEL_LEAKAGE_RELATIVE_DB = -18.0
SPEAKER_MIN_SEGMENT_DURATION_SEC = 0.10
SPEAKER_MERGE_SAME_SPEAKER_GAP_SEC = 0.30

# Legacy topic settings: not read by the active tagging pipeline.
TOPIC_ENABLE = False
TOPIC_PROVIDER = "openai_responses"
TOPIC_MODEL = ""
TOPIC_BASE_URL = ""
TOPIC_API_KEY = ""
TOPIC_API_KEY_PATH = "api.txt"
TOPIC_MODEL_PROVIDER = ""
TOPIC_CODEX_CONFIG_PATH = ""
TOPIC_TIMEOUT_SEC = 180
TOPIC_TEMPERATURE = None
TOPIC_USE_JSON_SCHEMA = False
TOPIC_CACHE_ENABLED = True
TOPIC_CACHE_PATH = "outputs/cache/topic_openai_responses_cache.jsonl"
TOPIC_FALLBACK = "null"
TOPIC_SHORT_GUARD_ENABLED = True
TOPIC_SHORT_GUARD_MAX_TOKENS = 3
