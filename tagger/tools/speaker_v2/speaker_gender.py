"""Language-routed speaker gender classification for speaker-v2 profiles."""

from __future__ import division

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from tagger.tools.subprocess_runner import run_subprocess_tool


SCHEMA_VERSION = "speaker_v2.speaker_gender.v1"
TOOL_NAME = "speaker_gender_estimate"
MINIMUM_MULTI_SPEAKER_CLEAN_SEC = 1.0
TARGET_SAMPLE_RATE_HZ = 16000

ALICE_MODEL = "alice"
AFRISPEECH_MODEL = "afrispeech"
GRIKO_MODEL = "griko"

ALICE_SOURCE_VERSION = (
    "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2"
    "@8aedeb1a399945e52e6f3f61714abe07c1c48742"
)
AFRISPEECH_SOURCE_VERSION = (
    "AfriSpeech/afrispeech-gender-id"
    "@e0137ae093a912a809fc38427a8edab2e62fce70"
)
GRIKO_SOURCE_VERSION = (
    "griko/gender_cls_svm_ecapa_voxceleb"
    "@25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64"
)

ALICE_MODEL_SHA256 = (
    "fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252"
)
AFRISPEECH_EMBEDDING_SHA256 = (
    "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"
)
AFRISPEECH_HEAD_SHA256 = (
    "d6b4f20502d01c3bf5b5c968397feeff99358c6b5c73fc8f8fb452eddb28310d"
)
GRIKO_SVM_SHA256 = (
    "74badd2f209f7bb09e511b85da387ca383e2734e3d092474edcff3b3549f2bfa"
)
GRIKO_SCALER_SHA256 = (
    "4e44e58d1e6602f61913b53f65bcc328e400a4c33a97410c543a5f1e2f357651"
)

# FireRed LID's African languages plus ISO aliases covered by the AfriSpeech
# model card. Arabic is intentionally absent: the deployed model explicitly
# classifies it as fallback-only, so it follows the Griko route.
AFRICAN_LANGUAGE_CODES = frozenset(
    (
        "af",
        "am",
        "amh",
        "ach",
        "fat",
        "ff",
        "ful",
        "ha",
        "hau",
        "ibo",
        "ig",
        "ki",
        "kik",
        "lg",
        "lin",
        "ln",
        "lug",
        "luo",
        "mas",
        "mg",
        "nyn",
        "om",
        "orm",
        "pcm",
        "sid",
        "sn",
        "sna",
        "so",
        "sog",
        "sw",
        "swa",
        "ti",
        "tir",
        "tw",
        "twi",
        "wal",
        "yo",
        "yor",
    )
)

MODEL_METADATA = {
    ALICE_MODEL: {
        "source_version": ALICE_SOURCE_VERSION,
        "dependency_group": "G_alice_voice_gender_classifier_onnx_q8_v2",
    },
    AFRISPEECH_MODEL: {
        "source_version": AFRISPEECH_SOURCE_VERSION,
        "dependency_group": "G_afrispeech_gender_id",
    },
    GRIKO_MODEL: {
        "source_version": GRIKO_SOURCE_VERSION,
        "dependency_group": "G_griko_gender_cls_svm_ecapa_voxceleb",
    },
}


class SpeakerGenderError(RuntimeError):
    """Raised when one configured gender model cannot return a prediction."""


class SpeakerGenderConfig:
    def __init__(
        self,
        alice_model_path="models/speaker/alice_voice_gender_classifier/onnx-q8-v2/model_quantized.onnx",
        afrispeech_model_dir="models/speaker/afrispeech_gender_id/2026-09-11",
        griko_model_dir="models/speaker/griko_gender_cls_ecapa_voxceleb/2025-07-31",
        ecapa_model_dir="models/speaker/speechbrain_ecapa_voxceleb",
        alice_python=".runtime/diarizen_official_py310/bin/python",
        afrispeech_python=".runtime/diarizen_official_py310/bin/python",
        griko_python=".runtime/fireredvad_rebuild_py310/bin/python",
        timeout_sec=300,
        minimum_multi_speaker_clean_sec=MINIMUM_MULTI_SPEAKER_CLEAN_SEC,
        verify_model_assets=True,
    ):
        self.alice_model_path = str(alice_model_path)
        self.afrispeech_model_dir = str(afrispeech_model_dir)
        self.griko_model_dir = str(griko_model_dir)
        self.ecapa_model_dir = str(ecapa_model_dir)
        self.alice_python = str(alice_python or "")
        self.afrispeech_python = str(afrispeech_python or "")
        self.griko_python = str(griko_python or "")
        self.timeout_sec = int(timeout_sec)
        self.minimum_multi_speaker_clean_sec = float(
            minimum_multi_speaker_clean_sec
        )
        self.verify_model_assets = bool(verify_model_assets)
        if self.timeout_sec <= 0:
            raise ValueError("speaker gender timeout_sec must be positive")
        if (
            not math.isfinite(self.minimum_multi_speaker_clean_sec)
            or self.minimum_multi_speaker_clean_sec <= 0
        ):
            raise ValueError(
                "speaker gender minimum_multi_speaker_clean_sec must be positive"
            )

    def to_record(self):
        return {
            "alice_model_path": self.alice_model_path,
            "afrispeech_model_dir": self.afrispeech_model_dir,
            "griko_model_dir": self.griko_model_dir,
            "ecapa_model_dir": self.ecapa_model_dir,
            "alice_python": self.alice_python,
            "afrispeech_python": self.afrispeech_python,
            "griko_python": self.griko_python,
            "timeout_sec": self.timeout_sec,
            "minimum_multi_speaker_clean_sec": (
                self.minimum_multi_speaker_clean_sec
            ),
            "verify_model_assets": self.verify_model_assets,
        }

    def subprocess_python_for(self, model_name):
        if model_name == ALICE_MODEL:
            return self.alice_python
        if model_name == AFRISPEECH_MODEL:
            return self.afrispeech_python
        if model_name == GRIKO_MODEL:
            return self.griko_python
        raise ValueError("unknown speaker gender model: %s" % model_name)


def gender_model_for_language(language):
    """Return the configured model family for one FireRed LID code."""

    normalized = " ".join(str(language or "").strip().lower().split())
    base = normalized.replace("_", "-").split("-", 1)[0].split(" ", 1)[0]
    if base in ("zh", "en") or normalized == "english":
        return ALICE_MODEL
    if base in AFRICAN_LANGUAGE_CODES:
        return AFRISPEECH_MODEL
    return GRIKO_MODEL


def infer_speaker_genders(
    decision_timeline,
    audio_path,
    language,
    config=None,
    context=None,
):
    """Classify speakers in first-speech order from one selected timeline."""

    config = config or SpeakerGenderConfig()
    model_name = gender_model_for_language(language)
    metadata = MODEL_METADATA[model_name]
    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "observed",
        "detected_language": language,
        "selected_model": model_name,
        "source_version": metadata["source_version"],
        "dependency_group": metadata["dependency_group"],
        "minimum_multi_speaker_clean_sec": (
            config.minimum_multi_speaker_clean_sec
        ),
        "speaker_order": "first_speech_start_sec",
        "speakers": [],
    }
    summary = _timeline_summary(decision_timeline)
    if summary is None:
        base["status"] = "no_timeline"
        return base
    segments = _activity_segments(summary)
    if not segments:
        base["status"] = "no_speech"
        return base

    ordered_ids = _speaker_ids_in_first_speech_order(segments)
    overlap = _overlap_intervals(summary, segments)
    multi_speaker = len(ordered_ids) > 1
    for index, source_speaker_id in enumerate(ordered_ids, 1):
        speaker_segments = [
            (item["start_sec"], item["end_sec"])
            for item in segments
            if item["speaker_id"] == source_speaker_id
        ]
        clean_intervals = _subtract_intervals(
            _merge_intervals(speaker_segments), overlap
        )
        clean_duration = _duration(clean_intervals)
        detail = {
            "speaker_id": "speaker_%d" % index,
            "source_speaker_id": source_speaker_id,
            "first_speech_start_sec": round(
                min(item[0] for item in speaker_segments), 6
            ),
            "clean_speech_duration_sec": round(clean_duration, 6),
            "input_mode": (
                "clean_non_overlap_concatenation"
                if multi_speaker
                else "full_audio"
            ),
            "model": model_name,
            "gender": None,
            "score": None,
            "reason": None,
        }
        if (
            multi_speaker
            and clean_duration + 1e-9
            < config.minimum_multi_speaker_clean_sec
        ):
            detail["reason"] = "insufficient_clean_speech"
            base["speakers"].append(detail)
            continue
        try:
            result = predict_gender(
                audio_path,
                model_name,
                intervals=clean_intervals if multi_speaker else None,
                config=config,
                context=context,
            )
            gender = result.get("gender")
            if gender not in ("male", "female"):
                raise SpeakerGenderError(
                    "gender model returned an unsupported label"
                )
            score = result.get("score")
            if score is not None:
                score = float(score)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise SpeakerGenderError(
                        "gender model returned an invalid score"
                    )
                score = round(score, 6)
            detail["gender"] = gender
            detail["score"] = score
        except Exception as exc:  # noqa: BLE001 - one speaker must not fail peers.
            detail["reason"] = "%s: %s" % (exc.__class__.__name__, exc)
        base["speakers"].append(detail)
    return base


def predict_gender(
    audio_path,
    model_name,
    intervals=None,
    config=None,
    context=None,
):
    config = config or SpeakerGenderConfig()
    if intervals is not None:
        intervals = _validate_intervals(intervals)
        if not intervals:
            raise SpeakerGenderError("speaker gender input intervals are empty")
    subprocess_python = config.subprocess_python_for(model_name)
    if subprocess_python:
        client = SpeakerGenderSubprocessClient(config, model_name)
    else:
        client = SpeakerGenderClient(config)
    return client.predict(
        audio_path,
        model_name,
        intervals=intervals,
        context=context,
    )


class SpeakerGenderSubprocessClient:
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name

    def predict(self, audio_path, model_name, intervals=None, context=None):
        if model_name != self.model_name:
            raise SpeakerGenderError("subprocess gender model route changed")
        result = run_subprocess_tool(
            self.config.subprocess_python_for(model_name),
            TOOL_NAME,
            {
                "audio_path": str(audio_path),
                "model_name": model_name,
                "intervals": intervals,
                "config": _subprocess_config(self.config),
            },
            context=context,
            timeout_sec=self.config.timeout_sec,
        )
        return result["output"]


class SpeakerGenderClient:
    def __init__(self, config=None):
        self.config = config or SpeakerGenderConfig(
            alice_python="", afrispeech_python="", griko_python=""
        )
        self._alice_session = None
        self._afrispeech_embedding_session = None
        self._afrispeech_session = None
        self._afrispeech_labels = None
        self._griko_encoder = None
        self._griko_scaler = None
        self._griko_svm = None
        self._griko_labels = None
        self._griko_feature_names = None

    def predict(self, audio_path, model_name, intervals=None, context=None):
        del context
        samples = _load_audio(audio_path, intervals=intervals)
        if model_name == ALICE_MODEL:
            return self._predict_alice(samples)
        if model_name == AFRISPEECH_MODEL:
            return self._predict_afrispeech(samples)
        if model_name == GRIKO_MODEL:
            return self._predict_griko(samples)
        raise SpeakerGenderError("unknown speaker gender model: %s" % model_name)

    def _predict_alice(self, samples):
        import numpy as np

        if self._alice_session is None:
            import onnxruntime as ort

            path = _required_file(
                self.config.alice_model_path, "Alice gender model"
            )
            _verify_hash(
                path, ALICE_MODEL_SHA256, self.config.verify_model_assets
            )
            self._alice_session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        input_name = self._alice_session.get_inputs()[0].name
        logits = self._alice_session.run(
            None,
            {input_name: np.asarray(samples, dtype=np.float32).reshape(1, -1)},
        )[0][0]
        return _prediction_from_logits(logits, ("male", "female"))

    def _predict_afrispeech(self, samples):
        import numpy as np

        if self._afrispeech_embedding_session is None:
            import onnxruntime as ort

            root = Path(self.config.afrispeech_model_dir).expanduser().resolve()
            embedding_path = _required_file(
                root
                / "embedding"
                / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
                "AfriSpeech CAM++ embedding model",
            )
            head_path = _required_file(
                root / "model.onnx", "AfriSpeech gender head"
            )
            config_path = _required_file(
                root / "config.json", "AfriSpeech config"
            )
            _verify_hash(
                embedding_path,
                AFRISPEECH_EMBEDDING_SHA256,
                self.config.verify_model_assets,
            )
            _verify_hash(
                head_path,
                AFRISPEECH_HEAD_SHA256,
                self.config.verify_model_assets,
            )
            self._afrispeech_embedding_session = ort.InferenceSession(
                str(embedding_path), providers=["CPUExecutionProvider"]
            )
            self._afrispeech_session = ort.InferenceSession(
                str(head_path), providers=["CPUExecutionProvider"]
            )
            with config_path.open("r", encoding="utf-8") as source:
                label_map = json.load(source)["label_map"]
            self._afrispeech_labels = tuple(
                label_map[str(index)] for index in range(len(label_map))
            )
        features = _afrispeech_fbank(samples)
        embedding_input = self._afrispeech_embedding_session.get_inputs()[0].name
        embedding = self._afrispeech_embedding_session.run(
            None, {embedding_input: features}
        )[0]
        logits = self._afrispeech_session.run(
            ["logits"], {"embedding": embedding}
        )[0][0]
        return _prediction_from_logits(logits, self._afrispeech_labels)

    def _predict_griko(self, samples):
        import numpy as np
        import pandas as pd
        import torch

        self._load_griko()
        signal = torch.from_numpy(
            np.asarray(samples, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            embedding = self._griko_encoder.encode_batch(
                signal, normalize=False
            )
        values = embedding.detach().cpu().reshape(-1).numpy()
        if len(values) != len(self._griko_feature_names):
            raise SpeakerGenderError(
                "Griko ECAPA embedding dimension does not match classifier"
            )
        frame = pd.DataFrame([values], columns=self._griko_feature_names)
        scaled = self._griko_scaler.transform(frame)
        prediction = self._griko_svm.predict(scaled)[0]
        try:
            index = int(prediction)
            gender = self._griko_labels[index]
        except (IndexError, KeyError, TypeError, ValueError):
            gender = str(prediction).strip().lower()
        if gender not in ("male", "female"):
            raise SpeakerGenderError("Griko returned an unsupported label")
        return {"gender": gender, "score": None}

    def _load_griko(self):
        if self._griko_encoder is not None:
            return
        import joblib
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        root = Path(self.config.griko_model_dir).expanduser().resolve()
        svm_path = _required_file(root / "svm_model.joblib", "Griko SVM")
        scaler_path = _required_file(
            root / "scaler.joblib", "Griko feature scaler"
        )
        config_path = _required_file(root / "config.json", "Griko config")
        _verify_hash(svm_path, GRIKO_SVM_SHA256, self.config.verify_model_assets)
        _verify_hash(
            scaler_path, GRIKO_SCALER_SHA256, self.config.verify_model_assets
        )
        ecapa_root = Path(self.config.ecapa_model_dir).expanduser().resolve()
        _required_file(ecapa_root / "hyperparams.yaml", "ECAPA hyperparams")
        torch.set_num_threads(4)
        try:
            self._griko_encoder = EncoderClassifier.from_hparams(
                source=str(ecapa_root),
                savedir=str(ecapa_root / "runtime_cache"),
                overrides={"pretrained_path": str(ecapa_root)},
                run_opts={"device": "cpu"},
            )
            self._griko_encoder.eval()
        except Exception as exc:
            raise SpeakerGenderError("Griko ECAPA model loading failed") from exc
        self._griko_scaler = joblib.load(str(scaler_path))
        self._griko_svm = joblib.load(str(svm_path))
        with config_path.open("r", encoding="utf-8") as source:
            config = json.load(source)
        self._griko_labels = tuple(config["labels"])
        self._griko_feature_names = tuple(config["feature_names"])


def _timeline_summary(evidence):
    if not isinstance(evidence, dict):
        return None
    if "speaker_timeline" not in evidence.get("capabilities", []):
        return None
    if evidence.get("status") not in ("observed", "estimated"):
        return None
    if evidence.get("quality", {}).get("usable", True) is False:
        return None
    summary = evidence.get("payload", {}).get("timeline_summary")
    return summary if isinstance(summary, dict) else None


def _activity_segments(summary):
    raw = summary.get("activity_segments") or summary.get("segments") or []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker_id")
        try:
            start = max(0.0, float(item.get("start_sec")))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if not speaker or not math.isfinite(start) or not math.isfinite(end):
            continue
        if end <= start:
            continue
        result.append(
            {
                "speaker_id": str(speaker),
                "start_sec": start,
                "end_sec": end,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["start_sec"], item["end_sec"], item["speaker_id"]
        ),
    )


def _speaker_ids_in_first_speech_order(segments):
    result = []
    seen = set()
    for item in segments:
        speaker = item["speaker_id"]
        if speaker not in seen:
            result.append(speaker)
            seen.add(speaker)
    return result


def _overlap_intervals(summary, segments):
    supplied = _validate_intervals(
        summary.get("overlap_activity_segments")
        or summary.get("overlap_segments")
        or []
    )
    if supplied:
        return _merge_intervals(supplied)
    overlaps = []
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            if left["speaker_id"] == right["speaker_id"]:
                continue
            start = max(left["start_sec"], right["start_sec"])
            end = min(left["end_sec"], right["end_sec"])
            if end > start:
                overlaps.append((start, end))
    return _merge_intervals(overlaps)


def _validate_intervals(raw):
    result = []
    for item in raw or []:
        if isinstance(item, dict):
            start, end = item.get("start_sec"), item.get("end_sec")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        try:
            start, end = max(0.0, float(start)), float(end)
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            result.append((start, end))
    return result


def _merge_intervals(intervals):
    result = []
    for start, end in sorted(_validate_intervals(intervals)):
        if result and start <= result[-1][1] + 1e-9:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _subtract_intervals(intervals, exclusions):
    result = []
    for start, end in intervals:
        pieces = [(start, end)]
        for cut_start, cut_end in exclusions:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    next_pieces.append((piece_start, cut_start))
                if cut_end < piece_end:
                    next_pieces.append((cut_end, piece_end))
            pieces = next_pieces
        result.extend(pieces)
    return _merge_intervals(result)


def _duration(intervals):
    return sum(max(0.0, end - start) for start, end in intervals)


def _load_audio(audio_path, intervals=None):
    import numpy as np

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=False
        )
    except Exception:
        audio, sample_rate = _decode_with_ffmpeg(audio_path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1 or not len(audio):
        raise SpeakerGenderError("speaker gender source audio is empty")
    if not bool(np.all(np.isfinite(audio))):
        raise SpeakerGenderError(
            "speaker gender source audio contains non-finite samples"
        )
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise SpeakerGenderError("speaker gender sample rate is invalid")
    if sample_rate != TARGET_SAMPLE_RATE_HZ:
        from scipy.signal import resample_poly

        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE_HZ)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE_HZ // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
        sample_rate = TARGET_SAMPLE_RATE_HZ
    if intervals is not None:
        pieces = []
        for start, end in _validate_intervals(intervals):
            start_frame = min(len(audio), max(0, int(round(start * sample_rate))))
            end_frame = min(
                len(audio), max(start_frame, int(round(end * sample_rate)))
            )
            if end_frame > start_frame:
                pieces.append(audio[start_frame:end_frame])
        if not pieces:
            raise SpeakerGenderError("speaker gender clean audio is empty")
        audio = np.concatenate(pieces)
    if not len(audio):
        raise SpeakerGenderError("speaker gender prepared audio is empty")
    return np.asarray(audio, dtype=np.float32)


def _decode_with_ffmpeg(audio_path):
    from array import array

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SpeakerGenderError(
            "audio decode failed and ffmpeg is unavailable"
        )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE_HZ),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        raise SpeakerGenderError("ffmpeg failed to decode source audio")
    values = array("f")
    values.frombytes(completed.stdout)
    return values, TARGET_SAMPLE_RATE_HZ


def _prediction_from_logits(logits, labels):
    import numpy as np

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(values) != len(labels) or not bool(np.all(np.isfinite(values))):
        raise SpeakerGenderError("gender model logits are invalid")
    probabilities = np.exp(values - np.max(values))
    probabilities /= np.sum(probabilities)
    index = int(np.argmax(probabilities))
    return {
        "gender": str(labels[index]),
        "score": float(probabilities[index]),
    }


def _afrispeech_fbank(samples):
    """Match the CAM++ metadata: 80-bin Kaldi fbank plus global mean."""

    import numpy as np
    import torch
    import torchaudio

    signal = torch.from_numpy(
        np.asarray(samples, dtype=np.float32)
    ).unsqueeze(0)
    try:
        features = torchaudio.compliance.kaldi.fbank(
            signal,
            num_mel_bins=80,
            sample_frequency=TARGET_SAMPLE_RATE_HZ,
            dither=0.0,
            energy_floor=0.0,
            snip_edges=False,
        )
    except Exception as exc:
        raise SpeakerGenderError("AfriSpeech fbank extraction failed") from exc
    if features.ndim != 2 or features.shape[0] < 1:
        raise SpeakerGenderError("AfriSpeech fbank is empty")
    features = features - features.mean(dim=0, keepdim=True)
    return features.unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)


def _required_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise SpeakerGenderError("%s is missing: %s" % (label, path))
    return path


def _verify_hash(path, expected, enabled):
    if not enabled:
        return
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SpeakerGenderError(
            "speaker gender model hash mismatch: %s" % path
        )


def _subprocess_config(config):
    record = config.to_record()
    record["alice_python"] = ""
    record["afrispeech_python"] = ""
    record["griko_python"] = ""
    return record
