"""JSONL worker entry point for subprocess-backed model tools."""

import contextlib
import json
import sys
import traceback


_CLIENTS = {}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        sys.stderr.write("usage: subprocess_worker <tool_name>\n")
        return 2
    tool_name = argv[0]

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("_command") == "close":
            return 0
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = dispatch(tool_name, request)
            response = {"status": "ok", "result": _jsonable(result)}
        except Exception as exc:  # noqa: BLE001 - returned to parent as tool failure.
            traceback.print_exc(file=sys.stderr)
            response = {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def dispatch(tool_name, request):
    if tool_name == "firered_vad_detect":
        return _run_firered_vad_detect(request)
    if tool_name == "firered_aed_detect":
        return _run_firered_aed_detect(request)
    if tool_name == "firered_lid_detect":
        return _run_firered_lid_detect(request)
    if tool_name == "panns_background_estimate":
        return _run_panns_background_estimate(request)
    if tool_name == "dass_noise_type_estimate":
        return _run_dass_noise_type_estimate(request)
    if tool_name == "brouhaha_estimate":
        return _run_brouhaha_estimate(request)
    if tool_name == "dnsmos_estimate":
        return _run_dnsmos_estimate(request)
    if tool_name == "recrir_estimate":
        return _run_recrir_estimate(request)
    if tool_name == "moss_diarize_estimate":
        return _run_moss_diarize_estimate(request)
    if tool_name == "firered_asr_estimate":
        return _run_firered_asr_estimate(request)
    if tool_name == "qwen3_asr_estimate":
        return _run_qwen3_asr_estimate(request)
    if tool_name == "diarizen_official_estimate":
        return _run_diarizen_official_estimate(request)
    if tool_name == "nemotron3_diarization_estimate":
        return _run_nemotron3_diarization_estimate(request)
    if tool_name == "campplus_identity_estimate":
        return _run_campplus_identity_estimate(request)
    if tool_name == "ecapa_identity_estimate":
        return _run_ecapa_identity_estimate(request)
    if tool_name == "whisper_lexical_estimate":
        return _run_whisper_lexical_estimate(request)
    if tool_name == "sortformer_timeline_estimate":
        return _run_sortformer_timeline_estimate(request)
    if tool_name == "pyannote_community1_estimate":
        return _run_pyannote_community1_estimate(request)
    if tool_name == "speaker_gender_estimate":
        return _run_speaker_gender_estimate(request)
    raise ValueError("unknown subprocess tool: %s" % tool_name)


def _run_firered_vad_detect(request):
    from tagger.tools.basic_acoustic.firered_vad_silence_detector import (
        FireRedVadClient,
        FireRedVadConfig,
    )

    config = FireRedVadConfig(**request["config"])
    client = _cached_client("firered_vad_detect", request["config"], FireRedVadClient, config)
    return {
        "speech_segments": client.detect_speech_segments(
            request["audio_path"],
            context=None,
        )
    }


def _run_firered_aed_detect(request):
    from tagger.tools.sound_field_scene.firered_aed_detector import (
        FireRedAedClient,
        FireRedAedConfig,
    )

    config = FireRedAedConfig(**request["config"])
    client = _cached_client(
        "firered_aed_detect",
        request["config"],
        FireRedAedClient,
        config,
    )
    return {
        "output": client.detect_audio_events(
            request["audio_path"],
            context=None,
        )
    }


def _run_firered_lid_detect(request):
    from tagger.tools.language_content.firered_lid_detector import (
        FireRedLidClient,
        FireRedLidConfig,
    )

    config = FireRedLidConfig(**request["config"])
    client = _cached_client(
        "firered_lid_detect",
        request["config"],
        FireRedLidClient,
        config,
    )
    return {
        "output": client.detect_language(
            request["audio_path"],
            context=None,
        )
    }


def _run_panns_background_estimate(request):
    from tagger.tools.sound_field_scene.panns_background_detector import (
        PannsBackgroundClient,
        PannsBackgroundConfig,
    )

    config = PannsBackgroundConfig(**request["config"])
    client = _cached_client(
        "panns_background_estimate",
        request["config"],
        PannsBackgroundClient,
        config,
    )
    return {"output": client.estimate(request["audio_path"], context=None)}


def _run_dass_noise_type_estimate(request):
    from tagger.tools.sound_field_scene.dass_noise_type_detector import (
        DassNoiseTypeClient,
        DassNoiseTypeConfig,
    )

    config = DassNoiseTypeConfig(**request["config"])
    client = _cached_client(
        "dass_noise_type_estimate",
        request["config"],
        DassNoiseTypeClient,
        config,
    )
    return {"output": client.estimate(request["audio_path"], context=None)}


def _run_brouhaha_estimate(request):
    from tagger.tools.audio_quality.brouhaha_signal_estimator import (
        BrouhahaClient,
        BrouhahaConfig,
    )

    config = BrouhahaConfig(**request["config"])
    client = _cached_client("brouhaha_estimate", request["config"], BrouhahaClient, config)
    return {"output": client.estimate(request["audio_path"], context=None)}


def _run_recrir_estimate(request):
    from tagger.tools.room_acoustic.rir_estimator import RecRirClient, RecRirConfig

    config = RecRirConfig(**request["config"])
    client = _cached_client("recrir_estimate", request["config"], RecRirClient, config)
    return {"output": client.estimate_rir(request["audio_path"], context=None)}


def _run_dnsmos_estimate(request):
    from tagger.tools.audio_quality.dnsmos_quality_estimator import (
        DnsmosClient,
        DnsmosConfig,
    )

    config = DnsmosConfig(**request["config"])
    client = _cached_client("dnsmos_estimate", request["config"], DnsmosClient, config)
    return {"output": client.estimate(request["audio_path"], context=None)}


def _run_moss_diarize_estimate(request):
    from tagger.tools.speaker.moss_diarizer import (
        MossDiarizeClient,
        MossDiarizeConfig,
    )

    config = MossDiarizeConfig(**request["config"])
    client = _cached_client(
        "moss_diarize_estimate",
        request["config"],
        MossDiarizeClient,
        config,
    )
    return {"output": client.diarize(request["audio_path"], context=None)}


def _run_firered_asr_estimate(request):
    """Run one FireRedASR2-AED request using a cached model client."""

    from tagger.tools.speaker_v2.firered_asr import (
        FireRedAsrClient,
        FireRedAsrConfig,
    )

    config_record = dict(request.get("config") or {})
    # The parent config normally carries the subprocess executable only as
    # provenance.  It must not recursively cause another subprocess client in
    # this worker.
    config_record["subprocess_python"] = ""
    config = FireRedAsrConfig(**config_record)
    client = _cached_client(
        "firered_asr_estimate",
        config_record,
        FireRedAsrClient,
        config,
    )
    return {"output": client.transcribe(request["audio_path"], context=None)}


def _run_qwen3_asr_estimate(request):
    """Run one Qwen3-ASR-0.6B request using a cached model client."""

    from tagger.tools.speaker_v2.qwen3_asr import (
        Qwen3AsrClient,
        Qwen3AsrConfig,
    )

    config_record = dict(request.get("config") or {})
    config_record["subprocess_python"] = ""
    config = Qwen3AsrConfig(**config_record)
    client = _cached_client(
        "qwen3_asr_estimate",
        config_record,
        Qwen3AsrClient,
        config,
    )
    return {"output": client.transcribe(request["audio_path"], context=None)}


def _run_diarizen_official_estimate(request):
    """Run official DiariZen using one process-local cached pipeline."""

    from tagger.tools.speaker_v2.diarizen_official import (
        OfficialDiariZenClient,
        OfficialDiariZenConfig,
    )

    config_record = dict(request.get("config") or {})
    config_record["subprocess_python"] = ""
    config = OfficialDiariZenConfig(**config_record)
    client = _cached_client(
        "diarizen_official_estimate",
        config_record,
        OfficialDiariZenClient,
        config,
    )
    return {"output": client.diarize(request["audio_path"], context=None)}


def _run_nemotron3_diarization_estimate(request):
    """Run Nemotron 3 through a process-local cached CUDA model."""

    from tagger.tools.speaker_v2.nemotron3_diarization import (
        Nemotron3DiarizationClient,
        Nemotron3DiarizationConfig,
    )

    config_record = dict(request.get("config") or {})
    config_record["subprocess_python"] = ""
    config = Nemotron3DiarizationConfig(**config_record)
    client = _cached_client(
        "nemotron3_diarization_estimate",
        config_record,
        Nemotron3DiarizationClient,
        config,
    )
    return {"output": client.diarize(request["audio_path"], context=None)}


def _run_campplus_identity_estimate(request):
    from tagger.tools.speaker_v2.campplus_identity import (
        CampPlusIdentityClient,
        CampPlusIdentityConfig,
    )

    config = CampPlusIdentityConfig(**request["config"])
    client = _cached_client(
        "campplus_identity_estimate",
        request["config"],
        CampPlusIdentityClient,
        config,
    )
    return {
        "output": client.compare_regions(
            request["audio_path"],
            request["regions"],
        )
    }


def _run_ecapa_identity_estimate(request):
    from tagger.tools.speaker_v2.ecapa_identity import (
        EcapaIdentityClient,
        EcapaIdentityConfig,
        validate_subprocess_request,
    )

    validate_subprocess_request(request)
    config = EcapaIdentityConfig(**request["config"])
    client = _cached_client(
        "ecapa_identity_estimate",
        request["config"],
        EcapaIdentityClient,
        config,
    )
    return {
        "output": client.compare_regions(
            request["audio_path"],
            request["regions"],
        )
    }


def _run_whisper_lexical_estimate(request):
    from tagger.tools.speaker_v2.whisper_lexical import (
        WhisperLexicalClient,
        WhisperLexicalConfig,
    )

    config = WhisperLexicalConfig(**request["config"])
    client = _cached_client(
        "whisper_lexical_estimate",
        request["config"],
        WhisperLexicalClient,
        config,
    )
    return {"output": client.transcribe(request["audio_path"])}


def _run_sortformer_timeline_estimate(request):
    from tagger.tools.speaker_v2.sortformer_timeline import (
        SortformerTimelineClient,
        SortformerTimelineConfig,
    )

    config = SortformerTimelineConfig(**request["config"])
    client = _cached_client(
        "sortformer_timeline_estimate",
        request["config"],
        SortformerTimelineClient,
        config,
    )
    return {"output": client.diarize(request["audio_path"])}


def _run_pyannote_community1_estimate(request):
    from tagger.tools.speaker_v2.pyannote_community1 import (
        PyannoteCommunity1Client,
        PyannoteCommunity1Config,
    )

    config = PyannoteCommunity1Config(**request["config"])
    client = _cached_client(
        "pyannote_community1_estimate",
        request["config"],
        PyannoteCommunity1Client,
        config,
    )
    return {"output": client.diarize(request["audio_path"])}


def _run_speaker_gender_estimate(request):
    from tagger.tools.speaker_v2.speaker_gender import (
        SpeakerGenderClient,
        SpeakerGenderConfig,
    )

    config = SpeakerGenderConfig(**request["config"])
    client = _cached_client(
        "speaker_gender_estimate",
        request["config"],
        SpeakerGenderClient,
        config,
    )
    return {
        "output": client.predict(
            request["audio_path"],
            request["model_name"],
            intervals=request.get("intervals"),
            context=None,
        )
    }


def _cached_client(tool_name, config_record, client_class, config):
    key = (tool_name, json.dumps(config_record, sort_keys=True))
    if key not in _CLIENTS:
        _CLIENTS[key] = client_class(config)
    return _CLIENTS[key]


def _jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if hasattr(value, "itertracks"):
        return [
            {
                "start_sec": float(segment.start),
                "end_sec": float(segment.end),
                "track": str(track),
            }
            for segment, track in value.itertracks()
        ]
    if hasattr(value, "itersegments"):
        return [
            {
                "start_sec": float(segment.start),
                "end_sec": float(segment.end),
            }
            for segment in value.itersegments()
        ]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
