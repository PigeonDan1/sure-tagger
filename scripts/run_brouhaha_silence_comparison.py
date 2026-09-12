#!/usr/bin/env python3
"""Compare FireRed VAD and Brouhaha VAD silence segments.

This diagnostic script is not the public tags-only pipeline. The public
basic_acoustic.silence_segments field remains FireRed-only.
"""

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.input_schema import InputSchemaError, validate_input_record  # noqa: E402
from tagger.pipelines.tagging import resolve_audio_path  # noqa: E402
from tagger.tools.acoustic_io import get_audio_info  # noqa: E402
from tagger.tools.audio_quality.brouhaha_signal_estimator import BrouhahaConfig  # noqa: E402
from tagger.tools.basic_acoustic.brouhaha_vad_silence_detector import (  # noqa: E402
    TOOL_NAME as BROUHAHA_VAD_TOOL_NAME,
    run as run_brouhaha_vad_silence_detector,
)
from tagger.tools.basic_acoustic.firered_vad_silence_detector import (  # noqa: E402
    TOOL_NAME as FIRERED_VAD_TOOL_NAME,
    FireRedVadConfig,
    run as run_firered_vad_silence_detector,
)
from tagger.tools.basic_acoustic.nonspeech_ratio_calculator import (  # noqa: E402
    run as run_nonspeech_ratio,
)


def run_manifest(
    manifest_path,
    output_path,
    firered_vad_config=None,
    brouhaha_config=None,
):
    manifest = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    context = {}

    count = 0
    with manifest.open("r", encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as sink:
        for row_index, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                validate_input_record(record)
            except InputSchemaError as exc:
                raise InputSchemaError("line %s: %s" % (row_index, exc))

            result = compare_record(
                record,
                manifest_dir=manifest.parent,
                context=context,
                firered_vad_config=firered_vad_config,
                brouhaha_config=brouhaha_config,
            )
            sink.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1

    return {
        "manifest_path": str(manifest),
        "output_path": str(output),
        "sample_count": count,
    }


def compare_record(
    record,
    manifest_dir,
    context,
    firered_vad_config=None,
    brouhaha_config=None,
):
    sample = record["sample"]
    sample_id = sample["sample_id"]
    dataset_name = record["corpus"]["dataset_name"]
    audio_path = resolve_audio_path(sample, manifest_dir)
    warnings = []
    duration_sec = None
    sample_rate_hz = None
    channels = None

    output = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "audio_path": str(audio_path) if audio_path is not None else None,
        "duration_sec": None,
        "sample_rate_hz": None,
        "channels": None,
        "firered_vad": _empty_vad_result(),
        "brouhaha_vad": _empty_vad_result(),
        "comparison": {
            "nonspeech_ratio_delta_brouhaha_minus_firered": None,
            "silence_segment_count_delta_brouhaha_minus_firered": None,
        },
        "warnings": warnings,
    }

    if audio_path is None:
        warnings.append({"type": "missing_audio_path"})
        return output
    if not audio_path.exists():
        warnings.append({"type": "missing_audio_file", "audio_path": str(audio_path)})
        return output

    try:
        info = get_audio_info(audio_path, context=context)
        duration_sec = round(info.duration_sec, 6)
        sample_rate_hz = info.sample_rate_hz
        channels = info.channels
        output["duration_sec"] = duration_sec
        output["sample_rate_hz"] = sample_rate_hz
        output["channels"] = channels
    except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
        warnings.append({"type": "audio_probe_error", "message": str(exc)})
        return output

    output["firered_vad"] = _run_one_vad(
        tool_name=FIRERED_VAD_TOOL_NAME,
        run_fn=run_firered_vad_silence_detector,
        audio_path=audio_path,
        duration_sec=duration_sec,
        context=context,
        config=firered_vad_config,
        warnings=warnings,
    )
    output["brouhaha_vad"] = _run_one_vad(
        tool_name=BROUHAHA_VAD_TOOL_NAME,
        run_fn=run_brouhaha_vad_silence_detector,
        audio_path=audio_path,
        duration_sec=duration_sec,
        context=context,
        config=brouhaha_config,
        warnings=warnings,
    )
    _fill_comparison(output)
    return output


def _run_one_vad(
    tool_name,
    run_fn,
    audio_path,
    duration_sec,
    context,
    config,
    warnings,
):
    result = _empty_vad_result()
    try:
        silence_result = run_fn(
            audio_path,
            duration_sec=duration_sec,
            context=context,
            config=config,
        )
        ratio_result = run_nonspeech_ratio(
            silence_result.value,
            duration_sec=duration_sec,
        )
        result["status"] = "ok"
        result["silence_segments"] = silence_result.value
        result["nonspeech_ratio"] = ratio_result.value
        result["silence_segment_count"] = len(silence_result.value)
    except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
        result["status"] = "failed"
        warnings.append(
            {
                "type": "vad_error",
                "tool_name": tool_name,
                "message": str(exc),
            }
        )
    return result


def _empty_vad_result():
    return {
        "status": "not_run",
        "nonspeech_ratio": None,
        "silence_segments": None,
        "silence_segment_count": None,
    }


def _fill_comparison(output):
    firered = output["firered_vad"]
    brouhaha = output["brouhaha_vad"]
    if firered["status"] != "ok" or brouhaha["status"] != "ok":
        return
    output["comparison"]["nonspeech_ratio_delta_brouhaha_minus_firered"] = round(
        brouhaha["nonspeech_ratio"] - firered["nonspeech_ratio"],
        6,
    )
    output["comparison"][
        "silence_segment_count_delta_brouhaha_minus_firered"
    ] = brouhaha["silence_segment_count"] - firered["silence_segment_count"]


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="phase1_asr_samples/manifest.jsonl",
        help="Input JSONL manifest using the closed raw-only schema.",
    )
    parser.add_argument(
        "--output",
        default="phase1_asr_samples/outputs/brouhaha_silence_comparison.jsonl",
        help="Output diagnostic JSONL path.",
    )
    parser.add_argument(
        "--firered-vad-use-gpu",
        action="store_true",
        help="Use GPU in FireRedVAD config. Defaults to CPU.",
    )
    parser.add_argument(
        "--brouhaha-use-gpu",
        action="store_true",
        help="Use GPU in Brouhaha config when CUDA is available. Defaults to CPU.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary = run_manifest(
        args.manifest,
        args.output,
        firered_vad_config=FireRedVadConfig(use_gpu=args.firered_vad_use_gpu),
        brouhaha_config=BrouhahaConfig(use_gpu=args.brouhaha_use_gpu),
    )
    public_summary = {
        "output_path": summary["output_path"],
        "sample_count": summary["sample_count"],
    }
    print(json.dumps(public_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
