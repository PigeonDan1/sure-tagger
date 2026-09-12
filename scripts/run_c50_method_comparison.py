#!/usr/bin/env python3
"""Compare C50 estimates from Brouhaha and Rec-RIR.

This diagnostic script is not the public tags-only pipeline. The public
`room_acoustic.c50_db` field remains derived from a Rec-RIR room impulse
response, while Brouhaha C50 is internal evidence (`internal.brouhaha_c50_db`).
"""

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagger.input_schema import InputSchemaError, validate_input_record  # noqa: E402
from tagger.pipelines.tagging import (  # noqa: E402
    resolve_audio_path,
    write_rir_artifact,
)
from tagger.tools.acoustic_io import get_audio_info  # noqa: E402
from tagger.tools.audio_quality.brouhaha_signal_estimator import (  # noqa: E402
    TOOL_NAME as BROUHAHA_TOOL_NAME,
    BrouhahaConfig,
    run as run_brouhaha_signal_estimator,
)
from tagger.tools.room_acoustic.c50_estimator import (  # noqa: E402
    TOOL_NAME as RECRIR_C50_TOOL_NAME,
    run as run_recrir_c50_estimator,
)
from tagger.tools.room_acoustic.rir_estimator import (  # noqa: E402
    TOOL_NAME as RECRIR_RIR_TOOL_NAME,
    METHOD as RECRIR_METHOD,
    RecRirConfig,
    run as run_recrir_rir_estimator,
)
from tagger.tools.subprocess_runner import close_subprocess_workers  # noqa: E402


def run_manifest(
    manifest_path,
    output_path,
    brouhaha_config=None,
    recrir_config=None,
    artifact_dir=None,
):
    manifest = Path(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None
    context = {}

    count = 0
    warning_count = 0
    brouhaha_estimated_count = 0
    recrir_estimated_count = 0
    paired_count = 0
    try:
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
                    brouhaha_config=brouhaha_config,
                    recrir_config=recrir_config,
                    artifact_dir=artifact_root,
                    record_index=row_index,
                )
                sink.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
                warning_count += len(result["warnings"])
                if result["brouhaha"]["status"] == "ok":
                    brouhaha_estimated_count += 1
                if result["recrir"]["status"] == "ok":
                    recrir_estimated_count += 1
                if result["comparison"]["paired"]:
                    paired_count += 1
    finally:
        close_subprocess_workers(context)

    return {
        "manifest_path": str(manifest),
        "output_path": str(output),
        "artifact_dir": str(artifact_root) if artifact_root is not None else None,
        "sample_count": count,
        "warning_count": warning_count,
        "brouhaha_estimated_count": brouhaha_estimated_count,
        "recrir_estimated_count": recrir_estimated_count,
        "paired_count": paired_count,
    }


def compare_record(
    record,
    manifest_dir,
    context,
    brouhaha_config=None,
    recrir_config=None,
    brouhaha_client=None,
    recrir_client=None,
    artifact_dir=None,
    record_index=None,
):
    sample = record["sample"]
    sample_id = sample["sample_id"]
    dataset_name = record["corpus"]["dataset_name"]
    audio_path = resolve_audio_path(sample, manifest_dir)
    warnings = []

    output = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "audio_path": str(audio_path) if audio_path is not None else None,
        "duration_sec": None,
        "sample_rate_hz": None,
        "channels": None,
        "brouhaha": _empty_c50_result(
            tag_path="internal.brouhaha_c50_db",
            tool_name=BROUHAHA_TOOL_NAME,
            method="Brouhaha",
        ),
        "recrir": _empty_c50_result(
            tag_path="room_acoustic.c50_db",
            tool_name=RECRIR_C50_TOOL_NAME,
            method="%s_clarity_c50" % RECRIR_METHOD,
        ),
        "comparison": {
            "paired": False,
            "c50_delta_recrir_minus_brouhaha_db": None,
            "c50_abs_delta_db": None,
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
        output["duration_sec"] = round(info.duration_sec, 6)
        output["sample_rate_hz"] = info.sample_rate_hz
        output["channels"] = info.channels
    except Exception as exc:  # noqa: BLE001 - diagnostic metadata only.
        warnings.append({"type": "audio_probe_error", "message": str(exc)})

    output["brouhaha"] = _run_brouhaha_c50(
        audio_path=audio_path,
        context=context,
        config=brouhaha_config,
        client=brouhaha_client,
        warnings=warnings,
    )
    output["recrir"] = _run_recrir_c50(
        audio_path=audio_path,
        context=context,
        config=recrir_config,
        client=recrir_client,
        warnings=warnings,
        artifact_dir=artifact_dir,
        sample_key=_artifact_sample_key(sample_id, record_index),
    )
    _fill_comparison(output)
    return output


def _run_brouhaha_c50(audio_path, context, config, client, warnings):
    result = _empty_c50_result(
        tag_path="internal.brouhaha_c50_db",
        tool_name=BROUHAHA_TOOL_NAME,
        method="Brouhaha",
    )
    try:
        tool_results = run_brouhaha_signal_estimator(
            audio_path,
            context=context,
            config=config,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
        result["status"] = "failed"
        result["error"] = str(exc)
        warnings.append(
            {
                "type": "brouhaha_c50_error",
                "tool_name": BROUHAHA_TOOL_NAME,
                "message": str(exc),
            }
        )
        return result

    c50_result = _find_result(tool_results, "internal.brouhaha_c50_db")
    if c50_result is None:
        result["status"] = "failed"
        result["error"] = "Brouhaha did not return internal.brouhaha_c50_db"
        warnings.append(
            {
                "type": "brouhaha_c50_missing",
                "tool_name": BROUHAHA_TOOL_NAME,
                "message": result["error"],
            }
        )
        return result

    _copy_tool_result(result, c50_result)
    if result["status"] != "ok":
        warnings.append(
            {
                "type": "brouhaha_c50_invalid",
                "tool_name": BROUHAHA_TOOL_NAME,
                "message": result.get("error", "invalid Brouhaha C50 output"),
            }
        )
    return result


def _run_recrir_c50(
    audio_path,
    context,
    config,
    client,
    warnings,
    artifact_dir,
    sample_key,
):
    result = _empty_c50_result(
        tag_path="room_acoustic.c50_db",
        tool_name=RECRIR_C50_TOOL_NAME,
        method="%s_clarity_c50" % RECRIR_METHOD,
    )
    result["rir_tool_name"] = RECRIR_RIR_TOOL_NAME
    result["rir_status"] = "not_run"
    result["rir_artifact_path"] = None

    try:
        rir_result = run_recrir_rir_estimator(
            audio_path,
            context=context,
            config=config,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
        result["status"] = "failed"
        result["rir_status"] = "failed"
        result["error"] = str(exc)
        warnings.append(
            {
                "type": "recrir_rir_error",
                "tool_name": RECRIR_RIR_TOOL_NAME,
                "message": str(exc),
            }
        )
        return result

    result["rir_status"] = rir_result.status
    result["rir_sample_count"] = rir_result.evidence.get("sample_count")
    if rir_result.status != "estimated" or rir_result.value is None:
        result["status"] = "failed"
        result["error"] = rir_result.evidence.get("error", "invalid Rec-RIR output")
        warnings.append(
            {
                "type": "recrir_rir_invalid",
                "tool_name": RECRIR_RIR_TOOL_NAME,
                "message": result["error"],
            }
        )
        return result

    if artifact_dir is not None:
        try:
            artifact_path = write_rir_artifact(
                rir_result.value,
                Path(artifact_dir) / "rir",
                sample_key,
            )
            result["rir_artifact_path"] = str(artifact_path)
        except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
            warnings.append(
                {
                    "type": "rir_artifact_write_error",
                    "tool_name": RECRIR_RIR_TOOL_NAME,
                    "message": str(exc),
                }
            )

    try:
        c50_result = run_recrir_c50_estimator(rir_result.value, context=context)
    except Exception as exc:  # noqa: BLE001 - diagnostic warning only.
        result["status"] = "failed"
        result["error"] = str(exc)
        warnings.append(
            {
                "type": "recrir_c50_error",
                "tool_name": RECRIR_C50_TOOL_NAME,
                "message": str(exc),
            }
        )
        return result

    _copy_tool_result(result, c50_result)
    if result["status"] != "ok":
        warnings.append(
            {
                "type": "recrir_c50_invalid",
                "tool_name": RECRIR_C50_TOOL_NAME,
                "message": result.get("error", "invalid Rec-RIR C50 output"),
            }
        )
    return result


def _empty_c50_result(tag_path, tool_name, method):
    return {
        "status": "not_run",
        "tool_status": None,
        "tag_path": tag_path,
        "tool_name": tool_name,
        "method": method,
        "c50_db": None,
    }


def _copy_tool_result(output, tool_result):
    output["tool_status"] = tool_result.status
    output["tag_path"] = tool_result.tag_path
    output["tool_name"] = tool_result.tool_name
    output["method"] = tool_result.method
    output["c50_db"] = tool_result.value
    if tool_result.status == "estimated" and _is_finite_number(tool_result.value):
        output["status"] = "ok"
        output.pop("error", None)
        return

    output["status"] = "failed"
    output["error"] = tool_result.evidence.get("error", "invalid C50 output")


def _find_result(results, tag_path):
    for result in results:
        if result.tag_path == tag_path:
            return result
    return None


def _fill_comparison(output):
    brouhaha = output["brouhaha"]
    recrir = output["recrir"]
    if brouhaha["status"] != "ok" or recrir["status"] != "ok":
        return
    delta = float(recrir["c50_db"]) - float(brouhaha["c50_db"])
    output["comparison"]["paired"] = True
    output["comparison"]["c50_delta_recrir_minus_brouhaha_db"] = round(delta, 6)
    output["comparison"]["c50_abs_delta_db"] = round(abs(delta), 6)


def _artifact_sample_key(sample_id, record_index):
    if record_index is None:
        return sample_id
    return "%06d_%s" % (record_index, sample_id)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="phase1_asr_samples/manifest.jsonl",
        help="Input JSONL manifest using the closed raw-only schema.",
    )
    parser.add_argument(
        "--output",
        default="phase1_asr_samples/outputs/c50_method_comparison.jsonl",
        help="Output diagnostic JSONL path.",
    )
    parser.add_argument(
        "--brouhaha-use-gpu",
        action="store_true",
        help="Use GPU in Brouhaha config when CUDA is available. Defaults to CPU.",
    )
    parser.add_argument(
        "--recrir-use-gpu",
        action="store_true",
        help="Use GPU in Rec-RIR config when CUDA is available. Defaults to CPU.",
    )
    parser.add_argument(
        "--brouhaha-python",
        default=None,
        help="Python executable for Brouhaha subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--recrir-python",
        default=None,
        help="Python executable for Rec-RIR subprocess. Defaults to local_config.py.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Optional directory for Rec-RIR waveform artifacts. "
            "Artifacts are written below ARTIFACT_DIR/rir."
        ),
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary = run_manifest(
        args.manifest,
        args.output,
        brouhaha_config=BrouhahaConfig(
            use_gpu=args.brouhaha_use_gpu,
            subprocess_python=args.brouhaha_python,
        ),
        recrir_config=RecRirConfig(
            use_gpu=args.recrir_use_gpu,
            subprocess_python=args.recrir_python,
        ),
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
