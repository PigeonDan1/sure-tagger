#!/usr/bin/env python3
"""Build deterministic browser data from the report's source artifacts."""

import gzip
import hashlib
import json
import re
import shutil
import wave
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parents[1]
TAGGER_DIR = Path("/hpc_stor03/sjtu_home/weihan.chen/share/tagger")
PIPELINE_DOC = TAGGER_DIR / "tagger/tools/speaker_v2/docs/speaker_v2_label_pipeline_inventory_20260818.md"
SURE_DOC = TAGGER_DIR / "tagger/tools/speaker_v1/model_evaluate/evaluation_matrix.md"
QWEN_DIR = TAGGER_DIR / "tagger/tools/speaker_v2/evalue"
QWEN_OUTPUT = QWEN_DIR / "outputs/quality_shadow_wav_cpu_sortformer_100_20260819"
DEMO_DIR = TAGGER_DIR / "ami_en2001a_utterances/outputs/speaker_v2_quality_demo_20260819_final"
AMI_MANIFEST = TAGGER_DIR / "ami_en2001a_utterances/manifest.jsonl"

PUBLIC_FIELDS = (
    "speaker_count",
    "multi_speaker",
    "speaker_change_count",
    "speaker_change",
    "overlap_ratio",
    "speaker_overlap",
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_cell(value: str):
    value = value.strip().replace("**", "").replace("`", "")
    return value.replace("<br>", " / ")


def markdown_table_after(text: str, heading_prefix: str, table_index: int = 0):
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(heading_prefix))
    tables = []
    index = start + 1
    while index < len(lines):
        if lines[index].startswith("## "):
            break
        if lines[index].lstrip().startswith("|"):
            table = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append([clean_cell(cell) for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            if len(table) >= 3:
                tables.append({"headers": table[0], "rows": table[2:]})
        else:
            index += 1
    return tables[table_index]


def parse_number(value):
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def table_records(table):
    return [dict(zip(table["headers"], row)) for row in table["rows"]]


def normalized_source(path):
    return str(path)


def build_pipeline_data():
    text = PIPELINE_DOC.read_text(encoding="utf-8")
    route_table = markdown_table_after(text, "## 2.")
    return {
        "source": normalized_source(PIPELINE_DOC),
        "sourceSha256": sha256(PIPELINE_DOC),
        "updated": "2026-08-18",
        "profile": "quality-shadow",
        "publicFields": [
            {"key": "speaker_count", "type": "integer >= 0 | null", "claim": "C", "meaning": "C timeline 中累计活动 >=0.10s 的 speaker ID 数"},
            {"key": "multi_speaker", "type": "boolean | null", "claim": "M", "meaning": "精确派生：speaker_count >= 2"},
            {"key": "speaker_change_count", "type": "integer >= 0 | null", "claim": "X 派生", "meaning": "X decision timeline 的 change point 数"},
            {"key": "speaker_change", "type": "boolean | null", "claim": "X", "meaning": "是否至少存在一个 change point"},
            {"key": "overlap_ratio", "type": "number [0,1] | null", "claim": "O 派生", "meaning": "overlap duration / speech union duration"},
            {"key": "speaker_overlap", "type": "boolean | null", "claim": "O", "meaning": "O timeline 是否存在合格 overlap event"},
        ],
        "inputContract": {
            "consumed": ["sample_id", "audio.path / 音频字节", "音频时长与基础媒体信息"],
            "excluded": ["native metadata", "人工/模型标签", "输入 transcript（不进入 resolver）"],
            "publicOutput": list(PUBLIC_FIELDS),
            "internalOnly": ["timeline segments", "evidence_id", "route / guard / fallback", "模型版本与运行信息", "置信与认证状态"],
        },
        "stages": [
            {"index": "01", "name": "Audio intake", "detail": "读取音频并建立 scope；不读取 native 标注。"},
            {"index": "02", "name": "Specialist evidence", "detail": "MOSS、Sortformer、Pyannote 生成独立 timeline evidence。"},
            {"index": "03", "name": "Claim routing", "detail": "C/M/O/X 按 per-claim primary、fallback 与 guard 路由。"},
            {"index": "04", "name": "Derived metrics", "detail": "从实际 decision timeline 派生 change count 与 overlap ratio。"},
            {"index": "05", "name": "Certification boundary", "detail": "shadow candidate 仅进入 evaluation_output；公开 adapter 保持 null。"},
        ],
        "timelineRules": [
            {"value": "< 0.10s", "label": "短 segment 丢弃"},
            {"value": "<= 0.30s", "label": "同 speaker continuity gap"},
            {"value": "<= 1.00s", "label": "相邻异 speaker 记 change"},
            {"value": ">= 0.10s", "label": "overlap event 最短时长"},
        ],
        "claimRoutes": [
            {"claim": "C", "fields": ["speaker_count"], "primary": "Sortformer", "fallback": "MOSS", "guard": "—", "excluded": "Pyannote", "headline": "73.5% exact / MAE 0.292"},
            {"claim": "M", "fields": ["multi_speaker"], "primary": "Sortformer", "fallback": "—", "guard": "MOSS · negative FP guard", "excluded": "Pyannote", "headline": "96.5% accuracy"},
            {"claim": "O", "fields": ["speaker_overlap", "overlap_ratio"], "primary": "Pyannote", "fallback": "Sortformer", "guard": "Sortformer + MOSS", "excluded": "—", "headline": "83.8% bool / event F1 52.18%"},
            {"claim": "X", "fields": ["speaker_change", "speaker_change_count"], "primary": "MOSS", "fallback": "Sortformer", "guard": "Sortformer · recall witness", "excluded": "Pyannote", "headline": "94.9% bool / F1@0.5s 62.47%"},
        ],
        "inventoryRows": table_records(route_table),
        "internalCapabilities": [
            {"key": "I", "name": "speaker identity", "state": "internal evidence", "model": "ECAPA / CAM++"},
            {"key": "V", "name": "speech coverage", "state": "internal evidence", "model": "Brouhaha / FireRedVAD"},
            {"key": "A", "name": "lexical timeline", "state": "internal audit", "model": "MOSS / Whisper"},
            {"key": "D", "name": "full diarization", "state": "not connected", "model": "无正式 claim/output"},
        ],
        "warnings": [
            "guard 当前只记录 observation，guards_affect_candidate=false。",
            "表中 SURE 分数是冻结数据集上的全局估计，不是单条样本概率。",
            "SURE overlap bool 使用 ratio>=0.05；runtime O claim 使用是否存在>=0.10s overlap event，83.8% 只能作为定义有差异的代理指标。",
            "当前 quality-shadow 即使开启 certification gate，六个 evaluation_output 字段仍全部为 null，不能产出 production-certified 标签。",
            "CLI 默认仍是 legacy-shadow；新版链路必须显式指定 --profile quality-shadow。",
        ],
    }


def build_sure_data():
    text = SURE_DOC.read_text(encoding="utf-8")
    asr = table_records(markdown_table_after(text, "## ASR"))
    cmo = table_records(markdown_table_after(text, "## C/M/O"))
    change = table_records(markdown_table_after(text, "## X"))
    der = table_records(markdown_table_after(text, "## D"))
    vad = table_records(markdown_table_after(text, "## V", 0))
    vad_boundary = table_records(markdown_table_after(text, "## V", 1))
    vad_combined = []
    for row in vad:
        boundary = next(item for item in vad_boundary if item["模型"] == row["模型"])
        combined = dict(row)
        combined.update({key: value for key, value in boundary.items() if key != "模型"})
        vad_combined.append(combined)
    identity = table_records(markdown_table_after(text, "## I"))
    status = table_records(markdown_table_after(text, "## 状态"))

    model_chart = []
    for row in cmo:
        name = row["模型"]
        change_row = next(item for item in change if item["模型"] == name)
        der_row = next(item for item in der if item["模型"] == name)
        model_chart.append(
            {
                "name": name,
                "shortName": "MOSS" if name.startswith("MOSS") else "Sortformer" if name.startswith("NVIDIA") else "Pyannote",
                "countAccuracy": parse_number(row["count_accuracy↑"]),
                "multiAccuracy": parse_number(row["multi_accuracy↑"]),
                "overlapAccuracy": parse_number(row["overlap_accuracy↑"]),
                "changeAccuracy": parse_number(change_row["change_bool_accuracy↑"]),
                "der": parse_number(der_row["DER↓"]),
            }
        )

    return {
        "source": normalized_source(SURE_DOC),
        "sourceSha256": sha256(SURE_DOC),
        "updated": "2026-08-18",
        "dataset": {
            "utterances": 1000,
            "meetings": 167,
            "durationSeconds": 27905.0095,
            "durationHours": 7.7514,
            "manifestSha256": "1edb22b5d524d4a551501e8e2aa8a768cc67a0aa66a2a31a38bbf00fa402bc94",
            "speakerCountDistribution": {"1": 131, "2": 266, "3": 316, "4": 287},
            "identityTrials": 23815,
        },
        "tables": {"asr": asr, "cmo": cmo, "change": change, "der": der, "vad": vad_combined, "vadBoundary": vad_boundary, "identity": identity, "status": status},
        "modelChart": model_chart,
        "takeaways": [
            "Sortformer 的 count exact 73.5% 与 MAE 0.292 最优，并保持最高 multi accuracy 96.5%。",
            "Pyannote 的 overlap bool 83.8%、frame F1 70.87%、event F1 52.18% 最优。",
            "MOSS 的 change count MAE 1.473 最低；Sortformer 的 change bool 95.5% 略高。",
            "Pyannote DER 0.1931 最低，但 confusion 时长最高，说明单一总分不能替代结构指标。",
            "ECAPA 在身份验证的 EER 10.79%、ROC-AUC 95.19%，全面优于 CAM++。",
        ],
        "limitations": [
            "DER 每个模型为 999 / 1,000 条可评 session；1 条在 0.25s collar 后无参考时长，被显式标为 insufficient。",
            "VAD 缓存没有连续帧概率，AUC-ROC 为 unsupported。",
            "AMI 分数仍需非 AMI、meeting-separated 数据复核，尤其关注预训练污染风险。",
        ],
    }


def pick_unique(rows, predicate, used):
    for row in rows:
        if row["sample_id"] not in used and predicate(row):
            used.add(row["sample_id"])
            return row
    raise ValueError("no representative Qwen-ground-truth case found")


def copy_qwen_audio(sample_id: str):
    source = QWEN_DIR / "data" / f"{sample_id}.mp3"
    target = REPORT_DIR / "assets/audio/qwen" / source.name
    shutil.copy2(source, target)
    return f"assets/audio/qwen/{source.name}"


def build_qwen_data():
    metrics_path = QWEN_OUTPUT / "evaluation/metrics.json"
    predictions_path = QWEN_OUTPUT / "evaluation/predictions.jsonl"
    run_manifest_path = QWEN_OUTPUT / "run_manifest.json"
    report_path = QWEN_DIR / "多说话人评测报告_20260819.md"
    metrics = read_json(metrics_path)
    rows = read_jsonl(predictions_path)
    run_manifest = read_json(run_manifest_path)
    if len(rows) != metrics["dataset"]["sample_count"]:
        raise ValueError("Qwen reference prediction count does not match metrics")

    used = set()
    cases = [
        ("多人误报", "multi_fp", pick_unique(rows, lambda row: row["gt_multi_speaker"] is False and row["pred_multi_speaker"] is True, used)),
        ("多人漏报", "multi_fn", pick_unique(rows, lambda row: row["gt_multi_speaker"] is True and row["pred_multi_speaker"] is False, used)),
        ("重叠误报", "overlap_fp", pick_unique(rows, lambda row: row["gt_speaker_overlap"] is False and row["pred_speaker_overlap"] is True, used)),
        ("高人数低估", "count_under", pick_unique(rows, lambda row: isinstance(row["gt_speaker_count"], int) and row["gt_speaker_count"] - row["pred_speaker_count"] >= 2, used)),
    ]
    case_payload = []
    for title, kind, row in cases:
        gt_path = QWEN_DIR / "data" / f"{row['sample_id']}.json"
        gt_record = read_json(gt_path)
        case_payload.append(
            {
                "title": title,
                "kind": kind,
                "sampleId": row["sample_id"],
                "audio": copy_qwen_audio(row["sample_id"]),
                "comparison": row,
                "groundTruthRecord": gt_record,
                "source": normalized_source(gt_path),
            }
        )

    return {
        "source": normalized_source(QWEN_DIR),
        "reportSource": normalized_source(report_path),
        "metricsSource": normalized_source(metrics_path),
        "metricsSha256": sha256(metrics_path),
        "predictionsSha256": sha256(predictions_path),
        "groundTruth": {
            "label": "Qwen-captioner / Qwen-derived reference labels",
            "provenance": "100 个评测 JSON 与上游 tag_extracted 逐字节一致；结构化抽取脚本默认模型为 Qwen3-8B。原始音频描述 captioner 的具体版本未在现有 artifact 中固化。",
            "extraction": "audio caption → Qwen3-8B 严格事实抽取 → company speaker schema",
            "leakageBoundary": "manifest 不含 tag、native metadata 或 transcript；JSON 标签只在推理完成后进入 scorer。",
            "caveat": "这是模型生成的 reference label，不等同于人工 gold；结果衡量与该 reference 的一致性。",
        },
        "run": {
            "profile": run_manifest["run_profile"],
            "processed": run_manifest["result"]["processed_sample_count"],
            "success": run_manifest["result"]["success_count"],
            "failure": run_manifest["result"]["failure_count"],
            "sortformerDevice": run_manifest["models"]["sortformer"]["config"]["device"],
            "mossDevice": run_manifest["models"]["moss"]["config"]["device"],
            "pyannoteDevice": run_manifest["models"]["pyannote"]["config"]["device"],
            "productionEligible": run_manifest["evaluation_output"]["production_eligible"],
        },
        "metrics": metrics,
        "predictions": rows,
        "cases": case_payload,
        "fieldCards": [
            {"field": "multi_speaker", "eligible": 97, "coverage": 1.0, "primary": "Accuracy", "value": 0.9278350515463918, "secondary": "F1 91.36%"},
            {"field": "speaker_count", "eligible": 97, "coverage": 1.0, "primary": "Exact", "value": 0.7422680412371134, "secondary": "MAE 0.340"},
            {"field": "speaker_change", "eligible": 95, "coverage": 1.0, "primary": "Accuracy", "value": 0.8210526315789474, "secondary": "F1 80.46%"},
            {"field": "speaker_change_count", "eligible": 85, "coverage": 1.0, "primary": "Exact", "value": 0.5882352941176471, "secondary": "MAE 1.071"},
            {"field": "speaker_overlap", "eligible": 91, "coverage": 1.0, "primary": "Accuracy", "value": 0.6703296703296703, "secondary": "Precision 21.62%"},
            {"field": "overlap_ratio", "eligible": 72, "coverage": 0.9722222222222222, "primary": "MAE", "value": 0.026392928571428574, "secondary": "RMSE 0.0742", "lowerIsBetter": True},
        ],
        "keyRisk": "Overlap 正例仅 9 条：TP / TN / FP / FN = 8 / 53 / 29 / 1。Recall 88.89%，但 precision 21.62%，误报是最突出的风险。",
    }


def native_reference(manifest_record):
    utterances = manifest_record["sample"]["native_metadata"]["utterances"]
    speakers = sorted({item["speaker"] for item in utterances})
    has_overlap = any(
        left["speaker"] != right["speaker"]
        and max(left["start"], right["start"]) < min(left["end"], right["end"])
        for index, left in enumerate(utterances)
        for right in utterances[index + 1 :]
    )
    return {
        "speaker_count": len(speakers),
        "multi_speaker": len(speakers) >= 2,
        "speaker_change_count": None,
        "speaker_change": len(speakers) >= 2,
        "overlap_ratio": None,
        "speaker_overlap": has_overlap,
    }


def wav_metadata(path: Path):
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        rate = source.getframerate()
        return {
            "channels": source.getnchannels(),
            "sampleRateHz": rate,
            "sampleWidthBytes": source.getsampwidth(),
            "frames": frames,
            "durationSeconds": frames / rate,
        }


def resolve_artifact_path(value: str):
    path = Path(value)
    if path.exists():
        return path
    marker = "/share/tagger/"
    if marker in value:
        candidate = TAGGER_DIR / value.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def timeline_evidence(sample_dir: Path):
    selected = {}
    wanted = {
        "nvidia_streaming_sortformer_4spk_v2": "count_multi",
        "moss_transcribe_diarize": "change",
        "pyannote_community_1": "overlap",
    }
    for path in sorted((sample_dir / "evidence").glob("*.json.gz")):
        value = read_gzip_json(path)
        name = value.get("source", {}).get("name")
        if name not in wanted:
            continue
        summary = value.get("payload", {}).get("timeline_summary", {})
        selected[wanted[name]] = {
            "source": name,
            "evidenceId": value.get("evidence_id"),
            "status": value.get("status"),
            "quality": value.get("quality"),
            "segments": summary.get("segments", []),
            "speakerIds": summary.get("speaker_ids", []),
            "changePoints": summary.get("change_candidate_points_sec", []),
            "overlapSegments": summary.get("overlap_segments", []),
        }
    return selected


def copy_demo_audio(sample_id: str):
    source = TAGGER_DIR / "ami_en2001a_utterances/audio" / f"{sample_id}.wav"
    target = REPORT_DIR / "assets/audio/demo" / source.name
    shutil.copy2(source, target)
    return f"assets/audio/demo/{source.name}", wav_metadata(source)


def build_demo_data():
    results_path = DEMO_DIR / "speaker_v2_shadow_results.jsonl"
    run_manifest_path = DEMO_DIR / "run_manifest.json"
    summary_path = DEMO_DIR / "DEMO_SUMMARY.md"
    results = read_jsonl(results_path)
    results_by_id = {row["sample_id"]: row for row in results}
    selected_ids = sorted(results_by_id)
    manifest_rows = {
        row["sample"]["sample_id"]: row
        for row in read_jsonl(AMI_MANIFEST)
        if row["sample"]["sample_id"] in results_by_id
    }
    if set(manifest_rows) != set(results_by_id):
        raise ValueError("demo manifest records are incomplete")

    samples = []
    for sample_id in selected_ids:
        result = results_by_id[sample_id]
        record = manifest_rows[sample_id]
        fusion_path = resolve_artifact_path(result["fusion_artifact"])
        fusion = read_gzip_json(fusion_path)
        public_output = fusion["public_adapter"]["speaker"]
        if set(public_output) != set(PUBLIC_FIELDS) or any(value is not None for value in public_output.values()):
            raise ValueError(f"unexpected public adapter output for {sample_id}")
        audio, media = copy_demo_audio(sample_id)
        native = native_reference(record)
        candidate = result["evaluation_output"]["speaker"]
        sample_dir = fusion_path.parent
        samples.append(
            {
                "sampleId": sample_id,
                "title": f"{native['speaker_count']} 人 · {'含重叠' if native['speaker_overlap'] else '无重叠'}",
                "audio": audio,
                "media": media,
                "rawInput": {"sample_id": sample_id, "audio": {"path": record["sample"]["audio"]["path"]}},
                "excludedInput": {"transcript": record["sample"]["text"]["transcript"], "native_metadata": record["sample"]["native_metadata"]},
                "nativeReference": native,
                "candidateOutput": candidate,
                "publicOutput": public_output,
                "countCorrect": candidate["speaker_count"] == native["speaker_count"],
                "routeEvidence": {claim: value.get("route", {}) for claim, value in fusion["claims"].items()},
                "modelTimelines": timeline_evidence(sample_dir),
                "nativeTimeline": [
                    {"start_sec": item["start"], "end_sec": item["end"], "speaker_id": item["speaker"], "text": item["text"]}
                    for item in record["sample"]["native_metadata"]["utterances"]
                ],
                "metadata": fusion,
                "metadataSource": normalized_source(fusion_path),
            }
        )

    run_manifest = read_json(run_manifest_path)
    return {
        "source": normalized_source(summary_path),
        "summarySha256": sha256(summary_path),
        "resultsSource": normalized_source(results_path),
        "resultsSha256": sha256(results_path),
        "runManifestSource": normalized_source(run_manifest_path),
        "run": {
            "profile": run_manifest["run_profile"],
            "processed": run_manifest["result"]["processed_sample_count"],
            "success": run_manifest["result"]["success_count"],
            "failure": run_manifest["result"]["failure_count"],
            "durationSeconds": 151.606,
            "audioSeconds": 91.817,
            "rtf": 1.65,
            "productionEligible": run_manifest["evaluation_output"]["production_eligible"],
            "policyVersion": run_manifest["policy_version"],
            "policyHash": run_manifest["policy_hash"],
        },
        "summary": {
            "countExact": 0.6,
            "countMae": 0.6,
            "multiAccuracy": 1.0,
            "overlapAccuracy": 1.0,
            "changeAccuracy": 1.0,
            "evidenceUsable": "30 / 30",
            "artifactsValidated": 88,
        },
        "modelComparison": [
            {"model": "MOSS", "countExact": 0.4, "countMae": 0.8, "multi": 1.0, "change": 1.0, "overlap": 0.6},
            {"model": "Sortformer", "countExact": 0.6, "countMae": 0.6, "multi": 1.0, "change": 1.0, "overlap": 1.0},
            {"model": "Pyannote", "countExact": 0.4, "countMae": 1.0, "multi": 1.0, "change": 1.0, "overlap": 1.0},
        ],
        "samples": samples,
        "warnings": [
            "5 条是按 1–5 人定向选择的功能 demo，不能外推总体精度。",
            "4 人和 5 人样本均输出为 3 人；高说话人数 exact count 是最明确短板。",
            "evaluation_output 为 candidate、unsafe_for_publication；公开 adapter 仍返回六个 null。",
            "Pyannote license review 仍为 pending；Brouhaha runtime 兼容性需在正式使用前固定。",
        ],
    }


def main():
    data = {
        "meta": {
            "title": "Speaker Intelligence · 技术进展汇报",
            "reportDate": "2026-08-19",
            "generatedBy": "scripts/build_report_data.py",
            "publicFields": list(PUBLIC_FIELDS),
        },
        "pipeline": build_pipeline_data(),
        "sure": build_sure_data(),
        "qwen": build_qwen_data(),
        "demo": build_demo_data(),
    }
    output = REPORT_DIR / "data/report-data.js"
    serialized = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
    output.write_text(f"window.REPORT_DATA = {serialized};\n", encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
