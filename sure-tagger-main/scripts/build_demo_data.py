"""Build demo/data.js from a manifest and a tags-only output.

The demo site (``demo/``) reads ``window.DEMO_DATA`` from ``data.js``. This
script regenerates that file from the current full-pipeline run so the site
always shows the freshest tags. Usage:

    PYTHONPATH=. python3 scripts/build_demo_data.py \
      --manifest phase2_asr_sample/manifest.jsonl \
      --tags outputs/phase2_full_tags.jsonl \
      --audio-dir demo/assets/audio \
      --rir-artifact-dir outputs/phase2_artifacts/rir \
      --output demo/data.js
"""

import argparse
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# dataset + filename -> human-readable sample title
TITLE_PATTERNS = [
    ("LibriSpeech", r"librispeech_", "LibriSpeech 干净朗读语音"),
    ("AISHELL-1", r"aishell_", "AISHELL-1 中文朗读语音"),
    ("TIMIT", r"timit_", "TIMIT 多音素朗读"),
    ("AMI", r"ami_", "AMI 会议片段"),
    ("WHAM! noise", r"wham_noise_", "WHAM! 非平稳背景噪声"),
]

CHIME_CONDITIONS = {"STR": "街道", "PED": "行人区", "CAF": "咖啡馆", "BUS": "公交"}
NOISEX_TYPES = {"babble": "人群", "f16": "飞机", "factory1": "工厂", "machinegun": "枪械", "volvo": "车辆"}
TUT_SCENES = {"airport": "机场", "bus": "公交", "metro": "地铁", "park": "公园", "street_traffic": "街道交通"}

PUBLIC_GROUPS = (
    "basic_acoustic",
    "audio_quality",
    "room_acoustic",
    "sound_field_scene",
    "language_content",
    "speaker",
)
COVERAGE_FIELDS = {
    "basic_acoustic": "duration_sec",
    "audio_quality": "snr_db",
    "room_acoustic": "rt60_sec",
    "sound_field_scene": "external_noise_type",
    "language_content": "language",
    "speaker": "speaker_count",
}


def dataset_title(dataset_name, audio_name):
    for name, pattern, label in TITLE_PATTERNS:
        if dataset_name == name and re.search(pattern, audio_name):
            return label
    if dataset_name == "CHiME4":
        match = re.search(r"_(STR|PED|CAF|BUS)_REAL", audio_name)
        condition = CHIME_CONDITIONS.get(match.group(1) if match else "", "真实")
        return f"CHiME4 {condition}噪声语音"
    if dataset_name == "TUT Urban Acoustic Scenes 2018":
        match = re.search(r"tut2018_([a-z_]+)-", audio_name)
        scene = TUT_SCENES.get(match.group(1) if match else "", "声景")
        return f"TUT 2018 {scene}声景"
    if dataset_name == "NOISEX-92":
        match = re.search(r"noisex92_([a-z0-9]+)", audio_name)
        noise = NOISEX_TYPES.get(match.group(1) if match else "", "噪声")
        return f"NOISEX-92 {noise}噪声"
    return dataset_name


# docs/DASS.md category keys published in external_noise_type
CATEGORY_LABELS = {
    "music": "音乐",
    "animal": "动物",
    "mechanical": "机械",
    "nature": "自然",
    "formless": "无明确声源",
    "channel_environment": "声道/环境",
}


def sample_note(tags):
    parts = []
    scene = tags.get("sound_field_scene") or {}
    noise = (scene.get("external_noise_type") or [])[:3]
    if noise:
        names = [CATEGORY_LABELS.get(key, key) for key in noise]
        parts.append("噪声: " + ", ".join(names))
    if scene.get("music_present"):
        parts.append("含背景音乐")
    speaker = tags.get("speaker") or {}
    count = speaker.get("speaker_count")
    if count is not None:
        parts.append(f"说话人: {count}")
    composition = scene.get("noise_composition") or {}
    nonempty = [
        CATEGORY_LABELS.get(key, key)
        for key, value in composition.items()
        if value
    ]
    if nonempty:
        parts.append("组成: " + "/".join(nonempty))
    return " · ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "phase2_asr_sample/manifest.jsonl"))
    parser.add_argument("--tags", default=str(PROJECT_ROOT / "outputs/phase2_full_tags.jsonl"))
    parser.add_argument("--audio-dir", default=str(PROJECT_ROOT / "demo/assets/audio"))
    parser.add_argument("--rir-artifact-dir", default=str(PROJECT_ROOT / "outputs/phase2_artifacts/rir"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "demo/data.js"))
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    tag_rows = [json.loads(line) for line in open(args.tags, encoding="utf-8")]
    if len(manifest_rows) != len(tag_rows):
        raise SystemExit(
            f"manifest ({len(manifest_rows)} rows) and tags ({len(tag_rows)} rows) disagree"
        )
    audio_dir = Path(args.audio_dir)
    missing = [
        Path(row["sample"]["audio"]["path"]).name
        for row in manifest_rows
        if not (audio_dir / Path(row["sample"]["audio"]["path"]).name).exists()
    ]
    if missing:
        raise SystemExit(f"audio files missing in {audio_dir}: {missing}")

    samples = []
    datasets = {}
    for index, (manifest, tags) in enumerate(zip(manifest_rows, tag_rows), start=1):
        sample = manifest["sample"]
        dataset = manifest["corpus"]["dataset_name"]
        datasets[dataset] = datasets.get(dataset, 0) + 1
        audio_name = Path(sample["audio"]["path"]).name
        samples.append(
            {
                "row": index,
                "sampleId": sample["sample_id"],
                "dataset": dataset,
                "title": dataset_title(dataset, audio_name),
                "note": sample_note(tags),
                "audio": f"assets/audio/{audio_name}",
                "transcript": sample["text"]["transcript"],
                "nativeMetadata": sample.get("native_metadata") or {},
                "durationSec": (tags.get("basic_acoustic") or {}).get("duration_sec"),
                "tags": tags,
            }
        )

    dataset_order = [row["corpus"]["dataset_name"] for row in manifest_rows]
    dataset_names = sorted(set(dataset_order), key=dataset_order.index)

    def group_covered(group):
        field = COVERAGE_FIELDS[group]
        return sum(1 for tags in tag_rows if (tags.get(group) or {}).get(field) is not None)

    coverage = {group: group_covered(group) for group in PUBLIC_GROUPS}
    speaker_tags = [tags.get("speaker") or {} for tags in tag_rows]
    speaker_count = sum(1 for tags in speaker_tags if tags.get("speaker_count") is not None)
    speaker_multi_count = sum(1 for tags in speaker_tags if tags.get("multi_speaker") is True)

    rir_artifact_count = 0
    rir_dir = Path(args.rir_artifact_dir)
    if rir_dir.is_dir():
        rir_artifact_count = sum(1 for _ in rir_dir.iterdir() if _.is_file())

    summary = {
        "sampleCount": len(samples),
        "datasetCount": len(dataset_names),
        "selectedCount": len(samples),
        "rirArtifactCount": rir_artifact_count,
        "datasets": [{"name": name, "count": datasets[name]} for name in dataset_names],
        "coverage": coverage,
        "speakerCount": speaker_count,
        "speakerMultiCount": speaker_multi_count,
        "generatedFrom": {
            "manifest": str(Path(args.manifest).relative_to(PROJECT_ROOT)),
            "tags": str(Path(args.tags).relative_to(PROJECT_ROOT)),
            "audioDir": str(audio_dir.relative_to(PROJECT_ROOT)),
        },
    }

    payload = {"summary": summary, "samples": samples}
    out = Path(args.output)
    out.write_text(
        "window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out} ({len(samples)} samples, coverage {coverage}, "
        f"speaker {speaker_count}/{speaker_multi_count} multi)"
    )


if __name__ == "__main__":
    main()
