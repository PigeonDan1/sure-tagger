#!/usr/bin/env python3
"""Build the curated phase2 ASR sample manifest and normalized audio files."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tagger.input_schema import validate_input_record  # noqa: E402


class SampleSpec(NamedTuple):
    dataset_name: str
    sample_id: str
    source_path: str
    transcript: str = ""
    max_duration_sec: Optional[float] = None


EMPTY_SOURCE_URLS = {
    "article": [],
    "github": [],
    "huggingface": [],
    "dataset_card": [],
}


def _read_keyed_text(path: Path) -> Dict[str, str]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            records[key] = " ".join(value.split())
    return records


def _read_timit_text(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    return " ".join(parts[2:])


def _read_ami_records(path: Path) -> Dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["sample"]["sample_id"]] = record
    return records


def _build_specs() -> List[SampleSpec]:
    libri_root = REPO_ROOT / "data/rawdata/LibriSpeech/test-clean/1089/134686"
    libri_text = _read_keyed_text(libri_root / "1089-134686.trans.txt")
    libri_ids = [f"1089-134686-{index:04d}" for index in range(5)]

    aishell_text = _read_keyed_text(
        REPO_ROOT
        / "data/rawdata/AIshell/data_aishell/transcript/aishell_transcript_v0.8.txt"
    )
    aishell_ids = [f"BAC009S0764W012{index}" for index in range(1, 6)]

    timit_root = REPO_ROOT / "data/rawdata/TIMIT/timit/test/dr1/faks0"
    timit_ids = ["sa1", "sa2", "si1573", "sx133", "sx223"]

    chime_text = _read_keyed_text(
        REPO_ROOT / "data/am/chime4/data-mfcc/dt05_all_noisy/text"
    )
    chime_entries = [
        ("F01_050C0101_PED_REAL", "dt05_ped_real/F01_050C0101_PED.wav"),
        ("F01_050C0102_CAF_REAL", "dt05_caf_real/F01_050C0102_CAF.wav"),
        ("F01_050C0102_STR_REAL", "dt05_str_real/F01_050C0102_STR.wav"),
        ("F01_050C0103_BUS_REAL", "dt05_bus_real/F01_050C0103_BUS.wav"),
        ("F01_050C0104_CAF_REAL", "dt05_caf_real/F01_050C0104_CAF.wav"),
    ]

    ami_records = _read_ami_records(
        REPO_ROOT / "ami_en2001a_utterances/manifest.jsonl"
    )
    ami_ids = [
        "EN2001a_utterance_00000",
        "EN2001a_utterance_00001",
        "EN2001a_utterance_00003",
        "EN2001a_utterance_00005",
        "EN2001a_utterance_00006",
    ]

    specs = [
        SampleSpec(
            "LibriSpeech",
            sample_id,
            str(libri_root / f"{sample_id}.flac"),
            libri_text[sample_id],
        )
        for sample_id in libri_ids
    ]
    specs.extend(
        SampleSpec(
            "AISHELL-1",
            sample_id,
            str(
                REPO_ROOT
                / "data/rawdata/AIshell/data_aishell/wav/test/S0764"
                / f"{sample_id}.wav"
            ),
            aishell_text[sample_id],
        )
        for sample_id in aishell_ids
    )
    specs.extend(
        SampleSpec(
            "TIMIT",
            sample_id,
            str(timit_root / f"{sample_id}.wav"),
            _read_timit_text(timit_root / f"{sample_id}.txt"),
        )
        for sample_id in timit_ids
    )
    specs.extend(
        SampleSpec(
            "CHiME4",
            sample_id,
            str(
                REPO_ROOT
                / "data/rawdata/CHIME3/CHiME4/data/audio/16kHz/isolated_1ch_track"
                / relative_path
            ),
            chime_text[sample_id],
        )
        for sample_id, relative_path in chime_entries
    )
    specs.extend(
        SampleSpec(
            "AMI",
            sample_id,
            str(
                REPO_ROOT
                / "ami_en2001a_utterances"
                / ami_records[sample_id]["sample"]["audio"]["path"]
            ),
            ami_records[sample_id]["sample"]["text"]["transcript"],
        )
        for sample_id in ami_ids
    )

    tut_root = (
        REPO_ROOT / "data/rawdata/TUT-urban-acoustic-scenes-2018-development/audio"
    )
    tut_ids = [
        "airport-barcelona-0-0-a",
        "bus-barcelona-15-599-a",
        "metro-barcelona-41-1221-a",
        "park-barcelona-89-2429-a",
        "street_traffic-barcelona-161-4901-a",
    ]
    specs.extend(
        SampleSpec(
            "TUT Urban Acoustic Scenes 2018",
            sample_id,
            str(tut_root / f"{sample_id}.wav"),
        )
        for sample_id in tut_ids
    )

    noisex_root = REPO_ROOT / "data/rawdata/NOISEX-92"
    noisex_ids = ["babble", "f16", "factory1", "machinegun", "volvo"]
    specs.extend(
        SampleSpec(
            "NOISEX-92",
            sample_id,
            str(noisex_root / f"{sample_id}.wav"),
            max_duration_sec=10.0,
        )
        for sample_id in noisex_ids
    )

    wham_root = REPO_ROOT / "data/rawdata/wham_noise/tt"
    wham_ids = [
        "050a0501_1.7783_442o030z_-1.7783",
        "050a0502_1.3461_440o030j_-1.3461",
        "050a0504_2.4414_443o0313_-2.4414",
        "050a0505_1.5097_440o030d_-1.5097",
        "050a0506_1.7744_447c0213_-1.7744",
    ]
    specs.extend(
        SampleSpec("WHAM! noise", sample_id, str(wham_root / f"{sample_id}.wav"))
        for sample_id in wham_ids
    )
    return specs


def _output_name(spec: SampleSpec) -> str:
    prefix = {
        "LibriSpeech": "librispeech",
        "AISHELL-1": "aishell",
        "TIMIT": "timit",
        "CHiME4": "chime4",
        "AMI": "ami",
        "TUT Urban Acoustic Scenes 2018": "tut2018",
        "NOISEX-92": "noisex92",
        "WHAM! noise": "wham_noise",
    }[spec.dataset_name]
    return f"{prefix}_{spec.sample_id}.wav"


def _normalize_audio(spec: SampleSpec, destination: Path) -> None:
    source = Path(spec.source_path)
    if not source.is_file():
        raise FileNotFoundError(f"missing source audio: {source}")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if spec.max_duration_sec is not None:
        command.extend(["-t", str(spec.max_duration_sec)])
    command.extend(
        [
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    subprocess.run(command, check=True)


def _record(spec: SampleSpec, audio_name: str) -> dict:
    record = {
        "corpus": {
            "dataset_name": spec.dataset_name,
            "source_urls": {key: list(value) for key, value in EMPTY_SOURCE_URLS.items()},
            "native_metadata": {},
        },
        "sample": {
            "sample_id": spec.sample_id,
            "audio": {"path": f"audio/{audio_name}"},
            "text": {"transcript": spec.transcript},
            "native_metadata": {},
        },
    }
    validate_input_record(record)
    return record


def main() -> None:
    output_dir = REPO_ROOT / "phase2_asr_sample"
    if (output_dir / "manifest.jsonl").exists() or (output_dir / "audio").exists():
        raise FileExistsError(
            f"output already exists: {output_dir}; move or remove it before rebuilding"
        )

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    specs = _build_specs()
    records = []
    for index, spec in enumerate(specs, start=1):
        audio_name = _output_name(spec)
        print(f"[{index:02d}/{len(specs)}] {spec.dataset_name}: {spec.sample_id}")
        _normalize_audio(spec, audio_dir / audio_name)
        records.append(_record(spec, audio_name))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
