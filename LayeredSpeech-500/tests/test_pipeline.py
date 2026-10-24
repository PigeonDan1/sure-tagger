import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from saasr_mini.audio import active_rms
from saasr_mini.editing import SpeechEditResult
from saasr_mini.pipeline import load_config, simulate


class _ScaledSpeechBackend:
    name = "scaled_test_backend"

    def synthesize(
        self, text: str, speaker_index: int, *, instructions: str | None = None
    ) -> np.ndarray:
        del text, instructions
        amplitude = 0.1 if speaker_index == 0 else 0.4
        body = np.full(800, amplitude, dtype=np.float32)
        return np.concatenate(
            [np.zeros(100, dtype=np.float32), body, np.zeros(100, dtype=np.float32)]
        )

    def close(self) -> None:
        return None


class _LengtheningEditBackend:
    name = "lengthening_test_editor"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    def resolve_instruction(self, text: str, explicit_instruction: str | None) -> str | None:
        del text
        return explicit_instruction

    def edit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        source_text: str,
        instruction: str,
        source_audio_path: Path | None = None,
    ) -> SpeechEditResult:
        self.calls.append(
            {
                "sample_rate": sample_rate,
                "source_text": source_text,
                "instruction": instruction,
                "source_audio_path": source_audio_path,
            }
        )
        return SpeechEditResult(
            audio=np.repeat(audio, 2),
            metadata={"request_id": "test-edit", "time_used": 0.01, "rtf": 0.001},
        )

    def close(self) -> None:
        self.close_count += 1


def test_end_to_end_fixture_pipeline(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "topic": "unit test",
                "language": "English",
                "speakers": [
                    {"speaker_index": 0, "role": "lead"},
                    {"speaker_index": 1, "role": "reviewer"},
                ],
                "utterances": [
                    {"utterance_id": "u0", "speaker_index": 0, "text": "Start the check."},
                    {
                        "utterance_id": "u1",
                        "speaker_index": 1,
                        "text": "I can confirm the output.",
                        "overlap_hint": "interrupt",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "audio": {"sample_rate": 8000, "peak_dbfs": -1},
                "scenes": [
                    {
                        "id": "test_scene",
                        "dialogue": "dialogue.json",
                        "speakers": 2,
                        "overlap_ratio": 0.1,
                        "max_simultaneous_speakers": 2,
                        "snr_db": 10,
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3], [3, 2, 1.3]],
                            "rt60_seconds": 0.3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    manifests = simulate(config_path, output)

    manifest = manifests[0]
    assert manifest["num_speakers"] == 2
    assert abs(manifest["overlap_ratio"] - 0.1) < 0.04
    assert manifest["achieved_snr_db"] == 10.0
    audio, sample_rate = sf.read(output / manifest["audio_path"])
    assert sample_rate == 8000
    assert audio.size > 0
    assert (output / "sessions" / "test_scene" / "segments.rttm").is_file()
    assert (output / "exports" / "sa_asr.jsonl").is_file()


def test_external_rir_asset_is_loaded_and_recorded(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "rir_scene",
                "topic": "external RIR",
                "language": "English",
                "speakers": [{"speaker_index": 0, "role": "speaker"}],
                "utterances": [{"utterance_id": "u0", "speaker_index": 0, "text": "Test the room response."}],
            }
        ),
        encoding="utf-8",
    )
    rir_path = tmp_path / "office_rir.wav"
    response = np.zeros(160, dtype=np.float32)
    response[0] = 1.0
    response[24] = 0.35
    sf.write(rir_path, response, 8000, subtype="FLOAT")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 8,
                "audio": {"sample_rate": 8000, "peak_dbfs": -1},
                "scenes": [
                    {
                        "id": "rir_scene",
                        "dialogue": "dialogue.json",
                        "speakers": 1,
                        "overlap_ratio": 0,
                        "max_simultaneous_speakers": 1,
                        "snr_db": None,
                        "environment": "meeting_room",
                        "rir_path": "office_rir.wav",
                        "rir_environment": "meeting_room",
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3]],
                            "rt60_seconds": 0.3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = simulate(config_path, tmp_path / "output", speech_backend=_ScaledSpeechBackend())[0]

    assert manifest["provenance"]["rir_backend"] == "audio_asset"
    assert manifest["provenance"]["rir_asset"] == "office_rir.wav"
    assert (tmp_path / "output/sessions/rir_scene/rirs/speaker_00.wav").is_file()


def test_mixed_rir_assets_are_rejected_without_same_room_metadata(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "mixed_rir_scene",
                "topic": "mixed RIR",
                "language": "English",
                "speakers": [
                    {"speaker_index": 0, "role": "speaker 0"},
                    {"speaker_index": 1, "role": "speaker 1"},
                ],
                "utterances": [
                    {"utterance_id": "u0", "speaker_index": 0, "text": "First."},
                    {"utterance_id": "u1", "speaker_index": 1, "text": "Second."},
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("room_a.wav", "room_b.wav"):
        response = np.zeros(32, dtype=np.float32)
        response[0] = 1.0
        sf.write(tmp_path / name, response, 8000, subtype="FLOAT")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 9,
                "audio": {"sample_rate": 8000, "peak_dbfs": -1},
                "scenes": [
                    {
                        "id": "mixed_rir_scene",
                        "dialogue": "dialogue.json",
                        "speakers": 2,
                        "overlap_ratio": 0,
                        "max_simultaneous_speakers": 2,
                        "snr_db": None,
                        "rir_paths": ["room_a.wav", "room_b.wav"],
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3], [3, 2, 1.3]],
                            "rt60_seconds": 0.3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one shared RIR asset"):
        load_config(config_path)


def test_noise_and_rir_environment_must_match(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "environment_mismatch",
                "topic": "environment mismatch",
                "language": "English",
                "speakers": [{"speaker_index": 0, "role": "speaker"}],
                "utterances": [{"utterance_id": "u0", "speaker_index": 0, "text": "Check the room."}],
            }
        ),
        encoding="utf-8",
    )
    rir_path = tmp_path / "h060_Office_ConferenceRoom_3txts.wav"
    response = np.zeros(32, dtype=np.float32)
    response[0] = 1.0
    sf.write(rir_path, response, 8000, subtype="FLOAT")
    noise_path = tmp_path / "TCAR" / "TCAR_ch01_clip00.wav"
    noise_path.parent.mkdir()
    sf.write(noise_path, np.zeros(128, dtype=np.float32), 8000, subtype="FLOAT")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 10,
                "audio": {"sample_rate": 8000, "peak_dbfs": -1},
                "scenes": [
                    {
                        "id": "environment_mismatch",
                        "dialogue": "dialogue.json",
                        "speakers": 1,
                        "overlap_ratio": 0,
                        "max_simultaneous_speakers": 1,
                        "snr_db": 10,
                        "environment": "meeting_room",
                        "rir_path": rir_path.name,
                        "noise_path": "TCAR/TCAR_ch01_clip00.wav",
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3]],
                            "rt60_seconds": 0.3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noise assets resolve to environments"):
        load_config(config_path)


def test_fixed_speaker_rms_calibration_is_shared_by_utterances(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "calibration_scene",
                "topic": "calibration",
                "language": "English",
                "speakers": [
                    {"speaker_index": 0, "role": "quiet"},
                    {"speaker_index": 1, "role": "loud"},
                ],
                "utterances": [
                    {"utterance_id": "u0", "speaker_index": 0, "text": "quiet one"},
                    {"utterance_id": "u1", "speaker_index": 1, "text": "loud one"},
                    {"utterance_id": "u2", "speaker_index": 0, "text": "quiet two"},
                    {"utterance_id": "u3", "speaker_index": 1, "text": "loud two"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "audio": {"sample_rate": 8000, "peak_dbfs": -1, "speaker_rms_dbfs": -20},
                "scenes": [
                    {
                        "id": "calibration_scene",
                        "dialogue": "dialogue.json",
                        "speakers": 2,
                        "overlap_ratio": 0,
                        "max_simultaneous_speakers": 1,
                        "snr_db": None,
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3], [3, 2, 1.3]],
                            "rt60_seconds": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    manifest = simulate(config_path, output, speech_backend=_ScaledSpeechBackend())[0]

    assert [row["gain_db"] for row in manifest["speaker_calibration"]] == pytest.approx(
        [0.0, -12.0412], abs=0.01
    )
    for index in range(2):
        calibrated, _ = sf.read(
            output
            / "sessions"
            / "calibration_scene"
            / "dry_stems"
            / f"speaker_{index:02d}.wav"
        )
        assert 20 * np.log10(active_rms(calibrated)) == pytest.approx(-20.0, abs=0.02)
    assert (output / "sessions" / "calibration_scene" / "raw_utterances" / "u0.wav").is_file()


def test_speech_edit_runs_before_leveling_and_preserves_source_layer(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        json.dumps(
            {
                "scene_id": "edit_scene",
                "topic": "edit ordering",
                "language": "Chinese",
                "speakers": [{"speaker_index": 0, "role": "speaker"}],
                "utterances": [
                    {
                        "utterance_id": "u0",
                        "speaker_index": 0,
                        "text": "请放慢这句话。",
                        "edit_instruction": (
                            "<rate, factor=0.90>请放慢这句话。</rate>"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 4,
                "audio": {
                    "sample_rate": 8000,
                    "peak_dbfs": -1,
                    "speaker_rms_dbfs": -26,
                },
                "scenes": [
                    {
                        "id": "edit_scene",
                        "dialogue": "dialogue.json",
                        "speakers": 1,
                        "overlap_ratio": 0,
                        "max_simultaneous_speakers": 1,
                        "snr_db": None,
                        "room": {
                            "dimensions_m": [4, 3, 2.6],
                            "microphone_position_m": [2, 1.5, 1.1],
                            "speaker_positions_m": [[1, 1, 1.3]],
                            "rt60_seconds": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    editor = _LengtheningEditBackend()
    output = tmp_path / "output"
    manifest = simulate(
        config_path,
        output,
        speech_backend=_ScaledSpeechBackend(),
        speech_edit_backend=editor,
    )[0]

    scene_dir = output / "sessions" / "edit_scene"
    pre_edit, _ = sf.read(scene_dir / "source_utterances" / "u0.wav")
    post_edit, _ = sf.read(scene_dir / "raw_utterances" / "u0.wav")
    calibrated, _ = sf.read(scene_dir / "utterances" / "u0.wav")
    assert post_edit.size == pre_edit.size * 2
    assert calibrated.size == post_edit.size
    assert 20 * np.log10(active_rms(calibrated)) == pytest.approx(-26.0, abs=0.02)
    assert manifest["provenance"]["speech_edit_backend"] == editor.name
    assert manifest["provenance"]["edited_utterance_count"] == 1
    assert manifest["speech_edits"][0]["file_duration_ratio"] == 2.0
    assert manifest["turns"][0]["pre_edit_source_audio_path"].endswith(
        "source_utterances/u0.wav"
    )
    assert editor.calls[0]["source_text"] == "请放慢这句话。"
    assert editor.close_count >= 1
