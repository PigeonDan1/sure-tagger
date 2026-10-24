from pathlib import Path

from saasr_mini.pipeline import load_config, load_dialogue

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_scenes_cover_two_four_and_twelve_speakers() -> None:
    config = load_config(ROOT / "configs" / "mini.yaml")
    assert [scene.speakers for scene in config.scenes] == [2, 4, 12]
    for scene in config.scenes:
        dialogue = load_dialogue(scene.dialogue, scene.speakers)
        assert len(dialogue.speakers) == scene.speakers


def test_dialogue_loads_optional_tts_instructions(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        """{
  "scene_id": "emotion",
  "topic": "emotion control",
  "language": "Chinese",
  "speakers": [{"speaker_index": 0, "role": "speaker"}],
  "utterances": [{
    "utterance_id": "u0",
    "speaker_index": 0,
    "text": "今天的测试顺利完成了。",
    "instructions": "请用开心、兴奋的语气说",
    "edit_instruction": "<emo type='happy' level='2'>今天的测试顺利完成了。</emo>"
  }]
}\n""",
        encoding="utf-8",
    )

    dialogue = load_dialogue(dialogue_path, expected_speakers=1)

    assert dialogue.utterances[0].instructions == "请用开心、兴奋的语气说"
    assert dialogue.utterances[0].edit_instruction == (
        "<emo type='happy' level='2'>今天的测试顺利完成了。</emo>"
    )


def test_dialogue_resolves_precomputed_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    audio_path.touch()
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(
        """{
  "scene_id": "cached",
  "topic": "cached speech",
  "language": "English",
  "speakers": [{"speaker_index": 0, "role": "speaker"}],
  "utterances": [{
    "utterance_id": "u0",
    "speaker_index": 0,
    "text": "Use the cached waveform.",
    "input_audio_path": "source.wav"
  }]
}\n""",
        encoding="utf-8",
    )

    dialogue = load_dialogue(dialogue_path, expected_speakers=1)

    assert dialogue.utterances[0].input_audio_path == audio_path.resolve()
