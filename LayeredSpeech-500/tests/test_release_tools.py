import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asset_inventory_matches_frozen_scene_inputs() -> None:
    inventory = json.loads((ROOT / "data/asset_hashes.json").read_text(encoding="utf-8"))
    assets = {item["path"] for item in inventory["files"]}
    voice_bank = json.loads((ROOT / "data/voice_bank_v4.json").read_text(encoding="utf-8"))
    for voice in voice_bank.values():
        path = voice["prompt_audio"]
        assert path.startswith("external_assets/")
        if path.startswith("external_assets/fleurs/"):
            stem = Path(path).stem
            lang, slot = stem.removeprefix("fleurs_").rsplit("_", 1)
            filename = f"fleurs_{lang}_{int(slot):02d}.wav"
            path = "external_assets/fleurs/audio_enhanced_deepfilternet/" + filename
        assert path.removeprefix("external_assets/") in assets
    scenes = [
        json.loads(line)
        for line in (ROOT / "data/scene_scripts_v4.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for scene in scenes:
        selection = scene["rir_selection"]
        if selection["mode"] == "measured":
            assert selection["asset_path"] in assets


def test_asset_verifier_reports_missing_and_hash_mismatch(tmp_path: Path) -> None:
    verifier = _load_script("verify_assets")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "path": "refs/sample.wav", "group": "reference_voice", "sha256": "0" * 64, "bytes": 3,
    }]}), encoding="utf-8")
    assert verifier.verify_assets(tmp_path, manifest) == ["missing: refs/sample.wav"]
    sample = tmp_path / "refs/sample.wav"
    sample.parent.mkdir()
    sample.write_bytes(b"abc")
    assert verifier.verify_assets(tmp_path, manifest) == ["SHA-256 mismatch: refs/sample.wav"]


def test_full_preflight_does_not_accept_empty_model_directories(tmp_path: Path) -> None:
    reproduce = _load_script("reproduce")
    errors = reproduce.check_models(tmp_path)
    assert any("missing model provenance" in error for error in errors)
    assert any("missing Dots runtime" in error for error in errors)
