from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repository_files() -> list[Path]:
    # Reference prompts and local environments are intentional runtime inputs,
    # not source-controlled model/data artifacts.
    excluded = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        ".venv",
        "refs",
        "runs",
        "models",
        "external_assets",
    }
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.parts)
    ]


def test_repository_contains_no_data_or_model_artifacts() -> None:
    blocked = {".wav", ".flac", ".mp3", ".sqlite", ".db", ".pt", ".pth", ".ckpt", ".onnx"}
    assert not [path for path in _repository_files() if path.suffix.lower() in blocked]
    allowed_metadata = {"data", "manifests", "docs"}
    assert not [
        path
        for path in _repository_files()
        if path.stat().st_size > 20_000_000 and not (set(path.parts) & allowed_metadata)
    ]


def test_text_files_contain_no_high_confidence_secret_or_private_path() -> None:
    secret_prefixes = ["".join(parts) for parts in (("s", "k-"), ("g", "hp_"), ("github", "_pat_"))]
    private_roots = ["/" + "".join(parts) + "/" for parts in (("c", "pfs"), ("new", "cpfs"))]
    private_key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    findings: list[str] = []
    for path in _repository_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(prefix in text for prefix in secret_prefixes + private_roots):
            findings.append(str(path.relative_to(ROOT)))
        if private_key_marker in text:
            findings.append(str(path.relative_to(ROOT)))
    assert not findings
