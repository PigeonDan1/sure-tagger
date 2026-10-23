from decimal import Decimal
from pathlib import Path

import pytest

from saasr_mini.editing import DotsTtsEditBackend, build_rate_instruction


def test_build_rate_instruction_escapes_text_and_formats_factor() -> None:
    assert build_rate_instruction("A < B & C", Decimal("0.90")) == (
        '<rate, factor=0.90>A &lt; B &amp; C</rate>'
    )


@pytest.mark.parametrize("factor", [0.54, 1.0, 1.951, float("nan")])
def test_build_rate_instruction_rejects_unsupported_factors(factor: float) -> None:
    with pytest.raises(ValueError):
        build_rate_instruction("test", factor)


def test_explicit_instruction_overrides_default_rate(tmp_path: Path) -> None:
    model_dir = tmp_path / "dots-edit"
    checkpoint = model_dir / "checkpoints"
    python = model_dir / ".venv" / "bin" / "python"
    server = tmp_path / "server.py"
    checkpoint.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.touch()
    server.touch()
    backend = DotsTtsEditBackend(
        model_dir,
        16_000,
        python=python,
        server=server,
        default_rate=0.88,
        use_xvector="on",
    )

    explicit = '<pitch, semitones=-5>测试文本。</pitch>'
    assert backend.resolve_instruction("测试文本。", explicit) == explicit
    assert backend.resolve_instruction("测试文本。", None) == (
        '<rate, factor=0.88>测试文本。</rate>'
    )
    assert backend.use_xvector is True
