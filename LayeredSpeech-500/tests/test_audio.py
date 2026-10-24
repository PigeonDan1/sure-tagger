import numpy as np
import pytest

from saasr_mini.audio import active_rms, interval_overlap_ratio


def test_interval_overlap_ratio() -> None:
    assert interval_overlap_ratio([(0.0, 2.0), (1.0, 3.0)]) == pytest.approx(1 / 3)
    assert interval_overlap_ratio([(0.0, 1.0), (2.0, 3.0)]) == 0.0


def test_active_rms_ignores_silence() -> None:
    audio = np.concatenate([np.zeros(100), np.full(200, 0.1), np.zeros(100)])
    assert 20 * np.log10(active_rms(audio)) == pytest.approx(-20.0, abs=0.01)
