from pathlib import Path

import numpy as np
import soundfile as sf

from saasr_mini.noise import (
    PROFILE_METADATA,
    SUPPORTED_PROFILES,
    generate_noise,
    load_noise_asset,
)


def test_environment_profiles_are_deterministic_and_distinct() -> None:
    generated = {
        profile: generate_noise(profile, 16000, 16000, seed=17)
        for profile in SUPPORTED_PROFILES
    }
    assert set(generated) == set(PROFILE_METADATA)
    assert all(audio.shape == (16000,) for audio in generated.values())
    assert all(np.isfinite(audio).all() for audio in generated.values())
    for profile, audio in generated.items():
        assert np.array_equal(audio, generate_noise(profile, 16000, 16000, seed=17))
    assert not np.array_equal(generated["office"], generated["train"])


def test_noise_asset_is_mono_resampled_and_tiled(tmp_path: Path) -> None:
    source = np.linspace(-0.4, 0.4, 800, dtype=np.float32)
    asset = tmp_path / "noise.wav"
    sf.write(asset, source, 16000, subtype="PCM_16")

    loaded = load_noise_asset(asset, samples=1200, sample_rate=8000)

    assert loaded.shape == (1200,)
    assert loaded.dtype == np.float32
    assert np.isfinite(loaded).all()
    assert abs(float(np.mean(loaded))) < 1e-4
