"""Acoustic-environment labels and asset compatibility checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


ENVIRONMENT_ALIASES = {
    "anechoic": "dry",
    "cafe": "restaurant",
    "car": "vehicle",
    "classroom": "meeting_room",
    "office": "meeting_room",
    "outdoor": "open_space",
    "street": "open_space",
    "station": "public_transit",
    "train": "public_transit",
}
SUPPORTED_ENVIRONMENTS = {
    "dry",
    "meeting_room",
    "open_space",
    "restaurant",
    "public_transit",
    "vehicle",
    "residential",
}

NOISE_ASSET_ENVIRONMENTS = {
    "OOFFICE": "meeting_room",
    "OMEETING": "meeting_room",
    "OHALLWAY": "meeting_room",
    "PCAFETER": "restaurant",
    "DKITCHEN": "restaurant",
    "PSTATION": "public_transit",
    "TMETRO": "public_transit",
    "TBUS": "public_transit",
    "TCAR": "vehicle",
    "NFIELD": "open_space",
    "NPARK": "open_space",
    "NRIVER": "open_space",
    "SPSQUARE": "open_space",
    "STRAFFIC": "open_space",
    "PRESTO": "public_transit",
    "DLIVING": "residential",
    "DWASHING": "residential",
}

NOISE_PROFILE_ENVIRONMENTS = {
    "office": "meeting_room",
    "airport": "public_transit",
    "station": "public_transit",
    "train": "public_transit",
    "cafe": "restaurant",
    "street": "open_space",
}


def canonical_environment(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ENVIRONMENT_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_ENVIRONMENTS:
        choices = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ValueError(f"unknown acoustic environment {value!r}; choose one of: {choices}")
    return normalized


def infer_noise_environment(path: Path) -> str | None:
    """Infer a DEMAND-style noise environment from its parent directory."""
    for part in reversed(path.parts):
        environment = NOISE_ASSET_ENVIRONMENTS.get(part.upper())
        if environment is not None:
            return environment
    return None


def infer_noise_profile_environment(profile: str) -> str | None:
    return NOISE_PROFILE_ENVIRONMENTS.get(profile.strip().lower())


def infer_rir_environment(path: Path) -> str | None:
    """Infer a broad environment from a measured RIR filename.

    Unknown filenames must be accompanied by an explicit ``rir_environment``
    in the scene config; guessing from an arbitrary name would weaken the
    alignment check.
    """
    name = path.stem.lower()
    compact = re.sub(r"[^a-z0-9]+", "", name)
    tokens = set(re.split(r"[^a-z0-9]+", name))
    if "officeconferenceroom" in compact or "classroom" in compact:
        return "meeting_room"
    if any(
        marker in compact
        for marker in ("coffeeshop", "restaurant", "cafeteria", "diningroom", "fastfood")
    ) or "bar" in tokens:
        return "restaurant"
    if "car" in tokens or compact.endswith("car"):
        return "vehicle"
    if "train" in tokens or "trainstation" in compact or "subwaystation" in compact:
        return "public_transit"
    if any(
        marker in compact
        for marker in (
            "outside",
            "outdoor",
            "parkinglot",
            "grassyfield",
            "forest",
            "streets",
            "streetsof",
            "playground",
            "backporch",
        )
    ):
        return "open_space"
    if any(marker in compact for marker in ("bedroom", "livingroom", "babysroom")):
        return "residential"
    return None


def _validate_asset_environment(
    *,
    scene_id: str,
    label: str,
    expected: str,
    inferred: Iterable[str | None],
    declared: str | None,
) -> None:
    inferred_values = {item for item in inferred if item is not None}
    if declared is not None:
        if declared != expected:
            raise ValueError(
                f"{scene_id}: {label}_environment={declared!r} does not match "
                f"environment={expected!r}"
            )
        if inferred_values and inferred_values != {declared}:
            raise ValueError(
                f"{scene_id}: {label} assets resolve to environments "
                f"{sorted(inferred_values)}, not {declared!r}"
            )
        return
    if not inferred_values:
        raise ValueError(
            f"{scene_id}: cannot infer {label} environment; set "
            f"{label}_environment explicitly"
        )
    if inferred_values != {expected}:
        raise ValueError(
            f"{scene_id}: {label} assets resolve to environments "
            f"{sorted(inferred_values)}, but scene environment is {expected!r}"
        )


def validate_scene_environment(
    *,
    scene_id: str,
    environment: str | None,
    noise_profile: str | None,
    noise_path: Path | None,
    noise_environment: str | None,
    rir_paths: list[Path],
    rir_environment: str | None,
) -> None:
    has_acoustic_assets = bool(noise_profile or noise_path or rir_paths)
    if not has_acoustic_assets:
        return
    if environment is None:
        raise ValueError(
            f"{scene_id}: environment is required when noise or measured RIR assets are configured"
        )

    if noise_profile is not None:
        profile_environment = infer_noise_profile_environment(noise_profile)
        if profile_environment is None:
            raise ValueError(
                f"{scene_id}: noise_profile {noise_profile!r} has no acoustic environment mapping"
            )
        _validate_asset_environment(
            scene_id=scene_id,
            label="noise",
            expected=environment,
            inferred=[profile_environment],
            declared=noise_environment,
        )
    elif noise_path is not None:
        _validate_asset_environment(
            scene_id=scene_id,
            label="noise",
            expected=environment,
            inferred=[infer_noise_environment(noise_path)],
            declared=noise_environment,
        )

    if rir_paths:
        _validate_asset_environment(
            scene_id=scene_id,
            label="rir",
            expected=environment,
            inferred=[infer_rir_environment(path) for path in rir_paths],
            declared=rir_environment,
        )
