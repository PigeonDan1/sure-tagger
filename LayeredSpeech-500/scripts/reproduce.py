#!/usr/bin/env python3
"""Run the offline fixture or the externally provisioned full profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from verify_assets import verify_assets
from download_models import check_model_files


ROOT = Path(__file__).resolve().parents[1]


def check_models(models_root: Path) -> list[str]:
    lock = yaml.safe_load((ROOT / "data/models.lock.yaml").read_text(encoding="utf-8"))
    errors = []
    for model in lock["models"]:
        target = models_root / model["id"]
        provenance = target / "MODEL_SOURCE.json"
        if not provenance.is_file():
            errors.append(f"missing model provenance: {provenance}")
            continue
        try:
            source = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid model provenance: {provenance}: {exc}")
            continue
        if (source.get("repository"), source.get("revision")) != (
            model["repository"], model["revision"]
        ):
            errors.append(f"model revision mismatch: {provenance}")
        errors.extend(check_model_files(model["id"], target / "checkpoints"))
    runtime_python = models_root / "runtime/.venv/bin/python"
    if not runtime_python.is_file():
        errors.append(f"missing Dots runtime: {runtime_python}; run scripts/setup_dots_runtime.sh")
    else:
        try:
            result = subprocess.run(
                [str(runtime_python), "-c", (
                    "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13); "
                    "import dots_tts.runtime; import dots_tts.edit_runtime"
                )],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if result.returncode:
                errors.append(f"Dots runtime import failed: {result.stderr.strip()[-500:]}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Dots runtime unavailable: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fixture", "full"), default="fixture")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v4_multilingual")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mini.yaml")
    parser.add_argument("--scene-ids")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "external_assets")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--check", action="store_true", help="Check prerequisites without generating audio.")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.profile == "fixture":
        if args.check:
            if not args.config.is_file():
                raise SystemExit(f"missing fixture config: {args.config}")
            print(f"Fixture config ready: {args.config}")
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        command = [sys.executable, "-m", "saasr_mini", "run", "--config", str(args.config), "--output", str(args.output)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        fixture_seed = yaml.safe_load(args.config.read_text(encoding="utf-8"))["seed"]
        (args.output / "release_run.json").write_text(
            json.dumps({"profile": "fixture", "seed": fixture_seed}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    assets_root = args.assets_root.expanduser().resolve()
    models_root = args.models_root.expanduser().resolve()
    errors = verify_assets(assets_root) + check_models(models_root)
    if errors:
        raise SystemExit("Full reproduction prerequisites failed:\n- " + "\n- ".join(errors[:25]) + (
            f"\n... and {len(errors) - 25} more" if len(errors) > 25 else ""
        ))
    if args.check:
        print("Full reproduction prerequisites passed")
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"full-profile output must be a new or empty directory: {args.output}")
    env = os.environ.copy()
    env.update({
        "RELEASE_ASSETS_ROOT": str(assets_root),
        "DOTS_BASE_MODEL": str(models_root / "dots_tts_base"),
        "DOTS_EDIT_MODEL": str(models_root / "dots_tts_edit"),
        "DOTS_BASE_PYTHON": str(models_root / "runtime/.venv/bin/python"),
        "DOTS_EDIT_PYTHON": str(models_root / "runtime/.venv/bin/python"),
    })
    command = [
        sys.executable, str(ROOT / "scripts/build_dots_dataset.py"),
        "--output", str(args.output), "--stage", "all", "--gpus", args.gpus,
    ]
    if args.scene_ids:
        command.extend(["--scene-ids", args.scene_ids])
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.scene_ids:
        print("Subset generation complete; 500-scene Ground Truth requires a full run.")
        return
    env["RELEASE_OUTPUT_ROOT"] = str(args.output)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_ground_truth.py")],
                   cwd=ROOT, env=env, check=True)
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    qc = json.loads((args.output / "qc_report.json").read_text(encoding="utf-8"))["summary"]
    gt_rows = sum(1 for line in (args.output / "ground_truth/ground_truth.jsonl").open(
        encoding="utf-8") if line.strip())
    if summary["scene_count"] != 500 or gt_rows != 500 or qc["hard_failure_count"]:
        raise SystemExit("full run failed acceptance: expected 500 scenes, 500 GT rows and zero hard QC failures")
    print("Full run accepted: 500 scenes, 500 GT rows, zero hard QC failures")


if __name__ == "__main__":
    main()
