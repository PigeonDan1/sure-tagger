from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .pipeline import simulate

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("demo_output"),
) -> None:
    """Generate all scenes in CONFIG using offline fixture audio."""
    manifests = simulate(config, output)
    for manifest in manifests:
        typer.echo(
            f"{manifest['session_id']}: {manifest['num_speakers']} speakers, "
            f"{manifest['duration_seconds']:.2f}s, overlap={manifest['overlap_ratio']:.3f}"
        )
    typer.echo(f"Artifacts: {output.resolve()}")


@app.command()
def inspect(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Print a compact summary of one generated session manifest."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            {
                key: data[key]
                for key in (
                    "session_id",
                    "audio_path",
                    "duration_seconds",
                    "num_speakers",
                    "overlap_ratio",
                    "max_concurrency",
                    "target_snr_db",
                    "achieved_snr_db",
                )
            },
            indent=2,
        )
    )

