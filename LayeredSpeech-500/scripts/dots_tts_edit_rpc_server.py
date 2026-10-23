#!/usr/bin/env python3
"""Line-delimited JSON-RPC bridge for the dots.tts.edit environment."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


runtime: Any = None


def initialize(params: dict[str, Any]) -> dict[str, Any]:
    global runtime
    from dots_tts.edit_runtime import DotsTtsEditRuntime

    kwargs: dict[str, Any] = {
        "precision": str(params.get("precision", "bfloat16")),
        "optimize": bool(params.get("optimize", False)),
        "max_generate_length": int(params.get("max_generate_length", 500)),
    }
    if params.get("max_sequence_length") is not None:
        kwargs["max_sequence_length"] = int(params["max_sequence_length"])
    runtime = DotsTtsEditRuntime.from_pretrained(str(params["checkpoint_path"]), **kwargs)
    return {
        "checkpoint_path": str(Path(params["checkpoint_path"]).resolve()),
        "sample_rate": int(runtime.sample_rate),
        "precision": kwargs["precision"],
    }


def edit(params: dict[str, Any]) -> dict[str, Any]:
    if runtime is None:
        raise RuntimeError("dots.tts.edit runtime has not been initialized")
    import soundfile as sf

    from dots_tts.utils.util import seed_everything

    seed_everything(int(params.get("seed", 20260414)))
    generation: dict[str, Any] = {
        "source_audio_path": str(params["source_audio_path"]),
        "instruction": str(params["instruction"]),
        "source_text": params.get("source_text"),
        "target_text": params.get("target_text"),
        "use_xvector": params.get("use_xvector", "auto"),
        "speaker_scale": float(params.get("speaker_scale", 1.5)),
    }
    for key in ("ode_method", "num_steps", "guidance_scale"):
        if params.get(key) is not None:
            generation[key] = params[key]
    result = runtime.generate_edit(**generation)
    output_path = Path(params["output_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = result["audio"].detach().cpu().squeeze().float().numpy()
    sf.write(output_path, audio, int(result["sample_rate"]), subtype="FLOAT")
    return {
        "audio_path": str(output_path),
        "sample_rate": int(result["sample_rate"]),
        "duration_seconds": float(result["duration_seconds"]),
        "time_used": float(result["time_used"]),
        "rtf": float(result["rtf"]),
        "request_id": str(result["request_id"]),
        "source_text_source": str(result["source_text_source"]),
        "target_text_source": str(result["target_text_source"]),
    }


def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return initialize(params)
    if method == "edit":
        return edit(params)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    for line in sys.stdin:
        request: dict[str, Any] | None = None
        try:
            request = json.loads(line)
            result = dispatch(str(request["method"]), dict(request.get("params") or {}))
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if request is not None else None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
