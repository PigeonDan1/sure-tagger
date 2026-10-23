#!/usr/bin/env python3
"""Line-delimited JSON-RPC bridge for the separately installed dots.tts runtime."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


runtime: Any = None


def initialize() -> dict[str, Any]:
    global runtime
    from dots_tts.runtime import DotsTtsRuntime

    checkpoint = Path(os.environ["MODEL_PATH"]).resolve()
    runtime = DotsTtsRuntime.from_pretrained(
        str(checkpoint),
        precision=os.environ.get("PRECISION", "float16"),
        max_generate_length=int(os.environ.get("MAX_GENERATE_LENGTH", "500")),
    )
    return {"checkpoint_path": str(checkpoint)}


def synthesize(arguments: dict[str, Any]) -> dict[str, Any]:
    if runtime is None:
        raise RuntimeError("dots.tts runtime has not been initialized")
    from dots_tts.utils.util import seed_everything
    import soundfile as sf

    if arguments.get("instructions"):
        raise ValueError("this benchmark sends plain text to the base TTS checkpoint")
    if arguments.get("seed") is not None:
        seed_everything(int(arguments["seed"]))
    result = runtime.generate(
        text=str(arguments["text"]),
        prompt_audio_path=str(arguments["prompt_audio_path"]),
        prompt_text=str(arguments.get("prompt_text") or ""),
        template_name="tts",
        language=arguments.get("language"),
        prompt_language=arguments.get("prompt_language"),
        speaker_scale=float(arguments.get("speaker_scale", 1.5)),
        ode_method=str(arguments.get("ode_method", "euler")),
        num_steps=int(arguments.get("num_steps", 10)),
        guidance_scale=float(arguments.get("guidance_scale", 1.2)),
        normalize_text=bool(arguments.get("normalize_text", False)),
    )
    output = Path(arguments["output_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = result["audio"].float().cpu().squeeze().numpy()
    peak = float(abs(audio).max()) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    sf.write(output, audio, int(result["sample_rate"]))
    return {"audio_path": str(output), "sample_rate": int(result["sample_rate"])}


def main() -> None:
    for line in sys.stdin:
        request: dict[str, Any] | None = None
        try:
            request = json.loads(line)
            method = request.get("method")
            if method == "initialize":
                result = initialize()
            elif method == "tools/call":
                result = synthesize(request["params"]["arguments"])
            else:
                raise ValueError(f"unknown method: {method}")
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
