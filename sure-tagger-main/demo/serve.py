#!/usr/bin/env python3
"""Serve the repository demo and its audio fixtures without stale caching."""
import http.server
import os
from pathlib import Path

PORT = 8765


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    print(f"Open http://127.0.0.1:{PORT}/sure-tagger-main/demo/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), NoCacheHandler).serve_forever()
