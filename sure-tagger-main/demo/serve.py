#!/usr/bin/env python3
"""Demo site server with no-cache headers.

data.js is regenerated in place by scripts/build_demo_data.py; plain
``python3 -m http.server`` sends no Cache-Control, so browsers heuristically
cache stale copies and keep showing removed tag fields (e.g. ``sound``).
This server serves everything with ``Cache-Control: no-store`` instead.
"""
import http.server
import os

PORT = 8765


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), NoCacheHandler).serve_forever()
