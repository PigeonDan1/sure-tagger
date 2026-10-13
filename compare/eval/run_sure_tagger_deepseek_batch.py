#!/usr/bin/env python3
"""Convenience entry point for the sure-tagger DeepSeek batch."""

from __future__ import print_function

import sys

import run_deepseek_batch


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--system" in args:
        raise SystemExit("sure-tagger wrapper does not accept --system")
    return run_deepseek_batch.main(["--system", "sure-tagger"] + args)


if __name__ == "__main__":
    raise SystemExit(main())
