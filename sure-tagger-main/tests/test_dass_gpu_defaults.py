"""Regression tests for DASS' GPU-first runtime selection.

These tests deliberately avoid importing the DASS model stack or allocating a
CUDA tensor.  They protect the configuration/CLI contract while remaining
runnable on CPU-only development and CI hosts.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tagger.local_config import DASS_PYTHON
from tagger.pipelines.tagging import build_arg_parser
from tagger.tools.sound_field_scene.dass_noise_type_detector import (
    DassNoiseTypeClient,
    DassNoiseTypeConfig,
    DassNoiseTypeError,
    _subprocess_config,
)


class DassGpuDefaultsTest(unittest.TestCase):
    def test_config_defaults_to_gpu(self):
        config = DassNoiseTypeConfig(subprocess_python="")

        self.assertTrue(config.use_gpu)
        self.assertTrue(config.to_record()["use_gpu"])

    def test_default_runtime_is_project_local(self):
        self.assertEqual(DASS_PYTHON, ".runtime/dass_runtime_py311/bin/python")

    def test_cli_is_gpu_first_with_explicit_cpu_opt_out(self):
        parser = build_arg_parser()

        self.assertTrue(parser.parse_args([]).dass_use_gpu)
        self.assertTrue(parser.parse_args(["--dass-use-gpu"]).dass_use_gpu)
        self.assertFalse(parser.parse_args(["--dass-use-cpu"]).dass_use_gpu)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--dass-use-gpu", "--dass-use-cpu"])

    def test_subprocess_payload_preserves_device_selection(self):
        for use_gpu in (True, False):
            with self.subTest(use_gpu=use_gpu):
                config = DassNoiseTypeConfig(
                    use_gpu=use_gpu,
                    subprocess_python="",
                )
                self.assertIs(_subprocess_config(config)["use_gpu"], use_gpu)

    def test_gpu_request_fails_fast_when_cuda_is_unavailable(self):
        # _load_runtime checks the checkpoint contract before importing the
        # model classes, so placeholder files are sufficient for this guard.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            for filename in (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ):
                (model_dir / filename).touch()

            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
            )
            client = DassNoiseTypeClient(
                DassNoiseTypeConfig(
                    model_dir=str(model_dir),
                    use_gpu=True,
                    subprocess_python="",
                )
            )
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(
                    DassNoiseTypeError,
                    "DASS GPU was requested but CUDA is unavailable",
                ):
                    client._load_runtime()


if __name__ == "__main__":
    unittest.main()
