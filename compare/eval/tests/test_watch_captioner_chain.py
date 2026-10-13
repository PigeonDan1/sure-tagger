import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import watch_captioner_chain as watcher


class WatcherProcessDiscoveryTest(unittest.TestCase):
    def test_qwen_shell_wrapper_is_not_reported_as_qwen(self):
        # ``ps`` can expose the complete ``bash -c`` string as unquoted words;
        # the script name therefore appears in the shell line even after the
        # actual Python child has exited.
        lines = [
            "100 /bin/bash -c while ps -p 1; do sleep 1; done; "
            "/runtime/bin/python -u /work/run_qwen_benchmark.py --workers 5; "
            "/runtime/bin/python -u /work/watch_captioner_chain.py",
            "101 /runtime/bin/python -u /work/run_qwen_benchmark.py "
            "--native-oss --audio-only --workers 5",
            "102 /bin/sh -c rg run_qwen_benchmark.py",
        ]
        with mock.patch.object(watcher, "process_lines", return_value=lines):
            self.assertEqual(watcher.pids_for("run_qwen_benchmark.py"), [101])

    def test_error_classifier_prioritizes_http_status(self):
        self.assertFalse(watcher.error_is_network("HTTP 400: timeout is invalid"))
        self.assertTrue(watcher.error_is_network("HTTP 503: timeout"))

    def test_deepseek_retry_requires_all_attempts_to_be_network(self):
        network = {
            "status": "failed",
            "error_type": "network_error",
            "error": "HTTP 503: busy; connection reset by peer",
        }
        mixed = {
            "status": "failed",
            "error_type": "network_error",
            "error": "HTTP 503: busy; HTTP 401: denied",
        }
        legacy_mixed = {
            "status": "failed",
            "error_type": "api_error",
            "error": "HTTP 503: busy; HTTP 401: denied",
        }
        self.assertTrue(watcher.deepseek_failure_retryable(network))
        self.assertFalse(watcher.deepseek_failure_retryable(mixed))
        self.assertFalse(watcher.deepseek_failure_retryable(legacy_mixed))


class AmiRetryManifestTest(unittest.TestCase):
    def test_status_filters_successful_items(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_path = root / "staging.index.json"
            runtime_path = root / "runtime.manifest.jsonl"
            raw_dir = root / "raw"
            item_dir = raw_dir / "items"
            item_dir.mkdir(parents=True)
            samples = []
            rows = []
            for index, sample_id in enumerate(("ami-ok", "ami-retry", "ami-skip")):
                audio_path = root / (sample_id + ".wav")
                audio_path.write_bytes(b"audio-" + str(index).encode("ascii"))
                audio_hash = watcher.hashlib.sha256(audio_path.read_bytes()).hexdigest()
                samples.append(
                    {
                        "sample_id": sample_id,
                        "source_audio": "/dataset/t1-ami/audio/" + sample_id + ".wav",
                        "source_sha256": audio_hash,
                    }
                )
                rows.append(
                    {
                        "corpus": {"dataset_name": "", "native_metadata": {}, "source_urls": {}},
                        "sample": {
                            "audio": {"path": str(audio_path)},
                            "native_metadata": {},
                            "sample_id": sample_id,
                            "text": {"transcript": ""},
                        },
                    }
                )
            stage_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            runtime_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (item_dir / "ami-ok.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "audio_sha256": samples[0]["source_sha256"],
                        "caption": "caption",
                    }
                ),
                encoding="utf-8",
            )
            (item_dir / "ami-retry.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "errors": ["native OSS upload failed: connection reset by peer"],
                    }
                ),
                encoding="utf-8",
            )
            (item_dir / "ami-skip.json").write_text(
                json.dumps(
                    {"status": "failed", "errors": ["native response content is empty"]}
                ),
                encoding="utf-8",
            )
            overrides = {
                "AMI_STAGE_INDEX": stage_path,
                "QWEN_MANIFEST": runtime_path,
                "RAW_DIR": raw_dir,
                "AMI_RETRY_ROOT": root / "retries",
                "AMI_COUNT": 3,
            }
            with mock.patch.multiple(watcher, **overrides):
                counts, retry_rows = watcher.ami_qwen_status()
                retry_path = watcher.write_ami_retry_manifest(retry_rows, 1)

            self.assertEqual(
                counts,
                {
                    "success": 1,
                    "failed": 2,
                    "retryable": 1,
                    "non_network": 1,
                    "pending": 0,
                },
            )
            self.assertEqual(len(retry_rows), 1)
            self.assertEqual(
                json.loads(retry_path.read_text(encoding="utf-8"))["sample"]["sample_id"],
                "ami-retry",
            )


class AmiOutputCompleteTest(unittest.TestCase):
    def _write_output(self, root, statuses, digest="digest", fingerprint="fingerprint"):
        output = root / ("ami-output-%d" % len(list(root.iterdir())))
        output.mkdir()
        rows = []
        for index, status in enumerate(statuses):
            row = {"sample_id": "sample-%d" % index, "status": status}
            if status == "failed":
                row.update(error_type="api_error", error="HTTP 400: invalid request")
            rows.append(row)
        (output / "status.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        success_count = sum(status == "success" for status in statuses)
        failed_count = sum(status == "failed" for status in statuses)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "sample_count": len(statuses),
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "input_manifest_sha256": digest,
                    "config_fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )
        return output

    def test_terminal_failures_are_complete_but_retryable_or_pending_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "ami-manifest.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps({"sample_id": "sample-%d" % index}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            with mock.patch.object(watcher, "AMI_COUNT", 3):
                output = self._write_output(root, ("success", "success", "failed"))
                with mock.patch.multiple(
                    watcher, AMI_OUT=output, AMI_MANIFEST=manifest
                ):
                    self.assertTrue(watcher.ami_output_complete("digest", "fingerprint"))
                    self.assertFalse(watcher.ami_output_complete("other", "fingerprint"))
                    self.assertFalse(watcher.ami_output_complete("digest", "other"))

                output = self._write_output(root, ("success", "failed", "failed"))
                rows = [
                    json.loads(line)
                    for line in (output / "status.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                rows[1]["error_type"] = "network_error"
                rows[1]["error"] = "connection reset by peer"
                (output / "status.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with mock.patch.multiple(
                    watcher, AMI_OUT=output, AMI_MANIFEST=manifest
                ):
                    self.assertFalse(watcher.ami_output_complete("digest", "fingerprint"))

                output = self._write_output(root, ("success", "success", "pending"))
                with mock.patch.multiple(
                    watcher, AMI_OUT=output, AMI_MANIFEST=manifest
                ):
                    self.assertFalse(watcher.ami_output_complete("digest", "fingerprint"))


class FullQwenStatusTest(unittest.TestCase):
    def test_status_ignores_terminal_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runtime.manifest.jsonl"
            raw_dir = root / "raw"
            item_dir = raw_dir / "items"
            item_dir.mkdir(parents=True)
            rows = []
            for sample_id in ("ok", "retry", "skip"):
                rows.append({"sample": {"sample_id": sample_id}})
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records = {
                "ok": {"status": "success", "caption": "caption"},
                "retry": {
                    "status": "failed",
                    "errors": ["connection reset by peer"],
                },
                "skip": {
                    "status": "failed",
                    "errors": ["HTTP 401: invalid key"],
                },
            }
            for sample_id, record in records.items():
                (item_dir / (sample_id + ".json")).write_text(
                    json.dumps(record), encoding="utf-8"
                )
            with mock.patch.multiple(
                watcher,
                QWEN_MANIFEST=manifest,
                RAW_DIR=raw_dir,
                EXPECTED_COUNT=3,
            ):
                counts, retry_rows = watcher.full_qwen_status()

        self.assertEqual(
            counts,
            {
                "success": 1,
                "failed": 2,
                "retryable": 1,
                "non_network": 1,
                "pending": 0,
            },
        )
        self.assertEqual(retry_rows, [rows[1]])


class FullCaptionOutputCompleteTest(unittest.TestCase):
    def _write_output(self, root, statuses, digest="digest", fingerprint="fingerprint"):
        output = root / "captioner-output"
        output.mkdir()
        rows = []
        for index, status in enumerate(statuses):
            row = {"sample_id": "sample-%d" % index, "status": status}
            if status == "failed":
                row.update(error_type="api_error", error="HTTP 400: invalid request")
            rows.append(row)
        (output / "status.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        success_count = sum(status == "success" for status in statuses)
        failed_count = sum(status == "failed" for status in statuses)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "sample_count": len(statuses),
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "input_manifest_sha256": digest,
                    "config_fingerprint": fingerprint,
                }
            ),
            encoding="utf-8",
        )
        return output

    def test_terminal_failures_are_complete_and_network_failures_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps({"sample_id": "sample-%d" % index}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            output = self._write_output(root, ("success", "success", "failed"))
            with mock.patch.multiple(
                watcher,
                MANIFEST=manifest,
                CAPTION_OUT=output,
                EXPECTED_COUNT=3,
            ):
                self.assertTrue(watcher.full_output_complete("digest", "fingerprint"))

                rows = [
                    json.loads(line)
                    for line in (output / "status.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                rows[1]["status"] = "failed"
                rows[1]["error_type"] = "network_error"
                rows[1]["error"] = "connection reset by peer"
                (output / "status.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                (output / "summary.json").write_text(
                    json.dumps(
                        {
                            "sample_count": 3,
                            "success_count": 1,
                            "failed_count": 2,
                            "input_manifest_sha256": "digest",
                            "config_fingerprint": "fingerprint",
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertFalse(watcher.full_output_complete("digest", "fingerprint"))

    def test_manifest_sample_ids_must_match_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps({"sample_id": "sample-%d" % index}) + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            output = self._write_output(root, ("success", "success"))
            rows = [
                json.loads(line)
                for line in (output / "status.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[1]["sample_id"] = "other"
            (output / "status.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with mock.patch.multiple(
                watcher,
                MANIFEST=manifest,
                CAPTION_OUT=output,
                EXPECTED_COUNT=2,
            ):
                self.assertFalse(watcher.full_output_complete("digest", "fingerprint"))


if __name__ == "__main__":
    unittest.main()
