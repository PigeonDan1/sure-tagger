import json
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_qwen_benchmark as qwen


class QwenAudioOnlyTest(unittest.TestCase):
    def test_request_shape_is_audio_only(self):
        body = qwen.build_request_body(b"audio-bytes", None, "model")
        self.assertEqual(
            [part["type"] for part in body["messages"][0]["content"]],
            ["input_audio"],
        )
        qwen.assert_request_body(body, None)

    def test_native_request_shape_is_one_audio_uri(self):
        body = qwen.build_native_request_body(
            "oss://benchmark/audio.wav", "qwen-omni-turbo"
        )
        self.assertEqual(
            body["input"]["messages"][0]["content"],
            [{"audio": "oss://benchmark/audio.wav"}],
        )
        qwen.assert_native_request_body(body)

    def test_native_response_text_extraction(self):
        payload = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [{"text": "an audio caption"}],
                        },
                    }
                ]
            }
        }
        self.assertEqual(qwen.native_content_text(payload), ("an audio caption", "stop"))

    def test_native_upload_uses_opaque_object_name_and_cache(self):
        class FakeResponse(object):
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = ""

            def json(self):
                return self._payload

        class FakeSession(object):
            get_calls = []
            post_calls = []

            def get(self, *args, **kwargs):
                self.get_calls.append((args, kwargs))
                return FakeResponse(
                    200,
                    {
                        "output": {
                            "upload_host": "https://oss.invalid/upload",
                            "upload_dir": "opaque-dir",
                            "oss_access_key_id": "id",
                            "signature": "signature",
                            "policy": "policy",
                            "x_oss_object_acl": "private",
                            "x_oss_forbid_overwrite": "true",
                        }
                    },
                )

            def post(self, *args, **kwargs):
                self.post_calls.append((args, kwargs))
                return FakeResponse(200, {})

        fake_requests = SimpleNamespace(Session=FakeSession)
        with tempfile.NamedTemporaryFile(
            prefix="sample-id-must-not-leak-", suffix=".wav"
        ) as stream, mock.patch.dict(sys.modules, {"requests": fake_requests}):
            stream.write(b"audio")
            stream.flush()
            uploader = qwen.NativeOssUploader(
                "model", "https://api.invalid/uploads", "secret", 10
            )
            audio_path = Path(stream.name)
            first_uri = uploader.upload(audio_path)
            second_uri = uploader.upload(audio_path)

        self.assertEqual(first_uri, second_uri)
        self.assertNotIn(audio_path.name, first_uri)
        self.assertEqual(len(FakeSession.get_calls), 1)
        self.assertEqual(len(FakeSession.post_calls), 1)
        upload_kwargs = FakeSession.post_calls[0][1]
        self.assertNotIn(audio_path.name, upload_kwargs["data"]["key"])
        self.assertNotEqual(
            upload_kwargs["files"]["file"][0], audio_path.name
        )

    def test_native_upload_refreshes_policy_after_network_error(self):
        class FakeResponse(object):
            status_code = 200
            text = ""

            def json(self):
                return {}

        class FakeSession(object):
            get_calls = 0
            post_calls = 0

            def get(self, *args, **kwargs):
                type(self).get_calls += 1
                return SimpleNamespace(
                    status_code=200,
                    text="",
                    json=lambda: {
                        "data": {
                            "upload_host": "https://oss.invalid/upload",
                            "upload_dir": "opaque-dir",
                            "oss_access_key_id": "id",
                            "signature": "signature",
                            "policy": "policy",
                            "x_oss_object_acl": "private",
                            "x_oss_forbid_overwrite": "true",
                            "expire_in_seconds": 300,
                        }
                    },
                )

            def post(self, *args, **kwargs):
                type(self).post_calls += 1
                if type(self).post_calls == 1:
                    raise OSError("expired upload policy")
                return FakeResponse()

        fake_requests = SimpleNamespace(Session=FakeSession)
        with tempfile.NamedTemporaryFile(suffix=".wav") as stream, mock.patch.dict(
            sys.modules, {"requests": fake_requests}
        ):
            stream.write(b"audio")
            stream.flush()
            uploader = qwen.NativeOssUploader(
                "model", "https://api.invalid/uploads", "secret", 10
            )
            audio_path = Path(stream.name)
            with self.assertRaises(qwen.RequestFailure):
                uploader.upload(audio_path)
            self.assertTrue(uploader.upload(audio_path).startswith("oss://"))

        self.assertEqual(FakeSession.get_calls, 2)
        self.assertEqual(FakeSession.post_calls, 2)

    def test_endpoint_resolver_rotates_and_restores(self):
        original = socket.getaddrinfo
        resolver = qwen.EndpointIPv4Resolver(
            "https://dashscope.aliyuncs.com/v1", ["192.0.2.1", "192.0.2.2"]
        )
        resolver.install()
        try:
            first = socket.getaddrinfo(
                "dashscope.aliyuncs.com", 443, 0, socket.SOCK_STREAM
            )
            second = socket.getaddrinfo(
                "dashscope.aliyuncs.com", 443, 0, socket.SOCK_STREAM
            )
            self.assertEqual(first[0][0], socket.AF_INET)
            self.assertEqual(first[0][4][0], "192.0.2.1")
            self.assertEqual(second[0][4][0], "192.0.2.2")

            # Non-TCP lookups and unrelated hosts continue through the original
            # resolver rather than being rewritten by the endpoint pinning.
            datagram = socket.getaddrinfo(
                "localhost", 0, socket.AF_INET, socket.SOCK_DGRAM
            )
            self.assertTrue(datagram)
        finally:
            resolver.restore()
        self.assertIs(socket.getaddrinfo, original)

    def test_max_inflight_is_optional_and_gate_releases_on_error(self):
        default_args = qwen.parse_args([])
        self.assertIsNone(default_args.max_inflight)
        self.assertEqual(default_args.effective_max_inflight, default_args.workers)
        limited_args = qwen.parse_args(
            ["--workers", "10", "--max-inflight", "2"]
        )
        self.assertEqual(limited_args.max_inflight, 2)
        self.assertEqual(limited_args.effective_max_inflight, 2)

        original = qwen.request_caption

        def fail(*args, **kwargs):
            raise RuntimeError("offline test failure")

        qwen.request_caption = fail
        gate = threading.BoundedSemaphore(1)
        args = SimpleNamespace(
            endpoint="https://example.invalid/v1",
            transport="compatible",
            model="model",
            audio_only=True,
            prompt="",
            timeout=1,
            max_tokens=1,
        )
        try:
            with self.assertRaises(RuntimeError):
                qwen.request_caption_with_gate(
                    {"audio_path": Path("unused.wav")},
                    "key",
                    args,
                    request_semaphore=gate,
                )
        finally:
            qwen.request_caption = original
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_retry_classifier_accepts_network_errors_only(self):
        self.assertTrue(
            qwen.is_network_failure(
                qwen.RequestFailure("native OSS upload failed: connection reset")
            )
        )
        self.assertTrue(qwen.is_network_failure(qwen.RequestFailure("HTTP 503")))
        self.assertFalse(
            qwen.is_network_failure(
                qwen.RequestFailure("native response content is empty")
            )
        )
        self.assertFalse(qwen.is_network_failure(qwen.RequestFailure("HTTP 401")))
        self.assertFalse(
            qwen.is_network_failure(qwen.RequestFailure("HTTP 400: timeout is invalid"))
        )
        self.assertTrue(
            qwen.is_network_failure(qwen.RequestFailure("HTTP 429: timeout"))
        )

    def test_terminal_failure_checkpoint_is_reused_without_request(self):
        sample = {
            "sample_id": "sample-1",
            "audio_sha256": "audio-hash",
        }
        record = {
            "sample_id": "sample-1",
            "audio_sha256": "audio-hash",
            "prompt_sha256": "prompt-hash",
            "model": "model",
            "endpoint": "https://example/v1",
            "input_contract": "audio_only",
            "status": "failed",
            "errors": ["HTTP 401: invalid API key"],
        }
        with tempfile.TemporaryDirectory() as directory:
            item_path = Path(directory) / "sample-1.json"
            item_path.write_text(json.dumps(record), encoding="utf-8")
            reused = qwen.read_terminal_failure(
                item_path,
                sample,
                "prompt-hash",
                "model",
                "https://example/v1",
                "audio_only",
            )
        self.assertEqual(reused["status"], "failed")

    def test_network_failure_checkpoint_is_not_terminal(self):
        sample = {
            "sample_id": "sample-1",
            "audio_sha256": "audio-hash",
        }
        record = {
            "sample_id": "sample-1",
            "audio_sha256": "audio-hash",
            "prompt_sha256": "prompt-hash",
            "model": "model",
            "endpoint": "https://example/v1",
            "input_contract": "audio_only",
            "status": "failed",
            "errors": ["connection reset by peer"],
        }
        with tempfile.TemporaryDirectory() as directory:
            item_path = Path(directory) / "sample-1.json"
            item_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(
                qwen.read_terminal_failure(
                    item_path,
                    sample,
                    "prompt-hash",
                    "model",
                    "https://example/v1",
                    "audio_only",
                )
            )


if __name__ == "__main__":
    unittest.main()
