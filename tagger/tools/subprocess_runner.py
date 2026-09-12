"""Helpers for running model tools in tool-specific Python subprocesses."""

import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SubprocessToolError(RuntimeError):
    """Raised when a subprocess-backed tool cannot return a valid result."""


class JsonSubprocessWorker:
    def __init__(self, python_executable, tool_name):
        self.python_executable = _resolve_python_executable(python_executable)
        self.tool_name = tool_name
        self.broken = False
        self._lock = threading.Lock()
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT)
        if env.get("PYTHONPATH"):
            pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath
        _add_runtime_cuda_library_path(env, self.python_executable)
        self.process = subprocess.Popen(
            [
                self.python_executable,
                "-m",
                "tagger.tools.subprocess_worker",
                self.tool_name,
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            universal_newlines=True,
            bufsize=1,
        )

    def call(self, request, timeout_sec=None):
        # A worker owns one stdin/stdout JSONL stream. Calls may arrive from
        # parallel sample tasks, but each request/response pair must remain
        # ordered on that stream.
        with self._lock:
            return self._call_locked(request, timeout_sec=timeout_sec)

    def _call_locked(self, request, timeout_sec=None):
        if self.process.poll() is not None:
            self.broken = True
            raise SubprocessToolError(
                "subprocess worker exited before request: %s" % self.tool_name
            )
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = None
        if timeout_sec is not None:
            timeout_sec = float(timeout_sec)
            if timeout_sec <= 0:
                raise ValueError("subprocess timeout must be positive")
            deadline = time.monotonic() + timeout_sec
        response = self._read_response(deadline)
        if response.get("status") != "ok":
            message = response.get("message") or "subprocess tool failed"
            error_type = response.get("error_type")
            if error_type:
                message = "%s: %s" % (error_type, message)
            raise SubprocessToolError(message)
        return response.get("result")

    def close(self):
        with self._lock:
            if self.process.poll() is not None:
                return
            try:
                self.process.stdin.write(json.dumps({"_command": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    self.process.kill()

    def _read_response(self, deadline=None):
        noise = []
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._mark_broken()
                    raise SubprocessToolError(
                        "subprocess worker timed out for %s" % self.tool_name
                    )
                readable, _writable, _exceptional = select.select(
                    [self.process.stdout], [], [], remaining
                )
                if not readable:
                    self._mark_broken()
                    raise SubprocessToolError(
                        "subprocess worker timed out for %s" % self.tool_name
                    )
            line = self.process.stdout.readline()
            if not line:
                self.broken = True
                code = self.process.poll()
                raise SubprocessToolError(
                    "subprocess worker produced no JSON response for %s; "
                    "returncode=%s; stdout_noise=%s"
                    % (self.tool_name, code, " | ".join(noise[-5:]))
                )
            stripped = line.strip()
            try:
                response = json.loads(stripped)
            except ValueError:
                noise.append(stripped)
                continue
            if isinstance(response, dict) and "status" in response:
                return response
            noise.append(stripped)

    def _mark_broken(self):
        self.broken = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()


def run_subprocess_tool(
    python_executable,
    tool_name,
    request,
    context=None,
    timeout_sec=None,
):
    if not python_executable:
        raise SubprocessToolError("subprocess python is not configured")
    if context is None:
        worker = JsonSubprocessWorker(python_executable, tool_name)
        try:
            return worker.call(request, timeout_sec=timeout_sec)
        finally:
            worker.close()

    key = (_resolve_python_executable(python_executable), tool_name)
    workers_lock = context.setdefault("_subprocess_workers_lock", threading.Lock())
    with workers_lock:
        workers = context.setdefault("_subprocess_workers", {})
        worker_pool = workers.setdefault(key, [])
        slots_by_tool = context.get("_subprocess_worker_slots", {})
        slots = max(1, int(slots_by_tool.get(tool_name, 1)))
        while len(worker_pool) < slots:
            worker_pool.append(JsonSubprocessWorker(python_executable, tool_name))
        # Round-robin requests across the pool. Each worker keeps its own
        # ordered JSON stream, and the per-worker lock protects that stream.
        next_by_key = context.setdefault("_subprocess_worker_next", {})
        index = int(next_by_key.get(key, 0)) % len(worker_pool)
        next_by_key[key] = index + 1
        worker = worker_pool[index]
    try:
        return worker.call(request, timeout_sec=timeout_sec)
    except Exception:
        if worker.broken:
            with workers_lock:
                pool = workers.get(key, [])
                workers[key] = [item for item in pool if item is not worker]
            worker.close()
        raise


def close_subprocess_workers(context):
    if not context:
        return
    workers_lock = context.setdefault("_subprocess_workers_lock", threading.Lock())
    with workers_lock:
        workers = context.pop("_subprocess_workers", {})
    for pool in workers.values():
        for worker in pool:
            worker.close()


def _resolve_python_executable(python_executable):
    path = Path(str(python_executable)).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _add_runtime_cuda_library_path(env, python_executable):
    """Expose CUDA libraries shipped by the selected subprocess runtime.

    Each model runtime carries its own CUDA/cuDNN wheel set.  Adding only the
    libraries belonging to the selected executable avoids ABI conflicts between
    the Python 3.10 and 3.11 environments used by the pipeline.
    """
    executable = Path(str(python_executable)).resolve()
    runtime_root = executable.parent.parent
    site_candidates = list(
        (runtime_root / "lib").glob("python*/site-packages")
    )
    library_dirs = []
    for site in site_candidates:
        nvidia_root = site / "nvidia"
        if nvidia_root.is_dir():
            library_dirs.extend(
                str(path)
                for path in sorted(nvidia_root.glob("*/lib"))
                if path.is_dir()
            )
        for pth in site.glob("*.pth"):
            try:
                candidates = [
                    Path(line.strip())
                    for line in pth.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("import ")
                ]
            except OSError:
                continue
            for external_site in candidates:
                nvidia_root = external_site / "nvidia"
                if nvidia_root.is_dir():
                    library_dirs.extend(
                        str(path)
                        for path in sorted(nvidia_root.glob("*/lib"))
                        if path.is_dir()
                    )
    if not library_dirs:
        return
    existing = env.get("LD_LIBRARY_PATH", "")
    merged = library_dirs + ([existing] if existing else [])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(merged)
