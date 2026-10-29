"""Versioned WUST subprocess boundary; no numerical code or CPU fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


class RuntimeFailure(RuntimeError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class Deadline:
    def __init__(self, seconds, *, started=None):
        if seconds is None:
            raise ValueError("fwi_wust requires an explicit elapsed-time budget")
        self.end = (time.monotonic() if started is None else started) + seconds

    def remaining(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise RuntimeFailure("time_budget", "Global FWI deadline exhausted")
        return remaining


class WUSTClient:
    def __init__(self, parameters, deadline, folder):
        self.parameters = parameters
        self.deadline = deadline
        self.folder = Path(folder)
        root = parameters.runtime_root or os.environ.get("USCT_WUST_ROOT")
        if not root:
            raise RuntimeFailure("incompatible_runtime", "Set USCT_WUST_ROOT")
        self.root = Path(root).resolve(strict=True)
        self.launcher = self.root / "Runtime/python/wust_runtime.py"
        self.counter = 0

    def command(self, arguments):
        self.counter += 1
        log = self.folder / f"runtime-{self.counter}.stderr.log"
        env = os.environ.copy()
        if self.parameters.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.parameters.cuda_visible_devices
        cmd = [sys.executable, str(self.launcher), *arguments]
        remaining = self.deadline.remaining()
        handlers = {}

        def interrupted(signum, _frame):
            raise RuntimeFailure(
                "process_termination", f"BenchLab received signal {signum}"
            )

        with log.open("w") as stream:
            child = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stream,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                if threading.current_thread() is threading.main_thread():
                    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                        handlers[signum] = signal.signal(signum, interrupted)
                stdout, _ = child.communicate(timeout=remaining)
            except BaseException as exc:
                for signum in handlers:
                    signal.signal(signum, signal.SIG_IGN)
                # WUST owns MATLAB's separate process group. Give its SIGTERM
                # handler time to reap that group before killing the launcher.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.communicate()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeFailure(
                        "time_budget", "WUST deadline; process cleanup requested"
                    ) from exc
                raise
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        if child.returncode:
            detail = {}
            for line in log.read_text().splitlines():
                try:
                    value = json.loads(line)
                    if value.get("schema") == "wust.execution_failure":
                        detail = value
                except (ValueError, AttributeError):
                    pass
            raise RuntimeFailure(
                detail.get("reason", "runtime_failure"),
                detail.get("message", f"WUST exit {child.returncode}; see {log}"),
            )
        self.deadline.remaining()
        try:
            return json.loads(stdout)
        except ValueError as exc:
            raise RuntimeFailure(
                "incompatible_runtime", "WUST stdout is not clean JSON"
            ) from exc

    def admit(self):
        info = self.command(["describe", "--json"])
        if (
            info.get("schema") != "wust.capabilities"
            or info.get("schema_version") != 2
            or "reconstruct" not in info.get("operations", [])
            or "wust.frequency_input"
            not in info.get("input_schemas", {}).get("reconstruct", [])
            or self.parameters.backend not in info.get("profiles", {})
        ):
            raise RuntimeFailure(
                "incompatible_runtime", "WUST v2 reconstruction capability required"
            )
        return info

    def invoke(self, request):
        path = self.folder / f"{request['operation']}.request.json"
        path.write_text(json.dumps(request, allow_nan=False, indent=2))
        return self.command(
            [
                "run",
                str(path),
                "--matlab",
                self.parameters.matlab_executable,
                "--timeout-s",
                str(self.deadline.remaining()),
            ]
        )
