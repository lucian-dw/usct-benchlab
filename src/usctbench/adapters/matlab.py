"""MATLAB adapter utilities with explicit graceful-skip behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MatlabUnavailable(RuntimeError):
    """Raised when a MATLAB-backed adapter cannot be executed."""


def find_matlab(configured_bin: str | None = None) -> str | None:
    """Find a MATLAB executable from config, environment, or PATH."""

    candidates = [configured_bin, os.environ.get("MATLAB_BIN"), shutil.which("matlab")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
        if candidate and shutil.which(candidate):
            return str(shutil.which(candidate))
    return None


@dataclass(frozen=True)
class MatlabAdapter:
    """Small wrapper around `matlab -batch` for optional classic methods."""

    matlab_bin: str
    work_dir: Path

    @classmethod
    def from_config(cls, *, matlab_bin: str | None = None, work_dir: str | Path | None = None) -> "MatlabAdapter":
        resolved = find_matlab(matlab_bin)
        if resolved is None:
            raise MatlabUnavailable("MATLAB executable not found; set MATLAB_BIN or parameters.matlab_bin")
        return cls(matlab_bin=resolved, work_dir=Path(work_dir or ".").resolve())

    def run_batch(self, code: str, *, log_name: str = "matlab.log", timeout_s: int | None = None) -> Path:
        """Run MATLAB batch code and save stdout/stderr to a log file."""

        self.work_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.work_dir / log_name
        completed = subprocess.run(
            [self.matlab_bin, "-batch", code],
            cwd=self.work_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise MatlabUnavailable(f"MATLAB command failed with exit code {completed.returncode}; see {log_path}")
        return log_path

