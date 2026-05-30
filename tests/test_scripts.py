from __future__ import annotations

from pathlib import Path


def test_run_smoke_runs_benchmark_when_cases_exist():
    text = Path("scripts/run_smoke.sh").read_text(encoding="utf-8")

    assert "usctbench.cli bench" in text
    assert "audit_v01_readiness.py" in text
    assert "USCT_SAMPLE_ROOT/cases" in text

