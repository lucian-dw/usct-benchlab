from __future__ import annotations

from pathlib import Path


def test_run_smoke_runs_benchmark_when_cases_exist():
    text = Path("scripts/run_smoke.sh").read_text(encoding="utf-8")

    assert "usctbench.cli bench" in text
    assert "audit_v01_readiness.py" in text
    assert "USCT_SAMPLE_ROOT/cases" in text
    assert "USCT_REQUIRE_SMOKE_CASES" in text


def test_shell_scripts_source_local_env():
    for path in [
        Path("scripts/setup_workspace.sh"),
        Path("scripts/check_server.sh"),
        Path("scripts/bootstrap_a100.sh"),
        Path("scripts/run_smoke.sh"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert 'if [ -f "$REPO_DIR/.env" ]; then' in text
        assert '. "$REPO_DIR/.env"' in text
