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
        Path("scripts/run_v01_release_check.sh"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert 'if [ -f "$REPO_DIR/.env" ]; then' in text
        assert '. "$REPO_DIR/.env"' in text


def test_v01_release_check_runs_full_evidence_chain():
    text = Path("scripts/run_v01_release_check.sh").read_text(encoding="utf-8")

    assert "pytest -q" in text
    assert "inspect-openbreastus" in text
    assert "make-smoke" in text
    assert "usctbench.cli bench" in text
    assert "--require-v01-dod" in text
    assert "--openbreastus-index" in text
    assert "--smoke-manifest" in text
