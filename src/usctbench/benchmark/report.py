"""Failure-report helpers for benchmark runs."""

from __future__ import annotations

from pathlib import Path


def write_failure_report(
    path: str | Path,
    *,
    algorithm: str,
    case_id: str,
    config: str,
    error_type: str,
    symptom: str,
    likely_causes: list[str] | None = None,
    actions: list[str] | None = None,
    logs: list[str] | None = None,
    plots: list[str] | None = None,
) -> Path:
    """Write the standard markdown failure report."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    likely_causes = likely_causes or []
    actions = actions or []
    logs = logs or []
    plots = plots or []
    text = [
        "# Failure report",
        "",
        f"- Algorithm: {algorithm}",
        f"- Case id: {case_id}",
        f"- Config: {config}",
        f"- Error type: {error_type}",
        f"- Symptom: {symptom}",
        f"- Most likely causes: {_format_list(likely_causes)}",
        f"- First three actions to try: {_format_list(actions[:3])}",
        f"- Logs: {_format_list(logs)}",
        f"- Plots: {_format_list(plots)}",
        "",
    ]
    out.write_text("\n".join(text), encoding="utf-8")
    return out


def _format_list(values: list[str]) -> str:
    if not values:
        return ""
    return "; ".join(values)
