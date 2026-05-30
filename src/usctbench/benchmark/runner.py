"""Benchmark run and evaluation helpers."""

from __future__ import annotations

import csv
import glob
import json
import math
import resource
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from usctbench.benchmark.report import write_failure_report
from usctbench.io.hdf5 import read_case_hdf5, write_result_hdf5
from usctbench.registry import get_algorithm
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus
from usctbench.viz.preview import write_preview_png


def load_algorithm_config(path: str | Path) -> AlgorithmConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("algorithm config must be a YAML mapping")
    parameters = payload.get("parameters", {})
    metadata = payload.get("metadata", {})
    if not isinstance(parameters, dict):
        raise ValueError("config parameters must be a mapping")
    if not isinstance(metadata, dict):
        raise ValueError("config metadata must be a mapping")
    return AlgorithmConfig(name=payload.get("name") or payload.get("algorithm"), parameters=parameters, metadata=metadata)


def run_algorithm_case(
    algorithm_name: str,
    case_path: str | Path,
    config_path: str | Path,
    out_root: str | Path,
) -> Path:
    """Run one algorithm on one case and write the standard artifact set."""

    out_root = Path(out_root)
    case_id = Path(case_path).stem
    algorithm_for_report = algorithm_name
    config_for_report = str(config_path)
    memory_before_mb = _peak_memory_mb()
    try:
        case = read_case_hdf5(case_path)
        case_id = case.case_id
        config = load_algorithm_config(config_path)
        algorithm = get_algorithm(algorithm_name)
        result = algorithm.run(case, config)
    except Exception as exc:
        result = ReconstructionResult(
            algorithm=algorithm_for_report,
            case_id=case_id,
            runtime_s=0.0,
            status=ResultStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    memory_after_mb = _peak_memory_mb()

    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_result_artifacts(
        result,
        out_dir,
        config=config_for_report,
        peak_memory_mb=max(memory_before_mb, memory_after_mb),
    )
    return out_dir


def evaluate_run(run_dir: str | Path, protocol_path: str | Path | None = None) -> dict[str, Any]:
    """Aggregate per-case metrics into CSV and markdown reports."""

    run_path = Path(run_dir)
    protocol = _load_yaml(protocol_path) if protocol_path is not None else {}
    records = []
    for metrics_path in sorted(run_path.rglob("metrics.json")):
        case_dir = metrics_path.parent
        metadata_path = case_dir / "metadata.yaml"
        metadata = _load_yaml(metadata_path) if metadata_path.exists() else {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record = {
            "case_id": metadata.get("case_id", case_dir.name),
            "algorithm": metadata.get("algorithm", case_dir.parent.name),
            "status": metadata.get("status", "unknown"),
            "runtime_s": metadata.get("runtime_s", ""),
            "peak_memory_mb": metadata.get("peak_memory_mb", ""),
            "failure_reason": metadata.get("failure_reason") or "",
            "failure_report_present": (case_dir / "failure_report.md").exists(),
        }
        record.update(metrics)
        record["artifacts_complete"], artifact_reasons = _artifact_check(case_dir, record)
        record["pass"], pass_reasons, fail_reasons = _assess_record(record, protocol, artifact_reasons)
        record["pass_reasons"] = "; ".join(pass_reasons)
        record["fail_reasons"] = "; ".join(fail_reasons)
        records.append(record)

    summary_csv = run_path / "benchmark_summary.csv"
    _write_summary_csv(records, summary_csv)
    report_md = run_path / "benchmark_report.md"
    _write_benchmark_report(records, report_md, protocol=protocol)
    return {"records": records, "summary_csv": str(summary_csv), "report_md": str(report_md)}


def run_benchmark_suite(suite_path: str | Path) -> dict[str, Any]:
    """Run all algorithms listed in a benchmark suite YAML."""

    suite_file = Path(suite_path)
    suite = _load_yaml(suite_file)
    suite_name = suite.get("name", suite_file.stem)
    case_glob = _expand(str(suite["case_glob"]))
    cases = sorted(Path(path) for path in glob.glob(case_glob, recursive=True))
    output_root = Path(_expand(str(suite.get("outputs", {}).get("root", "runs/usctbench_runs"))))
    run_id = suite.get("run_id") or f"{suite_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = output_root / run_id

    algorithms = suite.get("algorithms", [])
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("benchmark suite must define a non-empty algorithms list")

    for algorithm in algorithms:
        algorithm_name = algorithm["name"]
        config_path = Path(_expand(str(algorithm["config"])))
        if not config_path.is_absolute():
            config_path = suite_file.parent.parent.parent / config_path if suite_file.parent.name == "benchmarks" else Path.cwd() / config_path
        for case_path in cases:
            run_algorithm_case(algorithm_name, case_path, config_path, run_root / algorithm_name)

    evaluation = evaluate_run(run_root, suite_file)
    return {"run_root": str(run_root), "num_cases": len(cases), **evaluation}


def _write_result_artifacts(result: ReconstructionResult, out_dir: Path, *, config: str, peak_memory_mb: float) -> None:
    preview_image = result.sound_speed_mps if result.sound_speed_mps is not None else result.attenuation_np_per_m
    if preview_image is not None:
        preview_path = write_preview_png(preview_image, out_dir / "preview.png")
        result.artifacts.setdefault("preview", str(preview_path))

    result_path = write_result_hdf5(result, out_dir / "result.h5")
    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": result.algorithm,
                "case_id": result.case_id,
                "config": config,
                "runtime_s": result.runtime_s,
                "peak_memory_mb": peak_memory_mb,
                "status": str(result.status),
                "failure_reason": result.failure_reason,
                "result_h5": str(result_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if result.status != ResultStatus.SUCCESS:
        write_failure_report(
            out_dir / "failure_report.md",
            algorithm=result.algorithm,
            case_id=result.case_id,
            config=config,
            error_type="unknown",
            symptom=result.failure_reason or "algorithm did not report success",
            likely_causes=["schema mismatch", "data/geometry mismatch", "numerical instability"],
            actions=["inspect case metadata", "inspect sinogram/features", "lower relaxation or iterations"],
        )


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _expand(value: str) -> str:
    import os

    return os.path.expandvars(os.path.expanduser(value))


def _assess_record(
    record: dict[str, Any],
    protocol: dict[str, Any],
    artifact_reasons: list[str],
) -> tuple[bool, list[str], list[str]]:
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    if str(record.get("status", "")).lower() != "success":
        fail_reasons.append(f"status is {record.get('status')}")
        if record.get("failure_reason"):
            fail_reasons.append(str(record["failure_reason"]))
    else:
        pass_reasons.append("status is success")

    if artifact_reasons:
        fail_reasons.extend(artifact_reasons)
    else:
        pass_reasons.append("required artifacts present")

    thresholds = protocol.get("thresholds", {})
    if isinstance(thresholds, dict):
        for key, limit in thresholds.items():
            if key in record and _is_number(record[key]):
                value = float(record[key])
                if value > float(limit):
                    fail_reasons.append(f"{key}={value:g} exceeds max {float(limit):g}")
                else:
                    pass_reasons.append(f"{key}={value:g} <= {float(limit):g}")

    minimums = protocol.get("minimums", {})
    if isinstance(minimums, dict):
        for key, limit in minimums.items():
            if key in record and _is_number(record[key]):
                value = float(record[key])
                if value < float(limit):
                    fail_reasons.append(f"{key}={value:g} below min {float(limit):g}")
                else:
                    pass_reasons.append(f"{key}={value:g} >= {float(limit):g}")

    return not fail_reasons, pass_reasons, fail_reasons


def _artifact_check(case_dir: Path, record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    required = {
        "result.h5": case_dir / "result.h5",
        "metrics.json": case_dir / "metrics.json",
        "metadata.yaml": case_dir / "metadata.yaml",
    }
    for label, path in required.items():
        if not path.exists():
            reasons.append(f"missing {label}")
    if str(record.get("status", "")).lower() == "success" and not (case_dir / "preview.png").exists():
        reasons.append("missing preview.png for successful run")
    if str(record.get("status", "")).lower() != "success" and not (case_dir / "failure_report.md").exists():
        reasons.append("missing failure_report.md for non-success run")
    return not reasons, reasons


def _write_summary_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    preferred = [
        "algorithm",
        "case_id",
        "status",
        "pass",
        "pass_reasons",
        "fail_reasons",
        "runtime_s",
        "peak_memory_mb",
        "artifacts_complete",
        "failure_report_present",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_benchmark_report(records: list[dict[str, Any]], path: Path, *, protocol: dict[str, Any]) -> None:
    passed = sum(1 for record in records if record.get("pass"))
    runtimes = [float(record["runtime_s"]) for record in records if _is_number(record.get("runtime_s"))]
    memory_values = [float(record["peak_memory_mb"]) for record in records if _is_number(record.get("peak_memory_mb"))]
    lines = [
        "# Benchmark report",
        "",
        f"- Records: {len(records)}",
        f"- Passed: {passed}",
        f"- Failed: {len(records) - passed}",
        f"- Protocol: {protocol.get('name', 'ad hoc')}",
        f"- Runtime total seconds: {sum(runtimes):.6g}" if runtimes else "- Runtime total seconds:",
        f"- Runtime max seconds: {max(runtimes):.6g}" if runtimes else "- Runtime max seconds:",
        f"- Peak memory max MB: {max(memory_values):.6g}" if memory_values else "- Peak memory max MB:",
        "",
        "## Results",
        "",
    ]
    for record in records:
        lines.append(
            f"- `{record.get('algorithm')}` / `{record.get('case_id')}`: "
            f"status={record.get('status')}, pass={record.get('pass')}, runtime_s={record.get('runtime_s')}, "
            f"reasons={record.get('fail_reasons') or record.get('pass_reasons')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if usage > 10_000_000:
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
