"""Prepare, reconstruct, persist and optionally evaluate one explicit request."""

import glob
import json
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone

from usctbench.benchmark.artifacts import publish, save_arrays, write_json
from usctbench.benchmark.report import write_summary
from usctbench.core.config import expand_config_value, load_yaml, prepare_config
from usctbench.core.io import read_case_hdf5
from usctbench.core.provenance import case_measurement_metadata
from usctbench.core.registry import get_algorithm
from usctbench.core.schema import (
    GroundTruthSpec,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)
from usctbench.metrics import compute_regional_image_metrics
from usctbench.viz import write_preview_png


def run(
    algorithm,
    case,
    config,
    out=None,
    *,
    host_limits=None,
    deployment=None,
    score=True,
    preview=True,
):
    """Return the result; optional ``out`` is the exact case output directory.

    This runner owns failure records and strips GT before numerical execution.
    Postprocessing can fail without destroying an already saved numerical result.
    """
    started = time.perf_counter()
    case_id = case.case_id if isinstance(case, USCTCase) else Path(case).stem
    prepared = None
    stages = {}
    stage = "prepare"
    measurement = {}
    directory = Path(out) if out is not None else None
    try:
        case = case if isinstance(case, USCTCase) else read_case_hdf5(case)
        case_id = case.case_id
        measurement = case_measurement_metadata(case.metadata)
        prepared = prepare_config(
            algorithm, config, case=case, host_limits=host_limits, deployment=deployment
        )
        if directory is not None and prepared.deployment.scratch_root is None:
            prepared.deployment = prepared.deployment.model_copy(
                update={"scratch_root": str(directory)}
            )
        stages[stage] = {"status": "completed"}
        stage = "solve"
        result = get_algorithm(algorithm).run(
            case.model_copy(update={"ground_truth": GroundTruthSpec()}), prepared
        )
        stages[stage] = {"status": str(result.status)}
        if result.failure_reason:
            stages[stage]["message"] = result.failure_reason
    except Exception as exc:
        stages[stage] = {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}
        result = ReconstructionResult(
            algorithm=algorithm,
            case_id=case_id,
            status=ResultStatus.FAILED,
            failure_reason=stages[stage]["message"],
        )
    result.runtime_s = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = float(peak) / (1024**2 if sys.platform == "darwin" else 1024)
    result.metrics["stages"] = stages
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            save_arrays(result, directory)
            stages["save"] = {"status": "completed"}
        except Exception as exc:
            stages["save"] = {
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
            result.status = ResultStatus.FAILED
            result.failure_reason = stages["save"]["message"]
        publish(
            result,
            directory,
            config=prepared,
            stages=stages,
            peak_rss_mib=peak_mib,
            measurement=measurement,
        )
    if result.sound_speed_mps is not None and result.status != ResultStatus.FAILED:
        if score:
            try:
                result.metrics.update(
                    compute_regional_image_metrics(
                        result.sound_speed_mps,
                        case.ground_truth.sound_speed_mps,
                        **prepared.image_evaluation.model_dump(exclude_none=True),
                    )
                )
                stages["evaluation"] = {"status": "completed"}
            except Exception as exc:
                stages["evaluation"] = {
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
        if preview and directory is not None:
            try:
                result.artifacts["preview"] = str(
                    write_preview_png(result.sound_speed_mps, directory / "preview.png")
                )
                stages["preview"] = {"status": "completed"}
            except Exception as exc:
                stages["preview"] = {
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
    if directory is not None:
        try:
            write_json(directory / "metrics.json", result.metrics)
        except (ValueError, TypeError, OSError) as exc:
            stages["evaluation"] = {
                "status": "failed",
                "message": f"metrics serialization: {exc}",
            }
        publish(
            result,
            directory,
            config=prepared,
            stages=stages,
            peak_rss_mib=peak_mib,
            measurement=measurement,
        )
    return result


def evaluate(directory):
    """Summarize saved outcomes without rerunning a model or inventing scores."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("run.json")):
        record = json.loads(path.read_text())
        metrics_path = path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        failures = [
            (key, value.get("message"))
            for key, value in record["stages"].items()
            if value["status"] == "failed"
        ]
        row = {
            key: record.get(key)
            for key in ("algorithm", "case_id", "status", "stop_reason", "runtime_s")
        }
        row.update(
            numerical_result_available=record["numerical_result_available"],
            measurement_provenance=record["measurement"].get(
                "measurement_provenance", "unknown"
            ),
            forward_model=metrics.get("forward_model"),
            observation_domain=metrics.get("observation_domain"),
            observation_units=metrics.get("observation_units"),
            primary_image_region=metrics.get("primary_image_region"),
            failure_stage=failures[0][0] if failures else None,
            message=failures[0][1] if failures else record.get("failure_reason"),
        )
        for source, target in (
            ("rmse", "rmse_mps"),
            ("psnr", "psnr_db"),
            ("ssim", "ssim"),
            ("full_image_rmse", "full_image_rmse_mps"),
            ("full_image_psnr", "full_image_psnr_db"),
            ("full_image_ssim", "full_image_ssim"),
            ("data_relative_residual", "data_relative_residual"),
        ):
            row[target] = metrics.get(source)
        records.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "records": records,
        "passed": bool(records)
        and all(
            row["status"] != "failed"
            and row["numerical_result_available"]
            and not row["failure_stage"]
            for row in records
        ),
        **write_summary(records, directory),
    }


def suite(path):
    """Resolve every relative path against the suite file, never the current cwd."""
    path = Path(path).resolve()
    definition = expand_config_value(load_yaml(path))

    def relative(value):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else path.parent / candidate

    cases = sorted(glob.glob(str(relative(definition["case_glob"])), recursive=True))
    if not cases:
        raise ValueError(f"benchmark suite matched no cases: {definition['case_glob']}")
    algorithms = definition.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("benchmark suite requires a nonempty algorithms list")
    run_id = (
        definition.get("run_id")
        or f"{definition.get('name', path.stem)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    root = relative(definition.get("outputs", {}).get("root", "../../runs")) / run_id
    configs = [
        (item["name"], expand_config_value(load_yaml(relative(item["config"]))))
        for item in algorithms
    ]
    for case_path in cases:
        case = read_case_hdf5(case_path)
        for name, config in configs:
            run(
                name,
                case,
                config,
                root / name / case.case_id,
                host_limits=definition.get("host_limits"),
                deployment=definition.get("deployment"),
            )
    return {"run_root": str(root), "num_cases": len(cases), **evaluate(root)}
