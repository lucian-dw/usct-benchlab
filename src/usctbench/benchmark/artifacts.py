"""Numerical arrays first, run record second, successful index last."""

from pathlib import Path
from dataclasses import dataclass, field
import json

from usctbench.core.io import write_result_hdf5
from usctbench.core.schema import ResultStatus


@dataclass(frozen=True)
class SavedArrays:
    artifacts: dict[str, str] = field(default_factory=dict)
    numerical_result_available: bool = False


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_arrays(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate generated reporting artifacts before replacing this run's arrays.
    for name in ("index.json", "metrics.json", "preview.png"):
        (directory / name).unlink(missing_ok=True)
    path = directory / "result.h5"
    temporary = directory / ".result.h5.tmp"
    try:
        write_result_hdf5(result, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return SavedArrays(
        artifacts={"result_h5": str(path)},
        numerical_result_available=result.sound_speed_mps is not None,
    )


def publish(result, directory, *, saved, config, stages, peak_rss_mib, measurement):
    stop = result.metrics.get("stopping", {})
    record = {
        "algorithm": result.algorithm,
        "experiment_id": result.experiment_id,
        "variant": result.variant,
        "case_id": result.case_id,
        "status": str(result.status),
        "stop_reason": result.metrics.get("stop_reason"),
        "runtime_s": result.runtime_s,
        "process_peak_rss_mib": peak_rss_mib,
        "failure_reason": result.failure_reason,
        "numerical_result_available": saved.numerical_result_available,
        "stages": stages,
        "config": config.model_dump(mode="json") if config is not None else None,
        "completed_iterations": stop.get(
            "completed_iterations", result.metrics.get("completed_iterations")
        ),
        "work": stop.get("work"),
        "measurement": measurement,
        "artifacts": result.artifacts,
    }
    write_json(Path(directory) / "run.json", record)
    if result.status != ResultStatus.FAILED and saved.numerical_result_available:
        write_json(
            Path(directory) / "index.json", {"run": "run.json", "arrays": "result.h5"}
        )
    return record
