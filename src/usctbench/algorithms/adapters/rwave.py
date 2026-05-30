"""Optional ray-Born/r-Wave MATLAB adapter."""

from __future__ import annotations

from pathlib import Path

from usctbench.adapters.matlab import MatlabAdapter, MatlabUnavailable
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


class RWaveAdapter:
    """Callable placeholder for an external ray-Born/r-Wave implementation."""

    name = "rwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        params = config.parameters
        try:
            adapter = MatlabAdapter.from_config(
                matlab_bin=params.get("matlab_bin"),
                work_dir=params.get("work_dir", "."),
            )
        except MatlabUnavailable as exc:
            return _skipped(self.name, case, str(exc))

        entrypoint = params.get("entrypoint")
        external_root = params.get("external_root")
        if not entrypoint or not external_root:
            return _skipped(
                self.name,
                case,
                "r-Wave adapter requires parameters.external_root and parameters.entrypoint",
            )

        entry_path = Path(external_root).expanduser() / str(entrypoint)
        if not entry_path.exists():
            return _skipped(self.name, case, f"r-Wave entrypoint not found: {entry_path}")

        code = f"addpath(genpath('{Path(external_root).expanduser()}')); disp('USCT rwave_adapter configured');"
        try:
            log_path = adapter.run_batch(code, log_name="rwave_matlab.log", timeout_s=int(params.get("timeout_s", 300)))
        except MatlabUnavailable as exc:
            return _skipped(self.name, case, str(exc))

        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason="External r-Wave entrypoint is configured, but data marshaling is not implemented yet.",
            artifacts={"matlab_log": str(log_path), "external_entrypoint": str(entry_path)},
        )


def _skipped(algorithm: str, case: USCTCase, reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=reason,
    )

