"""Optional refraction-corrected Gauss-Newton MATLAB adapter."""

from __future__ import annotations

from pathlib import Path

from usctbench.adapters.matlab import MatlabAdapter, MatlabUnavailable
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


class BentRayGNAdapter:
    """Callable placeholder for established MATLAB bent-ray GN code.

    The adapter is intentionally conservative: if MATLAB or the external
    entrypoint is unavailable, it returns `skipped` instead of crashing or
    pretending to reconstruct.
    """

    name = "bent_ray_gn"

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
                "MATLAB refraction GN adapter requires parameters.external_root and parameters.entrypoint",
            )

        entry_path = Path(external_root).expanduser() / str(entrypoint)
        if not entry_path.exists():
            return _skipped(self.name, case, f"MATLAB refraction GN entrypoint not found: {entry_path}")

        # Execution is deliberately opt-in because external packages have
        # different input conventions. The generated log still proves the
        # adapter can call MATLAB when a user supplies an entrypoint.
        code = f"addpath(genpath('{Path(external_root).expanduser()}')); disp('USCT bent_ray_gn adapter configured');"
        try:
            log_path = adapter.run_batch(code, log_name="bent_ray_gn_matlab.log", timeout_s=int(params.get("timeout_s", 300)))
        except MatlabUnavailable as exc:
            return _skipped(self.name, case, str(exc))

        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason="External MATLAB entrypoint is configured, but data marshaling is not implemented yet.",
            artifacts={"matlab_log": str(log_path), "external_entrypoint": str(entry_path)},
        )


def _skipped(algorithm: str, case: USCTCase, reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=reason,
        metrics={"adapter_dependency_available": False},
        artifacts={"adapter_status": "skipped", "skip_reason": reason},
    )
