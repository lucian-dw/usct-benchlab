"""One production FWI path: USCTCase -> WUST v2 -> final sound speed."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import numpy as np

from usctbench.core.config import prepared_run
from usctbench.core.schema import ReconstructionResult, ResultStatus

from .io import frequency_input, read_artifact, write_artifact
from .runtime import Deadline, RuntimeFailure, WUSTClient


def initialization(case, p):
    if p.sound_speed_bounds_mps is None or p.pml_m is None:
        raise ValueError(
            "Explicit sound_speed_bounds_mps and pml_m required for execution"
        )
    if p.initialization == "map":
        initial = np.asarray(p.initial_map_mps, dtype=np.float64)
    else:
        value = (
            p.initial_sound_speed_mps
            if p.initialization == "scalar"
            else case.metadata.get("reference_sound_speed_mps")
        )
        if value is None:
            raise ValueError(
                "Reference initialization requires declared reference_sound_speed_mps"
            )
        initial = np.full(case.grid.shape, value, dtype=np.float64)
    low, high = p.sound_speed_bounds_mps
    if (
        initial.shape != case.grid.shape
        or not np.isfinite(initial).all()
        or np.any((initial < low) | (initial > high))
    ):
        raise ValueError(
            "Initialization must match exact grid and bounds; no resize/clipping"
        )
    if p.update_mask is not None:
        mask = np.asarray(p.update_mask, dtype=bool)
        policy = "explicit_expert_mask"
    else:
        # BenchLab's conservative physical-region policy, not a copy of WUST's
        # snapping or PML validator. WUST independently admits the requested mask.
        ny, nx = case.grid.shape
        dy, dx = case.grid.spacing_m
        clearance = p.pml_m + max(dy, dx)
        y = (np.arange(ny) + 0.5) * dy
        x = (np.arange(nx) + 0.5) * dx
        mask = ((y > clearance) & (ny * dy - y > clearance))[:, None] & (
            (x > clearance) & (nx * dx - x > clearance)
        )[None, :]
        policy = "edge_clearance_pml_plus_one_max_spacing"
    if mask.shape != case.grid.shape or not mask.any():
        raise ValueError(
            "Update mask must match grid and contain requested update pixels"
        )
    return initial, mask, policy


def prepare_request(case, p, folder, max_updates, backend):
    initial, mask, mask_policy = initialization(case, p)
    input_path = folder / "frequency.json"
    initial_path = folder / "initial.json"
    frequency_input(case, input_path)
    write_artifact(
        initial_path,
        "wust.initial_model",
        {"initialization": p.initialization, "mask_policy": mask_policy},
        {"initial_mps": initial, "update_mask": mask},
        {"initial_mps": "y,x", "update_mask": "y,x"},
        {"initial_mps": "m/s", "update_mask": "1"},
    )
    numerical = {
        key: getattr(p, key)
        for key in (
            "max_update_mps",
            "step_damping",
            "source_batch_size",
            "pml_strength",
            "pml_m",
            "filter_cutoff",
            "filter_order",
        )
    }
    numerical.update(
        backend=backend,
        bounds_mps=list(p.sound_speed_bounds_mps),
        stencil_bounds=list(p.stencil_bounds_mps or p.sound_speed_bounds_mps),
        wavenumber=p.wavenumber_model,
    )
    if p.dispersion is not None:
        numerical["dispersion"] = p.dispersion
    return {
        "schema": "wust.request",
        "schema_version": 2,
        "operation": "reconstruct",
        "input_manifest": str(input_path),
        "initial_manifest": str(initial_path),
        "output_manifest": str(folder / "reconstruction.json"),
        "config": numerical,
        "frequency_schedule_hz": p.frequency_schedule_hz,
        "max_updates": max_updates,
        "save_history": False,
    }


def read_result(path, case, backend):
    doc, arrays = read_artifact(path, "wust.reconstruction")
    speed = arrays["c_mps"]
    spec = doc["arrays"]["c_mps"]
    if (
        spec["axes"] != "y,x"
        or spec["units"] != "m/s"
        or speed.shape != case.grid.shape
        or speed.dtype != np.float64
        or not np.isfinite(speed).all()
        or np.any(speed <= 0)
    ):
        raise ValueError("Invalid final WUST model; no resizing or clipping permitted")
    meta = doc["metadata"]
    if meta.get("environment", {}).get("backend") != backend:
        raise RuntimeFailure(
            "incompatible_runtime", "WUST backend differs; no fallback permitted"
        )
    if (
        meta.get("completion", {}).get("reason")
        not in ("zero_updates", "caller_truncated_schedule", "schedule_complete")
        or meta["completion"].get("converged") is not False
    ):
        raise ValueError("Invalid WUST completion semantics")
    return speed, meta


class WUSTFWIAlgorithm:
    name = "fwi_wust"

    @prepared_run
    def run(self, case, config):
        started = time.monotonic()
        folder = None
        try:
            p = config.parameters
            deployment = config.deployment
            resolved = config.run_controls
            deadline = Deadline(resolved.max_elapsed_s, started=started)
            base = deployment.scratch_root
            if base is not None:
                Path(base).mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="wust-", dir=base)).resolve()
            request = prepare_request(
                case, p, folder, resolved.max_iterations, deployment.backend
            )
            client = WUSTClient(deployment, deadline, folder)
            capability = client.admit()
            client.invoke(request)
            speed, meta = read_result(
                request["output_manifest"], case, deployment.backend
            )
            reason = meta["completion"]["reason"]
            complete = reason == "schedule_complete"
            metrics = {
                "forward_model": "WUST_Helmholtz",
                "final_data_residual": None,
                "stop_reason": "schedule_complete" if complete else "max_iterations",
                "termination_category": "completion" if complete else "budget",
                "converged": False,
                "completed_iterations": meta["completed_updates"],
                "wust": meta,
                "runtime_version": capability.get("runtime_version"),
                "requested_schedule_hz": p.frequency_schedule_hz,
                "canonical_frequencies_hz": meta["canonical_frequencies_hz"],
                "executed_frequencies_hz": meta["executed_frequencies_hz"],
                "resolved_run_policy": resolved.model_dump(),
            }
            deadline.remaining()
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=speed,
                metrics=metrics,
                runtime_s=time.monotonic() - started,
                artifacts={"wust_directory": str(folder)},
            )
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            reason = getattr(exc, "reason", "incompatible_request")
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"{reason}: {exc}",
                runtime_s=time.monotonic() - started,
                metrics={
                    "stop_reason": reason,
                    "termination_category": (
                        "budget" if reason == "time_budget" else "failure"
                    ),
                    "converged": False,
                },
                artifacts={} if folder is None else {"wust_directory": str(folder)},
            )
