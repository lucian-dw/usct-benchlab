"""Straight-ray attenuation tomography."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import run_with_failure_capture, sirt_solve, target_attenuation_integral
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class AttenuationSIRTAlgorithm:
    name = "attenuation_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_attenuation_integral(case, projector)
            iterations = int(config.parameters.get("iterations", 50))
            relaxation = float(config.parameters.get("relaxation", 0.8))
            upper = float(config.parameters.get("attenuation_upper_np_per_m", 80.0))
            attenuation, residual_norms = sirt_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
                nonnegative=True,
            )
            attenuation = np.clip(attenuation, 0.0, upper)
            final_residual = target - projector.forward(attenuation)
            metrics = {
                "data_residual_norm": float(np.linalg.norm(final_residual[mask])),
                "initial_data_residual_norm": residual_norms[0],
                "iterations": iterations,
            }
            if case.ground_truth.attenuation_np_per_m is not None:
                metrics.update(
                    compute_image_metrics(
                        attenuation,
                        np.asarray(case.ground_truth.attenuation_np_per_m, dtype=float),
                        mask=case.grid.roi_mask,
                        prefix="attenuation_",
                    )
                )
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                attenuation_np_per_m=attenuation,
                metrics=metrics,
            )

        return run_with_failure_capture(self.name, case, _run)

