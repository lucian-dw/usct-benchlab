"""Straight-ray attenuation tomography baseline."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray import (
    StraightRayProjector,
    configured_ray_weights,
    masked_norm,
    ray_weight_metrics,
    residual_metrics,
    run_with_failure_capture,
    sirt_solve,
    target_attenuation_integral,
)
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.metrics import compute_image_metrics


class AttenuationSIRTAlgorithm:
    """SIRT reconstruction from log-amplitude attenuation features."""

    name = "attenuation_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_attenuation_integral(case, projector)
            weights = configured_ray_weights(case, projector, mask, config)
            iterations = int(config.parameters.get("iterations", 50))
            relaxation = float(config.parameters.get("relaxation", 0.8))
            upper = float(config.parameters.get("attenuation_upper_np_per_m", 80.0))
            initial_norm = masked_norm(target, mask, weights)
            attenuation, _residual_norms = sirt_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
                nonnegative=True,
                weights=weights,
            )
            attenuation = np.clip(attenuation, 0.0, upper)
            final_residual = target - projector.forward(attenuation)
            final_norm = masked_norm(final_residual, mask, weights)
            metrics = {
                **residual_metrics(initial_norm, final_norm),
                "attenuation_input_signal_norm": initial_norm,
                "attenuation_input_has_signal": bool(initial_norm > 0.0),
                "attenuation_input_is_surrogate": _is_surrogate_attenuation_case(case),
                "iterations": iterations,
                **ray_weight_metrics(weights, mask, config),
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


def _is_surrogate_attenuation_case(case: USCTCase) -> bool:
    text = " ".join(
        str(item) for item in case.metadata.get("measurement_limitations", [])
    )
    text = f"{text} {case.metadata.get('attenuation_note', '')} {case.metadata.get('feature_provenance', '')}".lower()
    return "zero surrogate" in text or "surrogate" in text and "log_amp" in text


def register_attenuation_algorithm(*, replace: bool = False) -> None:
    register_algorithm(
        "attenuation_sirt",
        AttenuationSIRTAlgorithm,
        description="Straight-ray SIRT attenuation reconstruction.",
        tags=("ray", "attenuation"),
        replace=replace,
    )


__all__ = ["AttenuationSIRTAlgorithm", "register_attenuation_algorithm"]
