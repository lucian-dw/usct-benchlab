"""Application wiring for straight-ray solvers; physics lives in operators/."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from usctbench.algorithms._control import InversionControl
from usctbench.algorithms.ray import (
    configured_ray_weights,
    coverage_preconditioner,
    ray_weight_metrics,
    reference_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.model_space import (
    BilinearBasis,
    FineGridRegularizer,
    ReducedLinearOperator,
)
from usctbench.solvers.projected_cg import projected_cg
from usctbench.solvers.row_action import row_action


def reconstruct(case, config, algorithm):
    projector = StraightRayProjector.from_case(case)
    target, mask = target_delta_tof(case, projector)
    weights = configured_ray_weights(case, projector, mask, config)
    p = config.parameters
    c0 = reference_sound_speed(case, config)
    measurement_c0 = float(case.metadata.get("reference_sound_speed_mps", 1500.0))
    if not np.isclose(c0, measurement_c0, rtol=1e-12, atol=0):
        raise ValueError(
            "reference_sound_speed_mps conflicts with the delta_tof_s measurement "
            "reference; explicitly rebase the observations before changing the model reference"
        )
    low, high = speed_bounds(config)
    if not low <= c0 <= high:
        raise ValueError("reference sound speed must lie within sound_speed_bounds_mps")
    is_cg = algorithm == "straight_cgls"
    lambda_value = p.regularization_lambda if is_cg else 0.0
    control = InversionControl(
        case,
        config,
        target.reshape(projector.ray_shape),
        weights=weights.reshape(projector.ray_shape),
        valid_mask=mask.reshape(projector.ray_shape),
        iteration_unit="cgls_step" if is_cg else "full_sweep",
    )
    roi_only = p.roi_update_only
    roi_laplacian = p.roi_laplacian if is_cg else False
    roi = case.grid.roi_mask if roi_only else None
    basis = None
    solver_operator = projector
    kind = p.regularization if is_cg else "identity"
    solver_kind = kind
    model_shape = case.grid.shape
    if p.model_grid_shape is not None:
        if roi is not None or (roi_laplacian and case.grid.roi_mask is not None):
            raise ValueError("reduced model currently requires no inversion ROI")
        basis = BilinearBasis(case.grid, p.model_grid_shape)
        model_shape = basis.shape
        solver_operator = ReducedLinearOperator(projector, basis)
        solver_kind = FineGridRegularizer(basis, kind)
    initial = np.zeros(model_shape)
    s0 = np.full(model_shape, 1.0 / c0)

    def project(x):
        bounded = np.clip(x, 1.0 / high - 1.0 / c0, 1.0 / low - 1.0 / c0)
        return bounded if roi is None else np.where(roi, bounded, 0)

    def to_speed(x):
        fine = x if basis is None else basis.forward(x)
        return 1.0 / (1.0 / c0 + fine)

    use_preconditioning = p.coverage_preconditioning if is_cg else False
    preconditioner = None
    # Coverage construction is a real adjoint application and is budgeted.
    # It cannot consume held-out receiver values or exceed the adjoint budget.
    if use_preconditioning:
        from usctbench.core.stopping import BudgetExhausted

        try:
            preconditioner, _ = control.call(
                "adjoint",
                coverage_preconditioner,
                solver_operator,
                control.split.train.ravel(),
                control.split.weights.ravel(),
                roi_mask=roi,
                eps=float(p.coverage_preconditioner_eps),
                max_scale=float(p.coverage_preconditioner_max_scale),
                normalize=p.coverage_preconditioner_normalize,
            )
        except BudgetExhausted as exc:
            control.monitor.finish(exc.reason)
    robust_loss = p.robust_loss if is_cg else "none"
    if control.monitor.reason is not None:
        state, metrics = control.output(initial)
    elif is_cg:
        state, metrics = projected_cg(
            solver_operator,
            control,
            initial=initial,
            reference=s0,
            lower=1.0 / high - 1.0 / c0,
            upper=1.0 / low - 1.0 / c0,
            damping=lambda_value**2,
            regularization=solver_kind,
            roi=roi,
            regularization_roi=case.grid.roi_mask if roi_laplacian else None,
            preconditioner=preconditioner,
            huber_delta=(float(p.huber_delta_s) if robust_loss != "none" else None),
            irls_max_inner_steps=p.irls_max_inner_steps,
            optimality_rtol=float(p.gradient_rtol),
        )
    else:
        sigma = float(p.smooth_sigma)
        every = int(p.smooth_every)
        if not np.isfinite(sigma) or sigma < 0 or every < 0:
            raise ValueError("smoothing parameters must be nonnegative and finite")

        smooth_scale = (
            sigma
            if basis is None
            else tuple(sigma * k / n for k, n in zip(basis.shape, case.grid.shape))
        )

        def smooth(x, iteration):
            return (
                gaussian_filter(x, smooth_scale, mode="nearest")
                if sigma > 0 and every > 0 and iteration % every == 0
                else x
            )

        state, metrics = row_action(
            solver_operator,
            control,
            row_sums=projector.row_norms(power=1),
            initial=initial,
            reference=s0,
            project=project,
            relaxation=float(p.relaxation),
            subsets=p.subsets if algorithm == "straight_sart" else 1,
            postprocess=smooth,
        )
    sound_speed = to_speed(state)
    metrics.update(
        {
            "backend": "native_siddon_straight_ray",
            "forward_model": "straight_ray",
            "observation_domain": "travel_time",
            "observation_units": "s",
            "model_parameterization": (
                {"kind": "pixels"} if basis is None else basis.metadata()
            ),
            "projector_csr_storage_bytes": projector.storage_bytes,
            "measurement_reference_sound_speed_mps": measurement_c0,
            **ray_weight_metrics(weights, mask, config),
        }
    )

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        sound_speed_mps=sound_speed,
        metrics=metrics,
    )


class StraightRayCGLSAlgorithm:
    name = "straight_cgls"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return reconstruct(case, config, self.name)


class StraightRaySIRTAlgorithm:
    name = "straight_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return reconstruct(case, config, self.name)


class StraightRaySARTAlgorithm:
    name = "straight_sart"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return reconstruct(case, config, self.name)
