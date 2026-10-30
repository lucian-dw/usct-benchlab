"""Nonlinear first-arrival Eikonal inversion; no straight-ray fallback."""

from __future__ import annotations

from usctbench.core.config import prepared_run

import numpy as np

from usctbench.algorithms._control import InversionControl
from usctbench.algorithms.ray import (
    configured_ray_weights,
    reference_sound_speed,
    speed_bounds,
)
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.core.stopping import BudgetExhausted
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.model_space import (
    BilinearBasis,
    FineGridRegularizer,
    ReducedLinearOperator,
)
from usctbench.solvers.linear import solve_linearized
from usctbench.operators.base import Linearization
from usctbench.solvers.nonlinear import (
    GaussNewtonProblem,
    project_each_trial,
    gauss_newton,
)


def prepare_bent_problem(
    forward,
    initial,
    reference,
    bounds,
    offset,
    *,
    basis=None,
    regularization="identity",
    roi=None,
):
    """Prepare s or theta with s=s0+B theta, J_theta=J_s B and L_theta=L B."""
    lower = 1 / bounds[1] - (0 if basis is None else reference.flat[0])
    upper = 1 / bounds[0] - (0 if basis is None else reference.flat[0])
    starting = initial if basis is None else np.zeros(basis.shape)
    prior = reference if basis is None else np.zeros(basis.shape)

    def physical(model):
        return model if basis is None else reference + basis.forward(model)

    def project(proposal, current):
        bounded = np.clip(proposal, lower, upper)
        return bounded if roi is None else np.where(roi, bounded, reference)

    def linearize(model):
        lin = forward.linearize(physical(model))
        jacobian = (
            lin.jacobian
            if basis is None
            else ReducedLinearOperator(lin.jacobian, basis)
        )
        return Linearization(
            lin.value.reshape(offset.shape) - offset, jacobian, lin.derivative_kind
        )

    return GaussNewtonProblem(
        starting,
        prior,
        lower,
        upper,
        linearize,
        project,
        physical,
        project_each_trial,
        regularization if basis is None else FineGridRegularizer(basis, regularization),
        roi,
        1e-3 * np.sqrt(starting.size) * reference.flat[0],
        "water_reference",
        {
            "strategy": "project_each_shortened_direction",
            "displacement_parameter": "slowness" if basis is None else "coefficients",
            "roi_fixed": roi is not None,
        },
    )


class BentRayAlgorithm:
    """Bounded, regularized Gauss-Newton inversion of absolute or differential TOF.

    ``gradient_rtol`` (default 1e-8) is relative to the first checked regularized
    training gradient. ``line_search=False`` disables backtracking, not Armijo.
    """

    name = "bent_ray_gn"

    @prepared_run
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        p = config.parameters
        c0 = reference_sound_speed(case, config)
        differential = case.measurement.delta_tof_s is not None
        measurement_c0 = float(case.metadata.get("reference_sound_speed_mps", 1500.0))
        if differential and not np.isclose(c0, measurement_c0, rtol=1e-12, atol=0):
            raise ValueError(
                "reference_sound_speed_mps conflicts with the delta_tof_s measurement "
                "reference; explicitly rebase the observations before changing the model reference"
            )
        bounds = speed_bounds(config)
        forward = EikonalForward(
            case.grid,
            case.geometry,
            background_speed_mps=c0,
            spatial_order=p.eikonal_order,
        )
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
        )
        observed = (
            case.measurement.delta_tof_s if differential else case.measurement.tof_s
        )
        if observed is None:
            raise ValueError("bent_ray_gn requires measured delta_tof_s or tof_s")
        observed = np.asarray(observed, dtype=float)
        valid = distance > 0
        if case.measurement.valid_mask is not None:
            valid &= case.measurement.valid_mask
        weights = configured_ray_weights(case, forward, valid.ravel(), config).reshape(
            observed.shape
        )
        control = InversionControl(
            case,
            config,
            observed,
            default_iterations=4,
            weights=weights,
            valid_mask=valid,
            iteration_unit="Gauss-Newton outer step",
        )
        control.declare_update(
            (
                "slowness"
                if p.model_grid_shape is None
                else "delta_slowness_coefficients"
            ),
            "full_slowness",
            "s/m",
            normalization="norm(q_new-q_old)/norm(q_old); positive bounded slowness",
        )
        inner = int(p.inner_iterations)
        inner_solver = p.inner_solver
        inner_options = p.inner_options.model_dump()
        kind = str(p.regularization)
        damping = p.regularization_lambda**2
        step = float(p.step_length)
        sigma = float(p.smooth_sigma)
        gradient_rtol = float(p.gradient_rtol)
        line_search = p.line_search
        roi_only = p.roi_update_only
        roi = case.grid.roi_mask if roi_only else None
        s0 = np.full(case.grid.shape, 1 / c0)
        initial_speed = np.broadcast_to(
            np.asarray(
                c0 if p.initial_sound_speed_mps is None else p.initial_sound_speed_mps,
                dtype=float,
            ),
            case.grid.shape,
        )
        if not np.isfinite(initial_speed).all() or np.any(initial_speed <= 0):
            raise ValueError("initial_sound_speed_mps must be finite and positive")
        s = np.clip(1 / initial_speed, 1 / bounds[1], 1 / bounds[0])
        if roi is not None:
            s = np.where(roi, s, s0)
        initial = s.copy()
        offset = distance / c0 if differential else np.zeros_like(distance)
        initialization = str(p.initialization)
        initialization_iterations = p.initialization_iterations

        basis = None
        if p.model_grid_shape is not None:
            if roi is not None:
                raise ValueError("reduced model currently requires no inversion ROI")
            if initialization != "configured" or not np.all(initial_speed == c0):
                raise ValueError(
                    "reduced Bent currently requires configured water initialization"
                )
            if not bounds[0] <= c0 <= bounds[1]:
                raise ValueError("reference speed must lie within sound-speed bounds")
            basis = BilinearBasis(case.grid, p.model_grid_shape)

        try:
            if initialization == "cgls":
                from usctbench.operators.straight_ray import (
                    StraightRayProjector,
                )

                initializer = control.call(
                    "initialization_setup", StraightRayProjector.from_case, case
                )
                water_prediction = (
                    np.zeros_like(observed) if differential else distance / c0
                )
                seed_update, info = solve_linearized(
                    initializer,
                    control.weighted_residual(water_prediction),
                    np.zeros_like(s),
                    control,
                    weights=control.precision,
                    iterations=initialization_iterations,
                    method=inner_solver,
                    **inner_options,
                    damping=damping,
                    regularization=kind,
                    roi=roi,
                )
                control.record_linear(info)
                s = np.clip(s0 + seed_update, 1 / bounds[1], 1 / bounds[0])
                if roi is not None:
                    s = np.where(roi, s, s0)
            problem = prepare_bent_problem(
                forward,
                s,
                s0,
                bounds,
                offset,
                basis=basis,
                regularization=kind,
                roi=roi,
            )
            smooth_scale = (
                sigma
                if basis is None
                else tuple(sigma * k / n for k, n in zip(basis.shape, case.grid.shape))
            )
            selected, metrics = gauss_newton(
                problem,
                control,
                inner_iterations=inner,
                inner_solver=inner_solver,
                inner_options=inner_options,
                damping=damping,
                step_length=step,
                smooth_sigma=smooth_scale,
                max_backtracks=9 if line_search else 0,
                gradient_rtol=gradient_rtol,
            )
            s = problem.physical_model(selected)
        except BudgetExhausted as exc:
            control.monitor.finish(exc.reason)
            s, metrics = control.output(initial)
        except FloatingPointError as exc:
            control.fail(exc)
            s, metrics = control.output(initial)
        sound_speed = 1 / s
        metrics.update(
            {
                "backend": "native_eikonal_fast_marching",
                "eikonal_order": forward.spatial_order,
                "forward_model": "eikonal",
                "observation_domain": "travel_time",
                "observation_units": "s",
                "model_parameterization": (
                    {"kind": "pixels"} if basis is None else basis.metadata()
                ),
                "true_bent_ray": True,
                "uses_true_bent_rays": True,
                "surrogate_travel_time_backend": False,
                "full_external_eikonal_solver": False,
                "method_family": "first_arrival_eikonal",
                "linearization_parameter": "slowness_s_per_m",
                "inner_iterations": inner,
                "inner_solver": inner_solver,
                "inner_options": inner_options,
                "initialization": initialization,
                "initialization_training_only": initialization == "cgls",
                "initialization_iterations": (
                    initialization_iterations if initialization == "cgls" else 0
                ),
                "smoothing_applied_to": "update_direction_not_accumulated_image",
                "regularization": kind,
                "regularization_lambda_squared": damping,
                "line_search": line_search,
                "line_search_acceptance": "projected_step_armijo_1e-4",
                "gradient_rtol": gradient_rtol,
                "gradient_space": "pixels" if basis is None else "coefficients",
                "gradient_scope": (
                    "training_regularized_objective_at_listed_iterates_not_necessarily_selected_checkpoint"
                ),
                "measurement_reference_sound_speed_mps": (
                    measurement_c0 if differential else None
                ),
                "roi_update_only": roi_only,
                "roi_laplacian": False,
                "ground_truth_used_for_initialization": False,
            }
        )

        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
        )


__all__ = ["BentRayAlgorithm"]
