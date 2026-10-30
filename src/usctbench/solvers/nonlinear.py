"""Shared Gauss-Newton driver; physical spaces and trials are prepared by callers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter

from usctbench.solvers.linear import solve_linearized
from usctbench.operators.regularization import prepare_regularizer


@dataclass
class GaussNewtonProblem:
    initial: np.ndarray
    reference: np.ndarray
    lower: np.ndarray | float
    upper: np.ndarray | float
    linearize: Callable
    project: Callable
    physical_model: Callable
    trials: Callable
    regularization: object
    roi: np.ndarray | None
    fallback_step_norm: float
    prior_source: str
    constraints: dict


def project_each_trial(problem, state, proposal, step_length, max_backtracks):
    """Bent: P(q + alpha * 2**-k * d), in pixels or coefficient space."""
    for k in range(max_backtracks + 1):
        fraction = 0.5**k
        yield problem.project(
            state + step_length * fraction * proposal, state
        ), fraction


def backtrack_feasible_displacement(
    problem, state, proposal, step_length, max_backtracks
):
    """Born: form q0=P(q+alpha*d) once, then q + 2**-k*(q0-q)."""
    first = problem.project(state + step_length * proposal, state)
    displacement = first - state
    for k in range(max_backtracks + 1):
        fraction = 0.5**k
        yield (first if k == 0 else state + fraction * displacement), fraction


def gauss_newton(
    problem,
    control,
    *,
    inner_iterations=12,
    inner_solver="lsmr",
    inner_options=None,
    damping=0.0,
    step_length=1.0,
    smooth_sigma=0.0,
    max_backtracks=10,
    gradient_rtol=1e-8,
):
    """Accept using the real objective and actual feasible displacement.

    The accepted Linearization is reused for the next direction. Validation data
    never enter gradients, curvature, priors, trial generation or acceptance.
    """
    inner_options = dict(inner_options or {})
    state = np.asarray(problem.initial, dtype=float).copy()
    initial_state = state.copy()
    reference = problem.reference
    penalty = prepare_regularizer(problem.regularization)
    roi = problem.roi
    attempts, direction_checks, gradient_checks = [], [], []
    diagnostics = control.config.diagnostics
    initial_gradient_norm = None
    derivative_kind = None

    def objective(model, prediction):
        active = control.split.train | control.split.validation
        if not np.isfinite(prediction[active]).all():
            raise FloatingPointError("nonfinite prediction on active data")
        residual = np.where(control.split.train, control.safe_observed - prediction, 0)
        reg = penalty.forward(model - reference)
        value = float(
            0.5 * np.sum(control.precision * np.abs(residual) ** 2)
            + 0.5 * damping * np.vdot(reg, reg).real
        )
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite regularized objective")
        return value

    try:
        linearization = control.call("forward", problem.linearize, state)
        derivative_kind = linearization.derivative_kind
        cost = objective(state, linearization.value)
        control.observe(
            0,
            state,
            linearization.value,
            objective=cost,
            linearization=linearization,
        )
        for iteration in range(1, control.policy.max_iterations + 1):
            if control.monitor.reason is not None:
                break
            current = control.current_state
            state, linearization, cost = (
                current.model,
                current.linearization,
                current.objective,
            )
            residual = control.weighted_residual(linearization.value)
            gradient = -control.call(
                "adjoint", linearization.jacobian.adjoint, residual
            )
            gradient += damping * penalty.adjoint(penalty.forward(state - reference))
            if roi is not None:
                gradient = np.where(roi, gradient, 0)
            if not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite nonlinear objective gradient")
            with np.errstate(over="raise", invalid="raise"):
                # Flatten to preserve the Frobenius norm using scale-safe BLAS.
                gradient_norm = float(norm(gradient.ravel()))
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("nonfinite objective gradient norm")
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient_norm
            gradient_relative = gradient_norm / max(
                initial_gradient_norm, np.finfo(float).tiny
            )
            control.monitor.history[-1]["optimizer_diagnostics"] = {
                "gradient_norm": gradient_norm,
                "gradient_relative": gradient_relative,
            }
            if diagnostics:
                gradient_checks.append(
                    {
                        "iteration": control.monitor.history[-1]["iteration"],
                        "norm": gradient_norm,
                        "relative_to_initial": gradient_relative,
                    }
                )
            if gradient_relative <= gradient_rtol:
                control.monitor.finish(
                    "stationary_surrogate_gradient"
                    if "approximation" in derivative_kind
                    else "stationary_gradient"
                )
                break
            feasible_gradient = np.where(
                ((state <= problem.lower) & (gradient > 0))
                | ((state >= problem.upper) & (gradient < 0)),
                0,
                gradient,
            )
            feasible_norm = float(norm(feasible_gradient.ravel()))
            if feasible_norm == 0:
                control.monitor.finish(
                    "stationary_surrogate_projected_gradient"
                    if "approximation" in derivative_kind
                    else "stationary_projected_gradient"
                )
                break
            direction, info = solve_linearized(
                linearization.jacobian,
                residual,
                state - reference,
                control,
                weights=control.precision,
                iterations=inner_iterations,
                damping=damping,
                regularization=penalty,
                roi=roi,
                method=inner_solver,
                **inner_options,
            )
            control.record_linear(info)
            raw_direction = direction.copy()
            raw_slope = float(np.vdot(gradient, direction).real)
            smoothed_slope = None
            if np.any(np.asarray(smooth_sigma) > 0):
                smoothed = gaussian_filter(direction, smooth_sigma, mode="nearest")
                if roi is not None:
                    smoothed = np.where(roi, smoothed, 0)
                smoothed_slope = float(np.vdot(gradient, smoothed).real)
                # Smoothing a Newton step can turn it uphill even for a convex
                # quadratic. It is only a proposal, not an accepted optimizer.
                if np.isfinite(smoothed_slope) and smoothed_slope < 0:
                    direction = smoothed
            if roi is not None:
                direction = np.where(roi, direction, 0)
            if not np.all(np.isfinite(direction)):
                raise FloatingPointError("nonfinite GN update")
            accepted = False
            proposals = [("newton_proposal", direction)]
            if not np.array_equal(direction, raw_direction):
                proposals.append(("unsmoothed_newton", raw_direction))
            # Projection/clipping can also destroy descent of a Newton step.
            # A projected negative gradient provides an independently checked
            # fallback, at the same model-step scale, without changing the loss.
            with np.errstate(over="raise", invalid="raise"):
                direction_scale = float(norm(raw_direction.ravel()))
            if not np.isfinite(direction_scale):
                raise FloatingPointError("nonfinite Newton direction norm")
            if direction_scale == 0:
                direction_scale = problem.fallback_step_norm
            gradient_step = -feasible_gradient * (direction_scale / feasible_norm)
            proposals.append(("projected_gradient", gradient_step))
            if diagnostics:
                direction_checks.append(
                    {
                        "iteration": iteration,
                        "raw_directional_derivative": raw_slope,
                        "smoothed_directional_derivative": smoothed_slope,
                        "smoothing_rejected": smoothed_slope is not None
                        and smoothed_slope >= 0,
                    }
                )
            finite_trial_seen = nonfinite_trial_seen = False
            for proposal_name, proposal in proposals:
                for backtrack, (candidate, fraction) in enumerate(
                    problem.trials(
                        problem, state, proposal, step_length, max_backtracks
                    )
                ):
                    displacement = candidate - state
                    slope = float(np.vdot(gradient, displacement).real)
                    row = {
                        "iteration": iteration,
                        "proposal": proposal_name,
                        "step_length": float(step_length),
                        "backtracking_fraction": fraction,
                        "displacement_norm": float(
                            norm(displacement.ravel(), check_finite=False)
                        ),
                        "changed_pixel_count": int(np.count_nonzero(displacement)),
                        "projected_directional_derivative": slope,
                        "objective": None,
                        "accepted": False,
                    }
                    if not np.isfinite(slope) or slope >= 0:
                        row["rejection"] = "non_descent_projected_step"
                        if diagnostics:
                            attempts.append(row)
                        continue
                    try:
                        trial = control.call(
                            "line_search", problem.linearize, candidate
                        )
                        trial_cost = objective(candidate, trial.value)
                    except FloatingPointError:
                        nonfinite_trial_seen = True
                        row["rejection"] = "nonfinite_trial_prediction_or_objective"
                        if diagnostics:
                            attempts.append(row)
                        continue
                    finite_trial_seen = True
                    accepted = bool(
                        np.isfinite(trial_cost) and trial_cost <= cost + 1e-4 * slope
                    )
                    row.update(
                        objective=trial_cost if np.isfinite(trial_cost) else None,
                        accepted=accepted,
                    )
                    if diagnostics:
                        attempts.append(row)
                    if accepted:
                        break
                if accepted:
                    break
            if not accepted:
                control.monitor.finish(
                    "numerical_failure"
                    if nonfinite_trial_seen and not finite_trial_seen
                    else "line_search_failed"
                )
                break
            physical = problem.physical_model(state)
            update = float(
                norm((problem.physical_model(candidate) - physical).ravel())
                / norm(physical.ravel())
            )
            state, linearization, cost = candidate, trial, trial_cost
            control.observe(
                iteration,
                state,
                trial.value,
                objective=cost,
                update_relative=update,
                linearization=trial,
            )
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except FloatingPointError as exc:
        control.fail(exc)
    selected, metrics = control.output(initial_state)
    metrics.update(
        prior_reference_source=problem.prior_source,
        gradient_rtol=gradient_rtol,
        inner_solver=inner_solver,
        inner_options=inner_options,
        line_search_acceptance="projected_step_armijo_1e-4",
        line_search_constraints=problem.constraints,
        derivative_kind=derivative_kind,
        nonlinear_background_updates=max(0, len(control.monitor.history) - 1),
    )
    if diagnostics:
        metrics.update(
            line_search_history=attempts,
            direction_checks=direction_checks,
            gradient_checks=gradient_checks,
        )
    return selected, metrics
