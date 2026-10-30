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


def _propose_directions(
    problem,
    current,
    control,
    gradient,
    feasible_gradient,
    penalty,
    *,
    inner_iterations,
    inner_solver,
    inner_options,
    damping,
    smooth_sigma,
):
    """Solve the linearized problem, then propose smoothing and feasible descent."""
    direction, info = solve_linearized(
        current.linearization.jacobian,
        control.weighted_residual(current.prediction),
        current.model - problem.reference,
        control,
        weights=control.precision,
        iterations=inner_iterations,
        damping=damping,
        regularization=penalty,
        roi=problem.roi,
        method=inner_solver,
        **inner_options,
    )
    control.record_linear(info)
    raw_direction = direction
    smoothed_slope = None
    if np.any(np.asarray(smooth_sigma) > 0):
        smoothed = gaussian_filter(direction, smooth_sigma, mode="nearest")
        if problem.roi is not None:
            smoothed = np.where(problem.roi, smoothed, 0)
        smoothed_slope = float(np.vdot(gradient, smoothed).real)
        # Smoothing is only a proposal: even a convex Newton step can turn uphill.
        if np.isfinite(smoothed_slope) and smoothed_slope < 0:
            direction = smoothed
    if problem.roi is not None:
        direction = np.where(problem.roi, direction, 0)
    if not np.isfinite(direction).all():
        raise FloatingPointError("nonfinite GN update")
    proposals = [("newton_proposal", direction)]
    if not np.array_equal(direction, raw_direction):
        proposals.append(("unsmoothed_newton", raw_direction))
    direction_scale = float(norm(raw_direction.ravel()))
    if not np.isfinite(direction_scale):
        raise FloatingPointError("nonfinite Newton direction norm")
    if direction_scale == 0:
        direction_scale = problem.fallback_step_norm
    gradient_step = -feasible_gradient * (
        direction_scale / float(norm(feasible_gradient.ravel()))
    )
    proposals.append(("projected_gradient", gradient_step))
    diagnostics = None
    if control.config.diagnostics:
        diagnostics = {
            "iteration": current.iteration + 1,
            "raw_directional_derivative": float(np.vdot(gradient, raw_direction).real),
            "smoothed_directional_derivative": smoothed_slope,
            "smoothing_rejected": smoothed_slope is not None and smoothed_slope >= 0,
        }
    return proposals, diagnostics


def _armijo_trial(
    problem,
    current,
    control,
    gradient,
    proposals,
    objective,
    *,
    step_length,
    max_backtracks,
    attempts,
):
    """Evaluate only descent trials; return one complete accepted candidate."""
    finite_trial_seen = nonfinite_trial_seen = False
    for proposal_name, proposal in proposals:
        for candidate, fraction in problem.trials(
            problem, current.model, proposal, step_length, max_backtracks
        ):
            displacement = candidate - current.model
            slope = float(np.vdot(gradient, displacement).real)
            row = None
            if attempts is not None:
                row = {
                    "iteration": current.iteration + 1,
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
                attempts.append(row)
            if not np.isfinite(slope) or slope >= 0:
                if row is not None:
                    row["rejection"] = "non_descent_projected_step"
                continue
            try:
                trial = control.call("line_search", problem.linearize, candidate)
                trial_cost = objective(candidate, trial.value)
            except FloatingPointError:
                trial = None
                nonfinite_trial_seen = True
                if row is not None:
                    row["rejection"] = "nonfinite_trial_prediction_or_objective"
                continue
            finite_trial_seen = True
            accepted = trial_cost <= current.objective + 1e-4 * slope
            if row is not None:
                row.update(objective=trial_cost, accepted=bool(accepted))
            if accepted:
                return candidate, trial, trial_cost
            del trial
    control.monitor.finish(
        "numerical_failure"
        if nonfinite_trial_seen and not finite_trial_seen
        else "line_search_failed"
    )
    return None


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
    initial_linearization=None,
):
    """Accept using the real objective and actual feasible displacement.

    The accepted Linearization is reused for the next direction. Validation data
    never enter gradients, curvature, priors, trial generation or acceptance.
    initial_linearization, when supplied, is the value/Jacobian at problem.initial
    and has already been charged by its caller.
    """
    inner_options = dict(inner_options or {})
    initial_state = np.asarray(problem.initial, dtype=float).copy()
    reference = problem.reference
    penalty = prepare_regularizer(problem.regularization)
    roi = problem.roi
    diagnostics = control.config.diagnostics
    attempts = [] if diagnostics else None
    direction_checks = [] if diagnostics else None
    gradient_checks = [] if diagnostics else None
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
        linearization = (
            initial_linearization
            if initial_linearization is not None
            else control.call("forward", problem.linearize, initial_state)
        )
        derivative_kind = linearization.derivative_kind
        cost = objective(initial_state, linearization.value)
        control.observe(
            0,
            initial_state,
            linearization.value,
            objective=cost,
            linearization=linearization,
        )
        initial_linearization = None
        del linearization
        for iteration in range(1, control.policy.max_iterations + 1):
            if control.monitor.reason is not None:
                break
            current = control.current_state
            residual = control.weighted_residual(current.prediction)
            gradient = -control.call(
                "adjoint", current.linearization.jacobian.adjoint, residual
            )
            gradient += damping * penalty.adjoint(
                penalty.forward(current.model - reference)
            )
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
                ((current.model <= problem.lower) & (gradient > 0))
                | ((current.model >= problem.upper) & (gradient < 0)),
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
            proposals, direction_diagnostics = _propose_directions(
                problem,
                current,
                control,
                gradient,
                feasible_gradient,
                penalty,
                inner_iterations=inner_iterations,
                inner_solver=inner_solver,
                inner_options=inner_options,
                damping=damping,
                smooth_sigma=smooth_sigma,
            )
            if diagnostics:
                direction_checks.append(direction_diagnostics)
            accepted = _armijo_trial(
                problem,
                current,
                control,
                gradient,
                proposals,
                objective,
                step_length=step_length,
                max_backtracks=max_backtracks,
                attempts=attempts,
            )
            if accepted is None:
                break
            candidate, trial, trial_cost = accepted
            physical = problem.physical_model(current.model)
            update = float(
                norm((problem.physical_model(candidate) - physical).ravel())
                / norm(physical.ravel())
            )
            control.observe(
                iteration,
                candidate,
                trial.value,
                objective=trial_cost,
                update_relative=update,
                linearization=trial,
            )
            del current, physical, accepted, candidate, trial
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except FloatingPointError as exc:
        control.fail(exc)
    selected, metrics = control.output(initial_state)
    metrics.update(
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
