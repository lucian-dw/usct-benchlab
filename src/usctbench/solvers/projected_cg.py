"""Bounded CGLS/IRLS for real models and real or complex observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usctbench.core.stopping import BudgetExhausted
from usctbench.operators.regularization import prepare_regularizer


@dataclass
class _Conjugacy:
    weights: np.ndarray | None = None
    direction: np.ndarray | None = None
    gamma: float | None = None
    free: np.ndarray | None = None
    stage_gamma: float = 0.0
    stage: int = -1
    steps_since_refresh: int = 0


@dataclass
class _CGStep:
    model: np.ndarray
    prediction: np.ndarray
    objective: float
    relative_update: float
    direction: np.ndarray
    gamma: float
    reset_conjugacy: bool


def projected_cg(
    operator,
    control,
    *,
    initial,
    reference,
    lower,
    upper,
    offset=0.0,
    damping=0.0,
    regularization="identity",
    roi=None,
    regularization_roi=None,
    preconditioner=None,
    huber_delta=None,
    irls_max_inner_steps=10,
    optimality_rtol=1e-10,
):
    """All CG steps share one global budget, including across IRLS restarts.

    Prediction is offset + J x. IRLS freezes training-only weights for at most
    irls_max_inner_steps, refreshing early on inner stationarity or restart.
    Huber objective and validation residuals use the original precision.
    Backtracking handles physical bound projection; rejected trials never
    replace a valid checkpoint. Preconditioner is a physical-coordinate scale D.
    """
    start = np.array(initial, dtype=float, copy=True)
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if huber_delta is not None and (not np.isfinite(huber_delta) or huber_delta <= 0):
        raise ValueError("huber_delta_s must be finite and positive")
    if (
        isinstance(irls_max_inner_steps, bool)
        or int(irls_max_inner_steps) != irls_max_inner_steps
        or irls_max_inner_steps <= 0
    ):
        raise ValueError("irls_max_inner_steps must be a positive integer")
    if not np.isfinite(optimality_rtol) or optimality_rtol < 0:
        raise ValueError("optimality_rtol must be finite and nonnegative")
    scale = (
        np.ones_like(start)
        if preconditioner is None
        else np.asarray(preconditioner, dtype=float)
    )
    if (
        scale.shape != start.shape
        or not np.all(np.isfinite(scale))
        or np.any(scale < 0)
    ):
        raise ValueError(
            "preconditioner must be finite, nonnegative and match the image"
        )
    scale = scale if roi is None else np.where(roi, scale, 0)
    regmask = (
        np.ones_like(start)
        if regularization_roi is None
        else np.asarray(regularization_roi, dtype=float)
    )
    penalty = prepare_regularizer(regularization)
    conjugacy = _Conjugacy()
    initial_gradient_norm = None

    def regularize(value):
        return penalty.forward(regmask * value)

    def normal(value):
        return regmask * penalty.adjoint(regularize(value))

    def residual(prediction):
        return np.where(control.split.train, control.safe_observed - prediction, 0)

    def objective(value, prediction):
        a = np.abs(residual(prediction))
        loss = (
            0.5 * a**2
            if huber_delta is None
            else np.where(
                a <= huber_delta, 0.5 * a**2, huber_delta * (a - 0.5 * huber_delta)
            )
        )
        reg = regularize(value)
        return float(
            np.sum(control.precision * loss) + 0.5 * damping * np.vdot(reg, reg).real
        )

    def precision(prediction):
        if huber_delta is None:
            return control.precision
        return control.precision * np.minimum(
            1.0,
            huber_delta
            / np.maximum(np.abs(residual(prediction)), np.finfo(float).tiny),
        )

    def negative_gradient(current, weights):
        return control.call(
            "adjoint", operator.adjoint, weights * residual(current.prediction)
        ).real - damping * normal(current.model)

    def refresh_weights(current_weights):
        """Begin a fresh frozen-weight stage and discard conjugate directions."""
        conjugacy.weights = current_weights
        conjugacy.direction = conjugacy.gamma = conjugacy.free = None
        conjugacy.stage_gamma = 0.0
        conjugacy.steps_since_refresh = 0
        conjugacy.stage += 1
        control.work.counts["cg_restarts"] = (
            control.work.counts.get("cg_restarts", 0) + 1
        )
        if huber_delta is not None:
            control.work.counts["irls_reweights"] = (
                control.work.counts.get("irls_reweights", 0) + 1
            )
            control.monitor.set_stage(f"irls_{conjugacy.stage}")

    def project(value):
        bounded = np.clip(value, lower, upper)
        return bounded if roi is None else np.where(roi, bounded, start)

    def free_variables(model, vector):
        free = ((model > lower) | (vector >= 0)) & ((model < upper) | (vector <= 0))
        return free if roi is None else free & roi

    def try_step(current, current_weights, true_gradient, free):
        """Try the frozen-weight CG direction against the actual objective."""
        weights = conjugacy.weights
        stale_weights = not np.array_equal(weights, current_weights)
        raw_gradient = (
            negative_gradient(current, weights) if stale_weights else true_gradient
        )
        gradient = scale * np.where(free, raw_gradient, 0)
        next_gamma = float(np.vdot(gradient, gradient).real)
        if not np.isfinite(next_gamma):
            raise FloatingPointError("nonfinite scaled gradient")
        if stale_weights and next_gamma <= max(
            np.finfo(float).tiny, conjugacy.stage_gamma * 1e-14
        ):
            return None
        if next_gamma <= np.finfo(float).tiny:
            control.monitor.finish("linear_solver_breakdown")
            return None
        search_direction = scale * gradient
        if conjugacy.direction is not None:
            search_direction = (
                search_direction + (next_gamma / conjugacy.gamma) * conjugacy.direction
            )
        feasible = free_variables(current.model, search_direction)
        if (
            np.any(search_direction[~feasible] != 0)
            or np.vdot(true_gradient, search_direction).real <= 0
        ):
            return None
        q = control.call("jacobian", operator.forward, search_direction).reshape(
            control.observed.shape
        )
        reg = regularize(search_direction)
        curvature = float(
            np.sum(weights * np.abs(q) ** 2) + damping * np.vdot(reg, reg).real
        )
        if not np.isfinite(curvature) or curvature <= 0:
            control.monitor.finish("linear_solver_breakdown")
            return None
        alpha = next_gamma / curvature
        for backtrack in range(12):
            step = alpha * 0.5**backtrack
            raw = current.model + step * search_direction
            candidate = project(raw)
            projected = not np.array_equal(candidate, raw)
            prediction = (
                offset
                + control.call("line_search", operator.forward, candidate).reshape(
                    control.observed.shape
                )
                if projected
                else current.prediction + step * q
            )
            cost = objective(candidate, prediction)
            if np.isfinite(cost) and cost <= current.objective + 1e-12 * max(
                current.objective, np.finfo(float).tiny
            ):
                break
        else:
            return None
        relative_update = float(
            np.linalg.norm(candidate - current.model)
            / max(np.linalg.norm(reference + current.model), np.finfo(float).tiny)
        )
        small_update = (
            control.policy.update_rtol is not None
            and relative_update <= control.policy.update_rtol
        )
        small_improvement = (
            control.policy.objective_rtol is not None
            and (current.objective - cost)
            / max(abs(current.objective), np.finfo(float).tiny)
            <= control.policy.objective_rtol
        )
        if stale_weights and (small_update or small_improvement):
            return None
        return _CGStep(
            candidate,
            prediction,
            cost,
            relative_update,
            search_direction,
            next_gamma,
            projected or backtrack > 0,
        )

    def step_with_conjugacy_retry(current, current_weights, true_gradient, free):
        step = try_step(current, current_weights, true_gradient, free)
        if step is None and control.monitor.reason is None:
            # One fresh feasible-gradient retry; no completed iteration is charged.
            refresh_weights(current_weights)
            step = try_step(current, current_weights, true_gradient, free)
        return step

    def output():
        state, metrics = control.output(start)
        selected = metrics["stopping"]["selected_iteration"]
        record = next(
            (
                row.get("optimality")
                for row in control.monitor.history
                if row["iteration"] == selected
            ),
            None,
        )
        # Selection never certifies one checkpoint with another point's gradient.
        metrics["solver_optimality"] = {
            "iteration": selected,
            "verified": record is not None,
            **(
                record
                if record is not None
                else {
                    "projected_gradient_norm": None,
                    "initial_projected_gradient_norm": None,
                    "projected_gradient_tolerance": None,
                    "converged": None,
                }
            ),
        }
        return state, metrics

    try:
        if huber_delta is not None:
            control.monitor.set_stage("irls_0")
        prediction = offset + control.call("forward", operator.forward, start).reshape(
            control.observed.shape
        )
        if control.observe(
            0, start, prediction, objective=objective(start, prediction)
        ):
            return output()
        for iteration in range(1, control.policy.max_iterations + 1):
            current = control.current_state
            current_weights = precision(current.prediction)
            true_gradient = negative_gradient(current, current_weights)
            if not np.all(np.isfinite(true_gradient)):
                raise FloatingPointError("nonfinite objective gradient")
            free = free_variables(current.model, true_gradient)
            gradient_norm = float(np.max(np.abs(np.where(free, true_gradient, 0))))
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient_norm
            tolerance = max(
                np.finfo(float).tiny, initial_gradient_norm * optimality_rtol
            )
            converged = gradient_norm <= tolerance
            control.monitor.history[-1]["optimality"] = {
                "projected_gradient_norm": gradient_norm,
                "initial_projected_gradient_norm": initial_gradient_norm,
                "projected_gradient_tolerance": tolerance,
                "converged": converged,
            }
            if converged:
                control.monitor.finish("stationary_gradient")
                break
            if (
                conjugacy.direction is None
                or (
                    huber_delta is not None
                    and conjugacy.steps_since_refresh >= irls_max_inner_steps
                )
                or not np.array_equal(free, conjugacy.free)
            ):
                refresh_weights(current_weights)
            step = step_with_conjugacy_retry(
                current, current_weights, true_gradient, free
            )
            if step is None:
                control.monitor.finish("line_search_failed")
                break
            if conjugacy.steps_since_refresh == 0:
                conjugacy.stage_gamma = step.gamma
            conjugacy.direction = None if step.reset_conjugacy else step.direction
            conjugacy.gamma, conjugacy.free = step.gamma, free
            conjugacy.steps_since_refresh += 1
            if control.observe(
                iteration,
                step.model,
                step.prediction,
                objective=step.objective,
                update_relative=step.relative_update,
            ):
                break
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError as exc:
        control.fail(exc)
    return output()
