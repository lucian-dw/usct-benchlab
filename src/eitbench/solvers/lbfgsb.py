"""L-BFGS-B retaining only fully evaluated, accepted iterates on interruption."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from ..core.run_controls import BudgetExhausted


@dataclass
class Evaluation:
    x: np.ndarray
    value: float
    gradient: np.ndarray
    payload: dict


class CachedObjective:
    def __init__(self, evaluate, ledger):
        self.evaluate = evaluate
        self.ledger = ledger
        self.cache = []
        self.evaluations = 0

    def record(self, x):
        for record in self.cache:
            if np.array_equal(record.x, x):
                return record
        self.ledger.check()
        value, gradient, payload = self.evaluate(x)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite objective or gradient")
        record = Evaluation(
            np.array(x, copy=True), float(value), np.array(gradient, copy=True), payload
        )
        self.cache = [record, *self.cache[:1]]
        self.evaluations += 1
        return record

    def __call__(self, x):
        record = self.record(x)
        return record.value, record.gradient


def optimize(
    evaluate, initial, bounds, ledger, *, maxiter, ftol, gtol, maxls=40, maxcor=20
):
    initial = np.clip(
        np.asarray(initial, dtype=float), np.array(bounds)[:, 0], np.array(bounds)[:, 1]
    )
    objective = CachedObjective(evaluate, ledger)
    accepted = None
    history = []

    def accept(x):
        nonlocal accepted
        accepted = objective.record(x)
        history.append(
            dict(
                iteration=len(history),
                objective=accepted.value,
                gradient_inf=float(np.linalg.norm(accepted.gradient, ord=np.inf)),
            )
        )

    status, reason, message = (
        "max_iterations",
        "max_iterations",
        "zero target updates requested",
    )
    scipy_status = None
    try:
        accept(initial)
        if maxiter > 0:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=accept,
                options=dict(
                    maxiter=maxiter, ftol=ftol, gtol=gtol, maxls=maxls, maxcor=maxcor
                ),
            )
            # result.x is an optimizer-accepted iterate. No trial is promoted on errors.
            if not np.array_equal(result.x, accepted.x):
                accept(result.x)
            scipy_status = int(result.status)
            message = str(result.message)
            if result.success:
                status, reason = "converged", (
                    "projected_gradient_tolerance"
                    if "PGTOL" in message or "PROJECTED_GRADIENT" in message
                    else "objective_tolerance"
                )
            elif result.status == 1:
                status, reason = "max_iterations", "max_iterations"
            else:
                status, reason = "numerical_error", "line_search_failed"
    except BudgetExhausted as exc:
        status, reason, message = "resource_exhausted", str(exc), str(exc)
    except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, reason, message = "numerical_error", "numerical_failure", str(exc)
    info = dict(
        status=status,
        success=status == "converged",
        stopping_reason=reason,
        message=message,
        scipy_status=scipy_status,
        iterations=max(0, len(history) - 1),
        evaluations=objective.evaluations,
        history=history,
        initial_objective=history[0]["objective"] if history else None,
        final_objective=accepted.value if accepted else None,
    )
    return accepted, info
