"""OR stopping rules, explicit work budgets and auditable termination records."""

from __future__ import annotations

import time

import numpy as np

from usctbench.core.run_controls import RunControls

STOP_CATEGORIES = {
    "not_terminated": "not_terminated",
    "max_iterations": "budget",
    "time_budget": "budget",
    "forward_calls_budget": "budget",
    "adjoint_calls_budget": "budget",
    "numerical_failure": "failure",
    "linear_solver_breakdown": "failure",
    "line_search_failed": "failure",
    "exact_data_fit": "quality_target",
    "target_residual": "quality_target",
    "noise_discrepancy": "quality_target",
    "stationary_gradient": "stationarity",
    "stationary_projected_gradient": "stationarity",
    "stationary_surrogate_gradient": "stationarity",
    "stationary_surrogate_projected_gradient": "stationarity",
    "small_model_update": "stagnation",
    "objective_plateau": "stagnation",
    "validation_plateau": "stagnation",
}


class BudgetExhausted(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class WorkLedger:
    """Count complete array operator calls, including validation and line search.

    Limits are checked *before* starting a call. A running external/NumPy/PDE
    solve cannot be preempted here; elapsed-time limits act between calls.
    Extra source-solve/subset counters are descriptive, not equivalent FLOPs.
    """

    def __init__(self, policy: RunControls, *, clock=time.perf_counter):
        self.policy, self.clock = policy, clock
        self.started = clock()
        self.counts = {"forward_calls": 0, "adjoint_calls": 0}
        self.seconds = {}

    @property
    def elapsed_s(self):
        return max(0.0, float(self.clock() - self.started))

    def check_time(self):
        """Cooperative deadline check for source/GMRES loops, without a call count."""
        if (
            self.policy.max_elapsed_s is not None
            and self.elapsed_s >= self.policy.max_elapsed_s
        ):
            raise BudgetExhausted("time_budget")

    def call(self, kind, function, *args, **kwargs):
        self.check_time()
        counter = "adjoint_calls" if kind == "adjoint" else "forward_calls"
        limit = getattr(self.policy, f"max_{counter}")
        if limit is not None and self.counts[counter] >= limit:
            raise BudgetExhausted(f"{counter}_budget")
        self.counts[counter] += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        started = self.clock()
        try:
            return function(*args, **kwargs)
        finally:
            self.seconds[kind] = self.seconds.get(kind, 0.0) + self.clock() - started


class StopMonitor:
    """Record complete iterates and the first OR trigger in deterministic order."""

    def __init__(
        self, policy: RunControls, work: WorkLedger, *, iteration_unit="iteration"
    ):
        self.policy, self.work = policy, work
        self.iteration_unit = iteration_unit
        self.history = []
        self.reason = None
        self.triggers = []
        self.plateau_count = 0
        self.small_update_count = 0
        self.validation_bad_count = 0
        self.best_validation = None
        self.significant_validation = None
        self.best_iteration = None
        self.stage_id = None

    def set_stage(self, stage_id):
        """Reset update patience at stage boundaries, never the global budget."""
        if stage_id != self.stage_id:
            self.small_update_count = 0
            self.stage_id = stage_id

    def observe(
        self,
        iteration,
        *,
        residual_norm,
        observed_norm,
        objective,
        update_relative=None,
        validation_relative=None,
    ):
        if self.reason is not None:
            return self.reason
        if (
            isinstance(iteration, (bool, np.bool_))
            or not isinstance(iteration, (int, np.integer))
            or iteration < 0
        ):
            raise ValueError("iteration must be a nonnegative integer")
        if self.history and iteration <= self.history[-1]["iteration"]:
            raise ValueError("complete iteration indices must be strictly increasing")
        values = [residual_norm, observed_norm, objective]
        values += [v for v in (update_relative, validation_relative) if v is not None]
        if not np.all(np.isfinite(values)):
            return self.finish("numerical_failure")
        relative = residual_norm / observed_norm if observed_norm > 0 else None
        if self.history and self.policy.objective_rtol is not None:
            previous = self.history[-1]["objective"]
            improvement = (previous - objective) / max(
                abs(previous), np.finfo(float).tiny
            )
            self.plateau_count = (
                self.plateau_count + 1
                if improvement <= self.policy.objective_rtol
                else 0
            )
        if validation_relative is not None:
            if (
                self.best_validation is None
                or validation_relative < self.best_validation
            ):
                self.best_validation = float(validation_relative)
                self.best_iteration = int(iteration)
            meaningful = self.significant_validation is None or validation_relative < (
                self.significant_validation * (1 - self.policy.validation_rtol)
            )
            if meaningful:
                self.significant_validation = float(validation_relative)
                self.validation_bad_count = 0
            else:
                self.validation_bad_count += 1
        self.history.append(
            {
                "iteration": int(iteration),
                "elapsed_s": self.work.elapsed_s,
                "residual_norm": float(residual_norm),
                "relative_residual": relative,
                "objective": float(objective),
                "relative_update": update_relative,
                "stage_id": self.stage_id,
                "validation_relative_residual": validation_relative,
                "work": dict(self.work.counts),
            }
        )
        p = self.policy
        triggers = []
        if p.max_elapsed_s is not None and self.work.elapsed_s >= p.max_elapsed_s:
            triggers.append("time_budget")
        for counter in ("forward_calls", "adjoint_calls"):
            limit = getattr(p, f"max_{counter}")
            if limit is not None and self.work.counts[counter] >= limit:
                triggers.append(f"{counter}_budget")
        # Exact fit and safety budgets do not wait for min_iterations.
        # Exact data fit alone is not stationarity when a penalty remains.
        # A zero *complete* nonnegative objective is a certified minimum.
        if residual_norm == 0 and objective == 0:
            triggers.append("exact_data_fit")
        eligible_small_update = (
            iteration >= p.min_iterations
            and p.update_rtol is not None
            and update_relative is not None
            and update_relative <= p.update_rtol
        )
        self.small_update_count = (
            self.small_update_count + 1 if eligible_small_update else 0
        )
        if iteration >= p.min_iterations:
            if (
                p.target_relative_residual is not None
                and relative is not None
                and relative <= p.target_relative_residual
            ):
                triggers.append("target_residual")
            if (
                p.noise_norm is not None
                and residual_norm <= p.discrepancy_factor * p.noise_norm
            ):
                triggers.append("noise_discrepancy")
            if (
                p.update_rtol is not None
                and update_relative is not None
                and self.small_update_count >= p.update_patience
            ):
                triggers.append("small_model_update")
            if (
                p.objective_rtol is not None
                and self.plateau_count >= p.objective_patience
            ):
                triggers.append("objective_plateau")
            if (
                p.validation_patience is not None
                and self.validation_bad_count >= p.validation_patience
            ):
                triggers.append("validation_plateau")
        if iteration >= p.max_iterations:
            triggers.append("max_iterations")
        # Targets belong to the selected checkpoint, not merely the final iterate.
        self.history[-1]["quality_triggered_rules"] = [
            rule
            for rule in triggers
            if rule
            in (
                "exact_data_fit",
                "target_residual",
                "noise_discrepancy",
            )
        ]
        if triggers:
            self.reason, self.triggers = triggers[0], triggers
        return self.reason

    def finish(self, reason):
        if self.reason is None:
            self.reason, self.triggers = reason, [reason]
        return self.reason

    @property
    def selected_iteration(self):
        if self.policy.restore_best_validation and self.best_iteration is not None:
            return self.best_iteration
        return self.history[-1]["iteration"] if self.history else None

    def record(self):
        selected = self.selected_iteration
        selected_quality_target_met = any(
            row["quality_triggered_rules"]
            for row in self.history
            if row["iteration"] == selected
        )
        return {
            "reason": self.reason or "not_terminated",
            "triggered_rules": list(self.triggers),
            "completed_iterations": (
                self.history[-1]["iteration"] if self.history else 0
            ),
            "selected_iteration": selected,
            "iteration_unit": self.iteration_unit,
            "stage_id": self.stage_id,
            "selected_stage_id": next(
                (
                    row["stage_id"]
                    for row in self.history
                    if row["iteration"] == selected
                ),
                None,
            ),
            "elapsed_s": self.work.elapsed_s,
            "work": dict(self.work.counts),
            "resolved_policy": self.policy.model_dump(),
            "budget_scope": "operator_calls; time_checked_between_calls",
            "has_complete_checkpoint": bool(self.history),
            "quality_target_met": selected_quality_target_met,
            "operator_seconds": dict(self.work.seconds),
            "termination_category": STOP_CATEGORIES[self.reason or "not_terminated"],
        }
