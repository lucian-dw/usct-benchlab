"""Shared run bookkeeping. Only training samples enter an optimization update."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from usctbench.core.stopping import StopMonitor, WorkLedger
from usctbench.evaluation.data import make_data_split, residual_statistics


@dataclass
class AcceptedState:
    model: np.ndarray
    prediction: np.ndarray
    objective: float
    iteration: int
    linearization: object = None


class InversionControl:
    def __init__(
        self,
        case,
        config,
        observed,
        *,
        weights=None,
        valid_mask=None,
        iteration_unit="iteration",
    ):
        self.case, self.config = case, config
        self.observed = np.asarray(observed)
        settings = config.evaluation.model_dump(exclude_none=True)
        self.split = make_data_split(
            self.observed,
            valid_mask=valid_mask,
            weights=weights,
            tx_positions=case.geometry.tx_pos_m,
            rx_positions=case.geometry.rx_pos_m,
            **settings,
        )
        self.policy = config.run_controls
        if self.policy.max_iterations is None:
            raise ValueError(
                "prepare_config must resolve iteration budget before numerical execution"
            )
        if self.policy.validation_patience is not None and not np.any(
            self.split.validation
        ):
            raise ValueError("validation_patience requires a nonempty hold-out split")
        self.work = WorkLedger(self.policy)
        self.monitor = StopMonitor(
            self.policy, self.work, iteration_unit=iteration_unit
        )
        self.precision = np.where(self.split.train, self.split.weights, 0.0)
        self.safe_observed = np.where(self.split.train, self.observed, 0)
        self.last_prediction = self.best_prediction = None
        self.current_state = None
        self.inner_solver_history = []
        self.pending_linear_info = None
        self.numerical_error = None
        self.update_semantics = {
            "optimization_variable": None,
            "update_variable": None,
            "update_units": None,
            "update_norm": None,
            "update_norm_scope": None,
            "update_normalization": None,
        }

    def declare_update(
        self,
        optimization_variable,
        update_variable,
        units,
        *,
        scope="full_physical_grid",
        normalization=None,
    ):
        """Describe the actual statistic without changing the numerical path."""
        self.update_semantics = {
            "optimization_variable": optimization_variable,
            "update_variable": update_variable,
            "update_units": units,
            "update_norm": "euclidean_l2",
            "update_norm_scope": scope,
            "update_normalization": normalization
            or "norm(q_new-q_old)/max(norm(q_old),float64_tiny)",
        }

    def call(self, kind, function, *args, **kwargs):
        return self.work.call(kind, function, *args, **kwargs)

    def fail(self, exception):
        self.numerical_error = exception
        self.monitor.finish("numerical_failure")

    def record_linear(self, info):
        self.pending_linear_info = asdict(info)
        if self.config.diagnostics:
            self.inner_solver_history.append(self.pending_linear_info)

    def weighted_residual(self, prediction):
        # Index first: 0 * NaN must never poison gradients on excluded channels.
        return (
            np.where(self.split.train, self.safe_observed - prediction, 0)
            * self.precision
        )

    def observe(
        self,
        iteration,
        state,
        prediction,
        *,
        objective=None,
        update_relative=None,
        optimizer_diagnostics=None,
        linearization=None,
    ):
        if self.monitor.reason is not None:
            return self.monitor.reason
        if optimizer_diagnostics is not None and any(
            value is not None and not np.isfinite(value)
            for value in optimizer_diagnostics.values()
        ):
            return self.monitor.finish("numerical_failure")
        train = residual_statistics(
            prediction, self.observed, mask=self.split.train, weights=self.split.weights
        )
        validation = residual_statistics(
            prediction,
            self.observed,
            mask=self.split.validation,
            weights=self.split.weights,
        )
        previous_best = self.monitor.best_iteration
        previous_history_length = len(self.monitor.history)
        reason = self.monitor.observe(
            iteration,
            state,
            residual_norm=train["weighted_residual_norm"],
            observed_norm=train["weighted_observed_norm"],
            objective=(
                0.5 * train["weighted_residual_norm"] ** 2
                if objective is None
                else objective
            ),
            update_relative=update_relative,
            validation_relative=validation["weighted_relative_residual"],
        )
        # A rejected nonfinite state/objective cannot advance the data cache.
        # Its image and its prediction must refer to the same complete iterate.
        if len(self.monitor.history) > previous_history_length:
            if self.pending_linear_info is not None:
                self.monitor.history[-1]["linear_solve"] = self.pending_linear_info
                self.pending_linear_info = None
            if optimizer_diagnostics is not None:
                self.monitor.history[-1]["optimizer_diagnostics"] = dict(
                    optimizer_diagnostics
                )
            self.last_prediction = np.array(prediction, copy=True)
            self.current_state = AcceptedState(
                self.monitor.last_state,
                self.last_prediction,
                self.monitor.history[-1]["objective"],
                iteration,
                linearization,
            )
            if self.monitor.best_iteration != previous_best:
                self.best_prediction = self.last_prediction.copy()
        return reason

    def output(self, fallback):
        state, _ = self.monitor.selected_state()
        if state is None:
            state = np.array(fallback, copy=True)
        selected_best = (
            self.policy.restore_best_validation and self.best_prediction is not None
        )
        prediction = self.best_prediction if selected_best else self.last_prediction
        stop = self.monitor.record()
        stop.update(self.update_semantics)
        metrics = {
            "stop_reason": stop["reason"],
            "stopping": stop,
            "iterations": stop["completed_iterations"],
            "iteration_history": self.monitor.history,
            "evaluation_split": self.split.metadata,
            "heldout_role": "validation_for_stopping_and_checkpoint_selection_not_untouched_test",
            "roi_mask_provided": self.case.grid.roi_mask is not None,
            "roi_update_restriction_active": bool(
                self.config.parameters.roi_update_only
                and self.case.grid.roi_mask is not None
            ),
            "residual_curve": [row["residual_norm"] for row in self.monitor.history],
        }
        if self.numerical_error is not None:
            metrics["numerical_error"] = (
                f"{type(self.numerical_error).__name__}: {self.numerical_error}"
            )
        if self.config.diagnostics and self.inner_solver_history:
            metrics["inner_solver_history"] = self.inner_solver_history
        if prediction is not None:
            evaluation = self.split.evaluate(prediction, self.observed)
            metrics["evaluation"] = evaluation
            train = evaluation["train"]
            metrics["data_relative_residual"] = train["weighted_relative_residual"]
            metrics["data_residual_norm"] = train["weighted_residual_norm"]
            first = self.monitor.history[0]["residual_norm"]
            metrics["initial_data_residual_norm"] = first
            metrics["data_residual_reduction"] = (
                1 - train["weighted_residual_norm"] / first if first else 0.0
            )
        return state, metrics
