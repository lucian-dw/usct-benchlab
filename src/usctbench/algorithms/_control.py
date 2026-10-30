"""Shared run bookkeeping. Only training samples enter an optimization update."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from usctbench.core.stopping import StopMonitor, WorkLedger
from usctbench.evaluation.data import make_data_split


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
        self.precision = self.split.train_weights
        self.safe_observed = np.where(self.split.train, self.observed, 0)
        self.current_state = None
        self.best_state = None
        self.inner_solver_history = []
        self.pending_linear_info = None
        self.numerical_error = None

    @property
    def selected_state(self):
        if self.policy.restore_best_validation and self.best_state is not None:
            return self.best_state
        return self.current_state

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
        linearization=None,
    ):
        if self.monitor.reason is not None:
            return self.monitor.reason
        if not np.all(np.isfinite(state)):
            return self.monitor.finish("numerical_failure")
        train = self.split.statistics(prediction, "train")
        validation = self.split.statistics(prediction, "validation")
        previous_best = self.monitor.best_iteration
        previous_history_length = len(self.monitor.history)
        reason = self.monitor.observe(
            iteration,
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
            self.current_state = AcceptedState(
                np.array(state, copy=True),
                np.array(prediction, copy=True),
                self.monitor.history[-1]["objective"],
                iteration,
                linearization,
            )
            if self.monitor.best_iteration != previous_best:
                current = self.current_state
                self.best_state = AcceptedState(
                    current.model.copy(),
                    current.prediction.copy(),
                    current.objective,
                    current.iteration,
                )
        return reason

    def output(self, fallback):
        selected = self.selected_state
        state = np.array(fallback if selected is None else selected.model, copy=True)
        prediction = None if selected is None else selected.prediction
        stop = self.monitor.record()
        metrics = {
            "stop_reason": stop["reason"],
            "stopping": stop,
            "iterations": stop["completed_iterations"],
            "iteration_history": self.monitor.history,
            "evaluation_split": self.split.metadata,
            "roi_mask_provided": self.case.grid.roi_mask is not None,
            "roi_update_restriction_active": bool(
                self.config.parameters.roi_update_only
                and self.case.grid.roi_mask is not None
            ),
        }
        if self.numerical_error is not None:
            metrics["numerical_error"] = (
                f"{type(self.numerical_error).__name__}: {self.numerical_error}"
            )
        if self.config.diagnostics and self.inner_solver_history:
            metrics["inner_solver_history"] = self.inner_solver_history
        if prediction is not None:
            evaluation = self.split.evaluate(prediction)
            metrics["evaluation"] = evaluation
            train = evaluation["train"]
            metrics["data_relative_residual"] = train["weighted_relative_residual"]
            metrics["data_residual_norm"] = train["weighted_residual_norm"]
            first = self.monitor.history[0]["residual_norm"]
            metrics["initial_data_residual_norm"] = first
            metrics["data_residual_reduction"] = (
                1 - train["weighted_residual_norm"] / first if first else None
            )
            metrics["data_residual_reduction_status"] = (
                "ok" if first else "undefined_zero_initial_residual"
            )
        return state, metrics
