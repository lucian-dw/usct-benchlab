"""Budgeted SIRT/SART with training-only normalization and atomic checkpoints."""

from __future__ import annotations

import numpy as np

from usctbench.core.stopping import BudgetExhausted


def row_action(
    operator,
    control,
    *,
    initial,
    reference,
    project,
    relaxation,
    subsets=1,
    postprocess=None,
):
    """One iteration is a full sweep; each subset's actual operator work is counted.

    SART subsets use disjoint training rays. Normalization excludes held-out
    channels. An interrupted partial sweep returns the last *complete* sweep.
    Physical projection precedes residual evaluation and checkpoint selection.

    SIRT's fixed-point objective has precision w_i / sum_j A_ij, not w_i.
    For subset SART this global objective is only a monitor: inconsistent-data
    limit cycles and postprocessing need not monotonically minimize it.
    """
    x = np.array(initial, dtype=float, copy=True)
    start = x.copy()

    def output():
        selected, metrics = control.output(start)
        metrics["objective_name"] = "row_normalized_weighted_least_squares"
        return selected, metrics

    try:
        row_sum = operator.row_norms(power=1).reshape(control.observed.shape)
        if not np.isfinite(row_sum).all() or np.any(row_sum < 0):
            raise ValueError(
                "row-action solver requires finite nonnegative ray lengths"
            )
        inverse_row_sum = np.divide(
            1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0
        )

        def objective(prediction):
            residual = np.where(
                control.split.train, control.safe_observed - prediction, 0
            )
            return float(
                0.5
                * np.sum(control.precision * np.abs(residual) ** 2 * inverse_row_sum)
            )

        prediction = control.call("forward", operator.forward, x).reshape(
            control.observed.shape
        )
        if control.observe(0, x, prediction, objective=objective(prediction)):
            return output()
        ids = np.flatnonzero(control.split.train)
        groups = np.array_split(ids, min(int(subsets), len(ids)))
        normalization = []
        for group in groups:
            precision = np.zeros_like(control.precision)
            precision.flat[group] = control.precision.flat[group]
            col = control.call("adjoint", operator.adjoint, precision)
            normalization.append((precision, np.where(col > 0, col, 1.0)))
        for iteration in range(1, control.policy.max_iterations + 1):
            previous = x.copy()
            for index, (precision, col) in enumerate(normalization):
                if index:
                    prediction = control.call("forward", operator.forward, x).reshape(
                        control.observed.shape
                    )
                residual = np.where(
                    precision > 0, control.safe_observed - prediction, 0.0
                )
                update = (
                    control.call(
                        "adjoint",
                        operator.adjoint,
                        precision * residual * inverse_row_sum,
                    )
                    / col
                )
                x = project(x + relaxation * update)
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("nonfinite projected iterate")
                control.work.counts["subset_updates"] = (
                    control.work.counts.get("subset_updates", 0) + 1
                )
            if postprocess is not None:
                x = project(postprocess(x, iteration))
            prediction = control.call("forward", operator.forward, x).reshape(
                control.observed.shape
            )
            denominator = np.linalg.norm(reference + previous)
            relative_update = (
                float(
                    np.linalg.norm(x - previous)
                    / max(denominator, np.finfo(float).tiny)
                )
                if denominator > 0
                else None
            )
            if control.observe(
                iteration,
                x,
                prediction,
                objective=objective(prediction),
                update_relative=relative_update,
            ):
                break
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError as exc:
        control.fail(exc)
    return output()
