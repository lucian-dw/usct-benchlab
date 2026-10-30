# Evaluation and stopping

## Observations and isolation

A run records **observation_domain**, **observation_units**, physical model and
measurement provenance. ToF residuals are seconds; pressure residuals retain
declared pressure units. Relative residuals are dimensionless but still describe
different observations and cannot be pooled into one ranking.

For valid training indices $I$,

$$
r_{\rm rel}=\frac{\|F(x)_I-d_I\|_2}{\|d_I\|_2},\qquad
r_{w,\rm rel}=\frac{\|\sqrt{w_I}(F(x)_I-d_I)\|_2}
{\|\sqrt{w_I}d_I\|_2}.
$$

Zero observed norm, empty sets and unavailable predictions are distinct states;
they do not become fabricated zero residuals. Nonfinite predictions on valid
channels are failures. Valid zero observations participate normally.

Specimen receiver/frequency holdouts are excluded from updates, weights and
source fitting. Born may calibrate on an independently acquired water reference
including frequencies held out from specimen fitting. It does not substitute
specimen validation pressure when water calibration is missing.
Reciprocal TX exclusion is explicit. A holdout used for checkpoint
selection is **validation**, not an independent test. GT never controls stopping.

## Execution rules

Native methods share RunControls. The effective budget is the intersection of
the requested budget and host caps; zero means no updates, not a missing value.

| Reason family | Examples | Meaning |
| --- | --- | --- |
| Stationarity | stationary_gradient | Algorithm-specific relative projected-gradient/KKT check |
| Data target | target_residual, noise_discrepancy | Requested residual or supplied noise level reached |
| Stagnation | small_model_update, objective_plateau, validation_plateau | Small change for the declared patience; not proof of convergence |
| Budget | max_iterations, forward_calls_budget, adjoint_calls_budget, time_budget | A supported resource limit reached |
| Completion | schedule_complete | WUST finished its frequency plan; not convergence |
| Failure | numerical_failure, line_search_failed, runtime_failure | Invalid arithmetic, no acceptable trial, or external execution failure |

Relative updates declare optimization/update variables, units, scope and
normalization. Typically native speed solvers report full slowness or full
squared-slowness change,

$$
u_k=\frac{\|z_{k+1}-z_k\|_2}{\|z_k\|_2},
$$

over their declared physical-grid scope. WUST reports slowness change over its
update mask but does **not** use update_rtol to terminate. The default threshold
is unset. Objective/validation patience is likewise opt-in.

| Algorithm / model | Variable used in the update norm | Scope |
| --- | --- | --- |
| Straight CGLS/SIRT/SART | $s_0+\delta s$ | Complete pixel grid; for a reduced model, $s_0+\theta$ on its coefficient grid |
| Bent | $s$ or $s_0+B\theta$ | Complete physical grid, including fixed ROI exterior |
| Fixed Born | $m_0+\delta m$, $m=1/c^2$ | Complete physical grid |
| Nonlinear Born | $m$ | Complete physical grid |
| WUST | $s$ | Explicit update mask; reporting only, not online stopping |

`current_state` holds one accepted model, prediction, objective, iteration and
linearization. A best-validation snapshot omits the linearization. `StopMonitor`
owns scalar history and selection indices, not arrays. `solver_optimality` is
read from the history row for `selected_iteration`; a later stationary point
cannot certify an earlier restored checkpoint. Residual curves are derived from
`iteration_history`, not duplicated in another array. Zero initial residual gives
`data_residual_reduction: null` and `undefined_zero_initial_residual`.

Native budgets are checked between expensive work units. A partial SART sweep
is not published as a complete iteration. Records distinguish selected
checkpoint from final attempted iteration. Native callers needing a hard
wall-clock limit must use a supervising process.

WUST currently supports only frequency-update count and a hard process-group
deadline. It rejects forward/adjoint caps and numerical-convergence rules.
A killed WUST process does not manufacture a final image. Its pre-update losses
at changing frequencies are not final-model residuals.

## Image metrics

With GT, the evaluator reports foreground, full-image and water-background
metrics separately. The primary region defaults to tissue excluding water;
the same explicit/derived speed range is used for PSNR/SSIM comparisons.
RMSE is m/s; PSNR is dB; SSIM is dimensionless.

SSIM averages complete local windows, up to 7 pixels. No complete window means
null plus an unavailable status, never a silently substituted global statistic.
Perfect PSNR uses null plus a perfect-reconstruction status, keeping valid JSON.
Missing GT returns unavailable image scores while data-domain evaluation remains
possible. GT region masks affect reporting only.

See [result lifecycle](result_lifecycle.md) and [numerical tests](physics_validation.md).
