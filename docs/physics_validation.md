# Maintained numerical checks

| Property | Test location |
| --- | --- |
| Siddon lengths and exact transpose on rectangular grids | tests/operators/test_straight_csr.py |
| Eikonal homogeneous water, Snell reference, active-stencil FD/adjoint | tests/operators/test_eikonal.py |
| Born point scatterer, real/complex adjoint, Full-Green derivative/remainder | tests/operators/test_ray_born.py |
| Reduced basis, physical-grid penalty and transpose | tests/operators/test_model_space.py |
| Weighted/complex/rank-deficient linear solves against dense references | tests/solvers/test_linear.py |
| Bounded/Huber/regularized solutions against independent optimization | tests/solvers/test_projected_cg.py |
| GN directions, both actual trial policies, Armijo, KKT and relative stationarity | tests/solvers/test_nonlinear.py |
| Bent weighted coarse AB/LB, physical mapping and native objective derivative | tests/algorithms/test_bent_ray.py; test_reduced_model.py |
| Actual native reconstruction and held-out data isolation | tests/algorithms/test_native_inversions.py; test_relinearized_ray_born.py |
| Absolute-time DTFT, axes and source calibration | tests/data/test_waveforms.py; test_calibration.py |
| Budget/stagnation and complete checkpoints | tests/core/test_run_controls.py |
| Failure propagation and numerical-save-before-report | tests/cli/test_benchmark_runner.py |
| Pressure contract, separate time/frequency water and first-read units | tests/data/test_pressure.py; test_waveforms.py; tests/core/test_io.py |
| Initial Born background reuse and configured phase reference | tests/algorithms/test_relinearized_ray_born.py |
| Executed Green work and shared device-field cache | tests/operators/test_ray_born.py |

## Shared GN test ownership

The former Bent solver-level checks now have one owner:

| Behavior | Maintained test in tests/solvers/test_nonlinear.py |
| --- | --- |
| Uphill smoothing vs regularized dense optimum (merged) | test_nonlinear_solver_rejects_uphill_smoothing_and_recovers_dense_solution |
| Clipped uphill Newton vs feasible gradient (merged) | test_projected_uphill_newton_uses_projected_gradient_without_evaluation |
| Zero inner step with nonzero feasible gradient (moved) | test_zero_inner_update_does_not_claim_stationarity |
| Bound KKT without an inner solve/trial (moved) | test_bound_kkt_stationarity_requires_no_newton_or_trial |
| Recoverable/all-failing nonfinite trial (moved) | test_nonfinite_trials_are_rejected_without_changing_checkpoint |
| Objective-scale-invariant relative gradient (moved) | test_relative_gradient_stopping_is_invariant_to_objective_scale |

`test_disabling_backtracking_does_not_allow_ascent` remains in the Bent algorithm
file as a short public parameter-wiring test. Independent coarse-space dense
references and real Eikonal/Green/WUST derivatives are not replaced by mock
solver checks. Both GN trial formulas are exercised with their actual feasible
displacements, not only a monotonic objective assertion.

These are numerical checks within specified models, not clinical performance
claims. Normal CI runs pytest once. Manual MATLAB CPU/GPU and cross-repository
commands are in [FWI](fwi.md); WUST's native comparison tests actual source-batch
encoding against independent MATLAB k-Wave simulation. Research-scale image
comparisons and performance campaigns live in [research archive](research_index.md).
