# Maintained numerical checks

| Property | Test location |
| --- | --- |
| Siddon lengths and exact transpose on rectangular grids | tests/operators/test_straight_csr.py |
| Eikonal homogeneous water, Snell reference, active-stencil FD/adjoint | tests/operators/test_eikonal.py |
| Born point scatterer, real/complex adjoint, Full-Green derivative/remainder | tests/operators/test_ray_born.py |
| Reduced basis, physical-grid penalty and transpose | tests/operators/test_model_space.py |
| Weighted/complex/rank-deficient linear solves against dense references | tests/solvers/test_linear.py |
| Bounded/Huber/regularized solutions against independent optimization | tests/solvers/test_projected_cg.py |
| Projected GN, accepted objective and fixed ROI | tests/solvers/test_nonlinear.py; tests/algorithms/test_bent_ray.py |
| Actual native reconstruction and held-out data isolation | tests/algorithms/test_native_inversions.py; test_relinearized_ray_born.py |
| Absolute-time DTFT, axes and source calibration | tests/data/test_waveforms.py; test_calibration.py |
| Budget/stagnation and complete checkpoints | tests/core/test_run_controls.py |
| Failure propagation and numerical-save-before-report | tests/cli/test_benchmark_runner.py |

These are numerical checks within specified models, not clinical performance
claims. Normal CI runs pytest once. Manual MATLAB CPU/GPU and cross-repository
commands are in [FWI](fwi.md); WUST's native comparison tests actual source-batch
encoding against independent MATLAB k-Wave simulation. Research-scale image
comparisons and performance campaigns live in [research archive](research_index.md).
