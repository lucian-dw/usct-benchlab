# Algorithms

All methods return sound speed in m/s. Parameters are defined only in
[typed models](../src/usctbench/algorithms/parameters.py) and
[FWI parameters](../src/usctbench/algorithms/fwi/parameters.py).
Use describe-algorithm for the current schema; YAML files are overrides.

| Command | Input / unknown | Objective and iteration | Main cost / limitation |
| --- | --- | --- | --- |
| straight_cgls | Differential ToF / $\delta s$ | Weighted quadratic or Huber data fit with $\lambda^2\|L\delta s\|^2/2$; projected CG/IRLS | Sparse forward/transpose; straight rays neglect refraction |
| straight_sirt | Differential ToF / $\delta s$ | Simultaneous row/column-normalized corrections | Sparse forward/transpose; row-normalized objective differs from ordinary LS |
| straight_sart | Differential ToF / $\delta s$ | Ordered subset corrections; one iteration is one complete sweep | Inconsistent data can produce subset cycles |
| bent_ray_gn | First-arrival ToF / $s$ | Native fast marching, discrete Jacobian, regularized GN and projected line search | Rebuild travel times each accepted/trial model; first arrivals omit scattering |
| born | Complex pressure / $m=1/c^2$ | Fixed Born linear solve or relinearized nonlinear objective | WKB Green approximation or volume-integral Green solves; explicit calibration required |
| fwi_wust | Complex total pressure / $s$ | Source-projected Helmholtz objective; one frequency entry is one model update | External MATLAB CPU or CUDA MEX; currently budget/deadline stopping only |

## Born variants

- **wkb_fixed**: frozen Eikonal phase/transport amplitudes and linear Born contrast.
- **wkb_nonlinear**: rebuild WKB background; its continuum Born direction is not
  the exact derivative of the discrete WKB forward implementation. Line search
  evaluates the actual nonlinear objective and may reject the direction.
- **full_green_fixed**: volume-integral background Green fields, fixed linearization.
- **full_green_nonlinear**: rebuild volume-integral fields, with the corresponding
  discrete Born derivative, including multiple scattering in the background.

The Full-Green solve is a volume-integral discretization, not WUST's optimized
finite-difference Helmholtz stencil. Neither name claims upstream r-Wave parity.
All pressure inputs need a compatible source spectrum or independent water
reference; unit sources are admitted only when explicitly declared.

## Scope

Bounds, ROI and optional reduced grids are physical-model choices. Coefficient
models map through a bilinear basis; the penalty remains on the fine physical
grid. Optional smoothing and early termination alter the delivered numerical
solution and are recorded. No method is required to reproduce FWI's image to
pass its own mathematical validation.

See [formulation](math_formulation.md), [operator contracts](operator_contract.md)
and [stopping](evaluation.md).
