# Operator contract

## Layout and units

| Quantity | BenchLab convention |
| --- | --- |
| Image, mask | [row=y, col=x] |
| Grid shape / spacing / origin | [ny,nx] / [dy,dx] in m / outer pixel edge [y,x] in m |
| Element positions | [element, yx] in m |
| Travel time | [tx,rx], seconds |
| Pressure | [frequency,tx,rx], complex; units explicitly declared |
| Time pressure | [time,tx,rx], with actual sample times |
| Source spectrum | prepared [frequency,tx], complex |
| WUST boundary | positions changed once to [x,y]; images stay [y,x] |

## Canonical numerical modules

Linear operators declare `data_dtype` independently of a particular observation.
Complex-valued Jacobians use both real and imaginary rows in real-model least
squares even when the supplied residual happens to be real. Reduced operators
preserve this declaration. Inner solvers report native residual estimates,
without spending extra forward/adjoint work to recertify after termination.

- **operators/straight_ray.py**: one CSR Siddon matrix. Forward $Ax$ and adjoint
  $A^\top y$ share identical ordering and lengths.
- **operators/eikonal.py**: fast marching and its local linearization. A stored
  active-stencil tape supplies forward sensitivities and reverse adjoint.
- **operators/ray_born.py** and **volume_integral.py**: complex pressure fields and Born
  sensitivities. Full-Green differentiates the discrete volume-integral solve;
  WKB uses a continuum sensitivity approximation.
- **operators/model_space.py**: bilinear coefficient basis $B$ and reduced $JB$.
- **operators/regularization.py**: fine-grid penalties and exact transposes.

A linear operator exposes forward and adjoint. A nonlinear forward exposes
linearize(model), returning its value, Jacobian and derivative kind.
Jacobians act on **perturbations of the declared variable**, not an arbitrary
sound-speed increment.

## Inner products

Arrays use Euclidean sums. A real model and complex data satisfy

$$
\Re\langle Jv,z\rangle_\mathbb C=\langle v,J^\ast z\rangle_\mathbb R.
$$

Pixel area is already included in the Born kernel and ray lengths in the
straight matrix; callers must not multiply either again. Weights are precision
weights; weighted least squares uses $\sqrt w$ in an augmented system.

Unknowns are straight $\delta s$ [s/m], Eikonal $s$ [s/m], Born $m$ [s²/m²],
and WUST $s$ [s/m]. Speed conversion and box projection belong to solver updates.

Invalid observations are masked before arithmetic, not filled with valid zeros.
An actual zero measurement remains valid. Specimen holdouts never enter training
normalization or source fitting. Independent water calibration is a distinct
acquisition, not specimen validation data. Observed masks are not GT-derived support.

## Boundaries

Prepared numerical objects are reused internally. Public inputs validate shape,
units, coordinates and conventions. Internal import paths point to canonical
operators, not historical forward/adjoint wrappers.

[Mathematics](math_formulation.md) · [algorithms](algorithms.md)
