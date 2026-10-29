# Mathematical formulation

## Propagation and inverse problem

For a constant-density lossless acoustic model,

$$
c(x)^{-2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x),
\qquad d_{sr}(t)=\mathcal M_rp_s(t,\cdot)+\eta_{sr}(t).
$$

The maintained reconstruction target is $c(x)$ in m/s. Data generation may
involve additional medium physics, but none of the six algorithms estimates
attenuation. A model-generated surrogate and an imported pressure acquisition
are different observations, even if they share a property map.

## Straight travel time

Let $s=1/c$, $s_0=1/c_0$, $x=s-s_0$. With straight paths,

$$
b_{sr}=\Delta t_{sr}\approx\int_{\gamma_{sr}}x(\xi)\,d\ell,\qquad Ax\approx b.
$$

Siddon intersection lengths form one sparse CSR matrix in meters. Thus $x$ is
s/m and $Ax$ is seconds. Ray weights $w_i$ are precisions, not square-root weights.

CGLS solves a bounded problem, using a quadratic data term or optional Huber loss:

$$
\min_{x\in C}\ \sum_{i\in\mathrm{train}}w_i\,\rho((Ax-b)_i)
+\frac{\lambda^2}{2}\|Lx\|_2^2 .
$$

The default $\rho(r)=r^2/2$. Bounds on speed become inverse bounds on slowness;
ROI pixels outside the update region remain fixed. Projected CG restarts when
constraints or IRLS weights change. A zero data residual alone is not
stationarity of the regularized objective.

For SIRT/SART, $r_i=\sum_j A_{ij}$ and subset column normalizers
$d_j=\sum_{i\in S}w_iA_{ij}$ give

$$
x^+=\Pi_C\left[x+\tau D^{-1}A_S^\top
\operatorname{diag}(w_i/r_i)(b_S-A_Sx)\right].
$$

SIRT uses all training rays simultaneously. SART sweeps all disjoint subsets
before completing one iteration. With inconsistent data, subset cycles and
optional smoothing need not monotonically minimize the global monitored
row-normalized objective. They are not identical to ordinary least squares.

## Eikonal / Bent

$$
|\nabla T_s(x)|=s(x),\qquad F(s)_{sr}=T_s(r).
$$

Native first/second-order fast marching includes a fixed homogeneous-water
discretization correction. Differentiating its active stencil yields $J$ and
$J^\top$; derivatives are local to that stencil, not globally smooth at switches.

$$
\Phi(s)=\frac12\|W(F(s)-t)\|_2^2+
\frac{\lambda^2}{2}\|L(s-s_0)\|_2^2 .
$$

Each Gauss–Newton step solves the weighted linearized problem, then evaluates
the actual projected trial point with Armijo acceptance. Water-relative data use
the same explicit $c_0$. With a reduced basis $B$, updates use $JB$ and the fine
grid penalty $LB$, not a newly invented coarse-grid regularizer.

## Born scattering

Let $m=1/c^2$ and $m_0$ be the background. With time convention
$p(t)=\Re[\hat p(\omega)e^{-i\omega t}]$,

$$
\delta\hat p_{sr}\approx
\omega^2\int_\Omega G_0(r,x)G_0(x,s)q_s\,\delta m(x)\,dx .
$$

Fixed variants solve for $m-m_0$ with frozen background fields. Nonlinear variants
rebuild fields and minimize the actual complex-pressure residual plus a
quadratic penalty. Its coefficient is $\lambda^2$, optionally scaled by the
Jacobian diagonal in the declared pressure units.

WKB fields use Eikonal phase and transport amplitudes. Their continuum Born
sensitivity is an **approximation**, not the exact derivative of this discrete
WKB construction. Full-Green instead solves the discrete volume-integral
equation,

$$
u=u_0+G_0 V(m)u,
\qquad
\delta u=(I-G_0V)^{-1}G_0\,\delta V\,u.
$$

That derivative is tested against the same discrete nonlinear solve. Cell area,
self-cell integration, nonperiodic convolution and complex conjugation are part
of the operator, not fitted display corrections. See [kernel details](ray_born_reference.md).

## WUST waveform inversion

At one frequency, WUST solves its optimized discrete Helmholtz system
$H_\omega(s)u_{\omega,s}=q_s$. For each TX it eliminates a complex source scale
using only admitted receivers:

$$
a_s^\star=\frac{u_s^Hd_s}{u_s^Hu_s},\qquad
\Phi_\omega(s)=\frac12\sum_s\|a_s^\star\mathcal M u_s(s)-d_s\|_2^2 .
$$

The exact discrete slowness gradient uses the adjoint solve and the mass-stencil
transpose. The source-scale derivative cancels by the variable-projection
stationarity condition. No separate explicit $\lambda R(c)$ penalty is added
by this runtime: bounds, update damping/capping and optional ringing removal
are its actual controls.

One entry in the resolved frequency schedule performs one nonlinear conjugate
gradient model update. Repeated entries repeat that frequency. Loss records
belong to the **pre-update** model at the recorded frequency; losses at different
frequencies are not one objective descent curve.

## Implementation map

| Model | Registered command | Mathematical implementation |
| --- | --- | --- |
| Straight ray | straight_cgls / straight_sirt / straight_sart | operators/straight_ray.py; solvers/projected_cg.py, row_action.py |
| Eikonal | bent_ray_gn | operators/eikonal.py; algorithms/bent_ray.py |
| Born | rwave_adapter | operators/ray_born.py, volume_integral.py; solvers/linear.py, nonlinear.py |
| Helmholtz FWI | fwi_wust | algorithms/fwi/algorithm.py; independent WUST Runtime |

See [operator contracts](operator_contract.md) for units/inner products and
[evaluation](evaluation.md) for stopping and reporting.
