# Mathematical Formulation

This document summarizes the mathematical models used by `usct-benchlab` v0.1.
The package uses one data/result interface, but the registered algorithms
correspond to different approximations of the USCT inverse problem.

## PDE Forward Problem

USCT starts from an acoustic wave equation. For source $s$, a simple
sound-speed-only model is

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

Here $p_s(t,x)$ is pressure, $q_s(t,x)$ is the emitted source term, and $c(x)$
is sound speed.

A more general model may include density and attenuation:

$$
\begin{aligned}
\frac{1}{c(x)^2}\partial_{tt}p_s
- \nabla\cdot\left(\frac{1}{\rho(x)}\nabla p_s\right)
+ \mathcal A_\alpha[p_s]
&= q_s.
\end{aligned}
$$

The density field is $\rho(x)$, and $\mathcal A_\alpha$ denotes an attenuation
operator controlled by $\alpha(x)$. Practical solvers also include boundary
conditions, transducer models, grids, and source wavelets.

## Receiver Operator

Receiver $r$ samples the wavefield through a measurement operator
$\mathcal M_r$:

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

The data $d_{sr}(t)$ may be stored as raw time traces, frequency-domain complex
pressure, or derived features such as travel-time delay and log-amplitude
ratio.

## Inverse Problem

The full USCT inverse problem is

$$
\text{recover } c(x),\rho(x),\alpha(x)
\quad
\text{from}
\quad
\{d_{sr}(t)\}_{s,r}.
$$

The v0.1 package focuses primarily on reconstructing the sound-speed map
$c(x)$. Attenuation is supported as a separate straight-ray baseline.

## Straight-Ray Approximation

Straight-ray travel-time tomography approximates the propagation path between
source $s$ and receiver $r$ by a fixed line segment $\gamma_{sr}$. Relative to a
reference speed $c_0$, the delay is

$$
\Delta t_{sr}
\approx
\int_{\gamma_{sr}}
\left(\frac{1}{c(x)}-\frac{1}{c_0}\right)d\ell .
$$

Define the slowness perturbation

$$
\delta s(x)=\frac{1}{c(x)}-\frac{1}{c_0}.
$$

After pixel discretization, the line integrals become

$$
A\delta s \approx b,
$$

where $A$ is the ray-length matrix and $b$ stacks measured or generated
$\Delta t_{sr}$ values.

## Algebraic Reconstruction

The straight-ray sound-speed algorithms solve weighted regularized systems:

$$
\min_{\delta s}
\|W(A\delta s-b)\|_2^2+\lambda^2R(\delta s).
$$

$W$ contains valid-ray masks and optional weights. $R$ is a regularizer such as
damping or smoothness. After solving for $\delta s$, sound speed is recovered
by

$$
c(x)=\frac{1}{\delta s(x)+1/c_0}.
$$

In v0.1:

- `straight_cgls` is a Krylov least-squares solver.
- `straight_sirt` is a simultaneous iterative reconstruction method.
- `straight_sart` is an ordered/subset algebraic reconstruction method.

## Attenuation Tomography

For amplitude-based attenuation tomography, the basic line-integral model is

$$
-\log |p/p_0|
\approx
\int_\gamma \alpha(x)\,d\ell .
$$

The registered `attenuation_sirt` command solves this straight-ray attenuation
problem with an algebraic update.

## Eikonal / Bent-Ray Model

When refraction matters, ray paths depend on the unknown sound speed. A common
high-frequency model is the eikonal equation

$$
|\nabla T_s(x)|=\frac{1}{c(x)},
\qquad
t_{sr}\approx T_s(r).
$$

A full refraction-corrected inversion would solve a nonlinear travel-time
least-squares problem:

$$
\min_c
\sum_{s,r}\left|t_{sr}^{\mathrm{obs}}-T_s(r;c)\right|^2
+\lambda R(c).
$$

The v0.1 `bent_ray_gn` command is a regularized bent-ray-style travel-time
baseline. It records `full_external_eikonal_solver = False` and
`v0_1_backend = "regularized_travel_time_baseline"`.

## Weak-Scattering / Ray-Born Model

Weak-scattering methods linearize pressure perturbations around a background
medium. A schematic ray-Born expression is

$$
\delta \hat p_{sr}(\omega)
\approx
\int_\Omega
G_0(\omega,r,x)K_\omega(x)G_0(\omega,x,s)\delta m(x)\,dx .
$$

$G_0$ is a background Green's function, $K_\omega$ is a frequency-dependent
kernel, and $\delta m(x)$ is a contrast parameter. A complete implementation
requires complex frequency-domain pressure data and careful reference-field
handling.

The v0.1 `rwave_adapter` command is an rWave/ray-Born-inspired adapter
baseline. It records `full_ray_born_solver = False` and
`v0_1_backend = "adapter_style_travel_time_baseline"`.

## FWI PDE-Constrained Objective

Full waveform inversion keeps the PDE forward model in the loop and compares
observed pressure to simulated pressure. In frequency-domain notation:

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

The simulated pressure $\hat p_s(\omega,r;c)$ is constrained by the acoustic
PDE and its discretization. In v0.1, `fwi_kwave_adapter` ingests external
k-Wave/FWI artifacts or calls a configured external pipeline, then reports the
result using the package-standard benchmark outputs.

## Mapping from Models to Commands

| Mathematical model | Registered command | Main input | Main output |
| --- | --- | --- | --- |
| Straight-ray weighted least squares | `straight_cgls` | `delta_tof_s` | Sound speed |
| Simultaneous iterative ray tomography | `straight_sirt` | `delta_tof_s` | Sound speed |
| Ordered/subset algebraic ray update | `straight_sart` | `delta_tof_s` | Sound speed |
| Straight-ray log-amplitude tomography | `attenuation_sirt` | `log_amp` | Attenuation |
| Regularized bent-ray-style travel-time baseline | `bent_ray_gn` | `delta_tof_s` | Sound speed |
| Ray-Born-inspired adapter baseline | `rwave_adapter` | `delta_tof_s` | Sound speed |
| PDE-level full-wave inversion adapter | `fwi_kwave_adapter` | External FWI artifact or command | Sound speed |
| Small waveform-inversion sanity model | `fwi_tiny` | Synthetic waveform case | Sound speed |
