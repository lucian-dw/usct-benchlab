# Mathematical Formulation

This document gives the mathematical model behind the v0.1 algorithms in
`usct-benchlab`. The package uses a common `USCTCase -> ReconstructionResult`
interface, but the registered algorithms make different measurement
assumptions.

## USCT Forward Problem

Ultrasound computed tomography seeks acoustic property maps from transmitted
and scattered waves. A compact abstract model is

```math
d = \mathcal{F}(c,\rho,\alpha;\theta) + \eta,
```

where:

- `d` is receiver-array pressure data or features extracted from it;
- `c(x)` is sound speed and is the main v0.1 reconstruction target;
- `rho(x)` is density;
- `alpha(x)` is acoustic attenuation;
- `theta` contains source, receiver, frequency, grid, and boundary settings;
- `eta` represents modeling and measurement noise.

Different algorithms approximate `mathcal{F}` at different fidelity levels.

## Travel-Time Linearization

For straight-ray sound-speed tomography, each transmitter-receiver pair
defines a ray path `gamma_ij`. With a water or homogeneous reference speed
`c0`, the delay is approximated by

```math
\Delta t_{ij}
=
\int_{\gamma_{ij}}
\left(\frac{1}{c(x)}-\frac{1}{c_0}\right)\,ds.
```

Let

```math
\delta s(x)=\frac{1}{c(x)}-\frac{1}{c_0}.
```

After discretization over image pixels, the model is

```math
A\delta s \approx b,
```

where `A` is a ray-length matrix and `b` stacks the measured or generated
travel-time delays.

## Algebraic Reconstruction Methods

The straight-ray sound-speed solvers use weighted and regularized algebraic
objectives:

```math
\min_{\delta s}
\|W(A\delta s-b)\|_2^2+\lambda^2R(\delta s).
```

`W` encodes valid rays and optional ray weights. `R` is usually a damping or
smoothness penalty.

- `straight_cgls` uses a Krylov least-squares iteration.
- `straight_sirt` uses simultaneous residual backprojection updates.
- `straight_sart` uses ordered transmitter-wise algebraic updates.

After solving for `delta s`, the sound-speed image is recovered by

```math
c(x)=\frac{1}{\delta s(x)+1/c_0}.
```

## Attenuation Tomography

For attenuation, the v0.1 straight-ray model uses a log-amplitude ratio:

```math
-\log |p/p_0| \approx \int_\gamma \alpha(x)\,ds.
```

This again becomes a ray transform, but the unknown is attenuation rather than
slowness. The registered attenuation baseline is `attenuation_sirt`.

## Refraction-Corrected Travel-Time Tomography

In a refracting medium, the direct-arrival travel time for a source `i` can be
modeled by the eikonal equation

```math
|\nabla T_i(x)| = \frac{1}{c(x)},
\qquad
t_{ij} \approx T_i(r_j).
```

A full bent-ray inversion alternates between updating the sound-speed model,
recomputing travel-time fields or ray paths, and solving a nonlinear
least-squares problem:

```math
\min_c \sum_{i,j} |t_{ij}^{obs} - T_i(r_j;c)|^2 + \lambda R(c).
```

The v0.1 command `bent_ray_gn` is intentionally narrower: it is a
bent-ray-style regularized travel-time baseline implemented with the package's
native case format. It records `full_external_eikonal_solver = False` in
metrics/metadata-like outputs and should not be interpreted as a complete
external eikonal solver.

## Weak-Scattering / Ray-Born Model

Weak-scattering methods linearize pressure perturbations around a background
medium. A schematic frequency-domain ray-Born form is

```math
\delta p(\omega,r_i,r_j)
\approx
\int_\Omega
G_0(\omega,r_j,x)
K_\omega(x)
G_0(\omega,x,r_i)
\delta m(x)\,dx.
```

Here `G0` is a background Green's function, `K_omega` is a frequency-dependent
kernel, and `delta m` is a contrast parameter. A complete ray-Born/rWave solver
requires complex frequency-domain pressure data and careful reference-field
handling.

The v0.1 `rwave_adapter` command is an rWave/ray-Born-inspired adapter
baseline using the common package I/O. It records `full_ray_born_solver =
False` and `adapter_style = True`.

## Waveform Inversion

Full waveform inversion uses the pressure data directly. A frequency-domain
sound-speed-only objective can be written as

```math
\min_c
\frac{1}{2}
\sum_{\omega,i,j}
\|P_{\omega,i,j}^{obs}-P_{\omega,i,j}(c)\|_2^2
+ \lambda R(c).
```

The forward pressure `P(c)` is produced by a wave solver such as k-Wave. In
v0.1, `fwi_kwave_adapter` ingests and reports external k-Wave/FWI outputs
rather than embedding a production FWI solver inside this package.

## Mapping to Registered Algorithms

| Registered name | Model family | Primary input | Output |
| --- | --- | --- | --- |
| `straight_cgls` | Straight-ray travel-time least squares | `delta_tof_s` | Sound speed |
| `straight_sirt` | Straight-ray algebraic iteration | `delta_tof_s` | Sound speed |
| `straight_sart` | Straight-ray ordered algebraic iteration | `delta_tof_s` | Sound speed |
| `attenuation_sirt` | Straight-ray log-amplitude tomography | `log_amp` | Attenuation |
| `bent_ray_gn` | Regularized bent-ray-style travel-time baseline | `delta_tof_s` | Sound speed |
| `rwave_adapter` | rWave/ray-Born-inspired adapter baseline | `delta_tof_s` | Sound speed |
| `fwi_kwave_adapter` | External full-wave inversion adapter | k-Wave/FWI artifact | Sound speed |
| `fwi_tiny` | Synthetic waveform sanity model | Synthetic case | Sound speed |
