# Born and Full-Green kernels

The maintained model uses $m=c^{-2}$ and $e^{-i\omega t}$. In homogeneous 2-D
water the outgoing kernel is

$$
G_0(r)=\frac{i}{4}H_0^{(1)}(k_0r),\qquad k_0=\omega/c_0.
$$

The pressure derivative combines source and receiver fields:

$$
(J\delta m)_{fsr}=\omega_f^2
\sum_j G_f(r,j)u_f(j,s)\,\delta m_j\,\Delta x\Delta y.
$$

Its adjoint takes the **real part** after conjugating the complex kernel,
because $\delta m$ is real. Source amplitudes scale both the background pressure
and the Jacobian. Fitting background pressure alone is insufficient.

## Discrete volume integral

The self-cell singularity is integrated over an equal-area disk rather than
evaluated at zero distance. FFT padding implements **linear**, not circular,
convolution. The volume-integral system includes $\omega^2(m-m_0)$ and cell
area in its convolution kernel. Its iterative solve reports achieved residual,
iteration count and status. Cache size changes storage, not the mathematical
operator.

Full-Green nonlinear updates differentiate this same system with distorted
background fields. WKB substitutes Eikonal phase and transport amplitude; its
Born sensitivity is explicitly approximate. Neither variant implements all
upstream r-Wave ray-shooting choices.

Implementation: [ray_born.py](../src/usctbench/operators/ray_born.py),
[volume_integral.py](../src/usctbench/operators/volume_integral.py).
Tests use a point-scatterer reference, an independent dense multiple-scattering
system, an adjoint relation and a second-order remainder.
