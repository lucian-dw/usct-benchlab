"""Free-space 2-D Helmholtz volume integral for full Green Born backgrounds.

Solve U = U0 + G0 V U, V = omega^2 * cell_area * (c^-2 - c0^-2).
FFT implements the linear (not periodic) cell convolution; GMRES sums multiple
scattering. This is a collocation reference implementation, not ray shooting,
not the upstream r-Wave package, and not the production MATLAB FWI optimizer.
"""

import numpy as np
from scipy.fft import fft2, ifft2, next_fast_len
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import hankel1


class VolumeIntegralGreen:
    def __init__(
        self,
        grid,
        frequency_hz,
        background,
        exterior_speed,
        *,
        rtol,
        maxiter,
        budget_check=None,
        device=None,
        work_counts=None,
    ):
        self.grid, self.rtol, self.maxiter = grid, float(rtol), int(maxiter)
        self.budget_check = budget_check
        self.device = device
        self.work_counts = work_counts if work_counts is not None else {}
        self.shape = tuple(grid.shape)
        self.fft_shape = tuple(next_fast_len(2 * n - 1) for n in self.shape)
        omega = 2 * np.pi * frequency_hz
        area = float(np.prod(grid.spacing_m))
        self.potential = (
            omega**2
            * area
            * (1 / np.asarray(background, dtype=float) ** 2 - 1 / exterior_speed**2)
        )
        ny, nx = self.shape
        dy = np.r_[
            np.arange(ny),
            np.zeros(self.fft_shape[0] - 2 * ny + 1),
            np.arange(1 - ny, 0),
        ]
        dx = np.r_[
            np.arange(nx),
            np.zeros(self.fft_shape[1] - 2 * nx + 1),
            np.arange(1 - nx, 0),
        ]
        distance = np.hypot(
            dy[:, None] * grid.spacing_m[0], dx[None, :] * grid.spacing_m[1]
        )
        k = omega / exterior_speed
        radius = np.sqrt(area / np.pi)
        kernel = np.zeros(self.fft_shape, dtype=complex)
        nonzero = distance > 0
        kernel[nonzero] = 0.25j * hankel1(0, k * distance[nonzero])
        # Disk-cell average of the logarithmic Green singularity. The equal-area
        # disk is an explicit quadrature approximation for the rectangular cell.
        kernel[0, 0] = 0.5j * hankel1(1, k * radius) / (k * radius) - 1 / (k * k * area)
        kernel[ny : self.fft_shape[0] - ny + 1, :] = 0
        kernel[:, nx : self.fft_shape[1] - nx + 1] = 0
        self.kernel_fft = fft2(kernel)

    def convolve(self, values):
        values = np.asarray(values).reshape(self.shape)
        transformed = fft2(values, s=self.fft_shape)
        return ifft2(self.kernel_fft * transformed)[: self.shape[0], : self.shape[1]]

    def _matvec(self, values):
        if self.budget_check is not None:
            self.budget_check()
        self.work_counts["green_matvecs"] = self.work_counts.get("green_matvecs", 0) + 1
        field = np.asarray(values).reshape(self.shape)
        return (field - self.convolve(self.potential * field)).ravel()

    def fields(self, incident, *, return_device=False):
        if self.device is not None:
            import cupy as cp
            from usctbench.operators.cuda_green import solve_fields

            fields = solve_fields(self, incident, self.device)
            return fields if return_device else cp.asnumpy(fields)
        incident = np.asarray(incident, dtype=complex)
        n = int(np.prod(self.shape))
        operator = LinearOperator((n, n), matvec=self._matvec, dtype=complex)
        result = np.empty_like(incident)
        for index, rhs in enumerate(incident):
            if self.budget_check is not None:
                self.budget_check()

            def completed_iteration(_):
                if self.budget_check is not None:
                    self.budget_check()

            self.work_counts["green_solves"] = (
                self.work_counts.get("green_solves", 0) + 1
            )
            value, info = gmres(
                operator,
                rhs,
                x0=rhs,
                rtol=self.rtol,
                atol=0.0,
                restart=min(40, rhs.size),
                maxiter=self.maxiter,
                callback=completed_iteration,
                callback_type="pr_norm",
            )
            residual = float(
                np.linalg.norm(self._matvec(value) - rhs) / np.linalg.norm(rhs)
            )
            converged = (
                info == 0 and np.isfinite(residual) and residual <= self.rtol * 1.01
            )
            if not converged:
                raise FloatingPointError(
                    f"volume-integral Green solve failed: info={info}, relative residual={residual:.3g}"
                )
            result[index] = value
        return result
