"""Optional CuPy acceleration of the SAME complex128 volume-integral model.

No CUDA dependency is imported on the default CPU path. CPU/GPU forward,
directional-derivative and adjoint agreement are required acceptance gates.
"""

import numpy as np


def solve_fields(solver, incident, device):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, gmres

    with cp.cuda.Device(device):
        potential = cp.asarray(solver.potential)
        kernel = cp.asarray(solver.kernel_fft)

        def matvec(values):
            if solver.budget_check is not None:
                solver.budget_check()
            solver.work_counts["green_matvecs"] = (
                solver.work_counts.get("green_matvecs", 0) + 1
            )
            field = values.reshape(solver.shape)
            transform = cp.fft.fft2(potential * field, s=solver.fft_shape)
            convolution = cp.fft.ifft2(kernel * transform)[
                : solver.shape[0], : solver.shape[1]
            ]
            return (field - convolution).ravel()

        n = int(np.prod(solver.shape))
        operator = LinearOperator((n, n), matvec=matvec, dtype=cp.complex128)
        incident = cp.asarray(incident, dtype=cp.complex128)
        result = cp.empty_like(incident)
        restart = min(40, n)
        for index, rhs in enumerate(incident):
            if solver.budget_check is not None:
                solver.budget_check()

            def completed_iteration(_):
                if solver.budget_check is not None:
                    solver.budget_check()

            b = rhs
            solver.work_counts["green_solves"] = (
                solver.work_counts.get("green_solves", 0) + 1
            )
            value, info = gmres(
                operator,
                b,
                x0=b,
                atol=0,
                restart=restart,
                # CuPy 13 counts inner iterations; CPU maxiter counts restarts.
                maxiter=restart * solver.maxiter,
                tol=solver.rtol,
                callback=completed_iteration,
            )
            residual = float(cp.linalg.norm(matvec(value) - b) / cp.linalg.norm(b))
            if info != 0 or not np.isfinite(residual) or residual > solver.rtol * 1.01:
                raise FloatingPointError(
                    f"CUDA Green solve failed: info={info}, residual={residual}"
                )
            result[index] = value
        return result
