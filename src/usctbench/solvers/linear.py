"""Real-model linearized least squares with one augmented scaling step.

Assemble A = [Re sqrt(W) J; Im sqrt(W) J; sqrt(mu) L], with the
imaginary block only for complex data. The input residual is W * (y - F).
Normalize the entire augmented problem, not its data and penalty separately.
"""

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from usctbench.operators.base import LinearSolveInfo
from usctbench.operators.regularization import regularizer


def solve_linearized(
    jacobian,
    residual,
    current,
    control,
    *,
    weights,
    iterations,
    damping=0.0,
    regularization="identity",
    roi=None,
    method="lsmr",
    rtol=1e-7,
    atol=1e-10,
    btol=1e-10,
    conlim=1e8,
    preconditioner="none",
    preconditioner_probes=8,
    preconditioner_seed=0,
):
    """Return a physical step and native estimates, not nonlinear convergence.

    residual is already W * (y - F(q)); current is q - q_prior.
    damping is the objective weight mu = lambda_config**2 * kappa.
    Column scaling changes only the coordinates of the inner solve.
    """
    if method not in {"normal_cg", "lsmr", "lsqr"}:
        raise ValueError("inner solver must be normal_cg, lsmr or lsqr")
    if preconditioner not in {"none", "column_rms"}:
        raise ValueError("preconditioner must be none or column_rms")
    current = np.asarray(current)
    if np.iscomplexobj(current) or not np.isfinite(current).all():
        raise ValueError("current must be a finite real model")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    current = current.astype(float, copy=False)
    active = np.ones(current.shape, bool) if roi is None else np.asarray(roi, bool)
    if active.shape != current.shape:
        raise ValueError("ROI shape must match model")
    precision = np.asarray(weights, dtype=float)
    if not np.isfinite(precision).all() or np.any(precision < 0):
        raise ValueError("precision must be finite and nonnegative")
    residual = np.asarray(residual).reshape(precision.shape)
    selected = precision > 0
    sqrt_w = np.sqrt(precision[selected])
    complex_data = np.issubdtype(
        jacobian.data_dtype, np.complexfloating
    ) or np.iscomplexobj(residual)
    ndata, nmodel = sqrt_w.size, int(active.sum())
    data_rows = ndata * (2 if complex_data else 1)
    sqrt_mu = np.sqrt(damping)
    reg_current = (
        np.asarray(regularizer(current, regularization)) if damping else np.empty(0)
    )
    reg_shape = reg_current.shape

    def expand(free):
        image = np.zeros(current.shape)
        image[active] = np.asarray(free).ravel()
        return image

    def matvec(free):
        image = expand(free)
        prediction = np.asarray(
            control.call("jacobian", jacobian.forward, image)
        ).reshape(precision.shape)
        # Index before arithmetic; excluded channels may contain NaN.
        data = sqrt_w * prediction[selected]
        parts = [data.real, data.imag] if complex_data else [data]
        if damping:
            parts.append(
                sqrt_mu * np.asarray(regularizer(image, regularization)).ravel()
            )
        return np.concatenate(parts)

    def rmatvec(vector):
        data = vector[:ndata]
        if complex_data:
            data = data + 1j * vector[ndata:data_rows]
        full = np.zeros(precision.shape, dtype=complex if complex_data else float)
        full[selected] = sqrt_w * data
        image = np.asarray(
            control.call("adjoint", jacobian.adjoint, full)
        ).real.reshape(current.shape)
        if damping:
            reg_data = vector[data_rows:].reshape(reg_shape)
            reg_adj = (
                regularization.adjoint(reg_data)
                if hasattr(regularization, "adjoint")
                else regularizer(reg_data, regularization)
            )
            image = image + sqrt_mu * reg_adj
        return image[active]

    target_data = residual[selected] / sqrt_w
    parts = [target_data.real, target_data.imag] if complex_data else [target_data]
    if damping:
        parts.append(-sqrt_mu * reg_current.ravel())
    target = np.concatenate(parts)
    if np.iscomplexobj(target) or not np.isfinite(target).all():
        raise FloatingPointError("nonfinite or nonreal augmented right-hand side")
    operator = LinearOperator(
        (data_rows + reg_current.size, nmodel),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    rhs_scale = float(np.max(np.abs(target), initial=0))
    if rhs_scale == 0 or nmodel == 0:
        return np.zeros_like(current), LinearSolveInfo(
            method, 0, "stationary_subproblem", 0.0, float(norm(target)), True
        )
    scaled_target = target / rhs_scale
    # One charged A* and A establish a representative physical operator scale.
    initial_gradient = operator.rmatvec(scaled_target)
    initial_norm = norm(initial_gradient)
    if initial_norm == 0:
        return np.zeros_like(current), LinearSolveInfo(
            method, 0, "stationary_subproblem", 0.0, float(norm(target)), True
        )
    operator_scale = float(norm(operator.matvec(initial_gradient / initial_norm)))
    if not np.isfinite(operator_scale) or operator_scale == 0:
        raise FloatingPointError("inconsistent adjoint or unresolved operator scale")
    normalized = LinearOperator(
        operator.shape,
        matvec=lambda z: operator.matvec(z) / operator_scale,
        rmatvec=lambda v: operator.rmatvec(v) / operator_scale,
        dtype=float,
    )
    scale = np.ones(nmodel)
    if preconditioner == "column_rms":
        rng = np.random.default_rng(preconditioner_seed)
        columns = np.zeros(nmodel)
        for _ in range(preconditioner_probes):
            probe = rng.choice([-1.0, 1.0], size=operator.shape[0])
            product = normalized.rmatvec(probe)
            columns += product**2 / preconditioner_probes
        positive = columns[columns > 0]
        if positive.size:
            floor = max(float(np.median(positive)) * 1e-3, np.finfo(float).tiny)
            scale = 1 / np.sqrt(np.maximum(columns, floor))
            scale /= np.median(scale)
            scale = np.clip(scale, 1e-3, 1e3)
    scaled = LinearOperator(
        operator.shape,
        matvec=lambda z: normalized.matvec(scale * z),
        rmatvec=lambda v: scale * normalized.rmatvec(v),
        dtype=float,
    )
    initial_gradient = scale * initial_gradient / operator_scale
    options = dict(atol=atol, btol=btol, conlim=conlim)
    if method == "lsmr":
        z, code, completed, normr, normar, _, acond, _ = lsmr(
            scaled, scaled_target, maxiter=iterations, **options
        )
    elif method == "lsqr":
        result = lsqr(scaled, scaled_target, iter_lim=iterations, **options)
        z, code, completed, normr, _, _, acond, normar = result[:8]
    else:
        z, code, completed, normr, normar = _normal_cg(
            scaled, scaled_target, initial_gradient, iterations, rtol
        )
        acond = None
    control.work.counts["inner_iterations"] = control.work.counts.get(
        "inner_iterations", 0
    ) + int(completed)
    with np.errstate(over="raise", invalid="raise"):
        free = (rhs_scale / operator_scale) * scale * z
    if not np.isfinite(free).all():
        raise FloatingPointError("nonfinite physical linearized step")
    relative = float(normar / norm(initial_gradient))
    converged = bool(code in {0, 1, 2, 4, 5} and relative <= rtol)
    reasons = {
        0: "zero_solution",
        1: "compatible_backward_error",
        2: "least_squares_backward_error",
        3: "condition_limit",
        4: "machine_compatible_error",
        5: "machine_least_squares_error",
        6: "machine_condition_limit",
        7: "iteration_limit",
    }
    return expand(free), LinearSolveInfo(
        method,
        int(completed),
        "normal_residual_target" if converged else reasons[int(code)],
        relative,
        float(rhs_scale * normr),
        converged,
        int(code),
        None if acond is None else float(acond),
        normal_residual_norm=(
            float(operator_scale * rhs_scale * normar)
            if preconditioner == "none"
            else float(normar)
        ),
        normal_residual_scope=(
            "physical_augmented"
            if preconditioner == "none"
            else "transformed_system_estimate"
        ),
    )


def _normal_cg(operator, target, gradient, iterations, rtol):
    """CG on the normalized normal equations; use recursive residual estimates."""
    solution = np.zeros(operator.shape[1])
    gradient = gradient.copy()
    direction = gradient.copy()
    data_residual = -target.copy()
    initial_norm = norm(gradient)
    gamma = float(np.vdot(gradient, gradient).real)
    completed = 0
    for _ in range(iterations):
        if norm(gradient) <= rtol * initial_norm:
            break
        product = operator.matvec(direction)
        curvature = float(np.vdot(product, product).real)
        if not np.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError("nonpositive or nonfinite normal-system curvature")
        alpha = gamma / curvature
        solution += alpha * direction
        data_residual += alpha * product
        gradient -= alpha * operator.rmatvec(product)
        next_gamma = float(np.vdot(gradient, gradient).real)
        direction = gradient + (next_gamma / gamma) * direction
        gamma = next_gamma
        completed += 1
    normar = float(norm(gradient))
    code = 2 if normar <= rtol * initial_norm else 7
    return solution, code, completed, float(norm(data_residual)), normar
